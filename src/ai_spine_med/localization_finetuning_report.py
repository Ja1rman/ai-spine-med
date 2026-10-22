"""Validation-only comparison of separately adapted XML/mixed severity heads.

The four prediction tables are joined by patient and level before comparison.
Public files contain recalculated aggregates and hashes, never patient rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .assets import sha256_file, write_json
from .localization_severity_report import (
    ARMS,
    CLASSES,
    DATASETS,
    LEVELS,
    NAMES,
    SEEDS,
    _key,
    _metrics,
    read_validation_predictions,
    verify_ensemble,
)
from .uncertainty import paired_study_bootstrap

STAGES = ("before", "after")
MODELS = tuple(f"{arm}_{stage}" for arm in ARMS for stage in STAGES)
CONTRASTS = {
    "xml_only_after_minus_before": (["xml_only_after"], ["xml_only_before"]),
    "mixed_after_minus_before": (["mixed_after"], ["mixed_before"]),
    "after_mixed_minus_xml_only": (["mixed_after"], ["xml_only_after"]),
    "before_mixed_minus_xml_only": (["mixed_before"], ["xml_only_before"]),
}
COLORS = ("#89b4be", "#236d82", "#e0bf97", "#a76e35")


def _paired(tables):
    if set(tables) != set(MODELS):
        raise ValueError("Require all four before/after localization arms")
    indexed = {name: {_key(row): row for row in tables[name]} for name in MODELS}
    first = indexed[MODELS[0]]
    for name, mapping in indexed.items():
        if len(mapping) != len(tables[name]) or mapping.keys() != first.keys():
            raise ValueError("Models have different or duplicate patient/level cohorts")
        for key, row in mapping.items():
            if row["split"] != "validation" or row["label"] != first[key]["label"]:
                raise ValueError("Require matching validation grades and missing-reference masks")
    for arm in ARMS:
        for key, row in indexed[f"{arm}_before"].items():
            if row["status"] != indexed[f"{arm}_after"][key]["status"]:
                raise ValueError("Fine-tuning cannot change a fixed ROI extraction coverage mask")
    return indexed


def _extra_metrics(rows):
    metric = _metrics(rows)
    matrix = metric["conditional"]["confusion_matrix"]
    metric["moderate_found"] = int(matrix[1][1])
    metric["false_severe"] = int(matrix[0][2] + matrix[1][2])
    return metric


def _changes(left, right):
    """Counts on common assessed references; left is the newer/alternative model."""
    result = dict.fromkeys(
        (
            "same_category",
            "improved",
            "worsened",
            "both_correct",
            "both_wrong",
            "severe_improved",
            "severe_worsened",
            "severe_both_found",
            "severe_both_missed",
        ),
        0,
    )
    improved_by_reference_class = dict.fromkeys(CLASSES, 0)
    for a, b in zip(left, right, strict=True):
        ac, bc = a["prediction"] == a["label"], b["prediction"] == b["label"]
        result["same_category"] += a["prediction"] == b["prediction"]
        outcome = (
            "both_correct"
            if ac and bc
            else "improved"
            if ac
            else "worsened"
            if bc
            else "both_wrong"
        )
        result[outcome] += 1
        if ac and not bc:
            improved_by_reference_class[CLASSES[a["label"]]] += 1
        if a["label"] == 2:
            outcome = (
                "severe_both_found"
                if ac and bc
                else "severe_improved"
                if ac
                else "severe_worsened"
                if bc
                else "severe_both_missed"
            )
            result[outcome] += 1
    return {
        **{key: int(value) for key, value in result.items()},
        "improved_by_reference_class": improved_by_reference_class,
    }


def compare_group(tables, *, resamples=2000):
    indexed = _paired(tables)
    first = indexed[MODELS[0]]
    if not first or len({key[0] for key in first}) != 1:
        raise ValueError("Compare one nonempty dataset at a time")
    common = [
        key
        for key, row in first.items()
        if row["label"] is not None and all(indexed[name][key]["status"] == "ok" for name in MODELS)
    ]
    paired = {name: [dict(indexed[name][key], series_id="") for key in common] for name in MODELS}
    bootstrap = None
    if common:
        bootstrap = paired_study_bootstrap(
            paired, contrasts=CONTRASTS, resamples=resamples, seed=93017
        )
        bootstrap["method"] = "Paired patient-level percentile bootstrap; both levels stay together"
    return {
        "n_requested": len(first),
        "n_common_labeled_assessed": len(common),
        "n_excluded_from_common": len(first) - len(common),
        "models_full_cohort": {name: _extra_metrics(tables[name]) for name in MODELS},
        "models_common_cohort": {name: _extra_metrics(paired[name]) for name in MODELS},
        "paired_changes": {
            contrast: _changes(paired[left[0]], paired[right[0]])
            for contrast, (left, right) in CONTRASTS.items()
        },
        "bootstrap": bootstrap,
    }


def public_snapshot(summary, tables, *, resamples=2000):
    required = {
        "phase": "validation",
        "center_reused_exactly": True,
        "severity_retrained": True,
        "test_evaluated": False,
        "localizer_ensemble": "mean_probabilities_before_argmax",
        "severity_training_source": "rsna_train_only",
        "localizers_frozen": True,
        "warm_start": "v6_same_seed",
        "normalization": "frozen_v6_train_statistics",
        "severity_seeds": list(SEEDS),
        "bootstrap_resamples": resamples,
        "bootstrap_seed": 93017,
    }
    if any(summary.get(k) != v or type(summary.get(k)) is not type(v) for k, v in required.items()):
        raise ValueError("Require fixed validation-only, RSNA-TRAIN severity adaptation protocol")
    indexed = _paired(tables)
    public = {
        "schema_version": 1,
        "experiment": "xml_only_vs_mixed_localization_separately_adapted_severity",
        **required,
        "severity_ensemble": "Equal mean of three seed probability vectors within each arm/stage",
        "aisslab_grade_crosswalk_verified": False,
        "roi_mm": [50, 50],
        "primary_dataset": "rsna",
        "scope": "Reused development validation; epoch selection and assessment share this cohort",
        "training_protocol": {
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "batch_size": 64,
            "max_epochs": 15,
            "patience": 4,
            "epoch_zero_eligible": True,
            "selection": "Overall RSNA VAL macro-F1; earliest epoch wins ties",
            "trained_layers": "All MLP layers; encoder remains frozen",
            "class_weighting": "Inverse-square-root frequency; RSNA TRAIN paired cohort only",
        },
        "groups": {},
    }
    for dataset in DATASETS:
        for level in ("all", *LEVELS):
            selected = {
                name: [
                    row
                    for key, row in indexed[name].items()
                    if key[0] == dataset and (level == "all" or key[2] == level)
                ]
                for name in MODELS
            }
            public["groups"][f"{dataset}/{level}"] = compare_group(selected, resamples=resamples)
    return public


def public_training(summary):
    """Whitelist numerical TRAIN accounting and epoch choices; reject inconsistent receipts."""
    expected = {
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "max_epochs": 15,
        "patience": 4,
        "epoch0_eligible": True,
    }
    config = summary.get("training_config")
    if not isinstance(config, dict) or any(
        config.get(k) != v or type(config.get(k)) is not type(v) for k, v in expected.items()
    ):
        raise ValueError("Unexpected fine-tuning optimization protocol")
    result = {}
    for arm in ARMS:
        source = summary["arms"][arm].get("training", {})
        for field in ("n_patients", "n_targets", "n_requested_targets"):
            if type(source.get(field)) is not int or source[field] <= 0:
                raise ValueError("Require positive TRAIN accounting counts")
        counts = source.get("class_counts")
        by_level = source.get("by_level_class_counts")
        if (
            not isinstance(counts, list)
            or len(counts) != 3
            or any(type(v) is not int or v <= 0 for v in counts)
            or not isinstance(by_level, dict)
            or set(by_level) != set(LEVELS)
            or any(
                not isinstance(values, list)
                or len(values) != 3
                or any(type(v) is not int or v < 0 for v in values)
                for values in by_level.values()
            )
            or sum(counts) != source["n_targets"]
            or not source["n_patients"] <= source["n_targets"] <= 2 * source["n_patients"]
            or source["n_requested_targets"] < source["n_targets"]
            or np.asarray(list(by_level.values())).sum(axis=0).tolist() != counts
        ):
            raise ValueError("Inconsistent TRAIN class, patient or target counts")
        candidate = {
            field: source[field]
            for field in (
                "n_patients",
                "n_targets",
                "n_requested_targets",
                "class_counts",
                "by_level_class_counts",
            )
        }
        for field in ("selected_epochs", "epoch0_scores", "selected_scores"):
            values = source.get(field)
            if not isinstance(values, dict) or set(values) != {str(seed) for seed in SEEDS}:
                raise ValueError("Require all three seed epoch selections and scores")
            if field == "selected_epochs":
                valid = all(type(v) is int and 0 <= v <= 15 for v in values.values())
            else:
                valid = all(
                    type(v) in (int, float) and np.isfinite(v) and 0 <= v <= 1
                    for v in values.values()
                )
            if not valid:
                raise ValueError("Invalid selected epoch or validation score")
            candidate[field] = {str(seed): values[str(seed)] for seed in SEEDS}
        if any(
            candidate["selected_scores"][str(seed)] < candidate["epoch0_scores"][str(seed)]
            for seed in SEEDS
        ):
            raise ValueError("Epoch-zero eligibility forbids a worse selected seed VAL score")
        result[arm] = candidate
    for field in ("n_patients", "n_targets", "class_counts", "by_level_class_counts"):
        if result["xml_only"][field] != result["mixed"][field]:
            raise ValueError("Both branches must fit the same paired TRAIN cohort")
    weights = np.asarray(summary.get("class_weights"), dtype=float)
    expected_weights = 1 / np.sqrt(result["xml_only"]["class_counts"])
    expected_weights /= expected_weights.mean()
    if weights.shape != (3,) or not np.allclose(weights, expected_weights, atol=1e-6, rtol=0):
        raise ValueError("Class weights differ from the paired TRAIN frequencies")
    return {"optimization": expected, "class_weights": weights.tolist(), "arms": result}


def _f(value):
    return "—" if value is None else f"{value:.3f}".replace(".", ",")


def _interval(group, contrast):
    if group["bootstrap"] is None:
        return "—"
    value = group["bootstrap"]["contrasts"][contrast]["metrics"]["macro_f1"]
    return f"{_f(value['point'])} [{_f(value['lower_95'])}; {_f(value['upper_95'])}]"


def _found(metric, category):
    matrix = metric["conditional"]["confusion_matrix"]
    index = CLASSES.index(category)
    return f"{matrix[index][index]}/{metric['class_counts'][category]}"


def render_markdown(public, *, comparison_image, confusion_image):
    groups = public["groups"]
    l5 = groups["rsna/L5-S1"]
    l5_xml, l5_mixed = [l5["models_full_cohort"][f"{arm}_after"] for arm in ARMS]
    observed = (
        f"После адаптации на RSNA L5–S1 macro-F1 составляет "
        f"{_f(l5_xml['conditional']['macro_f1'])} с XML-only и "
        f"{_f(l5_mixed['conditional']['macro_f1'])} со смешанной локализацией. "
        f"Тяжёлых найдено {_found(l5_xml, 'severe')} и {_found(l5_mixed, 'severe')}; "
        f"умеренных — {_found(l5_xml, 'moderate')} и {_found(l5_mixed, 'moderate')}."
    )
    mixed_all = groups["rsna/all"]["models_full_cohort"]
    mixed_l4 = groups["rsna/L4-L5"]["models_full_cohort"]
    tradeoff = (
        "У смешанной ветки после дообучения больше найденных тяжёлых L4–L5: "
        f"{_found(mixed_l4['mixed_before'], 'severe')} → "
        f"{_found(mixed_l4['mixed_after'], 'severe')}. Одновременно ложных ответов «тяжёлый» "
        f"на RSNA стало больше: {mixed_all['mixed_before']['false_severe']} → "
        f"{mixed_all['mixed_after']['false_severe']} на обоих уровнях вместе. "
        "Поэтому прирост найденных тяжёлых нужно оценивать вместе с ложными тревогами."
    )
    lines = [
        "# Дообучение классификатора под два варианта локализации",
        "",
        "Я отдельно адаптировал три классификатора степени стеноза под ROI локализатора, обученного "
        "только по XML AISSLab, и под ROI смешанного локализатора AISSLab + RSNA. Для каждой ветки "
        "сравнил прежние ответы v8 и ответы после адаптации на той же валидации. Локализаторы, сеть "
        "центра, размер области 50×50 мм и извлечение признаков оставил прежними.",
        "",
        "Обучение степени использует только RSNA TRAIN. AISSLab здесь нужен для дополнительной "
        "проверки переноса; его категории не участвовали в обучении или выборе эпох.",
        "",
        observed,
        "",
        tradeoff,
        "",
        "Каждый запуск начал с классификатора v6 с тем же seed; среднее и стандартное отклонение "
        "признаков v6 зафиксировал. Обновлял все слои небольшого MLP, оставив ResNet18 замороженным. "
        "AdamW: learning rate 0,0001, weight decay 0,0001, batch 64; максимум 15 эпох и patience 4. "
        "Исходную эпоху 0 тоже разрешил выбирать. Критерий — общий macro-F1 RSNA VAL, при равенстве "
        "остаётся более ранняя эпоха. Для двух веток использовал общий обработанный RSNA TRAIN, "
        "одинаковые классовые веса, порядок batch и seeds.",
        "",
        "## RSNA: до и после",
        "",
        "Macro-F1 и счётчики находок в таблицах относятся ко всему доступному набору каждой "
        "модели. Парные разницы, интервалы и переходы ответов рассчитаны на случаях с меткой и "
        "результатом всех четырёх моделей; исключения видны в разделе покрытия.",
        "",
        "| Уровень | Локализация | До, macro-F1 | После, macro-F1 | После − до, 95% ДИ | Умеренные до → после | Тяжёлые до → после |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for level in ("all", *LEVELS):
        group = groups[f"rsna/{level}"]
        for arm in ARMS:
            a, b = [group["models_full_cohort"][f"{arm}_{stage}"] for stage in STAGES]
            lines.append(
                f"| {'Оба уровня' if level == 'all' else level} | {NAMES[arm]} | "
                f"{_f(a['conditional']['macro_f1'])} | {_f(b['conditional']['macro_f1'])} | "
                f"{_interval(group, f'{arm}_after_minus_before')} | "
                f"{_found(a, 'moderate')} → {_found(b, 'moderate')} | "
                f"{_found(a, 'severe')} → {_found(b, 'severe')} |"
            )
    lines += [
        "",
        f"![Дообучение и локализация: результаты]({comparison_image})",
        "",
        "## После адаптации: смешанная против XML-only",
        "",
        "| Данные | Уровень | XML-only, macro-F1 | Смешанная, macro-F1 | Смешанная − XML, 95% ДИ |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for level in ("all", *LEVELS):
            group = groups[f"{dataset}/{level}"]
            a, b = [group["models_full_cohort"][f"{arm}_after"] for arm in ARMS]
            lines.append(
                f"| {dataset.upper()}{' · условные категории' if dataset == 'aisslab' else ''} | "
                f"{'Оба уровня' if level == 'all' else level} | {_f(a['conditional']['macro_f1'])} | "
                f"{_f(b['conditional']['macro_f1'])} | {_interval(group, 'after_mixed_minus_xml_only')} |"
            )
    lines += [
        "",
        "## L5–S1: отдельные ошибки",
        "",
        "| Данные | Ветка | До/после | Метки: норма / умеренный / тяжёлый | Умеренные найдены | Тяжёлые найдены | Тяжёлый → норма | Тяжёлый → умеренный | Ложный тяжёлый | Не обработано с меткой |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        group = groups[f"{dataset}/L5-S1"]
        for arm in ARMS:
            for stage in STAGES:
                metric = group["models_full_cohort"][f"{arm}_{stage}"]
                severe = metric["severe"]
                counts = "/".join(str(metric["class_counts"][name]) for name in CLASSES)
                lines.append(
                    f"| {dataset.upper()} | {NAMES[arm]} | {'До' if stage == 'before' else 'После'} | "
                    f"{counts} | {_found(metric, 'moderate')} | {_found(metric, 'severe')} | "
                    f"{severe['predicted_normal_mild']} | {severe['predicted_moderate']} | "
                    f"{metric['false_severe']} | {metric['n_not_assessed']} |"
                )
    for arm in ARMS:
        changes = l5["paired_changes"][f"{arm}_after_minus_before"]
        lines += [
            "",
            f"RSNA L5–S1, {NAMES[arm]}: после адаптации исправлено "
            f"{changes['improved']} ошибок, испорчено {changes['worsened']} правильных ответов. "
            f"Среди тяжёлых: найдено дополнительно {changes['severe_improved']}, потеряно "
            f"{changes['severe_worsened']}, обе версии пропустили {changes['severe_both_missed']}.",
        ]
        normal = changes["improved_by_reference_class"]["normal_mild"]
        if normal == changes["improved"]:
            lines += [
                f"Все {normal} исправления относятся к исходной метке «норма/лёгкая степень». "
                "Снизилось число ложных тревог; распознавание умеренного и тяжёлого L5–S1 "
                "не улучшилось."
            ]
    lines += [
        "",
        f"![RSNA L5–S1: матрицы до и после]({confusion_image})",
        "",
        "Строки — исходная метка, столбцы — ответ модели. Матрицы включают только обработанные "
        "случаи с меткой; ошибки обработки и отсутствующие метки сохраняются отдельно.",
        "",
        "## Покрытие",
        "",
        "| Данные | Запрошено | Без метки | Обработано с меткой XML / смешанная | Общих для сравнения четырёх моделей |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        group = groups[f"{dataset}/all"]
        a, b = [group["models_full_cohort"][f"{arm}_after"] for arm in ARMS]
        lines.append(
            f"| {dataset.upper()} | {group['n_requested']} | {a['n_missing_reference_label']} | "
            f"{a['n_assessed']} / {b['n_assessed']} | {group['n_common_labeled_assessed']} |"
        )
    if "training" in public:
        train = public["training"]
        cohort = train["arms"]["xml_only"]
        counts = "/".join(str(v) for v in cohort["class_counts"])
        l5_counts = "/".join(str(v) for v in cohort["by_level_class_counts"]["L5-S1"])
        lines += [
            "",
            "## На чём дообучал",
            "",
            f"В обеих ветках один и тот же набор из {cohort['n_patients']} пациентов и "
            f"{cohort['n_targets']} обработанных уровней RSNA TRAIN. Число меток норма/лёгкий / "
            f"умеренный / тяжёлый: {counts}; отдельно L5–S1: {l5_counts}. "
            "Пропуски извлечения исключены из градиентного обучения, но сохранены в учёте.",
            "",
            "| Локализация | Выбранные эпохи: 42 / 2024 / 3407 | Число запрошенных TRAIN уровней |",
            "|---|---:|---:|",
        ]
        for arm in ARMS:
            value = train["arms"][arm]
            epochs = "/".join(str(value["selected_epochs"][str(seed)]) for seed in SEEDS)
            lines.append(f"| {NAMES[arm]} | {epochs} | {value['n_requested_targets']} |")
        lines += [
            "",
            "Нулевая эпоха означает, что для этого запуска исходные веса оказались "
            "лучшими по заданному критерию. Все исходные и выбранные оценки отдельных "
            "запусков сохранены в JSON; ансамбль не обязан улучшаться, даже когда F1 каждого "
            "выбранного запуска не ниже исходного.",
        ]
    lines += [
        "",
        "## Как трактовать результат",
        "",
        "У каждой ветки три фиксированных запуска (42, 2024, 3407); финальный ответ — равное "
        "усреднение вероятностей, а не выбор удачного запуска. До и после используются одни "
        "и те же ROI. После адаптации сравниваются системы с классификаторами, обученными "
        "под свою локализацию; это уже не проверка одной только замены среза.",
        "",
        "RSNA VAL участвует в выборе эпохи классификаторов и уже использовалась в предыдущих "
        "экспериментах. Это результат разработки на повторно используемой валидации, а не "
        "оценка на независимом тесте. TEST в этом эксперименте не оценивал. Число тяжёлых "
        "L5–S1 мало, поэтому рядом с F1 показываю абсолютные счётчики и все переходы категорий.",
        "",
        f"95% интервалы рассчитаны {public['bootstrap_resamples']} парными bootstrap-повторами "
        "по пациентам; оба уровня остаются вместе. Они описывают вариацию состава этой "
        "выборки для фиксированных моделей, не исправляют смещение от выбора эпох на VAL и "
        "не учитывают неопределённость разметки, новое обучение или другую клинику.",
        "",
        "Категории AISSLab сопоставлены с RSNA условно: исходная шкала не подтверждена как "
        "эквивалентная. Эти результаты не объединяю с RSNA. Для вывода о тяжёлом L5–S1 врачу "
        "по-прежнему нужно проверить степень стеноза, выбранный срез и попадание канала в ROI.",
        "",
        "Агрегаты: [localization-finetuning-results.json](localization-finetuning-results.json). "
        "Предыдущее сравнение с неизменным классификатором: "
        "[localization-severity.md](localization-severity.md). Снимки, индивидуальные ответы и "
        "веса остаются в игнорируемых локальных папках.",
        "",
        "## Повторить",
        "",
        "Протокол: [localization-finetuning-v9.json](../configs/localization-finetuning-v9.json). "
        "Готовые кэши и обученные веса защищены от перезаписи; команды предназначены для нового запуска.",
        "",
        "```bash",
        "python -m ai_spine_med.localization_finetuning_data --device mps",
        "python -m ai_spine_med.localization_finetuning --device mps",
        "python -m ai_spine_med.localization_finetuning_report",
        "```",
        "",
    ]
    return "\n".join(lines)


def draw_comparison(public, path):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), sharey=True)
    x = np.arange(3)
    for dataset, ax in zip(DATASETS, axes):
        for offset, name, color in zip((-0.27, -0.09, 0.09, 0.27), MODELS, COLORS):
            values = [
                public["groups"][f"{dataset}/{level}"]["models_full_cohort"][name]["conditional"][
                    "macro_f1"
                ]
                for level in ("all", *LEVELS)
            ]
            arm, stage = name.rsplit("_", 1)
            bars = ax.bar(
                x + offset,
                [v or 0 for v in values],
                width=0.17,
                color=color,
                label=f"{NAMES[arm]} · {'до' if stage == 'before' else 'после'}",
            )
            for bar, value in zip(bars, values):
                ax.annotate(
                    _f(value),
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
        ax.set(
            title=dataset.upper() + (" · условные категории" if dataset == "aisslab" else ""),
            xticks=x,
            xticklabels=["Оба уровня", *LEVELS],
            ylim=(0, 1),
            ylabel="macro-F1",
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    axes[0].legend(loc="upper left", frameon=False, fontsize=8)
    fig.suptitle(
        "Отдельная адаптация классификатора под XML-only и смешанную локализацию", fontsize=13
    )
    fig.text(
        0.5,
        0.02,
        "Повторно используемая VAL · три запуска каждой ветки · одинаковые ROI до/после",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def draw_confusion(public, path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for arm, row_axes in zip(ARMS, axes):
        for stage, ax in zip(STAGES, row_axes):
            metric = public["groups"]["rsna/L5-S1"]["models_full_cohort"][f"{arm}_{stage}"]
            matrix = np.asarray(metric["conditional"]["confusion_matrix"])
            totals = matrix.sum(axis=1, keepdims=True)
            colors = np.divide(
                matrix, totals, out=np.zeros_like(matrix, dtype=float), where=totals > 0
            )
            ax.imshow(colors, cmap="Blues", vmin=0, vmax=1)
            for i in range(3):
                for j in range(3):
                    ax.text(
                        j,
                        i,
                        str(matrix[i, j]),
                        ha="center",
                        va="center",
                        fontsize=15,
                        color="white" if colors[i, j] > 0.55 else "#183142",
                    )
            ax.set(
                title=f"{NAMES[arm]} · {'до' if stage == 'before' else 'после'}",
                xticks=range(3),
                yticks=range(3),
                xticklabels=["Норма/лёгкий", "Умеренный", "Тяжёлый"],
                yticklabels=["Норма/лёгкий", "Умеренный", "Тяжёлый"],
                xlabel="Ответ модели",
                ylabel="Исходная метка",
            )
            ax.tick_params(axis="both", length=0, labelsize=9)
    fig.suptitle("RSNA L5–S1: результат адаптации классификатора", fontsize=14)
    fig.text(
        0.5,
        0.015,
        "Числа — случаи с меткой и ответом · цвет нормирован по исходной степени",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def report_localization_finetuning(
    summary_path=Path("reports/runs/localization-finetuning-v9/summary.json"),
    *,
    out_json=Path("docs/localization-finetuning-results.json"),
    out_markdown=Path("docs/localization-finetuning.md"),
    comparison_image=Path("docs/images/localization_finetuning_comparison.png"),
    confusion_image=Path("docs/images/localization_finetuning_l5_confusion.png"),
    resamples=2000,
):
    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text())
    if set(summary["arms"]) != set(ARMS):
        raise ValueError("Unexpected localization arms")
    tables, hashes, seed_tables = {}, {}, {}
    for arm in ARMS:
        for stage in STAGES:
            source = summary["arms"][arm][stage]
            path = Path(source["predictions"])
            if not path.is_absolute():
                path = summary_path.parent / path
            checksum = sha256_file(path)
            if source.get("predictions_sha256") != checksum:
                raise ValueError("Prediction CSV changed after the inference summary was sealed")
            name = f"{arm}_{stage}"
            tables[name] = read_validation_predictions(path)
            seed_paths = source["seed_predictions"]
            if set(seed_paths) != {str(seed) for seed in SEEDS}:
                raise ValueError("Require all three seed prediction paths")
            seeds, seed_hashes = {}, {}
            for seed in SEEDS:
                seed_path = Path(seed_paths[str(seed)])
                if not seed_path.is_absolute():
                    seed_path = summary_path.parent / seed_path
                seeds[seed] = read_validation_predictions(seed_path)
                seed_hashes[str(seed)] = sha256_file(seed_path)
            verify_ensemble(tables[name], seeds)
            seed_tables[name] = seeds
            hashes[name] = {"ensemble_sha256": checksum, "seeds": seed_hashes}
    public = public_snapshot(summary, tables, resamples=resamples)
    public["training"] = public_training(summary)
    for dataset in DATASETS:
        for level in ("all", *LEVELS):
            public["groups"][f"{dataset}/{level}"]["individual_seed_models_full_cohort"] = {
                name: {
                    str(seed): _extra_metrics(
                        [
                            row
                            for row in seed_tables[name][seed]
                            if row["dataset"] == dataset
                            and (level == "all" or row["level"] == level)
                        ]
                    )
                    for seed in SEEDS
                }
                for name in MODELS
            }
    individual = public["groups"]["rsna/all"]["individual_seed_models_full_cohort"]
    for arm in ARMS:
        training = public["training"]["arms"][arm]
        for seed in SEEDS:
            for stage, field in (("before", "epoch0_scores"), ("after", "selected_scores")):
                measured = individual[f"{arm}_{stage}"][str(seed)]["conditional"]["macro_f1"]
                if not np.isclose(measured, training[field][str(seed)], rtol=0, atol=1e-12):
                    raise ValueError(
                        "Selected seed validation scores disagree with actual predictions"
                    )
            if training["selected_epochs"][str(seed)] == 0:
                before = {_key(row): row for row in seed_tables[f"{arm}_before"][seed]}
                after = {_key(row): row for row in seed_tables[f"{arm}_after"][seed]}
                for key, row in before.items():
                    if row["status"] == "ok" and not np.allclose(
                        [row[f"p_{name}"] for name in CLASSES],
                        [after[key][f"p_{name}"] for name in CLASSES],
                        rtol=0,
                        atol=2e-7,
                    ):
                        raise ValueError(
                            "Epoch-zero selection changed the unchanged seed predictions"
                        )
    public["private_input_sha256"] = hashes
    public["source_summary_sha256"] = sha256_file(summary_path)
    write_json(Path(out_json), public)
    draw_comparison(public, comparison_image)
    draw_confusion(public, confusion_image)
    out_markdown = Path(out_markdown)
    out_markdown.parent.mkdir(parents=True, exist_ok=True)
    out_markdown.write_text(
        render_markdown(
            public,
            comparison_image="images/" + Path(comparison_image).name,
            confusion_image="images/" + Path(confusion_image).name,
        )
    )
    return public


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", type=Path, default=Path("reports/runs/localization-finetuning-v9/summary.json")
    )
    args = parser.parse_args(argv)
    report_localization_finetuning(args.summary)


if __name__ == "__main__":
    main()
