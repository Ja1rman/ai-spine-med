"""Paired validation comparison of frozen ResNet18 and layer4 adaptation.

Only completed three-seed prediction tables and numerical training receipts are
read. Public artifacts contain aggregates, never patient IDs or arbitrary notes.
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
from .localization_finetuning_report import _changes, _extra_metrics, _f
from .localization_severity_report import (
    ARMS,
    CLASSES,
    LEVELS,
    SEEDS,
    _key,
    read_validation_predictions,
    verify_ensemble,
)
from .uncertainty import paired_study_bootstrap

VARIANTS = ("frozen", "layer4")
MODELS = tuple(f"{arm}_{variant}" for arm in ARMS for variant in VARIANTS)
NAMES = {"xml_only": "XML-only", "mixed": "Смешанная локализация"}
VARIANT_NAMES = {"frozen": "Замороженная сеть", "layer4": "layer4 + классификатор"}
COLORS = {"frozen": "#82919c", "layer4": "#286e82"}
CONTRASTS = {f"{arm}_layer4_minus_frozen": ([f"{arm}_layer4"], [f"{arm}_frozen"]) for arm in ARMS}
BOOTSTRAP_SEED = 113017
OPTIMIZATION = {
    "head_learning_rate": 1e-4,
    "encoder_learning_rate": 1e-5,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "max_epochs": 15,
    "patience": 4,
    "epoch0_eligible": True,
}


def _paired_maps(tables):
    if set(tables) != set(MODELS):
        raise ValueError("Require both localization arms and both fixed encoder variants")
    indexed = {name: {_key(row): row for row in tables[name]} for name in MODELS}
    first = indexed[MODELS[0]]
    if not first:
        raise ValueError("Empty RSNA validation cohort")
    for name, mapping in indexed.items():
        if len(mapping) != len(tables[name]) or mapping.keys() != first.keys():
            raise ValueError("Models have different or duplicate patient/level cohorts")
        for key, row in mapping.items():
            if (
                row["dataset"] != "rsna"
                or row["split"] != "validation"
                or row["level"] not in LEVELS
                or row["label"] != first[key]["label"]
            ):
                raise ValueError("Require matching RSNA validation grades and reference masks")
    for arm in ARMS:
        for key, row in indexed[f"{arm}_frozen"].items():
            if row["status"] != indexed[f"{arm}_layer4"][key]["status"]:
                raise ValueError("Encoder adaptation cannot change the fixed ROI coverage mask")
    return indexed


def compare_group(tables, *, resamples=2000):
    """Compare identical requested levels while retaining all extraction failures."""
    indexed = _paired_maps(tables)
    first = indexed[MODELS[0]]
    common = [
        key
        for key, row in first.items()
        if row["label"] is not None and all(indexed[name][key]["status"] == "ok" for name in MODELS)
    ]
    paired = {name: [dict(indexed[name][key], series_id="") for key in common] for name in MODELS}
    bootstrap = None
    if common:
        bootstrap = paired_study_bootstrap(
            paired, contrasts=CONTRASTS, resamples=resamples, seed=BOOTSTRAP_SEED
        )
        bootstrap["method"] = "Paired patient-level percentile bootstrap; both levels stay together"
    return {
        "n_requested": len(first),
        "n_requested_patients": len({key[1] for key in first}),
        "n_common_labeled_assessed": len(common),
        "n_common_patients": len({key[1] for key in common}),
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
        "test_evaluated": False,
        "localizers_frozen": True,
        "center_reused_exactly": True,
        "roi_reused_exactly": True,
        "batch_norm_frozen": True,
        "severity_training_source": "rsna_train_only",
        "seeds": list(SEEDS),
        "bootstrap_resamples": resamples,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    if any(summary.get(k) != v or type(summary.get(k)) is not type(v) for k, v in required.items()):
        raise ValueError("Require fixed paired validation-only encoder adaptation")
    indexed = _paired_maps(tables)
    result = {
        "schema_version": 1,
        "experiment": "paired_roi_resnet18_layer4_and_head_vs_frozen_encoder",
        **required,
        "primary_dataset": "rsna",
        "severity_ensemble": "Equal probability mean of seeds 42/2024/3407, never best-seed selection",
        "scope": "Reused development validation also used for epoch selection; not independent confirmation",
        "roi_mm": [50, 50],
        "unchanged": "Predicted level, predicted canal center, ROI and neighbors, mean pooling, v6 feature normalizers and BatchNorm buffers",
        "initialization": "Matching v6 MLP seed and original ImageNet ResNet18 for both variants",
        "groups": {},
    }
    for level in ("all", *LEVELS):
        selected = {
            name: [row for key, row in indexed[name].items() if level == "all" or key[2] == level]
            for name in MODELS
        }
        result["groups"][f"rsna/{level}"] = compare_group(selected, resamples=resamples)
    return result


def public_training(summary):
    """Only typed TRAIN totals and the prespecified numerical optimization settings."""
    source_config = summary.get("training_config", {})
    if any(
        source_config.get(k) != v or type(source_config.get(k)) is not type(v)
        for k, v in OPTIMIZATION.items()
    ):
        raise ValueError("Unexpected paired layer4 training protocol")
    source = summary.get("training", {})
    totals = ("n_patients", "n_targets", "n_requested_targets")
    if any(type(source.get(k)) is not int or source[k] <= 0 for k in totals):
        raise ValueError("Require positive TRAIN patient and target accounting")
    counts = source.get("class_counts")
    by_level = source.get("by_level_class_counts")
    if (
        not isinstance(counts, list)
        or len(counts) != 3
        or any(type(v) is not int or v <= 0 for v in counts)
        or not isinstance(by_level, dict)
        or set(by_level) != set(LEVELS)
        or any(
            not isinstance(v, list) or len(v) != 3 or any(type(x) is not int or x < 0 for x in v)
            for v in by_level.values()
        )
        or sum(counts) != source["n_targets"]
        or not source["n_patients"] <= source["n_targets"] <= 2 * source["n_patients"]
        or source["n_requested_targets"] < source["n_targets"]
        or np.asarray(list(by_level.values())).sum(axis=0).tolist() != counts
    ):
        raise ValueError("Inconsistent paired TRAIN class and target counts")
    weights = np.asarray(source.get("class_weights"), dtype=float)
    expected = 1 / np.sqrt(counts)
    expected /= expected.mean()
    if weights.shape != (3,) or not np.allclose(weights, expected, rtol=0, atol=1e-6):
        raise ValueError("Class weights differ from paired TRAIN frequencies")
    return {
        "optimization": dict(OPTIMIZATION),
        **{k: source[k] for k in totals},
        "class_counts": counts,
        "by_level_class_counts": by_level,
        "class_weights": weights.tolist(),
    }


def _history(source, selected, score):
    if not isinstance(source, list) or not source or len(source) > OPTIMIZATION["max_epochs"] + 1:
        raise ValueError("Require a numerical history including epoch zero")
    result = []
    for i, item in enumerate(source):
        value, loss = item.get("validation_macro_f1"), item.get("weighted_train_loss")
        if (
            type(item.get("epoch")) is not int
            or item["epoch"] != i
            or type(value) not in (int, float)
            or not np.isfinite(value)
            or not 0 <= value <= 1
            or (i == 0 and loss is not None)
            or (i > 0 and (type(loss) not in (int, float) or not np.isfinite(loss) or loss < 0))
        ):
            raise ValueError("Invalid validation history or training loss")
        result.append(
            {"epoch": i, "validation_macro_f1": float(value), "weighted_train_loss": loss}
        )
    best = int(np.argmax([entry["validation_macro_f1"] for entry in result]))
    if type(selected) is not int or selected != best:
        raise ValueError("Selected epoch is not the earliest maximum including epoch zero")
    if not np.isclose(result[selected]["validation_macro_f1"], score, rtol=0, atol=1e-12):
        raise ValueError("Selected history score disagrees with actual validation predictions")
    return {"selected_epoch": selected, "history": result}


def _source(summary_path, relative, checksums):
    relative = Path(relative)
    root = summary_path.parent.resolve()
    path = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root):
        raise ValueError("Prediction/history paths must stay inside the completed run")
    checksum = checksums.get(relative.as_posix())
    if not isinstance(checksum, str) or sha256_file(path) != checksum:
        raise ValueError("Prediction or history checksum changed")
    return path, checksum


def _found(metric, category):
    i = CLASSES.index(category)
    return f"{metric['conditional']['confusion_matrix'][i][i]}/{metric['class_counts'][category]}"


def _interval(group, arm):
    if group["bootstrap"] is None:
        return "—"
    metric = group["bootstrap"]["contrasts"][f"{arm}_layer4_minus_frozen"]["metrics"]["macro_f1"]
    return f"{_f(metric['point'])} [{_f(metric['lower_95'])}; {_f(metric['upper_95'])}]"


def render_markdown(public, *, comparison_image, confusion_image, learning_image):
    groups = public["groups"]
    training = public["training"]
    lines = [
        "# Дообучение layer4 ResNet18 на прежних областях",
        "",
        "Сравнил замороженный ResNet18 с вариантом, где дообучал его последний блок `layer4` "
        "вместе с классификатором степени стеноза. Проверил обе прежние локализации: XML-only "
        "и смешанную AISSLab + RSNA. Для каждой сделал три запуска: 42, 2024 и 3407.",
        "",
        "Оба варианта начинал с одинаковых весов: исходный ImageNet ResNet18 и классификатор "
        "v6 соответствующего seed. Замороженный контроль обучил заново. Локализатор, центр, "
        "области 50×50 мм, соседние срезы и усреднение их признаков оставил прежними. "
        "Нормализацию признаков v6 и статистики BatchNorm не обновлял. Аугментации не добавлял.",
        "",
        f"Для обучения использовал одни и те же {training['n_patients']} пациентов и "
        f"{training['n_targets']} обработанных уровней RSNA TRAIN. Для сравнения оставил "
        f"{groups['rsna/all']['n_common_labeled_assessed']} общих обработанных уровней с меткой "
        f"у {groups['rsna/all']['n_common_patients']} пациентов RSNA VAL. Отказы и отсутствующие "
        "метки учитываю отдельно; в норму их не превращаю.",
        "",
        "## Результаты на одинаковых уровнях",
        "",
        "В таблице — ансамбли равного среднего вероятностей трёх запусков. Разница F1 "
        "рассчитана как `layer4 − замороженная сеть`, рядом парный 95% интервал.",
        "",
        "| Уровень | Локализация | Замороженная, macro-F1 | layer4, macro-F1 | Разница, 95% ДИ | Умеренные до → после | Тяжёлые до → после |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    if public.get("frozen_control_reproduced_v9") is True:
        lines[6:6] = [
            "Замороженный контроль точно повторил прежний опыт v9: совпали выбранные эпохи "
            "и все вероятности шести запусков. Это проверил отдельно по сохранённым ответам.",
            "",
        ]
    for level in ("all", *LEVELS):
        group = groups[f"rsna/{level}"]
        for arm in ARMS:
            before, after = [group["models_common_cohort"][f"{arm}_{v}"] for v in VARIANTS]
            lines.append(
                f"| {'Оба уровня' if level == 'all' else level} | {NAMES[arm]} | "
                f"{_f(before['conditional']['macro_f1'])} | {_f(after['conditional']['macro_f1'])} | "
                f"{_interval(group, arm)} | {_found(before, 'moderate')} → {_found(after, 'moderate')} | "
                f"{_found(before, 'severe')} → {_found(after, 'severe')} |"
            )
    lines += [
        "",
        f"![Сравнение дообучения layer4]({comparison_image})",
        "",
        "## L5–S1 отдельно",
        "",
        "| Локализация | Вариант | Метки: норма / умеренный / тяжёлый | Умеренные найдены | Тяжёлые найдены | Тяжёлый → норма | Тяжёлый → умеренный | Ложный тяжёлый |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    l5 = groups["rsna/L5-S1"]
    for arm in ARMS:
        for variant in VARIANTS:
            metric = l5["models_common_cohort"][f"{arm}_{variant}"]
            severe = metric["severe"]
            counts = "/".join(str(metric["class_counts"][c]) for c in CLASSES)
            lines.append(
                f"| {NAMES[arm]} | {VARIANT_NAMES[variant]} | {counts} | {_found(metric, 'moderate')} | "
                f"{_found(metric, 'severe')} | {severe['predicted_normal_mild']} | "
                f"{severe['predicted_moderate']} | {metric['false_severe']} |"
            )
    for arm in ARMS:
        changes = l5["paired_changes"][f"{arm}_layer4_minus_frozen"]
        lines += [
            "",
            f"L5–S1, {NAMES[arm]}: после дообучения исправлено ошибок: {changes['improved']}; "
            f"правильных ответов потеряно: {changes['worsened']}. Среди тяжёлых дополнительно "
            f"найдено {changes['severe_improved']}, потеряно {changes['severe_worsened']}; "
            f"обе версии пропустили {changes['severe_both_missed']}.",
        ]
        if (
            changes["improved"] > 0
            and changes["improved_by_reference_class"]["normal_mild"] == changes["improved"]
        ):
            lines += [
                "Все исправления относятся к исходной метке «норма/лёгкая степень». "
                "Число найденных умеренных и тяжёлых случаев не увеличилось.",
            ]
    lines += [
        "",
        f"![Матрицы ошибок L5–S1]({confusion_image})",
        "",
        "Строки — исходная степень, столбцы — ответ. Числа — случаи; цвет нормирован "
        "по исходной степени. Малое число умеренных и тяжёлых L5–S1 делает F1 нестабильным, "
        "поэтому важны абсолютные счётчики и ложные тревоги.",
        "",
        "## Запуски и обучение",
        "",
        "В обоих вариантах AdamW, learning rate классификатора 0,0001. Для layer4 learning rate "
        "0,00001. Batch 64, weight decay 0,0001, максимум 15 эпох, patience 4. Эпоху выбирал по "
        "общему macro-F1 RSNA VAL; при равенстве оставлял раннюю. Эпоха 0 тоже могла быть выбрана.",
        "",
        "| Локализация | Вариант | Выбранные эпохи: 42 / 2024 / 3407 | F1 отдельных запусков: 42 / 2024 / 3407 |",
        "|---|---|---:|---:|",
    ]
    for arm in ARMS:
        for variant in VARIANTS:
            model = f"{arm}_{variant}"
            epochs = "/".join(str(public["runs"][model][str(s)]["selected_epoch"]) for s in SEEDS)
            scores = "/".join(
                _f(
                    groups["rsna/all"]["individual_seed_models_full_cohort"][model][str(s)][
                        "conditional"
                    ]["macro_f1"]
                )
                for s in SEEDS
            )
            lines.append(f"| {NAMES[arm]} | {VARIANT_NAMES[variant]} | {epochs} | {scores} |")
    counts = "/".join(str(x) for x in training["by_level_class_counts"]["L5-S1"])
    lines += [
        "",
        f"В TRAIN L5–S1 число нормальных/лёгких, умеренных и тяжёлых меток: {counts}. "
        "В обоих вариантах использовал одинаковые веса классов, вычисленные только по общему TRAIN. "
        "Нулевая выбранная эпоха означает сохранение исходных весов. Выбор эпох каждого seed "
        "не гарантирует улучшение их вероятностного ансамбля.",
        "",
        f"![Ход обучения]({learning_image})",
        "",
        "## Покрытие",
        "",
        "| Уровень | Локализация | Запрошено | Без метки | Обработано с меткой | Не обработано с меткой | Общих у четырёх вариантов |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for level in ("all", *LEVELS):
        group = groups[f"rsna/{level}"]
        for arm in ARMS:
            metric = group["models_full_cohort"][f"{arm}_frozen"]
            lines.append(
                f"| {'Оба уровня' if level == 'all' else level} | {NAMES[arm]} | "
                f"{group['n_requested']} | {metric['n_missing_reference_label']} | "
                f"{metric['n_assessed']} | {metric['n_not_assessed']} | {group['n_common_labeled_assessed']} |"
            )
    lines += [
        "",
        "Маски обработки замороженного и дообученного вариантов внутри каждой локализации "
        "совпадают. Полные и общие метрики, все отдельные запуски и матрицы ошибок сохранены в JSON.",
        "",
        "## Ограничения",
        "",
        "Эта валидация уже использовалась в предыдущих опытах и сейчас выбирает эпохи. "
        "Это результат разработки, а не независимое подтверждение. TEST в этом эксперименте не оценивал. "
        "Истинные координаты и степени не используются для автоматического выбора областей.",
        "",
        f"95% интервалы рассчитаны {public['bootstrap_resamples']} парными bootstrap-повторами по "
        "пациентам; оба уровня одного пациента остаются вместе. Они описывают состав этой выборки "
        "при фиксированных моделях и не исправляют смещение от выбора эпох на VAL, ошибки разметки "
        "или различия между клиниками. Для итоговой проверки нужны новые не просмотренные пациенты.",
        "",
        "[Численные результаты](roi-layer4-results.json), [протокол](../configs/roi-layer4-v11.json). "
        "Снимки, кэши, индивидуальные ответы и веса остаются локально в игнорируемых папках.",
        "",
        "Пересобрать отчёт по сохранённым ответам:",
        "",
        "```bash",
        "python -m ai_spine_med.roi_layer4_report",
        "```",
        "",
    ]
    overall_intervals = [
        groups["rsna/all"]["bootstrap"]["contrasts"][f"{arm}_layer4_minus_frozen"]["metrics"][
            "macro_f1"
        ]
        for arm in ARMS
    ]
    if all(value["lower_95"] <= 0 <= value["upper_95"] for value in overall_intervals):
        lines[4:4] = [
            "Уверенного общего прироста от layer4 в этом опыте не получил: у обеих "
            "локализаций парный интервал разницы macro-F1 включает ноль.",
            "",
        ]
    return "\n".join(lines)


def draw_comparison(public, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), sharey=True)
    for arm, ax in zip(ARMS, axes, strict=True):
        for variant, offset in zip(VARIANTS, (-0.12, 0.12), strict=True):
            name = f"{arm}_{variant}"
            for i, level in enumerate(("all", *LEVELS)):
                group = public["groups"][f"rsna/{level}"]
                interval = group["bootstrap"]["models"][name]["macro_f1"]
                value = interval["point"]
                ax.errorbar(
                    i + offset,
                    value,
                    yerr=[[value - interval["lower_95"]], [interval["upper_95"] - value]],
                    fmt="o",
                    markersize=8,
                    capsize=5,
                    color=COLORS[variant],
                    label=VARIANT_NAMES[variant] if i == 0 else None,
                )
                ax.annotate(
                    _f(value),
                    (i + offset, interval["upper_95"]),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=10,
                )
        ax.set(
            title=NAMES[arm],
            xticks=range(3),
            xticklabels=["Оба уровня", *LEVELS],
            ylim=(0, 1.02),
            ylabel="macro-F1 и 95% ДИ",
            xlim=(-0.5, 2.5),
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
        ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.suptitle("RSNA VAL: дообучение layer4 на тех же областях", fontsize=15)
    fig.text(
        0.5,
        0.02,
        "Три запуска · парный bootstrap по пациентам · VAL также выбирает эпохи",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def draw_confusion(public, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    labels = ["Норма/лёгкий", "Умеренный", "Тяжёлый"]
    for arm, row_axes in zip(ARMS, axes, strict=True):
        for variant, ax in zip(VARIANTS, row_axes, strict=True):
            metric = public["groups"]["rsna/L5-S1"]["models_common_cohort"][f"{arm}_{variant}"]
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
                        fontsize=17,
                        color="white" if colors[i, j] > 0.55 else "#183142",
                    )
            ax.set(
                title=f"{NAMES[arm]} · {VARIANT_NAMES[variant]}\nУмеренные {_found(metric, 'moderate')} · тяжёлые {_found(metric, 'severe')}",
                xticks=range(3),
                yticks=range(3),
                xticklabels=labels,
                yticklabels=labels,
                xlabel="Ответ модели",
                ylabel="Исходная степень",
            )
            ax.tick_params(axis="both", length=0, labelsize=9)
    fig.suptitle("RSNA L5–S1: матрицы на общих обработанных уровнях", fontsize=14)
    fig.text(
        0.5,
        0.015,
        "Числа — случаи · цвет нормирован по исходной степени",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def draw_learning(public, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    palette = ("#287c8e", "#b77b42", "#80639d")
    for arm, row_axes in zip(ARMS, axes, strict=True):
        for variant, ax in zip(VARIANTS, row_axes, strict=True):
            for seed, color in zip(SEEDS, palette, strict=True):
                run = public["runs"][f"{arm}_{variant}"][str(seed)]
                curve = run["history"]
                x = [item["epoch"] for item in curve]
                y = [item["validation_macro_f1"] for item in curve]
                ax.plot(x, y, color=color, label=f"seed {seed}", linewidth=1.5)
                selected = run["selected_epoch"]
                ax.scatter([selected], [y[selected]], marker="o", color=color, s=45, zorder=3)
            ax.set(
                title=f"{NAMES[arm]} · {VARIANT_NAMES[variant]}",
                xlabel="Эпоха",
                ylabel="VAL macro-F1",
                ylim=(0, 1),
            )
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=0.15)
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Истории отдельных запусков; точка — выбранная эпоха", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report_roi_layer4(
    summary_path=Path("reports/runs/roi-layer4-v11/summary.json"),
    *,
    out_json=Path("docs/roi-layer4-results.json"),
    out_markdown=Path("docs/roi-layer4.md"),
    comparison_image=Path("docs/images/roi_layer4_comparison.png"),
    confusion_image=Path("docs/images/roi_layer4_l5_confusion.png"),
    learning_image=Path("docs/images/roi_layer4_learning.png"),
    resamples=2000,
):
    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text())
    checksums = summary.get("input_files", {})
    tables, seed_tables, histories, hashes = {}, {}, {}, {}
    frozen_audits = []
    for arm in ARMS:
        for variant in VARIANTS:
            name = f"{arm}_{variant}"
            source = summary["arms"][arm]["variants"][variant]
            path, digest = _source(summary_path, source["ensemble"], checksums)
            tables[name] = read_validation_predictions(path)
            expected_seeds = {str(seed) for seed in SEEDS}
            for field in ("seed_predictions", "histories", "selected_epochs"):
                if set(source[field]) != expected_seeds:
                    raise ValueError("Require all three fixed seeds and their histories")
            seeds, runs, files = {}, {}, {"ensemble": digest}
            for seed in SEEDS:
                seed_path, seed_sha = _source(
                    summary_path, source["seed_predictions"][str(seed)], checksums
                )
                history_path, history_sha = _source(
                    summary_path, source["histories"][str(seed)], checksums
                )
                seeds[seed] = read_validation_predictions(seed_path)
                score = _extra_metrics(seeds[seed])["conditional"]["macro_f1"]
                runs[str(seed)] = _history(
                    json.loads(history_path.read_text()),
                    source["selected_epochs"][str(seed)],
                    score,
                )
                files[str(seed)] = {"predictions_sha256": seed_sha, "history_sha256": history_sha}
                if variant == "frozen":
                    relative_audit = (
                        Path(source["histories"][str(seed)]).parent / "audit.json"
                    ).as_posix()
                    if relative_audit in checksums:
                        audit_path, audit_sha = _source(summary_path, relative_audit, checksums)
                        audit = json.loads(audit_path.read_text())
                        frozen_audits.append(
                            audit.get("v9_reproduced") is True
                            and type(audit.get("maximum_probability_error")) in (int, float)
                            and audit["maximum_probability_error"] == 0.0
                        )
                        files[str(seed)]["control_audit_sha256"] = audit_sha
            verify_ensemble(tables[name], seeds)
            seed_tables[name], histories[name], hashes[name] = seeds, runs, files
    public = public_snapshot(summary, tables, resamples=resamples)
    public["training"] = public_training(summary)
    public["runs"] = histories
    if len(frozen_audits) == len(ARMS) * len(SEEDS) and all(frozen_audits):
        public["frozen_control_reproduced_v9"] = True
    for arm in ARMS:
        for seed in SEEDS:
            scores = [
                histories[f"{arm}_{v}"][str(seed)]["history"][0]["validation_macro_f1"]
                for v in VARIANTS
            ]
            if not np.isclose(*scores, rtol=0, atol=1e-12):
                raise ValueError(
                    "Paired variants do not start with the same epoch-zero validation score"
                )
    for level in ("all", *LEVELS):
        public["groups"][f"rsna/{level}"]["individual_seed_models_full_cohort"] = {
            name: {
                str(seed): _extra_metrics(
                    [
                        row
                        for row in seed_tables[name][seed]
                        if level == "all" or row["level"] == level
                    ]
                )
                for seed in SEEDS
            }
            for name in MODELS
        }
    public["private_input_sha256"] = hashes
    public["source_summary_sha256"] = sha256_file(summary_path)
    write_json(Path(out_json), public)
    draw_comparison(public, comparison_image)
    draw_confusion(public, confusion_image)
    draw_learning(public, learning_image)
    out_markdown = Path(out_markdown)
    out_markdown.parent.mkdir(parents=True, exist_ok=True)
    out_markdown.write_text(
        render_markdown(
            public,
            comparison_image="images/" + Path(comparison_image).name,
            confusion_image="images/" + Path(confusion_image).name,
            learning_image="images/" + Path(learning_image).name,
        )
    )
    return public


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", type=Path, default=Path("reports/runs/roi-layer4-v11/summary.json")
    )
    args = parser.parse_args(argv)
    report_roi_layer4(args.summary)


if __name__ == "__main__":
    main()
