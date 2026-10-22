"""Validation-only sensitivity of a frozen severity classifier to level localization.

Patient-level CSV files stay private. All public results are recalculated from
strictly joined predictions; arbitrary input metadata never enters publication.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .assets import sha256_file, write_json
from .probability_mean_report import _metric_snapshot
from .training import evaluate_predictions
from .uncertainty import paired_study_bootstrap

ARMS = ("xml_only", "mixed")
SEEDS = (42, 2024, 3407)
DATASETS = ("rsna", "aisslab")
LEVELS = ("L4-L5", "L5-S1")
CLASSES = ("normal_mild", "moderate", "severe")
NAMES = {"xml_only": "Только XML AISSLab", "mixed": "AISSLab + RSNA"}
COLORS = {"xml_only": "#287c8e", "mixed": "#be8757"}


def _key(row):
    return row["dataset"], row["study_id"], row["level"]


def read_validation_predictions(path):
    """Keep missing references and real failures; forbid non-VAL predictions."""
    required = {
        "dataset",
        "study_id",
        "split",
        "level",
        "label",
        "status",
        "reason",
        "prediction",
        *(f"p_{name}" for name in CLASSES),
    }
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("Missing validation prediction fields")
        raw = list(reader)
    if not raw:
        raise ValueError("Empty validation predictions")
    result, seen = [], set()
    for source in raw:
        row = {name: source[name] for name in required}
        if (
            row["dataset"] not in DATASETS
            or row["split"] != "validation"
            or row["level"] not in LEVELS
            or not row["study_id"]
            or _key(row) in seen
        ):
            raise ValueError("Invalid, duplicate or non-validation target")
        seen.add(_key(row))
        try:
            row["label"] = int(row["label"]) if row["label"] else None
            row["prediction"] = int(row["prediction"]) if row["prediction"] else None
        except (ValueError, TypeError):
            raise ValueError("Invalid category") from None
        if row["label"] is not None and row["label"] not in (0, 1, 2):
            raise ValueError("Invalid reference grade")
        if row["status"] == "ok":
            try:
                p = np.array([row[f"p_{name}"] for name in CLASSES], dtype=float)
            except (TypeError, ValueError):
                raise ValueError("Invalid probability vector") from None
            if (
                row["reason"]
                or not np.isfinite(p).all()
                or (p < 0).any()
                or (p > 1).any()
                or not np.isclose(p.sum(), 1, atol=1e-6, rtol=0)
                or row["prediction"] != int(p.argmax())
            ):
                raise ValueError("Invalid probability vector or argmax")
            row.update({f"p_{name}": float(value) for name, value in zip(CLASSES, p)})
        else:
            if (
                row["status"] not in ("not_assessed", "failed")
                or not row["reason"]
                or row["prediction"] is not None
                or any(row[f"p_{name}"] != "" for name in CLASSES)
            ):
                raise ValueError("A failed target needs a reason and no imputed probabilities")
            row["status"] = "not_assessed"
            row.update({f"p_{name}": None for name in CLASSES})
        result.append(row)
    return result


def _paired_maps(tables):
    if set(tables) != set(ARMS):
        raise ValueError("Require both predetermined localization arms")
    indexed = {arm: {_key(r): r for r in tables[arm]} for arm in ARMS}
    first = indexed[ARMS[0]]
    if any(len(indexed[arm]) != len(tables[arm]) for arm in ARMS):
        raise ValueError("Duplicate patient/level target")
    if any(mapping.keys() != first.keys() for mapping in indexed.values()):
        raise ValueError("Arms have different requested patient/level cohorts")
    for key, row in first.items():
        if indexed[ARMS[1]][key]["label"] != row["label"]:
            raise ValueError("Arms have different reference grades or missing-reference masks")
    return indexed


def verify_ensemble(ensemble, seed_tables):
    """A three-seed probability mean must match, independently of CSV row order."""
    if set(seed_tables) != set(SEEDS):
        raise ValueError("Require all three predetermined severity seeds")
    indexed = [{_key(r): r for r in seed_tables[seed]} for seed in SEEDS]
    expected = {_key(row): row for row in ensemble}
    if any(len(m) != len(seed_tables[s]) for m, s in zip(indexed, SEEDS)) or any(
        mapping.keys() != expected.keys() for mapping in indexed
    ):
        raise ValueError("Severity seed cohorts differ from the ensemble")
    for key, row in expected.items():
        triple = [mapping[key] for mapping in indexed]
        if any((r["label"], r["status"]) != (row["label"], row["status"]) for r in triple):
            raise ValueError("Severity seed reference or coverage differs")
        if row["status"] == "ok":
            expected_p = np.mean([[r[f"p_{c}"] for c in CLASSES] for r in triple], axis=0)
            if not np.allclose(
                expected_p, [row[f"p_{c}"] for c in CLASSES], rtol=0, atol=1e-7
            ) or row["prediction"] != int(expected_p.argmax()):
                raise ValueError("Published ensemble is not the equal seed probability mean")


def _metrics(rows):
    result = _metric_snapshot(evaluate_predictions(rows), levels=False)
    result["n_patients"] = len({r["study_id"] for r in rows})
    result["n_predictions_including_missing_reference"] = sum(r["status"] == "ok" for r in rows)
    return result


def compare_group(tables, *, resamples=2000):
    """Keep full coverage separately; paired CIs use common assessed labels only."""
    indexed = _paired_maps(tables)
    first = indexed[ARMS[0]]
    if not first or len({k[0] for k in first}) != 1:
        raise ValueError("Compare one nonempty dataset at a time")
    common = [
        key
        for key, row in first.items()
        if row["label"] is not None and all(indexed[arm][key]["status"] == "ok" for arm in ARMS)
    ]
    paired = {arm: [dict(indexed[arm][key], series_id="") for key in common] for arm in ARMS}
    bootstrap = None
    if common:
        bootstrap = paired_study_bootstrap(
            paired,
            contrasts={"mixed_minus_xml_only": (["mixed"], ["xml_only"])},
            resamples=resamples,
            seed=83017,
        )
        # Fixed text plus recalculated numeric aggregates; no arbitrary metadata.
        bootstrap["method"] = "Paired patient-level percentile bootstrap; both levels stay together"
    changes = {
        "n_common": len(common),
        "same_category": 0,
        "xml_only_correct": 0,
        "mixed_only_correct": 0,
        "both_correct": 0,
        "both_wrong": 0,
        "severe_xml_only_detected": 0,
        "severe_mixed_only_detected": 0,
        "severe_both_detected": 0,
        "severe_both_missed": 0,
    }
    for key in common:
        a, b = [indexed[arm][key] for arm in ARMS]
        ac, bc = a["prediction"] == a["label"], b["prediction"] == b["label"]
        changes["same_category"] += a["prediction"] == b["prediction"]
        changes[
            "both_correct"
            if ac and bc
            else "xml_only_correct"
            if ac
            else "mixed_only_correct"
            if bc
            else "both_wrong"
        ] += 1
        if a["label"] == 2:
            changes[
                "severe_both_detected"
                if ac and bc
                else "severe_xml_only_detected"
                if ac
                else "severe_mixed_only_detected"
                if bc
                else "severe_both_missed"
            ] += 1
    return {
        "n_requested": len(first),
        "n_common_labeled_assessed": len(common),
        "n_excluded_from_common": len(first) - len(common),
        "models_full_cohort": {arm: _metrics(tables[arm]) for arm in ARMS},
        "models_common_cohort": {arm: _metrics(paired[arm]) for arm in ARMS},
        "paired_changes": {k: int(v) for k, v in changes.items()},
        "bootstrap": bootstrap,
    }


def public_snapshot(summary, tables, *, resamples=2000):
    required = {
        "phase": "validation",
        "center_reused_exactly": True,
        "severity_retrained": False,
        "test_evaluated": False,
        "localizer_ensemble": "mean_probabilities_before_argmax",
        "grade_labels_joined_after_both_arms_image_inference": True,
        "bootstrap_resamples": resamples,
        "bootstrap_seed": 83017,
        "severity_seeds": list(SEEDS),
    }
    if any(
        summary.get(key) != value or type(summary.get(key)) is not type(value)
        for key, value in required.items()
    ):
        raise ValueError("Require the fixed VAL-only protocol with unchanged center and severity")
    indexed = _paired_maps(tables)
    if any(r["split"] != "validation" for table in tables.values() for r in table):
        raise ValueError("All report predictions must be validation-only")
    result = {
        "schema_version": 1,
        "experiment": "xml_only_vs_mixed_localization_frozen_severity",
        **required,
        "severity_ensemble": "equal mean of three frozen v6 severity probability vectors",
        "severity_training_source": "RSNA only; originally fitted on v6 localized ROIs",
        "roi_mm": [50, 50],
        "scope": "Reused development validation; fixed classifier sensitivity to ROI selection",
        "primary_dataset": "rsna",
        "aisslab_grade_crosswalk_verified": False,
        "groups": {},
    }
    for dataset in DATASETS:
        for level in ("all", *LEVELS):
            selected = {
                arm: [
                    r
                    for key, r in indexed[arm].items()
                    if key[0] == dataset and (level == "all" or key[2] == level)
                ]
                for arm in ARMS
            }
            result["groups"][f"{dataset}/{level}"] = compare_group(selected, resamples=resamples)
    return result


def _f(value):
    return "—" if value is None else f"{value:.3f}".replace(".", ",")


def _detected(metric):
    severe = metric["severe"]
    return f"{severe['correctly_predicted_severe']}/{severe['n_total']}"


def _difference(group):
    if group["bootstrap"] is None:
        return "—"
    value = group["bootstrap"]["contrasts"]["mixed_minus_xml_only"]["metrics"]["macro_f1"]
    return f"{_f(value['point'])} [{_f(value['lower_95'])}; {_f(value['upper_95'])}]"


def render_markdown(public, *, comparison_image, confusion_image):
    groups = public["groups"]
    all_rsna = groups["rsna/all"]
    primary = all_rsna["bootstrap"]
    if primary is None:
        finding = "Для парного сравнения не оказалось общих обработанных случаев с метками."
    else:
        delta = primary["contrasts"]["mixed_minus_xml_only"]["metrics"]["macro_f1"]
        if delta["lower_95"] <= 0 <= delta["upper_95"]:
            finding = "Убедительного различия по общему macro-F1 RSNA в этом сравнении нет: интервал разницы включает ноль."
        elif delta["lower_95"] > 0:
            finding = "На этой валидации смешанная локализация даёт более высокий общий macro-F1 RSNA; парный интервал разницы не включает ноль."
        else:
            finding = "На этой валидации локализация только по XML даёт более высокий общий macro-F1 RSNA; парный интервал разницы не включает ноль."
    l5 = groups["rsna/L5-S1"]["models_full_cohort"]
    l4 = groups["rsna/L4-L5"]["models_full_cohort"]
    moderate = {
        arm: f"{l5[arm]['conditional']['confusion_matrix'][1][1]}/"
        f"{l5[arm]['class_counts']['moderate']}"
        for arm in ARMS
    }
    tradeoff = (
        f"На RSNA тяжёлые L5–S1: {_detected(l5['xml_only'])} с XML и "
        f"{_detected(l5['mixed'])} со смешанной локализацией; умеренные L5–S1: "
        f"{moderate['xml_only']} и {moderate['mixed']}. Тяжёлые L4–L5: "
        f"{_detected(l4['xml_only'])} и {_detected(l4['mixed'])}. Поэтому один общий macro-F1 "
        "не заменяет разбор ошибок по отдельным степеням и уровням."
    )
    l5_changes = groups["rsna/L5-S1"]["paired_changes"]
    changed = l5_changes["n_common"] - l5_changes["same_category"]
    changed_text = (
        f"При замене XML-локализации на смешанную ответ RSNA L5–S1 изменился в {changed} случаях: "
        f"{l5_changes['mixed_only_correct']} исходно ошибочных ответов стали правильными, "
        f"{l5_changes['xml_only_correct']} правильных — ошибочными."
    )
    lines = [
        "# Как локализация влияет на оценку стеноза",
        "",
        "Я подставил два варианта выбора уровней L4–L5 и L5–S1 в одну и ту же систему оценки степени стеноза. "
        "В первом локализатор обучался только по XML AISSLab, во втором — по AISSLab и RSNA. "
        "Модель центра области, размер ROI 50×50 мм, обработку изображений, исходный ResNet18 и три "
        "классификатора v6 оставил без изменений. Нового обучения здесь нет.",
        "",
        finding,
        "",
        tradeoff,
        "",
        changed_text,
        "",
        "## RSNA: основное сравнение",
        "",
        "В таблице macro-F1 и найденные тяжёлые случаи посчитаны на полном доступном наборе каждой "
        "ветки. Разница macro-F1 и 95% интервал — на одних и тех же пациентах с меткой и результатом "
        "обеих веток. Два уровня пациента остаются вместе в bootstrap.",
        "",
        "| Уровень | XML, macro-F1 | Смешанная, macro-F1 | Смешанная − XML, 95% ДИ | Тяжёлые найдены: XML / смешанная |",
        "|---|---:|---:|---:|---:|",
    ]
    for level in ("all", *LEVELS):
        group = groups[f"rsna/{level}"]
        a, b = [group["models_full_cohort"][arm] for arm in ARMS]
        lines.append(
            f"| {'Оба уровня' if level == 'all' else level} | {_f(a['conditional']['macro_f1'])} | "
            f"{_f(b['conditional']['macro_f1'])} | {_difference(group)} | {_detected(a)} / {_detected(b)} |"
        )
    lines += ["", f"![Результаты оценки стеноза]({comparison_image})", "", "## Отдельно L5–S1", ""]
    lines += [
        "| Данные и локализация | Метки: норма / умеренный / тяжёлый | Умеренный найден | Тяжёлый найден | Тяжёлый → норма | Тяжёлый → умеренный | Тяжёлый не обработан |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    change_paragraphs = []
    for dataset in DATASETS:
        group = groups[f"{dataset}/L5-S1"]
        for arm in ARMS:
            metric = group["models_full_cohort"][arm]
            s = metric["severe"]
            classes = "/".join(str(metric["class_counts"][c]) for c in CLASSES)
            moderate = metric["conditional"]["confusion_matrix"][1][1]
            n_moderate = metric["class_counts"]["moderate"]
            lines.append(
                f"| {dataset.upper()} · {NAMES[arm]} | {classes} | {moderate}/{n_moderate} | "
                f"{_detected(metric)} | "
                f"{s['predicted_normal_mild']} | {s['predicted_moderate']} | {s['not_assessed']} |"
            )
        changes = group["paired_changes"]
        change_paragraphs.extend(
            [
                "",
                f"На {dataset.upper()} среди общих обработанных тяжёлых L5–S1: "
                f"только XML нашёл {changes['severe_xml_only_detected']}, только смешанный вариант — "
                f"{changes['severe_mixed_only_detected']}, оба нашли {changes['severe_both_detected']}, "
                f"оба пропустили {changes['severe_both_missed']}.",
                "",
            ]
        )
    lines.extend(change_paragraphs)
    lines.extend(
        [
            f"![Матрицы ошибок L5–S1]({confusion_image})",
            "",
            "Строки матриц — исходная метка, столбцы — ответ модели. Порядок: норма/лёгкий, умеренный, "
            "тяжёлый. Матрицы включают только обработанные случаи с известной меткой; пропуски обработки "
            "остаются в таблице и JSON.",
            "",
            "## AISSLab: дополнительная проверка",
            "",
            "Соответствие Normal / Stenosis / Severe Stenosis категориям RSNA остаётся условным. "
            "Эти результаты показываю отдельно и не объединяю с RSNA.",
            "",
            "| Уровень | XML, macro-F1 | Смешанная, macro-F1 | Смешанная − XML, 95% ДИ |",
            "|---|---:|---:|---:|",
        ]
    )
    for level in ("all", *LEVELS):
        group = groups[f"aisslab/{level}"]
        a, b = [group["models_full_cohort"][arm] for arm in ARMS]
        lines.append(
            f"| {'Оба уровня' if level == 'all' else level} | {_f(a['conditional']['macro_f1'])} | "
            f"{_f(b['conditional']['macro_f1'])} | {_difference(group)} |"
        )
    lines.extend(
        [
            "",
            "## Покрытие",
            "",
            "| Данные | Запрошено | Без метки | Обработано с меткой: XML / смешанная | Общих для парного сравнения |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for dataset in DATASETS:
        group = groups[f"{dataset}/all"]
        a, b = [group["models_full_cohort"][arm] for arm in ARMS]
        lines.append(
            f"| {dataset.upper()} | {group['n_requested']} | {a['n_missing_reference_label']} | "
            f"{a['n_assessed']} / {b['n_assessed']} | {group['n_common_labeled_assessed']} |"
        )
    lines.extend(
        [
            "",
            "## Что означает этот эксперимент",
            "",
            "Каждая ветка сначала усредняет вероятности срезов трёх локализаторов, затем выбирает срез "
            "и строит ROI. Степень стеноза определяется равным усреднением вероятностей трёх уже обученных "
            "классификаторов v6. Метки степени не участвуют в выборе среза или вырезании ROI.",
            "",
            "Классификаторы v6 обучались на ROI прежнего локализатора v6 и только на RSNA TRAIN. Здесь "
            "проверяется их чувствительность к замене локализации, а не лучший возможный классификатор "
            "для каждой ветки. Отдельное переобучение классификаторов на новых ROI могло бы дать другие результаты.",
            "",
            "Это повторно используемая валидация, а не независимый тест. AISSLab VAL уже участвовал в выборе "
            "эпох локализаторов, RSNA VAL — в выборе эпох классификаторов v6. TEST не оценивал, веса не "
            f"обновлял, пороги по этим результатам не подбирал. Интервалы получены {public['bootstrap_resamples']} парными "
            "bootstrap-повторами по пациентам для фиксированных моделей: они не учитывают "
            "неопределённость врачебной разметки и нового обучения.",
            "",
            "Попадание в нужный уровень ещё не гарантирует правильную степень стеноза: важны конкретный "
            "срез, положение центра и содержимое ROI. Проверка с врачом нужна прежде всего для "
            "пропущенных тяжёлых L5–S1. При малом числе тяжёлых случаев счётчики информативнее одного процента.",
            "",
            "Агрегированные результаты: [localization-severity-results.json](localization-severity-results.json). "
            "Предыдущий этап: [сравнение локализаторов](xml-localizer.md). Снимки и ответы по отдельным "
            "пациентам остаются в локальных игнорируемых папках.",
            "",
            "## Повторить",
            "",
            "Протокол: [localization-severity-v8.json](../configs/localization-severity-v8.json).",
            "",
            "```bash",
            "python -m ai_spine_med.localization_severity_validation --device mps",
            "python -m ai_spine_med.localization_severity_report",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def draw_comparison(public, path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    x = np.arange(3)
    for ax, dataset in zip(axes, DATASETS):
        for offset, arm in zip((-0.18, 0.18), ARMS):
            values = [
                public["groups"][f"{dataset}/{level}"]["models_full_cohort"][arm]["conditional"][
                    "macro_f1"
                ]
                for level in ("all", *LEVELS)
            ]
            bars = ax.bar(
                x + offset, [v or 0 for v in values], 0.32, color=COLORS[arm], label=NAMES[arm]
            )
            for bar, value in zip(bars, values):
                ax.annotate(
                    _f(value),
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
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
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle("Замена локализации при неизменном классификаторе стеноза", fontsize=14)
    fig.text(
        0.5,
        0.025,
        "Повторно используемая VAL · три фиксированных классификатора v6 · новые веса не обучались",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def draw_confusion(public, path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for dataset, row_axes in zip(DATASETS, axes):
        for arm, ax in zip(ARMS, row_axes):
            metric = public["groups"][f"{dataset}/L5-S1"]["models_full_cohort"][arm]
            matrix = np.asarray(metric["conditional"]["confusion_matrix"])
            # Row-normalized shade, while annotated numbers preserve rare-case counts.
            sums = matrix.sum(axis=1, keepdims=True)
            shaded = np.divide(matrix, sums, out=np.zeros_like(matrix, dtype=float), where=sums > 0)
            ax.imshow(shaded, cmap="Blues", vmin=0, vmax=1)
            for i in range(3):
                for j in range(3):
                    ax.text(
                        j,
                        i,
                        str(matrix[i, j]),
                        ha="center",
                        va="center",
                        fontsize=15,
                        color="white" if shaded[i, j] > 0.55 else "#183142",
                    )
            ax.set(
                title=f"{dataset.upper()} · {NAMES[arm]}",
                xticks=range(3),
                yticks=range(3),
                xticklabels=["Норма/лёгкий", "Умеренный", "Тяжёлый"],
                yticklabels=["Норма/лёгкий", "Умеренный", "Тяжёлый"],
                xlabel="Ответ модели",
                ylabel="Исходная метка",
            )
            ax.tick_params(axis="both", length=0, labelsize=9)
    fig.suptitle("L5–S1: куда попадают исходные степени стеноза", fontsize=14)
    fig.text(
        0.5,
        0.015,
        "Числа — обработанные случаи с меткой · цвет нормирован по строкам · AISSLab: условное соответствие категорий",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def report_localization_severity(
    summary_path=Path("reports/runs/localization-severity-v8/summary.json"),
    *,
    out_json=Path("docs/localization-severity-results.json"),
    out_markdown=Path("docs/localization-severity.md"),
    comparison_image=Path("docs/images/localization_severity_comparison.png"),
    confusion_image=Path("docs/images/localization_severity_l5_confusion.png"),
    resamples=2000,
):
    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text())
    if set(summary["arms"]) != set(ARMS):
        raise ValueError("Unexpected localization arms")
    tables, hashes = {}, {}
    for arm in ARMS:
        source = summary["arms"][arm]
        path = Path(source["predictions"])
        if not path.is_absolute():
            path = summary_path.parent / path
        tables[arm] = read_validation_predictions(path)
        hashes[arm] = {"ensemble_sha256": sha256_file(path), "seeds": {}}
        if source.get("predictions_sha256", sha256_file(path)) != sha256_file(path):
            raise ValueError("Prediction CSV changed after the inference summary was sealed")
        paths = source["seed_predictions"]
        if set(paths) != {str(seed) for seed in SEEDS}:
            raise ValueError("Require paths for all three frozen severity seed predictions")
        seed_tables = {}
        for seed in SEEDS:
            seed_path = Path(paths[str(seed)])
            if not seed_path.is_absolute():
                seed_path = summary_path.parent / seed_path
            seed_tables[seed] = read_validation_predictions(seed_path)
            hashes[arm]["seeds"][str(seed)] = sha256_file(seed_path)
        verify_ensemble(tables[arm], seed_tables)
    public = public_snapshot(summary, tables, resamples=resamples)
    public["private_input_sha256"] = hashes
    public["source_summary_sha256"] = sha256_file(summary_path)
    write_json(Path(out_json), public)
    comparison_image, confusion_image = Path(comparison_image), Path(confusion_image)
    draw_comparison(public, comparison_image)
    draw_confusion(public, confusion_image)
    out_markdown = Path(out_markdown)
    out_markdown.parent.mkdir(parents=True, exist_ok=True)
    out_markdown.write_text(
        render_markdown(
            public,
            comparison_image="images/" + comparison_image.name,
            confusion_image="images/" + confusion_image.name,
        )
    )
    return public


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", type=Path, default=Path("reports/runs/localization-severity-v8/summary.json")
    )
    parser.add_argument(
        "--out-json", type=Path, default=Path("docs/localization-severity-results.json")
    )
    parser.add_argument("--out-markdown", type=Path, default=Path("docs/localization-severity.md"))
    parser.add_argument(
        "--comparison-image",
        type=Path,
        default=Path("docs/images/localization_severity_comparison.png"),
    )
    parser.add_argument(
        "--confusion-image",
        type=Path,
        default=Path("docs/images/localization_severity_l5_confusion.png"),
    )
    args = parser.parse_args(argv)
    report_localization_severity(
        args.summary,
        out_json=args.out_json,
        out_markdown=args.out_markdown,
        comparison_image=args.comparison_image,
        confusion_image=args.confusion_image,
    )


if __name__ == "__main__":
    main()
