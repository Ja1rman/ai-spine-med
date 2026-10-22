"""Paired RSNA TEST comparison of frozen v9 axial heads and the v1 sagittal ensemble.

This module only reads completed inference: no fitting, thresholds or seed choice.
Public outputs whitelist aggregates and hashes, never individual MRI/patient rows.
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
from .localization_severity_report import CLASSES, LEVELS, SEEDS, _key, verify_ensemble
from .localized_stenosis_report import read_predictions
from .uncertainty import paired_study_bootstrap

MODELS = ("xml_only", "mixed", "sagittal")
NAMES = {
    "xml_only": "Аксиальная · XML-only",
    "mixed": "Аксиальная · смешанная",
    "sagittal": "Сагиттальная · frozen-head v1",
}
COLORS = {"xml_only": "#287c8e", "mixed": "#b77b42", "sagittal": "#80639d"}
CONTRASTS = {
    "xml_only_minus_sagittal": (["xml_only"], ["sagittal"]),
    "mixed_minus_sagittal": (["mixed"], ["sagittal"]),
    "mixed_minus_xml_only": (["mixed"], ["xml_only"]),
}
BOOTSTRAP_SEED = 103017


def _paired_maps(tables):
    if set(tables) != set(MODELS):
        raise ValueError("Require the two adapted axial arms and the fixed sagittal ensemble")
    indexed = {name: {_key(row): row for row in tables[name]} for name in MODELS}
    first = indexed[MODELS[0]]
    if not first:
        raise ValueError("Empty RSNA TEST cohort")
    for name, mapping in indexed.items():
        if len(mapping) != len(tables[name]) or mapping.keys() != first.keys():
            raise ValueError("Requested RSNA patient/level cohorts differ or contain duplicates")
        for key, row in mapping.items():
            if (
                row["dataset"] != "rsna"
                or row["split"] != "test"
                or row["level"] not in LEVELS
                or row["label"] != first[key]["label"]
            ):
                raise ValueError("Require matching RSNA TEST grades, levels and reference masks")
    return indexed


def compare_group(tables, *, resamples=2000):
    """Compare exactly the same targets; preserve complete requested coverage separately."""
    indexed = _paired_maps(tables)
    first = indexed[MODELS[0]]
    common = [
        key
        for key, row in first.items()
        if row["label"] is not None and all(indexed[name][key]["status"] == "ok" for name in MODELS)
    ]
    if not common:
        raise ValueError("No commonly assessed labeled targets for three-way TEST comparison")
    paired = {name: [dict(indexed[name][key], series_id="") for key in common] for name in MODELS}
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
        "phase": "test",
        "scope": "historically_reused_test_exploratory",
        "severity_seeds": list(SEEDS),
        "fixed_before_test": True,
        "no_test_selection": True,
        "labels_joined_after_image_inference": True,
        "bootstrap_resamples": resamples,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    if any(
        summary.get(field) != expected or type(summary.get(field)) is not type(expected)
        for field, expected in required.items()
    ):
        raise ValueError("Require the sealed, inference-only exploratory TEST protocol")
    indexed = _paired_maps(tables)
    public = {
        "schema_version": 1,
        "experiment": "v9_adapted_xml_only_and_mixed_axial_vs_v1_sagittal_on_paired_rsna_test",
        **required,
        "severity_ensemble": "Equal mean of three probability vectors; no best-seed selection",
        "sagittal_baseline": "Prespecified v1 frozen-head ensemble; not the original unadapted SpineNet",
        "axial_models": "v9 separately adapted heads with frozen v7 XML-only or mixed localizers",
        "selection_scope": "All fitting and epoch choices were completed on TRAIN/VAL before this TEST inference",
        "test_limitation": "These TEST patients were already inspected in historical error analysis; exploratory follow-up, not fresh independent confirmation",
        "roi_mm": [50, 50],
        "bootstrap_resamples": resamples,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "primary_dataset": "rsna",
        "groups": {},
        "complete_case_groups": {},
    }
    complete_patients = {
        key[1]
        for key in indexed[MODELS[0]]
        if all(
            ("rsna", key[1], level) in indexed[name]
            and indexed[name][("rsna", key[1], level)]["status"] == "ok"
            and indexed[name][("rsna", key[1], level)]["label"] is not None
            for name in MODELS
            for level in LEVELS
        )
    }
    if not complete_patients:
        raise ValueError("No complete patients with both levels assessed by all models")
    for level in ("all", *LEVELS):
        selected = {
            name: [row for key, row in indexed[name].items() if level == "all" or key[2] == level]
            for name in MODELS
        }
        public["groups"][f"rsna/{level}"] = compare_group(selected, resamples=resamples)
        complete = {
            name: [row for row in rows if row["study_id"] in complete_patients]
            for name, rows in selected.items()
        }
        public["complete_case_groups"][f"rsna/{level}"] = compare_group(
            complete, resamples=resamples
        )
    return public


def _found(metric, category):
    index = CLASSES.index(category)
    matrix = metric["conditional"]["confusion_matrix"]
    return f"{matrix[index][index]}/{metric['class_counts'][category]}"


def _interval(item):
    return f"{_f(item['point'])} [{_f(item['lower_95'])}; {_f(item['upper_95'])}]"


def render_markdown(public, *, comparison_image, confusion_image):
    groups = public["groups"]
    all_group, l5_group = groups["rsna/all"], groups["rsna/L5-S1"]
    all_models = all_group["models_common_cohort"]
    l5_models = l5_group["models_common_cohort"]
    l5_full_counts = "/".join(
        str(l5_group["models_full_cohort"]["xml_only"]["class_counts"][c]) for c in CLASSES
    )
    l5_common_counts = "/".join(str(l5_models["xml_only"]["class_counts"][c]) for c in CLASSES)
    lines = [
        "# Проверка последних аксиальных моделей на RSNA TEST",
        "",
        "Я проверил два отдельно дообученных аксиальных классификатора v9: под XML-only "
        "и под смешанную локализацию AISSLab + RSNA. Для сравнения взял прежний сагиттальный "
        "ансамбль frozen-head v1. У всех трёх вариантов усреднил вероятности заранее "
        "заданных запусков 42, 2024 и 3407; лучший запуск по тесту не выбирал.",
        "",
        f"Основное сравнение — на **{all_group['n_common_labeled_assessed']} общих уровнях "
        f"у {all_group['n_common_patients']} пациентов**, где все три модели получили ответ "
        "и есть исходная метка. Полное число запрошенных уровней и отказы каждой модели "
        "сохранил отдельно. Модели, эпохи, нормализация и способ усреднения здесь не меняются.",
        "",
        "На общей выборке macro-F1: "
        + "; ".join(
            f"{NAMES[name]} — **{_f(all_models[name]['conditional']['macro_f1'])}**"
            for name in MODELS
        )
        + ".",
        "",
        "Это продолжение на прежнем TEST: его пациенты уже участвовали в прошлых "
        "разборах ошибок. Результат показывает поведение зафиксированных моделей на "
        "одинаковых пациентах, но не является новым независимым подтверждением.",
        "",
        "## Сравнение на одинаковых пациентах и уровнях",
        "",
        "F1 ниже рассчитан только на общем обработанном наборе. В каждой строке уровня "
        "у всех моделей одинаковые пациенты и метки. Для обоих уровней вместе "
        "bootstrap сохраняет уровни одного пациента вместе.",
        "",
        "| Уровень | Модель | Macro-F1, 95% ДИ | Умеренные найдены | Тяжёлые найдены | Ложный тяжёлый |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for level in ("all", *LEVELS):
        group = groups[f"rsna/{level}"]
        for name in MODELS:
            metric = group["models_common_cohort"][name]
            interval = group["bootstrap"]["models"][name]["macro_f1"]
            lines.append(
                f"| {'Оба уровня' if level == 'all' else level} | {NAMES[name]} | "
                f"{_interval(interval)} | {_found(metric, 'moderate')} | "
                f"{_found(metric, 'severe')} | {metric['false_severe']} |"
            )
    lines += [
        "",
        f"![Результаты на общей тестовой выборке]({comparison_image})",
        "",
        "## Парные разницы",
        "",
        "Положительная разница означает больший macro-F1 у первой модели. "
        "Интервал, пересекающий ноль, не показывает убедительного различия в этом сравнении.",
        "",
        "| Уровень | Сравнение | Разница macro-F1, 95% ДИ |",
        "|---|---|---:|",
    ]
    for level in ("all", *LEVELS):
        group = groups[f"rsna/{level}"]
        for contrast, (left, right) in CONTRASTS.items():
            lines.append(
                f"| {'Оба уровня' if level == 'all' else level} | "
                f"{NAMES[left[0]]} − {NAMES[right[0]]} | "
                f"{_interval(group['bootstrap']['contrasts'][contrast]['metrics']['macro_f1'])} |"
            )
    lines += [
        "",
        "## L5–S1 отдельно",
        "",
        "| Модель | Метки: норма / умеренный / тяжёлый | Умеренные найдены | Тяжёлые найдены | Тяжёлый → норма | Тяжёлый → умеренный | Ложный тяжёлый |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in MODELS:
        metric = l5_models[name]
        severe = metric["severe"]
        counts = "/".join(str(metric["class_counts"][c]) for c in CLASSES)
        lines.append(
            f"| {NAMES[name]} | {counts} | {_found(metric, 'moderate')} | "
            f"{_found(metric, 'severe')} | {severe['predicted_normal_mild']} | "
            f"{severe['predicted_moderate']} | {metric['false_severe']} |"
        )
    for contrast, (left, right) in CONTRASTS.items():
        changes = l5_group["paired_changes"][contrast]
        lines += [
            "",
            f"L5–S1, «{NAMES[left[0]]}» против «{NAMES[right[0]]}»: "
            f"ошибок исправлено: {changes['improved']}; правильных ответов потеряно: "
            f"{changes['worsened']}. Среди тяжёлых первая модель дополнительно нашла "
            f"{changes['severe_improved']}, потеряла {changes['severe_worsened']}; "
            f"обе пропустили {changes['severe_both_missed']}.",
        ]
    lines += [
        "",
        f"![L5–S1: матрицы ошибок]({confusion_image})",
        "",
        "Строки — исходная степень, столбцы — ответ. Числа — случаи; цвет нормирован "
        "внутри исходной степени. Здесь только общий обработанный набор, а не вся "
        "запрошенная выборка. При небольшом числе умеренных и тяжёлых L5–S1 один ответ "
        "сильно меняет метрики; высокая точность нормы не означает, что стеноз этого уровня решён.",
        "",
        "## Покрытие всех запрошенных целей",
        "",
        "Отказы обработки не превращал в «норму». F1 полного набора условен на тех "
        "случаях, которые конкретная модель обработала; поэтому напрямую сравнивать "
        "эти F1 вместо парного набора выше нельзя.",
        "",
        "| Уровень | Модель | Запрошено | Без метки | Обработано с меткой | Не обработано с меткой | F1 обработанных | Тяжёлые найдены / всех запрошенных тяжёлых |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for level in ("all", *LEVELS):
        for name in MODELS:
            metric = groups[f"rsna/{level}"]["models_full_cohort"][name]
            lines.append(
                f"| {'Оба уровня' if level == 'all' else level} | {NAMES[name]} | {metric['n_requested']} | "
                f"{metric['n_missing_reference_label']} | {metric['n_assessed']} | "
                f"{metric['n_not_assessed']} | {_f(metric['conditional']['macro_f1'])} | "
                f"{_found(metric, 'severe')} |"
            )
    lines += [
        "",
        f"На L5–S1 число исходных меток норма/лёгкий / умеренный / тяжёлый: "
        f"{l5_full_counts} во всей запрошенной выборке и {l5_common_counts} в общем наборе. "
        "Исключённые из парного сравнения метки не потеряны: они входят в таблицу покрытия и полные метрики.",
        "",
        "Число уровней, исключённых из общего сравнения: "
        f"{all_group['n_excluded_from_common']}. "
        "Полные и парные метрики, исходные числа классов и индивидуальные результаты "
        "трёх запусков сохранены в JSON. Отдельные запуски показаны для воспроизводимости, "
        "а не для выбора удачного seed.",
        "",
        "## Дополнительная проверка: у каждого пациента обработаны оба уровня",
        "",
        "В основном парном наборе исключается только конкретный необработанный уровень. "
        "Поэтому число пациентов на L4–L5 и L5–S1 может различаться. Дополнительно "
        f"оставил набор из {public['complete_case_groups']['rsna/all']['n_common_patients']} "
        "пациентов, у которых все три модели обработали оба уровня: "
        f"{public['complete_case_groups']['rsna/all']['n_common_labeled_assessed']} целей. "
        "Это отдельный анализ чувствительности покрытия; модели и критерии не менял.",
        "",
        "| Уровень | Модель | Macro-F1, 95% ДИ | Тяжёлые найдены |",
        "|---|---|---:|---:|",
    ]
    for level in ("all", *LEVELS):
        group = public["complete_case_groups"][f"rsna/{level}"]
        for name in MODELS:
            metric = group["models_common_cohort"][name]
            lines.append(
                f"| {'Оба уровня' if level == 'all' else level} | {NAMES[name]} | "
                f"{_interval(group['bootstrap']['models'][name]['macro_f1'])} | "
                f"{_found(metric, 'severe')} |"
            )
    lines += [
        "",
        "## Что именно проверял",
        "",
        "Аксиальные модели используют замороженные локализаторы v7, прежнюю сеть центра "
        "и область 50×50 мм. Классификаторы v9 ранее отдельно адаптировал только на "
        "RSNA TRAIN; эпохи выбраны по RSNA VAL. Сагиттальный вариант — обученная голова "
        "на признаках SpineNet, а не исходный SpineNet без адаптации. Сейчас не "
        "дообучал веса, не подбирал пороги и не выбирал эпохи или seed по TEST. "
        "Истинные степени не используются при выборе среза и ROI.",
        "",
        f"95% интервалы — {public['bootstrap_resamples']} парных bootstrap-повторов по "
        "пациентам. Они описывают разброс состава этой выборки для фиксированных моделей; "
        "не учитывают ошибки меток, новые клиники, разброс повторного обучения или "
        "историю использования TEST. Для независимой оценки нужен ещё не просмотренный набор.",
        "",
        "Агрегаты: [localization-test-results.json](localization-test-results.json). "
        "Предыдущее сравнение на валидации: "
        "[localization-finetuning.md](localization-finetuning.md). Снимки, веса и "
        "индивидуальные ответы сохраняю локально; в Git попадают только код и агрегаты.",
        "",
        "## Пересобрать отчёт",
        "",
        "```bash",
        "python -m ai_spine_med.localization_test_report",
        "```",
        "",
        "Команда читает уже полученные тестовые ответы и не запускает обучение.",
        "",
    ]
    return "\n".join(lines)


def draw_comparison(public, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), sharey=True)
    for ax, level in zip(axes, ("all", *LEVELS), strict=True):
        group = public["groups"][f"rsna/{level}"]
        for x, name in enumerate(MODELS):
            value = group["bootstrap"]["models"][name]["macro_f1"]
            point = value["point"]
            ax.errorbar(
                x,
                point,
                yerr=[[max(0, point - value["lower_95"])], [max(0, value["upper_95"] - point)]],
                fmt="o",
                ms=10,
                capsize=6,
                color=COLORS[name],
                lw=2,
            )
            ax.annotate(
                _f(point),
                (x, value["upper_95"]),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=11,
            )
        ax.set(
            title=("Оба уровня" if level == "all" else level)
            + f" · {group['n_common_labeled_assessed']} уровней",
            xticks=range(3),
            xticklabels=["XML-only", "Смешанная", "Сагиттальная"],
            xlim=(-0.5, 2.5),
            ylim=(0, 1),
            ylabel="macro-F1 и 95% ДИ",
        )
        ax.tick_params(axis="x", labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
    fig.suptitle("RSNA TEST: три зафиксированных ансамбля на одинаковых пациентах", fontsize=14)
    fig.text(
        0.5,
        0.015,
        "Парный bootstrap по пациентам · TEST использовался в прежних разборах · без обучения и выбора seed",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.93))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def draw_confusion(public, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    group = public["groups"]["rsna/L5-S1"]
    for name, ax in zip(MODELS, axes, strict=True):
        metric = group["models_common_cohort"][name]
        matrix = np.asarray(metric["conditional"]["confusion_matrix"])
        totals = matrix.sum(axis=1, keepdims=True)
        values = np.divide(matrix, totals, out=np.zeros_like(matrix, dtype=float), where=totals > 0)
        ax.imshow(values, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(
                    j,
                    i,
                    str(matrix[i, j]),
                    ha="center",
                    va="center",
                    fontsize=16,
                    color="white" if values[i, j] > 0.55 else "#183142",
                )
        ax.set(
            title=f"{NAMES[name]}\nУмеренные {_found(metric, 'moderate')} · тяжёлые {_found(metric, 'severe')}",
            xticks=range(3),
            yticks=range(3),
            xticklabels=["Норма/лёгкий", "Умеренный", "Тяжёлый"],
            yticklabels=["Норма/лёгкий", "Умеренный", "Тяжёлый"],
            xlabel="Ответ модели",
            ylabel="Исходная степень",
        )
        ax.tick_params(axis="both", labelsize=9, length=0)
        ax.text(
            0.5,
            -0.25,
            f"Ложный тяжёлый: {metric['false_severe']}",
            transform=ax.transAxes,
            ha="center",
            fontsize=10,
        )
    fig.suptitle("RSNA TEST L5–S1: матрицы на общем обработанном наборе", fontsize=14)
    fig.text(
        0.5,
        0.01,
        "Цвет нормирован по исходной степени · число случаев важнее одного процента",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def report_localization_test(
    summary_path=Path("reports/runs/localization-test-v10/summary.json"),
    *,
    out_json=Path("docs/localization-test-results.json"),
    out_markdown=Path("docs/localization-test.md"),
    comparison_image=Path("docs/images/localization_test_comparison.png"),
    confusion_image=Path("docs/images/localization_test_l5_confusion.png"),
    resamples=2000,
):
    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text())
    if set(summary.get("models", {})) != set(MODELS):
        raise ValueError("Require exactly three predetermined model families")
    tables, hashes, individual = {}, {}, {}
    for name in MODELS:
        source = summary["models"][name]
        path = Path(source["predictions"])
        if not path.is_absolute():
            path = summary_path.parent / path
        checksum = sha256_file(path)
        if source.get("predictions_sha256") != checksum:
            raise ValueError("Prediction CSV changed after inference was sealed")
        tables[name] = read_predictions(path, default_dataset="rsna")
        seed_paths = source["seed_predictions"]
        if set(seed_paths) != {str(seed) for seed in SEEDS}:
            raise ValueError("Require all three predetermined seed prediction paths")
        seeds, seed_hashes = {}, {}
        for seed in SEEDS:
            seed_path = Path(seed_paths[str(seed)])
            if not seed_path.is_absolute():
                seed_path = summary_path.parent / seed_path
            seeds[seed] = read_predictions(seed_path, default_dataset="rsna")
            seed_hashes[str(seed)] = sha256_file(seed_path)
        verify_ensemble(tables[name], seeds)
        individual[name] = seeds
        hashes[name] = {"ensemble_sha256": checksum, "seeds": seed_hashes}
    public = public_snapshot(summary, tables, resamples=resamples)
    for level in ("all", *LEVELS):
        public["groups"][f"rsna/{level}"]["individual_seed_models_full_cohort"] = {
            name: {
                str(seed): _extra_metrics(
                    [
                        row
                        for row in individual[name][seed]
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
        "--summary", type=Path, default=Path("reports/runs/localization-test-v10/summary.json")
    )
    args = parser.parse_args(argv)
    report_localization_test(args.summary)


if __name__ == "__main__":
    main()
