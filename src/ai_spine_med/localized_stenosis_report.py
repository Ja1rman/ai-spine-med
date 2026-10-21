"""Verified public aggregates for explicit axial localization and ROI severity.

Source prediction tables are private. Publication uses numeric/typed whitelists,
strict patient/level joins, equal assessed cohorts and patient-cluster bootstrap.
Previously inspected TEST remains an exploratory follow-up, not a fresh holdout.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .assets import sha256_file, write_json
from .localized_stenosis import CLASSES, LEVELS, SEEDS, load_sealed_models
from .probability_mean_report import _metric_snapshot, _number, _same_tree
from .training import evaluate_predictions
from .uncertainty import paired_study_bootstrap

DATASETS = ("rsna", "aisslab")
NAMES = {
    "before": "Прежняя аксиальная",
    "after": "После переноса v4",
    "localized": "Уровень + канал + ROI",
    "sagittal": "Сагиттальный ансамбль",
}
COLORS = {"before": "#6c7d91", "after": "#b78172", "localized": "#267f82", "sagittal": "#8c6ea4"}


def _json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Expected report metadata object")
    return value


def _key(row):
    return row["dataset"], row["study_id"], row["level"]


def read_predictions(path, *, default_dataset=None):
    """Read strict severity probabilities; accept the older sagittal CSV schema."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"study_id", "split", "level", "label", "status", "reason", "prediction"}
        required |= {f"p_{name}" for name in CLASSES}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("Prediction table lacks required fields")
        raw = list(reader)
    rows, seen = [], set()
    for item in raw:
        row = {name: item[name] for name in required}
        row["dataset"] = item.get("dataset", default_dataset)
        if (
            row["dataset"] not in DATASETS
            or not row["study_id"]
            or row["split"] != "test"
            or row["level"] not in LEVELS
            or _key(row) in seen
        ):
            raise ValueError("Invalid, duplicate or non-TEST prediction identity")
        seen.add(_key(row))
        try:
            row["label"] = int(row["label"]) if row["label"] else None
            row["prediction"] = int(row["prediction"]) if row["prediction"] else None
        except (ValueError, TypeError):
            raise ValueError("Invalid prediction/reference category") from None
        if row["label"] is not None and row["label"] not in (0, 1, 2):
            raise ValueError("Invalid reference category")
        availability = item.get("reference_status")
        if availability is not None and availability not in (
            ("missing",) if row["label"] is None else ("known", "available")
        ):
            raise ValueError("Reference availability differs from its label")
        if row["status"] == "ok":
            probability = np.asarray([item[f"p_{name}"] for name in CLASSES], dtype=float)
            if (
                row["reason"]
                or not np.isfinite(probability).all()
                or (probability < 0).any()
                or (probability > 1).any()
                or not np.isclose(probability.sum(), 1, rtol=0, atol=1e-6)
                or row["prediction"] != int(probability.argmax())
            ):
                raise ValueError("Invalid probability vector or predicted category")
            row.update({f"p_{name}": float(value) for name, value in zip(CLASSES, probability)})
        elif (
            row["status"] not in {"not_assessed", "failed"}
            or not row["reason"]
            or row["prediction"] is not None
            or any(item[f"p_{name}"] != "" for name in CLASSES)
        ):
            raise ValueError("A failed target needs a reason and no prediction/probabilities")
        else:
            row["status"] = "not_assessed"
            row.update({f"p_{name}": None for name in CLASSES})
        rows.append(row)
    if not rows:
        raise ValueError("Empty prediction table")
    return rows


def equal_probability_ensemble(tables):
    if len(tables) != 3:
        raise ValueError("Require all three predetermined seeds")
    indexed = [{_key(row): row for row in table} for table in tables]
    if any(len(mapping) != len(table) for mapping, table in zip(indexed, tables)):
        raise ValueError("Duplicate ensemble prediction key")
    if any(mapping.keys() != indexed[0].keys() for mapping in indexed[1:]):
        raise ValueError("Seed ensembles have different target sets")
    result = []
    for key, first in indexed[0].items():
        triple = [mapping[key] for mapping in indexed]
        if any(
            (row["label"], row["status"]) != (first["label"], first["status"]) for row in triple
        ):
            raise ValueError("Seed ensemble reference/coverage mismatch")
        row = dict(first)
        if row["status"] == "ok":
            probability = np.mean([[r[f"p_{name}"] for name in CLASSES] for r in triple], axis=0)
            row["prediction"] = int(probability.argmax())
            row.update({f"p_{name}": float(value) for name, value in zip(CLASSES, probability)})
        result.append(row)
    return result


def metric_snapshot(rows):
    metric = _metric_snapshot(evaluate_predictions(rows))
    metric["n_predictions_including_missing_reference"] = sum(r["status"] == "ok" for r in rows)
    metric["n_patients"] = len({(r["dataset"], r["study_id"]) for r in rows})
    return metric


def compare_equal_cohort(tables, *, resamples=2000):
    """Pair by dataset/patient/level, keeping all model coverage separately."""
    if len(tables) < 2:
        raise ValueError("Require at least two fixed model tables")
    index = {name: {_key(r): r for r in rows} for name, rows in tables.items()}
    names = list(index)
    first = index[names[0]]
    if any(len(index[name]) != len(tables[name]) for name in names):
        raise ValueError("Duplicate comparison target")
    if any(mapping.keys() != first.keys() for mapping in index.values()):
        raise ValueError("Compared models have different requested patient/level keys")
    if len({key[0] for key in first}) != 1:
        raise ValueError("Bootstrap comparison must keep datasets separate")
    for key, row in first.items():
        if any(mapping[key]["label"] != row["label"] for mapping in index.values()):
            raise ValueError("Compared models disagree on a reference grade")
    common = [
        key
        for key, row in first.items()
        if row["label"] is not None
        and all(mapping[key]["status"] == "ok" for mapping in index.values())
    ]
    if not common:
        raise ValueError("No commonly assessed labeled targets for comparison")
    paired = {name: [dict(index[name][key], series_id="") for key in common] for name in names}
    contrasts = (
        {
            f"localized_minus_{name}": (["localized"], [name])
            for name in names
            if name != "localized"
        }
        if "localized" in names
        else {}
    )
    bootstrap = paired_study_bootstrap(paired, contrasts=contrasts, resamples=resamples, seed=2026)
    return {
        "n_requested": len(first),
        "n_common_labeled_assessed": len(common),
        "n_excluded_from_common": len(first) - len(common),
        "models_full_cohort": {name: metric_snapshot(rows) for name, rows in tables.items()},
        "models_common_cohort": {name: metric_snapshot(rows) for name, rows in paired.items()},
        "bootstrap": bootstrap,
    }


def localization_snapshot(summary):
    def levels(value):
        groups = {}
        for dataset in DATASETS:
            for level in LEVELS:
                key = f"{dataset}/{level}"
                item = value["groups"][key]
                groups[key] = {
                    "n": _number(item["n"], integer=True),
                    "hit10": _number(item["hit10"], unit=True),
                    "hit5": _number(item["hit5"], unit=True),
                    "median_distance_mm_when_verifiable": _number(
                        item["median_distance_mm_when_verifiable"]
                    ),
                    "unverifiable_top_series": _number(
                        item["unverifiable_top_series"], integer=True
                    ),
                    "uniform_expected_hit10": _number(item["uniform_expected_hit10"], unit=True),
                }
        return {
            "groups": groups,
            "macro_hit10": _number(value["macro_hit10"], unit=True),
            "cross_entropy": _number(value["cross_entropy"]),
        }

    center = summary["validation_centers"]
    return {
        "level_epoch": _number(summary["level_epoch"], integer=True),
        "center_epoch": _number(summary["center_epoch"], integer=True),
        "fitting_studies": _number(summary["fitting_studies"], integer=True),
        "validation_studies": _number(summary["validation_studies"], integer=True),
        "localization_gate_passed": summary["localization_gate_passed"] is True,
        "positional_baseline": levels(summary["positional_baseline"]),
        "validation_levels": levels(summary["validation_levels"]),
        "validation_centers": {
            key: _number(center[key], integer=key == "n", unit=key == "within10mm")
            for key in ("n", "mean_error_mm", "median_error_mm", "within10mm", "p90_error_mm")
        },
        "center_scope": "AISSLab exact annotated validation slices; conditional on a known slice, not complete end-to-end canal localization",
    }


def localization_test_snapshot(summary):
    """Only typed end-to-end anatomy aggregates; no patient-level references."""
    if summary.get("grade_labels_used") is not False or summary.get("scored_splits") != ["test"]:
        raise ValueError("Need separate reference-only TEST anatomy scoring")
    result = {"groups": {}}
    counts = (
        "requested",
        "predicted",
        "reference_levels",
        "missing_reference",
        "missing_prediction_with_reference",
        "unverifiable_other_series_or_frame",
        "verifiable_plane_predictions",
        "hit5_count",
        "hit10_count",
    )
    for dataset in DATASETS:
        for level in LEVELS:
            key = f"{dataset}/test/{level}"
            source = summary["groups"][key]
            item = {field: _number(source[field], integer=True) for field in counts}
            for field in (
                "hit5",
                "hit10",
                "uniform_expected_hit10",
                "frozen_train_positional_baseline_hit5",
                "frozen_train_positional_baseline_hit10",
            ):
                item[field] = _number(source[field], unit=True)
            for tolerance in (5, 10):
                expected = (
                    item[f"hit{tolerance}_count"] / item["reference_levels"]
                    if item["reference_levels"]
                    else None
                )
                if expected is None or not math.isclose(
                    expected, item[f"hit{tolerance}"], abs_tol=1e-12
                ):
                    raise ValueError("TEST localization hit rate uses an inconsistent denominator")
            center = source["true_ccs_center"]
            item["true_ccs_center"] = {"n": _number(center["n"], integer=True)}
            for error in ("plane_error_mm", "inplane_error_mm", "distance3d_mm"):
                item["true_ccs_center"][error] = {
                    field: _number(center[error][field], integer=field == "n")
                    for field in ("n", "mean", "median", "p90")
                }
            for field in ("inplane_within10mm", "three_d_within10mm"):
                item["true_ccs_center"][field] = _number(center[field], unit=True)
            result["groups"][key] = item
    result["hit_denominator"] = (
        "All targets with an anatomy reference; absent predictions/unverifiable series count as misses"
    )
    result["center_scope"] = (
        "End-to-end predicted slice/center versus AISSLab CCS only; RSNA has no true canal XY here"
    )
    return result


def _verify_localizer(root):
    selected, summary = _json(root / "selection.json"), _json(root / "summary.json")
    checkpoint = (root / selected["checkpoint"]).resolve()
    if (
        selected.get("model_family") != "explicit_axial_localizer_v1"
        or selected.get("frozen") is not True
        or selected.get("localization_gate_passed") is not True
        or selected.get("summary_sha256") != sha256_file(root / "summary.json")
        or not checkpoint.is_relative_to(root.resolve())
        or selected.get("checkpoint_sha256") != sha256_file(checkpoint)
    ):
        raise ValueError("Localizer must be unchanged, frozen and pass its validation gate")
    return localization_snapshot(summary)


def _verified_localized(severity_root):
    _, manifest, _ = load_sealed_models(severity_root)
    training = _json(severity_root / "summary.json")
    evaluation = _json(severity_root / "evaluation/summary.json")
    if evaluation.get("status") != "complete" or evaluation.get("selection_sha256") != sha256_file(
        severity_root / "selection.json"
    ):
        raise ValueError("Need completed evaluation of the unchanged severity selection")
    tables, hashes = {}, {}
    for name in [f"seed_{s}.csv" for s in SEEDS] + ["ensemble.csv"]:
        path = severity_root / "evaluation" / name
        if evaluation["outputs"][name]["sha256"] != sha256_file(path):
            raise ValueError("Localized prediction table checksum changed")
        rows = read_predictions(path)
        _same_tree(
            _metric_snapshot(evaluation["outputs"][name]["metrics"]),
            _metric_snapshot(evaluate_predictions(rows)),
        )
        tables[name], hashes[name] = rows, sha256_file(path)
    expected = equal_probability_ensemble([tables[f"seed_{s}.csv"] for s in SEEDS])
    observed = {_key(row): row for row in tables["ensemble.csv"]}
    if {_key(row) for row in expected} != observed.keys():
        raise ValueError("Localized ensemble has different target keys from its three seeds")
    for row in expected:
        other = observed[_key(row)]
        if (row["label"], row["status"]) != (other["label"], other["status"]):
            raise ValueError("Localized ensemble differs in references or coverage from its seeds")
        if row["status"] == "ok":
            if any(
                not math.isclose(row[f"p_{c}"], other[f"p_{c}"], rel_tol=0, abs_tol=1e-7)
                for c in CLASSES
            ):
                raise ValueError("Localized ensemble is not the fixed equal probability mean")
    clean_runs = []
    for run in training["runs"]:
        history = json.loads(
            (severity_root / "runs" / f"seed-{run['seed']}" / "history.json").read_text()
        )
        if [h["epoch"] for h in history] != list(range(1, len(history) + 1)):
            raise ValueError("Severity training history has gaps")
        clean = [
            {
                "epoch": _number(h["epoch"], integer=True),
                "weighted_training_loss": _number(h["weighted_training_loss"]),
                "validation_macro_f1": _number(h["validation_macro_f1"], unit=True),
            }
            for h in history
        ]
        selected_epoch = int(np.argmax([h["validation_macro_f1"] for h in clean])) + 1
        if run["selected_epoch"] != selected_epoch:
            raise ValueError("Severity epoch differs from earliest validation maximum")
        clean_runs.append(
            {
                "seed": run["seed"],
                "selected_epoch": selected_epoch,
                "epochs_trained": len(clean),
                "history": clean,
            }
        )
    clean_training = {
        "runs": clean_runs,
        "train_targets": _number(manifest["train_targets"], integer=True),
        "validation_targets": _number(manifest["validation_targets"], integer=True),
        "class_counts": [_number(v, integer=True) for v in manifest["class_counts"]],
        "class_weights": [_number(v) for v in manifest["class_weights"]],
        "AISSLab_grades_used_for_training_or_selection": False,
    }
    return tables["ensemble.csv"], clean_training, hashes, tables


def _verified_transfer(root):
    summary = _json(root / "evaluation/summary.json")
    if (
        summary.get("model_family") != "axial_transfer_v1"
        or summary.get("training_performed") is not False
        or summary.get("selection_sha256") != sha256_file(root / "selection.json")
    ):
        raise ValueError("Old axial evaluation is incomplete")
    tables, hashes = {}, {}
    for phase in ("before", "after"):
        name = f"{phase}_ensemble.csv"
        path = root / "evaluation" / name
        entry = summary["entries"][name.removesuffix(".csv")]
        if entry.get("sha256", entry.get("predictions_sha256")) != sha256_file(path):
            raise ValueError("Old axial prediction checksum changed")
        rows = read_predictions(path)
        _same_tree(_metric_snapshot(entry["metrics"]), _metric_snapshot(evaluate_predictions(rows)))
        tables[phase], hashes[phase] = rows, sha256_file(path)
    return tables, hashes


def _verified_sagittal(root):
    manifest, summary = _json(root / "manifest.json"), _json(root / "summary.json")
    if (
        manifest.get("scope") != "heldout_test"
        or summary.get("scope") != "test"
        or manifest.get("summary_sha256") != sha256_file(root / "summary.json")
        or manifest.get("selection_sha256") != summary.get("selection_sha256")
    ):
        raise ValueError("Historical sagittal test manifest changed")
    tables, hashes = [], {}
    for seed in SEEDS:
        path = root / "entries/frozen_head" / f"seed-{seed}" / "predictions.csv"
        rows = read_predictions(path, default_dataset="rsna")
        entry = next(
            r for r in summary["entries"] if r["variant"] == "frozen_head" and r["seed"] == seed
        )
        _same_tree(_metric_snapshot(entry["metrics"]), _metric_snapshot(evaluate_predictions(rows)))
        tables.append(rows)
        hashes[str(seed)] = sha256_file(path)
    return equal_probability_ensemble(tables), hashes


def _fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def _table_lines(comparison):
    lines = ["| Модель | Macro-F1 | Тяжёлые: найдены / всего |", "| --- | ---: | ---: |"]
    for name, metric in comparison["models_common_cohort"].items():
        severe = metric["severe"]
        lines.append(
            f"| {NAMES[name]} | {_fmt(metric['conditional']['macro_f1'])} | "
            f"{severe['correctly_predicted_severe']} / {severe['n_total']} |"
        )
    return lines


def render_markdown(result):
    rsna = result["comparisons"]["rsna"]["axial_common"]
    delta = rsna["bootstrap"]["contrasts"]["localized_minus_before"]["metrics"]["macro_f1"]
    change = "вырос" if delta["point"] > 0 else "снизился" if delta["point"] < 0 else "не изменился"
    confident = delta["lower_95"] > 0 or delta["upper_95"] < 0
    assessment = (
        "Интервал не пересекает ноль."
        if confident
        else "Интервал пересекает ноль; уверенного изменения по этому сравнению нет."
    )
    loc = result["localization"]
    severe_before = result["models"]["rsna"]["before"]["severe"]
    severe_now = result["models"]["rsna"]["localized"]["severe"]
    l5_rsna = result["models"]["rsna"]["localized"]["by_level"]["L5-S1"]["severe"]
    l5_ais = result["models"]["aisslab"]["localized"]["by_level"]["L5-S1"]["severe"]
    lines = [
        "# Явная локализация аксиальных уровней и классификация по области канала",
        "",
        f"Проверил схему «найти уровень → найти канал → вырезать область → оценить степень». "
        f"На одинаковых обработанных целях RSNA macro-F1 {change} относительно прежней аксиальной модели "
        f"на {_fmt(abs(delta['point']))}; парный 95% интервал разницы "
        f"[{_fmt(delta['lower_95'])}; {_fmt(delta['upper_95'])}]. {assessment}",
        "",
        f"Тяжёлых случаев RSNA нашёл {severe_now['correctly_predicted_severe']} из {severe_now['n_total']}; "
        f"прежняя аксиальная модель находила {severe_before['correctly_predicted_severe']} из {severe_before['n_total']}. "
        f"Для L5–S1 получилось {l5_rsna['correctly_predicted_severe']}/{l5_rsna['n_total']} на RSNA "
        f"и {l5_ais['correctly_predicted_severe']}/{l5_ais['n_total']} на AISSLab. "
        "Попадание в нужный уровень само по себе ещё не означает правильную оценку степени стеноза.",
        "",
        "Это исследовательское продолжение на прежних разбиениях. Эти тестовые пациенты уже участвовали "
        "в разборе ошибок, поэтому результат не считаю новым независимым подтверждением. Прежние модели и их отчёты сохранил.",
        "",
        "Одновременно изменились локализация, физическая область, обучаемый классификатор и состав данных "
        "для оценки степени. Это сравнение целых схем, а не отдельная абляция выбора срезов. "
        "Разницу F1 нельзя приписать одному из этих изменений.",
        "",
        "## Что изменил",
        "",
        "Локализатор обучил по анатомическим ориентирам RSNA и XML AISSLab. Он выбирает аксиальный срез "
        "для L4–L5 и L5–S1. Отдельная небольшая сеть находит центр канала; её учил только по точным рамкам AISSLab, "
        "без использования AP-диаметра или степени стеноза. Аксиальные subarticular-точки RSNA служат "
        "приблизительными ориентирами уровня, а не разметкой центрального канала.",
        "",
        "Из предсказанных координат получаю физическую область 50×50 мм и до трёх соседних центров в ±10 мм, "
        "только в той же серии. Положение центра переношу через DICOM-геометрию. Признаки извлекает исходный "
        "замороженный ImageNet ResNet18; обучается MLP 514→64→3 с номером уровня. Для степени использовал "
        "**только RSNA TRAIN**, эпоху выбирал только по RSNA validation. Метки степени AISSLab в обучение не входят.",
        "",
        "```mermaid",
        "flowchart LR",
        '  A["Аксиальная T2-серия"] --> X["Замороженный ResNet18: полное поле"]',
        '  X --> B["Обученные выбор уровня и центр канала"]',
        '  B --> C["Область 50×50 мм и соседние срезы"]',
        '  C --> D["Замороженный ResNet18"]',
        '  D --> E["Три MLP: среднее вероятностей"]',
        '  E --> F["Норма/лёгкий · умеренный · тяжёлый"]',
        "```",
        "",
        "Где есть сагиттальная серия, подготовил проверку переноса нумерации SpineNet. Без подтверждённой общей "
        "системы координат перенос блокируется. В текущем RSNA надёжный общий FrameOfReferenceUID не подтверждён; "
        "в этом эксперименте работает аксиальный локализатор. Сагиттальные точки вслепую не проецируются.",
        "",
        "## Локализация на validation",
        "",
        "| Данные и уровень | Позиционный ориентир: ±10 мм | Обученный локализатор: ±10 мм | ±5 мм |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, value in loc["validation_levels"]["groups"].items():
        prior = loc["positional_baseline"]["groups"][key]["hit10"]
        lines.append(f"| {key} | {prior:.1%} | {value['hit10']:.1%} | {value['hit5']:.1%} |")
    center = loc["validation_centers"]
    lines += [
        "",
        f"Средний по четырём группам hit ±10 мм — **{loc['validation_levels']['macro_hit10']:.1%}**. "
        f"Медианная ошибка центра на AISSLab validation — **{center['median_error_mm']:.2f} мм** "
        f"({center['n']} размеченных срезов). Центр проверен на заранее известном XML-срезе: это ещё не "
        "полная точность цепочки на автоматически выбранном срезе и не проверка центра на RSNA.",
        "",
        "![Локализация на validation и TEST](images/localized_stenosis_localization.png)",
        "",
    ]
    if "localization_test" in result:
        test_loc = result["localization_test"]
        lines += [
            "## Локализация на TEST",
            "",
            "Здесь срез и центр уже выбирает модель; аннотации используются только после прогноза для оценки. "
            "В знаменателе — все уровни с анатомическим ориентиром. Непроверяемая серия считается непопаданием, "
            "а отсутствие анатомической координаты не означает отсутствия метки степени стеноза.",
            "",
            "| Данные и уровень | Позиционный ориентир: ±10 мм | Локализатор: найдено / ориентиров | ±10 мм | Непроверяемая серия |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for dataset in DATASETS:
            for level in LEVELS:
                value = test_loc["groups"][f"{dataset}/test/{level}"]
                lines.append(
                    f"| {dataset}/{level} | {value['frozen_train_positional_baseline_hit10']:.1%} | "
                    f"{value['hit10_count']} / {value['reference_levels']} | {value['hit10']:.1%} | "
                    f"{value['unverifiable_other_series_or_frame']} |"
                )
        ais_l5 = test_loc["groups"]["aisslab/test/L5-S1"]
        if ais_l5["frozen_train_positional_baseline_hit10"] > ais_l5["hit10"]:
            lines += [
                "",
                "На AISSLab L5–S1 простой позиционный ориентир оказался лучше обученного локализатора. "
                "Поэтому улучшение выбора уровня не обобщаю на все группы.",
            ]
        lines += [
            "",
            "На AISSLab отдельно проверил центр канала после автоматического выбора среза:",
            "",
            "| Уровень | Медианная ошибка в плоскости, мм | Медианная 3D-ошибка, мм | 3D-ошибка ≤10 мм |",
            "| --- | ---: | ---: | ---: |",
        ]
        for level in LEVELS:
            value = test_loc["groups"][f"aisslab/test/{level}"]["true_ccs_center"]
            lines.append(
                f"| {level} | {value['inplane_error_mm']['median']:.2f} | "
                f"{value['distance3d_mm']['median']:.2f} | {value['three_d_within10mm']:.1%} |"
            )
        lines += [
            "",
            "Это сравнение с точками и рамками AISSLab. В RSNA истинной аксиальной разметки центра "
            "центрального канала здесь нет, поэтому точность переноса этого центра требует отдельной проверки врачом.",
            "",
        ]
    lines += [
        "## Степень стеноза на RSNA TEST",
        "",
        f"Для сравнения трёх аксиальных вариантов оставил **{rsna['n_common_labeled_assessed']} общих "
        f"размеченных и обработанных уровней из {rsna['n_requested']}**. Отказы не превращал в норму; "
        "полное покрытие каждой модели приведено ниже.",
        "",
    ]
    lines += _table_lines(rsna)
    lines += [
        "",
        "Результат по уровням на этой же общей выборке:",
        "",
        "| Уровень и модель | Macro-F1 | Тяжёлые: найдены / всего | Тяжёлый → норма/лёгкий |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for level in LEVELS:
        for name, metric in rsna["models_common_cohort"].items():
            value = metric["by_level"][level]
            severe = value["severe"]
            lines.append(
                f"| {level}, {NAMES[name]} | {_fmt(value['conditional']['macro_f1'])} | "
                f"{severe['correctly_predicted_severe']} / {severe['n_total']} | {severe['predicted_normal_mild']} |"
            )
    training = result["severity_training"]
    lines += [
        "",
        f"Классификатор обучил на {training['train_targets']} размеченных областях RSNA TRAIN; "
        f"для выбора эпохи использовал {training['validation_targets']} областей RSNA validation. "
        "Число целей TRAIN по классам «норма/лёгкий, умеренный, тяжёлый»: "
        f"{training['class_counts']}. Нормализацию и веса классов вычислял только по TRAIN.",
        "",
        "| Seed | Выбранная эпоха | Macro-F1 на RSNA validation | Macro-F1 на RSNA TEST |",
        "| --- | ---: | ---: | ---: |",
    ]
    for run in training["runs"]:
        validation_score = run["history"][run["selected_epoch"] - 1]["validation_macro_f1"]
        test_score = result["localized_seed_metrics"][str(run["seed"])]["rsna"]["conditional"][
            "macro_f1"
        ]
        lines.append(
            f"| {run['seed']} | {run['selected_epoch']} | {_fmt(validation_score)} | {_fmt(test_score)} |"
        )
    if "sagittal_common" in result["comparisons"]["rsna"]:
        sagittal = result["comparisons"]["rsna"]["sagittal_common"]
        lines += [
            "",
            f"Дополнительно сравнил с заранее выбранным сагиттальным ансамблем трёх frozen-head "
            f"моделей v1 на **{sagittal['n_common_labeled_assessed']} общих целях**. Лучший seed по новому тесту не выбирал.",
            "",
        ]
        lines += _table_lines(sagittal)
    lines += [
        "",
        "![Сравнение моделей на одинаковых целях](images/localized_stenosis_comparison.png)",
        "",
        "## AISSLab: отдельная проверка переноса",
        "",
        "В прежнем аудите классы AISSLab точно совпали с диапазонами AP-диаметра; их соответствие "
        "шкале modified Lee в RSNA пока не проверено врачом. "
        "Ниже сохраняю прежнее условное соответствие трёх классов только для исследовательского сравнения; "
        "эти числа не доказывают клиническую точность на AISSLab.",
        "",
    ]
    lines += _table_lines(result["comparisons"]["aisslab"]["axial_common"])
    lines += [
        "",
        "## Покрытие и неопределённость",
        "",
        "| Данные и модель | Получено прогнозов / запрошено | Есть эталон | Обработано с эталоном | Не обработано с эталоном |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset, models in result["models"].items():
        for name, metric in models.items():
            lines.append(
                f"| {dataset}, {NAMES[name]} | {metric['n_predictions_including_missing_reference']} / "
                f"{metric['n_requested']} | {metric['n_labeled']} | {metric['n_assessed']} | {metric['n_not_assessed']} |"
            )
    lines += [
        "",
        "Прогноз без врачебной метки сохраняю, но в F1 и матрицы ошибок его не включаю. "
        f"Интервалы получены {rsna['bootstrap']['resamples']} парными bootstrap-перевыборками пациентов: оба уровня одного пациента "
        "всегда остаются вместе. Они описывают разброс на этих пациентах для фиксированных моделей, "
        "не неопределённость разметки и не качество в новой клинике. Локализатор обучен один раз с seed 42; "
        "три seed относятся только к классификатору степени. Разброс обучения локализатора здесь не оценён.",
        "",
        "![Матрицы ошибок на общих целях](images/localized_stenosis_confusion.png)",
        "",
        f"На L5–S1 тяжёлых случаев в RSNA TEST — "
        f"{result['models']['rsna']['localized']['by_level']['L5-S1']['severe']['n_total']}. Даже несколько верных ответов не позволяют "
        "считать проблему этого уровня решённой. Отдельно требуется врачебная проверка нумерации, "
        "видимости канала в автоматически полученных областях и соответствия шкал. Для нового независимого "
        "подтверждения нужен ещё не просмотренный набор пациентов.",
        "",
        "## Как повторить",
        "",
        "Все исходные снимки, координаты, индивидуальные CSV и веса сохраняю локально. В репозитории "
        "публикую код, [агрегированные результаты](localized-stenosis-results.json) и графики.",
        "",
        "Параметры зафиксировал в [протоколе эксперимента](../configs/axial-localizer-v6.json). "
        "Полная последовательность для подготовленных исходных данных:",
        "",
        "```bash",
        "python -m ai_spine_med.axial_localizer_workflow prepare --device mps",
        "python -m ai_spine_med.axial_localizer_workflow localizer --device mps",
        "python -m ai_spine_med.axial_localizer_workflow features --device mps",
        "python -m ai_spine_med.axial_localizer_workflow severity --device mps",
        "python -m ai_spine_med.axial_localizer_workflow prepare-test --device mps",
        "python -m ai_spine_med.axial_localizer_workflow test-features --device mps",
        "python -m ai_spine_med.axial_localizer_workflow evaluate --device mps",
        "python -m ai_spine_med.axial_localizer_workflow score",
        "python -m ai_spine_med.axial_localizer_workflow report",
        "```",
        "",
        "Отчёт можно пересобрать отдельно, без обучения: `python -m ai_spine_med.localized_stenosis_report`.",
        "",
        "Источник внешних анатомических данных: Kwon et al., 2026, "
        "[Medical Lumbar Spine 3D Axial MRI, v1](https://data.mendeley.com/datasets/wjxkrkxhb8/1), "
        "[DOI:10.17632/wjxkrkxhb8.1](https://doi.org/10.17632/wjxkrkxhb8.1), CC BY 4.0.",
        "",
    ]
    return "\n".join(lines)


def render_figures(result, images):
    images = Path(images)
    images.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    loc = result["localization"]
    keys = list(loc["validation_levels"]["groups"])
    x = np.arange(len(keys))
    stages = [
        (
            "Validation",
            {
                key: {
                    "uniform": loc["validation_levels"]["groups"][key]["uniform_expected_hit10"],
                    "prior": loc["positional_baseline"]["groups"][key]["hit10"],
                    "learned": loc["validation_levels"]["groups"][key]["hit10"],
                }
                for key in keys
            },
        )
    ]
    if "localization_test" in result:
        stages.append(
            (
                "TEST · ранее просмотренные пациенты",
                {
                    key: {
                        "uniform": result["localization_test"]["groups"][
                            key.replace("/", "/test/")
                        ]["uniform_expected_hit10"],
                        "prior": result["localization_test"]["groups"][key.replace("/", "/test/")][
                            "frozen_train_positional_baseline_hit10"
                        ],
                        "learned": result["localization_test"]["groups"][
                            key.replace("/", "/test/")
                        ]["hit10"],
                    }
                    for key in keys
                },
            )
        )
    figure, axes = plt.subplots(1, len(stages), figsize=(12, 5.3), squeeze=False)
    for axis, (stage, values) in zip(axes[0], stages):
        for offset, field, label, color in (
            (-0.26, "uniform", "Равномерный: ожидаемая доля", "#c8d1d9"),
            (0, "prior", "Позиционный ориентир", "#6c7d91"),
            (0.26, "learned", "Обученный локализатор", "#267f82"),
        ):
            scores = [values[key][field] for key in keys]
            bars = axis.bar(x + offset, scores, width=0.25, label=label, color=color)
            axis.bar_label(bars, [f"{value:.0%}" for value in scores], fontsize=8, padding=3)
        axis.set(
            xticks=x,
            xticklabels=[k.replace("/", "\n") for k in keys],
            ylim=(0, 1.12),
            ylabel="Доля попаданий в ±10 мм",
            title=stage,
        )
        axis.grid(axis="y", alpha=0.18)
        axis.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    figure.suptitle(
        "Выбор аксиального уровня: приблизительные ориентиры RSNA и точные CCS-срезы AISSLab"
    )
    figure.tight_layout(rect=(0, 0.09, 1, 0.94))
    figure.savefig(images / "localized_stenosis_localization.png", dpi=170)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for axis, dataset in zip(axes, DATASETS):
        comparison = result["comparisons"][dataset]["axial_common"]
        metrics = comparison["models_common_cohort"]
        names = list(metrics)
        values = [metrics[name]["conditional"]["macro_f1"] for name in names]
        intervals = comparison["bootstrap"]["models"]
        low = [
            max(0, value - intervals[name]["macro_f1"]["lower_95"])
            for name, value in zip(names, values)
        ]
        high = [
            max(0, intervals[name]["macro_f1"]["upper_95"] - value)
            for name, value in zip(names, values)
        ]
        bars = axis.bar(
            range(len(names)), values, color=[COLORS[n] for n in names], yerr=[low, high], capsize=4
        )
        axis.bar_label(bars, [f"{v:.3f}" for v in values], padding=5)
        axis.set(
            xticks=range(len(names)),
            xticklabels=[NAMES[n].replace(" ", "\n", 1) for n in names],
            ylim=(0, 1.05),
            ylabel="Macro-F1",
            title=f"{dataset.upper()}: {comparison['n_common_labeled_assessed']} общих целей",
        )
        axis.grid(axis="y", alpha=0.18)
        axis.set_axisbelow(True)
    figure.suptitle("Степень стеноза: фиксированные ансамбли и 95% patient-bootstrap интервалы")
    figure.tight_layout()
    figure.savefig(images / "localized_stenosis_comparison.png", dpi=170)
    plt.close(figure)
    figure, axes = plt.subplots(2, 3, figsize=(12, 7.4))
    for row_index, dataset in enumerate(DATASETS):
        for column, name in enumerate(("before", "after", "localized")):
            axis = axes[row_index, column]
            metric = result["comparisons"][dataset]["axial_common"]["models_common_cohort"][name]
            matrix = np.asarray(metric["conditional"]["confusion_matrix"])
            axis.imshow(matrix, cmap="Blues")
            for (i, j), value in np.ndenumerate(matrix):
                axis.text(
                    j,
                    i,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value > matrix.max() * 0.55 else "#223344",
                )
            axis.set(
                xticks=range(3),
                yticks=range(3),
                xticklabels=("Норма/л.", "Умер.", "Тяж."),
                yticklabels=("Норма/л.", "Умер.", "Тяж."),
                title=f"{dataset.upper()} · {NAMES[name]}",
                xlabel="Прогноз",
                ylabel="Эталон",
            )
    figure.suptitle("Матрицы ошибок — одинаковые размеченные и обработанные цели")
    figure.tight_layout()
    figure.savefig(images / "localized_stenosis_confusion.png", dpi=170)
    plt.close(figure)


def generate_report(
    root="reports/runs/explicit-localization-v6",
    *,
    transfer_root="reports/runs/axial-transfer-v4",
    sagittal_root="reports/runs/test-v1",
    output_doc="docs/localized-stenosis.md",
    output_json="docs/localized-stenosis-results.json",
    images="docs/images",
    resamples=2000,
):
    """Publish only after all three severity selections and evaluation are complete."""
    root, transfer_root = Path(root), Path(transfer_root)
    localization = _verify_localizer(root / "localizer")
    localized, training, localized_hashes, seed_tables = _verified_localized(root / "severity")
    old, old_hashes = _verified_transfer(transfer_root)
    models, comparisons = {}, {}
    for dataset in DATASETS:
        tables = {name: [r for r in rows if r["dataset"] == dataset] for name, rows in old.items()}
        tables["localized"] = [r for r in localized if r["dataset"] == dataset]
        comparisons[dataset] = {"axial_common": compare_equal_cohort(tables, resamples=resamples)}
        models[dataset] = {name: metric_snapshot(rows) for name, rows in tables.items()}
    hashes = {"localized": localized_hashes, "old_axial": old_hashes}
    if sagittal_root is not None:
        sagittal, sagittal_hashes = _verified_sagittal(Path(sagittal_root))
        comparisons["rsna"]["sagittal_common"] = compare_equal_cohort(
            {
                "before": [r for r in old["before"] if r["dataset"] == "rsna"],
                "localized": [r for r in localized if r["dataset"] == "rsna"],
                "sagittal": sagittal,
            },
            resamples=resamples,
        )
        models["rsna"]["sagittal"] = metric_snapshot(sagittal)
        hashes["sagittal"] = sagittal_hashes
    result = {
        "schema_version": 1,
        "experiment": "explicit_axial_localization_v6",
        "scope": "Exploratory follow-up on previously inspected TEST; no new independent confirmation",
        "clinical_scale_equivalence_verified": False,
        "AISSLab_grades_used_for_training_or_selection": False,
        "localization": localization,
        "severity_training": training,
        "models": models,
        "comparisons": comparisons,
        "input_prediction_sha256": hashes,
        "localized_seed_metrics": {
            str(seed): {
                dataset: metric_snapshot(
                    [r for r in seed_tables[f"seed_{seed}.csv"] if r["dataset"] == dataset]
                )
                for dataset in DATASETS
            }
            for seed in SEEDS
        },
    }
    localization_test = root / "localization-test/summary.json"
    if localization_test.is_file():
        result["localization_test"] = localization_test_snapshot(_json(localization_test))
    write_json(Path(output_json), result)
    render_figures(result, images)
    Path(output_doc).parent.mkdir(parents=True, exist_ok=True)
    Path(output_doc).write_text(render_markdown(result), encoding="utf-8")
    return result


def report_localized_stenosis(
    severity_dir="reports/runs/explicit-localization-v6/severity",
    *,
    out_document="docs/localized-stenosis.md",
    figures="docs/images",
    localization_summary=None,
):
    """Workflow-friendly alias using the sealed localizer next to severity outputs."""
    severity_dir = Path(severity_dir)
    if localization_summary is not None:
        actual = _json(severity_dir.parent / "localizer/summary.json")
        supplied = (
            _json(localization_summary)
            if isinstance(localization_summary, (str, Path))
            else localization_summary
        )
        if localization_snapshot(supplied) != localization_snapshot(actual):
            raise ValueError("Supplied localization summary differs from its sealed source")
    return generate_report(
        severity_dir.parent,
        output_doc=out_document,
        output_json=Path(out_document).with_name("localized-stenosis-results.json"),
        images=figures,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("reports/runs/explicit-localization-v6"))
    parser.add_argument(
        "--transfer-root", type=Path, default=Path("reports/runs/axial-transfer-v4")
    )
    parser.add_argument("--sagittal-root", type=Path, default=Path("reports/runs/test-v1"))
    parser.add_argument("--output-doc", type=Path, default=Path("docs/localized-stenosis.md"))
    parser.add_argument(
        "--output-json", type=Path, default=Path("docs/localized-stenosis-results.json")
    )
    parser.add_argument("--images", type=Path, default=Path("docs/images"))
    args = parser.parse_args(argv)
    result = generate_report(
        args.root,
        transfer_root=args.transfer_root,
        sagittal_root=args.sagittal_root,
        output_doc=args.output_doc,
        output_json=args.output_json,
        images=args.images,
    )
    print(
        json.dumps(
            {
                dataset: entry["axial_common"]["models_common_cohort"]["localized"]["conditional"][
                    "macro_f1"
                ]
                for dataset, entry in result["comparisons"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
