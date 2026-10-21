"""Verified, aggregate-only reporting of the sealed axial transfer comparison."""

from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .assets import sha256_file, write_json
from .probability_mean_report import _metric_snapshot, _number, _same_tree
from .training import evaluate_predictions
from .uncertainty import paired_study_bootstrap

SEEDS = (42, 2024, 3407)
CLASSES = ("normal_mild", "moderate", "severe")
LEVELS = ("L4-L5", "L5-S1")
DATASETS = ("aisslab", "rsna")
PHASES = ("before", "after")
ENTRIES = tuple(f"{phase}_seed_{seed}" for phase in PHASES for seed in SEEDS) + (
    "before_ensemble",
    "after_ensemble",
)
_HASH = re.compile(r"[0-9a-f]{64}")
_REASONS = frozenset(
    {
        "missing_reference_label",
        "missing_or_invalid_geometry",
        "unsupported_inplane_rotation",
        "unsafe_source_path",
        "incomplete_series_download",
        "incompatible_series_receipt",
        "series_download_in_progress",
        "source_changed_during_extraction",
        "missing_dicom_identity",
        "unsupported_dicom_modality_or_multiframe",
        "unsupported_color_image",
        "unsupported_photometric_interpretation",
        "mixed_dicom_identity_or_frame",
        "inconsistent_photometric_interpretation",
        "duplicate_sop_instance_uid",
        "missing_or_duplicate_instance_number",
        "nonorthonormal_image_orientation",
        "not_axial_within_supported_obliquity",
        "invalid_pixel_spacing",
        "invalid_image_shape",
        "invalid_rescale_parameters",
        "invalid_pixel_array",
        "dicom_decode_or_header_failed",
        "empty_axial_series",
        "flat_series_intensity_distribution",
        "mixed_study_instance_uid",
        "duplicate_series_instance_key",
    }
)


def _json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Expected typed report metadata")
    return value


def _key(row):
    return row["dataset"], row["study_id"], row["level"]


def _read_predictions(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = {
            "dataset",
            "study_id",
            "level",
            "split",
            "label",
            "reference_status",
            "status",
            "reason",
            "prediction",
        } | {f"p_{name}" for name in CLASSES}
        if set(reader.fieldnames or ()) != fields:
            raise ValueError("Unexpected transfer prediction columns")
        source = list(reader)
    rows, seen = [], set()
    for raw in source:
        row = dict(raw)
        if (
            row["dataset"] not in DATASETS
            or not row["study_id"]
            or row["level"] not in LEVELS
            or row["split"] != "test"
            or _key(row) in seen
        ):
            raise ValueError("Invalid or duplicate TEST prediction identity")
        seen.add(_key(row))
        row["label"] = None if row["label"] == "" else int(row["label"])
        row["prediction"] = None if row["prediction"] == "" else int(row["prediction"])
        if row["label"] is not None and row["label"] not in (0, 1, 2):
            raise ValueError("Invalid reference severity")
        if row["reference_status"] != ("missing" if row["label"] is None else "available"):
            raise ValueError("Reference availability disagrees with the supplied label")
        values = [row[f"p_{name}"] for name in CLASSES]
        if row["status"] == "ok":
            probability = np.asarray(values, dtype=float)
            if (
                row["reason"]
                or not np.isfinite(probability).all()
                or np.any(probability < 0)
                or np.any(probability > 1)
                or not np.isclose(probability.sum(), 1, rtol=0, atol=1e-6)
                or row["prediction"] != int(probability.argmax())
            ):
                raise ValueError("Invalid transfer probability vector or prediction")
            row.update({f"p_{name}": float(value) for name, value in zip(CLASSES, probability)})
        elif (
            row["status"] != "not_assessed"
            or row["prediction"] is not None
            or any(value != "" for value in values)
            or row["reason"] not in _REASONS
        ):
            raise ValueError("A failure needs a safe reason and no class/probabilities")
        else:
            row.update({f"p_{name}": None for name in CLASSES})
        rows.append(row)
    if not rows:
        raise ValueError("Empty transfer prediction table")
    return rows


def _training(root):
    source, selected = _json(root / "summary.json"), _json(root / "selection.json")
    if (
        source.get("status") != "complete"
        or source.get("model_family") != "axial_transfer_v1"
        or source.get("seeds") != list(SEEDS)
        or selected.get("frozen") is not True
        or selected.get("model_family") != "axial_transfer_v1"
        or selected.get("seeds") != list(SEEDS)
        or selected.get("training_summary_sha256") != sha256_file(root / "summary.json")
        or selected.get("protocol_sha256") != source.get("protocol_sha256")
        or selected.get("training_index_sha256") != source.get("index_sha256")
    ):
        raise ValueError("Require completed unchanged three-seed training before publication")
    if [r.get("seed") for r in source.get("runs", [])] != list(SEEDS) or [
        r.get("seed") for r in selected.get("runs", [])
    ] != list(SEEDS):
        raise ValueError("Missing or reordered selected transfer seeds")
    public_runs = []
    for run, frozen in zip(source["runs"], selected["runs"]):
        seed = run["seed"]
        for key in (
            "seed",
            "best_epoch",
            "checkpoint",
            "checkpoint_sha256",
            "before_checkpoint",
            "before_checkpoint_sha256",
        ):
            if run[key] != frozen[key]:
                raise ValueError("Training summary differs from the fixed selection")
        for prefix in ("checkpoint", "before_checkpoint"):
            path = root / run[prefix]
            if (
                not path.resolve().is_relative_to(root.resolve())
                or not _HASH.fullmatch(run[prefix + "_sha256"])
                or sha256_file(path) != run[prefix + "_sha256"]
            ):
                raise ValueError("Selected transfer checkpoint checksum changed")
        history = json.loads((root / "runs" / f"seed-{seed}" / "history.json").read_text())
        if (
            not isinstance(history, list)
            or not history
            or [h["epoch"] for h in history] != list(range(len(history)))
            or run["epochs_run"] != len(history) - 1
        ):
            raise ValueError("Training history must include consecutive epochs starting at zero")
        clean = []
        best_epoch, best_score = 0, -math.inf
        for item in history:
            score = _number(item["validation_macro_f1"], unit=True)
            if score is None:
                raise ValueError("Validation macro-F1 cannot be missing")
            if score > best_score + 1e-12:
                best_epoch, best_score = item["epoch"], score
            loss = _number(item["training_loss"])
            validation_loss = _number(item["validation_loss_unweighted"])
            if (
                (item["epoch"] == 0 and loss is not None)
                or (item["epoch"] > 0 and (loss is None or loss < 0))
                or validation_loss is None
                or validation_loss < 0
            ):
                raise ValueError("Invalid learning-curve losses")
            clean.append(
                {
                    "epoch": int(item["epoch"]),
                    "training_loss": loss,
                    "validation_loss_unweighted": validation_loss,
                    "validation_macro_f1": score,
                }
            )
        if type(run["best_epoch"]) is not int or run["best_epoch"] != best_epoch:
            raise ValueError(
                "Selected epoch is not the earliest validation improvement, including epoch zero"
            )
        if not math.isclose(
            run["metrics"]["conditional"]["macro_f1"], best_score, abs_tol=1e-12, rel_tol=0
        ):
            raise ValueError("Selected validation score differs from history")
        public_runs.append(
            {
                "seed": seed,
                "best_epoch": best_epoch,
                "epochs_run": len(history) - 1,
                "history": clean,
            }
        )
    result = {
        "runs": public_runs,
        "epoch_zero_selected": sum(r["best_epoch"] == 0 for r in public_runs),
    }
    for field in ("n_fitting_studies", "n_replay_studies", "n_new_validation_studies"):
        result[field] = _number(source[field], integer=True)
    result["fitting_class_counts"] = [
        _number(v, integer=True) for v in source["fitting_class_counts"]
    ]
    result["class_weights"] = [_number(v) for v in source["class_weights"]]
    if (
        len(result["fitting_class_counts"]) != 3
        or len(result["class_weights"]) != 3
        or any(v is None or v <= 0 for v in result["class_weights"])
    ):
        raise ValueError("Invalid fitting-only class weights")
    return source, selected, result


def _cohort(prepared, sources):
    manifest = _json(prepared / "manifest.json")
    if (
        manifest.get("dataset") != "aisslab"
        or manifest.get("archive_verified") is not True
        or manifest.get("output_sha256", {}).get("records.json")
        != sha256_file(prepared / "records.json")
    ):
        raise ValueError("External preparation or patient records checksum changed")
    records = json.loads((prepared / "records.json").read_text())
    groups, keys = Counter(), set()
    counts = {
        split: {level: Counter() for level in LEVELS} for split in ("train", "validation", "test")
    }
    for row in records:
        if (
            row["dataset"] != "aisslab"
            or row["split"] not in counts
            or row["study_id"] in keys
            or len(row["labels"]) != 2
        ):
            raise ValueError("Invalid external patient grouping")
        keys.add(row["study_id"])
        groups[row["split"]] += 1
        for level, label in zip(LEVELS, row["labels"]):
            if label is not None and (type(label) is not int or label not in (0, 1, 2)):
                raise ValueError("Invalid external level grade")
            counts[row["split"]][level]["missing" if label is None else CLASSES[label]] += 1
    if dict(groups) != manifest["patients_by_split"] or len(records) != manifest["patients"]:
        raise ValueError("External prepared population differs from its manifest")
    source_summary = _json(sources / "summary.json")
    for name, key in (
        ("fitting-sources.json", "fitting_sources_sha256"),
        ("new-test-sources.json", "new_test_sources_sha256"),
    ):
        if sha256_file(sources / name) != source_summary[key]:
            raise ValueError("Source cohort changed after preparation")
    evaluation = _json(sources / "evaluation-sources.json")
    if (
        evaluation.get("schema_version") != 1
        or evaluation.get("scope") != "external_axial_v4_rsna_test"
    ):
        raise ValueError("Require the sealed TEST-only source inventory")
    expected, patients = {}, set()
    for row in evaluation["records"]:
        identity = row["dataset"], row["study_id"]
        if (
            row["dataset"] not in DATASETS
            or row["split"] != "test"
            or identity in patients
            or len(row["labels"]) != 2
        ):
            raise ValueError("Evaluation source patients overlap or contain a development split")
        patients.add(identity)
        for level, label in zip(LEVELS, row["labels"]):
            expected[(*identity, level)] = label
    new_expected = {
        (r["dataset"], r["study_id"], level): label
        for r in records
        if r["split"] == "test"
        for level, label in zip(LEVELS, r["labels"])
    }
    if {k: v for k, v in expected.items() if k[0] == "aisslab"} != new_expected or {
        k[0] for k in expected
    } != set(DATASETS):
        raise ValueError(
            "Evaluation must include every new held-out patient and the RSNA cohort separately"
        )
    public = {
        "new_patients_by_split": {split: int(groups[split]) for split in counts},
        "new_labels_by_split_level": {
            split: {
                level: {name: int(values[name]) for name in (*CLASSES, "missing")}
                for level, values in levels.items()
            }
            for split, levels in counts.items()
        },
        "replay_studies": _number(source_summary["replay_studies"], integer=True),
    }
    return expected, public


def _interval(item):
    result = {key: _number(item[key]) for key in ("point", "lower_95", "upper_95")}
    result["valid_resamples"] = _number(item["valid_resamples"], integer=True)
    if result["lower_95"] is not None and (
        result["upper_95"] is None or result["lower_95"] > result["upper_95"]
    ):
        raise ValueError("Invalid paired bootstrap interval")
    return result


def _dataset_report(tables, dataset):
    groups = {name: [r for r in rows if r["dataset"] == dataset] for name, rows in tables.items()}
    reference = groups["before_ensemble"]
    known = {
        name: [{**r, "series_id": ""} for r in rows if r["label"] is not None]
        for name, rows in groups.items()
    }
    if not known["before_ensemble"]:
        raise ValueError("Each evaluated dataset requires known reference targets")
    bootstrap = paired_study_bootstrap(
        known,
        contrasts={
            "after_minus_before_ensemble": (["after_ensemble"], ["before_ensemble"]),
            "after_minus_before_seed_mean": (
                [f"after_seed_{s}" for s in SEEDS],
                [f"before_seed_{s}" for s in SEEDS],
            ),
        },
        resamples=2000,
        seed=2026,
    )
    matrices = {name: _metric_snapshot(evaluate_predictions(rows)) for name, rows in groups.items()}
    clean_bootstrap = {
        "resamples": 2000,
        "seed": 2026,
        "n_studies_with_known_reference": bootstrap["n_studies"],
        "n_labeled_targets": bootstrap["n_targets"],
        "n_assessed": bootstrap["n_assessed"],
        "excluded_missing_reference_targets": sum(r["label"] is None for r in reference),
        "excluded_studies_without_reference": len({r["study_id"] for r in reference})
        - bootstrap["n_studies"],
        "models": {
            name: {key: _interval(value) for key, value in bootstrap["models"][name].items()}
            for name in ENTRIES
        },
        "contrasts": {
            name: {"metrics": {key: _interval(value) for key, value in item["metrics"].items()}}
            for name, item in bootstrap["contrasts"].items()
        },
    }
    before, after = (
        {_key(row): row for row in groups[name]} for name in ("before_ensemble", "after_ensemble")
    )
    transitions = Counter()
    for key, row in before.items():
        if row["status"] == "ok" and row["label"] is not None:
            old, new = row["prediction"] == row["label"], after[key]["prediction"] == row["label"]
            transitions[
                "both_correct"
                if old and new
                else "corrected"
                if new
                else "worsened"
                if old
                else "both_wrong"
            ] += 1
    return {
        "n_studies": len({r["study_id"] for r in reference}),
        "entries": {name: {"metrics": matrices[name]} for name in ENTRIES},
        "bootstrap": clean_bootstrap,
        "failures_by_reason": dict(
            sorted(Counter(r["reason"] for r in reference if r["status"] != "ok").items())
        ),
        "missing_reference": {
            "n_targets": sum(r["label"] is None for r in reference),
            "n_predicted": sum(r["label"] is None and r["status"] == "ok" for r in reference),
            "n_unprocessed": sum(r["label"] is None and r["status"] != "ok" for r in reference),
        },
        "transitions": {
            name: int(transitions[name])
            for name in ("corrected", "worsened", "both_correct", "both_wrong")
        },
    }


def _plots(public, figures):
    files = [f"axial_transfer_{name}.png" for name in ("comparison", "confusion", "learning")]
    with plt.rc_context({"font.size": 10, "figure.dpi": 130, "savefig.dpi": 160}):
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
        for axis, metric, title in zip(
            axes,
            ("macro", "severe"),
            ("Macro-F1 на обработанных уровнях", "Найдены тяжёлые · все известные цели"),
        ):
            for j, phase in enumerate(PHASES):
                values = []
                for dataset in DATASETS:
                    m = public["datasets"][dataset]["entries"][f"{phase}_ensemble"]["metrics"]
                    value = (
                        m["conditional"]["macro_f1"]
                        if metric == "macro"
                        else m["severe"]["detection_rate_all_requested"]
                    )
                    values.append(np.nan if value is None else value)
                bars = axis.bar(
                    np.arange(2) + (j - 0.5) * 0.34,
                    values,
                    width=0.32,
                    color=("#7a94ad", "#d28a36")[j],
                    label=("До", "После")[j],
                )
                axis.bar_label(bars, fmt="%.3f", padding=3)
            axis.set(
                xticks=range(2),
                xticklabels=["AISSLab · новый TEST", "RSNA · regression TEST"],
                ylim=(0, 1.12),
                title=title,
            )
            axis.spines[["top", "right"]].set_visible(False)
            axis.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figures / files[0])
        plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5))
        for i, dataset in enumerate(DATASETS):
            for j, phase in enumerate(PHASES):
                axis = axes[i, j]
                matrix = np.asarray(
                    public["datasets"][dataset]["entries"][f"{phase}_ensemble"]["metrics"][
                        "conditional"
                    ]["confusion_matrix"]
                )
                normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
                axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
                for a in range(3):
                    for b in range(3):
                        axis.text(
                            b,
                            a,
                            str(matrix[a, b]),
                            ha="center",
                            va="center",
                            color="white" if normalized[a, b] > 0.55 else "#17212b",
                        )
                axis.set(
                    xticks=range(3),
                    yticks=range(3),
                    xticklabels=["Норма /\nлёгкий", "Умеренный", "Тяжёлый"],
                    yticklabels=["Норма / лёгкий", "Умеренный", "Тяжёлый"],
                    xlabel="Предсказание",
                    ylabel="Исходная разметка",
                    title=f"{dataset.upper()} · {('до', 'после')[j]}",
                )
        fig.tight_layout()
        fig.savefig(figures / files[1])
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
        for run in public["training"]["runs"]:
            history = run["history"]
            axes[0].plot(
                [h["epoch"] for h in history[1:]],
                [h["training_loss"] for h in history[1:]],
                marker=".",
                label=str(run["seed"]),
            )
            axes[1].plot(
                [h["epoch"] for h in history],
                [h["validation_macro_f1"] for h in history],
                marker=".",
                label=str(run["seed"]),
            )
            axes[1].scatter(
                run["best_epoch"],
                history[run["best_epoch"]]["validation_macro_f1"],
                s=70,
                facecolors="none",
                edgecolors="black",
            )
        axes[0].set(title="TRAIN · взвешенная cross-entropy", xlabel="Эпоха", ylabel="Loss")
        axes[1].set(
            title="Новая validation · выбор эпохи",
            xlabel="Эпоха; кругом выбран checkpoint",
            ylabel="Macro-F1",
            ylim=(0, 1),
        )
        for axis in axes:
            axis.spines[["top", "right"]].set_visible(False)
            axis.legend(title="Seed", frameon=False)
        fig.tight_layout()
        fig.savefig(figures / files[2])
        plt.close(fig)
    return files


def _fmt(value):
    return "н/д" if value is None else f"{value:.3f}"


def _markdown(public, links):
    lines = [
        "# Дообучение аксиальной модели на AISSLab",
        "",
        "Я взял три ранее обученные аксиальные модели и дообучил последний блок ResNet18, механизм внимания и классификатор. Остальные слои, статистики BatchNorm и исходные нормализаторы оставил фиксированными. В обучение добавил заранее выбранные исследования из старого RSNA TRAIN, чтобы сохранить качество на исходных данных.",
        "",
        "Использовал [Medical Lumbar Spine 3D Axial MRI, Kwon et al., 2026](https://data.mendeley.com/datasets/wjxkrkxhb8/1), DOI: 10.17632/wjxkrkxhb8.1, лицензия CC BY 4.0. Из исходных DICOM и XML подготовил набор для этого эксперимента; авторскую разметку сопоставил с тремя классами по правилу ниже.",
        "",
        "Для эксперимента принял соответствие Normal / Stenosis / Severe классам 0 / 1 / 2. Клиническая эквивалентность шкал AISSLab и RSNA не установлена: здесь проверяю это исследовательское допущение. Отсутствующую аннотацию не считаю нормой; целевая степень — максимум явно размеченного центрального стеноза для уровня. Я сохраняю прогноз для неразмеченного уровня, но точность считаю только по врачебной разметке. Новый набор содержит аксиальные снимки. Прогнозы SpineNet не использую как эталон.",
        "",
        "Пациентов разделил до обучения. Все серии и оба уровня пациента остаются в одной части. Эпоху выбирал только по новой validation; исходная эпоха 0 тоже участвовала. Все три checkpoint зафиксировал до TEST. Предсказания сравниваю на одних и тех же целях и обработанных областях, без обрезки по эталонным координатам.",
        "",
        "| Часть AISSLab | Пациентов |",
        "| --- | ---: |",
    ]
    lines += [
        f"| {split} | {n} |" for split, n in public["cohort"]["new_patients_by_split"].items()
    ]
    lines += [
        "",
        "| Выборка | Уровней без врачебной метки | Из них с прогнозом | Без обработки |",
        "| --- | ---: | ---: | ---: |",
    ]
    for dataset in DATASETS:
        missing = public["datasets"][dataset]["missing_reference"]
        lines.append(
            f"| {dataset.upper()} | {missing['n_targets']} | {missing['n_predicted']} | {missing['n_unprocessed']} |"
        )
    lines += [
        "",
        f"В обучение вошли {public['training']['n_fitting_studies']} исследований, из них {public['training']['n_replay_studies']} — из исходного RSNA TRAIN. Эпоха 0 выбрана у {public['training']['epoch_zero_selected']} из трёх моделей; в этих случаях итоговый вариант сохранил исходные веса.",
        "",
        "## Результаты",
        "",
        "В основной таблице усредняю вероятности всех трёх seed с равными весами, затем выбираю класс. Это отличается от среднего трёх значений F1.",
        "",
        "| Выборка | Вариант | Macro-F1 | Найдены тяжёлые / всего | Обработано / с известной меткой | Без метки |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset in DATASETS:
        for phase in PHASES:
            metric = public["datasets"][dataset]["entries"][f"{phase}_ensemble"]["metrics"]
            severe = metric["severe"]
            lines.append(
                f"| {dataset.upper()} | {('До' if phase == 'before' else 'После')} | {_fmt(metric['conditional']['macro_f1'])} | {severe['correctly_predicted_severe']} / {severe['n_total']} | {metric['n_assessed']} / {metric['n_labeled']} | {metric['n_missing_reference_label']} |"
            )
    lines += [
        "",
        f"![Сравнение до и после]({links[0]})",
        "",
        "AISSLab TEST — новые отложенные пациенты при принятом сопоставлении шкал. Результаты прежнего RSNA TEST уже использовались в проекте; здесь это фиксированная проверка сохранения качества, а не новое независимое клиническое подтверждение.",
        "",
        "| Выборка | Seed | F1 до | F1 после | Выбранная эпоха |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset in DATASETS:
        entries = public["datasets"][dataset]["entries"]
        for run in public["training"]["runs"]:
            seed = run["seed"]
            lines.append(
                f"| {dataset.upper()} | {seed} | {_fmt(entries[f'before_seed_{seed}']['metrics']['conditional']['macro_f1'])} | {_fmt(entries[f'after_seed_{seed}']['metrics']['conditional']['macro_f1'])} | {run['best_epoch']} |"
            )
    lines += [
        "",
        "| Выборка | Δ F1 ансамбля: после − до | Парный 95% интервал |",
        "| --- | ---: | --- |",
    ]
    for dataset in DATASETS:
        ci = public["datasets"][dataset]["bootstrap"]["contrasts"]["after_minus_before_ensemble"][
            "metrics"
        ]["macro_f1"]
        lines.append(
            f"| {dataset.upper()} | {_fmt(ci['point'])} | [{_fmt(ci['lower_95'])}; {_fmt(ci['upper_95'])}] |"
        )
    lines += [
        "",
        "Bootstrap выполняю отдельно внутри каждой выборки: 2000 повторов по пациентам, оба уровня и отказы остаются вместе. Цели без эталонной метки исключаю из bootstrap и явно сохраняю в учёте покрытия. Интервалы описывают выборку для фиксированных моделей, не неопределённость шкал, разметки или повторного обучения.",
        "",
    ]
    regression = False
    for dataset in DATASETS:
        result = public["datasets"][dataset]
        ci = result["bootstrap"]["contrasts"]["after_minus_before_ensemble"]["metrics"]["macro_f1"]
        if ci["upper_95"] < 0:
            finding = "macro-F1 снизился; 95% интервал разницы целиком ниже нуля"
            regression = True
        elif ci["lower_95"] > 0:
            finding = "macro-F1 вырос; 95% интервал разницы целиком выше нуля"
        else:
            finding = "устойчивого изменения macro-F1 не выявил: 95% интервал разницы включает ноль"
        before = result["entries"]["before_ensemble"]["metrics"]["severe"]
        after = result["entries"]["after_ensemble"]["metrics"]["severe"]
        lines += [
            f"На {dataset.upper()} {finding}. Тяжёлых случаев нашёл {before['correctly_predicted_severe']} до и {after['correctly_predicted_severe']} после дообучения из {before['n_total']}.",
            "",
        ]
    if regression:
        lines += [
            "Из-за ухудшения на отложенной выборке прежние веса оставляю основными. Новые сохраняю отдельно как результат эксперимента. Причина ухудшения пока не установлена: соответствие шкал, выбор срезов и ошибки на конкретных уровнях нужно проверять отдельно.",
            "",
        ]
    lines += [
        f"![Матрицы ошибок]({links[1]})",
        "",
        "## Уровни и отказы",
        "",
        "| Выборка | Уровень | Вариант | Известно / обработано | Тяжёлые: найдены / всего | Тяжёлый → норма | Тяжёлый → умеренный | Тяжёлый без ответа |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in DATASETS:
        for level in LEVELS:
            for phase in PHASES:
                m = public["datasets"][dataset]["entries"][f"{phase}_ensemble"]["metrics"][
                    "by_level"
                ][level]
                s = m["severe"]
                lines.append(
                    f"| {dataset.upper()} | {level} | {('До' if phase == 'before' else 'После')} | {m['n_labeled']} / {m['n_assessed']} | {s['correctly_predicted_severe']} / {s['n_total']} | {s['predicted_normal_mild']} | {s['predicted_moderate']} | {s['not_assessed']} |"
                )
    lines += [
        "",
        "Macro-F1 считаю по обработанным целям с фиксированными классами 0, 1, 2. Для найденных тяжёлых случаев использую все известные тяжёлые цели, включая отказы. Необработанному случаю класс не присваиваю.",
        "",
        "| Выборка | Причина отсутствия оценки | Целей |",
        "| --- | --- | ---: |",
    ]
    for dataset in DATASETS:
        failures = public["datasets"][dataset]["failures_by_reason"]
        lines += [f"| {dataset.upper()} | {reason} | {n} |" for reason, n in failures.items()] or [
            f"| {dataset.upper()} | Нет | 0 |"
        ]
    lines += [
        "",
        f"![Ход обучения всех seed]({links[2]})",
        "",
        "Полученные числа относятся к этому исследовательскому протоколу и этим пациентам. Они не подтверждают клиническую пригодность или возможность сертификации сами по себе.",
        "",
    ]
    return "\n".join(lines)


def report_axial_transfer(
    run_dir, prepared_dir, sources_dir, *, document="docs/axial-transfer.md", figures="docs/images"
) -> dict:
    """Verify completed private outputs before writing whitelist-only public artifacts."""
    root, prepared, sources, document, figures = map(
        Path, (run_dir, prepared_dir, sources_dir, document, figures)
    )
    training, selection, public_training = _training(root)
    expected, cohort = _cohort(prepared, sources)
    evaluation = _json(root / "evaluation" / "summary.json")
    if (
        evaluation.get("schema_version") != 1
        or evaluation.get("model_family") != "axial_transfer_v1"
        or evaluation.get("training_performed") is not False
        or evaluation.get("selection_sha256") != sha256_file(root / "selection.json")
        or evaluation.get("training_index_sha256") != training["index_sha256"]
        or set(evaluation.get("entries", {})) != set(ENTRIES)
    ):
        raise ValueError(
            "Require completed TEST evaluation for exactly all eight before/after entries"
        )
    tables, reference = {}, None
    for name in ENTRIES:
        entry = evaluation["entries"][name]
        path = root / "evaluation" / f"{name}.csv"
        if sha256_file(path) != entry["predictions_sha256"]:
            raise ValueError("Transfer prediction CSV checksum changed")
        rows = _read_predictions(path)
        if {_key(r): r["label"] for r in rows} != expected:
            raise ValueError("Prediction cohort or labels differ from frozen TEST sources")
        identity = {
            _key(r): (r["label"], r["reference_status"], r["status"], r["reason"]) for r in rows
        }
        if reference is not None and identity != reference:
            raise ValueError("Before/after reference or assessed cohorts differ")
        reference = identity
        _same_tree(_metric_snapshot(entry["metrics"]), _metric_snapshot(evaluate_predictions(rows)))
        if set(entry["by_dataset"]) != set(DATASETS):
            raise ValueError("Dataset metrics must remain separate")
        for dataset in DATASETS:
            _same_tree(
                _metric_snapshot(entry["by_dataset"][dataset]),
                _metric_snapshot(
                    evaluate_predictions([r for r in rows if r["dataset"] == dataset])
                ),
            )
        tables[name] = rows
    for phase in PHASES:
        seeded = [{_key(r): r for r in tables[f"{phase}_seed_{seed}"]} for seed in SEEDS]
        for row in tables[f"{phase}_ensemble"]:
            if row["status"] == "ok":
                mean = np.mean(
                    [[entry[_key(row)][f"p_{name}"] for name in CLASSES] for entry in seeded],
                    axis=0,
                )
                if not np.allclose(mean, [row[f"p_{name}"] for name in CLASSES], atol=1e-7, rtol=0):
                    raise ValueError("Reported ensemble is not the equal mean of all three seeds")
    config = training["configuration"]
    if (
        config.get("label_mapping_authorized_by_user") is not True
        or config.get("clinical_scale_equivalence_verified") is not False
        or config.get("classes") != list(CLASSES)
        or config.get("levels") != list(LEVELS)
        or config.get("bootstrap_resamples") != 2000
        or config.get("bootstrap_seed") != 2026
    ):
        raise ValueError("Report requires the predefined research mapping and bootstrap protocol")
    public = {
        "schema_version": 1,
        "experiment": "external_axial_transfer_v4",
        "dataset_source": {
            "title": "Medical Lumbar Spine 3D Axial MRI",
            "authors": "Kwon et al.",
            "year": 2026,
            "doi": "10.17632/wjxkrkxhb8.1",
            "url": "https://data.mendeley.com/datasets/wjxkrkxhb8/1",
            "license": "CC BY 4.0",
        },
        "protocol": {
            "mapping_authorized_by_user": True,
            "clinical_scale_equivalence_verified": False,
            "mapping": {"normal": 0, "stenosis": 1, "severe_stenosis": 2},
            "classes": list(CLASSES),
            "levels": list(LEVELS),
            "seeds": list(SEEDS),
            "ensemble": "equal probability mean of all three seeds",
            "new_test": "held-out external patients under the predefined label mapping",
            "rsna_test": "previously opened cohort; fixed regression evaluation",
        },
        "cohort": cohort,
        "training": public_training,
        "datasets": {dataset: _dataset_report(tables, dataset) for dataset in DATASETS},
        "provenance": {
            "training_summary_sha256": sha256_file(root / "summary.json"),
            "selection_sha256": sha256_file(root / "selection.json"),
            "evaluation_summary_sha256": sha256_file(root / "evaluation" / "summary.json"),
            "prepared_manifest_sha256": sha256_file(prepared / "manifest.json"),
            "protocol_sha256": training["protocol_sha256"],
        },
    }
    # No output is written until all files, metrics, cohorts and ensembles pass.
    figures.mkdir(parents=True, exist_ok=True)
    document.parent.mkdir(parents=True, exist_ok=True)
    files = _plots(public, figures)
    links = [Path(os.path.relpath(figures / name, document.parent)).as_posix() for name in files]
    document.write_text(_markdown(public, links), encoding="utf-8")
    write_json(document.parent / "axial-transfer-results.json", public)
    return public
