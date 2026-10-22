"""Paired attention scope, source verification and useful error counts."""

import copy
import csv
import json

import matplotlib.image as mpimg
import numpy as np
import pytest

from ai_spine_med import roi_attention_report as report
from ai_spine_med.assets import sha256_file


def row(patient, level, label, prediction, *, failed=False):
    return {
        "dataset": "rsna",
        "study_id": f"PRIVATE_{patient}",
        "split": "validation",
        "level": level,
        "label": label,
        "status": "not_assessed" if failed else "ok",
        "reason": "PRIVATE /Users/PRIVATE/mri.dcm" if failed else "",
        "prediction": None if failed else prediction,
        **{
            f"p_{c}": None if failed else 0.8 if i == prediction else 0.1
            for i, c in enumerate(report.CLASSES)
        },
    }


@pytest.fixture
def comparison():
    tables = {name: [] for name in report.MODELS}
    for patient, label in enumerate((0, 0, 1, 1, 2, 2)):
        for level in report.LEVELS:
            for name in report.MODELS:
                failed = name.startswith("xml_only") and patient == 4 and level == "L5-S1"
                reference = None if patient == 0 and level == "L4-L5" else label
                prediction = 0 if name.endswith("mean") else label
                tables[name].append(row(patient, level, reference, prediction, failed=failed))
    counts = np.asarray([6, 4, 2])
    weights = 1 / np.sqrt(counts)
    weights /= weights.mean()
    summary = {
        "phase": "validation",
        "test_evaluated": False,
        "encoder_frozen": True,
        "localizers_frozen": True,
        "center_reused_exactly": True,
        "roi_reused_exactly": True,
        "batch_norm_frozen": True,
        "severity_training_source": "rsna_train_only",
        "seeds": list(report.SEEDS),
        "bootstrap_resamples": 100,
        "bootstrap_seed": report.BOOTSTRAP_SEED,
        "training_config": dict(report.OPTIMIZATION),
        "training": {
            "n_patients": 6,
            "n_targets": 12,
            "n_requested_targets": 14,
            "class_counts": counts.tolist(),
            "by_level_class_counts": {level: [3, 2, 1] for level in report.LEVELS},
            "class_weights": weights.tolist(),
            "private_note": "PRIVATE /Users/PRIVATE TRAIN_IDENTITIES SECRET",
        },
        "private_note": "PRIVATE https://example.invalid/?token=SECRET",
    }
    return summary, tables


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_run(tmp_path, summary, tables):
    summary = copy.deepcopy(summary)
    summary["input_files"] = {}
    summary["arms"] = {arm: {"variants": {}} for arm in report.ARMS}
    for arm in report.ARMS:
        baseline = report._extra_metrics(tables[f"{arm}_mean"])["conditional"]["macro_f1"]
        for variant in report.VARIANTS:
            name = f"{arm}_{variant}"
            base = tmp_path / arm / variant
            ensemble = base / "ensemble.csv"
            write_csv(ensemble, tables[name])
            source = {
                "ensemble": ensemble.relative_to(tmp_path).as_posix(),
                "seed_predictions": {},
                "histories": {},
                "selected_epochs": {},
            }
            for seed in report.SEEDS:
                path = base / f"seed_{seed}.csv"
                write_csv(path, list(reversed(tables[name])))
                history_path = base / f"history_{seed}.json"
                score = report._extra_metrics(tables[name])["conditional"]["macro_f1"]
                history = [
                    {
                        "epoch": 0,
                        "validation_macro_f1": baseline,
                        "weighted_train_loss": None,
                        "private_note": "PRIVATE SECRET /Users/PRIVATE",
                    }
                ]
                selected = 0
                if variant == "attention":
                    history.append(
                        {"epoch": 1, "validation_macro_f1": score, "weighted_train_loss": 0.5}
                    )
                    selected = 1
                history_path.write_text(json.dumps(history))
                if variant == "mean":
                    audit_path = history_path.parent / "audit.json"
                    audit_path.write_text(
                        json.dumps({"v9_reproduced": True, "maximum_probability_error": 0.0})
                    )
                    summary["input_files"][audit_path.relative_to(tmp_path).as_posix()] = (
                        sha256_file(audit_path)
                    )
                elif variant == "attention":
                    weights_path = history_path.parent / "attention_weights.npz"
                    n_assessed = sum(r["status"] == "ok" for r in tables[name])
                    np.savez(
                        weights_path,
                        weights=np.full((n_assessed, 3), 1 / 3, np.float32),
                        counts=np.full(n_assessed, 3, np.int64),
                    )
                    summary["input_files"][weights_path.relative_to(tmp_path).as_posix()] = (
                        sha256_file(weights_path)
                    )
                source["seed_predictions"][str(seed)] = path.relative_to(tmp_path).as_posix()
                source["histories"][str(seed)] = history_path.relative_to(tmp_path).as_posix()
                source["selected_epochs"][str(seed)] = selected
                for entry in (path, history_path):
                    summary["input_files"][entry.relative_to(tmp_path).as_posix()] = sha256_file(
                        entry
                    )
            summary["input_files"][ensemble.relative_to(tmp_path).as_posix()] = sha256_file(
                ensemble
            )
            summary["arms"][arm]["variants"][variant] = source
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))
    return path, summary


def report_args(tmp_path):
    return {
        "out_json": tmp_path / "public/results.json",
        "out_markdown": tmp_path / "public/report.md",
        "comparison_image": tmp_path / "public/comparison.png",
        "confusion_image": tmp_path / "public/l5.png",
        "learning_image": tmp_path / "public/learning.png",
        "resamples": 100,
    }


def test_same_patient_bootstrap_and_failure_denominators(comparison):
    summary, tables = comparison
    public = report.public_snapshot(summary, tables, resamples=100)
    group = public["groups"]["rsna/all"]
    assert (group["n_requested"], group["n_common_labeled_assessed"]) == (12, 10)
    assert (group["bootstrap"]["n_studies"], group["bootstrap"]["n_targets"]) == (6, 10)
    assert group["bootstrap"]["seed"] == report.BOOTSTRAP_SEED
    l5 = public["groups"]["rsna/L5-S1"]
    assert l5["models_full_cohort"]["xml_only_attention"]["severe"]["n_total"] == 2
    assert l5["models_full_cohort"]["xml_only_attention"]["severe"]["not_assessed"] == 1
    assert l5["models_common_cohort"]["mixed_attention"]["severe"]["n_total"] == 1
    assert l5["paired_changes"]["mixed_attention_minus_mean"]["severe_improved"] == 1
    for arm in report.ARMS:
        metric = group["models_common_cohort"]
        delta = (
            metric[f"{arm}_attention"]["conditional"]["macro_f1"]
            - metric[f"{arm}_mean"]["conditional"]["macro_f1"]
        )
        contrast = group["bootstrap"]["contrasts"][f"{arm}_attention_minus_mean"]
        assert contrast["metrics"]["macro_f1"]["point"] == pytest.approx(delta)


@pytest.mark.parametrize(
    "field,value",
    [
        ("phase", "test"),
        ("test_evaluated", True),
        ("encoder_frozen", False),
        ("localizers_frozen", False),
        ("center_reused_exactly", False),
        ("roi_reused_exactly", False),
        ("batch_norm_frozen", 1),
        ("severity_training_source", "rsna_and_aisslab"),
        ("seeds", [42]),
        ("bootstrap_seed", 1),
    ],
)
def test_rejects_changed_scope(comparison, field, value):
    summary, tables = comparison
    summary[field] = value
    with pytest.raises(ValueError, match="fixed paired validation-only"):
        report.public_snapshot(summary, tables, resamples=100)


def test_identical_counts_do_not_replace_matching_targets_and_grades(comparison):
    summary, tables = comparison
    changed = copy.deepcopy(tables)
    changed["mixed_attention"][0]["study_id"] = "PRIVATE_other"
    with pytest.raises(ValueError, match="patient/level"):
        report.public_snapshot(summary, changed, resamples=100)
    changed = copy.deepcopy(tables)
    changed["mixed_attention"][0]["label"] = 2
    with pytest.raises(ValueError, match="matching RSNA"):
        report.public_snapshot(summary, changed, resamples=100)
    changed = copy.deepcopy(tables)
    changed["mixed_attention"][0]["status"] = "not_assessed"
    with pytest.raises(ValueError, match="coverage mask"):
        report.public_snapshot(summary, changed, resamples=100)


def test_reports_false_severe_when_recall_increases(comparison):
    summary, tables = comparison
    for r in tables["mixed_attention"]:
        if r["label"] == 1:
            r["prediction"] = 2
    group = report.public_snapshot(summary, tables, resamples=100)["groups"]["rsna/L5-S1"]
    after = group["models_common_cohort"]["mixed_attention"]
    assert after["moderate_found"] == 0 and after["false_severe"] == 2
    assert after["severe"]["correctly_predicted_severe"] == 1


def test_history_selection_is_earliest_maximum_and_matches_predictions():
    history = [
        {"epoch": 0, "validation_macro_f1": 0.5, "weighted_train_loss": None},
        {"epoch": 1, "validation_macro_f1": 0.5, "weighted_train_loss": 0.2},
    ]
    assert report._history(history, 0, 0.5)["selected_epoch"] == 0
    with pytest.raises(ValueError, match="earliest maximum"):
        report._history(history, 1, 0.5)
    with pytest.raises(ValueError, match="disagrees"):
        report._history(history, 0, 0.6)
    history[1]["weighted_train_loss"] = float("nan")
    with pytest.raises(ValueError, match="Invalid validation history"):
        report._history(history, 0, 0.5)


def test_training_counts_and_class_weights_are_verified(comparison):
    summary, _ = comparison
    assert report.public_training(summary)["class_counts"] == [6, 4, 2]
    summary["training"]["class_weights"] = [1.0, 1.0, 1.0]
    with pytest.raises(ValueError, match="Class weights"):
        report.public_training(summary)
    summary["training"]["n_targets"] += 1
    with pytest.raises(ValueError, match="Inconsistent paired TRAIN"):
        report.public_training(summary)


def test_full_report_source_checks_and_safe_public_artifacts(comparison, tmp_path):
    summary, tables = comparison
    source, _ = write_run(tmp_path, summary, tables)
    args = report_args(tmp_path)
    public = report.report_roi_attention(source, **args)
    encoded = args["out_json"].read_text()
    assert "PRIVATE" not in encoded and "SECRET" not in encoded and "/Users/" not in encoded
    text = args["out_markdown"].read_text()
    assert "TEST в этом эксперименте не оценивал" in text
    assert "не исправляют смещение" in text and "Эпоха 0" in text
    assert "Ложный тяжёлый" in text and "Тяжёлый → норма" in text
    assert public["runs"]["mixed_attention"]["42"]["selected_epoch"] == 1
    assert public["mean_control_reproduced_v9"] is True
    assert "точно повторил v9" in text
    diagnostic = public["attention_diagnostics"]["mixed"]["42"]
    assert diagnostic["mean_effective_windows"] == pytest.approx(3)
    assert diagnostic["mean_l1_distance_from_uniform"] < 1e-6
    assert "Что получилось с весами attention" in text
    l5_section = text.split("## L5–S1 отдельно", 1)[1]
    assert l5_section.index("| Смешанная локализация | Attention |") < l5_section.index(
        "L5–S1, XML-only:"
    )
    for key in ("comparison_image", "confusion_image", "learning_image"):
        pixels = mpimg.imread(args[key])
        assert pixels.shape[0] > 500 and pixels.shape[1] > 1000


def test_actual_run_rejects_changed_or_escaped_prediction_sources(comparison, tmp_path):
    summary, tables = comparison
    source, saved = write_run(tmp_path, summary, tables)
    entry = saved["arms"]["mixed"]["variants"]["attention"]
    path = tmp_path / entry["seed_predictions"]["42"]
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum changed"):
        report.report_roi_attention(source, **report_args(tmp_path))
    with pytest.raises(ValueError, match="stay inside"):
        report._source(source, "../PRIVATE.csv", {})


def test_selected_scores_cannot_disagree_with_csv(comparison, tmp_path):
    summary, tables = comparison
    source, saved = write_run(tmp_path, summary, tables)
    entry = saved["arms"]["mixed"]["variants"]["attention"]
    path = tmp_path / entry["histories"]["42"]
    history = json.loads(path.read_text())
    history[1]["validation_macro_f1"] = 0.9
    path.write_text(json.dumps(history))
    saved["input_files"][entry["histories"]["42"]] = sha256_file(path)
    source.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="disagrees"):
        report.report_roi_attention(source, **report_args(tmp_path))


def test_missing_seed_is_not_an_allowed_ensemble(comparison, tmp_path):
    summary, tables = comparison
    source, saved = write_run(tmp_path, summary, tables)
    del saved["arms"]["mixed"]["variants"]["attention"]["seed_predictions"]["42"]
    source.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="all three fixed seeds"):
        report.report_roi_attention(source, **report_args(tmp_path))


def test_attention_statistics_use_known_window_counts_not_probability_zeros():
    values = np.array([[1.0, 0, 0], [0.5, 0.5, 0], [1.0, 0, 0]])
    statistic = report.attention_statistics(values, 3, np.array([1, 2, 3]))
    assert statistic["mean_effective_windows"] == pytest.approx(4 / 3)
    assert statistic["window_count_distribution"] == {"1": 1, "2": 1, "3": 1}
    assert statistic["mean_l1_distance_from_uniform"] == pytest.approx(4 / 9)
    without_counts = report.attention_statistics(values, 3)
    assert "window_count_distribution" not in without_counts
    assert "mean_l1_distance_from_uniform" not in without_counts


@pytest.mark.parametrize(
    "weights",
    [np.ones((2, 2)), np.ones((2, 3)), np.full((2, 3), np.nan), np.array([[1, 0, 0], [-1, 1, 1]])],
)
def test_attention_statistics_reject_invalid_probabilities(weights):
    with pytest.raises(ValueError, match="attention probabilities"):
        report.attention_statistics(weights, 2)


def test_attention_statistics_reject_nonzero_padding_and_invalid_counts():
    weights = np.full((2, 3), 1 / 3)
    with pytest.raises(ValueError, match="Padded"):
        report.attention_statistics(weights, 2, np.array([2, 3]))
    with pytest.raises(ValueError, match="counts"):
        report.attention_statistics(weights, 2, np.array([0, 3]))
