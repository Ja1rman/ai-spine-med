"""Synthetic reporting checks: paired targets, coverage, seals and publication."""

import copy
import csv
import json

import numpy as np
import pytest
import torch

from ai_spine_med import localized_stenosis as severity
from ai_spine_med import localized_stenosis_report as report
from ai_spine_med.assets import sha256_file, write_json
from ai_spine_med.training import evaluate_predictions


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def prediction(study, level, label, category, *, dataset="rsna", status="ok"):
    return {
        "dataset": dataset,
        "study_id": study,
        "split": "test",
        "level": level,
        "label": label,
        "status": status,
        "reason": "" if status == "ok" else "missing_predicted_landmark",
        "prediction": category if status == "ok" else None,
        **{
            f"p_{c}": (0.8 if j == category else 0.1) if status == "ok" else None
            for j, c in enumerate(report.CLASSES)
        },
    }


def write_predictions(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_features(root, split, labels, *, include_aisslab=False):
    rows, features = [], []
    for i, label in enumerate(labels):
        rows.append(
            {
                "dataset": "rsna",
                "study_id": f"PRIVATE_{split}_{i // 2}",
                "split": split,
                "level": report.LEVELS[i % 2],
                "label": label,
                "status": "ok",
                "reason": "",
                "feature_index": i,
            }
        )
        feature = np.zeros(512, np.float32)
        feature[label] = 1
        features.append(feature)
    if include_aisslab:
        for j, label in enumerate((2, None)):
            rows.append(
                {
                    "dataset": "aisslab",
                    "study_id": "PRIVATE_AIS",
                    "split": split,
                    "level": report.LEVELS[j],
                    "label": label,
                    "status": "ok",
                    "reason": "",
                    "feature_index": len(features),
                }
            )
            features.append(np.zeros(512, np.float32))
    root.mkdir(parents=True, exist_ok=True)
    np.savez(root / "features.npz", features=np.asarray(features, np.float32))
    with (root / "targets.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=severity.FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def localization_summary():
    def level_metrics(hit):
        return {
            "macro_hit10": hit,
            "cross_entropy": 1.2,
            "groups": {
                f"{dataset}/{level}": {
                    "n": 10,
                    "hit10": hit,
                    "hit5": hit - 0.1,
                    "median_distance_mm_when_verifiable": 3.0,
                    "unverifiable_top_series": 0,
                    "uniform_expected_hit10": 0.2,
                }
                for dataset in report.DATASETS
                for level in report.LEVELS
            },
        }

    return {
        "level_epoch": 2,
        "center_epoch": 3,
        "fitting_studies": 10,
        "validation_studies": 5,
        "localization_gate_passed": True,
        "positional_baseline": level_metrics(0.6),
        "validation_levels": level_metrics(0.9),
        "validation_centers": {
            "n": 10,
            "mean_error_mm": 5.0,
            "median_error_mm": 4.0,
            "within10mm": 0.9,
            "p90_error_mm": 9.0,
        },
        "original_failures": [
            {"study_id": "PRIVATE_FAILURE", "url": "https://example.invalid/?token=PRIVATE_TOKEN"}
        ],
    }


def localization_test_summary():
    groups = {}
    for dataset in report.DATASETS:
        for level in report.LEVELS:
            n = 10 if dataset == "aisslab" else 0
            error = {
                "n": n,
                "mean": 6.0 if n else None,
                "median": 5.0 if n else None,
                "p90": 10.0 if n else None,
            }
            groups[f"{dataset}/test/{level}"] = {
                "requested": 10,
                "predicted": 10,
                "reference_levels": 10,
                "missing_reference": 0,
                "missing_prediction_with_reference": 0,
                "unverifiable_other_series_or_frame": 1,
                "verifiable_plane_predictions": 9,
                "hit5_count": 6,
                "hit10_count": 8,
                "hit5": 0.6,
                "hit10": 0.8,
                "uniform_expected_hit10": 0.2,
                "frozen_train_positional_baseline_hit5": 0.7,
                "frozen_train_positional_baseline_hit10": 0.9,
                "true_ccs_center": {
                    "n": n,
                    "plane_error_mm": error,
                    "inplane_error_mm": error,
                    "distance3d_mm": error,
                    "inplane_within10mm": 0.9 if n else None,
                    "three_d_within10mm": 0.8 if n else None,
                },
            }
    return {
        "grade_labels_used": False,
        "scored_splits": ["test"],
        "groups": groups,
        "private_records": "PRIVATE_ANATOMY_REFERENCE",
    }


@pytest.fixture
def completed(tmp_path):
    root, old = tmp_path / "run", tmp_path / "old"
    fit = tmp_path / "fit"
    train_rows = write_features(fit, "train", [0, 1, 2, 0, 1, 2])
    train_features = np.load(fit / "features.npz")["features"].copy()
    val_rows = write_features(tmp_path / "val", "validation", [0, 1, 2, 0, 1, 2])
    val_features = np.load(tmp_path / "val/features.npz")["features"].copy()
    for row in val_rows:
        row["feature_index"] += len(train_rows)
    np.savez(fit / "features.npz", features=np.concatenate((train_features, val_features)))
    with (fit / "targets.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=severity.FIELDS)
        writer.writeheader()
        writer.writerows(train_rows + val_rows)
    severity.train_localized_stenosis(
        fit / "features.npz",
        fit / "targets.csv",
        root / "severity",
        {"severity_max_epochs": 1, "severity_patience": 1},
    )
    test = tmp_path / "test"
    write_features(test, "test", [0, 1, 2, 2], include_aisslab=True)
    severity.evaluate_localized_stenosis(
        test / "features.npz", test / "targets.csv", root / "severity"
    )
    localizer = root / "localizer"
    localizer.mkdir()
    (localizer / "model.pt").write_bytes(b"SYNTHETIC_FROZEN_LOCALIZER")
    write_json(localizer / "summary.json", localization_summary())
    write_json(
        localizer / "selection.json",
        {
            "model_family": "explicit_axial_localizer_v1",
            "frozen": True,
            "localization_gate_passed": True,
            "checkpoint": "model.pt",
            "checkpoint_sha256": sha256_file(localizer / "model.pt"),
            "summary_sha256": sha256_file(localizer / "summary.json"),
        },
    )
    write_json(root / "localization-test/summary.json", localization_test_summary())
    source = report.read_predictions(root / "severity/evaluation/ensemble.csv")
    (old / "evaluation").mkdir(parents=True)
    (old / "selection.json").write_text("SYNTHETIC_OLD_SELECTION")
    old_summary = {
        "model_family": "axial_transfer_v1",
        "training_performed": False,
        "selection_sha256": sha256_file(old / "selection.json"),
        "entries": {},
    }
    for phase in ("before", "after"):
        rows = [
            prediction(
                r["study_id"],
                r["level"],
                r["label"],
                r["label"] if phase == "before" and r["label"] is not None else 0,
                dataset=r["dataset"],
            )
            for r in source
        ]
        path = old / "evaluation" / f"{phase}_ensemble.csv"
        write_predictions(path, rows)
        old_summary["entries"][f"{phase}_ensemble"] = {
            "predictions_sha256": sha256_file(path),
            "metrics": evaluate_predictions(rows),
        }
    old_summary["private_extra"] = "PRIVATE_OLD_SECRET"
    write_json(old / "evaluation/summary.json", old_summary)
    return root, old


def test_equal_cohort_is_order_independent_and_groups_both_patient_levels():
    rows = [
        prediction("A", "L4-L5", 0, 0),
        prediction("A", "L5-S1", 2, 2),
        prediction("B", "L4-L5", 1, 0),
        prediction("B", "L5-S1", 2, 0),
    ]
    comparison = report.compare_equal_cohort(
        {"before": rows, "localized": list(reversed(rows))}, resamples=100
    )
    assert comparison["n_common_labeled_assessed"] == 4
    assert comparison["bootstrap"]["n_studies"] == 2
    assert (
        comparison["bootstrap"]["contrasts"]["localized_minus_before"]["metrics"]["macro_f1"][
            "point"
        ]
        == 0
    )


def test_common_comparison_does_not_hide_full_coverage_or_missing_references():
    first = [
        prediction("A", "L4-L5", 0, 0),
        prediction("A", "L5-S1", 2, 2),
        prediction("B", "L4-L5", 1, 1),
        prediction("B", "L5-S1", None, 0),
    ]
    second = copy.deepcopy(first)
    second[1] = prediction("A", "L5-S1", 2, 0, status="not_assessed")
    result = report.compare_equal_cohort({"before": first, "localized": second}, resamples=100)
    assert result["n_common_labeled_assessed"] == 2
    assert result["models_full_cohort"]["localized"]["n_not_assessed"] == 1
    assert result["models_full_cohort"]["localized"]["severe"]["missed_all_requested"] == 1
    assert (
        result["models_full_cohort"]["localized"]["n_predictions_including_missing_reference"] == 3
    )


def test_reference_and_requested_key_mismatches_cannot_be_paired():
    rows = [prediction("A", "L4-L5", 0, 0), prediction("B", "L5-S1", 2, 2)]
    changed = copy.deepcopy(rows)
    changed[0]["label"] = 1
    with pytest.raises(ValueError, match="reference grade"):
        report.compare_equal_cohort({"before": rows, "localized": changed}, resamples=100)
    with pytest.raises(ValueError, match="requested patient/level"):
        report.compare_equal_cohort({"before": rows, "localized": rows[:1]}, resamples=100)


def test_read_predictions_rejects_false_argmax_and_keeps_unknown_reference(tmp_path):
    rows = [prediction("A", "L4-L5", None, 2)]
    path = tmp_path / "rows.csv"
    write_predictions(path, rows)
    read = report.read_predictions(path)
    assert read[0]["label"] is None and read[0]["prediction"] == 2
    rows[0]["prediction"] = 0
    write_predictions(path, rows)
    with pytest.raises(ValueError, match="probability vector"):
        report.read_predictions(path)


def test_ensemble_uses_probabilities_and_joins_by_patient_level():
    rows = [prediction("A", "L4-L5", 2, 0), prediction("B", "L5-S1", 0, 2)]
    second = [prediction("B", "L5-S1", 0, 2), prediction("A", "L4-L5", 2, 2)]
    ensemble = report.equal_probability_ensemble([rows, second, second])
    assert ensemble[0]["prediction"] == 2
    assert ensemble[0]["p_severe"] == pytest.approx((0.1 + 0.8 + 0.8) / 3)
    assert ensemble[1]["study_id"] == "B"


def test_whitelisted_localization_drops_all_private_fields():
    result = report.localization_snapshot(localization_summary())
    assert "PRIVATE" not in json.dumps(result)
    assert "annotated validation slices" in result["center_scope"]


def test_test_anatomy_snapshot_counts_unverifiable_targets_in_hit_denominator():
    source = localization_test_summary()
    clean = report.localization_test_snapshot(source)
    assert "PRIVATE" not in json.dumps(clean)
    assert clean["groups"]["rsna/test/L5-S1"]["hit10"] == 8 / 10
    source["groups"]["rsna/test/L5-S1"]["hit10"] = 8 / 9
    with pytest.raises(ValueError, match="inconsistent denominator"):
        report.localization_test_snapshot(source)


def test_public_generation_retains_no_patient_identifier_capability_or_source_path(
    completed, tmp_path
):
    root, old = completed
    doc, data, images = (
        tmp_path / "docs/report.md",
        tmp_path / "docs/result.json",
        tmp_path / "docs/images",
    )
    result = report.generate_report(
        root,
        transfer_root=old,
        sagittal_root=None,
        output_doc=doc,
        output_json=data,
        images=images,
        resamples=100,
    )
    text = data.read_text() + doc.read_text()
    assert "PRIVATE" not in text and "example.invalid" not in text and str(tmp_path) not in text
    assert result["models"]["aisslab"]["localized"]["n_missing_reference_label"] == 1
    assert "не отдельная абляция" in text or "не отдельная абляция" in text.replace("а ", "")
    assert "Локализатор обучен один раз" in text
    assert len(list(images.glob("*.png"))) == 3


def test_unchanged_severity_evaluation_required_before_prediction_reads(completed, monkeypatch):
    root, _ = completed
    path = root / "severity/evaluation/summary.json"
    value = json.loads(path.read_text())
    value["selection_sha256"] = "0" * 64
    write_json(path, value)
    monkeypatch.setattr(
        report,
        "read_predictions",
        lambda *a, **k: pytest.fail("TEST CSV opened before seal validation"),
    )
    with pytest.raises(ValueError, match="unchanged severity"):
        report._verified_localized(root / "severity")


def test_prediction_hash_change_blocks_publication(completed):
    root, _ = completed
    path = root / "severity/evaluation/ensemble.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum changed"):
        report._verified_localized(root / "severity")


def test_localization_gate_and_checkpoint_are_checked(completed):
    root, _ = completed
    path = root / "localizer/model.pt"
    path.write_bytes(b"CHANGED")
    with pytest.raises(ValueError, match="Localizer must"):
        report._verify_localizer(root / "localizer")


def test_conclusion_is_computed_and_can_report_a_negative_result(completed, tmp_path):
    root, old = completed
    result = report.generate_report(
        root,
        transfer_root=old,
        sagittal_root=None,
        output_doc=tmp_path / "doc.md",
        output_json=tmp_path / "public.json",
        images=tmp_path / "images",
        resamples=100,
    )
    delta = result["comparisons"]["rsna"]["axial_common"]["bootstrap"]["contrasts"][
        "localized_minus_before"
    ]["metrics"]["macro_f1"]
    delta.update(point=-0.1, lower_95=-0.2, upper_95=-0.01)
    text = report.render_markdown(result)
    assert "снизился" in text and "Интервал не пересекает ноль" in text
    delta.update(point=0.01, lower_95=-0.1, upper_95=0.1)
    assert "уверенного изменения" in report.render_markdown(result)


def test_historical_sagittal_ensemble_gets_its_own_strict_common_cohort(completed, tmp_path):
    root, old = completed
    sagittal = tmp_path / "sagittal"
    source = [
        r
        for r in report.read_predictions(root / "severity/evaluation/ensemble.csv")
        if r["dataset"] == "rsna"
    ]
    rows = [prediction(r["study_id"], r["level"], r["label"], r["label"]) for r in source]
    rows[0] = prediction(
        rows[0]["study_id"], rows[0]["level"], rows[0]["label"], 0, status="not_assessed"
    )
    for row in rows:
        row.pop("dataset")
        row["series_id"] = "PRIVATE_SERIES"
    summary = {"scope": "test", "selection_sha256": "a" * 64, "entries": []}
    for seed in report.SEEDS:
        write_predictions(
            sagittal / "entries/frozen_head" / f"seed-{seed}" / "predictions.csv", rows
        )
        summary["entries"].append(
            {"variant": "frozen_head", "seed": seed, "metrics": evaluate_predictions(rows)}
        )
    write_json(sagittal / "summary.json", summary)
    write_json(
        sagittal / "manifest.json",
        {
            "scope": "heldout_test",
            "selection_sha256": "a" * 64,
            "summary_sha256": sha256_file(sagittal / "summary.json"),
        },
    )
    result = report.generate_report(
        root,
        transfer_root=old,
        sagittal_root=sagittal,
        output_doc=tmp_path / "doc.md",
        output_json=tmp_path / "public.json",
        images=tmp_path / "images",
        resamples=100,
    )
    assert result["comparisons"]["rsna"]["axial_common"]["n_common_labeled_assessed"] == 4
    assert result["comparisons"]["rsna"]["sagittal_common"]["n_common_labeled_assessed"] == 3
    assert result["models"]["rsna"]["sagittal"]["n_not_assessed"] == 1
