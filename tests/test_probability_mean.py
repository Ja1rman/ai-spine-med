import copy
import csv
import hashlib
import json

import numpy as np
import pytest

from ai_spine_med.probability_mean import (
    load_sources,
    mean_probability_rows,
    metrics_for_rows,
    read_predictions,
)

PROBABILITIES = ("p_normal_mild", "p_moderate", "p_severe")


def prediction_row(study, level, label, probability=None):
    row = {
        "study_id": str(study),
        "series_id": str(1000 + study),
        "level": level,
        "split": "validation",
        "label": label,
        "status": "ok" if probability is not None else "not_assessed",
        "reason": "" if probability is not None else "axial:processing_failed",
        "sagittal_available": True,
        "axial_available": probability is not None,
        "prediction": int(np.argmax(probability)) if probability is not None else None,
    }
    row.update(zip(PROBABILITIES, probability if probability is not None else (None,) * 3))
    return row


@pytest.fixture
def sources():
    sagittal = [
        prediction_row(100, "L4-L5", 2, (0.1, 0.2, 0.7)),
        prediction_row(100, "L5-S1", 0, (0.8, 0.1, 0.1)),
        prediction_row(101, "L4-L5", 1, (0.6, 0.3, 0.1)),
        prediction_row(101, "L5-S1", 2),
        prediction_row(102, "L4-L5", 0, (0.2, 0.7, 0.1)),
        prediction_row(102, "L5-S1", 1, (0.1, 0.8, 0.1)),
    ]
    axial = copy.deepcopy(sagittal)
    vectors = [
        (0.2, 0.4, 0.4),
        (0.6, 0.3, 0.1),
        (0.2, 0.7, 0.1),
        None,
        (0.5, 0.4, 0.1),
        (0.1, 0.7, 0.2),
    ]
    for row, probability in zip(axial, vectors):
        if probability is not None:
            row.update(zip(PROBABILITIES, probability))
            row["prediction"] = int(np.argmax(probability))
    fusion = copy.deepcopy(sagittal)
    for row in fusion:
        if row["status"] == "ok":
            row.update(zip(PROBABILITIES, (0.3, 0.4, 0.3)))
            row["prediction"] = 1
    return sagittal, axial, fusion


def target_key(row):
    return row["study_id"], row["series_id"], row["level"]


def test_csv_row_order_cannot_change_target_alignment_or_mutate_sources(sources):
    expected = mean_probability_rows(*sources)
    reordered = copy.deepcopy(sources)
    reordered[0].reverse()
    reordered[1][:] = reordered[1][2:] + reordered[1][:2]
    reordered[2][:] = reordered[2][1::2] + reordered[2][::2]
    original = copy.deepcopy(reordered)
    actual = mean_probability_rows(*reordered)
    assert actual == expected
    assert reordered == original
    assert [target_key(row) for row in actual] == sorted(target_key(row) for row in actual)
    assert [row["prediction"] for row in actual] == [2, 0, 1, None, 1, 1]


@pytest.mark.parametrize(
    "field,value",
    [
        ("series_id", "9999"),
        ("label", 1),
        ("status", "not_assessed"),
        ("reason", "unexpected"),
        ("axial_available", False),
    ],
)
def test_any_cross_source_identity_reference_or_coverage_mismatch_is_rejected(
    sources, field, value
):
    sources[1][0][field] = value
    with pytest.raises(ValueError):
        mean_probability_rows(*sources)


def test_duplicate_study_level_is_rejected_even_when_full_series_keys_differ(sources):
    for rows in sources:
        extra = copy.deepcopy(rows[0])
        extra["series_id"] = "9999"
        rows.append(extra)
    with pytest.raises(ValueError):
        mean_probability_rows(*sources)


@pytest.mark.parametrize("split", ["train", "test"])
def test_nonvalidation_rows_are_rejected_even_when_all_sources_agree(sources, split):
    for rows in sources:
        for row in rows:
            row["split"] = split
    with pytest.raises(ValueError):
        mean_probability_rows(*sources)


@pytest.mark.parametrize(
    "bad_vector",
    [(float("nan"), 0.2, 0.8), (float("inf"), 0, 0), (-0.1, 0.2, 0.9), (0.1, 0.2, 0.6)],
)
def test_invalid_probability_distribution_is_never_silently_normalized(sources, bad_vector):
    sources[1][0].update(zip(PROBABILITIES, bad_vector))
    with pytest.raises(ValueError):
        mean_probability_rows(*sources)


def test_source_prediction_must_match_its_probability_argmax(sources):
    sources[0][0]["prediction"] = 0
    with pytest.raises(ValueError):
        mean_probability_rows(*sources)


@pytest.mark.parametrize("field,value", [("prediction", 0), ("p_normal_mild", 1.0)])
def test_unassessed_source_cannot_carry_a_normal_prediction(sources, field, value):
    sources[1][3][field] = value
    with pytest.raises(ValueError):
        mean_probability_rows(*sources)


def test_unassessed_severe_remains_missing_in_all_requested_denominator(sources):
    rows = mean_probability_rows(*sources)
    missing = next(row for row in rows if row["status"] != "ok")
    assert missing["label"] == 2 and missing["prediction"] is None
    assert all(missing[field] is None for field in PROBABILITIES)
    metrics = metrics_for_rows(rows)
    assert metrics["n_requested"] == 6 and metrics["n_assessed"] == 5
    assert metrics["by_status"] == {"ok": 5, "not_assessed": 1}
    assert metrics["severe"]["n_total"] == 2
    assert metrics["severe"]["n_assessed"] == metrics["severe"]["correctly_predicted_severe"] == 1
    assert metrics["severe"]["recall_conditional"] == 1
    assert metrics["severe"]["detection_rate_all_requested"] == 0.5
    assert metrics["severe"]["missed_all_requested"] == 1
    assert metrics["by_level"]["L5-S1"]["severe"]["not_assessed"] == 1


def test_reference_labels_affect_metrics_but_never_blended_predictions(sources):
    original = mean_probability_rows(*sources)
    changed = copy.deepcopy(sources)
    for rows in changed:
        for row in rows:
            row["label"] = (row["label"] + 1) % 3
    perturbed = mean_probability_rows(*changed)
    for left, right in zip(original, perturbed):
        assert target_key(left) == target_key(right)
        assert left["prediction"] == right["prediction"]
        assert [left[field] for field in PROBABILITIES] == [right[field] for field in PROBABILITIES]
    assert (
        metrics_for_rows(original)["conditional"]["confusion_matrix"]
        != metrics_for_rows(perturbed)["conditional"]["confusion_matrix"]
    )


def test_fusion_reference_is_validated_but_does_not_participate_in_probability_mean(sources):
    original = mean_probability_rows(*sources)
    for row in sources[2]:
        if row["status"] == "ok":
            row.update(zip(PROBABILITIES, (1.0, 0.0, 0.0)))
            row["prediction"] = 0
    changed = mean_probability_rows(*sources)
    assert changed == original
    sources[2][0]["p_normal_mild"] = float("nan")
    with pytest.raises(ValueError):
        mean_probability_rows(*sources)


def test_csv_reader_canonicalizes_missing_fields_and_availability_without_filling_predictions(
    sources, tmp_path
):
    path = tmp_path / "predictions.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sources[0][0]))
        writer.writeheader()
        writer.writerows(sources[0])
    loaded = read_predictions(path)
    assert isinstance(loaded[0]["label"], int)
    assert isinstance(loaded[0]["p_severe"], float)
    assert loaded[0]["sagittal_available"] is True
    assert loaded[3]["axial_available"] is False
    assert loaded[3]["prediction"] is None
    assert all(loaded[3][field] is None for field in PROBABILITIES)
    mean_probability_rows(loaded, sources[1], sources[2])


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_experiment(tmp_path, sources):
    variants = ("sagittal", "axial", "fusion")
    seeds = [42, 2024, 3407]
    protocol = {
        "schema_version": 1,
        "experiment": "probability_mean_v3",
        "weights": {"sagittal": 0.5, "axial": 0.5},
        "seeds": seeds,
        "classes": ["normal_mild", "moderate", "severe"],
        "levels": ["L4-L5", "L5-S1"],
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 2026,
        "training_performed": False,
        "weight_or_threshold_tuning": False,
        "expected_validation": {
            "n_studies": len({r["study_id"] for r in sources[0]}),
            "n_targets": len(sources[0]),
            "n_assessed": sum(r["status"] == "ok" for r in sources[0]),
        },
        "source_prediction_sha256": {},
    }
    summary = {
        "test_evaluated": False,
        "all_seeds_reported": True,
        "all_variants_share_paired_cohort": True,
        "configuration": {
            "protocol": {key: protocol[key] for key in ("seeds", "classes", "levels")}
        },
        "variants": {},
    }
    for name, rows in zip(variants, sources):
        directory = tmp_path / "validation" / name / "ensemble"
        directory.mkdir(parents=True)
        path = directory / "predictions.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        digest = file_hash(path)
        protocol["source_prediction_sha256"][name] = digest
        summary["variants"][name] = {
            "validation": {
                "ensemble": {
                    "seeds": seeds,
                    "weights": [1 / 3] * 3,
                    "predictions_sha256": digest,
                    "metrics": metrics_for_rows(rows),
                }
            }
        }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary))
    protocol["source_summary_sha256"] = file_hash(summary_path)
    return protocol, summary


def test_load_sources_uses_only_pinned_prediction_files_without_source_mutation(
    sources, tmp_path, monkeypatch
):
    import torch

    protocol, _ = synthetic_experiment(tmp_path, sources)
    files = list(tmp_path.rglob("*.csv")) + [tmp_path / "summary.json"]
    before = {str(path): file_hash(path) for path in files}

    def forbidden(*args, **kwargs):
        raise AssertionError("Probability comparison must not load model weights or feature arrays")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(np, "load", forbidden)
    loaded, inputs = load_sources(tmp_path, protocol)
    assert set(loaded) == {"sagittal", "axial", "fusion"}
    assert len(inputs) == 4 and inputs["summary.json"] == protocol["source_summary_sha256"]
    assert all(len(rows) == 6 for rows in loaded.values())
    assert {str(path): file_hash(path) for path in files} == before


@pytest.mark.parametrize("target", ["summary.json", "validation/axial/ensemble/predictions.csv"])
def test_pinned_summary_or_prediction_tampering_is_rejected(sources, tmp_path, target):
    protocol, _ = synthetic_experiment(tmp_path, sources)
    path = tmp_path / target
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        load_sources(tmp_path, protocol)


@pytest.mark.parametrize("field", ["classes", "levels", "seeds"])
def test_source_category_level_or_seed_order_mismatch_is_rejected_even_with_valid_sha(
    sources, tmp_path, field
):
    protocol, summary = synthetic_experiment(tmp_path, sources)
    summary["configuration"]["protocol"][field] = list(
        reversed(summary["configuration"]["protocol"][field])
    )
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))
    protocol["source_summary_sha256"] = file_hash(path)
    with pytest.raises(ValueError):
        load_sources(tmp_path, protocol)


def test_missing_level_is_rejected_even_when_all_models_and_declared_counts_match(
    sources, tmp_path
):
    for rows in sources:
        rows.pop()
    protocol, _ = synthetic_experiment(tmp_path, sources)
    assert protocol["expected_validation"] == {"n_studies": 3, "n_targets": 5, "n_assessed": 4}
    with pytest.raises(ValueError):
        load_sources(tmp_path, protocol)


def test_changed_expected_cohort_cannot_silently_drop_a_requested_target(sources, tmp_path):
    protocol, _ = synthetic_experiment(tmp_path, sources)
    protocol["expected_validation"]["n_targets"] = 5
    with pytest.raises(ValueError):
        load_sources(tmp_path, protocol)
