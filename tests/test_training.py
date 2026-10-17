"""Protocol checks use small synthetic feature vectors, never clinical images."""

import csv
import hashlib
import json

import numpy as np
import pytest

from ai_spine_med.training import (
    collapse_stenosis_probabilities,
    evaluate_baseline,
    evaluate_predictions,
    inverse_sqrt_class_weights,
    load_feature_data,
    predict_head,
    train_head,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def update_hashes(root, **updates):
    metadata = json.loads((root / "metadata.json").read_text())
    metadata.update(updates)
    metadata["outputs_sha256"] = {
        name: digest(root / name) for name in ("features.npz", "target_manifest.csv")
    }
    (root / "metadata.json").write_text(json.dumps(metadata))


def make_cache(root):
    root.mkdir()
    arrays = {
        "features": np.zeros((15, 512), dtype=np.float32),
        "original_logits": np.zeros((15, 4), dtype=np.float32),
        "labels": np.arange(15, dtype=np.int64) % 3,
        "study_id": np.asarray([str(1000 + index) for index in range(15)]),
        "series_id": np.asarray([str(2000 + index) for index in range(15)]),
        "level": np.asarray(["L4-L5"] * 15),
        "split": np.asarray(["train"] * 9 + ["validation"] * 6),
    }
    rows = []
    for index, grade in enumerate(arrays["labels"]):
        arrays["features"][index, grade] = 3
        if grade == 0:
            # Native argmax is moderate; correct probability collapse is normal/mild.
            arrays["original_logits"][index] = np.log([0.30, 0.30, 0.35, 0.05])
        else:
            arrays["original_logits"][index] = -5
            arrays["original_logits"][index, grade + 1] = 5
        rows.append(
            {
                **{
                    name: arrays[name][index]
                    for name in ("study_id", "series_id", "level", "split")
                },
                "label": int(grade),
                "status": "ok",
                "reason": "",
            }
        )
    rows.append(
        {
            "study_id": "9999",
            "series_id": "9998",
            "level": "L5-S1",
            "split": "validation",
            "label": 2,
            "status": "missing_roi",
            "reason": "target_not_detected",
        }
    )
    # Deliberately reverse array order relative to the target manifest.
    np.savez(root / "features.npz", **{name: value[::-1] for name, value in arrays.items()})
    with (root / "target_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "extraction_complete": True,
                "encoder_sha256": "synthetic-original-encoder",
                "preprocessing_version": "synthetic-test-only",
            }
        )
    )
    update_hashes(root)
    return root


def rewrite_array(root, name, transform):
    with np.load(root / "features.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays[name] = transform(arrays[name])
    np.savez(root / "features.npz", **arrays)
    update_hashes(root)


def test_probability_mass_is_merged_before_argmax():
    logits = np.log([[0.30, 0.30, 0.35, 0.05]])
    assert logits.argmax(axis=1).tolist() == [2]
    merged = collapse_stenosis_probabilities(logits)
    np.testing.assert_allclose(merged, [[0.60, 0.35, 0.05]])
    assert merged.argmax(axis=1).tolist() == [0]
    np.testing.assert_allclose(collapse_stenosis_probabilities(logits + 10000), merged)


def test_baseline_joins_identifiers_and_retains_failed_severe_case(tmp_path):
    root = make_cache(tmp_path / "cache")
    metrics = evaluate_baseline(root, tmp_path / "evaluation")
    assert metrics["n_labeled"] == 7
    assert metrics["n_assessed"] == 6
    assert metrics["n_not_assessed"] == 1
    assert metrics["coverage"] == 6 / 7
    assert metrics["conditional"]["macro_f1"] == 1
    assert metrics["conditional"]["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    assert metrics["severe"]["recall_conditional"] == 1
    assert metrics["severe"]["detection_rate_all_requested"] == 2 / 3
    assert metrics["severe"]["not_assessed"] == 1
    with (tmp_path / "evaluation" / "predictions.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["prediction"] == "0"
    assert rows[0]["native_ccs_4grade"] == "3"
    assert rows[-1]["prediction"] == ""


def test_missing_references_are_never_normal_and_fixed_macro_includes_all_classes():
    rows = [
        {"label": 0, "prediction": 0, "status": "ok", "level": "L4-L5"},
        {"label": None, "prediction": None, "status": "missing_label", "level": "L5-S1"},
    ]
    metrics = evaluate_predictions(rows)
    assert metrics["n_requested"] == 2
    assert metrics["n_labeled"] == 1
    assert metrics["n_missing_reference_label"] == 1
    assert metrics["conditional"]["macro_f1"] == 1 / 3
    assert metrics["class_counts"] == {"normal_mild": 1, "moderate": 0, "severe": 0}
    assert metrics["by_level"]["L5-S1"]["conditional"]["macro_f1"] is None


def test_cache_checksum_detects_modified_manifest(tmp_path):
    root = make_cache(tmp_path / "cache")
    with (root / "target_manifest.csv").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        load_feature_data(root)


@pytest.mark.parametrize("field", ["labels", "split"])
def test_key_alignment_rejects_label_or_split_disagreement(tmp_path, field):
    root = make_cache(tmp_path / "cache")
    if field == "labels":
        rewrite_array(root, field, lambda values: (values + 1) % 3)
    else:
        rewrite_array(
            root, field, lambda values: np.where(values == "train", "validation", "train")
        )
    with pytest.raises(ValueError, match="disagree"):
        load_feature_data(root)


def test_test_features_are_refused_even_if_manifest_hash_is_valid(tmp_path):
    root = make_cache(tmp_path / "cache")
    rewrite_array(root, "split", lambda values: np.asarray(["test"] * len(values)))
    with pytest.raises(ValueError, match="test must remain untouched"):
        load_feature_data(root)


@pytest.mark.parametrize("complete", [False, "true", None])
def test_unfinished_cache_is_not_an_experiment(tmp_path, complete):
    root = make_cache(tmp_path / "cache")
    update_hashes(root, extraction_complete=complete)
    with pytest.raises(ValueError, match="incomplete"):
        evaluate_baseline(root, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_class_weights_use_counts_and_require_all_training_categories():
    weights = inverse_sqrt_class_weights(np.asarray([0, 0, 0, 0, 1, 2]))
    np.testing.assert_allclose(weights, [0.6, 1.2, 1.2])
    with pytest.raises(ValueError, match="per class"):
        inverse_sqrt_class_weights(np.asarray([0, 1]))


def test_three_cpu_head_runs_export_reproducible_predictions_without_mutating_cache(tmp_path):
    root = make_cache(tmp_path / "cache")
    before = digest(root / "features.npz")
    summary = train_head(
        root,
        tmp_path / "training",
        max_epochs=12,
        patience=3,
        learning_rate=0.05,
        batch_size=9,
        threads=1,
    )
    assert summary["configuration"]["encoder_updated"] is False
    assert summary["configuration"]["n_training_features"] == 9
    assert summary["configuration"]["assessed_training_class_counts"] == [3, 3, 3]
    assert summary["test_evaluated"] is False
    assert [run["seed"] for run in summary["runs"]] == [42, 2024, 3407]
    assert digest(root / "features.npz") == before
    cache = load_feature_data(root)
    validation = {
        tuple(str(cache.arrays[name][index]) for name in ("study_id", "series_id", "level")): index
        for index, split in enumerate(cache.arrays["split"])
        if split == "validation"
    }
    for result in summary["runs"]:
        seed_dir = tmp_path / "training" / "runs" / f"seed-{result['seed']}"
        probabilities = predict_head(cache.arrays["features"], seed_dir / "head.pt")
        with (seed_dir / "predictions.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if row["status"] == "ok":
                index = validation[tuple(row[name] for name in ("study_id", "series_id", "level"))]
                np.testing.assert_allclose(
                    [float(row[f"p_{name}"]) for name in ("normal_mild", "moderate", "severe")],
                    probabilities[index],
                    rtol=1e-6,
                )
        history = json.loads((seed_dir / "history.json").read_text())
        assert 1 <= result["best_epoch"] <= result["epochs_run"] <= 12
        best_score = max(epoch["validation_macro_f1"] for epoch in history)
        assert result["best_epoch"] == next(
            epoch["epoch"] for epoch in history if epoch["validation_macro_f1"] == best_score
        )
        assert history[-1]["training_loss"] < history[0]["training_loss"]


def test_exported_checkpoint_rejects_other_feature_dimensions(tmp_path):
    with pytest.raises(ValueError, match="512"):
        predict_head(np.zeros((1, 128)), tmp_path / "unused.pt")
