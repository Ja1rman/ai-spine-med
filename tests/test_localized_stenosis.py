"""Synthetic, patient-grouped checks for the RSNA-only localized ROI heads."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ai_spine_med import localized_stenosis as severity


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def protocol():
    return {
        "severity_seeds": list(severity.SEEDS),
        "severity_max_epochs": 2,
        "severity_patience": 1,
    }


def write_data(root, *, test=False, aisslab=True):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rows, features = [], []
    rng = np.random.default_rng(8)
    groups = [("test", 3)] if test else [("train", 6), ("validation", 3)]
    for split, n in groups:
        for i in range(n):
            for j, level in enumerate(severity.LEVELS):
                label = (i + j) % 3
                feature = rng.normal(0, 0.05, 512).astype(np.float32)
                feature[label] += 2
                if split != "train":
                    feature[10:] += 20
                rows.append(
                    {
                        "dataset": "rsna",
                        "study_id": f"{split}-{i}",
                        "split": split,
                        "level": level,
                        "label": label,
                        "status": "ok",
                        "reason": "",
                        "feature_index": len(features),
                    }
                )
                features.append(feature)
    if aisslab:
        split = "test" if test else "validation"
        for j, level in enumerate(severity.LEVELS):
            rows.append(
                {
                    "dataset": "aisslab",
                    "study_id": f"ais-{split}",
                    "split": split,
                    "level": level,
                    "label": j,
                    "status": "ok",
                    "reason": "",
                    "feature_index": len(features),
                }
            )
            features.append(np.full(512, 1000 + j, np.float32))
    features = np.asarray(features, np.float32)
    np.savez(root / "features.npz", features=features)
    write_rows(root / "targets.csv", rows)
    return root / "features.npz", root / "targets.csv", rows, features


def write_rows(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=severity.FIELDS)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def fitted(tmp_path):
    features, targets, rows, matrix = write_data(tmp_path / "fit")
    out = tmp_path / "run"
    summary = severity.train_localized_stenosis(features, targets, out, protocol())
    return out, summary, features, targets, rows, matrix


def test_protocol_reads_frozen_root_severity_fields():
    p = severity._protocol({"severity_learning_rate": 0.001, "severity_max_epochs": 4})
    assert p["max_epochs"] == 4
    assert p["seeds"] == [42, 2024, 3407]
    with pytest.raises(ValueError, match="fixed"):
        severity._protocol({"severity_learning_rate": 0.1})


def test_training_rejects_test_before_any_feature_file_read(tmp_path):
    _, targets, _, _ = write_data(tmp_path / "test", test=True)
    with pytest.raises(ValueError, match="TEST"):
        severity.train_localized_stenosis(
            tmp_path / "does-not-exist.npz", targets, tmp_path / "run", protocol()
        )


def test_patient_group_cannot_cross_splits(tmp_path):
    _, targets, rows, _ = write_data(tmp_path)
    rows[-1]["dataset"] = "rsna"
    rows[-1]["study_id"] = "train-0"
    with pytest.raises(ValueError, match="crosses splits"):
        write_rows(targets, rows)
        severity.read_targets(targets)


def test_feature_join_rejects_duplicates_and_noncontiguous_indices(tmp_path):
    _, targets, rows, _ = write_data(tmp_path)
    rows[0]["feature_index"] = len(rows) + 1
    write_rows(targets, rows)
    with pytest.raises(ValueError, match="contiguous"):
        severity.read_targets(targets)
    rows[0]["feature_index"] = rows[1]["feature_index"]
    write_rows(targets, rows)
    with pytest.raises(ValueError, match="unique feature"):
        severity.read_targets(targets)


def test_feature_shape_dtype_and_finiteness_are_checked(tmp_path):
    feature_path, targets, _, matrix = write_data(tmp_path)
    rows = severity.read_targets(targets)
    matrix[0, 0] = np.nan
    np.savez(feature_path, features=matrix)
    with pytest.raises(ValueError, match="finite float32"):
        severity.load_features(feature_path, rows)
    np.savez(feature_path, features=np.zeros((len(rows), 512), np.float64))
    with pytest.raises(ValueError, match="finite float32"):
        severity.load_features(feature_path, rows)


def test_fit_normalizers_weights_and_groups_use_rsna_train_only(fitted):
    out, summary, _, _, rows, features = fitted
    models, manifest, seal = severity.load_sealed_models(out)
    train = [r for r in rows if r["dataset"] == "rsna" and r["split"] == "train"]
    expected_mean = features[[r["feature_index"] for r in train]].mean(0, dtype=np.float64)
    np.testing.assert_allclose(models[0].mean.numpy(), expected_mean, atol=1e-7)
    assert manifest["class_counts"] == [4, 4, 4]
    assert manifest["class_weights"] == [1, 1, 1]
    assert not manifest["AISSLab_grades_used_for_training_or_selection"]
    assert all(dataset == "rsna" for dataset, _ in manifest["gradient_patient_groups"])
    assert all(dataset == "rsna" for dataset, _ in manifest["selection_patient_groups"])
    assert summary["status"] == "complete"
    assert seal["frozen"] and seal["seeds"] == list(severity.SEEDS)
    assert len(seal["checkpoints"]) == 3


def test_aisslab_reference_changes_cannot_change_fitted_weights(tmp_path):
    features, targets, rows, _ = write_data(tmp_path / "data")
    severity.train_localized_stenosis(features, targets, tmp_path / "a", protocol())
    first = [
        torch.load(tmp_path / "a" / f"runs/seed-{s}/model.pt", weights_only=True)["state_dict"]
        for s in severity.SEEDS
    ]
    for row in rows:
        if row["dataset"] == "aisslab":
            row["label"] = 2 - row["label"]
    write_rows(targets, rows)
    severity.train_localized_stenosis(features, targets, tmp_path / "b", protocol())
    second = [
        torch.load(tmp_path / "b" / f"runs/seed-{s}/model.pt", weights_only=True)["state_dict"]
        for s in severity.SEEDS
    ]
    for left, right in zip(first, second, strict=True):
        assert left.keys() == right.keys()
        assert all(torch.equal(left[key], right[key]) for key in left)


def test_missing_reference_still_gets_prediction_and_failures_remain_blank(fitted, tmp_path):
    out, _, _, _, _, _ = fitted
    features, targets, rows, matrix = write_data(tmp_path / "test", test=True)
    rows[0]["label"] = ""
    rows.append(
        {
            "dataset": "rsna",
            "study_id": "missing-test",
            "split": "test",
            "level": "L5-S1",
            "label": 2,
            "status": "not_assessed",
            "reason": "localizer_unavailable",
            "feature_index": "",
        }
    )
    write_rows(targets, rows)
    summary = severity.evaluate_localized_stenosis(features, targets, out)
    with (out / "evaluation/ensemble.csv").open() as stream:
        predicted = list(csv.DictReader(stream))
    assert predicted[0]["label"] == "" and predicted[0]["reference_status"] == "missing"
    assert predicted[0]["prediction"] != "" and predicted[0]["p_normal_mild"] != ""
    assert predicted[-1]["prediction"] == "" and predicted[-1]["p_severe"] == ""
    metric = summary["outputs"]["ensemble.csv"]["metrics"]
    assert metric["n_requested"] == 9 and metric["n_labeled"] == 8
    assert metric["n_predictions_including_missing_reference"] == len(matrix)
    assert metric["by_dataset"]["aisslab"]["clinical_scale_equivalence_verified"] is False


def test_missing_seal_rejects_before_test_reads(tmp_path):
    with pytest.raises(FileNotFoundError):
        severity.evaluate_localized_stenosis("no-features.npz", "no-targets.csv", tmp_path)


def test_fitting_sources_are_pinned(fitted):
    out, _, _, targets, _, _ = fitted
    targets.write_text(targets.read_text() + "\n")
    with pytest.raises(ValueError, match="fitting source changed"):
        severity.load_sealed_models(out)


def test_checkpoint_tampering_is_rejected(fitted):
    out, *_ = fitted
    checkpoint = out / "runs/seed-42/model.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        severity.load_sealed_models(out)


def test_test_patient_overlap_rejected_before_feature_read(fitted, tmp_path):
    out, _, _, _, _, _ = fitted
    _, targets, rows, _ = write_data(tmp_path / "test", test=True)
    rows[0]["study_id"] = rows[1]["study_id"] = "train-0"
    write_rows(targets, rows)
    with pytest.raises(ValueError, match="occurred in fitting"):
        severity.evaluate_localized_stenosis("missing.npz", targets, out)


def test_ensemble_is_equal_probability_mean_and_dropout_is_off(fitted, tmp_path):
    out, _, _, _, _, _ = fitted
    features_path, targets_path, raw_rows, features = write_data(tmp_path / "test", test=True)
    severity.evaluate_localized_stenosis(features_path, targets_path, out)
    models, _, _ = severity.load_sealed_models(out)
    levels = [severity.LEVELS.index(r["level"]) for r in raw_rows]
    probabilities = [severity.predict_localized_features(m, features, levels) for m in models]
    np.testing.assert_array_equal(
        probabilities[0], severity.predict_localized_features(models[0], features, levels)
    )
    expected = np.mean(probabilities, axis=0)
    with (out / "evaluation/ensemble.csv").open() as stream:
        observed = list(csv.DictReader(stream))
    actual = np.asarray([[float(r[f"p_{c}"]) for c in severity.CLASSES] for r in observed])
    np.testing.assert_allclose(actual, expected, atol=1e-7)


def test_existing_experiments_cannot_be_overwritten(fitted):
    out, _, features, targets, _, _ = fitted
    with pytest.raises(ValueError, match="never overwrite"):
        severity.train_localized_stenosis(features, targets, out, protocol())


def test_all_three_seed_results_required_by_seal(fitted):
    out, *_ = fitted
    seal_path = out / "selection.json"
    seal = json.loads(seal_path.read_text())
    seal["checkpoints"].pop("runs/seed-3407/model.pt")
    seal_path.write_text(json.dumps(seal))
    with pytest.raises(ValueError, match="hashes differ"):
        severity.load_sealed_models(out)
