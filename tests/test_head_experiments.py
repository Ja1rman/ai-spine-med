import copy
import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from ai_spine_med import head_experiments as experiments
from ai_spine_med.data import CLASS_NAMES
from ai_spine_med.training import FeatureData, inverse_sqrt_class_weights


def fixture_data():
    targets, arrays = [], []
    for study in range(22):
        split = "train" if study < 20 else "validation"
        for level_index, level in enumerate(("L4-L5", "L5-S1")):
            label = study % 3
            status, reason, index = "ok", "", len(arrays)
            if study == 18:
                status, reason, index = "not_assessed", "missing_roi", None
            if study == 19:
                label, status, reason, index = None, "missing_label", "no_reference", None
            row = {
                "study_id": str(study),
                "series_id": str(100 + study),
                "level": level,
                "split": split,
                "label": label,
                "status": status,
                "reason": reason,
                "feature_index": index,
            }
            targets.append(row)
            if index is not None:
                arrays.append(row)
    values = {
        "features": np.zeros((len(arrays), 512), dtype=np.float32),
        "original_logits": np.zeros((len(arrays), 4), dtype=np.float32),
        "labels": np.asarray([row["label"] for row in arrays], dtype=np.int64),
    }
    for name in ("study_id", "series_id", "level", "split"):
        values[name] = np.asarray([row[name] for row in arrays])
    return FeatureData(values, targets, {"cache_id": "fixture"}, {"features.npz": "fixture"})


def fixture_protocol():
    return {
        "schema_version": 1,
        "feature_cache_id": "fixture",
        "candidates": ["linear", "mlp", "coral"],
        "seeds": [42, 2024, 3407],
        "classes": list(CLASS_NAMES),
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 64,
        "max_epochs": 30,
        "patience": 5,
        "cv_folds": 3,
        "cv_seed": 2026,
        "feature_standardization": False,
        "encoder_training": False,
        "augmentation": False,
    }


def test_coral_order_probabilities_and_gradients():
    torch.manual_seed(5)
    model = experiments.CoralHead()
    x = torch.randn(9, 512)
    labels = torch.arange(9) % 3
    logits = model(x)
    assert torch.all(logits[:, 0] >= logits[:, 1])
    probabilities = experiments.head_probabilities("coral", logits)
    assert torch.all(probabilities >= 0)
    torch.testing.assert_close(probabilities.sum(1), torch.ones(9))
    loss = experiments.weighted_head_loss("coral", logits, labels, torch.tensor([0.5, 1, 2]))
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert model.raw_gap.grad.abs() > 0
    with torch.no_grad():
        model.raw_gap.fill_(-80)
        model.threshold0.fill_(50)
    extreme = experiments.head_probabilities("coral", model(x * 1000))
    assert torch.isfinite(extreme).all() and torch.all(extreme >= 0)
    torch.testing.assert_close(extreme.sum(1), torch.ones(9))


def test_coral_weighting_is_per_row_and_categorical_matches_ce():
    labels = torch.tensor([0, 1, 2, 2])
    weights = torch.tensor([0.4, 1.1, 1.5])
    logits = torch.tensor([[2.0, 1], [0, -1], [1, 0], [-2, -3]], requires_grad=True)
    binary = labels[:, None] > torch.arange(2)[None, :]
    row_losses = F.binary_cross_entropy_with_logits(logits, binary.float(), reduction="none").mean(
        1
    )
    expected = (row_losses * weights[labels]).sum() / weights[labels].sum()
    torch.testing.assert_close(
        experiments.weighted_head_loss("coral", logits, labels, weights), expected
    )
    categorical = torch.randn(4, 3)
    torch.testing.assert_close(
        experiments.weighted_head_loss("linear", categorical, labels, weights),
        F.cross_entropy(categorical, labels, weight=weights),
    )


def test_two_tiny_coral_epochs_use_fit_only_weights_and_final_fit_has_no_held_score():
    torch.set_num_threads(1)
    x = torch.randn(9, 512)
    y = torch.tensor([0, 0, 0, 0, 1, 2, 0, 1, 2])
    fit, held = np.arange(6), np.arange(6, 9)
    config = {"learning_rate": 0.001, "weight_decay": 0.0001, "batch_size": 64, "patience": 5}
    model, history, best_epoch, weights = experiments._fit(
        "coral", 42, x, y, fit, config=config, epochs=2, held_indices=held
    )
    np.testing.assert_array_equal(weights, inverse_sqrt_class_weights(y[fit].numpy()))
    assert not np.array_equal(weights, inverse_sqrt_class_weights(y.numpy()))
    earliest_best = max(history, key=lambda row: row["heldout_macro_f1"])["epoch"]
    assert best_epoch == earliest_best
    assert all(np.isfinite(row["training_loss"]) for row in history)
    assert all(parameter.grad is not None for parameter in model.parameters())
    _, fixed_history, fixed_epoch, _ = experiments._fit(
        "coral", 42, x, y, fit, config=config, epochs=2
    )
    assert fixed_epoch == 2
    assert [row["epoch"] for row in fixed_history] == [1, 2]
    assert all("heldout_macro_f1" not in row for row in fixed_history)


@pytest.mark.parametrize("name", experiments.CANDIDATES)
def test_exported_head_reloads_with_identical_probabilities(tmp_path, name):
    torch.manual_seed(19)
    model = experiments.build_head(name).eval()
    features = np.random.default_rng(7).normal(size=(5, 512)).astype(np.float32)
    with torch.inference_mode():
        expected = experiments.head_probabilities(name, model(torch.from_numpy(features))).numpy()
    checkpoint = {
        "schema_version": 1,
        "candidate": name,
        "architecture": experiments.architecture(name),
        "class_names": list(CLASS_NAMES),
        "feature_size": 512,
        "state_dict": model.state_dict(),
    }
    path = tmp_path / "head.pt"
    torch.save(checkpoint, path)
    np.testing.assert_array_equal(experiments.predict_experiment_head(features, path), expected)
    checkpoint["architecture"]["input_size"] = 511
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="architecture"):
        experiments.predict_experiment_head(features, path)


def test_folds_assign_unassessed_and_unlabeled_studies_without_validation():
    data = fixture_data()
    folds = experiments.make_folds(data)
    assigned = folds["assignments"]
    assert len(assigned) == 40
    by_study = {}
    for row in assigned:
        by_study.setdefault(row["study_id"], set()).add(row["fold"])
    assert set(by_study) == {str(study) for study in range(20)}
    assert all(len(value) == 1 for value in by_study.values())
    assert {row["status"] for row in assigned if row["study_id"] == "18"} == {"not_assessed"}
    assert {row["label"] for row in assigned if row["study_id"] == "19"} == {None}
    changed = copy.deepcopy(data)
    for row in changed.targets:
        if row["split"] == "validation":
            row["label"] = (row["label"] + 1) % 3
    assert experiments.make_folds(changed) == folds
    for fold in folds["folds"]:
        fit_rows = [
            row
            for row in data.targets
            if row["split"] == "train"
            and row["feature_index"] is not None
            and next(iter(by_study[row["study_id"]])) != fold["fold"]
        ]
        expected = np.bincount([row["label"] for row in fit_rows], minlength=3)
        assert fold["fitting_class_counts"] == expected.tolist()


def test_ensemble_joins_keys_preserves_missing_and_rejects_changed_reference():
    data = fixture_data()
    targets = [row for row in data.targets if row["study_id"] in {"0", "18"}]
    indices = experiments._indices(targets)
    probabilities = np.tile([0.1, 0.2, 0.7], (len(indices), 1))
    rows = experiments._prediction_rows(targets, probabilities, indices)
    combined = experiments.ensemble_rows([rows, rows[::-1], rows])
    assert len(combined) == 4
    assert sum(row["prediction"] is None for row in combined) == 2
    assert all(row["prediction"] == 2 for row in combined if row["status"] == "ok")
    wrong = copy.deepcopy(rows)
    wrong[0]["label"] = 2
    with pytest.raises(ValueError, match="labels or coverage"):
        experiments.ensemble_rows([rows, wrong, rows])
    with pytest.raises(ValueError, match="Duplicate"):
        experiments.ensemble_rows([rows + rows[:1], rows, rows])


def test_cv_preselection_precedes_validation_and_final_epochs_use_only_cv(tmp_path, monkeypatch):
    data = fixture_data()
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(fixture_protocol()))
    out = tmp_path / "out"
    monkeypatch.setattr(experiments, "load_feature_data", lambda _: data)
    calls = []

    def fake_fit(name, seed, x, y, fit_indices, *, config, epochs, held_indices=None):
        assert np.all(data.arrays["split"][fit_indices] == "train")
        if held_indices is not None:
            assert not (out / "selection.json").exists()
            assert np.all(data.arrays["split"][held_indices] == "train")
            assert set(data.arrays["study_id"][fit_indices]).isdisjoint(
                data.arrays["study_id"][held_indices]
            )
            best = (2, 4, 6)[(len(calls) // 3) % 3]
        else:
            assert len(calls) >= 27
            selection = json.loads((out / "selection.json").read_text())
            assert selection["created_before_original_validation_evaluation"] is True
            assert epochs == 4
            best = epochs
        calls.append((name, seed, best, held_indices is None))
        model = experiments.build_head(name)
        for parameter in model.parameters():
            parameter.data.zero_()
        return (
            model.eval(),
            [{"epoch": epoch} for epoch in range(1, best + 1)],
            best,
            inverse_sqrt_class_weights(y[fit_indices].numpy()),
        )

    monkeypatch.setattr(experiments, "_fit", fake_fit)
    result = experiments.run_head_experiments("unused", out, protocol_path=protocol, threads=1)
    assert len(calls) == 36 and sum(call[3] for call in calls) == 9
    assert result["validation_used_for_epoch_or_candidate_selection"] is False
    assert result["test_evaluated"] is False
    assert all(set(epochs.values()) == {4} for epochs in result["final_epochs"].values())
    for name in experiments.CANDIDATES:
        cv = result["candidates"][name]["cv"]
        assert cv["oof"]["ensemble"]["metrics"]["n_requested"] == 40
        assert cv["oof"]["ensemble"]["metrics"]["n_missing_reference_label"] == 2
        assert cv["oof"]["ensemble"]["metrics"]["n_assessed"] == 36
        assert result["candidates"][name]["validation"]["ensemble"]["metrics"]["n_requested"] == 4
    with pytest.raises(ValueError, match="never overwrite"):
        experiments.run_head_experiments("unused", out, protocol_path=protocol)
