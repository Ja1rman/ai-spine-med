import copy
import csv

import numpy as np
import pytest
import torch

from ai_spine_med import localization_finetuning as ft
from ai_spine_med.localized_stenosis import LocalizedSeverityHead


def target(study="a", level="L5-S1", status="ok", label=2, split="train", dataset="rsna"):
    return {
        "dataset": dataset,
        "study_id": study,
        "split": split,
        "level": level,
        "status": status,
        "label": label,
        "reason": "" if status == "ok" else "failed",
        "feature_index": 0 if status == "ok" else None,
    }


def test_common_training_keeps_same_successful_grade_keys_only():
    a = [target(), target("b"), target("c", label=None)]
    b = [target("c", label=None), target("b", status="failed"), target()]
    keys, _ = ft.common_training_keys({"xml_only": a, "mixed": b})
    assert keys == [("rsna", "a", "L5-S1")]


@pytest.mark.parametrize("change", ["grade", "identity", "split", "domain"])
def test_unpaired_or_non_rsna_train_targets_rejected(change):
    a, b = [target()], [target()]
    if change == "grade":
        b[0]["label"] = 0
    elif change == "identity":
        b[0]["study_id"] = "other"
    elif change == "split":
        b[0]["split"] = "validation"
    else:
        b[0]["dataset"] = "aisslab"
    with pytest.raises(ValueError):
        ft.common_training_keys({"xml_only": a, "mixed": b})


def inputs():
    torch.manual_seed(21)
    initial = LocalizedSeverityHead(np.zeros(512), np.ones(512)).eval().requires_grad_(False)
    x = np.random.default_rng(3).normal(size=(6, 512)).astype(np.float32)
    levels, labels = np.array([0, 1, 0, 1, 0, 1]), np.array([0, 0, 1, 1, 2, 2])
    val = [
        target(
            str(i), label=int(labels[i]), level=("L4-L5", "L5-S1")[levels[i]], split="validation"
        )
        for i in range(6)
    ]
    for i, row in enumerate(val):
        row["feature_index"] = i
    config = {
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "batch_size": 3,
        "max_epochs": 2,
        "patience": 2,
    }
    return initial, x, levels, labels, val, config


def test_epoch_zero_restored_when_continuing_training_is_worse(monkeypatch):
    initial, x, levels, labels, val, config = inputs()
    before = ft._state(initial)
    scores = iter([0.8, 0.7, 0.6])
    monkeypatch.setattr(
        ft, "evaluate_predictions", lambda _: {"conditional": {"macro_f1": next(scores)}}
    )
    model, epoch, history, _ = ft.fit_head(
        initial, x, levels, labels, val, x, np.ones(3), config, 42, "cpu"
    )
    assert epoch == 0 and len(history) == 3
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    assert all(torch.equal(v, initial.state_dict()[k]) for k, v in before.items())
    assert not any(p.requires_grad for p in initial.parameters())


def test_same_seed_same_features_produce_same_adaptation_and_fixed_normalizers(monkeypatch):
    initial, x, levels, labels, val, config = inputs()
    original = copy.deepcopy(initial.state_dict())
    outputs = []
    for _ in range(2):
        scores = iter([0.5, 0.6, 0.55])
        monkeypatch.setattr(
            ft, "evaluate_predictions", lambda _: {"conditional": {"macro_f1": next(scores)}}
        )
        outputs.append(
            ft.fit_head(initial, x, levels, labels, val, x, np.ones(3), config, 42, "cpu")
        )
    first, second = outputs[0][0].state_dict(), outputs[1][0].state_dict()
    assert outputs[0][1] == outputs[1][1] == 1
    assert all(torch.equal(first[k], second[k]) for k in first)
    assert any(not torch.equal(first[k], original[k]) for k in first if k.startswith("network."))
    assert torch.equal(first["mean"], original["mean"]) and torch.equal(
        first["std"], original["std"]
    )
    assert all(torch.equal(initial.state_dict()[k], original[k]) for k in original)


def test_ais_validation_cannot_select_epoch():
    initial, x, levels, labels, val, config = inputs()
    val[0]["dataset"] = "aisslab"
    with pytest.raises(ValueError, match="Only RSNA validation"):
        ft.fit_head(initial, x, levels, labels, val, x, np.ones(3), config, 42, "cpu")


def test_epoch_zero_requires_actual_frozen_probabilities(tmp_path):
    path = tmp_path / "before.csv"
    fields = [
        "dataset",
        "study_id",
        "level",
        "label",
        "status",
        "prediction",
        "p_normal_mild",
        "p_moderate",
        "p_severe",
    ]
    row = {
        "dataset": "rsna",
        "study_id": "a",
        "level": "L5-S1",
        "label": 2,
        "status": "ok",
        "prediction": 2,
        "p_normal_mild": 0.1,
        "p_moderate": 0.2,
        "p_severe": 0.7,
    }
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    ft.epoch0_matches([row], path)
    row.update(p_normal_mild=0.2, p_severe=0.6)
    with pytest.raises(ValueError, match="Epoch zero does not reproduce"):
        ft.epoch0_matches([row], path)
