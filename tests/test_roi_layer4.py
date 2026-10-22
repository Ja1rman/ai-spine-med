import copy
import json

import numpy as np
import pytest
import torch
from torch import nn

from ai_spine_med import roi_layer4 as ft
from ai_spine_med.localized_stenosis import LocalizedSeverityHead


def model():
    torch.manual_seed(9)
    tail = nn.Sequential(nn.Conv2d(256, 512, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU()).eval()
    head = LocalizedSeverityHead(np.zeros(512), np.ones(512)).eval()
    return ft.RoiLayer4(tail, head)


def rows(split):
    return [
        {
            "dataset": "rsna",
            "study_id": f"{split}-{i}",
            "split": split,
            "level": ("L4-L5", "L5-S1")[i % 2],
            "label": i % 3,
            "status": "ok",
            "reason": "",
            "feature_index": i,
        }
        for i in range(6)
    ]


def cache(split):
    offsets = np.array([[0, 1], [1, 3], [3, 6], [6, 7], [7, 9], [9, 12]], np.int64)
    maps = np.random.default_rng(7).normal(size=(12, 256, 14, 14)).astype(np.float32)
    return rows(split), maps, offsets, np.zeros((6, 512), np.float32), {}


def test_masked_pool_uses_only_actual_centers_and_preserves_gradients():
    net = model().eval()
    values = torch.randn(2, 3, 256, 14, 14, requires_grad=True)
    mask = torch.tensor([[True, False, False], [True, True, True]])
    features = net.roi_features(values, mask)
    with torch.no_grad():
        first = net.layer4(values[0, :1]).mean((2, 3))[0]
        second = net.layer4(values[1]).mean((2, 3)).mean(0)
    assert torch.allclose(features[0], first, atol=1e-6)
    assert torch.allclose(features[1], second, atol=1e-6)
    features.sum().backward()
    assert values.grad[0, 1:].count_nonzero() == 0
    assert values.grad[0, :1].abs().sum() > 0
    assert net.layer4[0].weight.grad.abs().sum() > 0


def test_training_keeps_all_bn_statistics_eval_but_head_dropout_trains():
    net = model().train()
    assert net.head.network[2].training
    assert not net.layer4[1].training
    assert net.layer4[1].weight.requires_grad
    before = copy.deepcopy(net.layer4[1].state_dict())
    net.roi_features(torch.randn(2, 3, 256, 14, 14), torch.ones(2, 3, dtype=torch.bool))
    assert all(torch.equal(v, net.layer4[1].state_dict()[k]) for k, v in before.items())


def test_bag_batch_copies_memmap_without_mutation_or_duplicating_labels():
    _, maps, offsets, _, _ = cache("train")
    original = maps.copy()
    values, mask = ft.bag_batch(maps, offsets, [2, 0], "cpu")
    assert values.shape == (2, 3, 256, 14, 14)
    assert mask.tolist() == [[True, True, True], [True, False, False]]
    assert np.array_equal(values[0].numpy(), maps[3:6])
    assert np.array_equal(maps, original)


@pytest.mark.parametrize("offsets", [[[0, 0]], [[-1, 2]], [[0, 4]], [[0, 30]]])
def test_invalid_bag_boundaries_rejected(offsets):
    with pytest.raises(ValueError):
        ft.bag_batch(np.zeros((6, 256, 14, 14), np.float32), np.array(offsets), [0], "cpu")


def fit_inputs():
    train, val = cache("train"), cache("validation")
    indexed = {(r["dataset"], r["study_id"], r["level"]): r for r in train[0]}
    keys = sorted(indexed)
    config = {
        "encoder_learning_rate": 1e-5,
        "head_learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "batch_size": 3,
        "max_epochs": 2,
        "patience": 2,
    }
    return train, val, indexed, keys, config


def test_fit_updates_layer4_and_head_selects_val_and_preserves_buffers(monkeypatch):
    initial = model()
    before = ft.v9._state(initial)
    train, val, indexed, keys, config = fit_inputs()
    scores = iter([0.5, 0.6, 0.55])
    monkeypatch.setattr(
        ft, "evaluate_predictions", lambda _: {"conditional": {"macro_f1": next(scores)}}
    )
    fitted, selected, history, audit = ft.fit_layer4(
        initial, train, keys, indexed, val, np.ones(3), config, 42, "cpu"
    )
    assert selected == 1 and len(history) == 3
    assert audit["layer4_updated_during_training"]
    state = fitted.state_dict()
    assert not torch.equal(state["layer4.0.weight"], before["layer4.0.weight"])
    assert not torch.equal(state["head.network.0.weight"], before["head.network.0.weight"])
    assert torch.equal(state["layer4.1.running_mean"], before["layer4.1.running_mean"])
    assert torch.equal(
        state["layer4.1.num_batches_tracked"], before["layer4.1.num_batches_tracked"]
    )
    assert torch.equal(state["head.mean"], before["head.mean"])
    assert torch.equal(state["head.std"], before["head.std"])
    assert all(torch.equal(v, initial.state_dict()[k]) for k, v in before.items())


def test_fit_can_restore_epoch_zero_after_actual_updates(monkeypatch):
    initial = model()
    before = ft.v9._state(initial)
    train, val, indexed, keys, config = fit_inputs()
    scores = iter([0.8, 0.7, 0.6])
    monkeypatch.setattr(
        ft, "evaluate_predictions", lambda _: {"conditional": {"macro_f1": next(scores)}}
    )
    fitted, selected, _, audit = ft.fit_layer4(
        initial, train, keys, indexed, val, np.ones(3), config, 42, "cpu"
    )
    assert selected == 0 and audit["layer4_updated_during_training"]
    assert all(torch.equal(v, fitted.state_dict()[k]) for k, v in before.items())


@pytest.mark.parametrize("bad", ["train_domain", "val_test", "overlap"])
def test_wrong_domains_splits_or_patient_overlap_rejected(bad):
    train, val, indexed, keys, config = fit_inputs()
    if bad == "train_domain":
        train[0][0]["dataset"] = "aisslab"
    elif bad == "val_test":
        val[0][0]["split"] = "test"
    else:
        val[0][0]["study_id"] = train[0][0]["study_id"]
    with pytest.raises(ValueError):
        ft.fit_layer4(model(), train, keys, indexed, val, np.ones(3), config, 42, "cpu")


def test_historical_control_reads_validation_and_preserves_failed_targets(tmp_path, monkeypatch):
    predictions = ft.severity.prediction_rows(
        rows("validation"), model().head, np.zeros((6, 512), np.float32)
    )
    predictions[-1].update(
        status="failed",
        reason="original_failure",
        prediction=None,
        p_normal_mild=None,
        p_moderate=None,
        p_severe=None,
    )
    directory = tmp_path / "arms" / "xml_only" / "after"
    directory.mkdir(parents=True)
    ft.severity._write_predictions(directory / "seed_42.csv", predictions)
    (tmp_path / "summary.json").write_text(
        json.dumps({"arms": {"xml_only": {"training": {"selected_epochs": {"42": 2}}}}})
    )
    monkeypatch.setattr(ft.v9, "RUN", tmp_path)
    assert ft._compare_v9_control("xml_only", 42, 2, predictions)["v9_reproduced"]
    with pytest.raises(ValueError, match="cohort"):
        ft._compare_v9_control("xml_only", 42, 2, predictions[:-1])
