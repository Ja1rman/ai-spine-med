import copy

import numpy as np
import pytest
import torch

from ai_spine_med import roi_attention as ft
from ai_spine_med.localized_stenosis import LocalizedSeverityHead


def model():
    torch.manual_seed(4)
    return ft.NeighborAttention(LocalizedSeverityHead(np.zeros(512), np.ones(512)))


def cache(split):
    offsets = np.array([[0, 1], [1, 3], [3, 6], [6, 7], [7, 9], [9, 12]], np.int64)
    features = np.random.default_rng(11).normal(size=(12, 512)).astype(np.float32)
    rows = [
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
    means = np.stack([features[a:b].mean(0) for a, b in offsets])
    return rows, features, offsets, means, {}


def test_zero_scorer_matches_actual_count_mean_and_initial_categories():
    net = model()
    data = cache("validation")
    pooled, weights = ft.pooled_features(net, data)
    assert np.allclose(pooled, data[3], atol=1e-6)
    assert np.allclose(weights[0], [1, 0, 0])
    assert np.allclose(weights[1], [0.5, 0.5, 0])
    assert np.allclose(weights[2], np.ones(3) / 3)
    audit = ft.check_initial_parity(net, data, 2e-5)
    assert audit["initial_weights_uniform"]
    assert audit["maximum_probability_error"] < 1e-6


def test_nonuniform_scores_pool_raw_vectors_and_ignore_padding():
    net = model().eval()
    with torch.no_grad():
        for p in net.scorer.parameters():
            p.zero_()
        net.scorer[0].weight[0, 0] = 1
        net.scorer[-1].weight[0, 0] = 1
    x = torch.zeros(1, 3, 512, requires_grad=True)
    with torch.no_grad():
        x[0, 1] = 2
        x[0, 2] = 1000
    mask = torch.tensor([[True, True, False]])
    pooled, weights = net.pool(x, mask)
    expected = torch.tensor([0.0, np.tanh(2.0)], dtype=torch.float32).softmax(0)
    assert torch.allclose(weights[0, :2], expected)
    assert weights[0, 2] == 0
    assert torch.allclose(pooled, 2 * expected[1] * torch.ones(1, 512))
    pooled.sum().backward()
    assert x.grad[0, 2].count_nonzero() == 0
    assert net.scorer[-1].weight.grad.abs().sum() > 0


def test_attention_pool_is_permutation_invariant_and_single_center_is_exact():
    net = model()
    with torch.no_grad():
        net.scorer[-1].weight.fill_(0.1)
    x = torch.randn(2, 3, 512)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    pooled, weights = net.pool(x, mask)
    perm = torch.tensor([2, 0, 1])
    other, otherweights = net.pool(x[:, perm], mask[:, perm])
    assert torch.allclose(pooled, other, atol=1e-6)
    assert torch.allclose(weights[:, perm], otherweights)
    assert torch.equal(pooled[1], x[1, 0])


def test_all_masked_bag_rejected_before_softmax():
    with pytest.raises(ValueError):
        model().pool(torch.zeros(2, 3, 512), torch.zeros(2, 3, dtype=torch.bool))


@pytest.mark.parametrize("boundaries", [[[0, 0]], [[-1, 2]], [[0, 4]], [[0, 99]]])
def test_invalid_original_bag_boundaries_rejected(boundaries):
    with pytest.raises(ValueError):
        ft.bag_batch(np.zeros((6, 512), np.float32), np.array(boundaries), [0], "cpu")


def inputs():
    train, val = cache("train"), cache("validation")
    indexed = {(r["dataset"], r["study_id"], r["level"]): r for r in train[0]}
    config = {
        "head_learning_rate": 1e-4,
        "attention_learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "batch_size": 3,
        "max_epochs": 2,
        "patience": 2,
    }
    return train, val, indexed, sorted(indexed), config


def test_fit_learns_score_projection_and_head_with_immutable_normalizers(monkeypatch):
    net = model()
    original = copy.deepcopy(net.state_dict())
    train, val, indexed, keys, config = inputs()
    scores = iter([0.5, 0.6, 0.55])
    monkeypatch.setattr(
        ft, "evaluate_predictions", lambda _: {"conditional": {"macro_f1": next(scores)}}
    )
    fitted, epoch, history, audit = ft.fit_attention(
        net, train, keys, indexed, val, np.ones(3), config, 42, "cpu"
    )
    assert epoch == 1 and len(history) == 3 and audit["candidate_attention_updated"]
    assert fitted.scorer[-1].weight.count_nonzero() > 0
    assert not torch.equal(fitted.head.network[0].weight, original["head.network.0.weight"])
    assert torch.equal(fitted.head.mean, original["head.mean"])
    assert torch.equal(fitted.head.std, original["head.std"])
    assert all(torch.equal(v, net.state_dict()[k]) for k, v in original.items())


def test_epoch_zero_restores_uniform_weights_after_real_training(monkeypatch):
    net = model()
    original = copy.deepcopy(net.state_dict())
    train, val, indexed, keys, config = inputs()
    scores = iter([0.8, 0.7, 0.6])
    monkeypatch.setattr(
        ft, "evaluate_predictions", lambda _: {"conditional": {"macro_f1": next(scores)}}
    )
    fitted, epoch, _, audit = ft.fit_attention(
        net, train, keys, indexed, val, np.ones(3), config, 42, "cpu"
    )
    assert epoch == 0 and audit["candidate_attention_updated"]
    assert all(torch.equal(v, fitted.state_dict()[k]) for k, v in original.items())
    assert fitted.scorer[-1].weight.count_nonzero() == 0


@pytest.mark.parametrize("bad", ["train_domain", "val_test", "overlap"])
def test_wrong_domains_splits_or_patient_overlap_rejected(bad):
    train, val, indexed, keys, config = inputs()
    if bad == "train_domain":
        train[0][0]["dataset"] = "aisslab"
    elif bad == "val_test":
        val[0][0]["split"] = "test"
    else:
        val[0][0]["study_id"] = train[0][0]["study_id"]
    with pytest.raises(ValueError):
        ft.fit_attention(model(), train, keys, indexed, val, np.ones(3), config, 42, "cpu")
