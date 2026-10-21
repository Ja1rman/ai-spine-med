import json

import numpy as np
import pytest
import torch

from ai_spine_med.axial_localizer import (
    CanalCenterModel,
    SliceLocalizer,
    predict_slice_weights,
    read_training_data,
)


def test_zero_residual_is_positional_prior_and_padding_does_not_change_predictions():
    model = SliceLocalizer(np.zeros(512), np.ones(512), [0.7, 0.9], [0.1, 0.1]).eval()
    features = np.random.default_rng(2).normal(size=(5, 512)).astype(np.float32)
    positions = np.repeat(np.linspace(0, 1, 5, dtype=np.float32)[:, None], 2, axis=1)
    actual = predict_slice_weights(model, features, positions)
    expected = np.exp(-0.5 * ((positions[:, 0][None] - np.array([0.7, 0.9])[:, None]) / 0.1) ** 2)
    expected /= expected.sum(-1, keepdims=True)
    assert np.allclose(actual, expected, atol=1e-6)
    x = torch.from_numpy(np.concatenate((features, np.zeros((2, 512), np.float32))))[None]
    p = torch.from_numpy(np.concatenate((positions, np.zeros((2, 2), np.float32))))[None]
    with torch.inference_mode():
        padded = model(x, p, torch.tensor([[1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool)).softmax(-1)[0]
    assert np.allclose(padded[:, :5], actual)
    assert torch.equal(padded[:, 5:], torch.zeros(2, 2))


def test_canals_return_normalized_xy_without_mutating_image_features():
    model = CanalCenterModel(np.zeros(512), np.ones(512)).eval()
    maps = torch.randn(3, 512, 7, 7)
    before = maps.clone()
    with torch.inference_mode():
        xy, logits = model(maps)
    assert xy.shape == (3, 2) and logits.shape == (3, 49)
    assert (xy >= 0).all() and (xy <= 1).all()
    assert torch.equal(maps, before)


def test_test_patient_rejected_before_feature_access(tmp_path):
    index = tmp_path / "index.json"
    index.write_text(
        json.dumps({"records": [{"split": "test", "arrays_path": "does-not-exist.npz"}]})
    )
    with pytest.raises(ValueError, match="TEST is forbidden"):
        read_training_data(index)
