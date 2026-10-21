import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ai_spine_med.axial_diagnosis_report import (
    diagnose_interventions,
    oracle_window_probabilities,
    summarize_targets,
)
from ai_spine_med.multimodal import MultimodalHead


def test_missing_reference_and_missing_prediction_are_distinct():
    result = summarize_targets(
        [
            {"label": None, "probabilities": [1, 0, 0]},
            {"label": 2, "probabilities": None},
            {"label": 2, "probabilities": [0, 0, 1]},
        ]
    )
    assert result["n_requested"] == 3
    assert result["n_labeled"] == 2
    assert result["n_assessed_labeled"] == 1
    assert result["n_missing_reference"] == result["n_missing_prediction"] == 1
    assert result["severe_found"] == 1 and result["severe_total"] == 2
    assert sum(map(sum, result["confusion_matrix"])) == 1


def test_oracle_uses_correct_level_identity_and_preserves_absent_window():
    torch.manual_seed(1)
    buffers = {
        n: torch.zeros(512) if "mean" in n else torch.ones(512)
        for n in ("sag_mean", "sag_std", "axial_mean", "axial_std")
    }
    head = MultimodalHead("axial", buffers).eval()
    model = SimpleNamespace(head=head, training=False)
    features = np.random.default_rng(2).normal(size=(4, 512)).astype(np.float32)
    weights = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]], dtype=np.float32)
    masks = np.array([[False, True, False, True], [False] * 4])
    actual = oracle_window_probabilities(model, features, weights, masks)
    with torch.inference_mode():
        pooled = torch.from_numpy((features[1] * 0.2 + features[3] * 0.4) / 0.6)
        expected = head.head(torch.cat((pooled, head.level_one_hot[0]))).softmax(-1).numpy()
    assert np.allclose(actual[0], expected, atol=1e-7)
    assert actual[1] is None


def test_different_checkpoint_selections_rejected_before_loading_model_or_pixels(tmp_path):
    attention, run = tmp_path / "attention", tmp_path / "run"
    attention.mkdir()
    run.mkdir()
    (attention / "index.json").write_text(json.dumps({"selection_sha256": "different"}))
    anchors = tmp_path / "anchors.json"
    anchors.write_text("{}")
    (run / "selection.json").write_text("{}")
    with pytest.raises(ValueError, match="different checkpoint selections"):
        diagnose_interventions(attention, anchors, run, tmp_path / "out")
