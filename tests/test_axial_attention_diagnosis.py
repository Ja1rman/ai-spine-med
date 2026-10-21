import pytest
import torch

from ai_spine_med.axial_attention_diagnosis import explain_features
from ai_spine_med.multimodal import MultimodalHead


def _head():
    torch.manual_seed(7)
    buffers = {
        name: torch.zeros(512) if "mean" in name else torch.ones(512)
        for name in ("sag_mean", "sag_std", "axial_mean", "axial_std")
    }
    return MultimodalHead("axial", buffers).eval()


def test_diagnosis_matches_forward_preserves_parameters_and_separates_interventions():
    head = _head()
    state = {k: v.clone() for k, v in head.state_dict().items()}
    features, positions = torch.randn(5, 512), torch.rand(5, 2)
    with torch.inference_mode():
        result = explain_features(head, features, positions)
        logits, weights = head(
            torch.zeros(1, 2, 512),
            features[None],
            positions[None],
            torch.ones(1, 5, dtype=torch.bool),
            return_attention=True,
        )
        assert torch.equal(result["probabilities"], logits[0].softmax(-1))
        assert torch.equal(result["attention"], weights[0])
        assert torch.allclose(result["attention"].sum(-1), torch.ones(2))
        pooled = weights[0] @ features
        expected = head.head(torch.cat((pooled, head.level_one_hot.flip(0)), -1)).softmax(-1)
        assert torch.equal(result["swapped_level_probabilities"], expected)
        expected = head.head(torch.cat((pooled.flip(0), head.level_one_hot), -1)).softmax(-1)
        assert torch.equal(result["swapped_attention_probabilities"], expected)
    assert all(torch.equal(v, head.state_dict()[k]) for k, v in state.items())


def test_flat_attention_equals_uniform_pooling():
    head = _head()
    for parameter in head.attention.parameters():
        parameter.data.zero_()
    with torch.inference_mode():
        result = explain_features(head, torch.randn(6, 512), torch.rand(6, 2))
    assert torch.allclose(
        result["probabilities"], result["uniform_attention_probabilities"], atol=1e-7
    )


def test_training_head_is_rejected():
    with pytest.raises(ValueError, match="evaluation mode"):
        explain_features(_head().train(), torch.randn(4, 512), torch.rand(4, 2))
