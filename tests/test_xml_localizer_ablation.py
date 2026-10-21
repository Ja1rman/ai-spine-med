import copy

import numpy as np
import pytest

from ai_spine_med.xml_localizer_ablation import (
    aggregate_cases,
    make_model,
    paired_bootstrap,
    partition_rows,
    score_weights,
)


def row(dataset="aisslab", split="train", identity="a"):
    return {
        "record": {"dataset": dataset, "split": split, "study_id": identity},
        "features": np.zeros((3, 512), np.float32),
        "positions": np.array([[0.0, 0], [0.5, 0], [1.0, 0]], np.float32),
        "level_mask": np.array([True, True]),
        "relevance": np.array([[0, 1, 0], [0, 0, 1]], np.float32),
        "anchor_distances_mm": np.array([[12, 0, 12], [24, 12, 0]], np.float32),
    }


def test_xml_only_filter_excludes_rsna_train_and_rsna_epoch_selection():
    data = [row(), row("rsna"), row(split="validation"), row("rsna", "validation")]
    train, selection = partition_rows(data, ["aisslab"])
    assert len(train) == len(selection) == 1
    assert train[0]["record"] == {"dataset": "aisslab", "split": "train", "study_id": "a"}
    assert selection[0]["record"]["dataset"] == "aisslab"
    mixed, selection_mixed = partition_rows(data, ["aisslab", "rsna"])
    assert len(mixed) == 2 and selection_mixed == selection


def test_test_is_rejected_even_if_not_in_requested_training_domain():
    with pytest.raises(ValueError, match="TEST forbidden"):
        partition_rows([row(), row(split="validation"), row("rsna", "test")], ["aisslab"])


def test_prior_and_normalizer_use_selected_train_only():
    r = row()
    r["features"][:] = 2
    model = make_model([r], "cpu")
    assert np.allclose(model.mean.numpy(), 2)
    assert np.allclose(model.prior_mean.numpy(), [0.5, 1.0])
    assert np.allclose(model.prior_std.numpy(), [0.08, 0.08])


def test_absent_reference_is_not_counted_as_correct_or_as_normal():
    r = row(split="validation")
    r["level_mask"][0] = False
    r["anchor_distances_mm"][0] = np.inf
    weights = np.array([[0, 1, 0], [0, 0, 1]], np.float32)
    cases = score_weights([r], [weights])
    groups = aggregate_cases(cases)["groups"]
    missing = groups["aisslab/L4-L5"]
    assert missing["requested"] == missing["predicted"] == missing["missing_reference"] == 1
    assert missing["n"] == 0 and missing["hit10"] is None
    assert groups["aisslab/L5-S1"]["hit10"] == 1


def test_other_series_is_failure_in_denominator_not_dropped():
    r = row(split="validation")
    r["anchor_distances_mm"][1, 0] = np.inf
    cases = score_weights([r], [np.array([[0, 1, 0], [1, 0, 0]])])
    group = aggregate_cases(cases)["groups"]["aisslab/L5-S1"]
    assert group["n"] == group["unverifiable_top_series"] == 1
    assert group["hit10_count"] == 0 and group["hit10"] == 0


def test_paired_comparison_rejects_same_counts_but_different_patients():
    cases = score_weights([row(split="validation")], [np.array([[0, 1, 0], [0, 0, 1]])])
    other = copy.deepcopy(cases)
    other[0]["study_id"] = "different"
    with pytest.raises(ValueError, match="cohorts differ"):
        paired_bootstrap(cases, other, 10, 42)


def test_paired_bootstrap_is_patient_paired_and_preserves_sign():
    cases = score_weights([row(split="validation")], [np.array([[0, 1, 0], [0, 0, 1]])])
    other = copy.deepcopy(cases)
    other[0]["hit10"] = False
    result = paired_bootstrap(cases, other, 100, 42)["groups"]
    assert result["aisslab/L4-L5"]["delta_hit10"] == 1
    assert result["aisslab/L4-L5"]["ci95"] == [1, 1]
    assert result["aisslab/L5-S1"]["ci95"] == [0, 0]


def test_invalid_slice_probabilities_cannot_produce_metrics():
    with pytest.raises(ValueError, match="Invalid slice probability"):
        score_weights([row(split="validation")], [np.ones((2, 3))])
