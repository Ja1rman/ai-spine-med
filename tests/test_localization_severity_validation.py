import csv
import json

import numpy as np
import pytest
import torch
from torch import nn

from ai_spine_med.axial_localizer import predict_slice_weights
from ai_spine_med.localization_severity_validation import (
    ProbabilitySliceEnsemble,
    image_descriptor,
    pack_features,
    read_validation_labels,
    validation_records,
)


class FixedSliceModel(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = nn.Parameter(torch.tensor(values, dtype=torch.float32))

    def forward(self, features, positions, mask):
        return (
            self.values[None].expand(len(features), -1, -1).masked_fill(~mask[:, None], -torch.inf)
        )


def test_localizer_ensemble_averages_probabilities_not_logits_or_indices():
    models = [
        FixedSliceModel([[8, 0, 0], [0, 8, 0]]),
        FixedSliceModel([[0, 2, 0], [0, 0, 2]]),
        FixedSliceModel([[0, 2, 0], [0, 0, 2]]),
    ]
    ensemble = ProbabilitySliceEnsemble(models)
    features, positions = np.zeros((3, 512), np.float32), np.zeros((3, 2), np.float32)
    expected = np.mean([predict_slice_weights(m, features, positions) for m in models], axis=0)
    result = predict_slice_weights(ensemble, features, positions)
    assert np.allclose(result, expected, atol=1e-7)
    # Averaging logits would pick slice0 for level0, whereas mean probability picks slice1.
    assert result[0].argmax() == 1
    assert np.argmax(np.mean([m.values.detach().numpy()[0] for m in models], axis=0)) == 0


def test_ensemble_preserves_masked_padding():
    model = FixedSliceModel([[0, 1, 20], [0, 1, 20]])
    ensemble = ProbabilitySliceEnsemble([model, model, model])
    output = ensemble(
        torch.zeros(1, 3, 512), torch.zeros(1, 3, 2), torch.tensor([[1, 1, 0]], dtype=torch.bool)
    )
    assert output.softmax(-1)[0, 0, 2] == 0
    assert output.softmax(-1)[0, 0].argmax() == 1


def test_validation_selector_preserves_original_failures_and_drops_train():
    rows = [
        {"dataset": "rsna", "study_id": "a", "split": "validation", "status": "failed"},
        {"dataset": "rsna", "study_id": "b", "split": "train", "status": "ok"},
    ]
    assert validation_records({"records": rows}) == rows[:1]


def test_test_rows_rejected_even_if_selector_would_drop_them():
    with pytest.raises(ValueError, match="TEST is forbidden"):
        validation_records({"records": [{"split": "test"}]})


def test_inference_descriptor_has_no_annotation_or_grade_fields():
    row = {
        "dataset": "aisslab",
        "study_id": "a",
        "split": "validation",
        "source": {},
        "selected_slices": [],
        "anchors": [{"center_xy": [0.3, 0.7]}],
        "labels": [0, 2],
        "level_mask": [True, True],
        "ap_diameter": 8,
    }
    result = image_descriptor(row)
    assert set(result) == {"dataset", "study_id", "split", "source", "selected_slices"}


def make_label_sources(tmp_path):
    rsna, ais = tmp_path / "rsna.csv", tmp_path / "aisslab.json"
    with rsna.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["study_id", "spinal_canal_stenosis_l4_l5", "spinal_canal_stenosis_l5_s1"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "study_id": "r",
                "spinal_canal_stenosis_l4_l5": "Moderate",
                "spinal_canal_stenosis_l5_s1": "Severe",
            }
        )
    ais.write_text(
        json.dumps(
            [
                {"study_id": "a", "split": "validation", "labels": [None, 2]},
                {"study_id": "a", "split": "test", "labels": [0, 0]},
                {"study_id": "a", "split": "train", "labels": [1, 1]},
            ]
        )
    )
    return rsna, ais


def test_reference_join_reads_validation_only_and_keeps_missing_grade(tmp_path):
    rsna, ais = make_label_sources(tmp_path)
    rows = [{"dataset": "rsna", "study_id": "r"}, {"dataset": "aisslab", "study_id": "a"}]
    labels = read_validation_labels(rows, rsna, ais)
    assert labels[("rsna", "r", "L5-S1")] == 2
    assert labels[("aisslab", "a", "L4-L5")] is None
    assert labels[("aisslab", "a", "L5-S1")] == 2


def test_absent_patient_is_not_silently_converted_to_missing_grade(tmp_path):
    rsna, ais = make_label_sources(tmp_path)
    with pytest.raises(ValueError, match="lacks a validation patient"):
        read_validation_labels([{"dataset": "rsna", "study_id": "unknown"}], rsna, ais)


def test_feature_packing_preserves_failure_and_missing_label(tmp_path):
    np.savez_compressed(tmp_path / "a.npz", features=np.ones((1, 512), np.float32))
    record = {
        "dataset": "aisslab",
        "study_id": "a",
        "split": "validation",
        "features_path": "a.npz",
        "targets": [
            {
                "level": "L4-L5",
                "status": "failed",
                "reason": "original_failure",
                "feature_index": None,
            },
            {"level": "L5-S1", "status": "ok", "reason": "", "feature_index": 0},
        ],
    }
    labels = {("aisslab", "a", "L4-L5"): 2, ("aisslab", "a", "L5-S1"): None}
    rows, features = pack_features([record], labels, tmp_path)
    assert len(rows) == 2 and features.shape == (1, 512)
    assert rows[0]["label"] == 2 and rows[0]["status"] == "failed"
    assert rows[1]["label"] is None and rows[1]["status"] == "ok"
