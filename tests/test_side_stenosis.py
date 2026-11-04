import numpy as np
import pytest
import torch

from ai_spine_med import side_stenosis as fit
from ai_spine_med import side_stenosis_data as data
from ai_spine_med.axial_features import _canonical_image
from ai_spine_med.axial_localizer_crops import canonical_to_source_xy


def target(**changes):
    return {
        "study_id": "1",
        "split": "train",
        "level": "L4-L5",
        "side": "left",
        "status": "ok",
        "reason": "",
        "feature_index": 0,
        "task_index": 0,
        "label": 2,
        "prediction": 2,
    } | changes


def test_anatomical_pools_are_not_image_sides():
    maps = torch.zeros(1, 512, 7, 7)
    maps[..., :3] = 2  # Image left is patient RIGHT.
    maps[..., 4:] = 7  # Image right is patient LEFT.
    values = data.spatial_pool(maps)
    assert torch.all(values[:, 512:1024] == 7)
    assert torch.all(values[:, 1024:] == 2)


@pytest.mark.parametrize(
    "column,row",
    [
        ([1, 0, 0], [0, 1, 0]),
        ([-1, 0, 0], [0, 1, 0]),
        ([0, 1, 0], [1, 0, 0]),
        ([0, -1, 0], [-1, 0, 0]),
    ],
)
def test_canonical_x_tracks_patient_lps_left_even_after_flips(column, row):
    column, row = np.asarray(column), np.asarray(row)
    image, action = _canonical_image(np.zeros((40, 50)), column, row)
    center = (np.asarray(image.shape)[::-1] - 1) / 2
    a = canonical_to_source_xy(center, (40, 50), action)
    b = canonical_to_source_xy(center + [1, 0], (40, 50), action)
    lps_delta = (b[0] - a[0]) * column + (b[1] - a[1]) * row
    assert lps_delta[0] == 1


def test_normalizer_cannot_fit_validation():
    with pytest.raises(ValueError, match="TRAIN"):
        fit.normalized_inputs(np.ones((1, 3), np.float32), [target(split="validation")])


def test_normalizer_uses_only_selected_train_indices():
    values = np.array([[1, 2], [3, 4], [1000, 1000]], np.float32)
    _, _, mean, _ = fit.normalized_inputs(values, [target(), target(feature_index=1)])
    np.testing.assert_array_equal(mean, [2, 3])


def test_missing_side_reference_is_not_normal():
    labels = {f"{side}_neural_foraminal_narrowing_l4_l5": "" for side in data.SIDES}
    rows = data.labels_for([target()], "foraminal", [{"study_id": "1", **labels}])
    assert all(r["label"] is None and r["reference_status"] == "missing" for r in rows)


def test_missing_reference_keeps_prediction_but_not_accuracy():
    rows = [target(label=None, prediction=1)]
    result = fit.metrics(rows)["overall"]
    assert result["assessed"] == 1 and result["known_references"] == 0
    assert result["macro_f1"] is None


def test_failure_remains_in_severe_denominator():
    rows = [target(), target(study_id="2", prediction=None, reason="missing_level")]
    result = fit.metrics(rows)["overall"]
    assert result["severe_total"] == 2 and result["severe_correct"] == 1
    assert result["severe_unassessed"] == 1 and result["known_and_assessed"] == 1


def test_each_level_side_head_has_distinct_weights():
    model = fit.BilateralHead(2, "linear")
    with torch.no_grad():
        model.net.weight.zero_()
        model.net.bias.copy_(torch.arange(12))
    scores = model(torch.zeros(4, 2), torch.arange(4))
    torch.testing.assert_close(scores, torch.arange(12).reshape(4, 3).float())


def test_predictions_cannot_hide_localization_failure():
    with pytest.raises(ValueError, match="failures"):
        fit.predictions([target(status="failed")], {("1", "L4-L5", "left"): np.array([1, 0, 0])})
