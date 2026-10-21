"""Synthetic anatomy and provenance tests, never patient images."""

import json
from pathlib import Path

import numpy as np
import pytest

from ai_spine_med import axial_localizer_data as data
from ai_spine_med.axial_features import _canonical_image


def header(orientation=(1, 0, 0, 0, 1, 0)):
    return {
        "image_shape": [5, 9],
        "orientation": list(orientation),
        "spacing": [2.0, 3.0],
        "origin_lps": [0.0, 0.0, 0.0],
        "series_id": "1",
        "instance": 1,
        "path": "series/1.dcm",
        "frame_sha256": "shared",
        "center_lps": [0, 0, 0],
        "normal": [0, 0, 1],
    }


@pytest.mark.parametrize(
    "orientation",
    [
        (1, 0, 0, 0, 1, 0),
        (-1, 0, 0, 0, 1, 0),
        (1, 0, 0, 0, -1, 0),
        (-1, 0, 0, 0, -1, 0),
        (0, 1, 0, 1, 0, 0),
        (0, -1, 0, 1, 0, 0),
        (0, 1, 0, -1, 0, 0),
        (0, -1, 0, -1, 0, 0),
    ],
)
def test_canonical_xy_matches_actual_decoder_pixels_and_physical_axes(orientation):
    item = header(orientation)
    image = np.zeros(item["image_shape"], np.float32)
    image[1, 6] = 1
    canonical, _ = _canonical_image(image, np.asarray(orientation[:3]), np.asarray(orientation[3:]))
    point = data.canonical_xy(item, 6, 1)
    row, column = np.argwhere(canonical == 1)[0]
    np.testing.assert_allclose(
        point, [column / (canonical.shape[1] - 1), row / (canonical.shape[0] - 1)]
    )
    geometry = data.canonical_geometry(item)
    assert geometry["canonical_shape"] == list(canonical.shape)
    expected_scale = [8.0, 24.0] if canonical.shape == (9, 5) else [24.0, 8.0]
    np.testing.assert_allclose(geometry["canonical_scale_mm"], expected_scale)


def slice_at(z, instance, series="1"):
    return {
        **header(),
        "center_lps": [0, 0, z],
        "instance": instance,
        "series_id": series,
        "path": f"series/{series}/{instance}.dcm",
    }


def anchor(item, level="L5-S1", kind="aisslab_xml_ccs_exact_slice"):
    return {
        "kind": kind,
        "level": level,
        "series_id": item["series_id"],
        "path": item["path"],
        "point_lps": item["center_lps"],
        "frame_sha256": item["frame_sha256"],
        "center_xy": [0.4, 0.6],
    }


def test_exact_ccs_slice_supervises_center_but_smooth_level_plane_target():
    slices = [slice_at(z, i + 1) for i, z in enumerate([10, 5, 0])]
    targets = data.build_targets(slices, [anchor(slices[-1])], sigma_mm=5)
    assert targets["level_mask"].tolist() == [False, True]
    assert targets["center_mask"].tolist() == [[False] * 3, [False, False, True]]
    np.testing.assert_allclose(targets["center_xy"][1, -1], [0.4, 0.6])
    assert targets["relevance"][1, -1] > targets["relevance"][1, 1] > targets["relevance"][1, 0] > 0
    assert targets["relevance"][1].sum() == pytest.approx(1)
    assert targets["relevance"][0].sum() == 0


def test_rsna_subarticular_point_never_supervises_canal_center():
    slices = [slice_at(10, 1), slice_at(0, 2)]
    reference = anchor(slices[-1], kind="rsna_axial_subarticular_approximate_level")
    targets = data.build_targets(slices, [reference])
    assert targets["level_mask"].tolist() == [False, True]
    assert not targets["center_mask"].any()


def test_unrelated_series_never_acquires_direct_reference_target():
    slices = [slice_at(0, 1), slice_at(0, 1, "2")]
    targets = data.build_targets(slices, [anchor(slices[0])])
    assert targets["relevance"][1].tolist() == [1, 0]
    assert targets["series_support"][1].tolist() == [True, False]


def test_localization_targets_invariant_to_severity_grade_and_ap_measurements():
    slices = [slice_at(0, 1), slice_at(4, 2)]
    reference = anchor(slices[0])
    one = data.build_targets(slices, [{**reference, "grade": 0, "AP_mm": 15.0}])
    two = data.build_targets(slices, [{**reference, "grade": 2, "AP_mm": 3.0}])
    for key in one:
        np.testing.assert_array_equal(one[key], two[key])


def test_missing_annotation_is_masked_not_normal_or_pseudo_truth():
    targets = data.build_targets([slice_at(0, 1)], [])
    assert not targets["level_mask"].any() and not targets["center_mask"].any()
    assert np.isinf(targets["anchor_distances_mm"]).all()
    assert not targets["relevance"].any()


def test_conflicting_exact_ccs_centers_rejected():
    slices = [slice_at(0, 1)]
    reference = anchor(slices[0])
    with pytest.raises(ValueError, match="Conflicting"):
        data.build_targets(slices, [reference, {**reference, "center_xy": [0.2, 0.2]}])


def test_preparation_refuses_test_before_any_medical_input_open(tmp_path, monkeypatch):
    monkeypatch.setattr(
        data, "_rsna_fitting_sources", lambda *a: pytest.fail("must reject before reading")
    )
    with pytest.raises(ValueError, match="fitting"):
        data.prepare_localizer_data(tmp_path / "out", phase="evaluation")
    assert not (tmp_path / "out").exists()


def test_rsna_fitting_metadata_refuses_test_before_reading_receipts(tmp_path):
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "config": {"encoder": {"sha256": data.WEIGHTS_SHA256}},
                "study_records": [{"split": "test"}],
            }
        )
    )
    with pytest.raises(ValueError, match="TEST"):
        data._rsna_fitting_sources(tmp_path, Path("missing-medical-root"))


def test_invalid_bandwidth_and_duplicate_slice_keys_rejected():
    item = slice_at(0, 1)
    with pytest.raises(ValueError, match="bandwidth"):
        data.build_targets([item], [], sigma_mm=float("nan"))
    with pytest.raises(ValueError, match="Duplicate"):
        data.build_targets([item, item], [])
