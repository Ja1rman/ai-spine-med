"""SpineNet coordinate inversion and registration gating use synthetic geometry."""

import hashlib
import itertools

import numpy as np
import pytest

from ai_spine_med import spinenet_axial_geometry as geometry


@pytest.mark.parametrize("transpose,flip_rows,flip_columns", list(itertools.product((False, True), repeat=3)))
def test_resampling_transpose_flips_roundtrip_without_rounding(transpose, flip_rows, flip_columns):
    shape, spacing = np.asarray([80, 120]), np.asarray([0.8, 0.5])
    source_xy = np.asarray([32.25, 23.75])
    canonical_shape = shape[::-1] if transpose else shape
    canonical_spacing = spacing[::-1] if transpose else spacing
    canonical_xy = source_xy[::-1].copy() if transpose else source_xy.copy()
    if flip_rows:
        canonical_xy[1] = canonical_shape[0] - 1 - canonical_xy[1]
    if flip_columns:
        canonical_xy[0] = canonical_shape[1] - 1 - canonical_xy[0]
    detector_xy = canonical_xy * canonical_spacing[::-1] / 0.5
    metadata = {
        "pixel_spacing_mm": 0.5,
        "resampled": True,
        "in_plane_transposed": transpose,
        "rows_flipped": flip_rows,
        "columns_flipped": flip_columns,
    }
    result = geometry.detector_to_source_xy(detector_xy, metadata, shape, spacing)
    np.testing.assert_allclose(result, source_xy, rtol=0, atol=1e-12)


def test_no_resampling_flag_preserves_pixels_when_spacing_within_original_tolerance():
    metadata = {
        "pixel_spacing_mm": 0.5,
        "resampled": False,
        "in_plane_transposed": False,
        "rows_flipped": False,
        "columns_flipped": False,
    }
    result = geometry.detector_to_source_xy([3.25, 4.5], metadata, [10, 20], [0.50001, 0.5])
    np.testing.assert_array_equal(result, [3.25, 4.5])


def test_oblique_dicom_conversion_respects_xy_spacing():
    header = {
        "orientation": [0, 1, 0, 0.6, 0, -0.8],
        "spacing": [2, 3],
        "origin_lps": [10, 20, 30],
    }
    np.testing.assert_allclose(geometry.source_xy_to_lps([4, 5], header), [16, 32, 22])
    header["orientation"] = [0, 1, 0, 0, 1, 0]
    with pytest.raises(ValueError, match="geometry"):
        geometry.source_xy_to_lps([4, 5], header)


def vertex(label, offset=0, slices=(0, 1, 2)):
    poly = [[60, 10 + offset], [60, 20 + offset], [20, 20 + offset], [20, 10 + offset]]
    return {"label": label, "slice_nos": list(slices), "polys": [poly] * len(slices)}


def test_disc_uses_posterior_endplate_midpoint_and_shared_slice():
    vertices = [vertex("L4"), vertex("L5", 15), vertex("S1", 35, (1, 2))]
    result = geometry.disc_landmark(vertices, "L5-S1")
    assert result["slice_index"] == 1
    np.testing.assert_allclose(result["detector_xy"], [60, 40])
    assert "not_canal_segmentation" in result["landmark_kind"]
    # A changed corner order still gives the same anatomical midpoint.
    for v in vertices:
        v["polys"] = [list(reversed(p)) for p in v["polys"]]
    assert geometry.disc_landmark(vertices, "L5-S1")["detector_xy"] == result["detector_xy"]


def test_disc_refuses_missing_duplicate_disjoint_and_reversed_levels():
    assert geometry.disc_landmark([vertex("L5")], "L5-S1")["reason"] == "missing_or_duplicate_vertebra"
    vertices = [vertex("L5"), vertex("S1", 30), vertex("S1", 40)]
    assert geometry.disc_landmark(vertices, "L5-S1")["reason"] == "missing_or_duplicate_vertebra"
    vertices = [vertex("L5", slices=(1,)), vertex("S1", 30, slices=(2,))]
    assert geometry.disc_landmark(vertices, "L5-S1")["reason"] == "no_shared_sagittal_slice"
    vertices = [vertex("L5", 30), vertex("S1")]
    assert geometry.disc_landmark(vertices, "L5-S1")["reason"] == "reversed_or_overlapping_endplates"


def axial(z, frame="frame", instance=1):
    return {
        "series_id": "series",
        "instance": instance,
        "origin_lps": [0, 0, z],
        "orientation": [1, 0, 0, 0, 1, 0],
        "spacing": [2, 3],
        "image_shape": [100, 100],
        "frame_sha256": hashlib.sha256(frame.encode()).hexdigest() if frame else None,
    }


def test_projection_refuses_nonmatching_or_missing_frame_even_same_positions():
    for frame in ("other", ""):
        result = geometry.project_registered_landmark([3, 4, 0], "frame", [axial(0, frame)])
        assert result["status"] == "not_localized"
        assert result["reason"] == "unregistered_frame"
    result = geometry.project_registered_landmark([3, 4, 0], "", [axial(0)])
    assert result["reason"] == "missing_sagittal_frame"


def test_projection_selects_nearest_plane_keeps_real_fov_and_source_xy():
    result = geometry.project_registered_landmark([30, 20, 9], "frame", [axial(0), axial(10, instance=2)])
    assert result["status"] == "ok"
    assert result["candidates"][0]["instance"] == 2
    assert result["candidates"][0]["distance_mm"] == 1
    assert result["candidates"][0]["source_xy"] == [10, 10]
    assert geometry.project_registered_landmark([1000, 20, 9], "frame", [axial(10)])["reason"] == "landmark_outside_axial_fov"


@pytest.mark.parametrize("point", [[np.nan, 1], [1], [1, 2, 3]])
def test_nonfinite_or_bad_shape_coordinates_rejected(point):
    metadata = {"pixel_spacing_mm": 1, "in_plane_transposed": False, "rows_flipped": False, "columns_flipped": False}
    with pytest.raises(ValueError, match="point"):
        geometry.detector_to_source_xy(point, metadata, [10, 20], [1, 1])
