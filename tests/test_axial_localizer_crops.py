"""Predicted physical crops use synthetic DICOM; no annotated patient scans."""

import itertools

import numpy as np
import pytest
import torch
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from ai_spine_med import axial_features as af
from ai_spine_med import axial_localizer_crops as crops


@pytest.fixture(autouse=True)
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def make_record(tmp_path, *, slices=None, orientation=None, spacing=(2, 1), extra_series=False):
    slices = slices or [(1, [0, 0, 10]), (2, [5, 0, 5]), (3, [10, 0, 0])]
    study_uid = generate_uid()
    series = []
    for sid in ["10", "20"] if extra_series else ["10"]:
        directory = tmp_path / sid
        directory.mkdir()
        series_uid, frame_uid = generate_uid(), generate_uid()
        files = []
        for instance, origin in slices:
            path = directory / f"{instance}.dcm"
            meta = FileMetaDataset()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            meta.MediaStorageSOPClassUID = MRImageStorage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
            ds.SOPClassUID = meta.MediaStorageSOPClassUID
            ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
            ds.StudyInstanceUID, ds.SeriesInstanceUID = study_uid, series_uid
            ds.FrameOfReferenceUID = frame_uid
            ds.Modality = "MR"
            ds.ImageOrientationPatient = orientation or [1, 0, 0, 0, 1, 0]
            ds.ImagePositionPatient = origin
            ds.PixelSpacing = list(spacing)
            ds.InstanceNumber = instance
            y, x = np.mgrid[:80, :120]
            pixels = (x + y * 2 + instance).astype(np.uint16)
            ds.Rows, ds.Columns = pixels.shape
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated = ds.BitsStored = 16
            ds.HighBit, ds.PixelRepresentation = 15, 0
            ds.PixelData = pixels.tobytes()
            ds.save_as(path, enforce_file_format=True)
            files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": af._sha(path)})
        series.append({"series_id": sid, "series_path": sid, "files": files})
    return {"source_root": str(tmp_path), "source": {"series": series}}


def prediction(*, xy=(0.5, 0.5), instance=2, series="10"):
    return {
        "levels": {
            "L5-S1": {
                "status": "ok",
                "series_id": series,
                "instance": instance,
                "center_xy_norm": list(xy),
            }
        }
    }


@pytest.mark.parametrize(
    "transpose,flip_rows,flip_columns", list(itertools.product((False, True), repeat=3))
)
def test_canonical_xy_roundtrip_all_orientation_actions(transpose, flip_rows, flip_columns):
    action = (
        f"transpose={int(transpose)},flip_rows={int(flip_rows)},flip_columns={int(flip_columns)}"
    )
    source = np.asarray([10.25, 30.75])
    canonical = crops.source_to_canonical_xy(source, [80, 120], action)
    np.testing.assert_allclose(crops.canonical_to_source_xy(canonical, [80, 120], action), source)


def test_actual_dicom_pipeline_propagates_lps_point_across_translated_context_planes(tmp_path):
    record = make_record(tmp_path)
    result = crops.crop_level_inputs(record, prediction())
    level = result["levels"]["L5-S1"]
    assert level["status"] == "ok"
    assert level["images"].shape == (3, 3, 224, 224)
    assert torch.isfinite(level["images"]).all()
    metadata = level["metadata"]
    np.testing.assert_allclose(metadata["anchor_source_xy"], [59.5, 39.5])
    np.testing.assert_allclose(metadata["anchor_lps"], [64.5, 79, 5])
    expected = {1: [64.5, 39.5], 2: [59.5, 39.5], 3: [54.5, 39.5]}
    for center in metadata["centers"]:
        for channel in center["channels"]:
            np.testing.assert_allclose(channel["source_xy"], expected[channel["instance"]])
            assert channel["canonical_spacing_mm"] == [2, 1]
            assert channel["outside_fov_fraction"] == 0
    assert result["levels"]["L4-L5"]["status"] == "not_localized"
    assert result["levels"]["L4-L5"]["images"].shape == (0, 3, 224, 224)


def test_actual_transpose_flips_respect_anisotropic_spacing(tmp_path):
    # Original columns point anterior, rows point patient right: transpose and
    # both flips produce canonical left/right and anterior/posterior axes.
    record = make_record(tmp_path, orientation=[0, -1, 0, -1, 0, 0], slices=[(1, [100, 200, 0])])
    result = crops.crop_level_inputs(record, prediction(xy=(0.25, 0.75), instance=1), max_centers=1)
    metadata = result["levels"]["L5-S1"]["metadata"]
    canonical_xy = [0.25 * 79, 0.75 * 119]
    source_xy = [119 - canonical_xy[1], 79 - canonical_xy[0]]
    np.testing.assert_allclose(metadata["anchor_source_xy"], source_xy)
    np.testing.assert_allclose(
        metadata["anchor_lps"], [100 - 2 * source_xy[1], 200 - source_xy[0], 0]
    )
    assert metadata["centers"][0]["channels"][0]["canonical_spacing_mm"] == [1, 2]


def test_only_selected_series_contributes_centers_and_context(tmp_path):
    record = make_record(tmp_path, extra_series=True)
    result = crops.crop_level_inputs(record, prediction(series="20"))
    centers = result["levels"]["L5-S1"]["metadata"]["centers"]
    assert len(centers) == 3
    assert all(center["series_id"] == "20" for center in centers)
    assert all(channel["series_id"] == "20" for center in centers for channel in center["channels"])


def test_nearest_each_side_window_and_missing_context_replication(tmp_path):
    record = make_record(
        tmp_path,
        slices=[(1, [0, 0, 30]), (2, [0, 0, 4]), (3, [0, 0, 0]), (4, [0, 0, -5]), (5, [0, 0, -8])],
    )
    result = crops.crop_level_inputs(record, prediction(instance=3), window_mm=6)
    centers = result["levels"]["L5-S1"]["metadata"]["centers"]
    assert [center["instance"] for center in centers] == [3, 2, 4]
    # Existing decoder refuses the 26-mm jump and replicates the center.
    center2 = next(center for center in centers if center["instance"] == 2)
    assert [channel["instance"] for channel in center2["channels"]] == [2, 2, 3]


def test_padding_is_zero_before_normalization_not_zero_normalized_signal(tmp_path):
    record = make_record(tmp_path, slices=[(1, [0, 0, 0])])
    result = crops.crop_level_inputs(record, prediction(xy=(0, 0), instance=1), max_centers=1)
    level = result["levels"]["L5-S1"]
    assert level["metadata"]["centers"][0]["channels"][0]["outside_fov_fraction"] > 0.5
    expected = -torch.tensor(af.IMAGENET_MEAN) / torch.tensor(af.IMAGENET_STD)
    torch.testing.assert_close(level["images"][0, :, 0, 0], expected)


def test_physical_square_preserves_anisotropic_scale_and_fractional_center():
    y, x = np.mgrid[:101, :201]
    image = (x + 2 * y).astype(np.float32) / 500
    patch, padding = crops._physical_crop(image, [100.25, 50.75], np.asarray([2, 1]), 50)
    assert patch.shape == (224, 224) and padding == 0
    # Both source gradients are one unit per mm, so the physical square has
    # identical horizontal and vertical intensity changes despite anisotropy.
    np.testing.assert_allclose(
        float(patch[112, -1] - patch[112, 0]), float(patch[-1, 112] - patch[0, 112]), atol=1e-5
    )
    assert float(patch.mean()) == pytest.approx((100.25 + 2 * 50.75) / 500, abs=1e-6)


def test_changing_oracle_grades_and_annotations_cannot_change_api_output(tmp_path):
    record = make_record(tmp_path, slices=[(1, [0, 0, 0])])
    pred = prediction(instance=1)
    original = crops.crop_level_inputs(record, pred)["levels"]["L5-S1"]["images"]
    # Deliberately invalid/unreadable ground truth fields. Neither they nor any
    # adjacent annotation file are inputs to crop selection or its geometry.
    record["labels"] = [2, 2]
    record["ccs_references"] = [{"xml_path": "/does/not/exist.xml", "instance": 999}]
    (tmp_path / "train_label_coordinates.csv").write_text("UNUSABLE ORACLE")
    (tmp_path / "labels.xml").write_text("<INVALID")
    pred["levels"]["L5-S1"]["grade"] = 2
    changed = crops.crop_level_inputs(record, pred)["levels"]["L5-S1"]["images"]
    torch.testing.assert_close(changed, original, rtol=0, atol=0)


@pytest.mark.parametrize("xy", [[-0.1, 0.5], [0.5, 1.1], [np.nan, 0.5], [0.5]])
def test_invalid_predictions_are_explicit_not_normal(tmp_path, xy):
    record = make_record(tmp_path, slices=[(1, [0, 0, 0])])
    result = crops.crop_level_inputs(record, prediction(instance=1, xy=xy))
    level = result["levels"]["L5-S1"]
    assert level["reason"] == "invalid_predicted_center"
    assert level["images"].numel() == 0


def test_unknown_predicted_slice_refused_and_source_change_fails(tmp_path):
    record = make_record(tmp_path, slices=[(1, [0, 0, 0])])
    result = crops.crop_level_inputs(record, prediction(instance=999))
    assert result["levels"]["L5-S1"]["reason"] == "predicted_slice_not_in_source"
    path = tmp_path / "10" / "1.dcm"
    content = bytearray(path.read_bytes())
    content[-1] ^= 1
    path.write_bytes(content)
    with pytest.raises(af.SourceNotReady, match="source_changed"):
        crops.crop_level_inputs(record, prediction(instance=1))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"roi_mm": 0},
        {"roi_mm": float("nan")},
        {"max_centers": 0},
        {"max_centers": 4},
        {"window_mm": -1},
    ],
)
def test_bad_parameters_fail_before_reading_sources(kwargs):
    with pytest.raises(ValueError, match="parameters"):
        crops.crop_level_inputs({}, {}, **kwargs)
