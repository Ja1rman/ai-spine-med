import hashlib
import json
import math

import numpy as np
import pytest
import torch
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from ai_spine_med import axial_features as af

STUDY_UID = generate_uid()
FRAME_UID = generate_uid()
MANIFEST_SHA = "a" * 64


def write_slice(path, *, z=0, instance=1, orientation=None, pixels=None, **tags):
    pixels = np.asarray(pixels if pixels is not None else [[1, 2, 3], [4, 5, 6]], np.uint16)
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = STUDY_UID
    ds.SeriesInstanceUID = "1.2.3.4.5"
    ds.FrameOfReferenceUID = FRAME_UID
    ds.Modality = "MR"
    ds.PatientID = "PRIVATE_ID_DO_NOT_EXPORT"
    ds.PatientName = "PRIVATE_NAME_DO_NOT_EXPORT"
    ds.ImageOrientationPatient = orientation or [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [0, 0, z]
    ds.PixelSpacing = [1, 1]
    ds.InstanceNumber = instance
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    for key, value in tags.items():
        setattr(ds, key, value)
    ds.PixelData = pixels.tobytes()
    ds.save_as(path, enforce_file_format=True)
    return ds


def source(directory, data_root, series_id="10"):
    return {
        "series_id": series_id,
        "series_path": str(directory.relative_to(data_root)),
        "files": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": af._sha(path)}
            for path in sorted(directory.glob("*.dcm"))
        ],
    }


def marker(directory):
    files = sorted(path.name for path in directory.glob("*.dcm"))
    (directory / af.rsna_bulk.COMPLETE_MARKER).write_text(
        json.dumps(
            {
                "complete": True,
                "expected_files": len(files),
                "files": files,
                "metadata_sha256": "b" * 64,
                "manifest_sha256": MANIFEST_SHA,
                "source_identity_sha256": "c" * 64,
                "modality": "axial_v3",
                "scope": "axial_v3_train_validation",
                "allowed_splits": ["train", "validation"],
            }
        )
    )


def row(study, series, split="train"):
    return {
        "study_id": str(study),
        "series_id": str(series),
        "split": split,
        "series_description": "Axial T2",
        "series_path": f"train_images/{study}/{series}",
    }


class DummyEncoder(torch.nn.Module):
    def forward(self, inputs):
        # This also checks that preprocessing gives a finite three-channel tensor.
        assert inputs.shape[1:] == (3, 224, 224)
        assert torch.isfinite(inputs).all()
        return inputs.mean(dim=(1, 2, 3))[:, None].repeat(1, 512)


def setup_extractor(monkeypatch, tmp_path, rows):
    preparation = {
        "manifest_sha256": MANIFEST_SHA,
        "output_sha256": {"series.csv": "d" * 64},
        "input_sha256": {"studies.csv": "e" * 64},
    }
    monkeypatch.setattr(af, "_manifest", lambda *_: (rows, preparation))
    monkeypatch.setattr(
        af,
        "_weight_signature",
        lambda *_: {
            "architecture": "torchvision.resnet18",
            "weights": "IMAGENET1K_V1",
            "url": af.WEIGHTS_URL,
            "sha256": af.WEIGHTS_SHA256,
        },
    )
    monkeypatch.setattr(af, "_implementation_signature", lambda: {"code.py": "f" * 64})
    monkeypatch.setattr(af, "_load_encoder", lambda *_: DummyEncoder())
    for item in rows:
        directory = tmp_path / "data" / item["series_path"]
        directory.mkdir(parents=True)
        write_slice(directory / "9.dcm", z=4, instance=80)
        write_slice(directory / "1.dcm", z=0, instance=2)
        marker(directory)
    return tmp_path / "data", tmp_path / "features", tmp_path / "weights.pth"


@pytest.mark.parametrize(
    "column,row_axis,expected",
    [
        ([1, 0, 0], [0, 1, 0], [[1, 2, 3], [4, 5, 6]]),
        ([-1, 0, 0], [0, 1, 0], [[3, 2, 1], [6, 5, 4]]),
        ([1, 0, 0], [0, -1, 0], [[4, 5, 6], [1, 2, 3]]),
        ([0, 1, 0], [1, 0, 0], [[1, 4], [2, 5], [3, 6]]),
    ],
)
def test_canonical_inplane_axes(column, row_axis, expected):
    image, _ = af._canonical_image(
        np.asarray([[1, 2, 3], [4, 5, 6]]), np.asarray(column), np.asarray(row_axis)
    )
    np.testing.assert_array_equal(image, expected)
    assert image.flags.c_contiguous and image.dtype == np.float32


def test_physical_center_sort_multiangle_and_same_series_context(tmp_path):
    angle = math.radians(30)
    orientation = [1, 0, 0, 0, math.cos(angle), math.sin(angle)]
    # IPP z=0, but a 20-row oblique image has center z=4.75, cranial to z=3.
    write_slice(
        tmp_path / "1.dcm",
        z=0,
        instance=2,
        orientation=orientation,
        pixels=np.arange(60).reshape(20, 3),
    )
    write_slice(tmp_path / "9.dcm", z=3, instance=100)
    images, positions, contexts, qc = af._study_inputs(
        tmp_path, {"series": [source(tmp_path, tmp_path)]}
    )
    assert [item.instance for item in images] == [2, 100]
    np.testing.assert_array_equal(positions, [[0, 0], [1, 1]])
    # Thirty degrees between normals is too large: both context channels replicate.
    assert all(context[0] is context[1] is context[2] for context in contexts.values())
    assert qc["context_channels_replicated"] == 4
    assert "PRIVATE" not in str(qc)


def test_context_does_not_cross_gap_series_or_angle():
    current = af._Slice(np.zeros((2, 2)), np.array([0, 0, 4]), np.array([0, 0, 1]), 1, "1", "")
    neighbor = af._Slice(np.zeros((2, 2)), np.array([0, 0, 0]), np.array([0, 0, 1]), 2, "1", "")
    assert af._neighbor_allowed(current, neighbor)
    neighbor.center_lps[2] = -20
    assert not af._neighbor_allowed(current, neighbor)
    neighbor.center_lps[2] = 0
    neighbor.series_id = "2"
    assert not af._neighbor_allowed(current, neighbor)


def test_monochrome1_inversion_is_common_across_entire_series(tmp_path):
    write_slice(tmp_path / "1.dcm", z=0, instance=1, PhotometricInterpretation="MONOCHROME1")
    write_slice(
        tmp_path / "2.dcm",
        z=4,
        instance=2,
        pixels=[[11, 12, 13], [14, 15, 16]],
        PhotometricInterpretation="MONOCHROME1",
        RescaleSlope=2,
        RescaleIntercept=-5,
    )
    images, qc = af._read_series(tmp_path, source(tmp_path, tmp_path))
    assert qc["monochrome1_inverted"]
    assert images[0].image.max() < images[1].image.min()
    assert 0 <= min(image.image.min() for image in images) <= 1


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"ImageOrientationPatient": [0, 1, 0, 0, 0, 1]}, "not_axial"),
        (
            {
                "ImageOrientationPatient": [
                    math.sqrt(0.5),
                    math.sqrt(0.5),
                    0,
                    -math.sqrt(0.5),
                    math.sqrt(0.5),
                    0,
                ]
            },
            "unsupported_inplane_rotation",
        ),
        ({"PixelSpacing": [1, 0]}, "invalid_pixel_spacing"),
        ({"InstanceNumber": 1}, "missing_or_duplicate_instance_number"),
        ({"SeriesInstanceUID": generate_uid()}, "mixed_dicom_identity_or_frame"),
        ({"Modality": "CT"}, "unsupported_dicom_modality_or_multiframe"),
    ],
)
def test_invalid_slice_fails_series(tmp_path, changes, reason):
    write_slice(tmp_path / "1.dcm", z=4, instance=1)
    write_slice(tmp_path / "2.dcm", z=0, instance=2, **changes)
    with pytest.raises(af.AxialDataError, match=reason):
        af._read_series(tmp_path, source(tmp_path, tmp_path))


@pytest.mark.parametrize("missing_frame", [False, True])
def test_unlinked_multiseries_reset_positions_without_comparing_cross_frame_z(
    tmp_path, missing_frame
):
    first, second = tmp_path / "1", tmp_path / "2"
    first.mkdir()
    second.mkdir()
    first_frame = "" if missing_frame else FRAME_UID
    write_slice(first / "1.dcm", z=0, instance=1, FrameOfReferenceUID=first_frame)
    write_slice(first / "2.dcm", z=4, instance=2, FrameOfReferenceUID=first_frame)
    other_frame, second_uid = ("" if missing_frame else generate_uid()), generate_uid()
    write_slice(
        second / "1.dcm",
        z=100,
        instance=1,
        FrameOfReferenceUID=other_frame,
        SeriesInstanceUID=second_uid,
    )
    write_slice(
        second / "2.dcm",
        z=104,
        instance=2,
        FrameOfReferenceUID=other_frame,
        SeriesInstanceUID=second_uid,
    )
    images, positions, _, qc = af._study_inputs(
        tmp_path, {"series": [source(second, tmp_path, "2"), source(first, tmp_path, "1")]}
    )
    assert [image.series_id for image in images] == ["1", "1", "2", "2"]
    assert [image.center_lps[2] for image in images] == [4, 0, 104, 100]
    np.testing.assert_array_equal(positions, [[0, 0], [1, 1], [0, 0], [1, 1]])
    assert not qc["all_series_share_frame_of_reference"]
    assert qc["frame_group_count"] == 2 and qc["series_frame_groups"] == {"1": 0, "2": 1}


def test_shared_frame_has_one_physical_position_scale(tmp_path):
    first, second = tmp_path / "1", tmp_path / "2"
    first.mkdir()
    second.mkdir()
    write_slice(first / "1.dcm", z=0, instance=1)
    write_slice(second / "1.dcm", z=100, instance=1, SeriesInstanceUID=generate_uid())
    images, positions, _, qc = af._study_inputs(
        tmp_path, {"series": [source(first, tmp_path, "1"), source(second, tmp_path, "2")]}
    )
    assert [image.series_id for image in images] == ["2", "1"]
    np.testing.assert_array_equal(positions, [[0, 0.5], [1, 0.5]])
    assert qc["all_series_share_frame_of_reference"] and qc["frame_group_count"] == 1


def test_cross_series_anonymized_sop_reuse_keeps_distinct_slices_and_audits_count(tmp_path):
    first, second = tmp_path / "1", tmp_path / "2"
    first.mkdir()
    second.mkdir()
    sop, other_series = generate_uid(), generate_uid()
    write_slice(first / "1.dcm", z=0, instance=1, SOPInstanceUID=sop)
    write_slice(
        second / "1.dcm", z=100, instance=1, SOPInstanceUID=sop, SeriesInstanceUID=other_series
    )
    images, positions, _, qc = af._study_inputs(
        tmp_path, {"series": [source(first, tmp_path, "1"), source(second, tmp_path, "2")]}
    )
    assert len(images) == 2 and {image.series_id for image in images} == {"1", "2"}
    assert positions.shape == (2, 2)
    assert qc["cross_series_reused_sop_uid_count"] == 1
    assert qc["cross_series_reused_sop_extra_slice_count"] == 1


def test_within_series_duplicate_sop_is_still_rejected(tmp_path):
    sop = generate_uid()
    write_slice(tmp_path / "1.dcm", z=0, instance=1, SOPInstanceUID=sop)
    write_slice(tmp_path / "2.dcm", z=4, instance=2, SOPInstanceUID=sop)
    with pytest.raises(af.AxialDataError, match="duplicate_sop_instance_uid"):
        af._read_series(tmp_path, source(tmp_path, tmp_path))


def test_extract_partial_then_resume_exact_arrays_and_receipts(monkeypatch, tmp_path):
    data, out, weights = setup_extractor(
        monkeypatch, tmp_path, [row(1, 10), row(2, 20, "validation")]
    )
    partial = af.extract_axial_features(data, "manifest", out, weights, device="cpu", limit=1)
    assert partial["n_successful_studies"] == 1 and not partial["extraction_complete"]
    first_sha = af._sha(out / "records/1.npz")
    summary = af.extract_axial_features(data, "manifest", out, weights, device="cpu")
    assert summary["extraction_complete"] and summary["n_successful_studies"] == 2
    assert first_sha == af._sha(out / "records/1.npz")
    metadata = json.loads((out / "metadata.json").read_text())
    assert metadata["scope"] == "train_validation"
    assert metadata["config"]["allowed_splits"] == ["train", "validation"]
    for name, digest in metadata["outputs_sha256"].items():
        assert af._sha(out / name) == digest
    with np.load(out / "records/1.npz", allow_pickle=False) as arrays:
        assert arrays.files == ["features", "positions", "instance", "series_id"]
        assert arrays["features"].shape == (2, 512) and arrays["features"].dtype == np.float32
        assert arrays["positions"].shape == (2, 2) and arrays["positions"].dtype == np.float32
        np.testing.assert_array_equal(arrays["instance"], [80, 2])
        np.testing.assert_array_equal(arrays["positions"], [[0, 0], [1, 1]])
        assert arrays["instance"].dtype == np.int64 and arrays["series_id"].dtype.kind == "U"
    record = json.loads((out / "records/1.json").read_text())
    assert record["source"]["series"][0]["files"][0]["sha256"]
    assert "PRIVATE" not in json.dumps(record)
    assert not summary["labels_or_coordinates_read"] and not summary["sagittal_guidance_used"]


def test_verified_axial_manifest_allows_missing_modality_with_explicit_qc(monkeypatch, tmp_path):
    from pydicom import dcmread

    data, out, weights = setup_extractor(monkeypatch, tmp_path, [row(1, 10)])
    path = data / "train_images/1/10/1.dcm"
    ds = dcmread(path)
    del ds.Modality
    ds.save_as(path, enforce_file_format=True)
    result = af.extract_axial_features(data, "manifest", out, weights, device="cpu")
    assert result["extraction_complete"] and result["n_successful_studies"] == 1
    record = json.loads((out / "records/1.json").read_text())
    assert record["qc"]["series"][0]["missing_Modality_slice_count"] == 1
    metadata = json.loads((out / "metadata.json").read_text())
    assert (
        "verified official RSNA Axial T2 manifest"
        in metadata["config"]["preprocessing"]["missing_modality"]
    )


def test_incomplete_download_is_pending_and_retried(monkeypatch, tmp_path):
    data, out, weights = setup_extractor(monkeypatch, tmp_path, [row(1, 10)])
    directory = data / "train_images/1/10"
    (directory / af.rsna_bulk.COMPLETE_MARKER).unlink()
    pending = af.extract_axial_features(data, "manifest", out, weights, device="cpu")
    assert not pending["extraction_complete"] and pending["n_successful_studies"] == 0
    assert json.loads((out / "records/1.json").read_text())["status"] == "pending"
    marker(directory)
    completed = af.extract_axial_features(data, "manifest", out, weights, device="cpu")
    assert completed["extraction_complete"] and completed["n_successful_studies"] == 1


def test_one_corrupt_series_fails_entire_study_no_partial_bag(monkeypatch, tmp_path):
    data, out, weights = setup_extractor(monkeypatch, tmp_path, [row(1, 10), row(1, 20)])
    (data / "train_images/1/20/1.dcm").write_bytes(b"broken DICOM")
    summary = af.extract_axial_features(data, "manifest", out, weights, device="cpu")
    assert summary["extraction_complete"] and summary["n_successful_studies"] == 0
    assert summary["failed_by_reason"] == {"dicom_decode_or_header_failed": 1}
    assert not (out / "records/1.npz").exists()


def test_cached_source_and_feature_tampering_are_rejected(monkeypatch, tmp_path):
    data, out, weights = setup_extractor(monkeypatch, tmp_path, [row(1, 10)])
    af.extract_axial_features(data, "manifest", out, weights, device="cpu")
    bag = out / "records/1.npz"
    original = bag.read_bytes()
    bag.write_bytes(b"not the saved features")
    with pytest.raises(ValueError, match="checksum changed"):
        af.extract_axial_features(data, "manifest", out, weights, device="cpu")
    bag.write_bytes(original)
    path = data / "train_images/1/10/1.dcm"
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="Source axial DICOM bytes changed"):
        af.extract_axial_features(data, "manifest", out, weights, device="cpu")


def test_model_device_runtime_failure_is_not_mislabeled_bad_mri(monkeypatch, tmp_path):
    data, out, weights = setup_extractor(monkeypatch, tmp_path, [row(1, 10)])

    class BrokenEncoder(torch.nn.Module):
        def forward(self, inputs):
            raise RuntimeError("simulated device failure")

    monkeypatch.setattr(af, "_load_encoder", lambda *_: BrokenEncoder())
    with pytest.raises(RuntimeError, match="simulated device failure"):
        af.extract_axial_features(data, "manifest", out, weights, device="cpu")
    assert not (out / "records/1.json").exists()
    assert not json.loads((out / "metadata.json").read_text())["extraction_complete"]


def test_test_split_rejected_before_weights_or_model(monkeypatch, tmp_path):
    from ai_spine_med import axial_data

    monkeypatch.setattr(
        axial_data, "validate_axial_manifest", lambda *_: ([row(1, 10, "test")], {})
    )
    with pytest.raises(ValueError, match="only original train and validation"):
        af.extract_axial_features(tmp_path, tmp_path, tmp_path / "out", tmp_path / "missing")


def test_wrong_pretrained_bytes_are_rejected(tmp_path):
    path = tmp_path / "weights.pth"
    path.write_bytes(b"wrong model")
    assert hashlib.sha256(path.read_bytes()).hexdigest() != af.WEIGHTS_SHA256
    with pytest.raises(ValueError, match="official IMAGENET1K_V1"):
        af._weight_signature(path)
