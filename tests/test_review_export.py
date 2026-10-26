"""The full review must preserve identities, references and real source slices."""

import hashlib
import io
import json
import zipfile

import numpy as np
import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from ai_spine_med import review_export as review


def prediction(study, level, label, output, status="ok"):
    return {
        "study_id": study,
        "series_id": "2",
        "level": level,
        "split": "validation",
        "label": str(label),
        "status": status,
        "reason": "" if status == "ok" else "missing_vertebra:S1",
        "prediction": str(output) if output is not None else "",
        "p_normal_mild": "0.8",
        "p_moderate": "0.1",
        "p_severe": "0.1",
    }


def test_full_cohort_includes_correct_errors_and_failures_with_stable_numbers():
    rows = [
        prediction("3", "L4-L5", 0, 0),
        prediction("1", "L5-S1", 2, 0),
        prediction("4", "L5-S1", 0, None, "not_assessed"),
    ]
    old_case = review.build_cases([rows[1]], {})[0]
    old_case["review_order"] = 1
    result = review.build_cases(rows, {"cases": [old_case]})
    assert len(result) == 3
    assert result[0]["case_id"] == old_case["case_id"]
    assert result[0]["review_order"] == 1
    assert result[0]["selection_flags"]["any_wrong"]
    assert sum(c["correct_control"] for c in result) == 1
    assert sum(c["selection_flags"]["not_assessed"] for c in result) == 1
    assert all(set(c["predictions"]) == {"sagittal"} for c in result)
    failed = next(c for c in result if c["status"] != "ok")
    assert failed["predictions"]["sagittal"]["prediction"] is None
    assert failed["predictions"]["sagittal"]["probabilities"] is None


def test_previous_case_cannot_silently_disappear():
    old = review.build_cases([prediction("1", "L4-L5", 0, 0)], {})
    with pytest.raises(ValueError, match="existing_cases_missing"):
        review.build_cases([prediction("2", "L4-L5", 0, 0)], {"cases": old})


def test_model_layers_use_rounding_at_each_vertebra_and_spatial_source_order():
    # SpineNet rounds medians first: round(5.5)=6, round(7.5)=8, mean=7.
    # Rounding only their mean would instead give round(6.5)=6.
    record = {
        "vertices": [{"label": "L4", "slice_nos": [5, 6]}, {"label": "L5", "slice_nos": [7, 8]}]
    }
    slices = [
        {
            "slice_key": str(i),
            "image_orientation_patient": [0, 1, 0, 0, 0, -1],
            "image_position_patient": [i * 3, 0, 0],
        }
        for i in reversed(range(12))
    ]
    layers = review.model_input_layers(record, {"slices": slices}, "L4-L5")
    assert [x["slice_key"] for x in layers] == [str(i) for i in range(3, 12)]
    assert not any(x["padding"] for x in layers)


def test_model_layers_mark_zero_padding_instead_of_clamping_to_wrong_slice():
    record = {"vertices": [{"label": "L4", "slice_nos": [0]}, {"label": "L5", "slice_nos": [0]}]}
    slices = [
        {
            "slice_key": str(i),
            "image_orientation_patient": [0, 1, 0, 0, 0, -1],
            "image_position_patient": [i * 3, 0, 0],
        }
        for i in range(6)
    ]
    layers = review.model_input_layers(record, {"slices": slices}, "L4-L5")
    assert [x["slice_key"] for x in layers] == [None] * 4 + list("01234")
    assert sum(x["padding"] for x in layers) == 4


def write_dicom(path, slice_number, series_uid, study_uid):
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = MRImageStorage, meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID, ds.StudyInstanceUID = series_uid, study_uid
    ds.Modality = "MR"
    ds.Rows = ds.Columns = 16
    ds.InstanceNumber = slice_number
    ds.ImageOrientationPatient = [0, 1, 0, 0, 0, -1]
    ds.ImagePositionPatient = [slice_number * 3, 0, 0]
    ds.PixelSpacing = [1, 1]
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = (np.arange(256, dtype=np.uint16).reshape(16, 16) + slice_number).tobytes()
    ds.save_as(path, enforce_file_format=True)


def test_export_preserves_original_dicom_bytes_and_can_reuse_assets(tmp_path):
    root = tmp_path / "repo"
    directory = root / "data/rsna/train_images/1/2"
    directory.mkdir(parents=True)
    series_uid, study_uid = generate_uid(), generate_uid()
    for i in range(1, 3):
        write_dicom(directory / f"{i}.dcm", i, series_uid, study_uid)
    marker = {
        "complete": True,
        "expected_files": 2,
        "files": ["1.dcm", "2.dcm"],
        "metadata_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "source_identity_sha256": "c" * 64,
    }
    (directory / "_series_complete.json").write_text(json.dumps(marker))
    series = {
        "series_id": "2",
        "series_description": "Sagittal T2/STIR",
        "relative_directory": "data/rsna/train_images/1/2",
        "files": [],
    }
    patient = {"study_id": "1", "patient_id": "patient-opaque"}
    first = tmp_path / "first"
    first_record = review.render_series(
        patient, series, source_root=root, output_root=first, selection_sha="d" * 64
    )
    assert first_record["series"]["n_slices"] == 2
    assert first_record["series"]["n_preview_slices"] == 2
    with zipfile.ZipFile(first / "site" / first_record["series"]["zip_url"]) as archive:
        assert all(i.compress_type == zipfile.ZIP_DEFLATED for i in archive.infolist())
        for source, item in zip(
            first_record["source_files"], first_record["series"]["slices"], strict=True
        ):
            content = archive.read(item["slice_key"] + ".dcm")
            assert content == (directory / source["name"]).read_bytes()
            assert hashlib.sha256(content).hexdigest() == item["sha256"]
            assert pydicom.dcmread(io.BytesIO(content)).InstanceNumber in (1, 2)
    second = tmp_path / "second"
    reused = review.render_series(
        patient,
        series,
        source_root=root,
        output_root=second,
        selection_sha="e" * 64,
        previous_root=first,
    )
    assert reused["output_sha256"] == first_record["output_sha256"]
    assert reused["selection_file_sha256"] == "e" * 64
    for name, sha in reused["output_sha256"].items():
        assert review.sha256_file(second / "site" / name) == sha
