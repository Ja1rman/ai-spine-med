"""Pinned release import tests use synthetic DICOM headers and tiny ZIPs only."""

import hashlib
import io
import json
import stat
import zipfile

import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from ai_spine_med import aisslab as a


def dicom_bytes(study=None, series=None, instance=1):
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = study or generate_uid()
    ds.SeriesInstanceUID = series or generate_uid()
    ds.FrameOfReferenceUID = generate_uid()
    ds.PatientName = "SYNTHETIC^DO_NOT_SERIALIZE"
    ds.PatientID = "SYNTHETIC_PRIVATE_ID"
    ds.Modality = "MR"
    ds.Rows, ds.Columns = 8, 8
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit, ds.PixelRepresentation = 15, 0
    ds.InstanceNumber = instance
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [0, 0, instance * 4]
    ds.PixelSpacing = [0.7, 0.7]
    ds.PixelData = bytes(range(128))
    stream = io.BytesIO()
    ds.save_as(stream, enforce_file_format=True)
    return stream.getvalue()


def annotation(objects):
    return (
        "<annotation><filename>original.png</filename><path>PRIVATE/PATH</path>"
        + "".join(
            f"<object><name>{name}</name><level>{level}</level></object>" for name, level in objects
        )
        + "</annotation>"
    ).encode()


def release(tmp_path, entries):
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, content in entries:
            z.writestr(name, content)
    return path


def prepare(path, out):
    return a.prepare_aisslab(
        path,
        out,
        expected_size=path.stat().st_size,
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def example_entries():
    study = generate_uid()
    return [
        (r"release\DICOM\1\a.dcm", dicom_bytes(study=study)),
        (r"release\DICOM\1\b.dcm", dicom_bytes(study=study)),
        (
            r"release\Localization_and_Severity_Classification\1\a.xml",
            annotation([("CCS_normal", "L4-L5"), ("LFS_severe_stenosis", "L5-S1")]),
        ),
        (
            r"release\Localization_and_Severity_Classification\1\b.xml",
            annotation([("CCS_severe_stenosis", "L4-L5")]),
        ),
        (r"release\Localization_and_Severity_Classification\1\a.png", b"not-read"),
        (r"release\Localization_and_Severity_Classification\1\b.png", b"not-read"),
        (r"release\Segmentation\1\mask.png", b"not-extracted"),
    ]


def test_explicit_ccs_worst_grade_preserves_references_and_unknown(tmp_path, monkeypatch):
    def forbid_pixels(_self):
        raise AssertionError("Preparation must not decode image pixels")

    monkeypatch.setattr(a.pydicom.dataset.Dataset, "pixel_array", property(forbid_pixels))
    archive = release(tmp_path, example_entries())
    output = tmp_path / "prepared"
    result = prepare(archive, output)
    records = json.loads((output / "records.json").read_text())
    assert len(records) == 1
    row = records[0]
    assert row["labels"] == [2, None]
    assert [ref["grade"] for ref in row["label_references"]] == [0, 2]
    assert {ref["level"] for ref in row["label_references"]} == {"L4-L5"}
    assert len(row["source"]["series"]) == 2
    assert result["annotation_linking"] == {"xml_dicom_stem_matches": 2, "xml_png_fallbacks": 0}
    assert result["patients_with_multiple_series"] == 1
    assert result["image_pixels_decoded"] is False
    assert "Maximum explicit CCS grade" in result["policies"]["aggregation"]
    assert len(list((output / "extracted").rglob("*.dcm"))) == 2
    assert not list((output / "extracted").rglob("*.png"))
    for series in row["source"]["series"]:
        assert series["series_id"].isdigit()
        assert len(series["series_path"].split("/")[2]) == 64
        for file in series["files"]:
            source = output / "extracted" / series["series_path"] / file["name"]
            assert hashlib.sha256(source.read_bytes()).hexdigest() == file["sha256"]
    for file in output.glob("*.json"):
        assert "SYNTHETIC_PRIVATE_ID" not in file.read_text()
        assert "SYNTHETIC^DO_NOT_SERIALIZE" not in file.read_text()
        assert "PRIVATE/PATH" not in file.read_text()
    assert prepare(archive, output) == result
    assert archive.exists()


def test_header_geometry_problem_remains_explicit_without_repair(tmp_path):
    ds = a.pydicom.dcmread(io.BytesIO(dicom_bytes()))
    ds.ImageOrientationPatient = [0, 1, 0, 0, 0, -1]
    stream = io.BytesIO()
    ds.save_as(stream, enforce_file_format=True)
    path = release(tmp_path, [("release/DICOM/1/a.dcm", stream.getvalue())])
    summary = prepare(path, tmp_path / "prepared")
    record = json.loads((tmp_path / "prepared/records.json").read_text())[0]
    assert summary["patients_with_metadata_geometry_issues"] == 1
    assert record["metadata_geometry_pass"] is False
    assert (
        "not_axial_within_supported_obliquity" in record["source"]["series"][0]["geometry_issues"]
    )


def test_missing_ccs_never_becomes_normal_and_png_fallback_is_explicit(tmp_path):
    archive = release(
        tmp_path,
        [
            ("release/DICOM/1/a.dcm", dicom_bytes()),
            (
                "release/Localization_and_Severity_Classification/1/other.xml",
                annotation([("RLS_normal", "L4-L5")]),
            ),
            ("release/Localization_and_Severity_Classification/1/other.png", b"image-fallback"),
        ],
    )
    result = prepare(archive, tmp_path / "prepared")
    row = json.loads((tmp_path / "prepared/records.json").read_text())[0]
    assert row["labels"] == [None, None]
    assert not row["label_references"]
    assert result["annotation_linking"]["xml_png_fallbacks"] == 1
    assert len(list((tmp_path / "prepared/extracted").rglob("*.png"))) == 1


@pytest.mark.parametrize(
    "name,level,error",
    [
        ("CCS_mild", "L4-L5", "unknown_ccs_grade"),
        ("CCS_normal", "", "missing_or_unknown_ccs_level"),
        ("CCS_stenosis", "L4", "missing_or_unknown_ccs_level"),
    ],
)
def test_unknown_ccs_grade_or_level_is_an_error(name, level, error):
    with pytest.raises(a.AisslabError, match=error):
        a._parse_labels(annotation([(name, level)]))


def test_no_target_annotation_is_unknown_and_xml_entities_rejected():
    assert a._parse_labels(annotation([("CCS_normal", "L1-L2")])) == []
    assert a._parse_labels(b"<annotation/>") == []
    with pytest.raises(a.AisslabError, match="unsafe_annotation_xml"):
        a._parse_labels(b'<!DOCTYPE a [<!ENTITY x "bad">]><annotation/>')


@pytest.mark.parametrize(
    "path",
    [
        "../escape.dcm",
        "/tmp/escape.dcm",
        r"C:\escape.dcm",
        r"root\..\escape.dcm",
        r"\\server\share\escape.dcm",
        "root/./file",
        "root//file",
    ],
)
def test_unsafe_zip_paths_rejected_before_extraction(tmp_path, path):
    archive = release(tmp_path, [(path, b"invalid")])
    with pytest.raises(a.AisslabError, match="unsafe_archive_path"):
        prepare(archive, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()
    assert not list(tmp_path.glob(".prepared.prepare-*"))


@pytest.mark.parametrize(
    "paths",
    [
        (r"release\DICOM\1\a.dcm", "release/DICOM/1/a.dcm"),
        ("release/DICOM/1/A.dcm", "release/DICOM/1/a.dcm"),
        ("release/DICOM/1", "release/DICOM/1/a.dcm"),
    ],
)
def test_normalized_and_file_directory_collisions_rejected(tmp_path, paths):
    archive = release(tmp_path, [(name, b"x") for name in paths])
    with pytest.raises(a.AisslabError, match="collision"):
        prepare(archive, tmp_path / "prepared")


def test_symlink_and_expansion_limit_rejected(tmp_path, monkeypatch):
    path = tmp_path / "source.zip"
    info = zipfile.ZipInfo("release/DICOM/1/a.dcm")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(info, "../../elsewhere")
    with pytest.raises(a.AisslabError, match="archive_special_file"):
        prepare(path, tmp_path / "prepared")
    path = release(tmp_path, [("release/DICOM/1/a.dcm", b"0" * 1000)])
    monkeypatch.setattr(a, "MAX_FILE_BYTES", 100)
    with pytest.raises(a.AisslabError, match="archive_file_limit"):
        prepare(path, tmp_path / "prepared")


def test_shared_payload_cannot_cross_patient_partitions(tmp_path):
    study = generate_uid()
    first = dicom_bytes(study=study)
    path = release(tmp_path, [("release/DICOM/1/a.dcm", first), ("release/DICOM/2/a.dcm", first)])
    with pytest.raises(a.AisslabError, match="crosses_patients"):
        prepare(path, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


def test_reused_uid_is_candidate_overlap_without_patient_merge(tmp_path):
    study = generate_uid()
    path = release(
        tmp_path,
        [
            ("release/DICOM/1/a.dcm", dicom_bytes(study=study)),
            ("release/DICOM/2/a.dcm", dicom_bytes(study=study)),
        ],
    )
    summary = prepare(path, tmp_path / "prepared")
    rows = json.loads((tmp_path / "prepared/records.json").read_text())
    assert summary["patients"] == 2
    assert summary["identity_audit"]["patients_with_candidate_uid_overlap"] == 2
    assert rows[0]["source"]["study_uid_sha256"] == rows[1]["source"]["study_uid_sha256"]
    assert rows[0]["candidate_uid_overlap"]["StudyInstanceUID"]
    assert rows[0]["study_id"] != rows[1]["study_id"]


def test_joint_stratification_keeps_all_series_in_fixed_patient_partition():
    records = [
        {
            "study_id": f"aisslab-{i}",
            "labels": ([i % 3, i % 2] if i < 490 else [None, 2]),
            "source": {"series": [{"series_id": str(i * 10 + j)} for j in range(3)]},
        }
        for i in range(500)
    ]
    first = a._patient_splits(records, 2026, (0.70, 0.15, 0.15))
    second = a._patient_splits(list(reversed(records)), 2026, (0.70, 0.15, 0.15))
    assert first == second
    assert {split: list(first.values()).count(split) for split in a.SPLITS} == {
        "train": 350,
        "validation": 75,
        "test": 75,
    }
    assert len(first) == 500
    with pytest.raises(a.AisslabError, match="duplicate_or_unassigned_patient"):
        a._patient_splits(records + [records[0]], 2026, (0.70, 0.15, 0.15))


def test_archive_pin_and_existing_manifest_tampering_are_rejected(tmp_path):
    path = release(tmp_path, example_entries())
    with pytest.raises(a.AisslabError, match="archive_size_or_path_mismatch"):
        a.prepare_aisslab(path, tmp_path / "prepared")
    prepare(path, tmp_path / "prepared")
    records = tmp_path / "prepared/records.json"
    records.write_text("[]")
    with pytest.raises(a.AisslabError, match="existing_manifest_changed"):
        prepare(path, tmp_path / "prepared")
