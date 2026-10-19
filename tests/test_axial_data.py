"""Axial selection and bulk integrity checks use synthetic archive bytes only."""

import csv
import hashlib
import io
import json
import zipfile
from types import SimpleNamespace

import pytest
from test_rsna_bulk import MemoryArchive

from ai_spine_med import axial_data as a
from ai_spine_med.rsna_bulk import COMPLETE_MARKER, ArchiveDownloadError, is_complete_series


def write_csv(path, fields, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def experiment(tmp_path):
    data, fixed, output = (tmp_path / name for name in ("data", "fixed", "axial"))
    data.mkdir()
    fixed.mkdir()
    studies = [
        {"study_id": str(i), "split": split}
        for i, split in ((1, "train"), (2, "validation"), (3, "test"))
    ]
    write_csv(fixed / "studies.csv", ["study_id", "split"], studies)
    descriptions = [
        {"study_id": str(study), "series_id": str(series), "series_description": sequence}
        for study, series, sequence in (
            (1, 10, "Sagittal T2/STIR"),
            (1, 11, "Axial T2"),
            (1, 12, "Axial T2"),
            (2, 20, "Sagittal T2/STIR"),
            (2, 21, "Axial T2"),
            (3, 31, "Axial T2"),
        )
    ]
    write_csv(
        data / "train_series_descriptions.csv",
        ["study_id", "series_id", "series_description"],
        descriptions,
    )
    (fixed / "summary.json").write_text(
        json.dumps(
            {
                "output_sha256": {"studies.csv": digest(fixed / "studies.csv")},
                "input_sha256": {
                    "train_series_descriptions.csv": digest(data / "train_series_descriptions.csv")
                },
            }
        )
    )
    # No labels.csv, train.csv or label_coordinates file exists in this fixture.
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for row in descriptions:
            for number in (1, 2):
                archive.writestr(
                    f"train_images/{row['study_id']}/{row['series_id']}/{number}.dcm",
                    f"SYNTHETIC-{row['study_id']}-{row['series_id']}-{number}".encode() * 10,
                )
    return data, fixed, output, MemoryArchive(blob.getvalue())


def test_all_axial_series_follow_fixed_groups_without_grades_or_coordinates(experiment):
    data, fixed, output, _source = experiment
    before = {p.name: p.read_bytes() for p in fixed.iterdir()}
    summary = a.prepare_axial_manifest(data, fixed, output)
    assert summary["n_series"] == 3
    assert summary["series_by_split"] == {"train": 2, "validation": 1}
    assert summary["n_studies_without_axial"] == 0
    rows, validated = a.validate_axial_manifest(data, output / "series.csv")
    assert {row["series_id"] for row in rows} == {"11", "12", "21"}
    assert {row["split"] for row in rows} == {"train", "validation"}
    assert validated == summary
    assert summary["selection_uses_reference_coordinates"] is False
    assert summary["test_images_accessed"] is False
    assert a.prepare_axial_manifest(data, fixed, output) == summary
    assert {p.name: p.read_bytes() for p in fixed.iterdir()} == before
    with pytest.raises(ArchiveDownloadError, match="different contents"):
        a.prepare_axial_manifest(data, fixed, fixed)


@pytest.mark.parametrize("changed", ["fixed_studies", "official_series", "axial_output"])
def test_changed_metadata_or_selection_rejected_before_remote_access(
    experiment, monkeypatch, changed
):
    data, fixed, output, source = experiment
    a.prepare_axial_manifest(data, fixed, output)
    if changed == "fixed_studies":
        path = fixed / "studies.csv"
    elif changed == "official_series":
        path = data / "train_series_descriptions.csv"
    else:
        path = output / "series.csv"
    path.write_text(
        path.read_text().replace("validation", "test")
        if changed != "official_series"
        else path.read_text() + "\n"
    )
    called = []
    monkeypatch.setattr(a, "_open_official_archive", lambda: called.append(True) or source)
    with pytest.raises(ArchiveDownloadError, match="changed|checksum"):
        a.download_axial(data, output)
    assert called == []
    assert not (data / "train_images").exists()


def test_download_atomic_axial_receipts_do_not_touch_sagittal_or_test(experiment, monkeypatch):
    data, fixed, output, source = experiment
    manifest = a.prepare_axial_manifest(data, fixed, output)
    sagittal = data / "train_images/1/10"
    sagittal.mkdir(parents=True)
    old_receipt = sagittal / COMPLETE_MARKER
    old_receipt.write_bytes(b"EXISTING SAGITTAL RECEIPT MUST STAY UNCHANGED")
    monkeypatch.setattr(a, "_open_official_archive", lambda: source)
    result = a.download_axial(data, output, workers=2)
    assert result["stage"] == "complete"
    assert result["completed_series"] == 3 and result["completed_files"] == 6
    assert result["modality"] == "axial_v3"
    assert old_receipt.read_bytes() == b"EXISTING SAGITTAL RECEIPT MUST STAY UNCHANGED"
    assert not (data / "train_images/3").exists()
    for study, series in ((1, 11), (1, 12), (2, 21)):
        directory = data / f"train_images/{study}/{series}"
        assert is_complete_series(directory)
        marker = json.loads((directory / COMPLETE_MARKER).read_text())
        assert marker["manifest_sha256"] == manifest["manifest_sha256"]
        assert marker["modality"] == "axial_v3"
        assert marker["allowed_splits"] == ["train", "validation"]
    assert (data / "_download_cache/axial_v3_download.json").is_file()
    assert not (data / "_download_cache/train_validation_download.json").exists()
    resumed = a.download_axial(data, output, workers=2)
    assert resumed["reused_series"] == 3
    assert resumed["range_bytes_received"] == 0


@pytest.mark.parametrize("cause", ["size", "space", "foreign_receipt", "symlink_escape"])
def test_size_space_and_receipt_guards_precede_pixel_download(
    experiment, monkeypatch, cause, tmp_path
):
    data, fixed, output, source = experiment
    a.prepare_axial_manifest(data, fixed, output)
    monkeypatch.setattr(a, "_open_official_archive", lambda: source)
    if cause == "size":
        monkeypatch.setattr(a, "MAX_UNCOMPRESSED_BYTES", 1)
    elif cause == "space":
        monkeypatch.setattr(a.shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    elif cause == "foreign_receipt":
        path = data / "train_images/1/11"
        path.mkdir(parents=True)
        (path / COMPLETE_MARKER).write_text(json.dumps({"modality": "another_pipeline"}))
    else:
        (data / "train_images/1").mkdir(parents=True)
        outside = tmp_path / "outside"
        outside.mkdir()
        (data / "train_images/1/11").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ArchiveDownloadError):
        a.download_axial(data, output)
    assert not (data / "_download_cache/axial_v3_download.json").exists()
    assert not list(data.glob("train_images/*/*/*.dcm"))
