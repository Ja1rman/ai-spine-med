"""Range extraction integrity checks use synthetic ZIPs, never patient images."""

import csv
import hashlib
import io
import json
import zipfile

import pytest

from ai_spine_med.rsna_bulk import (
    COMPLETE_MARKER,
    ArchiveDownloadError,
    _archive_plan,
    download_train_validation,
    is_complete_series,
)


class MemoryArchive:
    def __init__(self, contents):
        self.contents = contents
        self.size = len(contents)
        self.identity = {"size": self.size, "etag": "synthetic-only"}

    def read_range(self, start, end):
        return self.contents[start : end + 1]


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_manifests(directory):
    directory.mkdir()
    studies = [
        {"study_id": str(index), "split": split}
        for index, split in [(1, "train"), (2, "validation"), (3, "test")]
    ]
    series = [
        {
            **row,
            "series_id": "10",
            "series_description": "Sagittal T2/STIR",
            "series_path": f"train_images/{row['study_id']}/10",
        }
        for row in studies
    ]
    write_csv(directory / "studies.csv", studies)
    write_csv(directory / "series.csv", series)
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "output_sha256": {
                    name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                    for name in ("studies.csv", "series.csv")
                }
            }
        )
    )


def make_archive(*, malicious=False, compression=zipfile.ZIP_DEFLATED):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=compression) as archive:
        for study in (1, 2, 3):
            for instance in (1, 2):
                archive.writestr(
                    f"train_images/{study}/10/{instance}.dcm",
                    f"SYNTHETIC:{study}:{instance}".encode() * 20,
                )
        if malicious:
            archive.writestr("../outside.dcm", b"synthetic-unsafe-path")
    return output.getvalue()


@pytest.mark.parametrize("compression", [zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED])
def test_selected_splits_crc_markers_and_resume(tmp_path, compression):
    manifests = tmp_path / "manifests"
    make_manifests(manifests)
    source = MemoryArchive(make_archive(compression=compression))
    root = tmp_path / "raw"
    report = download_train_validation(root, manifests, workers=2, source=source)
    assert report["stage"] == "complete"
    assert report["completed_series"] == 2
    assert report["completed_files"] == 4
    assert not (root / "train_images/3").exists()
    for study in (1, 2):
        directory = root / f"train_images/{study}/10"
        assert is_complete_series(directory)
        marker = json.loads((directory / COMPLETE_MARKER).read_text())
        assert marker["expected_files"] == 2
        assert len(marker["metadata_sha256"]) == 64
        assert not list(directory.glob("*.part"))
        assert (directory / "1.dcm").read_bytes() == f"SYNTHETIC:{study}:1".encode() * 20
    repeated = download_train_validation(root, manifests, workers=2, source=source)
    assert repeated["reused_series"] == 2
    assert repeated["range_bytes_received"] == 0
    assert (
        json.loads((root / "_download_cache/train_validation_download.json").read_text())["stage"]
        == "complete"
    )


def test_existing_unmarked_sample_is_crc_checked_and_reused(tmp_path):
    manifests = tmp_path / "manifests"
    make_manifests(manifests)
    root = tmp_path / "raw"
    sample = root / "train_images/1/10"
    sample.mkdir(parents=True)
    for index in (1, 2):
        (sample / f"{index}.dcm").write_bytes(f"SYNTHETIC:1:{index}".encode() * 20)
    assert not is_complete_series(sample)
    report = download_train_validation(root, manifests, source=MemoryArchive(make_archive()))
    assert report["reused_series"] == 1
    assert is_complete_series(sample)


def test_crc_failure_never_publishes_partial_series(tmp_path):
    manifests = tmp_path / "manifests"
    make_manifests(manifests)
    contents = bytearray(make_archive())
    source = MemoryArchive(contents)
    plan = _archive_plan(source, {"train_images/1/10", "train_images/2/10"})[0]
    member = plan.members[0]
    contents[member.header_offset + 30 + len(member.filename.encode()) + 2] ^= 0xFF
    root = tmp_path / "raw"
    with pytest.raises(ArchiveDownloadError, match="CRC"):
        download_train_validation(root, manifests, workers=1, source=MemoryArchive(contents))
    assert not (root / "train_images/1/10").exists()
    assert not list(root.glob("train_images/*/.bulk-1-*"))
    assert (
        json.loads((root / "_download_cache/train_validation_download.json").read_text())["stage"]
        == "failed"
    )


def test_missing_file_invalidates_ready_marker_and_retry_repairs_it(tmp_path):
    manifests = tmp_path / "manifests"
    make_manifests(manifests)
    root = tmp_path / "raw"
    source = MemoryArchive(make_archive())
    download_train_validation(root, manifests, source=source)
    directory = root / "train_images/1/10"
    (directory / "1.dcm").unlink()
    assert not is_complete_series(directory)
    report = download_train_validation(root, manifests, source=source)
    assert report["reused_series"] == 1
    assert is_complete_series(directory)
    assert list(directory.parent.glob(".bulk-previous-10-*"))


def test_same_size_corrupted_local_image_is_not_reused(tmp_path):
    manifests = tmp_path / "manifests"
    make_manifests(manifests)
    root = tmp_path / "raw"
    source = MemoryArchive(make_archive())
    download_train_validation(root, manifests, source=source)
    image = root / "train_images/1/10/1.dcm"
    expected = image.read_bytes()
    altered = bytearray(expected)
    altered[0] ^= 0xFF
    image.write_bytes(altered)
    result = download_train_validation(root, manifests, source=source)
    assert result["reused_series"] == 1
    assert image.read_bytes() == expected


def test_manifest_mutation_is_rejected_before_any_download(tmp_path):
    manifests = tmp_path / "manifests"
    make_manifests(manifests)
    file = manifests / "series.csv"
    file.write_text(file.read_text().replace("validation", "train"))
    with pytest.raises(ArchiveDownloadError, match="checksum"):
        download_train_validation(tmp_path / "raw", manifests, source=MemoryArchive(make_archive()))
    assert not (tmp_path / "raw").exists()


def test_zip_traversal_member_is_rejected(tmp_path):
    manifests = tmp_path / "manifests"
    make_manifests(manifests)
    root = tmp_path / "raw"
    with pytest.raises(ArchiveDownloadError, match="Unsafe member path"):
        download_train_validation(
            root, manifests, source=MemoryArchive(make_archive(malicious=True))
        )
    assert not (root / "train_images").exists()
    assert not (tmp_path / "outside.dcm").exists()
    assert (
        json.loads((root / "_download_cache/train_validation_download.json").read_text())["stage"]
        == "failed"
    )


def test_bad_ready_marker_is_not_trusted(tmp_path):
    directory = tmp_path / "series"
    directory.mkdir()
    (directory / "1.dcm").write_bytes(b"synthetic")
    (directory / COMPLETE_MARKER).write_text(
        json.dumps({"complete": True, "expected_files": True, "files": ["1.dcm"]})
    )
    assert not is_complete_series(directory)
