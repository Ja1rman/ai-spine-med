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
    _selected_series,
    download_test,
    download_train_validation,
    is_complete_series,
    validate_test_selection,
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


def write_selection(manifests, directory):
    """A synthetic completed six-run seal; contains no patient images."""
    from ai_spine_med import rsna_bulk

    directory.mkdir(exist_ok=True)
    labels = manifests / "labels.csv"
    if not labels.exists():
        labels.write_text("study_id,split,level,label\n3,test,L4-L5,1\n")
    summary = json.loads((manifests / "summary.json").read_text())
    summary["output_sha256"]["labels.csv"] = hashlib.sha256(labels.read_bytes()).hexdigest()
    (manifests / "summary.json").write_text(json.dumps(summary))
    protocol = directory / "protocol.json"
    protocol.write_text('{"locked":true}\n')
    lock = json.loads(rsna_bulk.Path(rsna_bulk.__file__).with_name("assets.lock.json").read_text())
    checkpoint_groups = {}
    for variant in ("frozen_head", "fine_tuned"):
        checkpoint_groups[variant] = []
        for seed in (42, 2024, 3407):
            path = directory / f"{variant}-{seed}.pt"
            path.write_bytes(f"synthetic-{variant}-{seed}".encode())
            checkpoint_groups[variant].append(
                {
                    "seed": seed,
                    "path": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    selection = {
        "schema_version": 1,
        "scope": "heldout_test",
        "frozen": True,
        "manifests_sha256": {
            name: summary["output_sha256"][name]
            for name in ("studies.csv", "series.csv", "labels.csv")
        },
        "protocol": {
            "path": protocol.name,
            "sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
        },
        "original_weights_sha256": {name: item["sha256"] for name, item in lock["weights"].items()},
        "checkpoints": checkpoint_groups,
    }
    path = directory / "selection.json"
    path.write_text(json.dumps(selection, indent=2) + "\n")
    return path


def test_heldout_gate_precedes_network_and_root_creation(tmp_path, monkeypatch):
    from ai_spine_med import rsna_bulk

    manifests = tmp_path / "manifests"
    make_manifests(manifests)
    called = []
    monkeypatch.setattr(rsna_bulk, "_open_official_archive", lambda: called.append(True))
    with pytest.raises(ArchiveDownloadError, match="test is sealed"):
        download_test(tmp_path / "raw", manifests, tmp_path / "missing-selection.json")
    assert called == []
    assert not (tmp_path / "raw").exists()


def test_test_download_separate_status_same_manifest_and_existing_train_markers(tmp_path):
    manifests = tmp_path / "manifests"
    make_manifests(manifests)
    selection = write_selection(manifests, tmp_path / "sealed")
    root = tmp_path / "raw"
    source = MemoryArchive(make_archive())
    download_train_validation(root, manifests, source=source)
    before = (root / "train_images/1/10" / COMPLETE_MARKER).read_bytes()
    result = download_test(root, manifests, selection, source=source)
    assert result["completed_series"] == 1
    assert result["scope"] == "heldout_test"
    assert result["selection_sha256"] == hashlib.sha256(selection.read_bytes()).hexdigest()
    assert (root / "train_images/1/10" / COMPLETE_MARKER).read_bytes() == before
    marker = json.loads((root / "train_images/3/10" / COMPLETE_MARKER).read_text())
    assert marker["manifest_sha256"] == _selected_series(manifests)[1]
    assert marker["selection_sha256"] == result["selection_sha256"]
    assert marker["scope"] == "heldout_test"
    assert is_complete_series(root / "train_images/3/10")
    assert (root / "_download_cache/test_download.json").is_file()
    assert (
        json.loads((root / "_download_cache/train_validation_download.json").read_text())[
            "total_series"
        ]
        == 2
    )


@pytest.mark.parametrize(
    "changed", ["checkpoint", "protocol", "manifest", "seeds", "original", "incomplete"]
)
def test_changed_or_incomplete_selection_cannot_unlock_test(tmp_path, changed):
    manifests = tmp_path / "manifests"
    make_manifests(manifests)
    path = write_selection(manifests, tmp_path / "sealed")
    selection = json.loads(path.read_text())
    if changed == "checkpoint":
        (path.parent / selection["checkpoints"]["fine_tuned"][0]["path"]).write_bytes(b"changed")
    elif changed == "protocol":
        (path.parent / "protocol.json").write_text("changed")
    elif changed == "manifest":
        (manifests / "labels.csv").write_text("changed")
    elif changed == "seeds":
        selection["checkpoints"]["fine_tuned"][0]["seed"] = 99
    elif changed == "original":
        selection["original_weights_sha256"]["grading/ckpt1.pt"] = "0" * 64
    else:
        selection["checkpoints"]["frozen_head"].pop()
    path.write_text(json.dumps(selection))
    with pytest.raises(ArchiveDownloadError, match="test is sealed"):
        validate_test_selection(manifests, path)
