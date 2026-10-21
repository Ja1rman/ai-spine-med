"""Sealed TEST access uses synthetic metadata/archive bytes, never real images."""

import csv
import hashlib
import io
import json
import zipfile
from types import SimpleNamespace

import pytest
from test_rsna_bulk import MemoryArchive

from ai_spine_med import axial_test as a
from ai_spine_med.rsna_bulk import COMPLETE_MARKER, ArchiveDownloadError, is_complete_series


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_csv(path, fields, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def experiment(tmp_path):
    data, fixed, manifest = (tmp_path / name for name in ("data", "fixed", "test-manifest"))
    data.mkdir()
    fixed.mkdir()
    studies = [
        {"study_id": "1", "split": "train"},
        {"study_id": "2", "split": "validation"},
        *({"study_id": str(3000 + i), "split": "test"} for i in range(295)),
    ]
    write_csv(fixed / "studies.csv", ["study_id", "split"], studies)
    descriptions = []
    for row in studies:
        study = row["study_id"]
        for number, description in [(1, "Axial T2"), (2, "Sagittal T2/STIR")]:
            descriptions.append(
                dict(study_id=study, series_id=f"{study}{number}", series_description=description)
            )
    for i in range(51):
        study = str(3000 + i)
        descriptions.append(
            dict(study_id=study, series_id=f"{study}3", series_description="Axial T2")
        )
    write_csv(
        data / "train_series_descriptions.csv",
        ["study_id", "series_id", "series_description"],
        descriptions,
    )
    write_json(
        fixed / "summary.json",
        {
            "output_sha256": {"studies.csv": sha(fixed / "studies.csv")},
            "input_sha256": {
                "train_series_descriptions.csv": sha(data / "train_series_descriptions.csv")
            },
        },
    )
    # Grade/coordinate files do not exist; no code in this adapter needs them.
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for row in descriptions:
            for number in (1, 2):
                archive.writestr(
                    f"train_images/{row['study_id']}/{row['series_id']}/{number}.dcm",
                    f"SYNTHETIC-{row['study_id']}-{row['series_id']}-{number}".encode(),
                )
    source = MemoryArchive(payload.getvalue())
    summary = a.prepare_rsna_axial_test_manifest(data, fixed, manifest)
    run_dir = tmp_path / "completed-training"
    protocol = tmp_path / "protocol.json"
    write_json(protocol, {"seeds": list(a.SEEDS), "fixed_before_test": True})
    runs = []
    for seed in a.SEEDS:
        run = {"seed": seed, "best_epoch": seed % 5}
        for key in ("checkpoint", "before_checkpoint"):
            path = run_dir / "runs" / f"seed-{seed}" / f"{key}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"SYNTHETIC-CHECKPOINT-{key}-{seed}".encode())
            run[key] = str(path.relative_to(run_dir))
            run[key + "_sha256"] = sha(path)
        runs.append(run)
    training_summary = {
        "schema_version": 1,
        "status": "complete",
        "model_family": "axial_transfer_v1",
        "protocol_sha256": sha(protocol),
        "index_sha256": "a" * 64,
        "seeds": list(a.SEEDS),
        "runs": runs,
    }
    write_json(run_dir / "summary.json", training_summary)
    selection = {
        "schema_version": 1,
        "model_family": "axial_transfer_v1",
        "frozen": True,
        "protocol_sha256": sha(protocol),
        "training_index_sha256": "a" * 64,
        "training_summary_sha256": sha(run_dir / "summary.json"),
        "fit_keys": [["aisslab", "patient-one"], ["rsna", "1"]],
        "validation_keys": [["aisslab", "patient-two"]],
        "runs": runs,
    }
    write_json(run_dir / "selection.json", selection)
    seal = {
        "schema_version": 1,
        "scope": a.SCOPE,
        "frozen": True,
        "test_manifest_sha256": summary["manifest_sha256"],
        "protocol": {"path": str(protocol), "sha256": sha(protocol)},
        "training_selection": {
            "path": str(run_dir / "selection.json"),
            "sha256": sha(run_dir / "selection.json"),
        },
    }
    seal_path = tmp_path / "test-seal.json"
    write_json(seal_path, seal)
    return SimpleNamespace(
        data=data,
        fixed=fixed,
        manifest=manifest,
        source=source,
        seal_path=seal_path,
        seal=seal,
        summary=summary,
        run_dir=run_dir,
        selection=selection,
        training_summary=training_summary,
        protocol=protocol,
    )


def refresh_training_receipts(e):
    write_json(e.run_dir / "summary.json", e.training_summary)
    e.selection["training_summary_sha256"] = sha(e.run_dir / "summary.json")
    write_json(e.run_dir / "selection.json", e.selection)
    e.seal["training_selection"]["sha256"] = sha(e.run_dir / "selection.json")
    write_json(e.seal_path, e.seal)


def test_manifest_is_exact_fixed_test_only_metadata_and_idempotent(experiment):
    e = experiment
    before = {path.name: path.read_bytes() for path in e.fixed.iterdir()}
    rows, summary = a.validate_rsna_axial_test_manifest(e.data, e.manifest / "series.csv")
    assert len(rows) == 346 and len({row["study_id"] for row in rows}) == 295
    assert {row["split"] for row in rows} == {"test"}
    assert {row["series_description"] for row in rows} == {"Axial T2"}
    assert not {"1", "2"} & {row["study_id"] for row in rows}
    assert summary["preparation_reads_pixels"] is False
    assert summary["selection_uses_labels_or_reference_coordinates"] is False
    assert a.prepare_rsna_axial_test_manifest(e.data, e.fixed, e.manifest) == summary
    assert {path.name: path.read_bytes() for path in e.fixed.iterdir()} == before
    assert not (e.data / "train_images").exists()
    with pytest.raises(ArchiveDownloadError):
        a.prepare_rsna_axial_test_manifest(e.data, e.fixed, e.fixed)


@pytest.mark.parametrize("changed", ["study_split", "descriptions", "manifest"])
def test_changed_selection_rejected_before_remote_access(experiment, monkeypatch, changed):
    e = experiment
    if changed == "study_split":
        path = e.fixed / "studies.csv"
        path.write_text(path.read_text().replace("3000,test", "3000,train"))
    elif changed == "descriptions":
        path = e.data / "train_series_descriptions.csv"
        path.write_text(path.read_text() + "\n")
    else:
        path = e.manifest / "series.csv"
        path.write_text(path.read_text().replace(",test,", ",train,", 1))
    opened = []
    monkeypatch.setattr(a, "_open_official_archive", lambda: opened.append(True) or e.source)
    with pytest.raises(ArchiveDownloadError):
        a.download_rsna_axial_test(e.data, e.manifest, e.seal_path)
    assert not opened and not (e.data / "train_images").exists()


@pytest.mark.parametrize(
    "cause",
    [
        "missing_seal",
        "not_frozen",
        "wrong_scope",
        "wrong_manifest",
        "protocol_changed",
        "selection_changed",
        "checkpoint_changed",
        "before_changed",
        "summary_changed",
        "incomplete_runs",
        "incomplete_summary",
        "duplicate_seed",
        "overlap",
        "old_test_fitting",
    ],
)
def test_seal_or_incomplete_training_blocks_any_archive_access(experiment, monkeypatch, cause):
    e = experiment
    if cause == "missing_seal":
        e.seal_path.unlink()
    elif cause == "not_frozen":
        e.seal["frozen"] = False
        write_json(e.seal_path, e.seal)
    elif cause == "wrong_scope":
        e.seal["scope"] = "train_validation"
        write_json(e.seal_path, e.seal)
    elif cause == "wrong_manifest":
        e.seal["test_manifest_sha256"] = "0" * 64
        write_json(e.seal_path, e.seal)
    elif cause == "protocol_changed":
        e.protocol.write_text("CHANGED")
    elif cause == "selection_changed":
        (e.run_dir / "selection.json").write_text("CHANGED")
    elif cause in {"checkpoint_changed", "before_changed"}:
        key = "checkpoint" if cause == "checkpoint_changed" else "before_checkpoint"
        (e.run_dir / e.selection["runs"][0][key]).write_bytes(b"CHANGED")
    elif cause == "summary_changed":
        (e.run_dir / "summary.json").write_text("CHANGED")
    else:
        if cause == "incomplete_runs":
            e.selection["runs"] = e.selection["runs"][:2]
        elif cause == "incomplete_summary":
            e.training_summary["status"] = "training"
        elif cause == "duplicate_seed":
            e.selection["runs"][1]["seed"] = 42
        elif cause == "overlap":
            e.selection["validation_keys"] = e.selection["fit_keys"]
        else:
            e.selection["fit_keys"].append(["rsna", "3000"])
        refresh_training_receipts(e)
    opened = []
    monkeypatch.setattr(a, "_open_official_archive", lambda: opened.append(True) or e.source)
    with pytest.raises(ArchiveDownloadError):
        a.download_rsna_axial_test(e.data, e.manifest, e.seal_path)
    assert not opened
    assert not (e.data / "train_images").exists()
    assert not (e.data / "_download_cache").exists()


def test_atomic_download_resume_and_exact_source_inventory(experiment, tmp_path):
    e = experiment
    # A pre-existing TRAIN receipt is outside the selected TEST series.
    train = e.data / "train_images/1/11"
    train.mkdir(parents=True)
    (train / COMPLETE_MARKER).write_bytes(b"KEEP ORIGINAL TRAIN RECEIPT")
    result = a.download_rsna_axial_test(e.data, e.manifest, e.seal_path, workers=2, source=e.source)
    assert result["stage"] == "complete" and result["completed_series"] == 346
    assert result["completed_files"] == 692 and result["scope"] == a.SCOPE
    assert (train / COMPLETE_MARKER).read_bytes() == b"KEEP ORIGINAL TRAIN RECEIPT"
    assert not (e.data / "train_images/2").exists()
    inventory_path = tmp_path / "source-inventory.json"
    inventory = a.build_source_inventory(e.data, e.manifest, e.seal_path, inventory_path)
    assert inventory["n_files"] == 692 and inventory["n_series"] == 346
    assert inventory["n_studies"] == 295 and inventory["allowed_splits"] == ["test"]
    assert inventory["selection_sha256"] == sha(e.seal_path)
    for row in inventory["source_series"]:
        directory = e.data / row["series_path"]
        assert is_complete_series(directory)
        marker = json.loads((directory / COMPLETE_MARKER).read_text())
        assert marker["scope"] == a.SCOPE
        assert marker["allowed_splits"] == ["test"]
        assert marker["manifest_sha256"] == e.summary["manifest_sha256"]
        assert row["completion_receipt_sha256"] == sha(directory / COMPLETE_MARKER)
        for item in row["files"]:
            path = directory / item["name"]
            assert item["sha256"] == sha(path) and item["bytes"] == path.stat().st_size
    resumed = a.download_rsna_axial_test(
        e.data, e.manifest, e.seal_path, workers=2, source=e.source
    )
    assert resumed["reused_series"] == 346 and resumed["range_bytes_received"] == 0
    assert a.build_source_inventory(e.data, e.manifest, e.seal_path, inventory_path) == inventory
    assert not (e.data / "_download_cache/axial_v3_download.json").exists()


@pytest.mark.parametrize("cause", ["size", "space", "foreign_receipt", "symlink_escape"])
def test_series_and_resource_guards_precede_pixels(experiment, monkeypatch, tmp_path, cause):
    e = experiment
    opened = []
    monkeypatch.setattr(a, "_open_official_archive", lambda: opened.append(True) or e.source)
    directory = e.data / "train_images/3000/30001"
    if cause == "size":
        monkeypatch.setattr(a, "MAX_UNCOMPRESSED_BYTES", 1)
    elif cause == "space":
        monkeypatch.setattr(a.shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    elif cause == "foreign_receipt":
        directory.mkdir(parents=True)
        write_json(directory / COMPLETE_MARKER, {"scope": "train_validation"})
    else:
        directory.parent.mkdir(parents=True)
        outside = tmp_path / "outside"
        outside.mkdir()
        directory.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ArchiveDownloadError):
        a.download_rsna_axial_test(e.data, e.manifest, e.seal_path)
    assert not list(e.data.glob("train_images/*/*/*.dcm"))
    assert not (e.data / "_download_cache").exists()
    if cause in {"foreign_receipt", "symlink_escape"}:
        assert not opened


def test_source_inventory_refuses_partial_series_without_creating_output(experiment, tmp_path):
    e = experiment
    out = tmp_path / "inventory.json"
    with pytest.raises(ArchiveDownloadError, match="complete"):
        a.build_source_inventory(e.data, e.manifest, e.seal_path, out)
    assert not out.exists()
