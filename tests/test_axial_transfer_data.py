"""Lightweight transfer source safety checks; no MRI decoding or real encoder."""

import copy
import csv
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ai_spine_med import axial_transfer_cache as cache
from ai_spine_med import axial_transfer_data as data
from ai_spine_med.axial_features import _embed


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def source_row(root, dataset, study, split, labels=(0, 2)):
    sid = str(10000 + int(study))
    directory = root / "images" / study / sid
    directory.mkdir(parents=True, exist_ok=True)
    dicom = directory / "1.dcm"
    dicom.write_bytes(f"SYNTHETIC-{dataset}-{study}".encode())
    return {
        "dataset": dataset,
        "study_id": study,
        "split": split,
        "labels": list(labels),
        "source_root": str(root),
        "source": {
            "study_uid_sha256": hashlib.sha256(f"{dataset}-{study}".encode()).hexdigest(),
            "series": [
                {
                    "series_id": sid,
                    "series_path": str(directory.relative_to(root)),
                    "files": [
                        {"name": dicom.name, "bytes": dicom.stat().st_size, "sha256": sha(dicom)}
                    ],
                }
            ],
        },
    }


@pytest.mark.parametrize(
    "location", ["series_parent", "absolute_series", "file_parent", "absolute_file", "symlink"]
)
def test_unsafe_source_path_is_rejected_before_decode(tmp_path, location):
    root = tmp_path / "source"
    row = source_row(root, "aisslab", "100", "train")
    series = row["source"]["series"][0]
    if location == "series_parent":
        series["series_path"] = "../outside"
    elif location == "absolute_series":
        series["series_path"] = str(root.resolve())
    elif location == "file_parent":
        series["files"][0]["name"] = "../other.dcm"
    elif location == "absolute_file":
        series["files"][0]["name"] = str((root / "1.dcm").resolve())
    else:
        path = root / series["series_path"] / "1.dcm"
        outside = tmp_path / "outside.dcm"
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
    with pytest.raises(ValueError):
        cache.validate_sources(row)


@pytest.mark.parametrize("count", [1, 95, 96, 97, 127, 192, 219, 500, 1001])
def test_capped_slice_indices_exactly_match_original_v3_policy(count):
    expected = np.arange(count) if count <= 96 else np.linspace(0, count - 1, 96, dtype=np.int64)
    actual = cache.select_slice_indices(count)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.int64
    assert len(actual) == min(count, 96)
    assert actual[0] == 0 and actual[-1] == count - 1


def test_new_cache_tensor_preprocessing_matches_old_encoder_input_exactly():
    images = [
        SimpleNamespace(
            series_id="1",
            instance=i,
            image=np.linspace(i / 10, 1 - i / 10, 20 * 30, dtype=np.float32).reshape(20, 30),
        )
        for i in range(3)
    ]
    contexts = {
        ("1", i): (images[max(0, i - 1)], images[i], images[min(2, i + 1)]) for i in range(3)
    }

    class Capture(torch.nn.Module):
        def forward(self, inputs):
            self.inputs = inputs.detach().clone()
            return torch.zeros((len(inputs), 512))

    original = Capture()
    _embed(images, contexts, original, "cpu")
    new_input = cache._inputs(images, contexts, np.arange(3))
    torch.testing.assert_close(new_input, original.inputs, rtol=0, atol=0)


@pytest.mark.parametrize(
    "dataset,split", [("aisslab", "test"), ("rsna", "test"), ("rsna", "validation")]
)
def test_fitting_inventory_rejects_heldout_before_weights_or_pixels(
    tmp_path, monkeypatch, dataset, split
):
    inventory = tmp_path / "fitting.json"
    row = source_row(tmp_path / "source", dataset, "100", split)
    write_json(
        inventory, {"schema_version": 1, "scope": "external_axial_v4_fitting", "records": [row]}
    )
    called = []

    def forbidden(*args, **kwargs):
        called.append(True)
        raise AssertionError("An encoder or source decoder was reached before the split gate")

    monkeypatch.setattr(cache, "_weight_signature", forbidden)
    monkeypatch.setattr(cache, "_study_inputs", forbidden)
    with pytest.raises(ValueError, match="split|TEST|test|TRAIN|fitting|Fitting"):
        cache.build_transfer_cache(inventory, tmp_path / "cache")
    assert not called
    assert not (tmp_path / "cache").exists()


@pytest.fixture
def prepared_sources(tmp_path):
    prepared, axial, old_root = (tmp_path / name for name in ("prepared", "axial", "old-data"))
    targets = tmp_path / "features" / "target_manifest.csv"
    targets.parent.mkdir()
    protocol = tmp_path / "protocol.json"
    write_json(protocol, {"replay_ratio": 1.0, "replay_seed": 2026})
    external = [
        source_row(prepared / "extracted", "aisslab", study, split)
        for study, split in (
            ("100", "train"),
            ("101", "train"),
            ("102", "validation"),
            ("103", "test"),
        )
    ]
    write_json(prepared / "records.json", {"records": external})
    write_json(prepared / "manifest.json", {"schema_version": 1})
    write_json(
        prepared / "patient_splits.json", {row["study_id"]: row["split"] for row in external}
    )
    old_labels, receipts = [], []
    for study, split, axial_status in [
        ("1", "train", "ok"),
        ("2", "train", "ok"),
        ("3", "train", "ok"),
        ("4", "validation", "ok"),
        ("5", "test", "ok"),
        ("6", "train", "not_assessed"),
    ]:
        old = source_row(old_root, "rsna", study, split)
        record = {
            **old,
            "status": axial_status,
            "reason": "" if axial_status == "ok" else "synthetic_geometry_failure",
            "qc": {"series": [{"study_uid_sha256": old["source"]["study_uid_sha256"]}]},
        }
        record_path = axial / "records" / f"{study}.json"
        write_json(record_path, record)
        if split != "test":
            receipts.append(
                {
                    "study_id": study,
                    "split": split,
                    "status": axial_status,
                    "record_path": str(record_path.relative_to(axial)),
                    "record_sha256": sha(record_path),
                }
            )
        for level, label in zip(("L4-L5", "L5-S1"), old["labels"]):
            old_labels.append(
                {"study_id": study, "split": split, "level": level, "label": label, "status": "ok"}
            )
    with targets.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(old_labels[0]))
        writer.writeheader()
        writer.writerows(old_labels)
    write_json(
        targets.parent / "metadata.json", {"outputs_sha256": {"target_manifest.csv": sha(targets)}}
    )
    write_json(
        axial / "metadata.json",
        {
            "schema_version": 1,
            "scope": "train_validation",
            "extraction_complete": True,
            "study_records": receipts,
            "outputs_sha256": {item["record_path"]: item["record_sha256"] for item in receipts},
            "config": {"allowed_splits": ["train", "validation"]},
        },
    )
    return dict(
        prepared=prepared, protocol=protocol, data_root=old_root, targets=targets, axial_cache=axial
    )


def test_replay_is_deterministic_and_fitting_inventory_contains_no_test(prepared_sources, tmp_path):
    args = prepared_sources
    first, second = tmp_path / "sources-1", tmp_path / "sources-2"
    summary = data.prepare_transfer_sources(out=first, **args)
    assert data.prepare_transfer_sources(out=second, **args) == summary
    fitting = json.loads((first / "fitting-sources.json").read_text())
    held = json.loads((first / "new-test-sources.json").read_text())
    assert fitting == json.loads((second / "fitting-sources.json").read_text())
    assert summary["replay_studies"] == 2
    assert all(row["split"] in {"train", "validation"} for row in fitting["records"])
    replay = [row for row in fitting["records"] if row["dataset"] == "rsna"]
    assert len(replay) == 2
    assert {row["study_id"] for row in replay} <= {"1", "2", "3"}
    assert all(row["split"] == "train" for row in replay)
    assert len(held["records"]) == 1 and held["records"][0]["split"] == "test"
    assert not (
        {(r["dataset"], r["study_id"]) for r in fitting["records"]}
        & {(r["dataset"], r["study_id"]) for r in held["records"]}
    )


def test_replay_excludes_missing_sagittal_reference_targets(prepared_sources, tmp_path):
    args = copy.copy(prepared_sources)
    rows = list(csv.DictReader(args["targets"].open(newline="")))
    for row in rows:
        if row["study_id"] == "1":
            row["status"] = "missing_vertebra"
            row["label"] = ""
    with args["targets"].open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        args["targets"].parent / "metadata.json",
        {"outputs_sha256": {"target_manifest.csv": sha(args["targets"])}},
    )
    data.prepare_transfer_sources(out=tmp_path / "out", **args)
    fitting = json.loads((tmp_path / "out/fitting-sources.json").read_text())
    replay = [row for row in fitting["records"] if row["dataset"] == "rsna"]
    assert {row["study_id"] for row in replay} == {"2", "3"}


@pytest.mark.parametrize("modified", ["target_manifest", "old_axial_receipt"])
def test_original_reference_or_axial_receipt_tampering_blocks_replay(
    prepared_sources, tmp_path, modified
):
    args = prepared_sources
    path = (
        args["targets"]
        if modified == "target_manifest"
        else args["axial_cache"] / "records" / "1.json"
    )
    path.write_text(path.read_text() + "\n")
    out = tmp_path / "should-not-publish"
    with pytest.raises(ValueError, match="checksum"):
        data.prepare_transfer_sources(out=out, **args)
    assert not out.exists()
