"""Sealed ROI extraction, label isolation and resumable image-only provenance."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ai_spine_med import axial_features as af
from ai_spine_med import axial_localizer_extract as extraction
from ai_spine_med.assets import sha256_file, write_json


def seal(tmp_path, *, phase="fitting", gate=True, split="train"):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"roi_mm": 50, "roi_max_centers": 3, "roi_window_mm": 10}))
    index = tmp_path / "index.json"
    index.write_text(
        json.dumps(
            {
                "configuration": {"phase": phase},
                "records": [{"dataset": "rsna", "study_id": "s", "split": split}],
            }
        )
    )
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"frozen-checkpoint")
    summary = tmp_path / "summary.json"
    summary.write_text('{"status":"complete"}')
    selection = tmp_path / "selection.json"
    selected = {
        "schema_version": 1,
        "model_family": extraction.localizer.FAMILY,
        "frozen": True,
        "localization_gate_passed": gate,
        "protocol_sha256": sha256_file(protocol),
        "implementation_sha256": sha256_file(Path(extraction.localizer.__file__)),
        "training_index_sha256": sha256_file(index),
        "checkpoint": "model.pt",
        "checkpoint_sha256": sha256_file(checkpoint),
        "summary_sha256": sha256_file(summary),
    }
    write_json(selection, selected)
    return index, selection, protocol


def test_fitting_seal_refuses_gate_failure_test_and_changed_checkpoint(tmp_path):
    index, selection, protocol = seal(tmp_path, gate=False)
    with pytest.raises(ValueError, match="gate"):
        extraction.verify_localizer_selection(index, selection, protocol)
    index, selection, protocol = seal(tmp_path, split="test")
    with pytest.raises(ValueError, match="training index"):
        extraction.verify_localizer_selection(index, selection, protocol)
    index, selection, protocol = seal(tmp_path)
    (tmp_path / "model.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        extraction.verify_localizer_selection(index, selection, protocol)


def test_evaluation_inputs_require_both_frozen_selections(tmp_path):
    index, selection, protocol = seal(tmp_path, phase="evaluation_image_inputs", split="test")
    with pytest.raises(ValueError, match="both model seals"):
        extraction.verify_localizer_selection(index, selection, protocol)
    value = json.loads(index.read_text())
    value["configuration"]["seals"] = {
        "both_selections_frozen_before_test": True,
        "localization_gate_passed": True,
        "localizer_selection_sha256": sha256_file(selection),
        "protocol_file_sha256": sha256_file(protocol),
        "fitting_index_sha256": json.loads(selection.read_text())["training_index_sha256"],
        "severity_selection_sha256": "a" * 64,
    }
    write_json(index, value)
    assert (
        extraction.verify_localizer_selection(index, selection, protocol)[3]
        == "evaluation_image_inputs"
    )


def test_input_loader_never_opens_reference_npz_keys(monkeypatch, tmp_path):
    path = tmp_path / "inputs.npz"
    path.write_bytes(b"synthetic")
    accessed = []

    class Archive:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __getitem__(self, key):
            accessed.append(key)
            assert key in {"features", "positions"}
            return np.zeros((3, 512) if key == "features" else (3, 2), np.float32)

    monkeypatch.setattr(extraction.np, "load", lambda *args, **kwargs: Archive())
    row = {
        "arrays_path": path.name,
        "arrays_sha256": sha256_file(path),
        "selected_slices": [{}, {}, {}],
    }
    features, positions = extraction.read_image_inputs(tmp_path / "index.json", row)
    assert features.shape == (3, 512) and positions.shape == (3, 2)
    assert accessed == ["features", "positions"]


def test_inference_never_uses_reference_centers_or_grades_and_decodes_once(monkeypatch):
    calls = []
    images = [SimpleNamespace(series_id="1", instance=i) for i in (1, 2, 3)]
    prepared = {"images": images, "contexts": {}}
    monkeypatch.setattr(
        extraction, "prepare_crop_source", lambda row: calls.append("decode") or prepared
    )
    monkeypatch.setattr(
        extraction.localizer,
        "predict_slice_weights",
        lambda *args: np.asarray([[0.1, 0.8, 0.1], [0.1, 0.2, 0.7]], np.float32),
    )
    monkeypatch.setattr(
        extraction,
        "_inputs",
        lambda images, contexts, indices: torch.tensor(indices, dtype=torch.float32)[
            :, None, None, None
        ].expand(2, 3, 224, 224),
    )
    monkeypatch.setattr(
        extraction, "_full_fov_maps", lambda encoder, inputs: torch.zeros((2, 512, 7, 7))
    )

    class Center:
        def __call__(self, maps):
            return torch.tensor([[0.25, 0.5], [0.75, 0.6]]), torch.zeros((2, 49))

    class Encoder:
        def __call__(self, images):
            return images.mean((1, 2, 3))[:, None].expand(-1, 512)

    predictions = []

    def crop(row, prediction, **kwargs):
        assert kwargs["_prepared"] is prepared
        predictions.append(prediction)
        return {
            "levels": {
                level: {
                    "status": "ok",
                    "reason": "",
                    "images": torch.ones((3, 3, 224, 224)) * i,
                    "metadata": {},
                }
                for i, level in enumerate(extraction.LEVELS)
            },
            "parameters": {},
            "source_qc": {},
        }

    monkeypatch.setattr(extraction, "crop_level_inputs", crop)
    row = {
        "selected_slices": [{"series_id": "1", "instance": i} for i in (1, 2, 3)],
        "anchors": "MUST NEVER BE READ",
        "labels": [2, 2],
    }
    args = (
        np.zeros((3, 512), np.float32),
        np.zeros((3, 2), np.float32),
        None,
        Center(),
        Encoder(),
        {"roi_mm": 50, "roi_max_centers": 3, "roi_window_mm": 10},
        "cpu",
    )
    before = extraction.infer_localized_study(row, *args)
    row.update(anchors={"wrong_level": 999}, labels=[0, 0], center_xy="ORACLE CENTER")
    after = extraction.infer_localized_study(row, *args)
    np.testing.assert_array_equal(before[0], after[0])
    assert predictions[0] == predictions[1]
    assert [predictions[0]["levels"][level]["instance"] for level in extraction.LEVELS] == [2, 3]
    assert calls == ["decode", "decode"]


def test_reference_join_fails_external_training_and_rsna_uses_direct_grades(tmp_path):
    external = [{"dataset": "aisslab", "study_id": "x", "split": "train"}]
    with pytest.raises(ValueError, match="cannot enter severity fitting"):
        extraction._reference_labels(external, "fitting", "/missing.csv", "/missing.json")
    path = tmp_path / "train.csv"
    path.write_text(
        "study_id,spinal_canal_stenosis_l4_l5,spinal_canal_stenosis_l5_s1\ns,Moderate,Severe\n"
    )
    labels, receipt = extraction._reference_labels(
        [{"dataset": "rsna"}], "fitting", path, "/missing.json"
    )
    assert labels == {("rsna", "s", "L4-L5"): 1, ("rsna", "s", "L5-S1"): 2}
    assert receipt["rsna_labels_sha256"] == sha256_file(path)


def test_end_to_end_cache_is_resumable_and_preserves_explicit_failures(monkeypatch, tmp_path):
    index_path, selection, protocol = seal(tmp_path)
    index = json.loads(index_path.read_text())
    ok = {
        "dataset": "rsna",
        "study_id": "s",
        "split": "train",
        "status": "ok",
        "reason": "",
        "arrays_path": "inputs.npz",
        "arrays_sha256": "a" * 64,
        "source_root": "/private",
        "source": {},
        "selected_slices": [],
    }
    failed = {
        "dataset": "rsna",
        "study_id": "f",
        "split": "validation",
        "status": "failed",
        "reason": "bad_geometry",
    }
    index["records"] = [ok, failed]
    write_json(index_path, index)
    seal_value = json.loads(selection.read_text())
    seal_value["training_index_sha256"] = sha256_file(index_path)
    write_json(selection, seal_value)
    monkeypatch.setattr(
        extraction.localizer, "load_localizer_checkpoint", lambda *args: (None, None, seal_value)
    )
    monkeypatch.setattr(af, "_weight_signature", lambda *args: {"sha256": "weights"})
    monkeypatch.setattr(af, "_load_encoder", lambda *args: None)
    monkeypatch.setattr(extraction, "read_image_inputs", lambda *args: (None, None))
    inferred = []

    def infer(row, *args):
        inferred.append(row["study_id"])
        return (
            np.ones((2, 512), np.float32),
            [
                {"level": level, "status": "ok", "reason": "", "feature_index": i}
                for i, level in enumerate(extraction.LEVELS)
            ],
            {"prediction": "image-only"},
        )

    monkeypatch.setattr(extraction, "infer_localized_study", infer)
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "study_id,spinal_canal_stenosis_l4_l5,spinal_canal_stenosis_l5_s1\ns,Moderate,Severe\nf,Normal/Mild,\n"
    )
    out = tmp_path / "roi"
    first = extraction.extract_localized_features(
        index_path, selection, out, protocol, device="cpu", rsna_labels=labels
    )
    second = extraction.extract_localized_features(
        index_path, selection, out, protocol, device="cpu", rsna_labels=labels
    )
    assert inferred == ["s"]
    assert first["n_targets"] == second["n_targets"] == 4
    assert first["by_status"] == {"ok": 2, "failed": 2}
    with np.load(out / "features.npz", allow_pickle=False) as archive:
        assert archive.files == ["features"]
        assert archive["features"].shape == (2, 512)
    with (out / "targets.csv").open() as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == list(extraction.FIELDS)
        targets = list(reader)
    assert targets[0]["feature_index"] == "0" and targets[1]["feature_index"] == "1"
    assert targets[2]["feature_index"] == "" and targets[2]["reason"] == "bad_geometry"
    assert targets[3]["label"] == ""
