"""Sealed TEST extraction uses tiny synthetic arrays and no network/model assets."""

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_features import Spine, make_available, make_cohort
from test_rsna_bulk import write_selection

from ai_spine_med import features as f
from ai_spine_med import heldout_features as module
from ai_spine_med.assets import sha256_file
from ai_spine_med.rsna_bulk import ArchiveDownloadError, validate_test_selection
from ai_spine_med.training import load_feature_data


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    data, manifests, output, rows, receipt = make_cohort(tmp_path)
    selection_path = write_selection(manifests, tmp_path / "sealed")
    selection, _ = validate_test_selection(manifests, selection_path)
    model = Spine()
    monkeypatch.setattr(
        f,
        "_source_signature",
        lambda _: {
            "upstream_commit": "fixed",
            "source_archive_sha256": "0" * 64,
            "source_python_sha256": {},
            "weights_sha256": selection["original_weights_sha256"],
        },
    )
    monkeypatch.setattr(f, "_load_model", lambda *_: model)

    def load(_path):
        return SimpleNamespace(
            volume=np.ones((16, 16, 9), np.float32),
            pixel_spacing=(0.5, 0.5),
            metadata={"shape_hws": [16, 16, 9]},
        )

    monkeypatch.setattr(f, "load_sagittal_series", load)
    selection["trainval_config"] = {
        **f._source_signature(tmp_path),
        "implementation_sha256": {
            "features.py": sha256_file(Path(f.__file__)),
            "dicom.py": sha256_file(Path(inspect.getfile(load))),
            "isotropic_inplane": hashlib.sha256(
                inspect.getsource(f.isotropic_inplane).encode()
            ).hexdigest(),
        },
    }
    selection_path.write_text(json.dumps(selection, indent=2) + "\n")
    selection_sha = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    for row in rows:
        make_available(data, row, receipt)
    marker_path = data / rows[2]["series_path"] / f.COMPLETE_MARKER
    marker = json.loads(marker_path.read_text())
    marker.update(scope="heldout_test", selection_sha256=selection_sha)
    marker_path.write_text(json.dumps(marker))
    return data, manifests, output, rows, selection_path, model


def test_only_test_exact_roi_schema_resume_and_training_reader_rejects(fixture, tmp_path):
    data, manifests, output, rows, selection, model = fixture
    before = {key: value.clone() for key, value in model.grading_model.state_dict().items()}
    result = module.extract_heldout_features(
        data, manifests, tmp_path / "cache", output, selection, device="cpu"
    )
    assert result["extraction_complete"] is True
    assert result["n_requested_series"] == 1
    assert result["n_features"] == 2
    assert set(result["coverage"]) == {"test"}
    assert model.calls == 1
    assert not (output / "records/10_11.json").exists()
    assert not (output / "records/20_21.json").exists()
    with np.load(output / "features.npz", allow_pickle=False) as arrays:
        assert set(arrays["split"]) == {"test"}
        assert set(arrays.files) == set(f.ARRAY_KEYS)
        assert arrays["features"].shape == (2, 512)
        assert arrays["original_logits"].shape == (2, 4)
    with np.load(output / "records/30_31.npz", allow_pickle=False) as arrays:
        assert arrays["rois"].dtype == np.float64
        assert arrays["rois"].shape == (2, 9, 112, 224)
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["config"]["scope"] == "heldout_test"
    assert metadata["config"]["allowed_splits"] == ["test"]
    assert metadata["config"]["selection_sha256"] == result["selection_sha256"]
    record = json.loads((output / "records/30_31.json").read_text())
    assert record["timings"]["full_original_seconds"] >= 0
    assert record["timings"]["dicom_hash_seconds"] >= 0
    assert record["timings"]["measured_grading_forward_seconds"] > 0
    assert record["timings"]["grading_forward_calls"] == 1
    assert "series_timings" not in result
    assert result["timing_aggregates"]["full_original_seconds"]["count"] == 1
    for key, value in model.grading_model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    repeated = module.extract_heldout_features(
        data, manifests, tmp_path / "cache", output, selection, device="cpu"
    )
    assert repeated["runtime"]["reused"] == 1
    assert model.calls == 1
    with pytest.raises(ValueError, match="Only train and validation"):
        load_feature_data(output)


def test_gate_before_source_model_or_output(tmp_path, monkeypatch):
    touched = []
    monkeypatch.setattr(f, "_cohort", lambda *_: touched.append("metadata"))
    monkeypatch.setattr(f, "_source_signature", lambda *_: touched.append("weights"))
    with pytest.raises(ArchiveDownloadError, match="test is sealed"):
        module.extract_heldout_features(
            tmp_path, tmp_path, tmp_path, tmp_path / "out", tmp_path / "missing.json", device="cpu"
        )
    assert touched == []
    assert not (tmp_path / "out").exists()


def test_modified_checkpoint_prevents_resume_before_test_inspection(fixture, tmp_path, monkeypatch):
    data, manifests, output, rows, selection, model = fixture
    info = json.loads(selection.read_text())
    checkpoint = selection.parent / info["checkpoints"]["fine_tuned"][0]["path"]
    checkpoint.write_bytes(b"new weights after unsealing")
    touched = []
    monkeypatch.setattr(f, "_available_source", lambda *_: touched.append("test"))
    with pytest.raises(ArchiveDownloadError, match="test is sealed"):
        module.extract_heldout_features(
            data, manifests, tmp_path / "cache", output, selection, device="cpu"
        )
    assert touched == []
    assert model.calls == 0


def test_test_receipt_must_match_selection_and_weights(fixture, tmp_path, monkeypatch):
    data, manifests, output, rows, selection, model = fixture
    marker_path = data / rows[2]["series_path"] / f.COMPLETE_MARKER
    marker = json.loads(marker_path.read_text())
    marker["selection_sha256"] = "0" * 64
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="receipt is not bound"):
        module.extract_heldout_features(
            data, manifests, tmp_path / "cache", output, selection, device="cpu"
        )
    assert model.calls == 0
    monkeypatch.setattr(f, "_source_signature", lambda _: {"weights_sha256": {}})
    with pytest.raises(ValueError, match="weights differ"):
        module.extract_heldout_features(
            data, manifests, tmp_path / "cache", output, selection, device="cpu"
        )


def test_failures_retained_with_timing_and_missing_roi_not_normal(fixture, tmp_path, monkeypatch):
    data, manifests, output, rows, selection, model = fixture
    model.missing_s1 = True
    result = module.extract_heldout_features(
        data, manifests, tmp_path / "cache", output, selection, device="cpu"
    )
    assert result["n_features"] == 1
    record = json.loads((output / "records/30_31.json").read_text())
    assert record["targets"][1]["status"] == "not_assessed"
    assert record["targets"][1]["reason"] == "missing_vertebra:S1"

    def fail(*_):
        raise ValueError("synthetic decode failure")

    monkeypatch.setattr(f, "_process_series", fail)
    other = tmp_path / "failed-test"
    failed = module.extract_heldout_features(
        data, manifests, tmp_path / "cache", other, selection, device="cpu"
    )
    assert failed["n_features"] == 0
    assert failed["extraction_complete"] is True
    record = json.loads((other / "records/30_31.json").read_text())
    assert {t["status"] for t in record["targets"]} == {"failed"}
    assert record["timings"]["full_original_seconds"] >= 0


def test_changed_preprocessing_cannot_inspect_test(fixture, tmp_path, monkeypatch):
    data, manifests, output, rows, selection, model = fixture
    info = json.loads(selection.read_text())
    info["trainval_config"]["implementation_sha256"]["features.py"] = "0" * 64
    selection.write_text(json.dumps(info))
    touched = []
    monkeypatch.setattr(f, "_available_source", lambda *_: touched.append("test"))
    with pytest.raises(ValueError, match="preprocessing differs"):
        module.extract_heldout_features(
            data, manifests, tmp_path / "cache", output, selection, device="cpu"
        )
    assert touched == []
    assert model.calls == 0
