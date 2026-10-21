"""TEST preparation cannot precede frozen choices or feed truth into inference."""

import hashlib
import json

import numpy as np
import pytest
import torch

from ai_spine_med import axial_localizer_eval_data as evaluation
from ai_spine_med.assets import sha256_file


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


@pytest.fixture
def seals(tmp_path):
    protocol = tmp_path / "protocol.json"
    write(protocol, {"seed": 42, "sigma_mm": 5.0})
    canonical = hashlib.sha256(
        json.dumps(json.loads(protocol.read_text()), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    fitting = tmp_path / "fitting.json"
    write(fitting, {"records": []})
    targets = tmp_path / "targets.csv"
    targets.write_text("study_id,level,label\n")
    localizer = tmp_path / "localizer" / "selection.json"
    localizer.parent.mkdir()
    (localizer.parent / "model.pt").write_bytes(b"localizer")
    write(
        localizer,
        {
            "schema_version": 1,
            "model_family": "explicit_axial_localizer_v1",
            "frozen": True,
            "protocol_sha256": sha256_file(protocol),
            "training_index_sha256": sha256_file(fitting),
            "checkpoint": "model.pt",
            "checkpoint_sha256": sha256_file(localizer.parent / "model.pt"),
            "localization_gate_passed": True,
        },
    )
    severity = tmp_path / "severity" / "selection.json"
    severity.parent.mkdir()
    runs = []
    for seed in [42, 2024, 3407]:
        path = severity.parent / f"seed-{seed}.pt"
        path.write_bytes(str(seed).encode())
        runs.append({"seed": seed, "checkpoint": path.name, "checkpoint_sha256": sha256_file(path)})
    write(severity.parent / "manifest.json", {"training": "fixed"})
    write(severity.parent / "summary.json", {"status": "complete"})
    write(
        severity,
        {
            "schema_version": 1,
            "manifest_sha256": sha256_file(severity.parent / "manifest.json"),
            "summary_sha256": sha256_file(severity.parent / "summary.json"),
            "family": "localized_stenosis_v1",
            "frozen": True,
            "status": "sealed_before_test",
            "protocol_sha256": canonical,
            "training_index_sha256": sha256_file(targets),
            "seeds": [42, 2024, 3407],
            "runs": runs,
        },
    )
    return localizer, severity, protocol, fitting, targets


def test_both_complete_unchanged_selections_verified_without_medical_files(seals):
    receipt = evaluation.verify_evaluation_seals(*seals)
    assert receipt["both_selections_frozen_before_test"] is True
    assert receipt["localization_gate_passed"] is True
    assert receipt["protocol_file_sha256"] != receipt["protocol_canonical_sha256"]


@pytest.mark.parametrize("mutate", ["gate", "frozen", "protocol", "training_index", "checkpoint"])
def test_localizer_tamper_or_failed_gate_blocks_test(seals, mutate):
    localizer, *_ = seals
    if mutate == "checkpoint":
        (localizer.parent / "model.pt").write_bytes(b"changed")
    else:
        data = json.loads(localizer.read_text())
        data[
            {
                "gate": "localization_gate_passed",
                "frozen": "frozen",
                "protocol": "protocol_sha256",
                "training_index": "training_index_sha256",
            }[mutate]
        ] = False
        write(localizer, data)
    with pytest.raises(ValueError, match="TEST"):
        evaluation.verify_evaluation_seals(*seals)


def test_all_three_severity_checkpoint_bytes_required(seals):
    _, severity, *_ = seals
    (severity.parent / "seed-2024.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="severity checkpoint"):
        evaluation.verify_evaluation_seals(*seals)


def test_missing_severity_seed_blocks_test(seals):
    _, severity, *_ = seals
    value = json.loads(severity.read_text())
    value["runs"].pop()
    write(severity, value)
    with pytest.raises(ValueError, match="three"):
        evaluation.verify_evaluation_seals(*seals)


def test_prepare_checks_models_before_opening_any_test_inventory(tmp_path, monkeypatch):
    def blocked(*args):
        raise ValueError("models are not frozen")

    monkeypatch.setattr(evaluation, "verify_evaluation_seals", blocked)
    monkeypatch.setattr(
        evaluation, "_json", lambda *a: pytest.fail("TEST inventory opened too early")
    )
    with pytest.raises(ValueError, match="not frozen"):
        evaluation.prepare_evaluation_data(
            "loc.json", "severity.json", "protocol.json", "targets.csv", out=tmp_path / "out"
        )
    assert not (tmp_path / "out").exists()


def test_image_feature_api_has_no_annotation_inputs_or_outputs():
    activations = np.zeros((2, 512, 7, 7), dtype=np.float32)
    activations[0] = 2
    activations[1] = 4
    positions = np.asarray([[0, 0.1], [1, 0.9]], dtype=np.float32)
    result = evaluation.image_features(activations, positions, torch.nn.Identity(), "cpu")
    assert set(result) == {"features", "positions"}
    np.testing.assert_allclose(result["features"][0], 2)
    np.testing.assert_allclose(result["features"][1], 4)
    np.testing.assert_array_equal(result["positions"], positions)
