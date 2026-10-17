import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ai_spine_med import head_experiments as experiments
from ai_spine_med import inference as module
from ai_spine_med.assets import sha256_file, write_json
from ai_spine_med.training import predict_head


def make_heads(tmp_path, *, biases=None, preprocessing=None):
    source = {
        "upstream_commit": "verified-pinned-source",
        "source_archive_sha256": "a" * 64,
        "source_python_sha256": {"spinenet/main.py": "b" * 64},
        "weights_sha256": {"grading/ckpt1.pt": "c" * 64},
    }
    config = {
        **source,
        "implementation_sha256": preprocessing or module._preprocessing_signature(),
        "classes": list(module.CLASS_NAMES),
        "levels": list(module.LEVELS),
        "original_class_order": ["none", "mild", "moderate", "severe"],
    }
    metadata = {
        "schema_version": 1,
        "cache_id": module._canonical_sha(config),
        "config": config,
        "outputs_sha256": {"features.npz": "0" * 64, "target_manifest.csv": "1" * 64},
        "extraction_complete": True,
    }
    cache_sha = {**metadata["outputs_sha256"], "metadata.json": module._canonical_sha(metadata)}
    configuration = {
        "model": "Linear(512,3)",
        "encoder_updated": False,
        "seeds": [42, 2024, 3407],
        "implementation_sha256": "d" * 64,
    }
    root = tmp_path / "heads"
    write_json(
        root / "manifest.json",
        {
            "schema_version": 1,
            "class_names": list(module.CLASS_NAMES),
            "configuration": configuration,
            "feature_cache_sha256": cache_sha,
            "feature_cache_metadata": metadata,
        },
    )
    paths, runs = [], []
    for index, seed in enumerate(configuration["seeds"]):
        path = root / "runs" / f"seed-{seed}" / "head.pt"
        path.parent.mkdir(parents=True)
        weight = torch.zeros(3, 512)
        if biases is None:
            weight[:, 0] = torch.tensor([0.01, -0.02, 0.03])
            bias = torch.tensor([0.2, -0.1, 0.0])
        else:
            bias = torch.tensor(biases[index], dtype=torch.float32)
        torch.save(
            {
                "schema_version": 1,
                "state_dict": {"weight": weight, "bias": bias},
                "model": "Linear(512,3)",
                "feature_size": 512,
                "class_names": list(module.CLASS_NAMES),
                "seed": seed,
                "best_epoch": index + 1,
                "feature_cache_sha256": cache_sha,
            },
            path,
        )
        paths.append(path)
        runs.append({"seed": seed, "best_epoch": index + 1, "checkpoint_sha256": sha256_file(path)})
    write_json(
        root / "summary.json",
        {
            "configuration": configuration,
            "runs": runs,
            "all_seeds_reported": True,
            "test_evaluated": False,
        },
    )
    return SimpleNamespace(paths=paths, source=source, root=root)


class Grading(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_ccs = torch.nn.Linear(512, 4)
        with torch.no_grad():
            self.fc_ccs.weight.zero_()
            self.fc_ccs.bias.copy_(torch.log(torch.tensor([0.30, 0.29, 0.31, 0.10])))

    def forward(self, images):
        frozen = images.mean(dim=(1, 2, 3, 4))[:, None].expand(-1, 512)
        return None, None, self.fc_ccs(frozen)


class Spine:
    def __init__(self):
        self.grading_model = Grading().eval().requires_grad_(False)
        self.mode = "complete"
        self.calls = 0

    def detect_vb(self, volume, spacing):
        self.calls += 1
        names = ["L4", "L5", "S1"]
        if self.mode == "partial":
            names.remove("S1")
        if self.mode == "none":
            names = []
        return [
            {
                "predicted_label": name,
                "slice_nos": [1, 2],
                "polys": np.zeros((2, 4, 2)),
                "average_polygon": np.zeros((4, 2)),
            }
            for name in names
        ]

    def get_ivds_from_vert_dicts(self, vertices, volume):
        roi = np.linspace(0.1, 2, 9 * 112 * 224).reshape(9, 112, 224)
        levels = ["L4-L5"] if self.mode == "partial" else list(module.LEVELS)
        if self.mode == "none":
            levels = []
        result = [
            {"level_name": level, "volume": roi + index * 0.5} for index, level in enumerate(levels)
        ]
        if self.mode == "duplicate":
            result.append(result[1])
        if self.mode == "invalid":
            result[1]["volume"] = np.ones((9, 112, 224))
        return result


@pytest.fixture
def setup(tmp_path, monkeypatch):
    heads = make_heads(tmp_path)
    spine = Spine()
    dicom_dir = tmp_path / "SECRET_PATIENT_NAME"
    dicom_dir.mkdir()
    monkeypatch.setattr(module.features, "_source_signature", lambda _cache: heads.source)
    monkeypatch.setattr(module.features, "_load_model", lambda _cache, _device: spine)
    monkeypatch.setattr(
        module.features,
        "load_sagittal_series",
        lambda _path: SimpleNamespace(
            volume=np.ones((16, 16, 9), np.float32),
            pixel_spacing=(0.5, 0.5),
            metadata={
                "PatientName": "SECRET_PATIENT_NAME",
                "study_instance_uid": "SECRET_UID",
                "shape_hws": [16, 16, 9],
            },
        ),
    )
    return SimpleNamespace(
        heads=heads, model=spine, dicom_dir=dicom_dir, out=tmp_path / "prediction"
    )


def test_single_head_reuses_exact_encoder_without_gt_and_keeps_phi_private(setup):
    row = {"study_id": "case_1", "series_id": "case_1", "split": "inference", "series_path": "."}
    _, arrays = module.features._process_series(
        setup.model,
        setup.dicom_dir,
        row,
        {("case_1", level): None for level in module.LEVELS},
        "cpu",
    )
    expected = predict_head(arrays["features"], setup.heads.paths[0])
    report = module.run_inference(
        setup.dicom_dir, setup.out, checkpoint=[setup.heads.paths[0]], device="cpu"
    )
    assert report["status"] == "ok"
    assert report["n_assessed"] == 2 and report["reference_labels_used"] is False
    for index, target in enumerate(report["targets"]):
        np.testing.assert_allclose(
            [target[f"p_{name}"] for name in module.CLASS_NAMES], expected[index], rtol=0, atol=1e-7
        )
        # Native four-grade argmax is moderate, but none+mild probability mass
        # is larger: this verifies the baseline merge before the three-way argmax.
        assert target["original_baseline"]["category"] == "normal_mild"
        np.testing.assert_allclose(
            [target["original_baseline"][f"p_{name}"] for name in module.CLASS_NAMES],
            [0.59, 0.31, 0.10],
            atol=1e-7,
        )
    assert report["model"]["checkpoints"][0]["sha256"] == sha256_file(setup.heads.paths[0])
    content = (setup.out / "predictions.json").read_text() + (
        setup.out / "predictions.csv"
    ).read_text()
    for private in (
        "SECRET_PATIENT_NAME",
        "SECRET_UID",
        str(setup.dicom_dir),
        "PatientName",
        "series_instance_uid",
        "study_instance_uid",
    ):
        assert private not in content
    with (setup.out / "predictions.csv").open(newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 2 and {row["case_id"] for row in csv_rows} == {"case_1"}
    assert set(path.name for path in setup.out.iterdir()) == {"predictions.csv", "predictions.json"}


@pytest.mark.parametrize(
    "mode,reason",
    [
        ("partial", "missing_vertebra:S1"),
        ("duplicate", "duplicate_target_roi"),
        ("invalid", "invalid_roi_shape_or_values"),
    ],
)
def test_partial_roi_is_an_explicit_abstention(setup, mode, reason):
    setup.model.mode = mode
    report = module.run_inference(
        setup.dicom_dir, setup.out, checkpoint=setup.heads.paths[0], device="cpu"
    )
    first, missing = report["targets"]
    assert report["status"] == "partial" and report["n_assessed"] == 1
    assert first["status"] == "ok"
    assert missing["status"] == "not_assessed" and missing["reason"] == reason
    assert missing["prediction"] is None and missing["category"] is None
    assert missing["original_baseline"] is None
    assert all(missing[f"p_{name}"] is None for name in module.CLASS_NAMES)
    with (setup.out / "predictions.csv").open(newline="") as stream:
        row = list(csv.DictReader(stream))[1]
    assert row["prediction"] == row["category"] == row["p_normal_mild"] == ""


def test_failed_input_and_no_detected_targets_never_become_normal(setup, monkeypatch):
    def bad_input(_path):
        raise module.dicom.DicomGeometryError("SECRET_PATIENT_NAME at /private/patient/file.dcm")

    monkeypatch.setattr(module.features, "load_sagittal_series", bad_input)
    report = module.run_inference(
        setup.dicom_dir, setup.out, checkpoint=setup.heads.paths[0], device="cpu"
    )
    assert report["status"] == "failed" and report["n_assessed"] == 0
    assert report["failure"]["reason"] == "invalid_dicom_series"
    assert all(
        row["status"] == "not_assessed" and row["prediction"] is None for row in report["targets"]
    )
    assert "SECRET_PATIENT_NAME" not in (setup.out / "predictions.json").read_text()
    monkeypatch.setattr(
        module.features,
        "load_sagittal_series",
        lambda _: SimpleNamespace(
            volume=np.ones((8, 8, 9), np.float32), pixel_spacing=(1, 1), metadata={}
        ),
    )
    setup.model.mode = "none"
    empty = module.run_inference(
        setup.dicom_dir, setup.out.parent / "empty", checkpoint=setup.heads.paths[0], device="cpu"
    )
    assert empty["status"] == "failed" and empty["n_assessed"] == 0
    assert empty["failure"]["reason"] == "no_supported_target_roi"
    assert all(row["prediction"] is None for row in empty["targets"])


def test_ensemble_averages_three_softmax_vectors_instead_of_logits(setup, monkeypatch, tmp_path):
    distributions = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.3, 0.2, 0.5]])
    heads = make_heads(tmp_path / "ensemble", biases=np.log(distributions))
    monkeypatch.setattr(module.features, "_source_signature", lambda _: heads.source)
    report = module.run_inference(
        setup.dicom_dir, setup.out, checkpoint=heads.paths[::-1], device="cpu"
    )
    expected = distributions.mean(axis=0)
    actual = [report["targets"][0][f"p_{name}"] for name in module.CLASS_NAMES]
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    geometric = np.exp(np.log(distributions).mean(axis=0))
    geometric /= geometric.sum()
    assert not np.allclose(actual, geometric)
    assert report["model"]["variant"] == "frozen_head_probability_ensemble"
    assert [row["seed"] for row in report["model"]["checkpoints"]] == [42, 2024, 3407]
    with pytest.raises(ValueError, match="three distinct"):
        module.run_inference(
            setup.dicom_dir,
            setup.out.parent / "duplicate",
            checkpoint=[heads.paths[0]] * 3,
            device="cpu",
        )


def test_incompatible_preprocessing_nonfinite_weights_or_changed_selection_are_rejected(
    setup, monkeypatch, tmp_path
):
    changed = module._preprocessing_signature()
    changed["features.py"] = "e" * 64
    incompatible = make_heads(tmp_path / "incompatible", preprocessing=changed)
    with pytest.raises(ValueError, match="encoder or preprocessing"):
        module.run_inference(
            setup.dicom_dir, setup.out, checkpoint=incompatible.paths[0], device="cpu"
        )
    assert not setup.out.exists()
    path = setup.heads.paths[0]
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint["state_dict"]["bias"][0] = float("nan")
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="nonfinite"):
        module.run_inference(setup.dicom_dir, setup.out, checkpoint=path, device="cpu")
    checkpoint["state_dict"]["bias"][0] = 0.0
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="selected head"):
        module.run_inference(setup.dicom_dir, setup.out, checkpoint=path, device="cpu")
    assert not setup.out.exists()


def test_existing_output_is_never_overwritten_or_processed(setup, monkeypatch):
    setup.out.mkdir()
    sentinel = setup.out / "important.txt"
    sentinel.write_text("preserve me")

    def must_not_load(_):
        raise AssertionError("Provenance/model should not be loaded")

    monkeypatch.setattr(module.features, "_source_signature", must_not_load)
    with pytest.raises(ValueError, match="Output already exists"):
        module.run_inference(
            setup.dicom_dir, setup.out, checkpoint=setup.heads.paths[0], device="cpu"
        )
    assert sentinel.read_text() == "preserve me" and setup.model.calls == 0


def make_v2_heads(tmp_path):
    original = make_heads(tmp_path / "original")
    original_manifest = json.loads((original.root / "manifest.json").read_text())
    metadata = original_manifest["feature_cache_metadata"]
    root = tmp_path / "v2"
    write_json(root / "folds.json", {"scope": "synthetic TRAIN folds, no patient inputs"})
    provenance = {
        "implementation_sha256": sha256_file(Path(experiments.__file__)),
        "protocol_sha256": "f" * 64,
        "folds_sha256": sha256_file(root / "folds.json"),
        "feature_cache_sha256": original_manifest["feature_cache_sha256"],
        "feature_cache_id": metadata["cache_id"],
    }
    configuration = {
        **provenance,
        "class_names": list(module.CLASS_NAMES),
        "architectures": {name: experiments.architecture(name) for name in experiments.CANDIDATES},
    }
    final_epochs = {
        name: {str(seed): 2 for seed in experiments.SEEDS} for name in experiments.CANDIDATES
    }
    selection = {
        **provenance,
        "selected_candidate": "mlp",
        "final_epochs": final_epochs,
        "created_before_original_validation_evaluation": True,
        "validation_used_for_epoch_or_candidate_selection": False,
        "test_accessed": False,
    }
    write_json(root / "selection.json", selection)
    selection_sha = sha256_file(root / "selection.json")
    paths, candidates = {}, {}
    for name in experiments.CANDIDATES:
        paths[name], runs = [], []
        for seed in experiments.SEEDS:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                model = experiments.build_head(name)
            path = root / "validation" / name / f"seed-{seed}" / "head.pt"
            path.parent.mkdir(parents=True)
            torch.save(
                {
                    **provenance,
                    "schema_version": 1,
                    "candidate": name,
                    "architecture": experiments.architecture(name),
                    "feature_size": 512,
                    "class_names": list(module.CLASS_NAMES),
                    "state_dict": model.state_dict(),
                    "seed": seed,
                    "best_epoch": 2,
                    "scope": "Original-validation single development check; epochs selected by TRAIN CV",
                    "fold": None,
                    "encoder_updated": False,
                    "selection_sha256": selection_sha,
                },
                path,
            )
            paths[name].append(path)
            runs.append(
                {
                    "seed": seed,
                    "fold": None,
                    "best_epoch": 2,
                    "checkpoint_sha256": sha256_file(path),
                }
            )
        candidates[name] = {"validation": {"runs": runs}}
    write_json(
        root / "manifest.json",
        {
            "schema_version": 1,
            "configuration": configuration,
            "feature_cache_metadata": metadata,
            "feature_cache_sha256": original_manifest["feature_cache_sha256"],
        },
    )
    write_json(
        root / "summary.json",
        {
            "configuration": configuration,
            "candidates": candidates,
            "all_seeds_reported": True,
            "test_evaluated": False,
            "new_version_independently_tested": False,
            "selected_candidate": "mlp",
            "selection_sha256": selection_sha,
            "final_epochs": final_epochs,
        },
    )
    return SimpleNamespace(paths=paths, source=original.source, root=root)


@pytest.mark.parametrize("candidate", ["linear", "mlp", "coral"])
def test_final_v2_heads_and_ensemble_match_the_experiment_predictor(
    setup, tmp_path, monkeypatch, candidate
):
    exported = make_v2_heads(tmp_path / "v2-fixture")
    monkeypatch.setattr(module.features, "_source_signature", lambda _: exported.source)
    row = {"study_id": "case_1", "series_id": "case_1", "split": "inference", "series_path": "."}
    _, arrays = module.features._process_series(
        setup.model,
        setup.dicom_dir,
        row,
        {("case_1", level): None for level in module.LEVELS},
        "cpu",
    )
    paths = exported.paths[candidate]
    expected = np.mean(
        [experiments.predict_experiment_head(arrays["features"], path) for path in paths],
        axis=0,
        dtype=np.float64,
    )
    report = module.run_inference(setup.dicom_dir, setup.out, checkpoint=paths, device="cpu")
    assert report["status"] == "ok"
    assert report["model"]["variant"] == "frozen_head_v2_probability_ensemble"
    assert report["model"]["candidate"] == candidate
    actual = np.array(
        [[row[f"p_{name}"] for name in module.CLASS_NAMES] for row in report["targets"]]
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
    assert np.all(actual >= 0)
    np.testing.assert_allclose(actual.sum(axis=1), 1, atol=1e-6)
    assert all(
        row["selected_by_training_cv"] is (candidate == "mlp")
        for row in report["model"]["checkpoints"]
    )


def test_v2_cv_heads_and_mixed_architecture_ensembles_are_rejected(setup, tmp_path, monkeypatch):
    exported = make_v2_heads(tmp_path / "v2-fixture")
    monkeypatch.setattr(module.features, "_source_signature", lambda _: exported.source)
    checkpoint = torch.load(exported.paths["mlp"][0], map_location="cpu", weights_only=True)
    checkpoint["fold"] = 0
    checkpoint["scope"] = "TRAIN CV held fold (development epoch selection)"
    cv_path = exported.root / "cv" / "mlp" / "fold-0" / "seed-42" / "head.pt"
    cv_path.parent.mkdir(parents=True)
    torch.save(checkpoint, cv_path)
    with pytest.raises(ValueError, match="CV heads"):
        module.run_inference(setup.dicom_dir, setup.out, checkpoint=cv_path, device="cpu")
    mixed = [exported.paths["linear"][0], exported.paths["mlp"][1], exported.paths["coral"][2]]
    with pytest.raises(ValueError, match="three selected seeds"):
        module.run_inference(setup.dicom_dir, setup.out, checkpoint=mixed, device="cpu")
    assert not setup.out.exists()
