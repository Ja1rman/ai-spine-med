"""Synthetic sealed comparisons; no held-out patient MRI is opened by these tests."""

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ai_spine_med import evaluation as e
from ai_spine_med.assets import locked_assets
from ai_spine_med.data import CLASS_NAMES


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def refresh_cache(root):
    metadata = json.loads((root / "metadata.json").read_text())
    metadata["outputs_sha256"] = {
        name: sha(root / name) for name in ("features.npz", "target_manifest.csv")
    }
    write_json(root / "metadata.json", metadata)


def cache(root, targets, config, cache_id):
    root.mkdir(parents=True)
    assessed = [row for row in targets if row["status"] == "ok"][::-1]
    count = len(assessed)
    arrays = {
        "features": np.zeros((count, 512), dtype=np.float32),
        "original_logits": np.zeros((count, 4), dtype=np.float32),
        "labels": np.asarray([row["label"] for row in assessed], np.int64),
        **{
            name: np.asarray([row[name] for row in assessed], dtype="U32")
            for name in ("study_id", "series_id", "level", "split")
        },
    }
    for index, row in enumerate(assessed):
        grade = int(row["label"])
        arrays["features"][index, grade] = 1
        arrays["original_logits"][index, 0 if grade == 0 else grade + 1] = 4
    np.savez(root / "features.npz", **arrays)
    write_csv(root / "target_manifest.csv", targets)
    write_json(
        root / "metadata.json",
        {"extraction_complete": True, "config": config, "cache_id": cache_id},
    )
    refresh_cache(root)


def fixture_experiment(tmp_path, monkeypatch, *, sealed=True, all_failed=False):
    implementations = {}
    for name in ("evaluation.py", "training.py", "finetuning.py", "heldout_features.py"):
        path = tmp_path / "synthetic-sources" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("# Fixed synthetic source artifact for a protocol-unit fixture\n")
        implementations[name] = path
    monkeypatch.setattr(e, "_implementation_paths", lambda: implementations)
    manifests = tmp_path / "manifest"
    studies, series, labels, targets = [], [], [], []
    test_grades = [0, 1, 2, 0, 2, 2]
    for split, start in (("train", 100), ("validation", 200), ("test", 300)):
        for offset in range(3):
            study, number = str(start + offset), str(start + offset + 1000)
            studies.append({"study_id": study, "split": split})
            series.append(
                {
                    "study_id": study,
                    "series_id": number,
                    "split": split,
                    "series_description": "Sagittal T2/STIR",
                    "series_path": f"train_images/{study}/{number}",
                }
            )
            for level_index, level in enumerate(e.LEVELS):
                index = offset * 2 + level_index
                grade = test_grades[index] if split == "test" else index % 3
                labels.append({"study_id": study, "level": level, "split": split, "label": grade})
                failed = split == "test" and (all_failed or index == 5)
                targets.append(
                    {
                        "study_id": study,
                        "series_id": number,
                        "level": level,
                        "split": split,
                        "label": grade,
                        "status": "not_assessed" if failed else "ok",
                        "reason": "missing_target_pair" if failed else "",
                        "roi_path": "",
                        "roi_index": "",
                    }
                )
    for filename, rows in (
        ("studies.csv", studies),
        ("series.csv", series),
        ("labels.csv", labels),
    ):
        write_csv(manifests / filename, rows)
    hashes = {name: sha(manifests / name) for name in ("studies.csv", "series.csv", "labels.csv")}
    write_json(manifests / "summary.json", {"output_sha256": hashes})
    weights = {name: item["sha256"] for name, item in locked_assets()["weights"].items()}
    source_config = {
        "manifests": hashes,
        "weights_sha256": weights,
        "upstream_commit": "synthetic-source-commit",
        "source_archive_sha256": "a" * 64,
        "source_python_sha256": {"synthetic.py": "b" * 64},
        "implementation_sha256": {name: "c" * 64 for name in e.PREPROCESSING_FIELDS},
    }
    trainval = tmp_path / "trainval"
    cache(
        trainval,
        [row for row in targets if row["split"] != "test"],
        source_config,
        "synthetic-trainval",
    )
    train_data = e.load_feature_data(trainval)
    fine_config = {
        "seeds": list(e.SEEDS),
        "optimizer": "AdamW",
        "learning_rate_block": 1e-5,
        "learning_rate_head": 1e-4,
        "weight_decay": 1e-4,
        "batch_size": 16,
        "max_epochs": 2,
        "patience": 1,
        "implementation_sha256": sha(implementations["finetuning.py"]),
    }
    protocol = {
        **fine_config,
        "classes": list(CLASS_NAMES),
        "levels": list(e.LEVELS),
        "source_feature_cache_id": "synthetic-trainval",
        "study_manifest_sha256": hashes["studies.csv"],
        "heldout_test": {"no_test_tuning": True},
    }
    protocol_path = tmp_path / "protocol.json"
    write_json(protocol_path, protocol)
    fine_config["protocol_sha256"] = sha(protocol_path)
    head_dir, fine_dir = tmp_path / "head", tmp_path / "fine"
    frozen_hashes = {}
    for variant, directory in (("frozen_head", head_dir), ("fine_tuned", fine_dir)):
        config = (
            fine_config
            if variant == "fine_tuned"
            else {
                "seeds": list(e.SEEDS),
                "implementation_sha256": sha(implementations["training.py"]),
            }
        )
        runs = []
        for seed in e.SEEDS:
            path = directory / "runs" / f"seed-{seed}"
            path.mkdir(parents=True)
            state = {"weight": torch.zeros((3, 512)), "bias": torch.zeros(3)}
            state["weight"][:, :3] = torch.eye(3) * 3
            best_epoch = 2 if variant == "frozen_head" else 0
            checkpoint = {
                "model": "Linear(512,3)" if variant == "frozen_head" else e.FINE_MODEL,
                "state_dict": state
                if variant == "frozen_head"
                else {
                    **{"head." + name: value for name, value in state.items()},
                    "block.synthetic": torch.ones(1),
                },
                "class_names": list(CLASS_NAMES),
                "seed": seed,
                "best_epoch": best_epoch,
                "feature_size": 512,
                "feature_cache_sha256": train_data.file_sha256,
            }
            if variant == "fine_tuned":
                checkpoint.update(
                    input_shape=list(e.ACTIVATION_SHAPE),
                    initial_head_sha256=frozen_hashes[seed],
                    original_weights_sha256=weights,
                    protocol_sha256=sha(protocol_path),
                    batchnorm_running_statistics_frozen=True,
                    earlier_encoder_updated=False,
                )
            checkpoint_path = path / ("head.pt" if variant == "frozen_head" else "last-block.pt")
            torch.save(checkpoint, checkpoint_path)
            if variant == "frozen_head":
                frozen_hashes[seed] = sha(checkpoint_path)
            history = (
                [{"epoch": 1, "validation_macro_f1": 0.5}, {"epoch": 2, "validation_macro_f1": 1.0}]
                if variant == "frozen_head"
                else [
                    {"epoch": 0, "validation_macro_f1": 1.0},
                    {"epoch": 1, "validation_macro_f1": 0.8},
                    {"epoch": 2, "validation_macro_f1": 0.7},
                ]
            )
            metrics = {"conditional": {"macro_f1": 1.0}}
            write_json(path / "history.json", history)
            write_json(path / "metrics.json", metrics)
            runs.append(
                {
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "validation_macro_f1": 1.0,
                    "epochs_run": 2,
                    "checkpoint_sha256": sha(checkpoint_path),
                    "metrics": metrics,
                    **(
                        {
                            "initialization_parity": {
                                "frozen_head_macro_f1": 1.0,
                                "epoch0_macro_f1": 1.0,
                                "n_prediction_mismatches": 0,
                                "max_probability_abs_error": 0.0,
                                "atol": 1e-4,
                            }
                        }
                        if variant == "fine_tuned"
                        else {}
                    ),
                }
            )
        write_json(
            directory / "summary.json",
            {
                "configuration": config,
                "runs": runs,
                "all_seeds_reported": True,
                "test_evaluated": False,
            },
        )
        write_json(
            directory / "manifest.json",
            {"configuration": config, "feature_cache_sha256": train_data.file_sha256},
        )
    selection_path = tmp_path / "selection.json"
    fixture = SimpleNamespace(
        manifests=manifests,
        trainval=trainval,
        head_dir=head_dir,
        fine_dir=fine_dir,
        protocol=protocol_path,
        selection=selection_path,
        sources=implementations,
        hashes=hashes,
        weights=weights,
        targets=targets,
        root=tmp_path,
    )
    if sealed:
        fixture.seal = e.freeze_selection(
            protocol_path, manifests, trainval, head_dir, fine_dir, selection_path
        )
        test_config = {
            **source_config,
            "scope": "heldout_test",
            "allowed_splits": ["test"],
            "selection_sha256": sha(selection_path),
            "weights_sha256": weights,
            "manifests": hashes,
        }
        fixture.test_dir = tmp_path / "test-features"
        cache(
            fixture.test_dir,
            [row for row in targets if row["split"] == "test"],
            test_config,
            "synthetic-test",
        )
    return fixture


def test_seal_requires_all_six_completed_runs_and_never_overwrites(tmp_path, monkeypatch):
    fixture = fixture_experiment(tmp_path, monkeypatch, sealed=False)
    missing = fixture.fine_dir / "summary.json"
    content = missing.read_text()
    missing.unlink()
    with pytest.raises((OSError, ValueError)):
        e.freeze_selection(
            fixture.protocol,
            fixture.manifests,
            fixture.trainval,
            fixture.head_dir,
            fixture.fine_dir,
            fixture.selection,
        )
    assert not fixture.selection.exists()
    missing.write_text(content)
    seal = e.freeze_selection(
        fixture.protocol,
        fixture.manifests,
        fixture.trainval,
        fixture.head_dir,
        fixture.fine_dir,
        fixture.selection,
    )
    assert len(seal["checkpoints"]["frozen_head"]) == 3
    assert [row["best_epoch"] for row in seal["checkpoints"]["fine_tuned"]] == [0, 0, 0]
    with pytest.raises(ValueError, match="never be overwritten"):
        e.freeze_selection(
            fixture.protocol,
            fixture.manifests,
            fixture.trainval,
            fixture.head_dir,
            fixture.fine_dir,
            fixture.selection,
        )


def test_seal_rejects_config_drift_from_the_written_protocol(tmp_path, monkeypatch):
    fixture = fixture_experiment(tmp_path, monkeypatch, sealed=False)
    protocol = json.loads(fixture.protocol.read_text())
    protocol["learning_rate_head"] = 0.01
    write_json(fixture.protocol, protocol)
    with pytest.raises(ValueError, match="protocol"):
        e.freeze_selection(
            fixture.protocol,
            fixture.manifests,
            fixture.trainval,
            fixture.head_dir,
            fixture.fine_dir,
            fixture.selection,
        )


def test_seal_rejects_unordered_validation_history(tmp_path, monkeypatch):
    fixture = fixture_experiment(tmp_path, monkeypatch, sealed=False)
    path = fixture.fine_dir / "runs/seed-42/history.json"
    history = json.loads(path.read_text())
    write_json(path, [history[1], history[0], history[2]])
    with pytest.raises(ValueError, match="unique, consecutive and ordered"):
        e.freeze_selection(
            fixture.protocol,
            fixture.manifests,
            fixture.trainval,
            fixture.head_dir,
            fixture.fine_dir,
            fixture.selection,
        )


def test_seal_requires_epoch_zero_to_reproduce_matching_frozen_head(tmp_path, monkeypatch):
    fixture = fixture_experiment(tmp_path, monkeypatch, sealed=False)
    directory = fixture.fine_dir / "runs/seed-42"
    history = json.loads((directory / "history.json").read_text())
    history[0]["validation_macro_f1"] = 0.9
    write_json(directory / "history.json", history)
    metrics = {"conditional": {"macro_f1": 0.9}}
    write_json(directory / "metrics.json", metrics)
    summary = json.loads((fixture.fine_dir / "summary.json").read_text())
    summary["runs"][0].update(validation_macro_f1=0.9, metrics=metrics)
    summary["runs"][0]["initialization_parity"]["epoch0_macro_f1"] = 0.9
    write_json(fixture.fine_dir / "summary.json", summary)
    with pytest.raises(ValueError, match="epoch zero"):
        e.freeze_selection(
            fixture.protocol,
            fixture.manifests,
            fixture.trainval,
            fixture.head_dir,
            fixture.fine_dir,
            fixture.selection,
        )


def test_changed_checkpoint_blocks_access_before_any_test_cache_is_opened(tmp_path, monkeypatch):
    fixture = fixture_experiment(tmp_path, monkeypatch)
    path = fixture.head_dir / "runs/seed-42/head.pt"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(RuntimeError, match="sealed"):
        e.load_test_feature_data(tmp_path / "does-not-exist", fixture.selection)


def test_seal_body_tampering_is_detected(tmp_path, monkeypatch):
    fixture = fixture_experiment(tmp_path, monkeypatch)
    seal = json.loads(fixture.selection.read_text())
    seal["frozen"] = False
    write_json(fixture.selection, seal)
    with pytest.raises(ValueError, match="content changed"):
        e.validate_selection(fixture.selection)


def test_heldout_reader_aligns_keys_and_never_uses_train_or_validation(tmp_path, monkeypatch):
    fixture = fixture_experiment(tmp_path, monkeypatch)
    data = e.load_test_feature_data(fixture.test_dir, fixture.selection)
    assert data.arrays["features"].shape == (5, 512)
    assert len(data.targets) == 6
    assert set(data.arrays["split"]) == {"test"}
    assert data.targets[0]["feature_index"] == 4  # Arrays deliberately reversed.
    assert data.targets[-1]["feature_index"] is None
    with np.load(fixture.test_dir / "features.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["split"][0] = "train"
    np.savez(fixture.test_dir / "features.npz", **arrays)
    refresh_cache(fixture.test_dir)
    with pytest.raises(ValueError, match="test features only"):
        e.load_test_feature_data(fixture.test_dir, fixture.selection)


def test_omitting_a_failed_target_cannot_inflate_heldout_coverage(tmp_path, monkeypatch):
    fixture = fixture_experiment(tmp_path, monkeypatch)
    rows = e._csv(fixture.test_dir / "target_manifest.csv")
    write_csv(fixture.test_dir / "target_manifest.csv", rows[:-1])
    refresh_cache(fixture.test_dir)
    with pytest.raises(ValueError, match="omits requested targets"):
        e.load_test_feature_data(fixture.test_dir, fixture.selection)


def test_test_hash_and_reference_alignment_are_both_checked(tmp_path, monkeypatch):
    fixture = fixture_experiment(tmp_path, monkeypatch)
    with np.load(fixture.test_dir / "features.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["labels"][0] = (arrays["labels"][0] + 1) % 3
    np.savez(fixture.test_dir / "features.npz", **arrays)
    with pytest.raises(ValueError, match="checksum"):
        e.load_test_feature_data(fixture.test_dir, fixture.selection)
    refresh_cache(fixture.test_dir)
    with pytest.raises(ValueError, match="alignment"):
        e.load_test_feature_data(fixture.test_dir, fixture.selection)


@pytest.mark.parametrize("field", ["source_archive_sha256", "implementation_sha256"])
def test_heldout_reader_rechecks_encoder_and_preprocessing_provenance(tmp_path, monkeypatch, field):
    fixture = fixture_experiment(tmp_path, monkeypatch)
    metadata = json.loads((fixture.test_dir / "metadata.json").read_text())
    if field == "source_archive_sha256":
        metadata["config"][field] = "d" * 64
    else:
        metadata["config"][field]["isotropic_inplane"] = "e" * 64
    write_json(fixture.test_dir / "metadata.json", metadata)
    with pytest.raises(ValueError, match="source or preprocessing"):
        e.load_test_feature_data(fixture.test_dir, fixture.selection)


@pytest.mark.parametrize(
    "field", ["cache_id", "selection_sha256", "n_requested_series", "extraction_complete"]
)
def test_stale_extraction_summary_cannot_supply_preprocessing_timing(tmp_path, monkeypatch, field):
    fixture = fixture_experiment(tmp_path, monkeypatch)
    data = e.load_test_feature_data(fixture.test_dir, fixture.selection)
    aggregates = {
        name: {"count": 3, "median_seconds": 0.5, "total_seconds": 1.5}
        for name in (
            "full_original_seconds",
            "measured_grading_forward_seconds",
            "common_DICOM_localization_ROI_plus_tensorIO_seconds",
            "dicom_hash_seconds",
        )
    }
    summary = {
        "scope": "heldout_test",
        "cache_id": data.metadata["cache_id"],
        "selection_sha256": sha(fixture.selection),
        "extraction_complete": True,
        "n_features": 5,
        "n_requested_series": 3,
        "coverage": {"test": {"requested_labeled_targets": 6, "assessed": 5}},
        "timing_aggregates": aggregates,
    }
    write_json(fixture.test_dir / "summary.json", summary)
    assert (
        e._preprocessing_timings(fixture.test_dir, data, sha(fixture.selection))["available"]
        is True
    )
    summary[field] = (
        False
        if field == "extraction_complete"
        else 99
        if field == "n_requested_series"
        else "stale"
    )
    write_json(fixture.test_dir / "summary.json", summary)
    assert (
        e._preprocessing_timings(fixture.test_dir, data, sha(fixture.selection))["available"]
        is False
    )


def fake_activations(data, feature_dir, cache_dir, output_dir, *, selection_sha256, **kwargs):
    output_dir = Path(output_dir)
    output_dir.mkdir()
    values = np.zeros((len(data.arrays["features"]), *e.ACTIVATION_SHAPE), np.float32)
    values[:, :3, 0, 0, 0] = data.arrays["features"][:, :3]
    np.save(output_dir / "activations.npy", values)
    np.savez(
        output_dir / "index.npz",
        **{
            name: data.arrays[name]
            for name in ("study_id", "series_id", "level", "split", "labels")
        },
        feature_index=np.arange(len(values)),
    )
    write_json(
        output_dir / "metadata.json",
        {
            "scope": "heldout_inference",
            "selection_sha256": selection_sha256,
            "extraction_complete": True,
            "feature_cache_sha256": data.file_sha256,
            "outputs_sha256": {
                name: sha(output_dir / name) for name in ("activations.npy", "index.npz")
            },
        },
    )


def fake_fine_predict(activations, checkpoint_path, cache_dir, **kwargs):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    features = torch.zeros((len(activations), 512))
    features[:, :3] = torch.tensor(np.array(activations[:, :3, 0, 0, 0]))
    return (
        torch.nn.functional.linear(
            features, checkpoint["state_dict"]["head.weight"], checkpoint["state_dict"]["head.bias"]
        )
        .softmax(1)
        .numpy()
    )


def test_seven_entries_share_roi_denominators_and_report_every_seed(tmp_path, monkeypatch):
    from ai_spine_med import finetuning

    fixture = fixture_experiment(tmp_path, monkeypatch)
    monkeypatch.setattr(finetuning, "cache_inference_activations", fake_activations)
    monkeypatch.setattr(finetuning, "predict_last_block", fake_fine_predict)
    summary = e.evaluate_test(
        fixture.test_dir,
        fixture.selection,
        fixture.head_dir,
        fixture.fine_dir,
        tmp_path / "activations",
        tmp_path / "comparison",
        device="cpu",
        benchmark=False,
    )
    assert len(summary["entries"]) == 7
    assert summary["coverage"]["n_labeled"] == 6
    assert summary["coverage"]["n_assessed"] == 5
    for entry in summary["entries"]:
        metrics = entry["metrics"]
        assert metrics["conditional"]["macro_f1"] == 1
        assert metrics["severe"]["n_total"] == 3
        assert metrics["severe"]["n_assessed"] == 2
        assert metrics["severe"]["recall_conditional"] == 1
        assert metrics["severe"]["detection_rate_all_requested"] == 2 / 3
        assert metrics["severe"]["missed_all_requested"] == 1
    assert summary["variants"]["fine_tuned"]["seeds"] == [42, 2024, 3407]
    assert summary["variants"]["fine_tuned"]["macro_f1_std"] == 0
    assert summary["test_tuning"] is False
    assert json.loads((tmp_path / "comparison/summary.json").read_text())[
        "selection_sha256"
    ] == sha(fixture.selection)


def test_no_assessed_roi_keeps_all_severe_failures_without_fake_scores(tmp_path, monkeypatch):
    fixture = fixture_experiment(tmp_path, monkeypatch, all_failed=True)
    summary = e.evaluate_test(
        fixture.test_dir,
        fixture.selection,
        fixture.head_dir,
        fixture.fine_dir,
        tmp_path / "unused",
        tmp_path / "comparison",
        device="cpu",
        benchmark=False,
    )
    assert len(summary["entries"]) == 7
    assert summary["coverage"]["n_assessed"] == 0
    for entry in summary["entries"]:
        assert entry["metrics"]["conditional"]["macro_f1"] is None
        assert entry["metrics"]["severe"]["not_assessed"] == 3
        assert entry["metrics"]["severe"]["detection_rate_all_requested"] == 0
    assert summary["variants"]["fine_tuned"]["macro_f1_mean"] is None


def test_activation_index_drift_is_rejected_even_with_updated_hashes(tmp_path, monkeypatch):
    fixture = fixture_experiment(tmp_path, monkeypatch)
    data = e.load_test_feature_data(fixture.test_dir, fixture.selection)
    directory = tmp_path / "activations"
    fake_activations(
        data, fixture.test_dir, "unused", directory, selection_sha256=sha(fixture.selection)
    )
    with np.load(directory / "index.npz", allow_pickle=False) as archive:
        index = {name: archive[name] for name in archive.files}
    index["labels"] = (index["labels"] + 1) % 3
    np.savez(directory / "index.npz", **index)
    metadata = json.loads((directory / "metadata.json").read_text())
    metadata["outputs_sha256"]["index.npz"] = sha(directory / "index.npz")
    write_json(directory / "metadata.json", metadata)
    with pytest.raises(ValueError, match="keys, labels or order"):
        e._load_test_activations(directory, data, sha(fixture.selection))


def test_complete_classifier_timing_uses_same_device_batch_draws_and_train_rois(
    tmp_path, monkeypatch
):
    from ai_spine_med import finetuning

    fixture = fixture_experiment(tmp_path, monkeypatch)
    rois = np.ones((2, 9, 112, 224), np.float32)
    monkeypatch.setattr(e, "_training_rois", lambda selection: rois)
    synchronized = []
    monkeypatch.setattr(e, "_synchronize", synchronized.append)

    class TinyGrading(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer4 = torch.nn.Sequential(torch.nn.Identity())
            self.fc_ccs = torch.nn.Linear(512, 4)

        def forward(self, roi):
            features = roi.mean(dim=(1, 2, 3, 4))[:, None].expand(-1, 512)
            return (None, None, self.fc_ccs(self.layer4(features)))

    def suffix(checkpoint_path, cache_dir, device):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["state_dict"]
        head = torch.nn.Linear(512, 3)
        head.load_state_dict({name: state["head." + name] for name in ("weight", "bias")})
        return SimpleNamespace(block=torch.nn.Identity(), head=head)

    monkeypatch.setattr(finetuning, "_load_original_grading", lambda *args, **kwargs: TinyGrading())
    monkeypatch.setattr(finetuning, "load_last_block_head", suffix)
    result = e.benchmark_classifiers(fixture.seal, "unused", "cpu", warmup=2, draws=8)
    assert result["roi_sample_split"] == "train"
    assert result["n_roi_samples"] == 2
    assert len(result["classifier_batch1"]) == 7
    assert len(synchronized) == 7 * (2 + 2 * 8)
    for row in result["classifier_batch1"]:
        assert row["draws"] == 8 and row["warmup"] == 2 and row["batch_size"] == 1
        assert row["median_seconds"] > 0
