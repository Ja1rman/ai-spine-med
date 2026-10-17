"""Small synthetic activations exercise the real optimizer/resume/BN contracts."""

import csv
import json

import numpy as np
import pytest
import torch
from torch import nn

from ai_spine_med import finetuning as module
from ai_spine_med.assets import sha256_file
from ai_spine_med.training import load_feature_data


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(512, 512, 1, bias=False)
        self.bn = nn.BatchNorm3d(512)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[:, :, 0, 0, 0].copy_(torch.eye(512))

    def forward(self, values):
        return torch.relu(values + self.bn(self.conv(values)))


class TinyGrading(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer4 = nn.Sequential(TinyBlock())
        self.fc_ccs = nn.Linear(512, 4)
        with torch.no_grad():
            self.fc_ccs.weight.zero_()
            self.fc_ccs.bias.zero_()

    def forward(self, rois):
        labels = rois.mean((1, 2, 3, 4)).round().long() - 1
        activation = nn.functional.one_hot(labels, 512).float()[:, :, None, None, None]
        activation = activation.expand(-1, *module.ACTIVATION_SHAPE)
        features = self.layer4(activation).mean((2, 3, 4))
        return None, None, self.fc_ccs(features)


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(module, "ACTIVATION_SHAPE", (512, 1, 2, 2))
    signature = {
        "upstream_commit": "synthetic",
        "weights_sha256": {"grading/ckpt1.pt": "1" * 64},
        "source_python_sha256": {"synthetic.py": "2" * 64},
    }
    monkeypatch.setattr(module, "_source_signature", lambda cache: signature)
    monkeypatch.setattr(module, "_original_block", lambda cache: (TinyBlock().eval(), signature))
    monkeypatch.setattr(
        module,
        "_load_original_grading",
        lambda cache, device: TinyGrading().eval().requires_grad_(False).to(device),
    )
    features, activation_dir, heads = (
        tmp_path / name for name in ("features", "activations", "heads")
    )
    features.mkdir()
    activation_dir.mkdir()
    count = 9
    labels = np.arange(count, dtype=np.int64) % 3
    activations = np.zeros((count, *module.ACTIVATION_SHAPE), np.float32)
    for index, label in enumerate(labels):
        activations[index, label] = 1
    original = TinyBlock().eval()
    with torch.inference_mode():
        frozen_features = original(torch.from_numpy(activations)).mean((2, 3, 4)).numpy()
    arrays = {
        "features": frozen_features.astype(np.float32),
        "original_logits": np.zeros((count, 4), np.float32),
        "labels": labels,
        "study_id": np.asarray([str(i + 1000) for i in range(count)]),
        "series_id": np.asarray([str(i + 2000) for i in range(count)]),
        "level": np.asarray(["L4-L5"] * count),
        "split": np.asarray(["train"] * 6 + ["validation"] * 3),
    }
    rows = []
    for index in range(count):
        relative = f"records/{arrays['study_id'][index]}_{arrays['series_id'][index]}.npz"
        path = features / relative
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(
            path,
            rois=np.full((1, 9, 112, 224), labels[index] + 1, dtype=np.float64),
            levels=np.asarray(["L4-L5"]),
        )
        path.with_suffix(".json").write_text(
            json.dumps({"cache_id": "synthetic-cache", "arrays_sha256": sha256_file(path)})
        )
        rows.append(
            {
                **{
                    key: str(arrays[key][index])
                    for key in ("study_id", "series_id", "level", "split")
                },
                "label": int(labels[index]),
                "status": "ok",
                "reason": "",
                "roi_path": relative,
                "roi_index": 0,
            }
        )
    np.savez(features / "features.npz", **arrays)
    with (features / "target_manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (features / "metadata.json").write_text(
        json.dumps(
            {
                "cache_id": "synthetic-cache",
                "extraction_complete": True,
                "config": signature,
                "outputs_sha256": {
                    name: sha256_file(features / name)
                    for name in ("features.npz", "target_manifest.csv")
                },
            }
        )
    )
    data = load_feature_data(features)
    np.save(activation_dir / "activations.npy", activations)
    np.savez(
        activation_dir / "index.npz",
        **{key: arrays[key] for key in module.INDEX_KEYS},
        feature_index=np.arange(count),
    )
    (activation_dir / "metadata.json").write_text(
        json.dumps(
            {
                "cache_id": "synthetic-activation",
                "scope": "train_validation",
                "extraction_complete": True,
                "activation_shape": list(module.ACTIVATION_SHAPE),
                "source_signature": signature,
                "feature_cache_sha256": data.file_sha256,
                "outputs_sha256": {
                    name: sha256_file(activation_dir / name)
                    for name in ("activations.npy", "index.npz")
                },
            }
        )
    )
    for seed in module.SEEDS:
        path = heads / "runs" / f"seed-{seed}" / "head.pt"
        path.parent.mkdir(parents=True)
        head = nn.Linear(512, 3)
        with torch.no_grad():
            head.weight.zero_()
            head.bias.zero_()
            head.weight[:, :3].copy_(torch.eye(3) * 3)
        torch.save(
            {
                "model": "Linear(512,3)",
                "feature_size": 512,
                "class_names": list(module.CLASS_NAMES),
                "seed": seed,
                "feature_cache_sha256": data.file_sha256,
                "state_dict": head.state_dict(),
            },
            path,
        )
    return features, activation_dir, heads, tmp_path / "source", signature


def test_cache_resume_preserves_original_features_and_refuses_corruption(experiment, tmp_path):
    features, _activations, _heads, source, _signature = experiment
    out = tmp_path / "new-cache"
    first = module.cache_last_block_activations(
        features, source, out, device="cpu", batch_size=2, limit=2
    )
    assert first["verified_features"] == 2 and first["extraction_complete"] is False
    second = module.cache_last_block_activations(features, source, out, device="cpu", batch_size=2)
    assert second["verified_features"] == 9 and second["extraction_complete"] is True
    assert second["maximum_feature_abs_error"] == 0
    assert second["maximum_original_logits_abs_error"] == 0
    assert second["original_encoder_and_bn_unchanged"] is True
    values = np.load(out / "activations.npy", mmap_mode="r+")
    values[0, 0, 0, 0, 0] += 1
    values.flush()
    with pytest.raises(ValueError, match="row is corrupt"):
        module.cache_last_block_activations(features, source, out, device="cpu")


def test_training_freezes_bn_keeps_prefix_and_old_heads_and_selects_epoch_zero(
    experiment, tmp_path
):
    features, activations, heads, source, _signature = experiment
    before_prefix = sha256_file(activations / "activations.npy")
    before_heads = {
        seed: sha256_file(heads / "runs" / f"seed-{seed}" / "head.pt") for seed in module.SEEDS
    }
    summary = module.train_last_block(
        features,
        activations,
        heads,
        source,
        tmp_path / "run",
        threads=1,
        max_epochs=2,
        patience=1,
        batch_size=3,
    )
    assert [run["seed"] for run in summary["runs"]] == list(module.SEEDS)
    assert sha256_file(activations / "activations.npy") == before_prefix
    for run in summary["runs"]:
        assert run["best_epoch"] == 0
        assert run["block_parameters_changed_during_optimization"] is True
        assert run["head_parameters_changed_during_optimization"] is True
        assert run["all_batchnorm_running_statistics_unchanged"] is True
        assert run["selected_block_parameters_changed"] is False
        assert run["earlier_encoder_updated"] is False
        assert run["initialization_parity"]["n_prediction_mismatches"] == 0
        assert (
            run["initialization_parity"]["frozen_head_macro_f1"]
            == run["initialization_parity"]["epoch0_macro_f1"]
            == 1
        )
        directory = tmp_path / "run/runs" / f"seed-{run['seed']}"
        history = json.loads((directory / "history.json").read_text())
        assert [row["epoch"] for row in history] == [0, 1]
        assert max(row["validation_macro_f1"] for row in history) == run["validation_macro_f1"]
        checkpoint = torch.load(directory / "last-block.pt", map_location="cpu", weights_only=True)
        initial = TinyBlock().eval()
        for key, value in initial.state_dict().items():
            torch.testing.assert_close(
                checkpoint["state_dict"][f"block.{key}"], value, rtol=0, atol=0
            )
        probabilities = module.predict_last_block(
            np.load(activations / "activations.npy", mmap_mode="r"),
            directory / "last-block.pt",
            source,
            batch_size=3,
        )
        rows = list(csv.DictReader((directory / "predictions.csv").open()))
        for index, row in enumerate(rows, start=6):
            np.testing.assert_allclose(
                [float(row[f"p_{name}"]) for name in module.CLASS_NAMES],
                probabilities[index],
                atol=1e-7,
            )
        assert (
            sha256_file(heads / "runs" / f"seed-{run['seed']}" / "head.pt")
            == before_heads[run["seed"]]
        )


def test_epoch_resume_matches_uninterrupted_optimizer_state(experiment, tmp_path, monkeypatch):
    features, activations, heads, source, _signature = experiment
    original_save = module._atomic_torch

    def interrupt_after_epoch_one(path, value):
        original_save(path, value)
        if path.name == "resume.pt" and value["history"][-1]["epoch"] == 1:
            raise KeyboardInterrupt("synthetic interruption after a committed epoch")

    monkeypatch.setattr(module, "_atomic_torch", interrupt_after_epoch_one)
    settings = {"seeds": (42,), "threads": 1, "max_epochs": 2, "patience": 3, "batch_size": 3}
    with pytest.raises(KeyboardInterrupt):
        module.train_last_block(
            features, activations, heads, source, tmp_path / "resumed", **settings
        )
    monkeypatch.setattr(module, "_atomic_torch", original_save)
    resumed = module.train_last_block(
        features, activations, heads, source, tmp_path / "resumed", **settings
    )
    uninterrupted = module.train_last_block(
        features, activations, heads, source, tmp_path / "full", **settings
    )
    assert resumed["runs"][0]["epochs_run"] == uninterrupted["runs"][0]["epochs_run"] == 2
    first = torch.load(
        tmp_path / "resumed/runs/seed-42/resume.pt", map_location="cpu", weights_only=True
    )
    second = torch.load(
        tmp_path / "full/runs/seed-42/resume.pt", map_location="cpu", weights_only=True
    )
    for key in first["state_dict"]:
        torch.testing.assert_close(
            first["state_dict"][key], second["state_dict"][key], rtol=0, atol=0
        )
    torch.testing.assert_close(first["generator_state"], second["generator_state"], rtol=0, atol=0)
    for index, state in first["optimizer"]["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(
                value, second["optimizer"]["state"][index][key], rtol=0, atol=0
            )


def test_training_refuses_test_scope_and_activation_index_mismatch(experiment, tmp_path):
    features, activations, heads, source, _signature = experiment
    path = activations / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["scope"] = "heldout_inference"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="forbidden"):
        module.train_last_block(features, activations, heads, source, tmp_path / "rejected")
    metadata["scope"] = "train_validation"
    with np.load(activations / "index.npz", allow_pickle=False) as archive:
        index = {key: archive[key] for key in archive.files}
    index["labels"] = (index["labels"] + 1) % 3
    np.savez(activations / "index.npz", **index)
    metadata["outputs_sha256"]["index.npz"] = sha256_file(activations / "index.npz")
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="not aligned"):
        module.train_last_block(features, activations, heads, source, tmp_path / "rejected")


def test_bn_buffers_stay_frozen_while_affine_parameters_train():
    model = module.LastBlockClassifier(TinyBlock())
    before = module._state_hash(module._bn_state(model))
    model.train()
    assert model.block.bn.training is False
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss = nn.functional.cross_entropy(model(torch.randn(3, 512, 1, 2, 2)), torch.tensor([0, 1, 2]))
    loss.backward()
    assert model.block.bn.weight.grad is not None
    optimizer.step()
    assert module._state_hash(module._bn_state(model)) == before
