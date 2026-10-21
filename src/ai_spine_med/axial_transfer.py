"""Last-block axial transfer with fixed old attention heads and sealed evaluation.

Training reads only new TRAIN/validation and a fixed old-TRAIN replay sample.
TEST activation files are opened exclusively by the separate evaluation function.
All per-study outputs and model files are private research artifacts.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

from .assets import sha256_file, write_json
from .axial_features import WEIGHTS_SHA256
from .multimodal import MultimodalHead, architecture, load_multimodal_checkpoint
from .training import evaluate_predictions, inverse_sqrt_class_weights

FAMILY = "axial_transfer_v1"
SEEDS = (42, 2024, 3407)
CLASSES = ("normal_mild", "moderate", "severe")
LEVELS = ("L4-L5", "L5-S1")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_NORMALIZERS = ("sag_mean", "sag_std", "axial_mean", "axial_std")


def _device(device):
    if device == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if device not in {"cpu", "mps"}:
        raise ValueError("Transfer supports cpu, mps or auto")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available")
    return device


def _safe(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Cache/checkpoint paths must be relative")
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("Cache/checkpoint path escapes its root")
    return result


@dataclass(frozen=True)
class ActivationRecord:
    dataset: str
    study_id: str
    split: str
    labels: tuple[int | None, int | None]
    status: str
    reason: str
    path: Path | None
    sha256: str | None

    @property
    def key(self):
        return self.dataset, self.study_id

    @property
    def fitting_eligible(self):
        return self.status == "ok" and any(label is not None for label in self.labels)


def read_activation_index(path: str | Path) -> tuple[list[ActivationRecord], str]:
    """Read identities/provenance only; this function never reads activation bytes."""
    path = Path(path)
    source = json.loads(path.read_text())
    if (
        not isinstance(source, dict)
        or source.get("schema_version") != 1
        or not isinstance(source.get("records"), list)
        or not source["records"]
    ):
        raise ValueError("Require a nonempty schema-1 activation index")
    result, seen = [], set()
    for row in source["records"]:
        if not isinstance(row, dict):
            raise ValueError("Each activation record must be an object")
        labels = row.get("labels")
        if (
            row.get("dataset") not in {"aisslab", "rsna"}
            or not isinstance(row.get("study_id"), str)
            or not row["study_id"]
            or row.get("split") not in {"train", "validation", "test"}
            or row.get("status") not in {"ok", "not_assessed"}
            or not isinstance(row.get("reason"), str)
            or not isinstance(labels, list)
            or len(labels) != 2
            or any(
                label is not None and (type(label) is not int or label not in (0, 1, 2))
                for label in labels
            )
        ):
            raise ValueError("Invalid activation record identity, split, labels or status")
        key = row["dataset"], row["study_id"]
        if key in seen:
            raise ValueError("A dataset/study occurs more than once or crosses splits")
        seen.add(key)
        array_path, digest = None, None
        if row["status"] == "ok":
            digest = row.get("arrays_sha256")
            if row["reason"] or not isinstance(digest, str) or not _HASH.fullmatch(digest):
                raise ValueError("Assessed activation record needs an empty reason and a checksum")
            array_path = _safe(path.parent, row.get("arrays_path"))
            if array_path.suffix != ".npz":
                raise ValueError("Expected an NPZ activation file")
        elif not row["reason"] or row.get("arrays_path") or row.get("arrays_sha256"):
            raise ValueError("A not-assessed record needs a reason and no feature arrays")
        result.append(
            ActivationRecord(
                *key, row["split"], tuple(labels), row["status"], row["reason"], array_path, digest
            )
        )
    return result, sha256_file(path)


def load_record(record: ActivationRecord, *, verify=True):
    if record.status != "ok" or record.path is None:
        raise ValueError("Cannot load a not-assessed activation record")
    if verify and sha256_file(record.path) != record.sha256:
        raise ValueError("Activation checksum mismatch")
    with np.load(record.path, allow_pickle=False) as archive:
        if set(archive.files) != {"activations", "positions"}:
            raise ValueError("Unexpected activation array keys")
        activations, positions = archive["activations"], archive["positions"]
    n = len(activations)
    if (
        not 1 <= n <= 96
        or activations.shape != (n, 512, 7, 7)
        or activations.dtype != np.float32
        or positions.shape != (n, 2)
        or positions.dtype != np.float32
        or not np.isfinite(activations).all()
        or not np.isfinite(positions).all()
        or np.any(positions < -1e-6)
        or np.any(positions > 1 + 1e-6)
    ):
        raise ValueError("Invalid last-block activation shape, type, position or finite values")
    return torch.from_numpy(activations), torch.from_numpy(positions)


def _original_encoder(weights: str | Path):
    if sha256_file(Path(weights)) != WEIGHTS_SHA256:
        raise ValueError("Require the original pinned ImageNet ResNet18 weights")
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    model.fc = nn.Identity()
    return model.eval().requires_grad_(False)


def build_frozen_prefix(weights: str | Path, device="cpu") -> nn.Module:
    """Original ImageNet encoder up to, but excluding, layer4[1]."""
    original = _original_encoder(weights)
    prefix = nn.Sequential(
        original.conv1,
        original.bn1,
        original.relu,
        original.maxpool,
        original.layer1,
        original.layer2,
        original.layer3,
        original.layer4[0],
    )
    return prefix.eval().requires_grad_(False).to(_device(device))


class AxialTransferModel(nn.Module):
    def __init__(self, block: nn.Module, head: MultimodalHead):
        super().__init__()
        if head.variant != "axial":
            raise ValueError("Transfer requires the original axial attention/head architecture")
        self.block = block
        self.head = head
        self.block.requires_grad_(True)
        self.head.requires_grad_(True)
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        # BatchNorm affine parameters may learn; all running statistics stay fixed.
        for layer in self.block.modules():
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                layer.eval()
        return self

    def forward(self, activations, positions, mask=None):
        if (
            activations.ndim != 4
            or activations.shape[1:] != (512, 7, 7)
            or not 1 <= len(activations) <= 96
        ):
            raise ValueError("Require one study with 1..96 activations [N,512,7,7]")
        if positions.shape != (len(activations), 2):
            raise ValueError("Position shape differs from activation slices")
        if mask is None:
            mask = torch.ones(len(activations), dtype=torch.bool, device=activations.device)
        if mask.shape != (len(activations),) or mask.dtype != torch.bool or not mask.any():
            raise ValueError("Require a nonempty boolean slice mask")
        features = F.adaptive_avg_pool2d(self.block(activations), 1).flatten(1)
        dummy = torch.zeros((1, 2, 512), dtype=features.dtype, device=features.device)
        return self.head(dummy, features[None], positions[None], mask[None])[0]


def predict_record(model: AxialTransferModel, activations, positions, mask=None) -> np.ndarray:
    device = next(model.parameters()).device
    a = torch.as_tensor(activations, dtype=torch.float32, device=device)
    p = torch.as_tensor(positions, dtype=torch.float32, device=device)
    m = None if mask is None else torch.as_tensor(mask, dtype=torch.bool, device=device)
    model.eval()
    with torch.inference_mode():
        probabilities = model(a, p, m).softmax(-1).cpu().numpy()
    if (
        probabilities.shape != (2, 3)
        or not np.isfinite(probabilities).all()
        or not np.allclose(probabilities.sum(-1), 1, rtol=0, atol=1e-6)
    ):
        raise RuntimeError("Invalid transferred-model probabilities")
    return probabilities


def _state_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _buffers(model):
    return {key: value.detach().cpu().clone() for key, value in model.named_buffers()}


def _assert_buffers(model, before):
    current = _buffers(model)
    if current.keys() != before.keys() or any(
        not torch.equal(current[k], before[k]) for k in before
    ):
        raise RuntimeError("Frozen BatchNorm statistics or original head normalizers changed")


def _parameter_delta(model, initial, *, prefix=""):
    return max(
        float((value.detach().cpu() - initial[key]).abs().max())
        for key, value in model.named_parameters()
        if key.startswith(prefix)
    )


def _new_model(original, head_path, seed):
    head, header = load_multimodal_checkpoint(head_path)
    if (
        header.get("variant") != "axial"
        or header.get("seed") != seed
        or header.get("fold") is not None
    ):
        raise ValueError("Require each seed's final original axial head, not a CV checkpoint")
    return AxialTransferModel(copy.deepcopy(original.layer4[1]), head)


def _save_checkpoint(path, model, *, seed, epoch, phase, provenance):
    torch.save(
        {
            "schema_version": 1,
            "model_family": FAMILY,
            "architecture": architecture("axial"),
            "classes": list(CLASSES),
            "levels": list(LEVELS),
            "seed": seed,
            "best_epoch": epoch,
            "phase": phase,
            "state_dict": _state(model),
            **provenance,
        },
        path,
    )


def load_transfer_checkpoint(path: str | Path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("schema_version") != 1
        or checkpoint.get("model_family") != FAMILY
        or checkpoint.get("architecture") != architecture("axial")
        or checkpoint.get("classes") != list(CLASSES)
        or checkpoint.get("levels") != list(LEVELS)
        or checkpoint.get("seed") not in SEEDS
        or checkpoint.get("phase") not in {"before", "after"}
        or type(checkpoint.get("best_epoch")) is not int
        or not 0 <= checkpoint["best_epoch"] <= 8
        or (checkpoint.get("phase") == "before" and checkpoint["best_epoch"] != 0)
        or checkpoint.get("original_encoder_weights_sha256") != WEIGHTS_SHA256
        or any(
            not isinstance(checkpoint.get(key), str) or not _HASH.fullmatch(checkpoint[key])
            for key in ("protocol_sha256", "training_index_sha256", "initial_head_sha256")
        )
    ):
        raise ValueError("Invalid transfer checkpoint architecture or provenance")
    state = checkpoint.get("state_dict", {})
    try:
        head = MultimodalHead("axial", {name: state[f"head.{name}"] for name in _NORMALIZERS})
        model = AxialTransferModel(resnet18(weights=None).layer4[1], head)
        expected = model.state_dict()
        if state.keys() != expected.keys() or any(
            not isinstance(value, torch.Tensor)
            or value.dtype != expected[key].dtype
            or value.shape != expected[key].shape
            or not torch.isfinite(value).all()
            for key, value in state.items()
        ):
            raise ValueError("Invalid or nonfinite transfer state")
        model.load_state_dict(state, strict=True)
    except (KeyError, RuntimeError, TypeError) as error:
        raise ValueError("Transfer checkpoint tensor shapes or keys differ") from error
    return model.to(_device(device)).eval(), checkpoint


def _validate_protocol(protocol):
    exact = {
        "seeds": list(SEEDS),
        "classes": list(CLASSES),
        "levels": list(LEVELS),
        "max_epochs": 8,
        "patience": 3,
        "learning_rate_block": 1e-5,
        "learning_rate_head": 1e-4,
        "weight_decay": 1e-4,
        "batch_size": 1,
        "replay_ratio": 1.0,
        "replay_seed": 2026,
    }
    if not isinstance(protocol, dict):
        raise ValueError("Require a typed fixed transfer protocol")
    for key, expected in exact.items():
        if protocol.get(key) != expected or type(protocol.get(key)) is not type(expected):
            raise ValueError(f"Fixed transfer protocol mismatch: {key}")


def select_training_records(records):
    new_train = sorted(
        [
            r
            for r in records
            if r.dataset == "aisslab" and r.split == "train" and r.fitting_eligible
        ],
        key=lambda r: r.key,
    )
    validation = sorted(
        [r for r in records if r.dataset == "aisslab" and r.split == "validation"],
        key=lambda r: r.key,
    )
    old_train = sorted(
        [r for r in records if r.dataset == "rsna" and r.split == "train" and r.fitting_eligible],
        key=lambda r: r.key,
    )
    if (
        not new_train
        or not validation
        or not any(r.fitting_eligible for r in validation)
        or not old_train
    ):
        raise ValueError(
            "Require new TRAIN/validation and original TRAIN replay with labeled targets"
        )
    selected = np.random.default_rng(2026).choice(
        len(old_train), min(len(new_train), len(old_train)), replace=False
    )
    replay = [old_train[i] for i in sorted(selected.tolist())]
    return new_train + replay, validation, replay


def _verify_records(records):
    for record in records:
        if record.status == "ok" and sha256_file(record.path) != record.sha256:
            raise ValueError("Selected TRAIN/validation activation cache changed")


def _prediction_rows(record, probabilities):
    result = []
    for level, label, index in zip(LEVELS, record.labels, range(2)):
        ok = record.status == "ok"
        row = {
            "dataset": record.dataset,
            "study_id": record.study_id,
            "level": level,
            "split": record.split,
            "label": label,
            "reference_status": "missing" if label is None else "available",
            "status": "ok" if ok else "not_assessed",
            "reason": "" if ok else record.reason,
            "prediction": None,
            **{f"p_{name}": None for name in CLASSES},
        }
        if ok:
            p = probabilities[index]
            row.update(
                prediction=int(p.argmax()),
                **{f"p_{name}": float(value) for name, value in zip(CLASSES, p)},
            )
        result.append(row)
    return result


def _assess(model, records):
    rows = []
    for record in records:
        probability = None
        if record.status == "ok":
            probability = predict_record(model, *load_record(record, verify=False))
        rows.extend(_prediction_rows(record, probability))
    metrics = evaluate_predictions(rows)
    assessed = [r for r in rows if r["status"] == "ok" and r["label"] is not None]
    loss = (
        -np.mean([math.log(max(row[f"p_{CLASSES[row['label']]}"], 1e-30)) for row in assessed])
        if assessed
        else math.nan
    )
    if (
        not assessed
        or not math.isfinite(metrics["conditional"]["macro_f1"])
        or not math.isfinite(loss)
    ):
        raise ValueError("New validation must contain finite assessed reference targets")
    return metrics, float(loss)


def _write_predictions(path, rows):
    fields = (
        "dataset",
        "study_id",
        "level",
        "split",
        "label",
        "reference_status",
        "status",
        "reason",
        "prediction",
        *(f"p_{name}" for name in CLASSES),
    )
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _weighted_target_loss(logits, targets, class_weights):
    """Average over known targets, preserving class weights for a single study."""
    return F.cross_entropy(logits, targets, weight=class_weights, reduction="sum") / len(targets)


def train_axial_transfer(
    index,
    protocol,
    out,
    *,
    head_dir="reports/runs/multimodal-v3/validation/axial",
    weights=".cache/axial-v3/resnet18-f37072fd.pth",
    device="auto",
    threads=4,
):
    """Warm-start three independent last-block models; seal only after all checks."""
    index, protocol, out, head_dir, weights = map(Path, (index, protocol, out, head_dir, weights))
    config = json.loads(protocol.read_text())
    _validate_protocol(config)
    if out.exists():
        raise ValueError("Output exists; never overwrite transfer experiments")
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    chosen_device = _device(device)
    records, index_sha = read_activation_index(index)
    fitting, validation, replay = select_training_records(records)
    used = fitting + validation
    _verify_records(used)  # Never hashes or loads any TEST array.
    labels = np.asarray(
        [label for r in fitting for label in r.labels if label is not None], np.int64
    )
    class_weights = inverse_sqrt_class_weights(labels)
    original = _original_encoder(weights)
    original_hash = _state_hash(original.state_dict())
    head_hashes = {str(seed): sha256_file(head_dir / f"seed-{seed}" / "model.pt") for seed in SEEDS}
    protocol_sha = sha256_file(protocol)
    implementation_sha = sha256_file(Path(__file__))
    torch.set_num_threads(threads)
    out.mkdir(parents=True)
    provenance = {
        "protocol_sha256": protocol_sha,
        "implementation_sha256": implementation_sha,
        "training_index_sha256": index_sha,
        "original_encoder_weights_sha256": WEIGHTS_SHA256,
    }
    write_json(
        out / "manifest.json",
        {
            "schema_version": 1,
            "model_family": FAMILY,
            "configuration": config,
            "device": chosen_device,
            "threads": threads,
            "initial_heads_sha256": head_hashes,
            "class_weights": class_weights.tolist(),
            "fitting_class_counts": np.bincount(labels, minlength=3).tolist(),
            "fit_keys": [list(r.key) for r in fitting],
            "validation_keys": [list(r.key) for r in validation],
            "replay_keys": [list(r.key) for r in replay],
            **provenance,
        },
    )
    runs = []
    for seed in SEEDS:
        began = time.perf_counter()
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        directory = out / "runs" / f"seed-{seed}"
        directory.mkdir(parents=True)
        head_path = head_dir / f"seed-{seed}" / "model.pt"
        model = _new_model(original, head_path, seed).to(chosen_device)
        initial, frozen_buffers = _state(model), _buffers(model)
        run_provenance = {**provenance, "initial_head_sha256": head_hashes[str(seed)]}
        _save_checkpoint(
            directory / "before.pt",
            model,
            seed=seed,
            epoch=0,
            phase="before",
            provenance=run_provenance,
        )
        before_metrics, before_loss = _assess(model, validation)
        best_score, best_epoch, stale, best_state = (
            before_metrics["conditional"]["macro_f1"],
            0,
            0,
            initial,
        )
        history = [
            {
                "epoch": 0,
                "training_loss": None,
                "validation_loss_unweighted": before_loss,
                "validation_macro_f1": best_score,
                "seconds": 0.0,
            }
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": model.block.parameters(), "lr": 1e-5},
                {"params": model.head.parameters(), "lr": 1e-4},
            ],
            weight_decay=1e-4,
        )
        loss_weights = torch.from_numpy(class_weights).to(chosen_device)
        for epoch in range(1, 9):
            started = time.perf_counter()
            model.train()
            numerator, denominator = 0.0, 0.0
            for step, position in enumerate(rng.permutation(len(fitting))):
                record = fitting[position]
                a, p = load_record(record, verify=False)
                a, p = a.to(chosen_device), p.to(chosen_device)
                valid = torch.tensor(
                    [v is not None for v in record.labels], dtype=torch.bool, device=chosen_device
                )
                y = torch.tensor(
                    [v for v in record.labels if v is not None],
                    dtype=torch.long,
                    device=chosen_device,
                )
                optimizer.zero_grad(set_to_none=True)
                loss = _weighted_target_loss(model(a, p)[valid], y, loss_weights)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite transfer training loss")
                loss.backward()
                optimizer.step()
                divisor = len(y)
                numerator += float(loss.detach().cpu()) * divisor
                denominator += divisor
                if (step + 1) % 50 == 0:
                    print(
                        f"axial-transfer seed={seed} epoch={epoch} study={step + 1}/{len(fitting)} seconds={time.perf_counter() - started:.1f}",
                        flush=True,
                    )
            _assert_buffers(model, frozen_buffers)
            metrics, held_loss = _assess(model, validation)
            score = metrics["conditional"]["macro_f1"]
            history.append(
                {
                    "epoch": epoch,
                    "training_loss": numerator / denominator,
                    "validation_loss_unweighted": held_loss,
                    "validation_macro_f1": score,
                    "seconds": time.perf_counter() - started,
                }
            )
            if score > best_score + 1e-12:
                best_score, best_epoch, stale, best_state = score, epoch, 0, _state(model)
            else:
                stale += 1
            write_json(directory / "history.json", history)
            print(
                f"axial-transfer seed={seed} epoch={epoch} new_validation_macro_f1={score:.6f} best_epoch={best_epoch} seconds={history[-1]['seconds']:.1f}",
                flush=True,
            )
            if stale >= 3:
                break
        trained_delta = _parameter_delta(model, initial)
        trained_block_delta = _parameter_delta(model, initial, prefix="block.")
        if not trained_delta > 0 or not trained_block_delta > 0:
            raise RuntimeError("Last-block transfer parameters did not change during training")
        model.load_state_dict(best_state, strict=True)
        _assert_buffers(model, frozen_buffers)
        selected_delta = _parameter_delta(model, initial)
        selected_block_delta = _parameter_delta(model, initial, prefix="block.")
        _save_checkpoint(
            directory / "model.pt",
            model,
            seed=seed,
            epoch=best_epoch,
            phase="after",
            provenance=run_provenance,
        )
        reopened, _ = load_transfer_checkpoint(directory / "model.pt", device=chosen_device)
        probe = load_record(fitting[0], verify=False)
        parity = float(
            np.max(np.abs(predict_record(model, *probe) - predict_record(reopened, *probe)))
        )
        if parity > 1e-6:
            raise RuntimeError("Transfer checkpoint roundtrip predictions changed")
        final_metrics, _ = _assess(reopened, validation)
        if (
            final_metrics["conditional"]["macro_f1"] + 1e-12
            < before_metrics["conditional"]["macro_f1"]
        ):
            raise RuntimeError("Selected model is worse than the epoch-0 candidate")
        run = {
            "seed": seed,
            "best_epoch": best_epoch,
            "epochs_run": len(history) - 1,
            "checkpoint": f"runs/seed-{seed}/model.pt",
            "checkpoint_sha256": sha256_file(directory / "model.pt"),
            "before_checkpoint": f"runs/seed-{seed}/before.pt",
            "before_checkpoint_sha256": sha256_file(directory / "before.pt"),
            "before_metrics": before_metrics,
            "metrics": final_metrics,
            "training_parameter_delta_max_abs": trained_delta,
            "block_parameter_delta_max_abs": trained_block_delta,
            "selected_parameter_delta_max_abs": selected_delta,
            "selected_block_parameter_delta_max_abs": selected_block_delta,
            "frozen_buffers_unchanged": True,
            "checkpoint_roundtrip_max_probability_error": parity,
            "seconds": time.perf_counter() - began,
        }
        write_json(directory / "result.json", run)
        runs.append(run)
        del model, reopened, optimizer
    _verify_records(used)
    if (
        sha256_file(index) != index_sha
        or sha256_file(Path(__file__)) != implementation_sha
        or sha256_file(protocol) != protocol_sha
        or sha256_file(weights) != WEIGHTS_SHA256
        or _state_hash(original.state_dict()) != original_hash
        or any(
            sha256_file(head_dir / f"seed-{seed}" / "model.pt") != head_hashes[str(seed)]
            for seed in SEEDS
        )
    ):
        raise RuntimeError("Training inputs, frozen encoder or original heads changed")
    summary = {
        "schema_version": 1,
        "status": "complete",
        "model_family": FAMILY,
        "seeds": list(SEEDS),
        "configuration": config,
        "protocol_sha256": protocol_sha,
        "implementation_sha256": implementation_sha,
        "index_sha256": index_sha,
        "device": chosen_device,
        "class_weights": class_weights.tolist(),
        "fitting_class_counts": np.bincount(labels, minlength=3).tolist(),
        "n_fitting_studies": len(fitting),
        "n_replay_studies": len(replay),
        "n_new_validation_studies": len(validation),
        "runs": runs,
        "frozen_encoder_unchanged": True,
        "old_heads_unchanged": True,
        "test_evaluated": False,
    }
    write_json(out / "summary.json", summary)
    selection = {
        "schema_version": 1,
        "model_family": FAMILY,
        "frozen": True,
        "seeds": list(SEEDS),
        "protocol_sha256": protocol_sha,
        "training_index_sha256": index_sha,
        "implementation_sha256": implementation_sha,
        "training_summary_sha256": sha256_file(out / "summary.json"),
        "fit_keys": [list(r.key) for r in fitting],
        "validation_keys": [list(r.key) for r in validation],
        "forbidden_evaluation_keys": [
            list(r.key) for r in records if r.split in {"train", "validation"}
        ],
        "runs": [
            {
                key: run[key]
                for key in (
                    "seed",
                    "best_epoch",
                    "checkpoint",
                    "checkpoint_sha256",
                    "before_checkpoint",
                    "before_checkpoint_sha256",
                )
            }
            for run in runs
        ],
    }
    write_json(out / "selection.json", selection)
    return summary


def _sealed_models(run_dir, device):
    root = Path(run_dir)
    selection = json.loads((root / "selection.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    if (
        not isinstance(selection, dict)
        or not isinstance(summary, dict)
        or selection.get("schema_version") != 1
        or selection.get("model_family") != FAMILY
        or selection.get("frozen") is not True
        or selection.get("seeds") != list(SEEDS)
        or summary.get("status") != "complete"
        or summary.get("model_family") != FAMILY
        or summary.get("seeds") != list(SEEDS)
        or selection.get("training_summary_sha256") != sha256_file(root / "summary.json")
        or selection.get("training_index_sha256") != summary.get("index_sha256")
        or selection.get("protocol_sha256") != summary.get("protocol_sha256")
        or selection.get("implementation_sha256") != summary.get("implementation_sha256")
    ):
        raise ValueError("Require an unchanged completed three-seed training selection")
    runs = selection.get("runs", [])
    summary_runs = summary.get("runs", [])
    if (
        not isinstance(runs, list)
        or not isinstance(summary_runs, list)
        or len(runs) != 3
        or len(summary_runs) != 3
        or any(not isinstance(r, dict) for r in runs + summary_runs)
        or [r.get("seed") for r in runs] != list(SEEDS)
        or [r.get("seed") for r in summary_runs] != list(SEEDS)
    ):
        raise ValueError("All three selected seeds are required")
    groups = {}
    for name in ("fit_keys", "validation_keys", "forbidden_evaluation_keys"):
        entries = selection.get(name)
        if (
            not isinstance(entries, list)
            or not entries
            or any(
                not isinstance(k, list)
                or len(k) != 2
                or k[0] not in {"aisslab", "rsna"}
                or not isinstance(k[1], str)
                or not k[1]
                for k in entries
            )
        ):
            raise ValueError("Invalid fitting/validation identity inventory")
        groups[name] = {tuple(k) for k in entries}
        if len(groups[name]) != len(entries):
            raise ValueError("Duplicate fitting/validation identity")
    if (
        groups["fit_keys"] & groups["validation_keys"]
        or not (groups["fit_keys"] | groups["validation_keys"])
        <= groups["forbidden_evaluation_keys"]
    ):
        raise ValueError(
            "Fitting/validation identities overlap or are absent from the exclusion set"
        )
    models = {}
    for run, summary_run in zip(runs, summary_runs):
        if any(summary_run.get(key) != value for key, value in run.items()):
            raise ValueError("Selected checkpoint inventory differs from the training summary")
        initial_head_sha = None
        for phase, key in (("before", "before_checkpoint"), ("after", "checkpoint")):
            path = _safe(root, run[key])
            if sha256_file(path) != run[key + "_sha256"]:
                raise ValueError("A sealed transfer checkpoint changed")
            model, header = load_transfer_checkpoint(path, device=device)
            if (
                header["seed"] != run["seed"]
                or header["phase"] != phase
                or header["protocol_sha256"] != selection["protocol_sha256"]
                or header["training_index_sha256"] != selection["training_index_sha256"]
                or header.get("implementation_sha256") != selection["implementation_sha256"]
                or header["best_epoch"] != (0 if phase == "before" else run["best_epoch"])
                or (
                    initial_head_sha is not None
                    and header.get("initial_head_sha256") != initial_head_sha
                )
            ):
                raise ValueError("Selected checkpoint header differs from its seal")
            initial_head_sha = header.get("initial_head_sha256")
            models[phase, run["seed"]] = model
    return selection, models


def evaluate_axial_transfer(
    index, run_dir, out, *, splits=("test",), datasets=("aisslab", "rsna"), device="auto"
):
    """After sealing, evaluate a separate TEST-only index without refitting anything."""
    if tuple(splits) != ("test",) or not datasets or not set(datasets) <= {"aisslab", "rsna"}:
        raise ValueError("Evaluation accepts TEST only and the two known datasets")
    index, out = Path(index), Path(out)
    if out.exists():
        raise ValueError("Evaluation output exists; preserve completed results")
    selection, models = _sealed_models(run_dir, _device(device))
    records, index_sha = read_activation_index(index)
    forbidden = {
        tuple(key)
        for key in selection["forbidden_evaluation_keys"]
        + selection["fit_keys"]
        + selection["validation_keys"]
    }
    if any(r.split != "test" or r.dataset not in datasets or r.key in forbidden for r in records):
        raise ValueError(
            "TEST evaluation overlaps fitting/validation or contains a forbidden cohort"
        )
    _verify_records(records)
    tables = {f"{phase}_seed_{seed}": [] for phase in ("before", "after") for seed in SEEDS}
    tables.update(before_ensemble=[], after_ensemble=[])
    for record in records:
        values = {}
        if record.status == "ok":
            arrays = load_record(record, verify=False)
            for (phase, seed), model in models.items():
                values[phase, seed] = predict_record(model, *arrays)
        for phase in ("before", "after"):
            for seed in SEEDS:
                tables[f"{phase}_seed_{seed}"].extend(
                    _prediction_rows(record, values.get((phase, seed)))
                )
            mean = (
                np.mean([values[phase, seed].astype(np.float64) for seed in SEEDS], axis=0)
                if values
                else None
            )
            tables[f"{phase}_ensemble"].extend(_prediction_rows(record, mean))
    _verify_records(records)
    if sha256_file(index) != index_sha:
        raise RuntimeError("TEST activation index changed during evaluation")
    # Recheck all six files and the training-summary seal before publishing results.
    _sealed_models(run_dir, "cpu")
    out.mkdir(parents=True)
    entries = {}
    for name, rows in tables.items():
        path = out / f"{name}.csv"
        _write_predictions(path, rows)
        metrics = evaluate_predictions(rows)
        metrics["scope"] = (
            "Fixed before/after TEST evaluation under assumed class mapping; no model selection; report the two datasets separately"
        )
        by_dataset = {}
        for dataset in sorted({r.dataset for r in records}):
            group = evaluate_predictions([r for r in rows if r["dataset"] == dataset])
            group["scope"] = (
                "Fixed before/after TEST evaluation under assumed class mapping; no model selection; new external held-out patients"
                if dataset == "aisslab"
                else "Fixed before/after TEST evaluation; no model selection; previously opened RSNA TEST regression comparison, not a fresh independent test"
            )
            by_dataset[dataset] = group
        entries[name] = {
            "metrics": metrics,
            "by_dataset": by_dataset,
            "predictions_sha256": sha256_file(path),
        }
    summary = {
        "schema_version": 1,
        "model_family": FAMILY,
        "selection_sha256": sha256_file(Path(run_dir) / "selection.json"),
        "evaluation_index_sha256": index_sha,
        "training_index_sha256": selection["training_index_sha256"],
        "entries": entries,
        "training_performed": False,
        "scope": "Fixed before/after models; new external held-out patients and already-opened RSNA TEST are reported separately; no new independent claim for RSNA",
    }
    write_json(out / "summary.json", summary)
    return summary
