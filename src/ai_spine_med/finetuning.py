"""Fine-tune only the final SpineNet residual block and the three-class head.

The preceding encoder is represented by immutable, verified cached activations.
BatchNorm running statistics remain frozen, including in the trainable block.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import platform
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .assets import sha256_file, source_dir, write_json
from .baseline import import_upstream
from .data import CLASS_NAMES
from .features import _source_signature
from .training import (
    SEEDS,
    FeatureData,
    _prediction_rows,
    _write_predictions,
    evaluate_predictions,
    inverse_sqrt_class_weights,
    load_feature_data,
    predict_head,
)

ACTIVATION_SHAPE = (512, 9, 7, 14)
MODEL_NAME = "SpineNetLayer4LastBlock+Linear(512,3)"
SCHEMA_VERSION = 1
PARITY_ATOL = 1e-4
PARITY_RTOL = 1e-5
INDEX_KEYS = ("study_id", "series_id", "level", "split", "labels")


def _hash_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        torch.save(value, stream)
    temporary.replace(path)


def _cpu_state(state: dict) -> dict:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def _state_hash(state: dict) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _bn_state(module: nn.Module) -> dict:
    return {
        f"{name}.{key}": value.detach().cpu().clone()
        for name, layer in module.named_modules()
        if isinstance(layer, nn.modules.batchnorm._BatchNorm)
        for key, value in layer.named_buffers(recurse=False)
    }


def _resolve_device(device: str) -> str:
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device not in {"cpu", "mps"}:
        raise ValueError("Last-block work supports cpu, mps or auto")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable in this process; use an authorized GPU process or cpu")
    return device


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def _original_weights(cache_dir: Path) -> tuple[dict, dict]:
    signature = _source_signature(cache_dir)
    path = source_dir(cache_dir) / "spinenet/weights/grading/ckpt1.pt"
    with torch.serialization.safe_globals(
        [np.core.multiarray.scalar, np.dtype, type(np.dtype(np.float64))]
    ):
        state = torch.load(path, map_location="cpu", weights_only=True)["model_weights"]
    return state, signature


def _load_original_grading(cache_dir: str | Path, device: str = "cpu") -> nn.Module:
    cache_dir = Path(cache_dir)
    state, _signature = _original_weights(cache_dir)
    import_upstream(cache_dir)
    from spinenet.models.grading import GradingModel

    model = GradingModel()
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False).to(device)


def _original_block(cache_dir: Path) -> tuple[nn.Module, dict]:
    state, signature = _original_weights(cache_dir)
    import_upstream(cache_dir)
    from spinenet.models.grading import BasicBlock

    block = BasicBlock(512, 512)
    prefix = "layer4.2."
    block.load_state_dict(
        {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)},
        strict=True,
    )
    return block.eval(), signature


class LastBlockClassifier(nn.Module):
    """Train the residual block/affine BN parameters while freezing BN buffers."""

    def __init__(self, block: nn.Module, head: nn.Linear | None = None):
        super().__init__()
        self.block = block
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.head = head if head is not None else nn.Linear(512, 3)

    def train(self, mode: bool = True):
        super().train(mode)
        for layer in self.modules():
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                layer.eval()
        return self

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        features = torch.flatten(self.pool(self.block(activation)), 1)
        return self.head(features)


def _capture_activations(model: nn.Module, rois: np.ndarray, device: str):
    activations, features = [], []
    first = model.layer4[-1].register_forward_pre_hook(
        lambda _module, args: activations.append(args[0].detach().cpu().numpy().copy())
    )
    second = model.fc_ccs.register_forward_pre_hook(
        lambda _module, args: features.append(args[0].detach().cpu().numpy().copy())
    )
    try:
        with torch.inference_mode():
            output = model(torch.from_numpy(np.asarray(rois, dtype=np.float32))[:, None].to(device))
            logits = output[2].detach().cpu().numpy().copy()
    finally:
        first.remove()
        second.remove()
    activation = np.concatenate(activations).astype(np.float32)
    feature = np.concatenate(features).astype(np.float32)
    if (
        activation.shape != (len(rois), *ACTIVATION_SHAPE)
        or feature.shape != (len(rois), 512)
        or logits.shape != (len(rois), 4)
    ):
        raise ValueError("Unexpected original last-block activation/feature/head shape")
    if not all(np.isfinite(value).all() for value in (activation, feature, logits)):
        raise ValueError("Nonfinite original grading output")
    return activation, feature, logits


def _target_rows(data: FeatureData) -> list[dict]:
    rows: list[dict | None] = [None] * len(data.arrays["features"])
    for target in data.targets:
        index = target["feature_index"]
        if index is not None:
            if rows[index] is not None:
                raise ValueError("Duplicate activation target index")
            rows[index] = target
    if any(row is None for row in rows):
        raise ValueError("A frozen feature has no matching ROI target")
    return rows


def _roi_batch(
    feature_dir: Path, data: FeatureData, targets: list[dict], indices: np.ndarray, verified: dict
) -> np.ndarray:
    loaded, rois = {}, []
    for index in indices:
        row = targets[int(index)]
        path = (feature_dir / row["roi_path"]).resolve()
        if not path.is_relative_to(feature_dir.resolve()):
            raise ValueError("ROI path escapes source feature cache")
        record_path = feature_dir / "records" / f"{row['study_id']}_{row['series_id']}.json"
        if str(path) not in verified:
            record = json.loads(record_path.read_text())
            if (
                record["cache_id"] != data.metadata["cache_id"]
                or sha256_file(path) != record["arrays_sha256"]
            ):
                raise ValueError("Original ROI record provenance/checksum mismatch")
            verified[str(path)] = record["arrays_sha256"]
        if str(path) not in loaded:
            with np.load(path, allow_pickle=False) as archive:
                loaded[str(path)] = (archive["rois"], archive["levels"])
        volumes, levels = loaded[str(path)]
        roi_index = int(row["roi_index"])
        if str(levels[roi_index]) != row["level"]:
            raise ValueError("Original ROI level/index alignment mismatch")
        roi = volumes[roi_index]
        if roi.shape != (9, 112, 224) or not np.isfinite(roi).all():
            raise ValueError("Invalid original ROI")
        rois.append(roi)
    return np.stack(rois)


def _activation_config(
    data: FeatureData, cache_dir: Path, device: str, scope: str, selection_sha256: str | None
) -> dict:
    signature = _source_signature(cache_dir)
    recorded = data.metadata.get("config", {})
    for field in ("upstream_commit", "weights_sha256", "source_python_sha256"):
        if recorded.get(field) != signature.get(field):
            raise ValueError("Activation encoder differs from the original feature cache")
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": scope,
        "selection_sha256": selection_sha256,
        "feature_cache_sha256": data.file_sha256,
        "feature_cache_id": data.metadata["cache_id"],
        "source_signature": signature,
        "device": device,
        "activation_shape": list(ACTIVATION_SHAPE),
        "dtype": "float32",
        "boundary": "grading_model.layer4.2 input; preceding encoder frozen in eval mode",
        "parity_atol": PARITY_ATOL,
        "parity_rtol": PARITY_RTOL,
        "implementation_sha256": {
            function.__name__: hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
            for function in (
                _cache_activations,
                _capture_activations,
                _roi_batch,
                _target_rows,
                _activation_config,
                _load_original_grading,
                _original_weights,
            )
        },
    }


def _cache_activations(
    data: FeatureData,
    feature_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    *,
    device: str,
    batch_size: int,
    limit: int | None,
    scope: str,
    selection_sha256: str | None,
) -> dict:
    if batch_size < 1 or (limit is not None and limit < 1):
        raise ValueError("Batch size and limit must be positive")
    allowed = {"train", "validation"} if scope == "train_validation" else {"test"}
    if not len(data.arrays["features"]) or set(data.arrays["split"].tolist()) - allowed:
        raise ValueError("Activation cache contains forbidden splits or no assessed rows")
    device = _resolve_device(device)
    config = _activation_config(data, cache_dir, device, scope, selection_sha256)
    cache_id = _hash_json(config)
    metadata_path = output_dir / "metadata.json"
    count = len(data.arrays["features"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("cache_id") != cache_id:
            raise ValueError(
                "Existing activation cache has different source/configuration provenance"
            )
    else:
        if any(output_dir.iterdir()):
            raise ValueError("Activation directory has files without a valid resume manifest")
        np.savez(
            output_dir / "index.npz",
            **{key: data.arrays[key] for key in INDEX_KEYS},
            feature_index=np.arange(count, dtype=np.int64),
        )
        for name, dtype, shape in (
            ("activations.npy", np.float32, (count, *ACTIVATION_SHAPE)),
            ("completed.npy", np.bool_, (count,)),
            ("row_sha256.npy", "S64", (count,)),
        ):
            mapping = np.lib.format.open_memmap(
                output_dir / name, mode="w+", dtype=dtype, shape=shape
            )
            if name != "activations.npy":
                mapping[:] = 0 if name == "completed.npy" else b""
            mapping.flush()
            del mapping
        metadata = {
            **config,
            "cache_id": cache_id,
            "config": config,
            "extraction_complete": False,
            "outputs_sha256": {},
            "verified_features": 0,
            "maximum_feature_abs_error": 0.0,
            "maximum_original_logits_abs_error": 0.0,
        }
        write_json(metadata_path, metadata)
    activations = np.load(output_dir / "activations.npy", mmap_mode="r+")
    completed = np.load(output_dir / "completed.npy", mmap_mode="r+")
    row_hashes = np.load(output_dir / "row_sha256.npy", mmap_mode="r+")
    if activations.shape != (count, *ACTIVATION_SHAPE) or activations.dtype != np.float32:
        raise ValueError("Invalid activation memmap shape or dtype")
    if (
        completed.shape != (count,)
        or completed.dtype != np.bool_
        or row_hashes.shape != (count,)
        or row_hashes.dtype != np.dtype("S64")
    ):
        raise ValueError("Invalid activation completion/hash arrays")
    with np.load(output_dir / "index.npz", allow_pickle=False) as index:
        if any(
            not np.array_equal(index[key], data.arrays[key]) for key in INDEX_KEYS
        ) or not np.array_equal(index["feature_index"], np.arange(count)):
            raise ValueError("Activation index and frozen feature rows disagree")
    for index in np.flatnonzero(completed):
        digest = (
            hashlib.sha256(np.ascontiguousarray(activations[index]).tobytes()).hexdigest().encode()
        )
        if digest != row_hashes[index] or not np.isfinite(activations[index]).all():
            raise ValueError("Completed activation row is corrupt")
    pending = np.flatnonzero(~completed)
    if limit is not None:
        pending = pending[:limit]
    if len(pending):
        model = _load_original_grading(cache_dir, device)
        frozen_before = _state_hash(model.state_dict())
        targets, verified = _target_rows(data), {}
        for offset in range(0, len(pending), batch_size):
            indices = pending[offset : offset + batch_size]
            rois = _roi_batch(feature_dir, data, targets, indices, verified)
            values, features, logits = _capture_activations(model, rois, device)
            if not np.allclose(
                features, data.arrays["features"][indices], atol=PARITY_ATOL, rtol=PARITY_RTOL
            ) or not np.allclose(
                logits, data.arrays["original_logits"][indices], atol=PARITY_ATOL, rtol=PARITY_RTOL
            ):
                raise ValueError(
                    "Original suffix/feature/logit reconstruction failed parity validation"
                )
            metadata["maximum_feature_abs_error"] = max(
                metadata["maximum_feature_abs_error"],
                float(np.abs(features - data.arrays["features"][indices]).max()),
            )
            metadata["maximum_original_logits_abs_error"] = max(
                metadata["maximum_original_logits_abs_error"],
                float(np.abs(logits - data.arrays["original_logits"][indices]).max()),
            )
            activations[indices] = values
            activations.flush()
            for index in indices:
                row_hashes[index] = (
                    hashlib.sha256(np.ascontiguousarray(activations[index]).tobytes())
                    .hexdigest()
                    .encode()
                )
            row_hashes.flush()
            completed[indices] = True
            completed.flush()
            metadata["verified_features"] = int(completed.sum())
            write_json(metadata_path, metadata)
            print(f"Cached last-block inputs {metadata['verified_features']}/{count}", flush=True)
        if _state_hash(model.state_dict()) != frozen_before:
            raise RuntimeError("The frozen original grading encoder or BN buffers changed")
        del model
        if device == "mps":
            torch.mps.empty_cache()
    metadata["verified_features"] = int(completed.sum())
    metadata["extraction_complete"] = bool(completed.all())
    metadata["original_encoder_and_bn_unchanged"] = True
    if metadata["extraction_complete"]:
        metadata["outputs_sha256"] = {
            name: sha256_file(output_dir / name)
            for name in ("activations.npy", "index.npz", "completed.npy", "row_sha256.npy")
        }
    write_json(metadata_path, metadata)
    return metadata


def cache_last_block_activations(
    feature_dir: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    *,
    device: str = "mps",
    batch_size: int = 4,
    limit: int | None = None,
) -> dict:
    """Build/resume train/validation activations; never permit test rows."""
    data = load_feature_data(feature_dir)
    return _cache_activations(
        data,
        Path(feature_dir),
        Path(cache_dir),
        Path(output_dir),
        device=device,
        batch_size=batch_size,
        limit=limit,
        scope="train_validation",
        selection_sha256=None,
    )


def cache_inference_activations(
    data: FeatureData,
    feature_dir: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    selection_sha256: str,
    *,
    device: str = "mps",
    batch_size: int = 4,
) -> dict:
    """Accept only test data already validated by the sealed evaluation reader.

    This helper does not read a test manifest or labels itself. Its output scope
    is rejected by the trainer. The evaluation caller verifies the selection seal.
    """
    if len(selection_sha256) != 64 or any(c not in "0123456789abcdef" for c in selection_sha256):
        raise ValueError("A valid immutable selection SHA256 is required for heldout inference")
    return _cache_activations(
        data,
        Path(feature_dir),
        Path(cache_dir),
        Path(output_dir),
        device=device,
        batch_size=batch_size,
        limit=None,
        scope="heldout_inference",
        selection_sha256=selection_sha256,
    )


@dataclass(frozen=True)
class ActivationData:
    values: np.ndarray
    index: dict[str, np.ndarray]
    metadata: dict
    file_sha256: dict[str, str]


def _load_activation_data(
    data: FeatureData, activation_dir: Path, *, scope: str = "train_validation"
) -> ActivationData:
    metadata = json.loads((activation_dir / "metadata.json").read_text())
    if metadata.get("scope") != scope or metadata.get("extraction_complete") is not True:
        raise ValueError("Activation cache is incomplete or has a forbidden inference/test scope")
    if metadata.get("feature_cache_sha256") != data.file_sha256 or metadata.get(
        "activation_shape"
    ) != list(ACTIVATION_SHAPE):
        raise ValueError("Activation provenance differs from the requested feature cache")
    hashes = {}
    for name, expected in metadata["outputs_sha256"].items():
        if name not in {"activations.npy", "index.npz", "completed.npy", "row_sha256.npy"}:
            raise ValueError("Unexpected activation cache artifact")
        hashes[name] = sha256_file(activation_dir / name)
        if hashes[name] != expected:
            raise ValueError(f"Activation cache checksum mismatch: {name}")
    if not {"activations.npy", "index.npz"}.issubset(hashes):
        raise ValueError("Activation output checksums are missing")
    hashes["metadata.json"] = sha256_file(activation_dir / "metadata.json")
    with np.load(activation_dir / "index.npz", allow_pickle=False) as archive:
        index = {key: archive[key].copy() for key in (*INDEX_KEYS, "feature_index")}
    if any(
        not np.array_equal(index[key], data.arrays[key]) for key in INDEX_KEYS
    ) or not np.array_equal(index["feature_index"], np.arange(len(data.arrays["features"]))):
        raise ValueError("Activation index labels/splits/keys are not aligned with frozen features")
    values = np.load(activation_dir / "activations.npy", mmap_mode="r")
    if values.dtype != np.float32 or values.shape != (len(index["labels"]), *ACTIVATION_SHAPE):
        raise ValueError("Activation array has invalid shape/dtype")
    for start in range(0, len(values), 32):
        if not np.isfinite(values[start : start + 32]).all():
            raise ValueError("Activation cache contains nonfinite values")
    return ActivationData(values, index, metadata, hashes)


def load_last_block_head(
    checkpoint_path: str | Path, cache_dir: str | Path, device: str = "cpu"
) -> LastBlockClassifier:
    device = _resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    block, signature = _original_block(Path(cache_dir))
    if (
        checkpoint.get("model") != MODEL_NAME
        or checkpoint.get("input_shape") != list(ACTIVATION_SHAPE)
        or checkpoint.get("class_names") != list(CLASS_NAMES)
    ):
        raise ValueError("Last-block checkpoint architecture/class order mismatch")
    if checkpoint.get("original_weights_sha256") != signature["weights_sha256"]:
        raise ValueError("Last-block checkpoint uses different original weights")
    model = LastBlockClassifier(block)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()


def predict_last_block(
    activations: np.ndarray,
    checkpoint_path: str | Path,
    cache_dir: str | Path,
    device: str = "cpu",
    batch_size: int = 16,
) -> np.ndarray:
    """Predict cached inputs; the caller must verify their complete provenance."""
    if (
        batch_size < 1
        or activations.dtype != np.float32
        or activations.shape[1:] != ACTIVATION_SHAPE
    ):
        raise ValueError("Expected float32 last-block inputs with the recorded architecture shape")
    model = load_last_block_head(checkpoint_path, cache_dir, device)
    values = []
    with torch.inference_mode():
        for start in range(0, len(activations), batch_size):
            batch = np.array(activations[start : start + batch_size], dtype=np.float32, copy=True)
            if not np.isfinite(batch).all():
                raise ValueError("Nonfinite prediction activations")
            values.append(model(torch.from_numpy(batch).to(device)).softmax(1).cpu().numpy())
    return np.concatenate(values) if values else np.empty((0, 3), np.float32)


def _tree_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _tree_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_tree_cpu(item) for item in value]
    return value


def _head_checkpoint(head_dir: Path, seed: int, data: FeatureData) -> tuple[Path, dict]:
    path = head_dir / "runs" / f"seed-{seed}" / "head.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("model") != "Linear(512,3)"
        or checkpoint.get("feature_size") != 512
        or checkpoint.get("class_names") != list(CLASS_NAMES)
        or checkpoint.get("seed") != seed
        or checkpoint.get("feature_cache_sha256") != data.file_sha256
    ):
        raise ValueError("Warm-start head seed, source feature cache or architecture mismatch")
    return path, checkpoint


def _initial_suffix(
    cache_dir: Path, head_checkpoint: dict, device: str
) -> tuple[LastBlockClassifier, dict]:
    block, signature = _original_block(cache_dir)
    head = nn.Linear(512, 3)
    head.load_state_dict(head_checkpoint["state_dict"], strict=True)
    model = LastBlockClassifier(block, head).to(device)
    model.requires_grad_(True)
    model.eval()
    return model, signature


def _logits_for_indices(
    model: nn.Module, activations: np.ndarray, indices: np.ndarray, device: str, batch_size: int
) -> np.ndarray:
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            values = np.array(
                activations[indices[start : start + batch_size]], dtype=np.float32, copy=True
            )
            result.append(model(torch.from_numpy(values).to(device)).cpu().numpy())
    logits = np.concatenate(result) if result else np.empty((0, 3), np.float32)
    if not np.isfinite(logits).all():
        raise RuntimeError("Nonfinite fine-tuned predictions")
    return logits


def _validate_model(model, data: FeatureData, activations, indices, device: str, batch_size: int):
    logits = _logits_for_indices(model, activations, indices, device, batch_size)
    labels = torch.from_numpy(data.arrays["labels"][indices].astype(np.int64))
    loss = float(nn.functional.cross_entropy(torch.from_numpy(logits), labels))
    probabilities = torch.from_numpy(logits).softmax(1).numpy()
    all_probabilities = np.zeros((len(data.arrays["features"]), 3), np.float32)
    all_probabilities[indices] = probabilities
    rows = _prediction_rows(data, all_probabilities)
    return evaluate_predictions(rows), rows, loss, probabilities


def _verify_protocol(path: Path | None, config: dict, data: FeatureData) -> str | None:
    if path is None:
        return None
    protocol = json.loads(path.read_text())
    for key in (
        "optimizer",
        "learning_rate_block",
        "learning_rate_head",
        "weight_decay",
        "batch_size",
        "max_epochs",
        "patience",
        "seeds",
        "classes",
    ):
        if protocol.get(key) != config[key]:
            raise ValueError(
                f"Training configuration differs from the prespecified protocol: {key}"
            )
    if (
        protocol.get("source_feature_cache_id") != data.metadata["cache_id"]
        or protocol.get("study_manifest_sha256")
        != data.metadata["config"]["manifests"]["studies.csv"]
    ):
        raise ValueError("Protocol source cache or split manifest differs")
    return sha256_file(path)


def _export_checkpoint(
    model: LastBlockClassifier,
    best_state: dict,
    seed: int,
    best_epoch: int,
    data: FeatureData,
    activation_data: ActivationData,
    signature: dict,
    initial_head_sha256: str,
    config: dict,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "model": MODEL_NAME,
        "state_dict": _cpu_state(best_state),
        "input_shape": list(ACTIVATION_SHAPE),
        "class_names": list(CLASS_NAMES),
        "feature_size": 512,
        "seed": seed,
        "best_epoch": best_epoch,
        "initial_head_sha256": initial_head_sha256,
        "original_weights_sha256": signature["weights_sha256"],
        "original_source_signature": signature,
        "feature_cache_sha256": data.file_sha256,
        "activation_cache_sha256": activation_data.file_sha256,
        "activation_cache_id": activation_data.metadata["cache_id"],
        "configuration_sha256": _hash_json(config),
        "protocol_sha256": config["protocol_sha256"],
        "batchnorm_running_statistics_frozen": True,
        "earlier_encoder_updated": False,
        "scope": "Only last original residual block layer4.2 and a three-class head; no test selection",
    }


def train_last_block(
    feature_dir: str | Path,
    activation_dir: str | Path,
    head_dir: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    *,
    seeds: tuple[int, ...] = SEEDS,
    device: str = "cpu",
    threads: int = 4,
    max_epochs: int = 10,
    patience: int = 3,
    learning_rate_block: float = 1e-5,
    learning_rate_head: float = 1e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 16,
    resume: bool = True,
    protocol_path: str | Path | None = None,
) -> dict:
    """Resume at epoch boundaries; epoch zero participates in earliest-best selection.

    Only the residual block/head enter the optimizer. Cached activations are
    read-only, all BatchNorm buffers stay frozen, and earlier layers are absent
    from the training graph. No test feature/cache scope is accepted.
    """
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(seed) is not int or seed < 0 for seed in seeds)
    ):
        raise ValueError("Use distinct non-negative integer seeds")
    if min(threads, max_epochs, patience, batch_size) < 1:
        raise ValueError("Threads, epochs, patience and batch size must be positive")
    if (
        any(
            not math.isfinite(value) or value <= 0
            for value in (learning_rate_block, learning_rate_head)
        )
        or not math.isfinite(weight_decay)
        or weight_decay < 0
    ):
        raise ValueError("Learning rates must be positive and weight decay nonnegative")
    device = _resolve_device(device)
    torch.set_num_threads(threads)
    data = load_feature_data(feature_dir)
    activation_dir, head_dir, cache_dir, out = map(
        Path, (activation_dir, head_dir, cache_dir, output_dir)
    )
    activation_data = _load_activation_data(data, activation_dir)
    train_indices = np.flatnonzero(data.arrays["split"] == "train")
    validation_indices = np.flatnonzero(data.arrays["split"] == "validation")
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("Both assessed training and validation targets are required")
    class_weights = inverse_sqrt_class_weights(data.arrays["labels"][train_indices])
    initial_heads = {seed: _head_checkpoint(head_dir, seed, data) for seed in seeds}
    head_hashes = {
        str(seed): sha256_file(path) for seed, (path, _checkpoint) in initial_heads.items()
    }
    config = {
        "model": MODEL_NAME,
        "optimizer": "AdamW",
        "seeds": list(seeds),
        "classes": list(CLASS_NAMES),
        "class_names": list(CLASS_NAMES),
        "learning_rate_block": learning_rate_block,
        "learning_rate_head": learning_rate_head,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "patience": patience,
        "trainable_modules": ["grading_model.layer4.2", "new_linear_head_512_3"],
        "batch_normalization": "All running statistics frozen; only last-block affine parameters may train",
        "earlier_encoder_updated": False,
        "encoder_updated": "last_residual_block_only",
        "class_weights": class_weights.tolist(),
        "class_weight_rule": "inverse sqrt assessed training counts, normalised to mean 1",
        "assessed_training_class_counts": np.bincount(
            data.arrays["labels"][train_indices], minlength=3
        ).tolist(),
        "n_training_features": len(train_indices),
        "n_validation_features": len(validation_indices),
        "selection": "strictly highest validation macro-F1 for fixed labels [0,1,2], earliest ties, including warm-start epoch 0",
        "training_loss_definition": "Class-weighted batch losses accumulated during optimization, divided by the sum of sample class weights",
        "initial_head_sha256": head_hashes,
        "activation_cache_sha256": activation_data.file_sha256,
        "activation_cache_id": activation_data.metadata["cache_id"],
        "feature_cache_sha256": data.file_sha256,
        "implementation_sha256": sha256_file(Path(__file__)),
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "device": device,
            "threads": threads,
        },
    }
    config["protocol_sha256"] = _verify_protocol(
        Path(protocol_path) if protocol_path else None, config, data
    )
    config_sha256 = _hash_json(config)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "configuration": config,
        "configuration_sha256": config_sha256,
        "class_names": list(CLASS_NAMES),
        "feature_cache_sha256": data.file_sha256,
        "feature_cache_metadata": data.metadata,
        "activation_cache_metadata": activation_data.metadata,
        "scope": "Train/validation only; includes epoch zero; no test selection or clinical validation",
    }
    if out.exists() and (out / "manifest.json").exists():
        if (
            not resume
            or json.loads((out / "manifest.json").read_text()).get("configuration_sha256")
            != config_sha256
        ):
            raise ValueError(
                "Existing experiment has different immutable configuration or resume is disabled"
            )
    elif out.exists() and any(out.iterdir()):
        raise ValueError("Output directory contains files without a valid resume manifest")
    else:
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "manifest.json", manifest)
    seed_results = []
    for seed in seeds:
        seed_out = out / "runs" / f"seed-{seed}"
        seed_out.mkdir(parents=True, exist_ok=True)
        if (seed_out / "result.json").exists():
            result = json.loads((seed_out / "result.json").read_text())
            if sha256_file(seed_out / "last-block.pt") != result["checkpoint_sha256"]:
                raise ValueError("A completed last-block checkpoint was modified")
            seed_results.append(result)
            continue
        torch.manual_seed(seed)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        initial_head_path, initial_head = initial_heads[seed]
        model, signature = _initial_suffix(cache_dir, initial_head, device)
        if (
            signature["weights_sha256"]
            != activation_data.metadata["source_signature"]["weights_sha256"]
        ):
            raise ValueError("Initial last block differs from the cached prefix's original weights")
        original_bn_hash = _state_hash(_bn_state(model))
        initial_block_hash = _state_hash(dict(model.block.named_parameters()))
        initial_head_hash = _state_hash(dict(model.head.named_parameters()))
        optimizer = torch.optim.AdamW(
            [
                {"params": model.block.parameters(), "lr": learning_rate_block},
                {"params": model.head.parameters(), "lr": learning_rate_head},
            ],
            weight_decay=weight_decay,
        )
        weight_tensor = torch.from_numpy(class_weights).to(device)
        history, stale, best_epoch, best_score = [], 0, 0, -math.inf
        best_state = None
        warm_error = None
        initialization_parity = None
        block_optimized, head_optimized = False, False
        resume_path = seed_out / "resume.pt"
        if resume_path.exists():
            saved = torch.load(resume_path, map_location="cpu", weights_only=True)
            if saved.get("configuration_sha256") != config_sha256 or saved.get("seed") != seed:
                raise ValueError("Epoch resume checkpoint provenance mismatch")
            model.load_state_dict(saved["state_dict"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            generator.set_state(saved["generator_state"])
            history, stale, best_epoch, best_score = (
                saved["history"],
                saved["stale"],
                saved["best_epoch"],
                saved["best_score"],
            )
            best_state = saved["best_state"]
            warm_error = saved["warm_start_max_probability_abs_error"]
            initialization_parity = saved["initialization_parity"]
            block_optimized, head_optimized = (
                saved["block_parameters_optimized"],
                saved["head_parameters_optimized"],
            )
            if _state_hash(_bn_state(model)) != original_bn_hash:
                raise ValueError("Resume checkpoint changed frozen BatchNorm buffers")
        else:
            started = time.perf_counter()
            metrics, _rows, val_loss, warm_probabilities = _validate_model(
                model, data, activation_data.values, validation_indices, device, batch_size
            )
            reference = predict_head(data.arrays["features"][validation_indices], initial_head_path)
            warm_error = float(np.abs(warm_probabilities - reference).max())
            if warm_error > PARITY_ATOL or not np.allclose(
                warm_probabilities, reference, atol=PARITY_ATOL, rtol=PARITY_RTOL
            ):
                raise ValueError(
                    "Warm-start last block/head does not reproduce the corresponding frozen head"
                )
            if not np.array_equal(warm_probabilities.argmax(1), reference.argmax(1)):
                raise ValueError(
                    "Warm-start argmax differs from its frozen head despite numerical tolerance"
                )
            reference_all = np.zeros((len(data.arrays["features"]), 3), np.float32)
            reference_all[validation_indices] = reference
            reference_metrics = evaluate_predictions(_prediction_rows(data, reference_all))
            initialization_parity = {
                "frozen_head_macro_f1": float(reference_metrics["conditional"]["macro_f1"]),
                "epoch0_macro_f1": float(metrics["conditional"]["macro_f1"]),
                "n_prediction_mismatches": 0,
                "max_probability_abs_error": warm_error,
                "atol": PARITY_ATOL,
            }
            best_score = float(metrics["conditional"]["macro_f1"])
            best_state = _cpu_state(model.state_dict())
            history.append(
                {
                    "epoch": 0,
                    "training_loss": None,
                    "validation_loss_unweighted": val_loss,
                    "validation_macro_f1": best_score,
                    "seconds": time.perf_counter() - started,
                }
            )

        def save_resume():
            _atomic_torch(
                resume_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "configuration_sha256": config_sha256,
                    "seed": seed,
                    "state_dict": _cpu_state(model.state_dict()),
                    "optimizer": _tree_cpu(optimizer.state_dict()),
                    "generator_state": generator.get_state(),
                    "history": history,
                    "best_state": best_state,
                    "best_score": best_score,
                    "best_epoch": best_epoch,
                    "stale": stale,
                    "warm_start_max_probability_abs_error": warm_error,
                    "initialization_parity": initialization_parity,
                    "block_parameters_optimized": block_optimized,
                    "head_parameters_optimized": head_optimized,
                },
            )
            write_json(seed_out / "history.json", history)

        if not resume_path.exists():
            save_resume()
        last_epoch = int(history[-1]["epoch"])
        if stale < patience:
            for epoch in range(last_epoch + 1, max_epochs + 1):
                started = time.perf_counter()
                model.train()
                permutation = train_indices[
                    torch.randperm(len(train_indices), generator=generator).numpy()
                ]
                numerator, denominator = 0.0, 0.0
                total_batches = math.ceil(len(permutation) / batch_size)
                for batch_number, offset in enumerate(
                    range(0, len(permutation), batch_size), start=1
                ):
                    indices = permutation[offset : offset + batch_size]
                    x = torch.from_numpy(
                        np.array(activation_data.values[indices], dtype=np.float32, copy=True)
                    ).to(device)
                    y = torch.from_numpy(data.arrays["labels"][indices].astype(np.int64)).to(device)
                    optimizer.zero_grad(set_to_none=True)
                    weighted_sum = nn.functional.cross_entropy(
                        model(x), y, weight=weight_tensor, reduction="sum"
                    )
                    normalizer = weight_tensor[y].sum()
                    loss = weighted_sum / normalizer
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite last-block training loss")
                    loss.backward()
                    optimizer.step()
                    numerator += float(weighted_sum.detach().cpu())
                    denominator += float(normalizer.detach().cpu())
                    if batch_number % 20 == 0 or batch_number == total_batches:
                        _sync(device)
                        print(
                            f"Last block seed={seed} epoch={epoch} batch={batch_number}/{total_batches} seconds={time.perf_counter() - started:.1f}",
                            flush=True,
                        )
                metrics, _rows, val_loss, _probabilities = _validate_model(
                    model, data, activation_data.values, validation_indices, device, batch_size
                )
                score = float(metrics["conditional"]["macro_f1"])
                if _state_hash(_bn_state(model)) != original_bn_hash:
                    raise RuntimeError(
                        "Frozen BatchNorm running statistics changed during optimization"
                    )
                block_optimized |= (
                    _state_hash(dict(model.block.named_parameters())) != initial_block_hash
                )
                head_optimized |= (
                    _state_hash(dict(model.head.named_parameters())) != initial_head_hash
                )
                history.append(
                    {
                        "epoch": epoch,
                        "training_loss": numerator / denominator,
                        "validation_loss_unweighted": val_loss,
                        "validation_macro_f1": score,
                        "seconds": time.perf_counter() - started,
                    }
                )
                if score > best_score + 1e-12:
                    best_score, best_epoch, stale = score, epoch, 0
                    best_state = _cpu_state(model.state_dict())
                else:
                    stale += 1
                save_resume()
                print(
                    f"Last block seed={seed}: epoch={epoch}, validation macro-F1={score:.4f}, best={best_epoch}, stale={stale}/{patience}",
                    flush=True,
                )
                if stale >= patience:
                    break
        if not block_optimized or not head_optimized:
            raise RuntimeError("Optimization did not change both the selected block and new head")
        model.load_state_dict(best_state, strict=True)
        model.eval()
        metrics, rows, _val_loss, _probabilities = _validate_model(
            model, data, activation_data.values, validation_indices, device, batch_size
        )
        if _state_hash(_bn_state(model)) != original_bn_hash:
            raise RuntimeError("Selected checkpoint has changed BatchNorm buffers")
        if sha256_file(initial_head_path) != head_hashes[str(seed)]:
            raise RuntimeError("The original frozen-head checkpoint was modified")
        _atomic_torch(
            seed_out / "last-block.pt",
            _export_checkpoint(
                model,
                best_state,
                seed,
                best_epoch,
                data,
                activation_data,
                signature,
                head_hashes[str(seed)],
                config,
            ),
        )
        _write_predictions(seed_out / "predictions.csv", rows)
        write_json(seed_out / "metrics.json", metrics)
        write_json(seed_out / "history.json", history)
        with (seed_out / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        result = {
            "seed": seed,
            "best_epoch": best_epoch,
            "epochs_run": int(history[-1]["epoch"]),
            "validation_macro_f1": best_score,
            "metrics": metrics,
            "checkpoint_sha256": sha256_file(seed_out / "last-block.pt"),
            "warm_start_max_probability_abs_error": warm_error,
            "initialization_parity": initialization_parity,
            "block_parameters_changed_during_optimization": bool(block_optimized),
            "head_parameters_changed_during_optimization": bool(head_optimized),
            "selected_block_parameters_changed": _state_hash(dict(model.block.named_parameters()))
            != initial_block_hash,
            "selected_head_parameters_changed": _state_hash(dict(model.head.named_parameters()))
            != initial_head_hash,
            "all_batchnorm_running_statistics_unchanged": True,
            "earlier_encoder_updated": False,
        }
        write_json(seed_out / "result.json", result)
        seed_results.append(result)
        model, optimizer = None, None
        if device == "mps":
            torch.mps.empty_cache()
    scores = [run["validation_macro_f1"] for run in seed_results]
    summary = {
        "configuration": config,
        "runs": seed_results,
        "validation_macro_f1_mean": float(np.mean(scores)),
        "validation_macro_f1_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        "std_definition": "Sample standard deviation across seeds, not a confidence interval",
        "all_seeds_reported": True,
        "test_evaluated": False,
    }
    if (
        sha256_file(activation_dir / "activations.npy")
        != activation_data.file_sha256["activations.npy"]
    ):
        raise RuntimeError("Read-only prefix activations changed during optimization")
    if (
        _source_signature(cache_dir)["weights_sha256"]
        != activation_data.metadata["source_signature"]["weights_sha256"]
    ):
        raise RuntimeError("Original source checkpoints changed during optimization")
    write_json(out / "summary.json", summary)
    return summary
