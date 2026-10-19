"""Frozen-encoder sagittal/axial/fusion development comparison.

Axial attention sees only axial ImageNet features and fixed geometric positions;
it never receives sagittal localization or annotation coordinates. All variants
share one complete-paired target mask. CV and the previously used validation are
development estimates, not a fresh independent test. Prediction/attention files
contain local dataset identifiers and must not be published.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn import functional as F

from .assets import sha256_file, write_json
from .data import CLASS_NAMES
from .head_experiments import ensemble_rows, make_folds
from .training import (
    LEVELS,
    FeatureData,
    evaluate_predictions,
    inverse_sqrt_class_weights,
    load_feature_data,
)
from .uncertainty import paired_study_bootstrap

VARIANTS = ("sagittal", "axial", "fusion")
SEEDS = (42, 2024, 3407)
MAX_SLICES = 96
_HASH = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_ENCODER = {
    "architecture": "torchvision.resnet18",
    "weights": "IMAGENET1K_V1",
    "url": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "sha256": "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
}


def _hash_json(value) -> str:
    return hashlib.sha256(
        (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()
    ).hexdigest()


def _axial_config_id(value) -> str:
    # The extractor uses compact JSON without a final newline for its cache ID.
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode()).hexdigest()


def _key(row: dict) -> tuple[str, str, str]:
    return str(row["study_id"]), str(row["series_id"]), row["level"]


def _safe_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Axial cache file paths must be relative")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Axial cache path escapes its directory")
    return path


@dataclass
class PairedData:
    sagittal: FeatureData
    study_ids: list[str]
    splits: np.ndarray
    sag: torch.Tensor
    axial: torch.Tensor
    positions: torch.Tensor
    slice_mask: torch.Tensor
    labels: torch.Tensor
    paired: torch.Tensor
    targets: list[dict]
    selected_indices: list[np.ndarray]
    selected_series: list[np.ndarray]
    selected_instances: list[np.ndarray]
    axial_metadata: dict
    axial_metadata_sha256: str
    axial_output_sha256: str


def load_paired_data(sag_features: str | Path, axial_features: str | Path) -> PairedData:
    """Validate both caches and construct deterministic, padded study bags."""
    sag = load_feature_data(sag_features)  # Strict reader rejects every TEST row.
    root = Path(axial_features)
    metadata = json.loads((root / "metadata.json").read_text())
    if (
        metadata.get("schema_version") != 1
        or metadata.get("scope") != "train_validation"
        or metadata.get("extraction_complete") is not True
        or metadata.get("config", {}).get("allowed_splits") != ["train", "validation"]
    ):
        raise ValueError("Require a complete TRAIN/validation axial cache; TEST is forbidden")
    encoder = metadata.get("config", {}).get("encoder", {})
    if not isinstance(encoder, dict) or any(
        encoder.get(k) != v for k, v in EXPECTED_ENCODER.items()
    ):
        raise ValueError("Axial encoder architecture or frozen ImageNet weights differ")
    config = metadata["config"]
    study_manifest = sag.metadata.get("config", {}).get("manifests", {}).get("studies.csv")
    if (
        config.get("schema_version") != 1
        or not isinstance(metadata.get("cache_id"), str)
        or not _HASH.fullmatch(metadata["cache_id"])
        or metadata["cache_id"] != _axial_config_id(config)
        or not isinstance(study_manifest, str)
        or not _HASH.fullmatch(study_manifest)
        or config.get("manifest_input_sha256", {}).get("studies.csv") != study_manifest
        or config.get("preprocessing", {}).get("encoder_updated") is not False
    ):
        raise ValueError("Axial cache identity, fixed study split or frozen preprocessing mismatch")
    expected_splits = {row["study_id"]: row["split"] for row in sag.targets}
    studies = sorted(expected_splits)
    study_index = {study: i for i, study in enumerate(studies)}
    records = metadata.get("study_records")
    if not isinstance(records, list) or len(records) != len(studies):
        raise ValueError("Axial cache must retain every requested sagittal TRAIN/validation study")
    count = len(studies)
    sagittal = np.zeros((count, 2, 512), np.float32)
    axial = np.zeros((count, MAX_SLICES, 512), np.float32)
    positions = np.zeros((count, MAX_SLICES, 2), np.float32)
    slice_mask = np.zeros((count, MAX_SLICES), bool)
    labels = np.full((count, 2), -100, np.int64)
    paired = np.zeros((count, 2), bool)
    selection = [np.empty(0, np.int64) for _ in studies]
    series = [np.empty(0, dtype="U1") for _ in studies]
    instances = [np.empty(0, np.int64) for _ in studies]
    indexed, outputs = {}, {}
    for entry in records:
        study = str(entry["study_id"])
        if (
            study not in study_index
            or study in indexed
            or entry.get("split") != expected_splits[study]
            or entry.get("split") not in {"train", "validation"}
            or entry.get("status") not in {"ok", "failed"}
            or entry.get("source_ready") is not True
            or type(entry.get("n_series")) is not int
            or entry["n_series"] < 1
        ):
            raise ValueError("Axial study identity, split, completion or status mismatch")
        path = _safe_path(root, entry["record_path"])
        digest = sha256_file(path)
        if digest != entry.get("record_sha256"):
            raise ValueError("Axial study record checksum mismatch")
        outputs[entry["record_path"]] = digest
        record = json.loads(path.read_text())
        if (
            record.get("schema_version") != 1
            or record.get("cache_id") != metadata["cache_id"]
            or any(
                record.get(field) != entry.get(field)
                for field in (
                    "study_id",
                    "split",
                    "status",
                    "reason",
                    "arrays_path",
                    "arrays_sha256",
                    "n_slices",
                    "n_series",
                    "source_ready",
                )
            )
        ):
            raise ValueError("Axial index and per-study record disagree")
        indexed[study] = entry
        if entry["status"] != "ok":
            if entry.get("arrays_path") or entry.get("arrays_sha256") or entry.get("n_slices") != 0:
                raise ValueError("Failed axial study unexpectedly carries feature arrays")
            continue
        array_path = _safe_path(root, entry["arrays_path"])
        digest = sha256_file(array_path)
        if digest != entry.get("arrays_sha256"):
            raise ValueError("Axial bag checksum mismatch")
        outputs[entry["arrays_path"]] = digest
        with np.load(array_path, allow_pickle=False) as archive:
            if set(archive.files) != {"features", "positions", "instance", "series_id"}:
                raise ValueError("Unexpected axial feature-array keys")
            bag, pos, instance, names = (
                archive[name] for name in ("features", "positions", "instance", "series_id")
            )
        n = len(bag)
        if (
            type(entry.get("n_slices")) is not int
            or n != entry["n_slices"]
            or n < 1
            or bag.shape != (n, 512)
            or bag.dtype != np.float32
            or pos.shape != (n, 2)
            or pos.dtype != np.float32
            or instance.shape != (n,)
            or instance.dtype != np.int64
            or names.shape != (n,)
            or names.dtype.kind != "U"
            or not np.isfinite(bag).all()
            or not np.isfinite(pos).all()
            or np.any(pos < -1e-6)
            or np.any(pos > 1 + 1e-6)
            or len(set(names.tolist())) != entry["n_series"]
            or any(np.any(np.diff(pos[names == name, 1]) < -1e-6) for name in set(names.tolist()))
            or len(set(zip(names.tolist(), instance.tolist()))) != n
        ):
            raise ValueError("Invalid or unordered axial features, positions or slice identifiers")
        selected = (
            np.arange(n) if n <= MAX_SLICES else np.linspace(0, n - 1, MAX_SLICES, dtype=np.int64)
        )
        i, length = study_index[study], len(selected)
        axial[i, :length] = bag[selected]
        positions[i, :length] = pos[selected]
        slice_mask[i, :length] = True
        selection[i], series[i], instances[i] = selected, names[selected], instance[selected]
    if set(indexed) != set(studies) or metadata.get("outputs_sha256") != outputs:
        raise ValueError("Axial cache output inventory or requested study set is incomplete")
    targets = []
    for original in sag.targets:
        row = dict(original)
        i, level = study_index[row["study_id"]], LEVELS.index(row["level"])
        entry = indexed[row["study_id"]]
        sag_ok = row["status"] == "ok"
        axial_ok = entry["status"] == "ok"
        if row["label"] is not None:
            labels[i, level] = row["label"]
        if sag_ok:
            sagittal[i, level] = sag.arrays["features"][row["feature_index"]]
        is_paired = sag_ok and axial_ok and row["label"] is not None
        paired[i, level] = is_paired
        reasons = []
        if not sag_ok:
            reasons.append("sagittal:" + (row["reason"] or row["status"]))
        if not axial_ok:
            # Keep only a fixed processing code in aggregate/public metrics.
            reasons.append("axial:processing_failed")
        if row["label"] is None:
            reasons.append("missing_reference_label")
        row.update(
            status="ok" if is_paired else "not_assessed",
            reason=";".join(reasons),
            sagittal_available=sag_ok,
            axial_available=axial_ok,
            study_index=i,
            level_index=level,
        )
        targets.append(row)
    return PairedData(
        sag,
        studies,
        np.asarray([expected_splits[s] for s in studies]),
        torch.from_numpy(sagittal),
        torch.from_numpy(axial),
        torch.from_numpy(positions),
        torch.from_numpy(slice_mask),
        torch.from_numpy(labels),
        torch.from_numpy(paired),
        targets,
        selection,
        series,
        instances,
        metadata,
        sha256_file(root / "metadata.json"),
        _hash_json(outputs),
    )


def fit_normalizers(data: PairedData, fitting: np.ndarray, *, epsilon: float = 1e-6) -> dict:
    """Fit separate channel moments exclusively on paired fitting TRAIN studies."""
    selected = np.asarray(fitting, dtype=np.int64)
    if (
        not len(selected)
        or len(set(selected.tolist())) != len(selected)
        or np.any(selected < 0)
        or np.any(selected >= len(data.study_ids))
    ):
        raise ValueError("Require distinct valid fitting study indices")
    if np.any(data.splits[selected] != "train"):
        raise ValueError("Normalization may fit original TRAIN only, never validation")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Normalization epsilon must be positive")

    def moments(kind):
        total, mean, m2 = 0, np.zeros(512), np.zeros(512)
        for i in selected:
            mask = data.paired[i].numpy() if kind == "sag" else data.slice_mask[i].numpy()
            if not data.paired[i].any():
                continue
            values = (
                (data.sag[i] if kind == "sag" else data.axial[i]).numpy()[mask].astype(np.float64)
            )
            n = len(values)
            if not n:
                continue
            part_mean = values.mean(axis=0)
            part_m2 = ((values - part_mean) ** 2).sum(axis=0)
            delta = part_mean - mean
            m2 += part_m2 + delta * delta * total * n / (total + n)
            mean += delta * n / (total + n)
            total += n
        if not total:
            raise ValueError("No paired fitting features for normalization")
        std = np.maximum(np.sqrt(np.maximum(m2 / total, 0)), epsilon)
        return (
            torch.from_numpy(mean.astype(np.float32)),
            torch.from_numpy(std.astype(np.float32)),
            total,
        )

    sag_mean, sag_std, n_sag = moments("sag")
    axial_mean, axial_std, n_axial = moments("axial")
    return {
        "sag_mean": sag_mean,
        "sag_std": sag_std,
        "axial_mean": axial_mean,
        "axial_std": axial_std,
        "n_sagittal_targets": n_sag,
        "n_axial_slices": n_axial,
        "fitting_studies_sha256": _hash_json([data.study_ids[i] for i in sorted(selected)]),
    }


def architecture(variant: str) -> dict:
    if variant not in VARIANTS:
        raise ValueError("Unknown multimodal variant")
    return {
        "variant": variant,
        "feature_size": 512,
        "position_size": 2,
        "levels": list(LEVELS),
        "max_slices": MAX_SLICES,
        "hidden_size": 64,
        "dropout": 0.2,
        "head_input": 1026 if variant == "fusion" else 514,
        "attention": None
        if variant == "sagittal"
        else "Linear(514,64),Tanh,Linear(64,2); masked slice softmax",
    }


class MultimodalHead(nn.Module):
    def __init__(self, variant: str, normalizers: dict):
        super().__init__()
        self.variant = variant
        spec = architecture(variant)
        for field in ("sag_mean", "sag_std", "axial_mean", "axial_std"):
            value = normalizers[field]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != (512,)
                or not torch.isfinite(value).all()
                or (field.endswith("std") and not (value > 0).all())
            ):
                raise ValueError("Invalid normalization tensors")
            self.register_buffer(field, value.detach().clone().to(torch.float32))
        self.register_buffer("level_one_hot", torch.eye(2))
        self.attention = (
            None
            if variant == "sagittal"
            else nn.Sequential(nn.Linear(514, 64), nn.Tanh(), nn.Linear(64, 2))
        )
        self.head = nn.Sequential(
            nn.Linear(spec["head_input"], 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 3)
        )

    def forward(self, sagittal, axial, positions, mask, *, return_attention=False):
        if sagittal.ndim != 3 or sagittal.shape[1:] != (2, 512):
            raise ValueError("Sagittal features must be [B,2,512]")
        batch = len(sagittal)
        one_hot = self.level_one_hot[None].expand(batch, -1, -1)
        sag = (sagittal - self.sag_mean) / self.sag_std if self.variant != "axial" else None
        weights = None
        if self.variant != "sagittal":
            if (
                axial.ndim != 3
                or axial.shape[:2] != mask.shape
                or axial.shape[2] != 512
                or positions.shape != (*mask.shape, 2)
                or not mask.any(dim=1).all()
            ):
                raise ValueError("Axial attention requires a nonempty masked bag per study")
            normalized = (axial - self.axial_mean) / self.axial_std
            scores = self.attention(torch.cat((normalized, positions), dim=-1)).transpose(1, 2)
            scores = scores.masked_fill(~mask[:, None], float("-inf"))
            weights = scores.softmax(dim=-1)
            pooled = torch.bmm(weights, normalized)
        if self.variant == "sagittal":
            content = sag
        elif self.variant == "axial":
            content = pooled
        else:
            content = torch.cat((sag, pooled), dim=-1)
        logits = self.head(torch.cat((content, one_hot), dim=-1))
        return (logits, weights) if return_attention else logits


def _batch(data: PairedData, indices: np.ndarray, variant: str):
    index = torch.as_tensor(indices, dtype=torch.int64)
    length = 1 if variant == "sagittal" else max(1, int(data.slice_mask[index].sum(dim=1).max()))
    return (
        data.sag[index],
        data.axial[index, :length],
        data.positions[index, :length],
        data.slice_mask[index, :length],
    )


def _probabilities(
    model: MultimodalHead, data: PairedData, indices: np.ndarray, *, batch_size=32, attention=False
):
    values = np.full((len(data.study_ids), 2, 3), np.nan, np.float32)
    weights = np.zeros((len(data.study_ids), 2, MAX_SLICES), np.float32) if attention else None
    valid = np.asarray([i for i in indices if data.paired[i].any()], np.int64)
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(valid), batch_size):
            selected = valid[offset : offset + batch_size]
            logits, per_slice = model(*_batch(data, selected, model.variant), return_attention=True)
            probabilities = logits.softmax(dim=-1).numpy()
            if (
                probabilities.shape != (len(selected), 2, 3)
                or not np.isfinite(probabilities).all()
                or not np.allclose(probabilities.sum(axis=-1), 1, rtol=0, atol=1e-6)
            ):
                raise RuntimeError("Invalid multimodal probabilities")
            values[selected] = probabilities
            if attention and per_slice is not None:
                weights[selected, :, : per_slice.shape[-1]] = per_slice.numpy()
    return values, weights


def _fit(
    variant: str,
    seed: int,
    data: PairedData,
    fitting: np.ndarray,
    normalizers: dict,
    config: dict,
    *,
    epochs: int,
    held: np.ndarray | None = None,
):
    fitting = np.asarray([i for i in fitting if data.paired[i].any()], np.int64)
    if not len(fitting) or np.any(data.splits[fitting] != "train"):
        raise ValueError("Fitting requires commonly assessed original TRAIN studies")
    if held is not None and (
        set(fitting.tolist()) & set(held.tolist()) or np.any(data.splits[held] != "train")
    ):
        raise ValueError("CV fitting and held TRAIN studies must be disjoint")
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model = MultimodalHead(variant, normalizers)
    truth = data.labels[fitting][data.paired[fitting]].numpy()
    class_weights = inverse_sqrt_class_weights(truth)
    weight = torch.from_numpy(class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    history, best_state, best_epoch, best_score, stale = [], None, 0, -math.inf, 0
    for epoch in range(1, epochs + 1):
        began = time.perf_counter()
        model.train()
        permutation = torch.randperm(len(fitting), generator=generator).numpy()
        numerator, denominator = 0.0, 0.0
        for offset in range(0, len(fitting), config["batch_size"]):
            selected = fitting[permutation[offset : offset + config["batch_size"]]]
            optimizer.zero_grad(set_to_none=True)
            logits = model(*_batch(data, selected, variant))
            mask = data.paired[selected]
            y = data.labels[selected][mask]
            loss = F.cross_entropy(logits[mask], y, weight=weight)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite multimodal training loss")
            loss.backward()
            optimizer.step()
            divisor = float(weight[y].sum())
            numerator += float(loss.detach()) * divisor
            denominator += divisor
        record = {"epoch": epoch, "training_loss": numerator / denominator}
        if held is not None:
            scores, _ = _probabilities(model, data, held, batch_size=config["batch_size"])
            mask = data.paired[held].numpy()
            y = data.labels[held].numpy()[mask]
            assessed = scores[held][mask]
            if not len(y):
                raise ValueError("Held fold lacks commonly assessed targets")
            macro = float(
                f1_score(y, assessed.argmax(-1), labels=[0, 1, 2], average="macro", zero_division=0)
            )
            record.update(
                heldout_macro_f1=macro,
                heldout_loss_unweighted=float(
                    -np.log(np.maximum(assessed[np.arange(len(y)), y], 1e-30)).mean()
                ),
            )
            if macro > best_score + 1e-12:
                best_score, best_epoch, stale = macro, epoch, 0
                best_state = {
                    key: value.detach().clone() for key, value in model.state_dict().items()
                }
            else:
                stale += 1
        record["seconds"] = time.perf_counter() - began
        history.append(record)
        print(
            f"{variant} seed={seed} epoch={epoch} seconds={record['seconds']:.2f}"
            + (
                f" held_macro_f1={record['heldout_macro_f1']:.6f}"
                if held is not None
                else " final_TRAIN_fit"
            ),
            flush=True,
        )
        if held is not None and stale >= config["patience"]:
            break
    if held is not None:
        if best_state is None:
            raise RuntimeError("No finite multimodal CV checkpoint")
        model.load_state_dict(best_state, strict=True)
    else:
        best_epoch = epochs
    model.eval()
    return model, history, best_epoch, class_weights


def _rows(data: PairedData, study_indices: np.ndarray, probabilities: np.ndarray) -> list[dict]:
    selected = set(np.asarray(study_indices).tolist())
    output = []
    for target in data.targets:
        i, level = target["study_index"], target["level_index"]
        if i not in selected:
            continue
        row = {
            field: target[field]
            for field in (
                "study_id",
                "series_id",
                "level",
                "split",
                "label",
                "status",
                "reason",
                "sagittal_available",
                "axial_available",
            )
        }
        row.update(prediction=None, **{f"p_{name}": None for name in CLASS_NAMES})
        if target["status"] == "ok":
            values = probabilities[i, level]
            if (
                not np.isfinite(values).all()
                or np.any(values < 0)
                or not np.isclose(values.sum(), 1, rtol=0, atol=1e-6)
            ):
                raise ValueError("An assessed target is missing valid probabilities")
            row["prediction"] = int(values.argmax())
            row.update({f"p_{name}": float(value) for name, value in zip(CLASS_NAMES, values)})
        output.append(row)
    return sorted(output, key=_key)


def _metrics(rows: list[dict], scope: str) -> dict:
    metrics = evaluate_predictions(rows)
    metrics["scope"] = (
        scope + "; fixed common-paired target subset; development estimate, no independent test"
    )
    metrics["by_status"] = dict(Counter(row["status"] for row in rows))
    metrics["not_assessed_by_reason"] = dict(
        Counter(row["reason"] for row in rows if row["status"] != "ok")
    )
    for group in [metrics, *metrics["by_level"].values()]:
        group["severe"]["missed_all_requested"] = (
            group["severe"]["n_total"] - group["severe"]["correctly_predicted_severe"]
        )
    return metrics


def _write_predictions(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        fields = (
            "study_id",
            "series_id",
            "level",
            "split",
            "label",
            "status",
            "reason",
            "sagittal_available",
            "axial_available",
            "prediction",
            "p_normal_mild",
            "p_moderate",
            "p_severe",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_multimodal_checkpoint(path: str | Path) -> tuple[MultimodalHead, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    variant = checkpoint.get("variant")
    if (
        checkpoint.get("schema_version") != 1
        or checkpoint.get("architecture") != architecture(variant)
        or checkpoint.get("classes") != list(CLASS_NAMES)
        or checkpoint.get("levels") != list(LEVELS)
        or checkpoint.get("encoder_updated") is not False
    ):
        raise ValueError("Invalid multimodal checkpoint architecture/category order")
    state = checkpoint.get("state_dict", {})
    model = MultimodalHead(
        variant,
        {field: state[field] for field in ("sag_mean", "sag_std", "axial_mean", "axial_std")},
    )
    if set(state) != set(model.state_dict()) or any(
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or not torch.isfinite(value).all()
        for value in state.values()
    ):
        raise ValueError("Invalid or nonfinite multimodal checkpoint state")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, checkpoint


def _export_run(
    directory: Path,
    model,
    history,
    epoch,
    seed,
    data,
    assessed,
    normalizers,
    class_weights,
    provenance,
    *,
    scope,
    fold,
):
    directory.mkdir(parents=True)
    checkpoint = {
        "schema_version": 1,
        "variant": model.variant,
        "architecture": architecture(model.variant),
        "classes": list(CLASS_NAMES),
        "levels": list(LEVELS),
        "seed": seed,
        "best_epoch": epoch,
        "fold": fold,
        "scope": scope,
        "encoder_updated": False,
        "state_dict": {key: value.detach().clone() for key, value in model.state_dict().items()},
        "class_weights": class_weights.tolist(),
        "normalization_fitting": {
            key: value for key, value in normalizers.items() if not isinstance(value, torch.Tensor)
        },
        **provenance,
    }
    path = directory / "model.pt"
    torch.save(checkpoint, path)
    reopened, _ = load_multimodal_checkpoint(path)
    before, _ = _probabilities(model, data, assessed)
    after, attention = _probabilities(
        reopened, data, assessed, attention=model.variant != "sagittal"
    )
    mask = data.paired.numpy() & np.isin(np.arange(len(data.study_ids)), assessed)[:, None]
    if not np.allclose(before[mask], after[mask], rtol=0, atol=1e-7):
        raise RuntimeError("Multimodal checkpoint roundtrip probabilities changed")
    rows = _rows(data, assessed, after)
    metrics = _metrics(rows, scope)
    _write_predictions(directory / "predictions.csv", rows)
    write_json(directory / "metrics.json", metrics)
    write_json(directory / "history.json", history)
    result = {
        "seed": seed,
        "fold": fold,
        "best_epoch": epoch,
        "epochs_run": len(history),
        "macro_f1": metrics["conditional"]["macro_f1"],
        "checkpoint_sha256": sha256_file(path),
        "predictions_sha256": sha256_file(directory / "predictions.csv"),
        "checkpoint_roundtrip_max_probability_error": float(
            np.max(np.abs(before[mask] - after[mask]))
        )
        if mask.any()
        else 0.0,
        "metrics": metrics,
    }
    write_json(directory / "result.json", result)
    return result, rows, attention


def _export_ensemble(directory: Path, rows: list[dict], scope: str):
    directory.mkdir(parents=True)
    _write_predictions(directory / "predictions.csv", rows)
    metrics = _metrics(rows, scope)
    write_json(directory / "metrics.json", metrics)
    return {
        "seeds": list(SEEDS),
        "weights": [1 / 3] * 3,
        "metrics": metrics,
        "predictions_sha256": sha256_file(directory / "predictions.csv"),
    }


def availability(data: PairedData) -> dict:
    def counts(rows):
        return {
            "n_requested": len(rows),
            "n_labeled": sum(row["label"] is not None for row in rows),
            "sagittal_available": sum(row["sagittal_available"] for row in rows),
            "axial_available": sum(row["axial_available"] for row in rows),
            "paired_assessed": sum(row["status"] == "ok" for row in rows),
            "not_assessed": sum(row["status"] != "ok" for row in rows),
            "not_assessed_by_reason": dict(
                Counter(row["reason"] for row in rows if row["status"] != "ok")
            ),
        }

    return {
        split: {
            **counts([r for r in data.targets if r["split"] == split]),
            "by_level": {
                level: counts(
                    [r for r in data.targets if r["split"] == split and r["level"] == level]
                )
                for level in LEVELS
            },
        }
        for split in ("train", "validation")
    }


def _attention_report(path: Path, data: PairedData, indices: np.ndarray, weights: np.ndarray):
    records = []
    for i in indices:
        if not data.paired[i].any():
            continue
        n = len(data.selected_indices[i])
        levels = {}
        for level, name in enumerate(LEVELS):
            values = weights[i, level, :n]
            if not data.paired[i, level]:
                levels[name] = {"status": "not_assessed", "weights": [], "top3": []}
                continue
            if not np.isfinite(values).all() or not np.isclose(values.sum(), 1, atol=1e-6):
                raise ValueError("Invalid attention weights")
            top = np.argsort(-values, kind="stable")[:3]
            levels[name] = {
                "status": "ok",
                "weights": values.tolist(),
                "top3": [
                    {
                        "bag_index": int(k),
                        "source_index": int(data.selected_indices[i][k]),
                        "series_id": str(data.selected_series[i][k]),
                        "instance": int(data.selected_instances[i][k]),
                        "weight": float(values[k]),
                    }
                    for k in top
                ],
            }
        records.append(
            {
                "study_id": data.study_ids[i],
                "split": "validation",
                "selected_source_indices": data.selected_indices[i].tolist(),
                "levels": levels,
            }
        )
    write_json(
        path,
        {
            "scope": "PRIVATE original-validation attention explanation; no confirmed anatomical localization or clinical validity",
            "seeds": list(SEEDS),
            "aggregation": "arithmetic mean of three seed attention weights",
            "records": records,
        },
    )


def _validate_protocol(protocol: dict, data: PairedData):
    exact = {
        "schema_version": 1,
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "classes": list(CLASS_NAMES),
        "levels": list(LEVELS),
        "encoder_training": False,
        "feature_standardization": True,
        "augmentation": False,
        "max_slices": 96,
        "position_size": 2,
        "hidden_size": 64,
        "dropout": 0.2,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 32,
        "max_epochs": 40,
        "patience": 7,
        "cv_folds": 3,
        "cv_seed": 2026,
        "feature_cache_id": data.sagittal.metadata["cache_id"],
        "sagittal_feature_cache_id": data.sagittal.metadata["cache_id"],
        "normalization_epsilon": 1e-6,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 2026,
    }
    for field, expected in exact.items():
        if protocol.get(field) != expected or type(protocol.get(field)) is not type(expected):
            raise ValueError(f"Fixed multimodal protocol mismatch: {field}")


def run_multimodal(sag_features, axial_features, protocol, out, threads=4) -> dict:
    """Run the fixed 27 CV + nine final CPU fits, without accessing TEST."""
    protocol_path, out = Path(protocol), Path(out)
    if out.exists():
        raise ValueError("Output already exists; never overwrite a multimodal experiment")
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be positive")
    began = time.perf_counter()
    config = json.loads(protocol_path.read_text())
    data = load_paired_data(sag_features, axial_features)
    _validate_protocol(config, data)
    folds = make_folds(data.sagittal, seed=config["cv_seed"])
    out.mkdir(parents=True)
    write_json(out / "folds.json", folds)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    provenance = {
        "protocol_sha256": sha256_file(protocol_path),
        "implementation_sha256": sha256_file(Path(__file__)),
        "sagittal_feature_cache_sha256": data.sagittal.file_sha256,
        "axial_metadata_sha256": data.axial_metadata_sha256,
        "axial_output_inventory_sha256": data.axial_output_sha256,
        "folds_sha256": sha256_file(out / "folds.json"),
    }
    configuration = {
        "protocol": config,
        "architectures": {v: architecture(v) for v in VARIANTS},
        "slice_sampling": "floor(linspace(0,N-1,96)) in cached frame-group/slice order when N>96; otherwise all slices; masks trimmed per batch; no global z comparison between unrelated frames",
        "normalization": "Fitting TRAIN only; paired sagittal targets and valid slices of fitting paired studies; population std floored at epsilon; position channels unchanged",
        "runtime": {
            "device": "cpu",
            "threads": threads,
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
        },
        **provenance,
    }
    audit = availability(data)
    write_json(
        out / "manifest.json",
        {
            "schema_version": 1,
            "configuration": configuration,
            "availability": audit,
            "scope": "TRAIN CV and previously used validation development only; TEST not loaded",
        },
    )
    train = np.flatnonzero(data.splits == "train")
    validation = np.flatnonzero(data.splits == "validation")
    study_fold = {row["study_id"]: row["fold"] for row in folds["assignments"]}
    variants = {variant: {"cv": {"folds": []}} for variant in VARIANTS}
    oof = {variant: {seed: [] for seed in SEEDS} for variant in VARIANTS}
    epochs = {variant: {str(seed): [] for seed in SEEDS} for variant in VARIANTS}
    for fold in range(3):
        fitting = np.asarray([i for i in train if study_fold[data.study_ids[i]] != fold], np.int64)
        held = np.asarray([i for i in train if study_fold[data.study_ids[i]] == fold], np.int64)
        normalizers = fit_normalizers(data, fitting, epsilon=config["normalization_epsilon"])
        for variant in VARIANTS:
            runs, predictions = [], []
            for seed in SEEDS:
                model, history, best_epoch, weights = _fit(
                    variant,
                    seed,
                    data,
                    fitting,
                    normalizers,
                    config,
                    epochs=config["max_epochs"],
                    held=held,
                )
                directory = out / "cv" / variant / f"fold-{fold}" / f"seed-{seed}"
                result, rows, _ = _export_run(
                    directory,
                    model,
                    history,
                    best_epoch,
                    seed,
                    data,
                    held,
                    normalizers,
                    weights,
                    provenance,
                    scope="TRAIN CV held fold; used for epoch selection",
                    fold=fold,
                )
                runs.append(result)
                predictions.append(rows)
                oof[variant][seed].extend(rows)
                epochs[variant][str(seed)].append(best_epoch)
            ensemble = _export_ensemble(
                out / "cv" / variant / f"fold-{fold}" / "ensemble",
                ensemble_rows(predictions),
                "TRAIN CV held-fold uniform ensemble",
            )
            variants[variant]["cv"]["folds"].append(
                {"fold": fold, "runs": runs, "ensemble": ensemble}
            )
    final_epochs = {
        variant: {seed: int(np.median(values)) for seed, values in selected.items()}
        for variant, selected in epochs.items()
    }
    # Freeze all epoch counts before the first original-validation forward pass.
    write_json(
        out / "final-epochs.json",
        {
            "final_epochs": final_epochs,
            "created_before_validation": True,
            "validation_used_for_epoch_selection": False,
            **provenance,
        },
    )
    expected = {_key(row) for row in data.targets if row["split"] == "train"}
    for variant in VARIANTS:
        runs = []
        for seed in SEEDS:
            rows = sorted(oof[variant][seed], key=_key)
            if len(rows) != len(expected) or {_key(r) for r in rows} != expected:
                raise RuntimeError("OOF rows do not cover each requested TRAIN target exactly once")
            directory = out / "cv" / variant / "oof" / f"seed-{seed}"
            directory.mkdir(parents=True)
            _write_predictions(directory / "predictions.csv", rows)
            metrics = _metrics(rows, "Pooled TRAIN OOF development; early-stopping folds reused")
            write_json(directory / "metrics.json", metrics)
            runs.append(
                {
                    "seed": seed,
                    "metrics": metrics,
                    "predictions_sha256": sha256_file(directory / "predictions.csv"),
                }
            )
        ensemble = _export_ensemble(
            out / "cv" / variant / "oof" / "ensemble",
            ensemble_rows([oof[variant][s] for s in SEEDS]),
            "Pooled TRAIN OOF uniform ensemble; development",
        )
        cv = variants[variant]["cv"]
        cv["oof"] = {"runs": runs, "ensemble": ensemble}
        cv["mean_fold_ensemble_macro_f1"] = float(
            np.mean([f["ensemble"]["metrics"]["conditional"]["macro_f1"] for f in cv["folds"]])
        )
        write_json(out / "cv" / variant / "summary.json", cv)
    normalizers = fit_normalizers(data, train, epsilon=config["normalization_epsilon"])
    ensemble_validation = {}
    for variant in VARIANTS:
        runs, predictions, attention_sum = [], [], None
        for seed in SEEDS:
            model, history, epoch, weights = _fit(
                variant,
                seed,
                data,
                train,
                normalizers,
                config,
                epochs=final_epochs[variant][str(seed)],
            )
            directory = out / "validation" / variant / f"seed-{seed}"
            result, rows, attention = _export_run(
                directory,
                model,
                history,
                epoch,
                seed,
                data,
                validation,
                normalizers,
                weights,
                provenance,
                scope="Previously used original validation; one development check after TRAIN-CV epoch fixation",
                fold=None,
            )
            runs.append(result)
            predictions.append(rows)
            if variant != "sagittal":
                attention_sum = (
                    attention.astype(np.float64)
                    if attention_sum is None
                    else attention_sum + attention
                )
        ensemble_rows_out = ensemble_rows(predictions)
        ensemble = _export_ensemble(
            out / "validation" / variant / "ensemble",
            ensemble_rows_out,
            "Previously used original validation uniform ensemble; development",
        )
        ensemble_validation[variant] = [
            row for row in ensemble_rows_out if row["label"] is not None
        ]
        scores = [run["macro_f1"] for run in runs]
        variants[variant]["validation"] = {
            "runs": runs,
            "ensemble": ensemble,
            "macro_f1_mean": float(np.mean(scores)),
            "macro_f1_std": float(np.std(scores, ddof=1)),
            "std_definition": "sample SD across all three fixed seeds, not a confidence interval",
        }
        if attention_sum is not None:
            attention_path = out / "validation" / variant / "attention.json"
            _attention_report(attention_path, data, validation, attention_sum / 3)
            variants[variant]["validation"]["attention_sha256"] = sha256_file(attention_path)
        write_json(out / "validation" / variant / "summary.json", variants[variant]["validation"])
    bootstrap = paired_study_bootstrap(
        ensemble_validation,
        contrasts={
            "fusion_minus_sagittal": (["fusion"], ["sagittal"]),
            "axial_minus_sagittal": (["axial"], ["sagittal"]),
        },
        resamples=config["bootstrap_resamples"],
        seed=config["bootstrap_seed"],
    )
    bootstrap["by_level"] = {
        level: paired_study_bootstrap(
            {
                name: [row for row in rows if row["level"] == level]
                for name, rows in ensemble_validation.items()
            },
            contrasts={
                "fusion_minus_sagittal": (["fusion"], ["sagittal"]),
                "axial_minus_sagittal": (["axial"], ["sagittal"]),
            },
            resamples=config["bootstrap_resamples"],
            seed=config["bootstrap_seed"],
        )
        for level in LEVELS
    }
    write_json(out / "bootstrap.json", bootstrap)
    if (
        sha256_file(protocol_path) != provenance["protocol_sha256"]
        or sha256_file(Path(__file__)) != provenance["implementation_sha256"]
        or sha256_file(Path(axial_features) / "metadata.json") != data.axial_metadata_sha256
        or any(
            sha256_file(Path(sag_features) / name) != digest
            for name, digest in data.sagittal.file_sha256.items()
        )
    ):
        raise RuntimeError("Protocol, implementation or source caches changed during training")
    summary = {
        "schema_version": 1,
        "scope": "TRAIN early-stopped CV and previously used original-validation development comparison; not an independent test",
        "configuration": configuration,
        "variants": variants,
        "availability": audit,
        "final_epochs": final_epochs,
        "bootstrap": bootstrap,
        "all_seeds_reported": True,
        "all_variants_share_paired_cohort": True,
        "validation_used_for_epoch_selection": False,
        "test_evaluated": False,
        "new_version_independently_tested": False,
        "attention_is_confirmed_localization": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - began,
    }
    write_json(out / "summary.json", summary)
    return summary
