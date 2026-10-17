"""Research evaluation and CPU training of a new head on frozen SpineNet features.

These routines never load test data or update the original encoder. Cache arrays
are joined to the complete target manifest by identifiers, never by row order.
Predictions contain deidentified dataset identifiers and must remain local.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .assets import write_json
from .data import CLASS_NAMES

SEEDS = (42, 2024, 3407)
LEVELS = ("L4-L5", "L5-S1")
FEATURE_SIZE = 512


@dataclass(frozen=True)
class FeatureData:
    arrays: dict[str, np.ndarray]
    targets: list[dict[str, Any]]
    metadata: dict[str, Any]
    file_sha256: dict[str, str]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["study_id"]), str(row["series_id"]), str(row["level"])


def load_feature_data(feature_dir: str | Path) -> FeatureData:
    """Validate all cache checksums and the exact array/manifest alignment."""
    root = Path(feature_dir)
    metadata = json.loads((root / "metadata.json").read_text())
    if not isinstance(metadata, dict) or metadata.get("extraction_complete") is not True:
        raise ValueError("Feature extraction is incomplete; finish all requested series first")
    if not isinstance(metadata.get("outputs_sha256"), dict):
        raise ValueError("Feature cache metadata is missing output checksums")
    hashes = {name: _sha(root / name) for name in ("features.npz", "target_manifest.csv")}
    for name, digest in hashes.items():
        if metadata.get("outputs_sha256", {}).get(name) != digest:
            raise ValueError(f"Feature cache checksum mismatch: {name}; rebuild a complete cache")
    hashes["metadata.json"] = _sha(root / "metadata.json")
    with np.load(root / "features.npz", allow_pickle=False) as archive:
        names = ("features", "original_logits", "labels", "study_id", "series_id", "level", "split")
        if not set(names).issubset(archive.files):
            raise ValueError("Feature cache is missing required arrays")
        arrays = {name: archive[name].copy() for name in names}
    features, logits, labels = arrays["features"], arrays["original_logits"], arrays["labels"]
    count = len(features)
    if features.shape != (count, FEATURE_SIZE) or features.dtype != np.float32:
        raise ValueError("Frozen features must have shape (N,512) and dtype float32")
    if logits.shape != (count, 4) or not np.issubdtype(logits.dtype, np.floating):
        raise ValueError("Original CCS logits must have shape (N,4)")
    if not np.isfinite(features).all() or not np.isfinite(logits).all():
        raise ValueError("Feature cache contains nonfinite values")
    if labels.shape != (count,) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Feature labels must be an integer array with shape (N,)")
    if not np.isin(labels, (0, 1, 2)).all():
        raise ValueError("Feature labels must be 0=normal_mild,1=moderate,2=severe")
    for name in ("study_id", "series_id", "level", "split"):
        if arrays[name].shape != (count,) or arrays[name].dtype.kind not in "US":
            raise ValueError(f"Cache {name} must be a string array with shape (N,)")
    if not set(arrays["split"].tolist()).issubset({"train", "validation"}):
        raise ValueError(
            "Only train and validation features are permitted; test must remain untouched"
        )
    feature_by_key: dict[tuple[str, str, str], int] = {}
    for index in range(count):
        key = tuple(str(arrays[name][index]) for name in ("study_id", "series_id", "level"))
        if key in feature_by_key:
            raise ValueError("Duplicate feature identifier")
        feature_by_key[key] = index
    with (root / "target_manifest.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"study_id", "series_id", "split", "level", "label", "status", "reason"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("Target manifest is missing required columns")
        raw_targets = list(reader)
    targets: list[dict[str, Any]] = []
    manifest_keys: set[tuple[str, str, str]] = set()
    patient_levels: set[tuple[str, str]] = set()
    study_splits: dict[str, str] = {}
    matched: set[tuple[str, str, str]] = set()
    for raw in raw_targets:
        row = dict(raw)
        key = _key(row)
        if key in manifest_keys or (key[0], key[2]) in patient_levels:
            raise ValueError("Manifest duplicates a study/level; choose one series per study first")
        manifest_keys.add(key)
        patient_levels.add((key[0], key[2]))
        if row["split"] not in {"train", "validation"}:
            raise ValueError(
                "Only train and validation targets are permitted; test must remain untouched"
            )
        if key[0] in study_splits and study_splits[key[0]] != row["split"]:
            raise ValueError("A study appears in both train and validation")
        study_splits[key[0]] = row["split"]
        if key[2] not in LEVELS:
            raise ValueError("Unexpected target level")
        try:
            row["label"] = None if row["label"] == "" else int(row["label"])
        except (TypeError, ValueError):
            raise ValueError("Invalid target label") from None
        if row["label"] is not None and row["label"] not in (0, 1, 2):
            raise ValueError("Invalid target severity category")
        if not row["status"]:
            raise ValueError("Every requested target needs an explicit status")
        if row["status"] in {"pending", "not_downloaded", "not_processed"}:
            raise ValueError("Feature extraction contains unfinished targets")
        index = feature_by_key.get(key)
        if row["status"] == "ok":
            if row["label"] is None or index is None:
                raise ValueError("Assessed target lacks a label or a matching feature row")
            if arrays["split"][index] != row["split"] or int(labels[index]) != row["label"]:
                raise ValueError("Feature labels/splits disagree with target identifiers")
            matched.add(key)
            row["feature_index"] = index
        elif index is not None:
            raise ValueError("An unassessed target unexpectedly has cached features")
        else:
            row["feature_index"] = None
        targets.append(row)
    if matched != set(feature_by_key):
        raise ValueError("Feature arrays contain rows absent from the complete target manifest")
    return FeatureData(arrays, targets, metadata, hashes)


def collapse_stenosis_probabilities(logits: np.ndarray) -> np.ndarray:
    """Merge none+mild probability mass before taking a three-category argmax."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all():
        raise ValueError("Expected finite original stenosis logits with shape (N,4)")
    shifted = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return np.column_stack((probabilities[:, :2].sum(axis=1), probabilities[:, 2:]))


def _score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labeled = [row for row in rows if row["label"] is not None]
    assessed = [row for row in labeled if row["status"] == "ok"]
    truth = np.asarray([row["label"] for row in assessed], dtype=np.int64)
    predicted = np.asarray([row["prediction"] for row in assessed], dtype=np.int64)
    if assessed:
        precision, recall, f1, support = precision_recall_fscore_support(
            truth, predicted, labels=[0, 1, 2], zero_division=0
        )
        matrix = confusion_matrix(truth, predicted, labels=[0, 1, 2])
    else:
        precision, recall, f1 = (np.zeros(3) for _ in range(3))
        support = np.zeros(3, dtype=np.int64)
        matrix = np.zeros((3, 3), dtype=np.int64)
    severe = [row for row in labeled if row["label"] == 2]
    severe_assessed = [row for row in severe if row["status"] == "ok"]
    severe_correct = sum(row["prediction"] == 2 for row in severe_assessed)
    return {
        "n_requested": len(rows),
        "n_labeled": len(labeled),
        "n_missing_reference_label": len(rows) - len(labeled),
        "n_assessed": len(assessed),
        "n_not_assessed": len(labeled) - len(assessed),
        "coverage": len(assessed) / len(labeled) if labeled else None,
        "class_counts": {
            name: sum(row["label"] == grade for row in labeled)
            for grade, name in enumerate(CLASS_NAMES)
        },
        "assessed_class_counts": dict(zip(CLASS_NAMES, support.astype(int).tolist())),
        "not_assessed_class_counts": {
            name: sum(row["label"] == grade and row["status"] != "ok" for row in labeled)
            for grade, name in enumerate(CLASS_NAMES)
        },
        "conditional": {
            "macro_f1": float(f1.mean()) if assessed else None,
            "accuracy": float(accuracy_score(truth, predicted)) if assessed else None,
            "precision": dict(zip(CLASS_NAMES, precision.tolist())),
            "recall": dict(zip(CLASS_NAMES, recall.tolist())),
            "f1": dict(zip(CLASS_NAMES, f1.tolist())),
            "confusion_matrix": matrix.tolist(),
            "confusion_axes": "rows=true, columns=predicted; normal_mild,moderate,severe",
        },
        "severe": {
            "n_total": len(severe),
            "n_assessed": len(severe_assessed),
            "correctly_predicted_severe": severe_correct,
            "predicted_normal_mild": sum(row["prediction"] == 0 for row in severe_assessed),
            "predicted_moderate": sum(row["prediction"] == 1 for row in severe_assessed),
            "not_assessed": len(severe) - len(severe_assessed),
            "recall_conditional": severe_correct / len(severe_assessed)
            if severe_assessed
            else None,
            "detection_rate_all_requested": severe_correct / len(severe) if severe else None,
        },
    }


def evaluate_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count abstentions separately from classification errors and reference gaps."""
    for row in rows:
        if row["label"] is not None and row["label"] not in (0, 1, 2):
            raise ValueError("Unexpected reference category")
        if row["status"] == "ok" and row.get("prediction") not in (0, 1, 2):
            raise ValueError("An assessed prediction must belong to one of the three categories")
    return {
        **_score(rows),
        "by_level": {
            level: _score([row for row in rows if row["level"] == level]) for level in LEVELS
        },
        "scope": "Cached validation targets only; classification metrics condition on assessed targets. Not a test-set result or clinical validation.",
    }


def _prediction_rows(data: FeatureData, probabilities: np.ndarray, *, original: bool = False):
    rows = []
    for target in data.targets:
        if target["split"] != "validation":
            continue
        row = {
            name: target[name]
            for name in ("study_id", "series_id", "level", "split", "label", "status", "reason")
        }
        row.update(prediction=None, p_normal_mild=None, p_moderate=None, p_severe=None)
        index = target["feature_index"]
        if index is not None:
            score = probabilities[index]
            row["prediction"] = int(score.argmax())
            for name, value in zip(CLASS_NAMES, score):
                row[f"p_{name}"] = float(value)
            if original:
                row["native_ccs_4grade"] = int(data.arrays["original_logits"][index].argmax()) + 1
        rows.append(row)
    return rows


def _write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "study_id",
        "series_id",
        "level",
        "split",
        "label",
        "status",
        "reason",
        "prediction",
        "p_normal_mild",
        "p_moderate",
        "p_severe",
    ]
    if any("native_ccs_4grade" in row for row in rows):
        fields.append("native_ccs_4grade")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def _new_output(output_dir: str | Path) -> Path:
    out = Path(output_dir)
    if out.exists():
        raise ValueError("Output directory already exists; choose a new experiment directory")
    out.mkdir(parents=True)
    return out


def evaluate_baseline(feature_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Evaluate the original four-class head, with a probability-correct collapse."""
    data = load_feature_data(feature_dir)
    if not any(row["split"] == "validation" for row in data.targets):
        raise ValueError("The cache has no validation targets")
    probabilities = collapse_stenosis_probabilities(data.arrays["original_logits"])
    rows = _prediction_rows(data, probabilities, original=True)
    metrics = evaluate_predictions(rows)
    out = _new_output(output_dir)
    _write_predictions(out / "predictions.csv", rows)
    write_json(out / "metrics.json", metrics)
    write_json(
        out / "manifest.json",
        {
            "schema_version": 1,
            "model": "original_SpineNet_ccs_four_classes",
            "class_names": CLASS_NAMES,
            "mapping": "softmax(logits4), then p_none+p_mild,p_moderate,p_severe; argmax after merging",
            "clinical_scale_equivalence_verified": False,
            "feature_cache_sha256": data.file_sha256,
            "feature_cache_metadata": data.metadata,
            "predictions_sha256": _sha(out / "predictions.csv"),
            "scope": "Research validation evaluation; test not loaded",
        },
    )
    return metrics


def inverse_sqrt_class_weights(labels: np.ndarray) -> np.ndarray:
    """Derive weights from successfully cached training targets only."""
    labels = np.asarray(labels)
    if (
        labels.ndim != 1
        or not np.issubdtype(labels.dtype, np.integer)
        or not np.isin(labels, [0, 1, 2]).all()
    ):
        raise ValueError("Expected integer training labels from the three categories")
    counts = np.bincount(labels, minlength=3)
    if np.any(counts == 0):
        raise ValueError(
            "Frozen-head training requires at least one assessed training target per class"
        )
    weights = 1 / np.sqrt(counts.astype(np.float64))
    return (weights / weights.mean()).astype(np.float32)


def predict_head(features: np.ndarray, checkpoint_path: str | Path) -> np.ndarray:
    """Reload a local exported head and return uncalibrated category scores.

    The caller must use the identical frozen encoder/preprocessing recorded in
    the checkpoint. A 512-dimensional vector alone cannot prove its provenance.
    """
    import torch
    from torch import nn

    values = np.asarray(features)
    if values.ndim != 2 or values.shape[1] != FEATURE_SIZE or not np.isfinite(values).all():
        raise ValueError("Expected finite frozen features with shape (N,512)")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("model") != "Linear(512,3)"
        or checkpoint.get("feature_size") != FEATURE_SIZE
        or checkpoint.get("class_names") != list(CLASS_NAMES)
    ):
        raise ValueError("Checkpoint model dimensions or class order do not match")
    head = nn.Linear(FEATURE_SIZE, 3)
    head.load_state_dict(checkpoint["state_dict"], strict=True)
    head.eval()
    with torch.no_grad():
        return head(torch.from_numpy(values.astype(np.float32))).softmax(dim=1).numpy()


def train_head(
    feature_dir: str | Path,
    output_dir: str | Path,
    *,
    seeds: tuple[int, ...] = SEEDS,
    max_epochs: int = 20,
    patience: int = 5,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    threads: int = 4,
) -> dict[str, Any]:
    """Train only Linear(512,3), using one identical frozen cache for all seeds."""
    import torch
    from torch import nn

    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(not isinstance(seed, int) or seed < 0 for seed in seeds)
    ):
        raise ValueError("Use distinct non-negative integer seeds")
    if min(max_epochs, patience, batch_size, threads) < 1:
        raise ValueError("Epochs, patience, batch size and CPU threads must be positive")
    if (
        not math.isfinite(learning_rate)
        or learning_rate <= 0
        or not math.isfinite(weight_decay)
        or weight_decay < 0
    ):
        raise ValueError("Learning rate must be positive and weight decay non-negative")
    data = load_feature_data(feature_dir)
    train_indices = np.flatnonzero(data.arrays["split"] == "train")
    validation_indices = np.flatnonzero(data.arrays["split"] == "validation")
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("Assessed train and validation features are both required")
    class_weights = inverse_sqrt_class_weights(data.arrays["labels"][train_indices])
    out = _new_output(output_dir)
    torch.set_num_threads(threads)
    x = torch.from_numpy(data.arrays["features"]).clone()
    y = torch.from_numpy(data.arrays["labels"].astype(np.int64)).clone()
    train_index = torch.from_numpy(train_indices)
    validation_index = torch.from_numpy(validation_indices)
    train_x, train_y = x[train_index], y[train_index]
    weighted_loss = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights))
    validation_loss = nn.CrossEntropyLoss()
    seed_results = []
    config = {
        "model": "Linear(512,3)",
        "encoder_updated": False,
        "seeds": list(seeds),
        "max_epochs": max_epochs,
        "patience": patience,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "threads": threads,
        "optimizer": "AdamW",
        "class_weights": class_weights.tolist(),
        "class_weight_rule": "inverse sqrt assessed training counts, normalised to mean 1",
        "assessed_training_class_counts": np.bincount(train_y.numpy(), minlength=3).tolist(),
        "n_training_features": len(train_indices),
        "n_validation_features": len(validation_indices),
        "selection": "strictly higher validation macro-F1; earliest epoch wins ties; fixed labels [0,1,2]",
        "normalization": "original cached SpineNet features, no additional fitted transformation",
        "implementation_sha256": _sha(Path(__file__)),
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "device": "cpu",
            "threads": torch.get_num_threads(),
        },
    }
    write_json(
        out / "manifest.json",
        {
            "schema_version": 1,
            "class_names": CLASS_NAMES,
            "configuration": config,
            "feature_cache_sha256": data.file_sha256,
            "feature_cache_metadata": data.metadata,
            "scope": "Research validation selection; test not loaded; no clinical accuracy claim",
        },
    )
    for seed in seeds:
        seed_out = out / "runs" / f"seed-{seed}"
        seed_out.mkdir(parents=True)
        torch.manual_seed(seed)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        head = nn.Linear(FEATURE_SIZE, 3)
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        history = []
        best_score, best_epoch, stale = -math.inf, 0, 0
        best_state = None
        for epoch in range(1, max_epochs + 1):
            head.train()
            permutation = torch.randperm(len(train_x), generator=generator)
            for offset in range(0, len(permutation), batch_size):
                indices = permutation[offset : offset + batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = weighted_loss(head(train_x[indices]), train_y[indices])
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                loss.backward()
                optimizer.step()
            head.eval()
            with torch.no_grad():
                logits = head(x)
                probabilities = logits.softmax(dim=1).numpy()
                training_loss = float(weighted_loss(logits[train_index], y[train_index]))
                val_loss = float(validation_loss(logits[validation_index], y[validation_index]))
            rows = _prediction_rows(data, probabilities)
            metrics = evaluate_predictions(rows)
            score = metrics["conditional"]["macro_f1"]
            history.append(
                {
                    "epoch": epoch,
                    "training_loss": training_loss,
                    "validation_loss_unweighted": val_loss,
                    "validation_macro_f1": score,
                }
            )
            if score > best_score + 1e-12:
                best_score, best_epoch, stale = score, epoch, 0
                best_state = {
                    name: value.detach().clone() for name, value in head.state_dict().items()
                }
            else:
                stale += 1
            if stale >= patience:
                break
        if best_state is None:
            raise RuntimeError("No valid frozen-head checkpoint was produced")
        head.load_state_dict(best_state)
        head.eval()
        with torch.no_grad():
            probabilities = head(x).softmax(dim=1).numpy()
        rows = _prediction_rows(data, probabilities)
        metrics = evaluate_predictions(rows)
        _write_predictions(seed_out / "predictions.csv", rows)
        write_json(seed_out / "metrics.json", metrics)
        write_json(seed_out / "history.json", history)
        with (seed_out / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        torch.save(
            {
                "schema_version": 1,
                "state_dict": best_state,
                "model": "Linear(512,3)",
                "class_names": list(CLASS_NAMES),
                "feature_size": FEATURE_SIZE,
                "seed": seed,
                "best_epoch": best_epoch,
                "feature_cache_sha256": data.file_sha256,
            },
            seed_out / "head.pt",
        )
        result = {
            "seed": seed,
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "validation_macro_f1": best_score,
            "checkpoint_sha256": _sha(seed_out / "head.pt"),
            "metrics": metrics,
        }
        seed_results.append(result)
        print(
            f"Frozen head seed={seed}: best epoch={best_epoch}, validation macro-F1={best_score:.4f}",
            flush=True,
        )
    scores = [result["validation_macro_f1"] for result in seed_results]
    summary = {
        "configuration": config,
        "runs": seed_results,
        "validation_macro_f1_mean": float(np.mean(scores)),
        "validation_macro_f1_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        "std_definition": "sample standard deviation across seeds; not a confidence interval",
        "all_seeds_reported": True,
        "test_evaluated": False,
    }
    write_json(out / "summary.json", summary)
    return summary
