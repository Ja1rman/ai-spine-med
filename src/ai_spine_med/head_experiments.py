"""Small frozen-feature head experiments using original train/validation only.

The three-fold scores are development-selection estimates: each held fold is
also used to stop training. Candidate and final epoch counts are fixed from CV
before any original-validation forward pass. The already opened v1 test is
neither loaded nor reused here. Predictions and fold identifiers stay local.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.nn import functional as F

from .assets import write_json
from .data import CLASS_NAMES
from .training import (
    FeatureData,
    evaluate_predictions,
    inverse_sqrt_class_weights,
    load_feature_data,
)

CANDIDATES = ("linear", "mlp", "coral")
SEEDS = (42, 2024, 3407)
FEATURE_SIZE = 512
CV_FOLDS = 3


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _key(row: dict) -> tuple[str, str, str]:
    return tuple(str(row[name]) for name in ("study_id", "series_id", "level"))


class CoralHead(nn.Module):
    """One shared severity score and two monotonically ordered thresholds."""

    def __init__(self) -> None:
        super().__init__()
        self.score = nn.Linear(FEATURE_SIZE, 1, bias=False)
        self.threshold0 = nn.Parameter(torch.tensor(-0.5))
        self.raw_gap = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))

    def thresholds(self) -> torch.Tensor:
        return torch.stack((self.threshold0, self.threshold0 + F.softplus(self.raw_gap)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.score(features) - self.thresholds()


def architecture(name: str) -> dict[str, Any]:
    if name == "linear":
        return {"name": name, "input_size": 512, "output_size": 3}
    if name == "mlp":
        return {
            "name": name,
            "input_size": 512,
            "hidden_size": 64,
            "activation": "ReLU",
            "dropout": 0.2,
            "output_size": 3,
        }
    if name == "coral":
        return {
            "name": name,
            "input_size": 512,
            "score_size": 1,
            "score_bias": False,
            "thresholds": "b,b+softplus(raw_gap)",
            "output_size": 3,
        }
    raise ValueError("Unknown frozen-feature head architecture")


def build_head(name: str) -> nn.Module:
    architecture(name)
    if name == "linear":
        return nn.Linear(FEATURE_SIZE, 3)
    if name == "mlp":
        return nn.Sequential(nn.Linear(512, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 3))
    return CoralHead()


def head_probabilities(name: str, logits: torch.Tensor) -> torch.Tensor:
    if name == "coral":
        if logits.ndim != 2 or logits.shape[1] != 2:
            raise ValueError("CORAL requires two ordered cumulative logits")
        q = logits.sigmoid()
        return torch.stack((1 - q[:, 0], q[:, 0] - q[:, 1], q[:, 1]), dim=1)
    if name not in CANDIDATES or logits.ndim != 2 or logits.shape[1] != 3:
        raise ValueError("Categorical heads require three logits")
    return logits.softmax(dim=1)


def weighted_head_loss(
    name: str, logits: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor
) -> torch.Tensor:
    """Normalize each batch by its sample class weights for both loss families."""
    if name == "coral":
        target = labels[:, None] > torch.arange(2, device=labels.device)[None, :]
        losses = F.binary_cross_entropy_with_logits(
            logits, target.to(logits.dtype), reduction="none"
        ).mean(dim=1)
    elif name in CANDIDATES:
        losses = F.cross_entropy(logits, labels, reduction="none")
    else:
        raise ValueError("Unknown head loss")
    sample_weights = class_weights[labels]
    return (losses * sample_weights).sum() / sample_weights.sum()


def predict_experiment_head(features: np.ndarray, checkpoint_path: str | Path) -> np.ndarray:
    values = np.asarray(features)
    if (
        values.ndim != 2
        or values.shape[1] != FEATURE_SIZE
        or not np.issubdtype(values.dtype, np.floating)
        or not np.isfinite(values).all()
    ):
        raise ValueError("Expected finite floating-point frozen features with shape (N,512)")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    name = checkpoint.get("candidate")
    if (
        name not in CANDIDATES
        or checkpoint.get("schema_version") != 1
        or checkpoint.get("class_names") != list(CLASS_NAMES)
        or checkpoint.get("architecture") != architecture(name)
        or checkpoint.get("feature_size") != FEATURE_SIZE
    ):
        raise ValueError("Checkpoint architecture or class order does not match")
    model = build_head(name)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    with torch.inference_mode():
        probabilities = head_probabilities(name, model(torch.from_numpy(values.astype(np.float32))))
    result = probabilities.numpy()
    _validate_probabilities(result, len(values))
    return result


def _validate_probabilities(values: np.ndarray, count: int) -> None:
    if (
        values.shape != (count, 3)
        or not np.isfinite(values).all()
        or (values < 0).any()
        or not np.allclose(values.sum(axis=1), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("Invalid three-class probability distribution")


def make_folds(data: FeatureData, *, seed: int = 2026) -> dict[str, Any]:
    """Assign every original TRAIN study, including all unassessed targets."""
    targets = [row for row in data.targets if row["split"] == "train"]
    validation_studies = {row["study_id"] for row in data.targets if row["split"] == "validation"}
    all_studies = {row["study_id"] for row in targets}
    if not targets or not all_studies.isdisjoint(validation_studies):
        raise ValueError("CV requires disjoint original TRAIN and validation studies")
    labeled = [row for row in targets if row["label"] is not None]
    groups = np.asarray([row["study_id"] for row in labeled])
    labels = np.asarray([row["label"] for row in labeled], dtype=np.int64)
    if len(set(groups)) < CV_FOLDS or set(labels.tolist()) != {0, 1, 2}:
        raise ValueError("Three grouped folds need labeled TRAIN studies and all three classes")
    splitter = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    study_fold: dict[str, int] = {}
    for fold, (_, heldout) in enumerate(splitter.split(np.zeros(len(labels)), labels, groups)):
        for study in set(groups[heldout].tolist()):
            if study in study_fold:
                raise ValueError("A study occurs in more than one held fold")
            study_fold[study] = fold
    # A future manifest can include studies with no reference at either level.
    # Keep them assigned locally without inventing a severity stratum or label.
    missing_studies = sorted(all_studies - set(study_fold))
    rng = np.random.default_rng(seed)
    for study in rng.permutation(missing_studies).tolist():
        counts = Counter(study_fold.values())
        study_fold[study] = min(range(CV_FOLDS), key=lambda fold: (counts[fold], fold))
    if set(study_fold) != all_studies:
        raise ValueError("CV did not assign every original TRAIN study")
    assignments = [
        {
            **{
                name: row[name]
                for name in ("study_id", "series_id", "level", "label", "status", "reason")
            },
            "fold": study_fold[row["study_id"]],
        }
        for row in sorted(targets, key=_key)
    ]
    if len({_key(row) for row in assignments}) != len(assignments):
        raise ValueError("Duplicate CV target key")
    folds = []
    for fold in range(CV_FOLDS):
        fit_studies = {study for study, assigned in study_fold.items() if assigned != fold}
        held_studies = {study for study, assigned in study_fold.items() if assigned == fold}
        if fit_studies & held_studies or fit_studies | held_studies != all_studies:
            raise ValueError("CV study grouping is not a disjoint partition")
        fit_rows = [row for row in targets if row["study_id"] in fit_studies]
        held_rows = [row for row in targets if row["study_id"] in held_studies]
        fit_labels = [row["label"] for row in fit_rows if row["feature_index"] is not None]
        held_labels = [row["label"] for row in held_rows if row["feature_index"] is not None]
        if set(fit_labels) != {0, 1, 2} or not held_labels:
            raise ValueError("A CV fitting fold lacks an assessed class or an assessed held target")
        folds.append(
            {
                "fold": fold,
                "n_fitting_studies": len(fit_studies),
                "n_heldout_studies": len(held_studies),
                "n_fitting_targets": len(fit_rows),
                "n_heldout_targets": len(held_rows),
                "n_fitting_features": len(fit_labels),
                "n_heldout_features": len(held_labels),
                "fitting_class_counts": np.bincount(fit_labels, minlength=3).tolist(),
                "heldout_class_counts": np.bincount(held_labels, minlength=3).tolist(),
            }
        )
    return {
        "schema_version": 1,
        "scope": "Original TRAIN only; target identifiers remain local",
        "algorithm": "StratifiedGroupKFold over complete labeled targets; groups=study_id",
        "n_splits": CV_FOLDS,
        "shuffle": True,
        "random_state": seed,
        "feature_cache_sha256": data.file_sha256,
        "n_studies": len(all_studies),
        "n_targets": len(targets),
        "folds": folds,
        "assignments": assignments,
    }


def _indices(targets: list[dict]) -> np.ndarray:
    return np.asarray(
        [row["feature_index"] for row in targets if row["feature_index"] is not None],
        dtype=np.int64,
    )


def _prediction_rows(
    targets: list[dict], probabilities: np.ndarray, indices: np.ndarray
) -> list[dict]:
    _validate_probabilities(probabilities, len(indices))
    if len(set(indices.tolist())) != len(indices) or set(indices.tolist()) != set(
        _indices(targets)
    ):
        raise ValueError("Prediction feature rows do not match the requested target keys")
    by_index = dict(zip(indices.tolist(), probabilities))
    result = []
    for target in targets:
        row = {
            name: target[name]
            for name in ("study_id", "series_id", "level", "split", "label", "status", "reason")
        }
        row.update(prediction=None, p_normal_mild=None, p_moderate=None, p_severe=None)
        if target["feature_index"] is not None:
            values = by_index[target["feature_index"]]
            row["prediction"] = int(values.argmax())
            row.update({f"p_{name}": float(value) for name, value in zip(CLASS_NAMES, values)})
        result.append(row)
    return result


def _metrics(rows: list[dict], scope: str) -> dict:
    metrics = evaluate_predictions(rows)
    metrics["scope"] = (
        scope + "; classification metrics condition on assessed targets; no test evaluation"
    )
    for section in (metrics, *metrics["by_level"].values()):
        matrix = section["conditional"]["confusion_matrix"]
        section["normal_mild_elevated"] = sum(matrix[0][1:])
        section["normal_mild_elevated_rate_conditional"] = (
            sum(matrix[0][1:]) / sum(matrix[0]) if sum(matrix[0]) else None
        )
        section["severe"]["missed_all_requested"] = (
            section["severe"]["n_total"] - section["severe"]["correctly_predicted_severe"]
        )
    return metrics


def _write_predictions(path: Path, rows: list[dict]) -> None:
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fit(
    name: str,
    seed: int,
    x: torch.Tensor,
    y: torch.Tensor,
    fit_indices: np.ndarray,
    *,
    config: dict,
    epochs: int,
    held_indices: np.ndarray | None = None,
) -> tuple[nn.Module, list[dict], int, np.ndarray]:
    """Final fits never receive original validation features or labels."""
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model = build_head(name)
    weights = inverse_sqrt_class_weights(y[fit_indices].numpy())
    weight_tensor = torch.from_numpy(weights)
    fit_x, fit_y = x[fit_indices], y[fit_indices]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    best_state, best_epoch, best_score, stale = None, 0, -math.inf, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(fit_x), generator=generator)
        for offset in range(0, len(permutation), config["batch_size"]):
            selected = permutation[offset : offset + config["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_head_loss(name, model(fit_x[selected]), fit_y[selected], weight_tensor)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite frozen-feature training loss")
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            training_loss = float(weighted_head_loss(name, model(fit_x), fit_y, weight_tensor))
            record = {"epoch": epoch, "training_loss": training_loss}
            if held_indices is not None:
                held_logits = model(x[held_indices])
                scores = head_probabilities(name, held_logits).numpy()
                _validate_probabilities(scores, len(held_indices))
                macro = float(
                    f1_score(
                        y[held_indices].numpy(),
                        scores.argmax(1),
                        labels=[0, 1, 2],
                        average="macro",
                        zero_division=0,
                    )
                )
                record.update(
                    heldout_macro_f1=macro,
                    heldout_loss_unweighted=float(
                        weighted_head_loss(name, held_logits, y[held_indices], torch.ones(3))
                    ),
                )
                if macro > best_score:
                    best_score, best_epoch, stale = macro, epoch, 0
                    best_state = {
                        key: value.detach().clone() for key, value in model.state_dict().items()
                    }
                else:
                    stale += 1
        history.append(record)
        if held_indices is not None and stale >= config["patience"]:
            break
    if held_indices is not None:
        if best_state is None:
            raise RuntimeError("No finite CV checkpoint was selected")
        model.load_state_dict(best_state)
    else:
        best_epoch = epochs
    model.eval()
    return model, history, best_epoch, weights


def _export_run(
    directory: Path,
    model: nn.Module,
    history: list[dict],
    best_epoch: int,
    name: str,
    seed: int,
    fit_indices: np.ndarray,
    targets: list[dict],
    data: FeatureData,
    x: torch.Tensor,
    class_weights: np.ndarray,
    provenance: dict,
    *,
    scope: str,
    fold: int | None,
) -> tuple[dict, list[dict]]:
    directory.mkdir(parents=True)
    indices = _indices(targets)
    checkpoint = {
        "schema_version": 1,
        "candidate": name,
        "architecture": architecture(name),
        "state_dict": {key: value.detach().clone() for key, value in model.state_dict().items()},
        "class_names": list(CLASS_NAMES),
        "feature_size": FEATURE_SIZE,
        "seed": seed,
        "best_epoch": best_epoch,
        "scope": scope,
        "fold": fold,
        "encoder_updated": False,
        "class_weights": class_weights.tolist(),
        "training_class_counts": np.bincount(
            data.arrays["labels"][fit_indices], minlength=3
        ).tolist(),
        **provenance,
    }
    torch.save(checkpoint, directory / "head.pt")
    # Reopened checkpoint, rather than an in-memory-only model, creates every
    # exported selected prediction. This also verifies safe serialization.
    probabilities = predict_experiment_head(x[indices].numpy(), directory / "head.pt")
    rows = _prediction_rows(targets, probabilities, indices)
    metrics = _metrics(rows, scope)
    _write_predictions(directory / "predictions.csv", rows)
    write_json(directory / "metrics.json", metrics)
    write_json(directory / "history.json", history)
    result = {
        "seed": seed,
        "fold": fold,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "macro_f1": metrics["conditional"]["macro_f1"],
        "checkpoint_sha256": _sha(directory / "head.pt"),
        "predictions_sha256": _sha(directory / "predictions.csv"),
        "class_weights": class_weights.tolist(),
        "training_class_counts": checkpoint["training_class_counts"],
        "metrics": metrics,
    }
    write_json(directory / "result.json", result)
    print(
        f"{scope}: {name} seed={seed} fold={fold} epoch={best_epoch} macro-F1={result['macro_f1']:.6f}",
        flush=True,
    )
    return result, rows


def ensemble_rows(seed_rows: list[list[dict]]) -> list[dict]:
    """Uniform probabilities with explicit key/reference/status alignment."""
    if len(seed_rows) != 3:
        raise ValueError("The protocol requires all three equally weighted seeds")
    mappings = [{_key(row): row for row in rows} for rows in seed_rows]
    reference = mappings[0]
    if any(len(mapping) != len(rows) for mapping, rows in zip(mappings, seed_rows)):
        raise ValueError("Duplicate ensemble target key")
    if any(set(mapping) != set(reference) for mapping in mappings[1:]):
        raise ValueError("Ensemble models have different requested target keys")
    output = []
    for key in sorted(reference):
        row = dict(reference[key])
        for mapping in mappings[1:]:
            if any(
                mapping[key][name] != row[name] for name in ("split", "label", "status", "reason")
            ):
                raise ValueError("Ensemble labels or coverage differ between seeds")
        if row["status"] == "ok":
            values = np.asarray(
                [[mapping[key][f"p_{name}"] for name in CLASS_NAMES] for mapping in mappings],
                dtype=np.float64,
            )
            _validate_probabilities(values, 3)
            probabilities = values.mean(axis=0)
            row["prediction"] = int(probabilities.argmax())
            row.update(
                {f"p_{name}": float(value) for name, value in zip(CLASS_NAMES, probabilities)}
            )
        elif any(mapping[key].get("prediction") is not None for mapping in mappings):
            raise ValueError("Unassessed ensemble target unexpectedly has a prediction")
        output.append(row)
    return output


def _export_ensemble(directory: Path, rows: list[dict], scope: str) -> dict:
    directory.mkdir(parents=True)
    _write_predictions(directory / "predictions.csv", rows)
    metrics = _metrics(rows, scope)
    write_json(directory / "metrics.json", metrics)
    return {
        "seeds": list(SEEDS),
        "weights": [1 / 3] * 3,
        "metrics": metrics,
        "predictions_sha256": _sha(directory / "predictions.csv"),
    }


def _validate_protocol(protocol: dict, data: FeatureData) -> None:
    exact = {
        "schema_version": 1,
        "candidates": list(CANDIDATES),
        "seeds": list(SEEDS),
        "classes": list(CLASS_NAMES),
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 64,
        "max_epochs": 30,
        "patience": 5,
        "cv_folds": 3,
        "cv_seed": 2026,
        "feature_standardization": False,
        "encoder_training": False,
        "augmentation": False,
        "feature_cache_id": data.metadata["cache_id"],
    }
    for name, expected in exact.items():
        if protocol.get(name) != expected or type(protocol.get(name)) is not type(expected):
            raise ValueError(f"Frozen v2 protocol mismatch: {name}")


def run_head_experiments(
    feature_dir: str | Path = "artifacts/features-v1",
    out: str | Path = "reports/runs/head-v2",
    *,
    protocol_path: str | Path,
    threads: int = 4,
) -> dict[str, Any]:
    """Train 27 CV heads, select by CV, then refit nine heads without val tuning."""
    if type(threads) is not int or threads < 1:
        raise ValueError("CPU threads must be a positive integer")
    protocol_path, out = Path(protocol_path), Path(out)
    protocol = json.loads(protocol_path.read_text())
    if not isinstance(protocol, dict):
        raise ValueError("Protocol must be a JSON object")
    data = load_feature_data(feature_dir)
    _validate_protocol(protocol, data)
    folds = make_folds(data, seed=protocol["cv_seed"])
    train_targets = [row for row in data.targets if row["split"] == "train"]
    validation_targets = [row for row in data.targets if row["split"] == "validation"]
    if not len(_indices(validation_targets)):
        raise ValueError("A final original-validation check needs assessed validation targets")
    if out.exists():
        raise ValueError("Output directory exists; never overwrite a completed or partial v2 run")
    out.mkdir(parents=True)
    write_json(out / "folds.json", folds)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    provenance = {
        "protocol_sha256": _sha(protocol_path),
        "implementation_sha256": _sha(Path(__file__)),
        "feature_cache_sha256": data.file_sha256,
        "feature_cache_id": data.metadata["cache_id"],
        "folds_sha256": _sha(out / "folds.json"),
    }
    config = {
        "protocol": protocol,
        "class_names": list(CLASS_NAMES),
        "architectures": {name: architecture(name) for name in CANDIDATES},
        "runtime": {
            "python": platform.python_version(),
            "numpy": str(np.__version__),
            "torch": str(torch.__version__),
            "device": "cpu",
            "threads": threads,
            "deterministic_algorithms": True,
        },
        "n_training_studies": folds["n_studies"],
        "n_training_features": len(_indices(train_targets)),
        "n_validation_features": len(_indices(validation_targets)),
        "normalization": "No fitted feature transformation",
        **provenance,
    }
    write_json(
        out / "manifest.json",
        {
            "schema_version": 1,
            "configuration": config,
            "feature_cache_metadata": data.metadata,
            "feature_cache_sha256": data.file_sha256,
            "scope": "Original TRAIN CV and original-validation development check only; no test access",
        },
    )
    x = torch.from_numpy(data.arrays["features"])
    y = torch.from_numpy(data.arrays["labels"].astype(np.int64))
    study_fold = {row["study_id"]: row["fold"] for row in folds["assignments"]}
    candidates: dict[str, dict] = {}
    final_epochs: dict[str, dict[str, int]] = {}
    for name in CANDIDATES:
        fold_results = []
        oof_seed_rows = {seed: [] for seed in SEEDS}
        selected_epochs = {seed: [] for seed in SEEDS}
        for fold in range(CV_FOLDS):
            fitting = [row for row in train_targets if study_fold[row["study_id"]] != fold]
            heldout = [row for row in train_targets if study_fold[row["study_id"]] == fold]
            fit_indices, held_indices = _indices(fitting), _indices(heldout)
            runs, seed_predictions = [], []
            for seed in SEEDS:
                model, history, best_epoch, weights = _fit(
                    name,
                    seed,
                    x,
                    y,
                    fit_indices,
                    config=protocol,
                    epochs=protocol["max_epochs"],
                    held_indices=held_indices,
                )
                directory = out / "cv" / name / f"fold-{fold}" / f"seed-{seed}"
                result, rows = _export_run(
                    directory,
                    model,
                    history,
                    best_epoch,
                    name,
                    seed,
                    fit_indices,
                    heldout,
                    data,
                    x,
                    weights,
                    provenance,
                    scope="TRAIN CV held fold (development epoch selection)",
                    fold=fold,
                )
                runs.append(result)
                seed_predictions.append(rows)
                oof_seed_rows[seed].extend(rows)
                selected_epochs[seed].append(best_epoch)
            ensemble = _export_ensemble(
                out / "cv" / name / f"fold-{fold}" / "ensemble",
                ensemble_rows(seed_predictions),
                "TRAIN CV held-fold ensemble; development selection",
            )
            fold_results.append({"fold": fold, "runs": runs, "ensemble": ensemble})
        expected_keys = {_key(row) for row in train_targets}
        oof_runs = []
        for seed in SEEDS:
            rows = oof_seed_rows[seed]
            if len(rows) != len(expected_keys) or {_key(row) for row in rows} != expected_keys:
                raise RuntimeError(
                    "OOF predictions do not cover each original TRAIN target exactly once"
                )
            directory = out / "cv" / name / "oof" / f"seed-{seed}"
            directory.mkdir(parents=True)
            rows.sort(key=_key)
            _write_predictions(directory / "predictions.csv", rows)
            metrics = _metrics(
                rows, "Pooled original-TRAIN OOF; used for development epoch selection"
            )
            write_json(directory / "metrics.json", metrics)
            oof_runs.append(
                {
                    "seed": seed,
                    "metrics": metrics,
                    "predictions_sha256": _sha(directory / "predictions.csv"),
                }
            )
        ensemble = _export_ensemble(
            out / "cv" / name / "oof" / "ensemble",
            ensemble_rows([oof_seed_rows[seed] for seed in SEEDS]),
            "Pooled original-TRAIN OOF ensemble; development selection",
        )
        selection_score = float(
            np.mean(
                [
                    result["ensemble"]["metrics"]["conditional"]["macro_f1"]
                    for result in fold_results
                ]
            )
        )
        candidates[name] = {
            "cv": {
                "folds": fold_results,
                "oof": {"runs": oof_runs, "ensemble": ensemble},
                "selection_macro_f1": selection_score,
            }
        }
        final_epochs[name] = {str(seed): int(np.median(selected_epochs[seed])) for seed in SEEDS}
        write_json(out / "cv" / name / "summary.json", candidates[name]["cv"])
    selected = max(CANDIDATES, key=lambda name: candidates[name]["cv"]["selection_macro_f1"])
    selection = {
        "schema_version": 1,
        "selected_candidate": selected,
        "candidate_scores": {
            name: candidates[name]["cv"]["selection_macro_f1"] for name in CANDIDATES
        },
        "criterion": "Mean of three CV-fold ensemble macro-F1; earliest candidate order wins exact ties",
        "final_epochs": final_epochs,
        "created_before_original_validation_evaluation": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_used_for_epoch_or_candidate_selection": False,
        "cv_scores_are_unbiased_estimates": False,
        "test_accessed": False,
        **provenance,
    }
    write_json(out / "selection.json", selection)
    selection_sha = _sha(out / "selection.json")
    print(
        f"CV preselection fixed before original-validation inference: {selected}; final epochs={final_epochs}",
        flush=True,
    )
    fit_indices = _indices(train_targets)
    for name in CANDIDATES:
        runs, seed_predictions = [], []
        for seed in SEEDS:
            model, history, best_epoch, weights = _fit(
                name, seed, x, y, fit_indices, config=protocol, epochs=final_epochs[name][str(seed)]
            )
            directory = out / "validation" / name / f"seed-{seed}"
            result, rows = _export_run(
                directory,
                model,
                history,
                best_epoch,
                name,
                seed,
                fit_indices,
                validation_targets,
                data,
                x,
                weights,
                {**provenance, "selection_sha256": selection_sha},
                scope="Original-validation single development check; epochs selected by TRAIN CV",
                fold=None,
            )
            runs.append(result)
            seed_predictions.append(rows)
        ensemble = _export_ensemble(
            out / "validation" / name / "ensemble",
            ensemble_rows(seed_predictions),
            "Original-validation ensemble development check; no validation epoch/candidate selection",
        )
        scores = [run["macro_f1"] for run in runs]
        candidates[name]["validation"] = {
            "runs": runs,
            "ensemble": ensemble,
            "macro_f1_mean": float(np.mean(scores)),
            "macro_f1_std": float(np.std(scores, ddof=1)),
            "std_definition": "Sample standard deviation across three fixed seeds; not a confidence interval",
        }
        write_json(out / "validation" / name / "summary.json", candidates[name]["validation"])
    if (
        _sha(protocol_path) != provenance["protocol_sha256"]
        or _sha(Path(__file__)) != provenance["implementation_sha256"]
        or _sha(out / "selection.json") != selection_sha
    ):
        raise RuntimeError(
            "Protocol, implementation or candidate preselection changed during training"
        )
    summary = {
        "schema_version": 1,
        "scope": "TRAIN-only development CV and original-validation development check; no test evaluation",
        "configuration": config,
        "candidates": candidates,
        "selected_candidate": selected,
        "selection_sha256": selection_sha,
        "final_epochs": final_epochs,
        "folds": folds["folds"],
        "all_seeds_reported": True,
        "cv_scores_are_unbiased_estimates": False,
        "validation_used_for_epoch_or_candidate_selection": False,
        "test_evaluated": False,
        "new_version_independently_tested": False,
    }
    write_json(out / "summary.json", summary)
    return summary
