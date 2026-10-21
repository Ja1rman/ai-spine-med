"""RSNA-only severity heads on frozen, automatically localized axial ROI features.

This component never selects slices/crops and never opens a DICOM or an anatomy
annotation. TEST features are accepted only after all three training checkpoints
are sealed. AISSLab grades never contribute to optimization or epoch selection.
Individual prediction CSVs/checkpoints are private research artifacts.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .assets import sha256_file, write_json
from .training import evaluate_predictions, inverse_sqrt_class_weights

FAMILY = "localized_stenosis_v1"
SEEDS = (42, 2024, 3407)
LEVELS = ("L4-L5", "L5-S1")
CLASSES = ("normal_mild", "moderate", "severe")
FIELDS = (
    "dataset", "study_id", "split", "level", "label", "status", "reason", "feature_index"
)
PREDICTION_FIELDS = (
    "dataset", "study_id", "split", "level", "label", "reference_status", "status", "reason",
    "prediction", "p_normal_mild", "p_moderate", "p_severe",
)


def _hash_json(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _device(device):
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device not in {"cpu", "mps"}:
        raise ValueError("Localized severity supports cpu, mps or auto")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    return device


def _protocol(protocol):
    if not isinstance(protocol, dict):
        raise ValueError("Require an explicit classifier protocol")
    values = protocol.get("classifier", protocol)
    defaults = {
        "seeds": list(SEEDS), "hidden_dim": 64, "dropout": 0.2,
        "batch_size": 64, "max_epochs": 30, "patience": 5,
        "learning_rate": 1e-3, "weight_decay": 1e-4,
    }
    result = {
        key: values.get(f"severity_{key}", values.get(key, default))
        for key, default in defaults.items()
    }
    if (
        result["seeds"] != list(SEEDS)
        or result["hidden_dim"] != 64
        or result["dropout"] != 0.2
        or result["batch_size"] != 64
        or result["learning_rate"] != 1e-3
        or result["weight_decay"] != 1e-4
        or type(result["max_epochs"]) is not int
        or not 1 <= result["max_epochs"] <= 30
        or type(result["patience"]) is not int
        or not 1 <= result["patience"] <= 5
    ):
        raise ValueError("Classifier protocol differs from the fixed small-MLP design")
    return result


def read_targets(path, *, fitting=False):
    """Validate identities/splits before opening any feature arrays."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not set(FIELDS).issubset(reader.fieldnames or ()):
            raise ValueError("Localized feature targets lack required columns")
        raw_rows = list(reader)
    if not raw_rows:
        raise ValueError("Require nonempty localized feature targets")
    rows, keys, groups, indices = [], set(), {}, set()
    for raw in raw_rows:
        row = {name: raw[name] for name in FIELDS}
        if (
            row["dataset"] not in {"rsna", "aisslab"}
            or not row["study_id"]
            or row["split"] not in {"train", "validation", "test"}
            or row["level"] not in LEVELS
            or row["status"] not in {"ok", "not_assessed", "failed"}
        ):
            raise ValueError("Invalid localized feature target identity/split/status")
        if fitting and row["split"] == "test":
            raise ValueError("Training never accepts TEST targets/features")
        key = row["dataset"], row["study_id"], row["level"]
        group = key[:2]
        if key in keys or (group in groups and groups[group] != row["split"]):
            raise ValueError("A patient/level duplicates or a patient crosses splits")
        keys.add(key)
        groups[group] = row["split"]
        try:
            row["label"] = int(row["label"]) if row["label"] != "" else None
            row["feature_index"] = (
                int(row["feature_index"]) if row["feature_index"] != "" else None
            )
        except (TypeError, ValueError):
            raise ValueError("Invalid label or localized feature index") from None
        if row["label"] is not None and row["label"] not in (0, 1, 2):
            raise ValueError("Invalid three-category reference label")
        index = row["feature_index"]
        if row["status"] == "ok":
            if index is None or index < 0 or index in indices or row["reason"]:
                raise ValueError("Assessed targets require unique feature indices and no reason")
            indices.add(index)
        elif index is not None or not row["reason"]:
            raise ValueError("Failed targets require a reason and no feature index")
        rows.append(row)
    if indices != set(range(len(indices))):
        raise ValueError("Feature indices must cover the exact contiguous matrix")
    return rows


def load_features(path, rows):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"features"}:
            raise ValueError("Require only the frozen ROI feature matrix in NPZ")
        features = archive["features"].copy()
    n = sum(row["status"] == "ok" for row in rows)
    if (
        features.shape != (n, 512)
        or features.dtype != np.float32
        or not np.isfinite(features).all()
    ):
        raise ValueError("Require finite float32 localized ROI features [N,512]")
    return features


class LocalizedSeverityHead(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        mean, std = torch.as_tensor(mean), torch.as_tensor(std)
        if (
            mean.shape != (512,) or std.shape != (512,)
            or not torch.isfinite(mean).all() or not torch.isfinite(std).all()
            or not (std > 0).all()
        ):
            raise ValueError("Require finite TRAIN-only 512-dimensional normalizers")
        self.register_buffer("mean", mean.float().clone())
        self.register_buffer("std", std.float().clone())
        self.network = nn.Sequential(nn.Linear(514, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 3))

    def forward(self, features, levels):
        if features.ndim != 2 or features.shape[1] != 512 or levels.shape != (len(features),):
            raise ValueError("Require ROI [N,512] and one target level index per ROI")
        if levels.dtype != torch.long or ((levels < 0) | (levels > 1)).any():
            raise ValueError("Target level indices must be 0=L4-L5, 1=L5-S1")
        one_hot = F.one_hot(levels, 2).to(features.dtype)
        return self.network(torch.cat(((features - self.mean) / self.std, one_hot), dim=1))


def predict_localized_features(model, features, levels, *, batch_size=256):
    features = np.asarray(features, dtype=np.float32)
    levels = np.asarray(levels, dtype=np.int64)
    if features.shape != (len(levels), 512) or not np.isfinite(features).all():
        raise ValueError("Invalid prediction feature/level arrays")
    device = next(model.parameters()).device
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            x = torch.from_numpy(features[start:start + batch_size]).to(device)
            level = torch.from_numpy(levels[start:start + batch_size]).to(device)
            result.append(model(x, level).softmax(-1).cpu().numpy())
    probabilities = np.concatenate(result) if result else np.empty((0, 3), np.float32)
    if (
        probabilities.shape != (len(levels), 3) or not np.isfinite(probabilities).all()
        or not np.allclose(probabilities.sum(1), 1, atol=1e-6)
    ):
        raise RuntimeError("Invalid localized severity probabilities")
    return probabilities


def prediction_rows(rows, model, features):
    """Missing grades stay missing while their model probabilities are preserved."""
    successful = [row for row in rows if row["status"] == "ok"]
    indices = [row["feature_index"] for row in successful]
    probabilities = predict_localized_features(
        model, features[indices], [LEVELS.index(row["level"]) for row in successful]
    )
    by_key = {
        (row["dataset"], row["study_id"], row["level"]): probability
        for row, probability in zip(successful, probabilities, strict=True)
    }
    result = []
    for target in rows:
        row = {key: target[key] for key in FIELDS if key != "feature_index"}
        row.update(
            reference_status="known" if row["label"] is not None else "missing",
            prediction=None, p_normal_mild=None, p_moderate=None, p_severe=None,
        )
        probability = by_key.get((row["dataset"], row["study_id"], row["level"]))
        if probability is not None:
            row["prediction"] = int(probability.argmax())
            for name, value in zip(CLASSES, probability, strict=True):
                row[f"p_{name}"] = float(value)
        result.append(row)
    return result


def _metrics(rows, scope):
    metrics = evaluate_predictions(rows)
    metrics["scope"] = scope
    metrics["n_predictions_including_missing_reference"] = sum(r["status"] == "ok" for r in rows)
    metrics["by_dataset"] = {
        dataset: evaluate_predictions([r for r in rows if r["dataset"] == dataset])
        for dataset in sorted({r["dataset"] for r in rows})
    }
    for dataset, item in metrics["by_dataset"].items():
        item["scope"] = scope
        if dataset == "aisslab":
            item["clinical_scale_equivalence_verified"] = False
    return metrics


def _write_predictions(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _ensemble(all_rows):
    if len(all_rows) != 3:
        raise ValueError("Require exactly the three fixed seeds for the ensemble")
    result = []
    for triple in zip(*all_rows, strict=True):
        reference = triple[0]
        identity = {key: reference[key] for key in FIELDS if key != "feature_index"}
        if any(
            {key: row[key] for key in identity} != identity for row in triple[1:]
        ):
            raise ValueError("Seed prediction rows differ in target identity/reference/status")
        row = dict(reference)
        if row["status"] == "ok":
            probability = np.mean([[r[f"p_{c}"] for c in CLASSES] for r in triple], axis=0)
            row["prediction"] = int(probability.argmax())
            row.update({f"p_{name}": float(value) for name, value in zip(CLASSES, probability)})
        result.append(row)
    return result


def _provenance(features_path, targets_path, protocol, source_protocol):
    return {
        "features_path": str(Path(features_path).resolve()),
        "features_sha256": sha256_file(Path(features_path)),
        "targets_path": str(Path(targets_path).resolve()),
        "targets_sha256": sha256_file(Path(targets_path)),
        "implementation_sha256": sha256_file(Path(__file__)),
        "protocol": protocol, "protocol_sha256": _hash_json(protocol),
        "source_protocol": source_protocol,
        "source_protocol_sha256": _hash_json(source_protocol),
    }


def _assert_provenance(provenance):
    if (
        sha256_file(Path(provenance["features_path"])) != provenance["features_sha256"]
        or sha256_file(Path(provenance["targets_path"])) != provenance["targets_sha256"]
        or sha256_file(Path(__file__)) != provenance["implementation_sha256"]
        or _hash_json(provenance["protocol"]) != provenance["protocol_sha256"]
        or _hash_json(provenance["source_protocol"]) != provenance["source_protocol_sha256"]
    ):
        raise ValueError("Classifier implementation, protocol or fitting source changed")


def train_localized_stenosis(features_npz, targets_csv, out, protocol, *, device="cpu"):
    """Fit three frozen-feature MLPs using RSNA TRAIN and select on RSNA validation."""
    source_protocol = copy.deepcopy(protocol)
    protocol = _protocol(protocol)
    targets_sha256 = sha256_file(Path(targets_csv))
    rows = read_targets(targets_csv, fitting=True)
    provenance = _provenance(features_npz, targets_csv, protocol, source_protocol)
    if provenance["targets_sha256"] != targets_sha256:
        raise ValueError("Fitting target identities changed while being read")
    def selected(split):
        return [
            r for r in rows if r["dataset"] == "rsna" and r["split"] == split
            and r["status"] == "ok" and r["label"] is not None
        ]
    train, validation = selected("train"), selected("validation")
    if not train or not validation:
        raise ValueError("Require labeled successfully localized RSNA TRAIN and validation")
    labels = np.asarray([r["label"] for r in train], dtype=np.int64)
    weights = inverse_sqrt_class_weights(labels)
    features = load_features(features_npz, rows)
    x = features[[r["feature_index"] for r in train]]
    mean = x.mean(0, dtype=np.float64).astype(np.float32)
    std = x.std(0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    out = Path(out)
    if out.exists():
        raise ValueError("Classifier output already exists; never overwrite an experiment")
    out.mkdir(parents=True)
    _assert_provenance(provenance)
    manifest = {
        "schema_version": 1, "family": FAMILY, "provenance": provenance,
        "classes": list(CLASSES), "levels": list(LEVELS),
        "architecture": "frozen ROI embedding512 + level one-hot2 -> Linear64/ReLU/Dropout0.2/Linear3",
        "optimizer": "AdamW; weighted CE sum / target count",
        "class_counts": np.bincount(labels, minlength=3).tolist(),
        "class_weights": weights.tolist(),
        "train_targets": len(train), "validation_targets": len(validation),
        "fitting_patient_groups": sorted({(r["dataset"], r["study_id"]) for r in rows}),
        "gradient_patient_groups": sorted({(r["dataset"], r["study_id"]) for r in train}),
        "selection_patient_groups": sorted({(r["dataset"], r["study_id"]) for r in validation}),
        "AISSLab_grades_used_for_training_or_selection": False,
        "selection": "highest RSNA validation macro-F1 over all 3 classes; earliest epoch wins ties",
        "device": _device(device),
    }
    write_json(out / "manifest.json", manifest)
    device = manifest["device"]
    tx = torch.from_numpy(x).to(device)
    tl = torch.tensor([LEVELS.index(r["level"]) for r in train], dtype=torch.long, device=device)
    ty = torch.from_numpy(labels).to(device)
    tw = torch.from_numpy(weights).to(device)
    val_rows = [r for r in rows if r["dataset"] == "rsna" and r["split"] == "validation"]
    runs, all_validation = [], []
    for seed in SEEDS:
        torch.manual_seed(seed)
        generator = np.random.default_rng(seed)
        model = LocalizedSeverityHead(mean, std).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=protocol["learning_rate"], weight_decay=protocol["weight_decay"]
        )
        best_f1, best_epoch, best_state, stale = -math.inf, None, None, 0
        history = []
        for epoch in range(1, protocol["max_epochs"] + 1):
            model.train()
            total = 0.0
            order = generator.permutation(len(train))
            for start in range(0, len(order), protocol["batch_size"]):
                batch = torch.tensor(order[start:start + protocol["batch_size"]], device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(tx[batch], tl[batch]), ty[batch], weight=tw, reduction="sum")
                loss = loss / len(batch)
                loss.backward()
                optimizer.step()
                total += float(loss.detach().cpu()) * len(batch)
            predicted = prediction_rows(val_rows, model, features)
            score = _metrics(predicted, "RSNA validation only; checkpoint selection")
            f1 = score["conditional"]["macro_f1"]
            history.append({"epoch": epoch, "weighted_training_loss": total / len(train),
                            "validation_macro_f1": f1})
            if f1 > best_f1:
                best_f1, best_epoch = f1, epoch
                best_state = copy.deepcopy({key: value.cpu() for key, value in model.state_dict().items()})
                stale = 0
            else:
                stale += 1
            print(f"localized severity seed={seed} epoch={epoch} validation_macro_f1={f1:.6f}", flush=True)
            if stale >= protocol["patience"]:
                break
        model.load_state_dict(best_state, strict=True)
        _assert_provenance(provenance)
        run = out / "runs" / f"seed-{seed}"
        run.mkdir(parents=True)
        checkpoint = run / "model.pt"
        torch.save({
            "schema_version": 1, "family": FAMILY, "state_dict": best_state,
            "seed": seed, "selected_epoch": best_epoch, "classes": list(CLASSES),
            "levels": list(LEVELS), "provenance_sha256": _hash_json(provenance),
        }, checkpoint)
        predicted = prediction_rows(val_rows, model, features)
        _write_predictions(run / "validation.csv", predicted)
        result = {
            "seed": seed, "selected_epoch": best_epoch, "epochs_trained": len(history),
            "checkpoint": str(checkpoint.relative_to(out)), "checkpoint_sha256": sha256_file(checkpoint),
            "validation_predictions_sha256": sha256_file(run / "validation.csv"),
            "metrics": _metrics(predicted, "RSNA validation only; used for checkpoint selection"),
        }
        write_json(run / "history.json", history)
        write_json(run / "result.json", result)
        runs.append(result)
        all_validation.append(predicted)
    ensemble = _ensemble(all_validation)
    _write_predictions(out / "validation_ensemble.csv", ensemble)
    summary = {
        "schema_version": 1, "family": FAMILY, "status": "complete", "runs": runs,
        "validation_ensemble": _metrics(ensemble, "RSNA validation only; not an independent test"),
        "validation_ensemble_sha256": sha256_file(out / "validation_ensemble.csv"),
        "manifest_sha256": sha256_file(out / "manifest.json"),
    }
    write_json(out / "summary.json", summary)
    _assert_provenance(provenance)
    write_json(out / "selection.json", {
        "schema_version": 1, "family": FAMILY, "status": "sealed_before_test",
        "frozen": True, "seeds": list(SEEDS),
        "protocol_sha256": provenance["source_protocol_sha256"],
        "training_index_sha256": provenance["targets_sha256"],
        "features_sha256": provenance["features_sha256"],
        "runs": [
            {key: run[key] for key in ("seed", "checkpoint", "checkpoint_sha256")}
            for run in runs
        ],
        "manifest_sha256": sha256_file(out / "manifest.json"),
        "summary_sha256": sha256_file(out / "summary.json"),
        "checkpoints": {r["checkpoint"]: r["checkpoint_sha256"] for r in runs},
    })
    return summary


def load_sealed_models(out, *, device="cpu"):
    """Check the complete three-seed seal before any TEST target/feature reads."""
    out = Path(out)
    seal = json.loads((out / "selection.json").read_text())
    if (seal.get("family") != FAMILY or seal.get("status") != "sealed_before_test"
            or seal.get("frozen") is not True or seal.get("seeds") != list(SEEDS)
            or seal.get("manifest_sha256") != sha256_file(out / "manifest.json")
            or seal.get("summary_sha256") != sha256_file(out / "summary.json")):
        raise ValueError("The complete classifier checkpoint selection is not sealed")
    manifest = json.loads((out / "manifest.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    if summary.get("status") != "complete" or [r["seed"] for r in summary["runs"]] != list(SEEDS):
        raise ValueError("Require all three completed fixed-seed classifier runs")
    _assert_provenance(manifest["provenance"])
    if (seal.get("protocol_sha256") != manifest["provenance"]["source_protocol_sha256"]
            or seal.get("training_index_sha256") != manifest["provenance"]["targets_sha256"]
            or seal.get("features_sha256") != manifest["provenance"]["features_sha256"]):
        raise ValueError("Selection provenance differs from the fitting sources")
    expected = {r["checkpoint"]: r["checkpoint_sha256"] for r in summary["runs"]}
    if seal["checkpoints"] != expected:
        raise ValueError("Selected checkpoint hashes differ from the training summary")
    models = []
    for result in summary["runs"]:
        path = (out / result["checkpoint"]).resolve()
        if not path.is_relative_to(out.resolve()) or sha256_file(path) != result["checkpoint_sha256"]:
            raise ValueError("A selected classifier checkpoint changed or escapes its run")
        data = torch.load(path, map_location="cpu", weights_only=True)
        if (data.get("family") != FAMILY or data.get("seed") != result["seed"]
                or data.get("classes") != list(CLASSES) or data.get("levels") != list(LEVELS)
                or data.get("provenance_sha256") != _hash_json(manifest["provenance"])):
            raise ValueError("Checkpoint identity/classes/provenance differs from the seal")
        state = data["state_dict"]
        model = LocalizedSeverityHead(state["mean"], state["std"])
        model.load_state_dict(state, strict=True)
        models.append(model.eval().requires_grad_(False).to(_device(device)))
    return models, manifest, seal


def evaluate_localized_stenosis(features_npz, targets_csv, out, *, device="cpu", evaluation_out=None):
    """Evaluate sealed heads; TEST anatomy/grades cannot alter their selection."""
    models, manifest, seal = load_sealed_models(out, device=device)
    selection_sha256 = sha256_file(Path(out) / "selection.json")
    targets_sha256 = sha256_file(Path(targets_csv))
    rows = read_targets(targets_csv)
    fitting_groups = {tuple(group) for group in manifest["fitting_patient_groups"]}
    if any(r["split"] != "test" for r in rows):
        raise ValueError("Sealed evaluation accepts only held-out TEST rows")
    if any((r["dataset"], r["study_id"]) in fitting_groups for r in rows):
        raise ValueError("An evaluation patient occurred in fitting/selection sources")
    features_sha256 = sha256_file(Path(features_npz))
    features = load_features(features_npz, rows)
    destination = Path(evaluation_out) if evaluation_out else Path(out) / "evaluation"
    if destination.exists():
        raise ValueError("Evaluation output already exists; never overwrite fixed results")
    destination.mkdir(parents=True)
    outputs, all_rows = {}, []
    for seed, model in zip(SEEDS, models, strict=True):
        predicted = prediction_rows(rows, model, features)
        name = f"seed_{seed}.csv"
        _write_predictions(destination / name, predicted)
        outputs[name] = {"sha256": sha256_file(destination / name),
                         "metrics": _metrics(predicted, "Exploratory held-out follow-up; previous TEST already inspected")}
        all_rows.append(predicted)
    ensemble = _ensemble(all_rows)
    _write_predictions(destination / "ensemble.csv", ensemble)
    outputs["ensemble.csv"] = {
        "sha256": sha256_file(destination / "ensemble.csv"),
        "metrics": _metrics(ensemble, "Exploratory held-out follow-up; previous TEST already inspected"),
    }
    summary = {
        "schema_version": 1, "family": FAMILY, "status": "complete", "outputs": outputs,
        "selection_sha256": selection_sha256,
        "features_sha256": features_sha256,
        "targets_sha256": targets_sha256,
        "patient_counts": dict(Counter(r["dataset"] for r in {
            (row["dataset"], row["study_id"]): row for row in rows
        }.values())),
        "n_targets": len(rows), "clinical_scale_equivalence_verified": False,
        "seal_status": seal["status"],
    }
    if (sha256_file(Path(out) / "selection.json") != selection_sha256
            or sha256_file(Path(features_npz)) != features_sha256
            or sha256_file(Path(targets_csv)) != targets_sha256):
        raise ValueError("Evaluation inputs or classifier seal changed during evaluation")
    _assert_provenance(manifest["provenance"])
    write_json(destination / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "evaluate"))
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("configs/axial-localizer-v6.json"))
    parser.add_argument("--device", choices=("cpu", "mps", "auto"), default="cpu")
    args = parser.parse_args(argv)
    if args.phase == "train":
        summary = train_localized_stenosis(
            args.features, args.targets, args.out, json.loads(args.protocol.read_text()),
            device=args.device,
        )
    else:
        summary = evaluate_localized_stenosis(args.features, args.targets, args.out, device=args.device)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
