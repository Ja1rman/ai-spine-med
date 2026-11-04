"""Train fixed linear/MLP bilateral heads on frozen RSNA TRAIN image features.

Both tasks have four independent level/side heads, each with three RSNA classes.
Validation selects an epoch within every predeclared run; neither architecture
nor a fortunate seed is claimed as an independently tested winner.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from . import side_stenosis_data as data
from .assets import sha256_file, write_json

OUT = Path("reports/runs/side-stenosis-v15")
SEEDS = (42, 2024, 3407)
ARCHITECTURES = ("linear", "mlp")


class BilateralHead(torch.nn.Module):
    def __init__(self, features, architecture):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError("Unknown fixed architecture")
        self.net = (
            torch.nn.Linear(features, 12)
            if architecture == "linear"
            else torch.nn.Sequential(
                torch.nn.Linear(features, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(64, 12),
            )
        )

    def forward(self, features, tasks):
        output = self.net(features).reshape(-1, 4, 3)
        return output[torch.arange(len(tasks)), tasks]


def normalized_inputs(values, rows, mean=None, scale=None):
    selected = [r for r in rows if r["status"] == "ok"]
    if not selected or values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Require nonempty finite frozen vectors")
    indices = [r["feature_index"] for r in selected]
    if any(type(i) is not int or not 0 <= i < len(values) for i in indices):
        raise ValueError("Invalid feature index")
    x = values[indices].astype(np.float32)
    if mean is None:
        if any(r["split"] != "train" for r in selected):
            raise ValueError("Normalizers may be fitted on TRAIN only")
        mean = x.mean(0)
        scale = np.maximum(x.std(0), 1e-5)
    x = np.clip((x - mean) / scale, -10, 10).astype(np.float32)
    return selected, x, mean, scale


def metrics(rows):
    def group(subset):
        known = [r for r in subset if r["label"] is not None]
        pairs = [r for r in known if r["prediction"] is not None]
        y = [r["label"] for r in pairs]
        p = [r["prediction"] for r in pairs]
        severe = [r for r in known if r["label"] == 2]
        return {
            "requested": len(subset),
            "assessed": sum(r["prediction"] is not None for r in subset),
            "known_references": len(known),
            "known_and_assessed": len(pairs),
            "class_counts_known": [sum(r["label"] == c for r in known) for c in range(3)],
            "macro_f1": float(f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0))
            if pairs
            else None,
            "confusion": confusion_matrix(y, p, labels=[0, 1, 2]).tolist()
            if pairs
            else [[0, 0, 0] for _ in range(3)],
            "severe_total": len(severe),
            "severe_correct": sum(r["prediction"] == 2 for r in severe),
            "severe_to_normal": sum(r["prediction"] == 0 for r in severe),
            "severe_to_moderate": sum(r["prediction"] == 1 for r in severe),
            "severe_unassessed": sum(r["prediction"] is None for r in severe),
            "failure_reasons": dict(
                Counter(r["reason"] for r in subset if r["prediction"] is None)
            ),
        }

    return {
        "overall": group(rows),
        "by_level": {
            level: group([r for r in rows if r["level"] == level]) for level in data.LEVELS
        },
        "by_level_side": {
            f"{level}:{side}": group([r for r in rows if r["level"] == level and r["side"] == side])
            for level in data.LEVELS
            for side in data.SIDES
        },
    }


def predictions(rows, probability_by_key):
    output = []
    for row in rows:
        key = row["study_id"], row["level"], row["side"]
        probabilities = probability_by_key.get(key)
        if (probabilities is not None) != (row["status"] == "ok"):
            raise ValueError("Predictions must preserve original image failures")
        output.append(
            row
            | {
                "prediction": int(np.argmax(probabilities)) if probabilities is not None else None,
                "probabilities": probabilities.tolist() if probabilities is not None else None,
            }
        )
    return output


def _save_predictions(path, rows):
    fields = (
        "study_id",
        "level",
        "side",
        "condition",
        "split",
        "label",
        "reference_status",
        "status",
        "reason",
        "prediction",
        "p_normal_mild",
        "p_moderate",
        "p_severe",
    )
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            result = {k: row[k] for k in fields if k in row}
            for name, p in zip(fields[-3:], row["probabilities"] or [None] * 3):
                result[name] = p
            writer.writerow(result)


def _fit(train_values, train_rows, val_values, val_rows, architecture, seed, config, directory):
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    train, x, mean, scale = normalized_inputs(train_values, train_rows)
    val, vx, _, _ = normalized_inputs(val_values, val_rows, mean, scale)
    known_indices = np.asarray([i for i, r in enumerate(train) if r["label"] is not None])
    tx = torch.from_numpy(x[known_indices])
    ty = torch.tensor([train[i]["label"] for i in known_indices], dtype=torch.long)
    tt = torch.tensor([train[i]["task_index"] for i in known_indices], dtype=torch.long)
    vx = torch.from_numpy(vx)
    vt = torch.tensor([r["task_index"] for r in val], dtype=torch.long)
    weights = []
    for task in range(4):
        count = np.bincount(ty[tt == task].numpy(), minlength=3)
        if np.any(count == 0):
            raise ValueError("Every training head needs all three classes")
        weight = 1 / np.sqrt(count)
        weights.append(weight / weight.mean())
    weights = torch.tensor(np.asarray(weights), dtype=torch.float32)
    model = BilateralHead(x.shape[1], architecture)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    best_score, best_epoch, best_state, stalled = -1.0, None, None, 0
    history, started = [], time.perf_counter()
    rng = torch.Generator().manual_seed(seed)
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        order = torch.randperm(len(tx), generator=rng)
        for start in range(0, len(order), config["batch_size"]):
            indices = order[start : start + config["batch_size"]]
            logits = model(tx[indices], tt[indices])
            loss = torch.nn.functional.cross_entropy(logits, ty[indices], reduction="none")
            loss = (loss * weights[tt[indices], ty[indices]]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            probabilities = model(vx, vt).softmax(-1).numpy()
        scores = {(r["study_id"], r["level"], r["side"]): p for r, p in zip(val, probabilities)}
        score = metrics(predictions(val_rows, scores))["overall"]["macro_f1"]
        history.append({"epoch": epoch, "validation_macro_f1": score})
        if score > best_score + 1e-12:
            best_score, best_epoch, best_state, stalled = (
                score,
                epoch,
                copy.deepcopy(model.state_dict()),
                0,
            )
        else:
            stalled += 1
        if stalled >= config["patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        probabilities = model(vx, vt).softmax(-1).numpy()
    scores = {(r["study_id"], r["level"], r["side"]): p for r, p in zip(val, probabilities)}
    rows = predictions(val_rows, scores)
    checkpoint = {
        "schema_version": 1,
        "architecture": architecture,
        "feature_count": x.shape[1],
        "seed": seed,
        "best_epoch": best_epoch,
        "state_dict": best_state,
        "mean": torch.from_numpy(mean),
        "scale": torch.from_numpy(scale),
        "classes": ["normal_mild", "moderate", "severe"],
        "tasks": [f"{level}:{side}" for level in data.LEVELS for side in data.SIDES],
        "encoder_updated": False,
    }
    torch.save(checkpoint, directory / "head.pt")
    # Reopen the exact serialized model rather than trusting the live instance.
    reloaded = torch.load(directory / "head.pt", map_location="cpu", weights_only=True)
    restored = BilateralHead(reloaded["feature_count"], reloaded["architecture"])
    restored.load_state_dict(reloaded["state_dict"], strict=True)
    restored.eval()
    with torch.inference_mode():
        actual = restored(vx, vt).softmax(-1).numpy()
    if not np.array_equal(actual, probabilities):
        raise RuntimeError("Reloaded checkpoint predictions changed")
    _save_predictions(directory / "validation.csv", rows)
    result = {
        "seed": seed,
        "best_epoch": best_epoch,
        "trained_epochs": len(history),
        "training_seconds": time.perf_counter() - started,
        "metrics": metrics(rows),
        "history": history,
        "train_class_counts_by_task": [
            torch.bincount(ty[tt == task], minlength=3).tolist() for task in range(4)
        ],
        "checkpoint_sha256": sha256_file(directory / "head.pt"),
        "predictions_sha256": sha256_file(directory / "validation.csv"),
        "checkpoint_reload_max_abs_error": 0,
    }
    write_json(directory / "summary.json", result)
    print(
        json.dumps(
            {"architecture": architecture, "seed": seed, "f1": best_score, "epoch": best_epoch}
        ),
        flush=True,
    )
    return rows, result


def fit(protocol=data.PROTOCOL, cache=data.OUT, out=OUT):
    config = data.verify_protocol(protocol)
    cache, out = Path(cache), Path(out)
    prepared = json.loads((cache / "summary.json").read_text())
    if (
        prepared["protocol_sha256"] != sha256_file(Path(protocol))
        or prepared["test_evaluated"] is not False
    ):
        raise ValueError("Fitting requires the matching development-only cache")
    if any(sha256_file(cache / file) != digest for file, digest in prepared["sha256"].items()):
        raise ValueError("Prepared cache changed")
    if (out / "summary.json").exists():
        raise ValueError("Completed experiment is immutable; use a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": config,
        "protocol_sha256": sha256_file(Path(protocol)),
        "cache_summary_sha256": sha256_file(cache / "summary.json"),
        "implementation_sha256": sha256_file(Path(__file__)),
        "test_evaluated": False,
    }
    write_json(out / "manifest.json", manifest)
    torch.set_num_threads(4)
    result = {
        "protocol_sha256": manifest["protocol_sha256"],
        "test_evaluated": False,
        "conditions": {},
    }
    for condition in data.CONDITIONS:
        partitions = prepared["conditions"][condition]
        rows = {s: json.loads((cache / partitions[s]["targets"]).read_text()) for s in data.SPLITS}
        values = {
            s: np.load(cache / partitions[s]["features"], allow_pickle=False) for s in data.SPLITS
        }
        if {r["study_id"] for r in rows["train"]} & {r["study_id"] for r in rows["validation"]}:
            raise ValueError("Study leakage between train and validation")
        baseline_scores = {}
        for task in range(4):
            count = Counter(
                r["label"]
                for r in rows["train"]
                if r["task_index"] == task and r["label"] is not None
            )
            prediction = max(range(3), key=lambda grade: (count[grade], -grade))
            for row in rows["validation"]:
                if row["task_index"] == task and row["status"] == "ok":
                    baseline_scores[row["study_id"], row["level"], row["side"]] = np.eye(3)[
                        prediction
                    ]
        baseline = predictions(rows["validation"], baseline_scores)
        _save_predictions(out / f"{condition}-majority.csv", baseline)
        condition_results = {"majority_train": metrics(baseline), "architectures": {}}
        for architecture in ARCHITECTURES:
            individual, summaries = [], []
            for seed in SEEDS:
                pred, summary = _fit(
                    values["train"],
                    rows["train"],
                    values["validation"],
                    rows["validation"],
                    architecture,
                    seed,
                    config,
                    out / condition / architecture / f"seed-{seed}",
                )
                individual.append(pred)
                summaries.append(summary)
            ensemble_scores = {}
            for i, row in enumerate(rows["validation"]):
                if row["status"] == "ok":
                    ensemble_scores[row["study_id"], row["level"], row["side"]] = np.mean(
                        [run[i]["probabilities"] for run in individual], axis=0
                    )
            ensemble = predictions(rows["validation"], ensemble_scores)
            ensemble_path = out / condition / architecture / "ensemble.csv"
            _save_predictions(ensemble_path, ensemble)
            condition_results["architectures"][architecture] = {
                "seeds": summaries,
                "ensemble": metrics(ensemble),
                "mean_seed_macro_f1": float(
                    np.mean([r["metrics"]["overall"]["macro_f1"] for r in summaries])
                ),
                "std_seed_macro_f1": float(
                    np.std([r["metrics"]["overall"]["macro_f1"] for r in summaries], ddof=1)
                ),
                "ensemble_sha256": sha256_file(ensemble_path),
            }
        result["conditions"][condition] = condition_results
    data.verify_protocol(protocol)
    write_json(out / "summary.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "report", "all"))
    parser.add_argument("--protocol", type=Path, default=data.PROTOCOL)
    parser.add_argument("--cache", type=Path, default=data.OUT)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    if args.stage in {"prepare", "all"}:
        data.prepare(args.protocol, args.cache)
    if args.stage in {"train", "all"}:
        fit(args.protocol, args.cache, args.out)
    if args.stage in {"report", "all"}:
        from .side_stenosis_report import report

        report(args.out)


if __name__ == "__main__":
    main()
