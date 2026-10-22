"""Paired RSNA-only adaptation of the v6 MLP to each frozen localizer's ROI.

The same v6 seed weights and normalizers initialize both arms. Epoch zero is
eligible; selection uses overall RSNA validation macro-F1, never L5 alone.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import localization_finetuning_data as data
from . import localization_severity_validation as v8
from . import localized_stenosis as severity
from .assets import sha256_file, write_json
from .training import evaluate_predictions, inverse_sqrt_class_weights

RUN = Path("reports/runs/localization-finetuning-v9")
FAMILY = "paired_localization_finetuning_v9"


def common_training_keys(tables):
    """Keep identical TRAIN identities/grades; do not silently impute failures."""
    indexed = {}
    for arm in v8.ARMS:
        rows = tables[arm]
        if any(r["dataset"] != "rsna" or r["split"] != "train" for r in rows):
            raise ValueError("Adaptation gradients require RSNA TRAIN only")
        indexed[arm] = {(r["dataset"], r["study_id"], r["level"]): r for r in rows}
        if len(indexed[arm]) != len(rows):
            raise ValueError("Duplicate TRAIN target")
    a, b = [indexed[arm] for arm in v8.ARMS]
    if a.keys() != b.keys() or any(a[k]["label"] != b[k]["label"] for k in a):
        raise ValueError("Training arms differ in requested identities or grades")
    keys = sorted(
        k
        for k, r in a.items()
        if r["label"] is not None and all(indexed[arm][k]["status"] == "ok" for arm in v8.ARMS)
    )
    if not keys:
        raise ValueError("No common assessed TRAIN targets")
    return keys, indexed


def _state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def epoch0_matches(predictions, source_csv):
    """Verify the warm start against the actual immutable v8 RSNA predictions."""
    with Path(source_csv).open(newline="") as stream:
        source = {
            (r["dataset"], r["study_id"], r["level"]): r
            for r in csv.DictReader(stream)
            if r["dataset"] == "rsna"
        }
    current = {(r["dataset"], r["study_id"], r["level"]): r for r in predictions}
    if current.keys() != source.keys():
        raise ValueError("Warm-start validation cohort differs from v8")
    for key, row in current.items():
        old = source[key]
        old_label = int(old["label"]) if old["label"] else None
        if row["label"] != old_label or row["status"] != old["status"]:
            raise ValueError("Warm-start grades/status differ from v8")
        if row["status"] == "ok":
            a = [row[f"p_{c}"] for c in severity.CLASSES]
            b = [float(old[f"p_{c}"]) for c in severity.CLASSES]
            if not np.allclose(a, b, rtol=0, atol=2e-7) or row["prediction"] != int(
                old["prediction"]
            ):
                raise ValueError("Epoch zero does not reproduce the frozen v8 classifier")


def fit_head(initial, x, levels, labels, val_rows, val_features, weights, config, seed, device):
    """Continue only MLP weights; keep buffers and epoch-zero predictions fixed."""
    if any(r["dataset"] != "rsna" or r["split"] != "validation" for r in val_rows):
        raise ValueError("Only RSNA validation can select a classifier epoch")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = copy.deepcopy(initial).to(device).requires_grad_(True).eval()
    normalizers = {k: getattr(model, k).detach().clone() for k in ("mean", "std")}
    baseline = severity.prediction_rows(val_rows, model, val_features)
    best_f1 = evaluate_predictions(baseline)["conditional"]["macro_f1"]
    best_state, selected, stale = _state(model), 0, 0
    history = [{"epoch": 0, "validation_macro_f1": best_f1, "weighted_train_loss": None}]
    optimizer = torch.optim.AdamW(
        model.network.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    tx = torch.as_tensor(x, dtype=torch.float32, device=device)
    tl = torch.as_tensor(levels, dtype=torch.long, device=device)
    ty = torch.as_tensor(labels, dtype=torch.long, device=device)
    tw = torch.as_tensor(weights, dtype=torch.float32, device=device)
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        order, total = rng.permutation(len(x)), 0.0
        for offset in range(0, len(order), config["batch_size"]):
            batch = torch.as_tensor(order[offset : offset + config["batch_size"]], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(
                model(tx[batch], tl[batch]), ty[batch], weight=tw, reduction="sum"
            ) / len(batch)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(batch)
        predictions = severity.prediction_rows(val_rows, model, val_features)
        score = evaluate_predictions(predictions)["conditional"]["macro_f1"]
        history.append(
            {"epoch": epoch, "validation_macro_f1": score, "weighted_train_loss": total / len(x)}
        )
        if score > best_f1:
            best_f1, best_state, selected, stale = score, _state(model), epoch, 0
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    model.load_state_dict(best_state)
    model.eval().requires_grad_(False)
    if any(not torch.equal(getattr(model, k), normalizers[k]) for k in normalizers):
        raise RuntimeError("Frozen v6 normalization changed during adaptation")
    return model, selected, history, baseline


def load_caches(cache, config):
    cache = Path(cache)
    receipt = json.loads((cache / "summary.json").read_text())
    if (
        receipt.get("phase") != "rsna_train"
        or receipt.get("grades_joined_after_both_arms_images") is not True
        or receipt["provenance"]["implementation_sha256"] != sha256_file(Path(data.__file__))
    ):
        raise ValueError("Need completed, unchanged image-only TRAIN extraction")
    train_rows, train_features, val_rows, val_features = {}, {}, {}, {}
    for arm in v8.ARMS:
        source = receipt["arms"][arm]
        paths = {key: cache / source[key] for key in ("features", "targets")}
        if any(
            not p.resolve().is_relative_to(cache.resolve())
            or sha256_file(p) != source[f"{key}_sha256"]
            for key, p in paths.items()
        ):
            raise ValueError("TRAIN features/targets changed or escaped their root")
        train_rows[arm] = severity.read_targets(paths["targets"], fitting=True)
        train_features[arm] = severity.load_features(paths["features"], train_rows[arm])
        directory = v8.RUN / "arms" / arm
        val_rows[arm] = severity.read_targets(directory / "targets.csv", fitting=True)
        if any(r["split"] != "validation" for r in val_rows[arm]):
            raise ValueError("Only validation targets belong to the fixed v8 cache")
        val_features[arm] = severity.load_features(directory / "features.npz", val_rows[arm])
        training_patients = {(r["dataset"], r["study_id"]) for r in train_rows[arm]}
        if training_patients & {(r["dataset"], r["study_id"]) for r in val_rows[arm]}:
            raise ValueError("A patient overlaps TRAIN and validation")
    if receipt["provenance"]["frozen_files"] != config["frozen_files"]:
        raise ValueError("TRAIN extraction and fitting protocol inputs differ")
    return train_rows, train_features, val_rows, val_features, receipt


def train_comparison(protocol_path=data.PROTOCOL, cache=data.OUT, out=RUN, device="cpu"):
    protocol_path, cache, out = Path(protocol_path), Path(cache), Path(out)
    config, frozen = data.verify_protocol(protocol_path)
    if (
        config["seeds"] != list(v8.SEEDS)
        or config["arms"] != list(v8.ARMS)
        or config["epoch0_eligible"] is not True
    ):
        raise ValueError("Use the fixed paired seeds, arms and epoch-zero candidate")
    device = severity._device(device)
    torch.set_num_threads(4)
    train_rows, train_features, val_rows, val_features, receipt = load_caches(cache, config)
    keys, indexed = common_training_keys(train_rows)
    labels = np.array([indexed["xml_only"][k]["label"] for k in keys], np.int64)
    levels = np.array([severity.LEVELS.index(k[2]) for k in keys], np.int64)
    class_weights = inverse_sqrt_class_weights(labels)
    initial, _, _ = severity.load_sealed_models(v8.GRADERS, device=device)
    cohort_sha = hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest()
    if out.exists():
        raise ValueError("Use a new output directory; fitted comparisons are immutable")
    out.mkdir(parents=True)
    provenance = {
        "protocol_sha256": sha256_file(protocol_path),
        "implementation_sha256": sha256_file(Path(__file__)),
        "training_summary_sha256": sha256_file(cache / "summary.json"),
        "common_train_cohort_sha256": cohort_sha,
        "frozen_files": frozen,
    }
    write_json(
        out / "manifest.json",
        {
            "family": FAMILY,
            "phase": "development_validation",
            "provenance": provenance,
            "config": config,
            "class_weights": class_weights.tolist(),
            "gradient_keys": keys,
            "AISSLab_grade_fit_or_selection": False,
        },
    )
    models, training = {}, {}
    for arm in v8.ARMS:
        x = train_features[arm][[indexed[arm][k]["feature_index"] for k in keys]]
        selection_rows = [r for r in val_rows[arm] if r["dataset"] == "rsna"]
        training[arm] = {
            "n_patients": len({k[1] for k in keys}),
            "n_targets": len(keys),
            "n_requested_targets": len(train_rows[arm]),
            "class_counts": np.bincount(labels, minlength=3).tolist(),
            "by_level_class_counts": {
                level: np.bincount(labels[levels == i], minlength=3).tolist()
                for i, level in enumerate(severity.LEVELS)
            },
            "selected_epochs": {},
            "epoch0_scores": {},
            "selected_scores": {},
        }
        models[arm] = []
        for seed, warm_start in zip(v8.SEEDS, initial, strict=True):
            baseline = severity.prediction_rows(selection_rows, warm_start, val_features[arm])
            epoch0_matches(baseline, v8.RUN / "arms" / arm / f"seed_{seed}.csv")
            model, epoch, history, _ = fit_head(
                warm_start,
                x,
                levels,
                labels,
                selection_rows,
                val_features[arm],
                class_weights,
                config,
                seed,
                device,
            )
            directory = out / "arms" / arm / f"seed_{seed}"
            directory.mkdir(parents=True)
            torch.save(
                {
                    "family": FAMILY,
                    "seed": seed,
                    "selected_epoch": epoch,
                    "state_dict": _state(model),
                    "classes": list(severity.CLASSES),
                    "levels": list(severity.LEVELS),
                    "provenance": provenance,
                },
                directory / "model.pt",
            )
            write_json(directory / "history.json", history)
            training[arm]["selected_epochs"][str(seed)] = epoch
            training[arm]["epoch0_scores"][str(seed)] = history[0]["validation_macro_f1"]
            training[arm]["selected_scores"][str(seed)] = history[epoch]["validation_macro_f1"]
            models[arm].append(model)
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "seed": seed,
                        "selected_epoch": epoch,
                        "epoch0_f1": history[0]["validation_macro_f1"],
                        "selected_f1": history[epoch]["validation_macro_f1"],
                    }
                ),
                flush=True,
            )
    checkpoints = {
        arm: {
            str(seed): sha256_file(out / "arms" / arm / f"seed_{seed}" / "model.pt")
            for seed in v8.SEEDS
        }
        for arm in v8.ARMS
    }
    write_json(
        out / "selection.json",
        {
            "status": "all_six_fixed_before_AISSLab_prediction",
            "frozen": True,
            "checkpoints": checkpoints,
            "provenance": provenance,
        },
    )
    summary = {
        "phase": "validation",
        "severity_retrained": True,
        "test_evaluated": False,
        "center_reused_exactly": True,
        "localizers_frozen": True,
        "warm_start": "v6_same_seed",
        "normalization": "frozen_v6_train_statistics",
        "severity_training_source": "rsna_train_only",
        "severity_seeds": list(v8.SEEDS),
        "localizer_ensemble": "mean_probabilities_before_argmax",
        "bootstrap_resamples": config["bootstrap_resamples"],
        "bootstrap_seed": config["bootstrap_seed"],
        "training_config": {
            k: config[k]
            for k in (
                "learning_rate",
                "weight_decay",
                "batch_size",
                "max_epochs",
                "patience",
                "epoch0_eligible",
            )
        },
        "class_weights": class_weights.tolist(),
        "provenance": provenance,
        "arms": {},
    }
    for arm in v8.ARMS:
        source, before, after = (
            v8.RUN / "arms" / arm,
            out / "arms" / arm / "before",
            out / "arms" / arm / "after",
        )
        before.mkdir()
        after.mkdir()
        all_predictions, paths = [], {}
        for seed, model in zip(v8.SEEDS, models[arm], strict=True):
            predictions = severity.prediction_rows(val_rows[arm], model, val_features[arm])
            severity._write_predictions(after / f"seed_{seed}.csv", predictions)
            all_predictions.append(predictions)
            paths[str(seed)] = str((after / f"seed_{seed}.csv").relative_to(out))
            shutil.copy2(source / f"seed_{seed}.csv", before / f"seed_{seed}.csv")
        severity._write_predictions(after / "ensemble.csv", severity._ensemble(all_predictions))
        shutil.copy2(source / "ensemble.csv", before / "ensemble.csv")
        summary["arms"][arm] = {"training": training[arm]}
        for stage, directory in (("before", before), ("after", after)):
            summary["arms"][arm][stage] = {
                "predictions": str((directory / "ensemble.csv").relative_to(out)),
                "predictions_sha256": sha256_file(directory / "ensemble.csv"),
                "seed_predictions": {
                    str(seed): str((directory / f"seed_{seed}.csv").relative_to(out))
                    for seed in v8.SEEDS
                },
            }
    data.verify_protocol(protocol_path)
    if (
        sha256_file(Path(__file__)) != provenance["implementation_sha256"]
        or sha256_file(cache / "summary.json") != provenance["training_summary_sha256"]
    ):
        raise RuntimeError("Fitting source or TRAIN cache changed during adaptation")
    write_json(out / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=data.PROTOCOL)
    parser.add_argument("--cache", type=Path, default=data.OUT)
    parser.add_argument("--out", type=Path, default=RUN)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args(argv)
    train_comparison(args.protocol, args.cache, args.out, args.device)


if __name__ == "__main__":
    main()
