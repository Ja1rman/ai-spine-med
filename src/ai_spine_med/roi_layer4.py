"""Paired frozen encoder versus layer4 adaptation on unchanged axial ROI bags.

Only RSNA TRAIN contributes gradients. VAL selects epochs; TEST is never read.
The frozen strategy reproduces v9, from the same v6 head initialization used by
the adapted strategy. Cached layer3 maps preserve each real crop separately.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import axial_features as af
from . import localization_finetuning as v9
from . import localization_severity_validation as v8
from . import localized_stenosis as severity
from . import roi_layer4_data as data
from .assets import sha256_file, write_json
from .training import evaluate_predictions, inverse_sqrt_class_weights

RUN = Path("reports/runs/roi-layer4-v11")
FAMILY = "paired_roi_layer4_v11"


class RoiLayer4(nn.Module):
    """Trainable residual block and original head; no localization network."""

    def __init__(self, layer4, head):
        super().__init__()
        self.layer4 = copy.deepcopy(layer4).requires_grad_(True)
        self.head = copy.deepcopy(head).requires_grad_(True)
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        for module in self.layer4.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def roi_features(self, maps, mask):
        if (
            maps.ndim != 5
            or maps.shape[2:] != (256, 14, 14)
            or mask.shape != maps.shape[:2]
            or mask.dtype != torch.bool
            or (mask.sum(1) == 0).any()
        ):
            raise ValueError("Require nonempty masked bags of layer3 maps")
        batch, centers = maps.shape[:2]
        values = self.layer4(maps.reshape(batch * centers, 256, 14, 14))
        values = F.adaptive_avg_pool2d(values, 1).flatten(1).reshape(batch, centers, 512)
        return (values * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]

    def forward(self, maps, mask, levels):
        return self.head(self.roi_features(maps, mask), levels)


def bag_batch(maps, offsets, indices, device):
    """Copy actual float32 maps from disk; pad only centers, never labels."""
    selected = np.asarray(offsets)[np.asarray(indices, dtype=np.int64)]
    if (
        selected.ndim != 2
        or selected.shape[1] != 2
        or not len(selected)
        or (selected[:, 0] < 0).any()
        or (selected[:, 1] > len(maps)).any()
        or ((selected[:, 1] - selected[:, 0]) < 1).any()
        or ((selected[:, 1] - selected[:, 0]) > 3).any()
    ):
        raise ValueError("Invalid ROI bag boundaries")
    values = np.zeros((len(selected), 3, 256, 14, 14), np.float32)
    mask = np.zeros((len(selected), 3), bool)
    for i, (start, end) in enumerate(selected):
        values[i, : end - start] = maps[start:end]
        mask[i, : end - start] = True
    return torch.from_numpy(values).to(device), torch.from_numpy(mask).to(device)


def extract_features(model, maps, offsets, *, batch_size=64):
    model.eval()
    device = next(model.parameters()).device
    features = []
    with torch.no_grad():
        for start in range(0, len(offsets), batch_size):
            indices = np.arange(start, min(start + batch_size, len(offsets)))
            values, mask = bag_batch(maps, offsets, indices, device)
            features.append(model.roi_features(values, mask).detach().cpu().numpy())
    return np.concatenate(features).astype(np.float32)


def _immutable_buffers(model):
    return {key: value.detach().cpu().clone() for key, value in model.named_buffers()}


def _check_buffers(model, original):
    current = dict(model.named_buffers())
    if current.keys() != original.keys() or any(
        not torch.equal(current[key].detach().cpu(), value) for key, value in original.items()
    ):
        raise RuntimeError("BatchNorm statistics or original feature normalizers changed")


def check_initial_parity(model, cache, frozen_predictions, atol):
    rows, maps, offsets, frozen_features, _ = cache
    features = extract_features(model, maps, offsets)
    maximum = float(np.max(np.abs(features - frozen_features)))
    predictions = severity.prediction_rows(rows, model.head, features)
    by_key = {(r["dataset"], r["study_id"], r["level"]): r for r in frozen_predictions}
    errors = []
    for row in predictions:
        old = by_key[(row["dataset"], row["study_id"], row["level"])]
        if row["status"] != old["status"] or row["label"] != old["label"]:
            raise ValueError("Initial ROI coverage or grade differs")
        if row["status"] == "ok":
            errors.extend(abs(row[f"p_{c}"] - old[f"p_{c}"]) for c in severity.CLASSES)
            if row["prediction"] != old["prediction"]:
                raise ValueError("Layer3 reconstruction changed an initial class prediction")
    probability_error = max(errors, default=0.0)
    if probability_error > atol:
        raise ValueError("Initial layer4 predictions differ from the frozen feature control")
    return {"maximum_feature_error": maximum, "maximum_probability_error": probability_error}


def fit_layer4(initial, train_cache, keys, indexed, val_cache, weights, config, seed, device):
    train_rows, maps, offsets, _, _ = train_cache
    val_rows, val_maps, val_offsets, _, _ = val_cache
    if any(r["dataset"] != "rsna" or r["split"] != "train" for r in train_rows) or any(
        r["dataset"] != "rsna" or r["split"] != "validation" for r in val_rows
    ):
        raise ValueError("Layer4 gradients require RSNA TRAIN; epoch selection requires VAL")
    if {r["study_id"] for r in train_rows} & {r["study_id"] for r in val_rows}:
        raise ValueError("TRAIN and VAL patients overlap")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = copy.deepcopy(initial).to(device)
    immutable = _immutable_buffers(model)
    original_layer4 = v9._state(model.layer4)
    indices = np.asarray([indexed[k]["feature_index"] for k in keys], np.int64)
    levels = np.asarray([severity.LEVELS.index(k[2]) for k in keys], np.int64)
    labels = np.asarray([indexed[k]["label"] for k in keys], np.int64)
    initial_features = extract_features(model, val_maps, val_offsets)
    predictions = severity.prediction_rows(val_rows, model.head, initial_features)
    best_f1 = evaluate_predictions(predictions)["conditional"]["macro_f1"]
    best_state, selected, stale = v9._state(model), 0, 0
    history = [{"epoch": 0, "validation_macro_f1": best_f1, "weighted_train_loss": None}]
    optimizer = torch.optim.AdamW(
        [
            {"params": model.layer4.parameters(), "lr": config["encoder_learning_rate"]},
            {"params": model.head.network.parameters(), "lr": config["head_learning_rate"]},
        ],
        weight_decay=config["weight_decay"],
    )
    class_weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
    saw_layer4_change = False
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        order, total = rng.permutation(len(keys)), 0.0
        started = time.monotonic()
        for offset in range(0, len(order), config["batch_size"]):
            ids = order[offset : offset + config["batch_size"]]
            values, mask = bag_batch(maps, offsets, indices[ids], device)
            target_levels = torch.as_tensor(levels[ids], dtype=torch.long, device=device)
            target_labels = torch.as_tensor(labels[ids], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(
                model(values, mask, target_levels),
                target_labels,
                weight=class_weights,
                reduction="sum",
            ) / len(ids)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite layer4 training loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(ids)
        _check_buffers(model, immutable)
        saw_layer4_change |= any(
            not torch.equal(value, original_layer4[k])
            for k, value in v9._state(model.layer4).items()
        )
        val_features = extract_features(model, val_maps, val_offsets)
        predictions = severity.prediction_rows(val_rows, model.head, val_features)
        score = evaluate_predictions(predictions)["conditional"]["macro_f1"]
        history.append(
            {"epoch": epoch, "validation_macro_f1": score, "weighted_train_loss": total / len(keys)}
        )
        print(
            json.dumps({"seed": seed, **history[-1], "seconds": time.monotonic() - started}),
            flush=True,
        )
        if score > best_f1:
            best_f1, best_state, selected, stale = score, v9._state(model), epoch, 0
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    if not saw_layer4_change:
        raise RuntimeError("No layer4 update occurred")
    model.load_state_dict(best_state)
    model.eval().requires_grad_(False)
    _check_buffers(model, immutable)
    return model, selected, history, {"layer4_updated_during_training": saw_layer4_change}


def _compare_v9_control(arm, seed, selected, predictions):
    from .localization_severity_report import read_validation_predictions

    summary = json.loads((v9.RUN / "summary.json").read_text())
    if selected != summary["arms"][arm]["training"]["selected_epochs"][str(seed)]:
        raise ValueError("Frozen-control epoch does not reproduce v9")
    source = read_validation_predictions(v9.RUN / "arms" / arm / "after" / f"seed_{seed}.csv")
    source = {
        (r["dataset"], r["study_id"], r["level"]): r for r in source if r["dataset"] == "rsna"
    }
    if source.keys() != {(r["dataset"], r["study_id"], r["level"]) for r in predictions}:
        raise ValueError("Frozen control requested cohort differs from v9")
    maximum = 0.0
    for row in predictions:
        old = source[(row["dataset"], row["study_id"], row["level"])]
        if (
            old["label"] != row["label"]
            or (old["status"] == "ok") != (row["status"] == "ok")
            or old["reason"] != row["reason"]
        ):
            raise ValueError("Frozen control changed requested labels or failures")
        if row["status"] == "ok":
            maximum = max(maximum, *(abs(row[f"p_{c}"] - old[f"p_{c}"]) for c in severity.CLASSES))
            if old["prediction"] != row["prediction"]:
                raise ValueError("Frozen control changed a v9 prediction")
    if maximum > 2e-7:
        raise ValueError("Frozen control no longer reproduces v9 probabilities")
    return {"v9_reproduced": True, "maximum_probability_error": maximum}


def train_comparison(protocol_path=data.PROTOCOL, cache=data.OUT, out=RUN, device="cpu"):
    protocol_path, cache, out = Path(protocol_path), Path(cache), Path(out)
    config, frozen = data.verify_protocol(protocol_path)
    if config["seeds"] != list(v8.SEEDS) or config["variants"] != ["frozen", "layer4"]:
        raise ValueError("Require fixed paired strategies and three seeds")
    device = af._device(device)
    torch.set_num_threads(4)
    cache_summary = json.loads((cache / "summary.json").read_text())
    cache_provenance = cache_summary["provenance"]
    if (
        cache_provenance["protocol_sha256"] != sha256_file(protocol_path)
        or cache_provenance["implementation_sha256"] != sha256_file(Path(data.__file__))
        or cache_provenance["frozen_files"] != frozen
    ):
        raise ValueError("Prefix cache was prepared with different source or protocol")
    cached = {
        arm: {split: data.load_cache(cache, arm, split) for split in ("train", "validation")}
        for arm in v8.ARMS
    }
    train_tables = {arm: cached[arm]["train"][0] for arm in v8.ARMS}
    keys, indexed = v9.common_training_keys(train_tables)
    labels = np.array([indexed["xml_only"][k]["label"] for k in keys], np.int64)
    levels = np.array([severity.LEVELS.index(k[2]) for k in keys], np.int64)
    weights = inverse_sqrt_class_weights(labels)
    initial_heads, _, _ = severity.load_sealed_models(v8.GRADERS, device=device)
    original_encoder = af._load_encoder(v8.extract.WEIGHTS, device)
    provenance = {
        "protocol_sha256": sha256_file(protocol_path),
        "implementation_sha256": sha256_file(Path(__file__)),
        "cache_implementation_sha256": sha256_file(Path(data.__file__)),
        "cache_summary_sha256": sha256_file(cache / "summary.json"),
        "frozen_files": frozen,
        "common_train_cohort_sha256": hashlib.sha256(
            json.dumps(keys, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    manifest = {"family": FAMILY, "provenance": provenance, "config": config}
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists() and json.loads(
        (out / "manifest.json").read_text()
    ) != manifest:
        raise ValueError("Changed experiment; use a new output directory")
    if (out / "summary.json").exists():
        raise ValueError("Completed experiment is immutable")
    write_json(out / "manifest.json", manifest)
    summary = {
        "phase": "validation",
        "test_evaluated": False,
        "localizers_frozen": True,
        "center_reused_exactly": True,
        "roi_reused_exactly": True,
        "batch_norm_frozen": True,
        "severity_training_source": "rsna_train_only",
        "seeds": list(v8.SEEDS),
        "bootstrap_resamples": config["bootstrap_resamples"],
        "bootstrap_seed": config["bootstrap_seed"],
        "provenance": provenance,
        "training_config": {
            k: config[k]
            for k in (
                "head_learning_rate",
                "encoder_learning_rate",
                "weight_decay",
                "batch_size",
                "max_epochs",
                "patience",
                "epoch0_eligible",
            )
        },
        "training": {
            "n_patients": len({k[1] for k in keys}),
            "n_targets": len(keys),
            "n_requested_targets": len(train_tables["xml_only"]),
            "class_counts": np.bincount(labels, minlength=3).tolist(),
            "by_level_class_counts": {
                level: np.bincount(labels[levels == i], minlength=3).tolist()
                for i, level in enumerate(severity.LEVELS)
            },
            "class_weights": weights.tolist(),
        },
        "input_files": {},
        "arms": {},
    }
    for arm in v8.ARMS:
        train_cache, val_cache = cached[arm]["train"], cached[arm]["validation"]
        val_rows, _, _, val_features, _ = val_cache
        x = train_cache[3][[indexed[arm][k]["feature_index"] for k in keys]]
        summary["arms"][arm] = {"variants": {}}
        for variant in ("frozen", "layer4"):
            entry = {"seed_predictions": {}, "histories": {}, "selected_epochs": {}}
            predictions_per_seed = []
            for seed, head in zip(v8.SEEDS, initial_heads, strict=True):
                directory = out / "arms" / arm / variant / f"seed_{seed}"
                directory.mkdir(parents=True, exist_ok=True)
                print(
                    json.dumps({"arm": arm, "variant": variant, "seed": seed, "status": "started"}),
                    flush=True,
                )
                if variant == "frozen":
                    fit_config = {**config, "learning_rate": config["head_learning_rate"]}
                    fitted, selected, history, _ = v9.fit_head(
                        head,
                        x,
                        levels,
                        labels,
                        val_rows,
                        val_features,
                        weights,
                        fit_config,
                        seed,
                        device,
                    )
                    predictions = severity.prediction_rows(val_rows, fitted, val_features)
                    audit = _compare_v9_control(arm, seed, selected, predictions)
                    state = {"head": v9._state(fitted)}
                else:
                    initial = RoiLayer4(original_encoder.layer4, head).to(device)
                    baseline = severity.prediction_rows(val_rows, head, val_features)
                    parity = check_initial_parity(
                        initial, val_cache, baseline, config["initial_probability_atol"]
                    )
                    fitted, selected, history, audit = fit_layer4(
                        initial,
                        train_cache,
                        keys,
                        indexed[arm],
                        val_cache,
                        weights,
                        config,
                        seed,
                        device,
                    )
                    audit["initial_parity"] = parity
                    features = extract_features(fitted, val_cache[1], val_cache[2])
                    predictions = severity.prediction_rows(val_rows, fitted.head, features)
                    state = {"layer4_and_head": v9._state(fitted)}
                torch.save(
                    {
                        "family": FAMILY,
                        "arm": arm,
                        "variant": variant,
                        "seed": seed,
                        "selected_epoch": selected,
                        "classes": list(severity.CLASSES),
                        "levels": list(severity.LEVELS),
                        "provenance": provenance,
                        **state,
                    },
                    directory / "model.pt",
                )
                write_json(directory / "history.json", history)
                write_json(directory / "audit.json", audit)
                severity._write_predictions(directory / "validation.csv", predictions)
                predictions_per_seed.append(predictions)
                entry["seed_predictions"][str(seed)] = str(
                    (directory / "validation.csv").relative_to(out)
                )
                entry["histories"][str(seed)] = str((directory / "history.json").relative_to(out))
                entry["selected_epochs"][str(seed)] = selected
                print(
                    json.dumps(
                        {
                            "arm": arm,
                            "variant": variant,
                            "seed": seed,
                            "selected_epoch": selected,
                            "macro_f1": evaluate_predictions(predictions)["conditional"][
                                "macro_f1"
                            ],
                        }
                    ),
                    flush=True,
                )
            ensemble_path = out / "arms" / arm / variant / "ensemble.csv"
            severity._write_predictions(ensemble_path, severity._ensemble(predictions_per_seed))
            entry["ensemble"] = str(ensemble_path.relative_to(out))
            summary["arms"][arm]["variants"][variant] = entry
    for path in sorted(out.glob("arms/**/*")):
        if path.is_file():
            summary["input_files"][str(path.relative_to(out))] = sha256_file(path)
    data.verify_protocol(protocol_path)
    if (
        sha256_file(Path(__file__)) != provenance["implementation_sha256"]
        or sha256_file(cache / "summary.json") != provenance["cache_summary_sha256"]
    ):
        raise RuntimeError("Training implementation or cache changed during fitting")
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
