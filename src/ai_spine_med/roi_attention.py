"""Mean versus masked attention over the exact frozen neighboring ROI features.

This development-only experiment changes pooling and trains the original MLP.
The encoder, coordinates, actual ROI bags, class weights and splits stay fixed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import axial_features as af
from . import localization_finetuning as v9
from . import localization_severity_validation as v8
from . import localized_stenosis as severity
from . import roi_attention_data as data
from . import roi_layer4 as previous
from .assets import sha256_file, write_json
from .training import evaluate_predictions, inverse_sqrt_class_weights

RUN = Path("reports/runs/roi-attention-v12")
FAMILY = "paired_roi_neighbor_attention_v12"


class NeighborAttention(nn.Module):
    """Score actual 2.5D windows, pool raw vectors, then use the original head."""

    def __init__(self, head, hidden=64):
        super().__init__()
        self.head = copy.deepcopy(head).requires_grad_(True)
        self.scorer = nn.Sequential(
            nn.Linear(512, hidden), nn.Tanh(), nn.Linear(hidden, 1, bias=False)
        )
        nn.init.zeros_(self.scorer[-1].weight)
        self.eval()

    def pool(self, features, mask):
        if (
            features.ndim != 3
            or features.shape[2] != 512
            or mask.shape != features.shape[:2]
            or mask.dtype != torch.bool
            or (mask.sum(1) == 0).any()
        ):
            raise ValueError("Require nonempty masked bags of frozen512features")
        normalized = (features - self.head.mean) / self.head.std
        scores = self.scorer(normalized).squeeze(-1).masked_fill(~mask, -torch.inf)
        weights = scores.softmax(dim=1)
        return (features * weights[:, :, None]).sum(1), weights

    def forward(self, features, mask, levels):
        pooled, _ = self.pool(features, mask)
        return self.head(pooled, levels)


def bag_batch(features, offsets, indices, device):
    selected = np.asarray(offsets)[np.asarray(indices, dtype=np.int64)]
    if (
        selected.ndim != 2
        or selected.shape[1] != 2
        or not len(selected)
        or (selected[:, 0] < 0).any()
        or (selected[:, 1] > len(features)).any()
        or ((selected[:, 1] - selected[:, 0]) < 1).any()
        or ((selected[:, 1] - selected[:, 0]) > 3).any()
    ):
        raise ValueError("Invalid original ROI bag boundaries")
    values = np.zeros((len(selected), 3, 512), np.float32)
    mask = np.zeros((len(selected), 3), bool)
    for i, (start, end) in enumerate(selected):
        values[i, : end - start] = features[start:end]
        mask[i, : end - start] = True
    return torch.from_numpy(values).to(device), torch.from_numpy(mask).to(device)


def pooled_features(model, cache, *, batch_size=256):
    _, features, offsets, _, _ = cache
    device = next(model.parameters()).device
    model.eval()
    pooled, attention = [], []
    with torch.no_grad():
        for start in range(0, len(offsets), batch_size):
            x, mask = bag_batch(
                features, offsets, np.arange(start, min(start + batch_size, len(offsets))), device
            )
            values, weights = model.pool(x, mask)
            pooled.append(values.detach().cpu().numpy())
            attention.append(weights.detach().cpu().numpy())
    return np.concatenate(pooled).astype(np.float32), np.concatenate(attention).astype(np.float32)


def check_initial_parity(model, cache, atol):
    rows, _, offsets, frozen_features, _ = cache
    features, weights = pooled_features(model, cache)
    maximum = float(np.abs(features - frozen_features).max())
    if not np.isfinite(features).all() or maximum > 1e-4:
        raise ValueError("Uniform attention does not reproduce the historical mean features")
    before = severity.prediction_rows(rows, model.head, frozen_features)
    after = severity.prediction_rows(rows, model.head, features)
    error = 0.0
    for first, second in zip(before, after, strict=True):
        if first["status"] == "ok":
            error = max(error, *(abs(first[f"p_{c}"] - second[f"p_{c}"]) for c in severity.CLASSES))
            if first["prediction"] != second["prediction"]:
                raise ValueError("Uniform attention changes an initial category")
    for i, (start, end) in enumerate(offsets):
        count = end - start
        if not np.allclose(weights[i, :count], 1 / count, rtol=0, atol=1e-7) or np.any(
            weights[i, count:]
        ):
            raise ValueError("Initial attention is not uniform over actual centers only")
    if error > atol:
        raise ValueError("Uniform attention does not reproduce initial mean probabilities")
    return {
        "maximum_feature_error": maximum,
        "maximum_probability_error": error,
        "initial_weights_uniform": True,
    }


def fit_attention(initial, train_cache, keys, indexed, val_cache, weights, config, seed, device):
    train_rows = train_cache[0]
    val_rows = val_cache[0]
    if any(r["dataset"] != "rsna" or r["split"] != "train" for r in train_rows) or any(
        r["dataset"] != "rsna" or r["split"] != "validation" for r in val_rows
    ):
        raise ValueError("Gradients require RSNA TRAIN; epoch selection requires VAL")
    if {r["study_id"] for r in train_rows} & {r["study_id"] for r in val_rows}:
        raise ValueError("TRAIN and VAL patients overlap")
    model = copy.deepcopy(initial).to(device)
    immutable = previous._immutable_buffers(model)
    initial_scorer = v9._state(model.scorer)
    indices = np.array([indexed[k]["feature_index"] for k in keys], np.int64)
    x, mask = bag_batch(train_cache[1], train_cache[2], indices, device)
    levels = torch.as_tensor(
        [severity.LEVELS.index(k[2]) for k in keys], dtype=torch.long, device=device
    )
    labels = torch.as_tensor([indexed[k]["label"] for k in keys], dtype=torch.long, device=device)
    class_weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    features, _ = pooled_features(model, val_cache)
    predictions = severity.prediction_rows(val_rows, model.head, features)
    best_f1 = evaluate_predictions(predictions)["conditional"]["macro_f1"]
    best_state, selected, stale = v9._state(model), 0, 0
    history = [{"epoch": 0, "validation_macro_f1": best_f1, "weighted_train_loss": None}]
    optimizer = torch.optim.AdamW(
        [
            {"params": model.scorer.parameters(), "lr": config["attention_learning_rate"]},
            {"params": model.head.network.parameters(), "lr": config["head_learning_rate"]},
        ],
        weight_decay=config["weight_decay"],
    )
    changed = False
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        order, total = rng.permutation(len(keys)), 0.0
        for offset in range(0, len(order), config["batch_size"]):
            batch = torch.as_tensor(order[offset : offset + config["batch_size"]], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(
                model(x[batch], mask[batch], levels[batch]),
                labels[batch],
                weight=class_weights,
                reduction="sum",
            ) / len(batch)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite attention training loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(batch)
        previous._check_buffers(model, immutable)
        changed |= not torch.equal(
            model.scorer[-1].weight.detach().cpu(), initial_scorer["2.weight"]
        )
        features, _ = pooled_features(model, val_cache)
        predictions = severity.prediction_rows(val_rows, model.head, features)
        score = evaluate_predictions(predictions)["conditional"]["macro_f1"]
        history.append(
            {"epoch": epoch, "validation_macro_f1": score, "weighted_train_loss": total / len(keys)}
        )
        print(json.dumps({"seed": seed, **history[-1]}), flush=True)
        if score > best_f1:
            best_f1, best_state, selected, stale = score, v9._state(model), epoch, 0
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    if not changed:
        raise RuntimeError("Attention scorer did not receive any parameter update")
    model.load_state_dict(best_state)
    model.eval().requires_grad_(False)
    previous._check_buffers(model, immutable)
    return model, selected, history, {"candidate_attention_updated": changed}


def train_comparison(protocol_path=data.PROTOCOL, cache=data.OUT, out=RUN, device="cpu"):
    protocol_path, cache, out = Path(protocol_path), Path(cache), Path(out)
    config, frozen = data.verify_protocol(protocol_path)
    if config["seeds"] != list(v8.SEEDS) or config["variants"] != ["mean", "attention"]:
        raise ValueError("Require the fixed paired attention experiment")
    device = af._device(device)
    torch.set_num_threads(4)
    receipt = json.loads((cache / "summary.json").read_text())["provenance"]
    if (
        receipt["protocol_sha256"] != sha256_file(protocol_path)
        or receipt["implementation_sha256"] != sha256_file(Path(data.__file__))
        or receipt["frozen_files"] != frozen
    ):
        raise ValueError("Attention cache source or protocol changed")
    cached = {
        arm: {split: data.load_cache(cache, arm, split) for split in ("train", "validation")}
        for arm in v8.ARMS
    }
    train_tables = {arm: cached[arm]["train"][0] for arm in v8.ARMS}
    keys, indexed = v9.common_training_keys(train_tables)
    labels = np.array([indexed["xml_only"][k]["label"] for k in keys], np.int64)
    levels = np.array([severity.LEVELS.index(k[2]) for k in keys], np.int64)
    weights = inverse_sqrt_class_weights(labels)
    heads, _, _ = severity.load_sealed_models(v8.GRADERS, device=device)
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
    manifest = {"family": FAMILY, "config": config, "provenance": provenance}
    if out.exists():
        raise ValueError("Use a new output directory; experiments are immutable")
    out.mkdir(parents=True)
    write_json(out / "manifest.json", manifest)
    summary = {
        "phase": "validation",
        "test_evaluated": False,
        "encoder_frozen": True,
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
                "attention_learning_rate",
                "attention_hidden",
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
        for variant in ("mean", "attention"):
            entry = {"seed_predictions": {}, "histories": {}, "selected_epochs": {}}
            per_seed = []
            for seed, head in zip(v8.SEEDS, heads, strict=True):
                directory = out / "arms" / arm / variant / f"seed_{seed}"
                directory.mkdir(parents=True)
                print(
                    json.dumps({"arm": arm, "variant": variant, "seed": seed, "status": "started"}),
                    flush=True,
                )
                if variant == "mean":
                    fitted, epoch, history, _ = v9.fit_head(
                        head,
                        x,
                        levels,
                        labels,
                        val_rows,
                        val_features,
                        weights,
                        {**config, "learning_rate": config["head_learning_rate"]},
                        seed,
                        device,
                    )
                    predictions = severity.prediction_rows(val_rows, fitted, val_features)
                    audit = previous._compare_v9_control(arm, seed, epoch, predictions)
                    state = {"head": v9._state(fitted)}
                else:
                    torch.manual_seed(seed)
                    initial = NeighborAttention(head, config["attention_hidden"]).to(device)
                    parity = {
                        split: check_initial_parity(
                            initial, cached[arm][split], config["initial_probability_atol"]
                        )
                        for split in ("train", "validation")
                    }
                    fitted, epoch, history, audit = fit_attention(
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
                    features, attention = pooled_features(fitted, val_cache)
                    predictions = severity.prediction_rows(val_rows, fitted.head, features)
                    audit["initial_parity"] = parity
                    audit["selected_attention_changed"] = bool(
                        torch.count_nonzero(fitted.scorer[-1].weight)
                    )
                    np.savez_compressed(
                        directory / "attention_weights.npz",
                        weights=attention,
                        counts=val_cache[2][:, 1] - val_cache[2][:, 0],
                    )
                    state = {"attention_and_head": v9._state(fitted)}
                torch.save(
                    {
                        "family": FAMILY,
                        "arm": arm,
                        "variant": variant,
                        "seed": seed,
                        "selected_epoch": epoch,
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
                entry["seed_predictions"][str(seed)] = str(
                    (directory / "validation.csv").relative_to(out)
                )
                entry["histories"][str(seed)] = str((directory / "history.json").relative_to(out))
                entry["selected_epochs"][str(seed)] = epoch
                per_seed.append(predictions)
                print(
                    json.dumps(
                        {
                            "arm": arm,
                            "variant": variant,
                            "seed": seed,
                            "selected_epoch": epoch,
                            "macro_f1": evaluate_predictions(predictions)["conditional"][
                                "macro_f1"
                            ],
                        }
                    ),
                    flush=True,
                )
            path = out / "arms" / arm / variant / "ensemble.csv"
            severity._write_predictions(path, severity._ensemble(per_seed))
            entry["ensemble"] = str(path.relative_to(out))
            summary["arms"][arm]["variants"][variant] = entry
    summary["input_files"] = {
        str(p.relative_to(out)): sha256_file(p)
        for p in sorted(out.glob("arms/**/*"))
        if p.is_file()
    }
    data.verify_protocol(protocol_path)
    if (
        sha256_file(Path(__file__)) != provenance["implementation_sha256"]
        or sha256_file(cache / "summary.json") != provenance["cache_summary_sha256"]
    ):
        raise RuntimeError("Training source or cache changed during fitting")
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
