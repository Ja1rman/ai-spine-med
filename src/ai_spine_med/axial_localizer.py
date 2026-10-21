"""Supervised axial level and canal-center localization, independent of grades."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .assets import sha256_file, write_json

FAMILY = "explicit_axial_localizer_v1"
LEVELS = ("L4-L5", "L5-S1")


class SliceLocalizer(nn.Module):
    def __init__(self, mean, std, prior_mean, prior_std):
        super().__init__()
        for name, value in (
            ("mean", mean),
            ("std", std),
            ("prior_mean", prior_mean),
            ("prior_std", prior_std),
        ):
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32).clone())
        self.scorer = nn.Sequential(nn.Linear(514, 64), nn.Tanh(), nn.Linear(64, 2))
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)

    def forward(self, features, positions, mask):
        if (
            features.ndim != 3
            or positions.shape != (*features.shape[:2], 2)
            or mask.shape != features.shape[:2]
            or not mask.any(1).all()
        ):
            raise ValueError("Require nonempty padded slice bags and matching geometry")
        normalized = (features - self.mean) / self.std
        residual = self.scorer(torch.cat((normalized, positions), -1)).transpose(1, 2)
        z = positions[..., 0][:, None]
        prior = (
            -0.5 * ((z - self.prior_mean[None, :, None]) / self.prior_std[None, :, None]).square()
        )
        return (prior + residual).masked_fill(~mask[:, None], float("-inf"))


class CanalCenterModel(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).clone())
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).clone())
        self.head = nn.Sequential(nn.Conv2d(512, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 1, 1))
        y, x = torch.meshgrid(torch.linspace(0, 1, 7), torch.linspace(0, 1, 7), indexing="ij")
        self.register_buffer("grid", torch.stack((x, y), -1).reshape(49, 2))

    def forward(self, maps):
        if maps.ndim != 4 or maps.shape[1:] != (512, 7, 7):
            raise ValueError("Canal center requires original frozen 512x7x7 maps")
        logits = self.head(
            (maps - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        ).flatten(1)
        probabilities = logits.softmax(-1)
        return probabilities @ self.grid, logits


def read_training_data(index_path):
    index_path = Path(index_path)
    index = json.loads(index_path.read_text())
    rows = index["records"]
    # Reject TEST identities before accessing any feature/target arrays.
    if any(r["split"] not in {"train", "validation"} for r in rows):
        raise ValueError("TEST is forbidden in localization fitting/selection")
    seen, data = set(), []
    for row in rows:
        key = row["dataset"], row["study_id"]
        if key in seen:
            raise ValueError("A localization patient occurs more than once")
        seen.add(key)
        if row["status"] != "ok":
            continue
        path = (index_path.parent / row["arrays_path"]).resolve()
        if (
            not path.is_relative_to(index_path.parent.resolve())
            or sha256_file(path) != row["arrays_sha256"]
        ):
            raise ValueError("Localization arrays changed or escaped their root")
        with np.load(path, allow_pickle=False) as archive:
            arrays = dict(archive)
        n = len(arrays["features"])
        if (
            arrays["features"].shape != (n, 512)
            or arrays["positions"].shape != (n, 2)
            or arrays["relevance"].shape != (2, n)
        ):
            raise ValueError("Invalid localization feature/target dimensions")
        if not np.isfinite(arrays["features"]).all() or not np.isfinite(arrays["positions"]).all():
            raise ValueError("Nonfinite model inputs")
        if arrays["center_maps"].shape[0] and row["dataset"] != "aisslab":
            raise ValueError("RSNA subarticular points cannot supervise canal centers")
        data.append({"record": row, **arrays})
    return index, data


def _batch(rows, device):
    n = max(len(r["features"]) for r in rows)
    x, p = np.zeros((len(rows), n, 512), np.float32), np.zeros((len(rows), n, 2), np.float32)
    target, mask, known = (
        np.zeros((len(rows), 2, n), np.float32),
        np.zeros((len(rows), n), bool),
        [],
    )
    for i, row in enumerate(rows):
        size = len(row["features"])
        x[i, :size], p[i, :size], target[i, :, :size], mask[i, :size] = (
            row["features"],
            row["positions"],
            row["relevance"],
            True,
        )
        known.append(row["level_mask"])
    return tuple(torch.as_tensor(a, device=device) for a in (x, p, mask, target, np.asarray(known)))


def predict_slice_weights(model, features, positions):
    """Inference reads only images' features and geometry, never annotations."""
    device = next(model.parameters()).device
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    p = torch.as_tensor(positions, dtype=torch.float32, device=device)
    model.eval()
    with torch.inference_mode():
        weights = model(
            x[None], p[None], torch.ones((1, len(x)), dtype=torch.bool, device=device)
        ).softmax(-1)[0]
    return weights.cpu().numpy()


def _assess_levels(model, rows, device):
    groups = defaultdict(list)
    losses = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(rows), 16):
            subset = rows[offset : offset + 16]
            x, p, mask, target, known = _batch(subset, device)
            logits = model(x, p, mask)
            # Avoid 0 * -inf for padded/unreferenced slices.
            logp = logits.log_softmax(-1).masked_fill(~mask[:, None], 0)
            ce = -(target * logp).sum(-1)
            weights = logits.softmax(-1).cpu().numpy()
            for i, row in enumerate(subset):
                for level in (0, 1):
                    if not row["level_mask"][level]:
                        continue
                    size = len(row["features"])
                    distances = row["anchor_distances_mm"][level]
                    top = int(weights[i, level, :size].argmax())
                    groups[row["record"]["dataset"], LEVELS[level]].append(
                        {
                            "hit10": bool(distances[top] <= 10),
                            "hit5": bool(distances[top] <= 5),
                            "distance": float(distances[top]),
                            "uniform_hit10": float(np.mean(distances <= 10)),
                        }
                    )
                    losses.append(float(ce[i, level]))
    summary = {}
    for (dataset, level), entries in groups.items():
        finite = [r["distance"] for r in entries if np.isfinite(r["distance"])]
        summary[f"{dataset}/{level}"] = {
            "n": len(entries),
            "hit10": float(np.mean([r["hit10"] for r in entries])),
            "hit5": float(np.mean([r["hit5"] for r in entries])),
            "median_distance_mm_when_verifiable": float(np.median(finite)) if finite else None,
            "unverifiable_top_series": sum(not np.isfinite(r["distance"]) for r in entries),
            "uniform_expected_hit10": float(np.mean([r["uniform_hit10"] for r in entries])),
        }
    return {
        "groups": summary,
        "macro_hit10": float(np.mean([r["hit10"] for r in summary.values()])),
        "cross_entropy": float(np.mean(losses)),
    }


def _center_data(rows):
    selected = [r for r in rows if len(r["center_maps"])]
    return tuple(
        torch.from_numpy(np.concatenate([r[key] for r in selected])).float()
        for key in ("center_maps", "center_xy", "center_scale_mm")
    )


def _assess_centers(model, tensors, device):
    maps, target, scale = tensors
    predictions = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(maps), 32):
            xy, _ = model(maps[offset : offset + 32].to(device))
            predictions.append(xy.cpu())
    xy = torch.cat(predictions)
    errors = ((xy - target) * scale).square().sum(-1).sqrt().numpy()
    return {
        "n": len(errors),
        "mean_error_mm": float(errors.mean()),
        "median_error_mm": float(np.median(errors)),
        "within10mm": float(np.mean(errors <= 10)),
        "p90_error_mm": float(np.quantile(errors, 0.9)),
    }


def train_localizer(index_path, protocol_path, out, device="cpu"):
    index_path, protocol_path, out = map(Path, (index_path, protocol_path, out))
    out.mkdir(parents=True, exist_ok=True)
    if (out / "selection.json").exists():
        raise ValueError("Localization selection already frozen; use a new experiment directory")
    provenance = {
        "protocol_sha256": sha256_file(protocol_path),
        "training_index_sha256": sha256_file(index_path),
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    config = json.loads(protocol_path.read_text())
    seed = config["localizer_seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(4)
    index, rows = read_training_data(index_path)
    train = [r for r in rows if r["record"]["split"] == "train"]
    val = [r for r in rows if r["record"]["split"] == "validation"]
    all_features = np.concatenate([r["features"] for r in train])
    mean, std = all_features.mean(0), np.maximum(all_features.std(0), 1e-4)
    prior_values = [[], []]
    for row in train:
        for level in (0, 1):
            if row["level_mask"][level]:
                prior_values[level].append(float(row["relevance"][level] @ row["positions"][:, 0]))
    pm = np.array([np.median(v) for v in prior_values], np.float32)
    ps = np.clip(
        [1.4826 * np.median(np.abs(np.asarray(v) - pm[i])) for i, v in enumerate(prior_values)],
        0.08,
        0.30,
    ).astype(np.float32)
    levels = SliceLocalizer(mean, std, pm, ps).to(device)
    optimizer = torch.optim.AdamW(
        levels.parameters(),
        lr=config["localizer_learning_rate"],
        weight_decay=config["localizer_weight_decay"],
    )
    baseline = _assess_levels(levels, val, device)
    history = [{"epoch": 0, "validation": baseline}]
    best, state, best_epoch, stale = (
        (baseline["macro_hit10"], -baseline["cross_entropy"]),
        copy.deepcopy(levels.state_dict()),
        0,
        0,
    )
    domains = Counter(r["record"]["dataset"] for r in train)
    generator = np.random.default_rng(seed)
    started = time.monotonic()
    for epoch in range(1, config["localizer_max_epochs"] + 1):
        levels.train()
        order, losses = generator.permutation(len(train)), []
        for offset in range(0, len(order), 8):
            subset = [train[i] for i in order[offset : offset + 8]]
            x, p, mask, target, known = _batch(subset, device)
            optimizer.zero_grad()
            logp = levels(x, p, mask).log_softmax(-1).masked_fill(~mask[:, None], 0)
            per_level = -(target * logp).sum(-1)
            per_study = (per_level * known).sum(-1) / known.sum(-1).clamp_min(1)
            domain_weight = torch.tensor(
                [len(train) / (len(domains) * domains[r["record"]["dataset"]]) for r in subset],
                device=device,
            )
            loss = (per_study * domain_weight).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        assessed = _assess_levels(levels, val, device)
        entry = {"epoch": epoch, "training_loss": float(np.mean(losses)), "validation": assessed}
        history.append(entry)
        score = assessed["macro_hit10"], -assessed["cross_entropy"]
        if score > best:
            best, state, best_epoch, stale = score, copy.deepcopy(levels.state_dict()), epoch, 0
        else:
            stale += 1
        print(
            json.dumps(
                {
                    "stage": "level",
                    "epoch": epoch,
                    "validation_hit10": assessed["macro_hit10"],
                    "best_epoch": best_epoch,
                }
            ),
            flush=True,
        )
        if stale >= config["localizer_patience"]:
            break
    levels.load_state_dict(state)
    levels.eval()
    center_train, center_val = _center_data(train), _center_data(val)
    centers = CanalCenterModel(mean, std).to(device)
    optimizer = torch.optim.AdamW(
        centers.parameters(),
        lr=config["center_learning_rate"],
        weight_decay=config["localizer_weight_decay"],
    )
    baseline_center = _assess_centers(centers, center_val, device)
    center_history = [{"epoch": 0, "validation": baseline_center}]
    best_center, center_state, center_epoch, stale = (
        baseline_center["mean_error_mm"],
        copy.deepcopy(centers.state_dict()),
        0,
        0,
    )
    maps, xy, scales = center_train
    for epoch in range(1, config["center_max_epochs"] + 1):
        centers.train()
        order, losses = generator.permutation(len(maps)), []
        for offset in range(0, len(order), config["center_batch_size"]):
            ids = order[offset : offset + config["center_batch_size"]]
            x, target = maps[ids].to(device), xy[ids].to(device)
            optimizer.zero_grad()
            predicted, logits = centers(x)
            heatmap = torch.exp(
                -0.5
                * ((centers.grid[None] - target[:, None]) / config["center_heatmap_sigma"])
                .square()
                .sum(-1)
            )
            heatmap = heatmap / heatmap.sum(-1, keepdim=True)
            loss = -(heatmap * logits.log_softmax(-1)).sum(-1).mean() + 10 * F.mse_loss(
                predicted, target
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        assessed = _assess_centers(centers, center_val, device)
        center_history.append(
            {"epoch": epoch, "training_loss": float(np.mean(losses)), "validation": assessed}
        )
        if assessed["mean_error_mm"] < best_center:
            best_center, center_state, center_epoch, stale = (
                assessed["mean_error_mm"],
                copy.deepcopy(centers.state_dict()),
                epoch,
                0,
            )
        else:
            stale += 1
        print(
            json.dumps(
                {
                    "stage": "center",
                    "epoch": epoch,
                    "validation_mean_error_mm": assessed["mean_error_mm"],
                    "best_epoch": center_epoch,
                }
            ),
            flush=True,
        )
        if stale >= config["center_patience"]:
            break
    centers.load_state_dict(center_state)
    centers.eval()
    val_levels, val_centers = (
        _assess_levels(levels, val, device),
        _assess_centers(centers, center_val, device),
    )
    gate = val_levels["macro_hit10"] >= 0.5 and val_centers["median_error_mm"] <= 10
    if provenance != {
        "protocol_sha256": sha256_file(protocol_path),
        "training_index_sha256": sha256_file(index_path),
        "implementation_sha256": sha256_file(Path(__file__)),
    }:
        raise RuntimeError("Localization data, protocol or implementation changed during fitting")
    checkpoint = {
        "schema_version": 1,
        "model_family": FAMILY,
        "levels": list(LEVELS),
        **provenance,
        "level_state": {k: v.cpu() for k, v in levels.state_dict().items()},
        "center_state": {k: v.cpu() for k, v in centers.state_dict().items()},
        "level_epoch": best_epoch,
        "center_epoch": center_epoch,
    }
    torch.save(checkpoint, out / "model.pt")
    summary = {
        "status": "complete",
        "model_family": FAMILY,
        **provenance,
        "fitting_studies": len(train),
        "validation_studies": len(val),
        "original_failures": index.get("excluded_original_failures", []),
        "level_epoch": best_epoch,
        "center_epoch": center_epoch,
        "positional_baseline": baseline,
        "validation_levels": val_levels,
        "validation_centers": val_centers,
        "localization_gate_passed": gate,
        "seconds": time.monotonic() - started,
    }
    write_json(out / "history.json", {"levels": history, "center": center_history})
    write_json(out / "summary.json", summary)
    selection = {
        "schema_version": 1,
        "model_family": FAMILY,
        "frozen": True,
        **provenance,
        "checkpoint": "model.pt",
        "checkpoint_sha256": sha256_file(out / "model.pt"),
        "summary_sha256": sha256_file(out / "summary.json"),
        "localization_gate_passed": gate,
    }
    write_json(out / "selection.json", selection)
    return summary


def load_localizer_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_family") != FAMILY or checkpoint.get("levels") != list(LEVELS):
        raise ValueError("Invalid explicit axial localizer checkpoint")
    state = checkpoint["level_state"]
    model = SliceLocalizer(state["mean"], state["std"], state["prior_mean"], state["prior_std"])
    model.load_state_dict(state, strict=True)
    centers = CanalCenterModel(state["mean"], state["std"])
    centers.load_state_dict(checkpoint["center_state"], strict=True)
    return model.to(device).eval(), centers.to(device).eval(), checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index", type=Path, default=Path("artifacts/axial-localizer-v6/index.json")
    )
    parser.add_argument("--protocol", type=Path, default=Path("configs/axial-localizer-v6.json"))
    parser.add_argument(
        "--out", type=Path, default=Path("reports/runs/explicit-localization-v6/localizer")
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args(argv)
    summary = train_localizer(args.index, args.protocol, args.out, args.device)
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "level_epoch",
                    "center_epoch",
                    "validation_levels",
                    "validation_centers",
                    "localization_gate_passed",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
