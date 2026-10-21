"""Compare XML-only and mixed level training; keep the v6 canal center fixed.

Both arms select epochs on AISSLab validation only. RSNA validation is read for
evaluation after every model is sealed. No TEST or severity labels are loaded.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from . import axial_localizer as base
from .assets import sha256_file, write_json

PROTOCOL = Path("configs/xml-localizer-v7.json")
RUN = Path("reports/runs/xml-localizer-v7")


def partition_rows(rows, datasets):
    """Apply the actual training-source filter, not just a descriptive config."""
    if not datasets or not set(datasets) <= {"aisslab", "rsna"}:
        raise ValueError("Invalid localization training datasets")
    if any(r["record"]["split"] not in {"train", "validation"} for r in rows):
        raise ValueError("TEST forbidden in XML localization comparison")
    train = [
        r for r in rows if r["record"]["split"] == "train" and r["record"]["dataset"] in datasets
    ]
    val = [
        r
        for r in rows
        if r["record"]["split"] == "validation" and r["record"]["dataset"] == "aisslab"
    ]
    if not train or not val:
        raise ValueError("Require TRAIN and AISSLab validation patients")
    return train, val


def make_model(train, device):
    features = np.concatenate([r["features"] for r in train])
    mean, std = features.mean(0), np.maximum(features.std(0), 1e-4)
    values = [[], []]
    for row in train:
        for level in (0, 1):
            if row["level_mask"][level]:
                values[level].append(float(row["relevance"][level] @ row["positions"][:, 0]))
    if any(not v for v in values):
        raise ValueError("Both levels require TRAIN references")
    pm = np.array([np.median(v) for v in values], np.float32)
    ps = np.clip(
        [1.4826 * np.median(np.abs(np.asarray(v) - pm[i])) for i, v in enumerate(values)],
        0.08,
        0.30,
    ).astype(np.float32)
    return base.SliceLocalizer(mean, std, pm, ps).to(device)


def fit_level(train, val, config, seed, device):
    """Same level loss and budget as v6; AISSLab-only epoch selection."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = make_model(train, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["localizer_learning_rate"],
        weight_decay=config["localizer_weight_decay"],
    )
    baseline = base._assess_levels(model, val, device)
    best, state, selected, stale = (
        (baseline["macro_hit10"], -baseline["cross_entropy"]),
        copy.deepcopy(model.state_dict()),
        0,
        0,
    )
    history = [{"epoch": 0, "selection_validation": baseline}]
    domains = Counter(r["record"]["dataset"] for r in train)
    rng = np.random.default_rng(seed)
    for epoch in range(1, config["localizer_max_epochs"] + 1):
        model.train()
        order = rng.permutation(len(train))
        for offset in range(0, len(order), 8):
            subset = [train[i] for i in order[offset : offset + 8]]
            x, p, mask, target, known = base._batch(subset, device)
            optimizer.zero_grad()
            logp = model(x, p, mask).log_softmax(-1).masked_fill(~mask[:, None], 0)
            per_level = -(target * logp).sum(-1)
            per_study = (per_level * known).sum(-1) / known.sum(-1).clamp_min(1)
            weights = torch.tensor(
                [len(train) / (len(domains) * domains[r["record"]["dataset"]]) for r in subset],
                device=device,
            )
            loss = (per_study * weights).mean()
            loss.backward()
            optimizer.step()
        score = base._assess_levels(model, val, device)
        history.append({"epoch": epoch, "selection_validation": score})
        key = score["macro_hit10"], -score["cross_entropy"]
        if key > best:
            best, state, selected, stale = key, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if stale >= config["localizer_patience"]:
            break
    model.load_state_dict(state)
    return model.eval(), selected, history


def score_weights(rows, weights):
    """Record every requested level, including absent anatomical references."""
    if len(rows) != len(weights):
        raise ValueError("Prediction cohort differs")
    cases = []
    for row, w in zip(rows, weights, strict=True):
        w = np.asarray(w)
        if (
            w.shape != (2, len(row["features"]))
            or not np.isfinite(w).all()
            or np.any(w < 0)
            or not np.allclose(w.sum(-1), 1, atol=1e-5)
        ):
            raise ValueError("Invalid slice probability distribution")
        for j, level in enumerate(base.LEVELS):
            known = bool(row["level_mask"][j])
            top = int(w[j].argmax())
            distance = float(row["anchor_distances_mm"][j, top])
            cases.append(
                {
                    "dataset": row["record"]["dataset"],
                    "study_id": row["record"]["study_id"],
                    "split": row["record"]["split"],
                    "level": level,
                    "reference_present": known,
                    "predicted": True,
                    "selected_index": top,
                    "distance_mm": distance if known and np.isfinite(distance) else None,
                    "hit5": known and distance <= 5,
                    "hit10": known and distance <= 10,
                }
            )
    return cases


def aggregate_cases(cases):
    groups = defaultdict(list)
    for case in cases:
        if case["split"] != "validation":
            raise ValueError("Only validation belongs to the XML ablation report")
        groups[f"{case['dataset']}/{case['level']}"].append(case)
    result = {}
    for key, entries in sorted(groups.items()):
        known = [c for c in entries if c["reference_present"]]
        distances = [c["distance_mm"] for c in known if c["distance_mm"] is not None]
        n = len(known)
        result[key] = {
            "requested": len(entries),
            "predicted": sum(c["predicted"] for c in entries),
            "n": n,
            "missing_reference": len(entries) - n,
            "unverifiable_top_series": sum(c["distance_mm"] is None for c in known),
            "hit5_count": sum(c["hit5"] for c in known),
            "hit10_count": sum(c["hit10"] for c in known),
            "hit5": sum(c["hit5"] for c in known) / n if n else None,
            "hit10": sum(c["hit10"] for c in known) / n if n else None,
            "median_distance_mm_when_verifiable": float(np.median(distances))
            if distances
            else None,
            "p90_distance_mm_when_verifiable": float(np.quantile(distances, 0.9))
            if distances
            else None,
        }
    return {"groups": result}


def paired_bootstrap(xml_cases, mixed_cases, resamples, seed):
    def keyed(cases):
        d = {(c["dataset"], c["study_id"], c["level"]): c for c in cases}
        if len(d) != len(cases):
            raise ValueError("Duplicate patient-level prediction")
        return d

    xml, mixed = keyed(xml_cases), keyed(mixed_cases)
    if xml.keys() != mixed.keys():
        raise ValueError("Paired prediction cohorts differ")
    groups = defaultdict(list)
    for key, left in xml.items():
        right = mixed[key]
        if left["reference_present"] != right["reference_present"]:
            raise ValueError("Paired reference masks differ")
        if left["reference_present"]:
            groups[f"{key[0]}/{key[2]}"].append(int(left["hit10"]) - int(right["hit10"]))
    rng = np.random.default_rng(seed)
    result = {}
    for group, differences in sorted(groups.items()):
        differences = np.asarray(differences, dtype=np.float64)
        # Exactly one level per patient within each reported group. Separate
        # level intervals never treat two levels as independent observations.
        boot = differences[rng.integers(0, len(differences), (resamples, len(differences)))].mean(1)
        result[group] = {
            "n_patients": len(differences),
            "delta_hit10": float(differences.mean()),
            "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
        }
    return {"unit": "patient within each domain/level", "resamples": resamples, "groups": result}


def run_comparison(
    index_path, protocol_path, base_protocol_path, center_checkpoint, out, device="cpu"
):
    index_path, protocol_path, base_protocol_path, center_checkpoint, out = map(
        Path, (index_path, protocol_path, base_protocol_path, center_checkpoint, out)
    )
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use an empty output directory; previous runs are immutable")
    protocol = json.loads(protocol_path.read_text())
    checks = {
        "training_index_sha256": index_path,
        "base_protocol_sha256": base_protocol_path,
        "base_localizer_sha256": center_checkpoint,
        "base_implementation_sha256": Path(base.__file__),
    }
    if any(sha256_file(path) != protocol[key] for key, path in checks.items()):
        raise ValueError("Frozen v6 source/data/checkpoint differs from protocol")
    if protocol["arms"] != {"xml_only": ["aisslab"], "mixed": ["aisslab", "rsna"]}:
        raise ValueError("Unexpected comparison arms")
    if protocol["selection_dataset"] != "aisslab":
        raise ValueError("Only AISSLab validation can select an epoch")
    provenance = {key: sha256_file(path) for key, path in checks.items()}
    provenance.update(
        {
            "protocol_sha256": sha256_file(protocol_path),
            "implementation_sha256": sha256_file(Path(__file__)),
        }
    )
    config = json.loads(base_protocol_path.read_text())
    source, rows = base.read_training_data(index_path)
    old_checkpoint = torch.load(center_checkpoint, map_location="cpu", weights_only=True)
    center_state = old_checkpoint["center_state"]
    write_json(
        out / "manifest.json", {"status": "declared_before_fit", **provenance, "protocol": protocol}
    )
    torch.set_num_threads(4)
    fitted = {}
    for arm, datasets in protocol["arms"].items():
        train, val = partition_rows(rows, datasets)
        fitted[arm] = []
        for seed in protocol["seeds"]:
            model, epoch, history = fit_level(train, val, config, seed, device)
            directory = out / arm / f"seed_{seed}"
            directory.mkdir(parents=True)
            checkpoint = {
                "schema_version": 1,
                "model_family": base.FAMILY,
                "levels": list(base.LEVELS),
                **provenance,
                "training_datasets": datasets,
                "seed": seed,
                "level_epoch": epoch,
                "center_epoch": old_checkpoint["center_epoch"],
                "level_state": {k: v.cpu() for k, v in model.state_dict().items()},
                "center_state": center_state,
            }
            torch.save(checkpoint, directory / "model.pt")
            write_json(directory / "history.json", history)
            selected = {
                "frozen": True,
                "checkpoint_sha256": sha256_file(directory / "model.pt"),
                "selected_epoch": epoch,
                "selection_dataset": "aisslab",
                **provenance,
            }
            write_json(directory / "selection.json", selected)
            fitted[arm].append(
                {
                    "seed": seed,
                    "selected_epoch": epoch,
                    "model": model,
                    "fit_counts": dict(Counter(r["record"]["dataset"] for r in train)),
                    "checkpoint_sha256": selected["checkpoint_sha256"],
                }
            )
            print(json.dumps({"arm": arm, "seed": seed, "sealed_epoch": epoch}), flush=True)
    # All checkpoints are now fixed before any RSNA validation predictions.
    write_json(
        out / "selection.json",
        {
            "status": "all_six_sealed_before_rsna_validation",
            "frozen": True,
            **provenance,
            "checkpoints": {
                arm: [r["checkpoint_sha256"] for r in models] for arm, models in fitted.items()
            },
        },
    )
    val = [r for r in rows if r["record"]["split"] == "validation"]
    cohort = sorted(
        (r["record"]["dataset"], r["record"]["study_id"], level, bool(r["level_mask"][j]))
        for r in val
        for j, level in enumerate(base.LEVELS)
    )
    cohort_hash = hashlib.sha256(json.dumps(cohort, separators=(",", ":")).encode()).hexdigest()
    summary = {
        "schema_version": 1,
        "experiment": protocol["experiment"],
        "phase": "validation",
        "provenance": provenance,
        "validation_cohort_sha256": cohort_hash,
        "original_validation_failures": sum(
            r["split"] == "validation" for r in source.get("excluded_original_failures", [])
        ),
        "center_reused_exactly": True,
        "arms": {},
    }
    private, ensembles = {}, {}
    for arm, models in fitted.items():
        seed_results, all_weights = [], []
        for run in models:
            model = run.pop("model")
            weights = [
                base.predict_slice_weights(model, r["features"], r["positions"]) for r in val
            ]
            all_weights.append(weights)
            cases = score_weights(val, weights)
            prior = copy.deepcopy(model)
            torch.nn.init.zeros_(prior.scorer[-1].weight)
            torch.nn.init.zeros_(prior.scorer[-1].bias)
            prior_weights = [
                base.predict_slice_weights(prior, r["features"], r["positions"]) for r in val
            ]
            seed_results.append(
                {
                    **run,
                    "validation_cohort_sha256": cohort_hash,
                    "validation_levels": aggregate_cases(cases),
                    "positional_baseline": aggregate_cases(score_weights(val, prior_weights)),
                }
            )
            private[f"{arm}/seed_{run['seed']}"] = cases
        ensemble_weights = [np.mean([w[i] for w in all_weights], axis=0) for i in range(len(val))]
        ensembles[arm] = score_weights(val, ensemble_weights)
        summary["arms"][arm] = {
            "training_datasets": protocol["arms"][arm],
            "seeds": seed_results,
            "validation_cohort_sha256": cohort_hash,
            "ensemble": aggregate_cases(ensembles[arm]),
        }
    summary["paired_ensemble_comparison"] = paired_bootstrap(
        ensembles["xml_only"],
        ensembles["mixed"],
        protocol["bootstrap_resamples"],
        protocol["bootstrap_seed"],
    )
    if any(sha256_file(path) != provenance[key] for key, path in checks.items()) or (
        sha256_file(protocol_path) != provenance["protocol_sha256"]
        or sha256_file(Path(__file__)) != provenance["implementation_sha256"]
    ):
        raise RuntimeError("Frozen inputs changed during comparison")
    write_json(out / "cases-private.json", {"seeds": private, "ensembles": ensembles})
    write_json(out / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index", type=Path, default=Path("artifacts/axial-localizer-v6/index.json")
    )
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument(
        "--base-protocol", type=Path, default=Path("configs/axial-localizer-v6.json")
    )
    parser.add_argument(
        "--center-checkpoint",
        type=Path,
        default=Path("reports/runs/explicit-localization-v6/localizer/model.pt"),
    )
    parser.add_argument("--out", type=Path, default=RUN)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args(argv)
    run_comparison(
        args.index, args.protocol, args.base_protocol, args.center_checkpoint, args.out, args.device
    )


if __name__ == "__main__":
    main()
