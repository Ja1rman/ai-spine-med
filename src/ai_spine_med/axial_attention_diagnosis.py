"""Private, post-evaluation explanation of fixed axial checkpoints.

This exports attention and controlled pooling/level-identity perturbations.
Attention is an internal weight, not a confirmed anatomical localization.
No labels or coordinates enter inference, fitting, selection or preprocessing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .assets import sha256_file, write_json
from .axial_transfer import _sealed_models, load_record, read_activation_index

ENTRIES = tuple(
    f"{phase}_seed_{seed}" for phase in ("before", "after") for seed in (42, 2024, 3407)
)


def explain_features(head, features, positions):
    """Return exact predictions and interventions without modifying the head."""
    n = len(features)
    if features.shape != (n, 512) or positions.shape != (n, 2) or n < 1:
        raise ValueError("Require one complete feature bag and matching positions")
    if head.variant != "axial" or head.training:
        raise ValueError("Require a fixed axial head in evaluation mode")
    normalized = (features - head.axial_mean) / head.axial_std
    dummy = features.new_zeros((1, 2, 512))
    mask = torch.ones((1, n), dtype=torch.bool, device=features.device)
    logits, weights = head(dummy, features[None], positions[None], mask, return_attention=True)
    weights = weights[0]
    pooled = weights @ normalized
    one_hot = head.level_one_hot

    def classify(content, identity):
        return head.head(torch.cat((content, identity), dim=-1)).softmax(-1)

    manual = classify(pooled, one_hot)
    probabilities = logits[0].softmax(-1)
    if not torch.allclose(manual, probabilities, rtol=1e-5, atol=1e-6):
        raise RuntimeError("Attention explanation differs from the original head")
    return {
        "features": features,
        "attention": weights,
        "probabilities": probabilities,
        "uniform_attention_probabilities": classify(normalized.mean(0).expand(2, -1), one_hot),
        "swapped_level_probabilities": classify(pooled, one_hot.flip(0)),
        "swapped_attention_probabilities": classify(pooled.flip(0), one_hot),
    }


def _reference_predictions(run_dir):
    result = {}
    for name in ENTRIES:
        with (run_dir / "evaluation" / f"{name}.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        result[name] = {
            (r["dataset"], r["study_id"], r["level"]): np.array(
                [float(r[f"p_{c}"]) for c in ("normal_mild", "moderate", "severe")]
            )
            for r in rows
            if r["status"] == "ok"
        }
    return result


def export_attention(index_path, run_dir, out, *, device="cpu", threads=4):
    """Verify checkpoint/activation seals and match all earlier predictions."""
    index_path, run_dir, out = map(Path, (index_path, run_dir, out))
    torch.set_num_threads(threads)
    records, index_hash = read_activation_index(index_path)
    selection, models = _sealed_models(run_dir, device)
    references = _reference_predictions(run_dir)
    signature = {
        "index_sha256": index_hash,
        "selection_sha256": sha256_file(run_dir / "selection.json"),
        "implementation_sha256": sha256_file(Path(__file__)),
        "entries": list(ENTRIES),
    }
    out.mkdir(parents=True, exist_ok=True)
    config = out / "configuration.json"
    if config.exists() and json.loads(config.read_text()) != signature:
        raise ValueError("Existing explanation has different inputs or implementation")
    write_json(config, signature)
    started, output, max_error = time.monotonic(), [], 0.0
    for number, record in enumerate(records):
        key_hash = hashlib.sha256(json.dumps(record.key).encode()).hexdigest()
        array_path = out / f"{key_hash}.npz"
        receipt_path = out / f"{key_hash}.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if (
                receipt.get("activation_sha256") != record.sha256
                or sha256_file(array_path) != receipt["arrays_sha256"]
            ):
                raise ValueError("Cached explanation changed")
            max_error = max(max_error, receipt["prediction_max_abs_error"])
            output.append(receipt)
            continue
        if record.status != "ok":
            raise ValueError("This diagnostic requires the unchanged fully processed test cohort")
        activations, positions = load_record(record)
        arrays, error = {}, 0.0
        with torch.inference_mode():
            for name in ENTRIES:
                phase, _, seed = name.partition("_seed_")
                model = models[phase, int(seed)]
                target_device = next(model.parameters()).device
                a, p = activations.to(target_device), positions.to(target_device)
                features = F.adaptive_avg_pool2d(model.block(a), 1).flatten(1)
                explanation = explain_features(model.head, features, p)
                for field, tensor in explanation.items():
                    arrays.setdefault(field, []).append(tensor.cpu().numpy().astype(np.float32))
                expected = np.stack(
                    [references[name][(*record.key, level)] for level in ("L4-L5", "L5-S1")]
                )
                delta = float(np.max(np.abs(arrays["probabilities"][-1] - expected)))
                if delta > 1e-4:
                    raise RuntimeError("Diagnostic predictions differ from sealed evaluation")
                error = max(error, delta)
        values = {field: np.stack(items) for field, items in arrays.items()}
        values["positions"] = positions.numpy()
        if not all(np.isfinite(v).all() for v in values.values()):
            raise RuntimeError("Nonfinite explanation output")
        if not np.allclose(values["attention"].sum(-1), 1, atol=1e-6, rtol=0):
            raise RuntimeError("Invalid attention normalization")
        temporary = array_path.with_suffix(".npz.part")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **values)
        temporary.replace(array_path)
        receipt = {
            "dataset": record.dataset,
            "study_id": record.study_id,
            "activation_sha256": record.sha256,
            "arrays_path": array_path.name,
            "arrays_sha256": sha256_file(array_path),
            "n_slices": len(positions),
            "prediction_max_abs_error": error,
        }
        write_json(receipt_path, receipt)
        output.append(receipt)
        max_error = max(max_error, error)
        if (number + 1) % 20 == 0 or number + 1 == len(records):
            print(
                json.dumps(
                    {
                        "completed": number + 1,
                        "total": len(records),
                        "seconds": round(time.monotonic() - started, 1),
                    }
                ),
                flush=True,
            )
    if sha256_file(index_path) != index_hash:
        raise RuntimeError("Activation index changed during diagnosis")
    write_json(out / "index.json", {"schema_version": 1, **signature, "records": output})
    summary = {
        "n_studies": len(output),
        "n_checkpoints": len(ENTRIES),
        "prediction_max_abs_error": max_error,
        "training_performed": False,
        "labels_or_coordinates_used_in_inference": False,
        "seconds": time.monotonic() - started,
    }
    write_json(out / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index", type=Path, default=Path("artifacts/axial-transfer-test-v4/index.json")
    )
    parser.add_argument("--run", type=Path, default=Path("reports/runs/axial-transfer-v4"))
    parser.add_argument(
        "--out", type=Path, default=Path("reports/runs/axial-diagnosis-v5/attention")
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            export_attention(
                args.index, args.run, args.out, device=args.device, threads=args.threads
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
