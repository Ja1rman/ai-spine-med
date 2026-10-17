"""Seal completed model selection and compare all fixed variants on held-out MRI.

The seal is a reproducibility guard, not a digital signature. Test predictions
are never fed to a trainer or used to choose a seed, epoch, or threshold.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .assets import locked_assets, write_json
from .data import CLASS_NAMES
from .training import (
    FEATURE_SIZE,
    LEVELS,
    SEEDS,
    FeatureData,
    collapse_stenosis_probabilities,
    evaluate_predictions,
    load_feature_data,
    predict_head,
)

FINE_MODEL = "SpineNetLayer4LastBlock+Linear(512,3)"
ACTIVATION_SHAPE = (512, 9, 7, 14)
SOURCE_FIELDS = (
    "upstream_commit",
    "source_archive_sha256",
    "source_python_sha256",
    "weights_sha256",
)
PREPROCESSING_FIELDS = ("features.py", "dicom.py", "isotropic_inplane")


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _key(row: dict[str, Any]) -> tuple[str, str, str]:
    return tuple(str(row[name]) for name in ("study_id", "series_id", "level"))


def _files(paths: list[Path]) -> list[dict[str, str]]:
    return [{"path": str(path.resolve()), "sha256": _sha(path)} for path in paths]


def _implementation_paths() -> dict[str, Path]:
    return {
        name: Path(__file__).with_name(name)
        for name in ("evaluation.py", "training.py", "finetuning.py", "heldout_features.py")
    }


def _completed_runs(directory: Path, variant: str) -> tuple[list[dict], dict, list[dict]]:
    """Require a completed three-seed summary and its actual selected checkpoints."""
    import torch

    manifest = _json(directory / "manifest.json")
    summary = _json(directory / "summary.json")
    runs = summary.get("runs", [])
    config = summary.get("configuration", {})
    if (
        not isinstance(runs, list)
        or len(runs) != len(SEEDS)
        or [run.get("seed") for run in runs] != list(SEEDS)
        or config.get("seeds") != list(SEEDS)
        or summary.get("test_evaluated") is not False
        or manifest.get("configuration") != config
    ):
        raise ValueError(f"{variant} needs all three completed, preselected validation runs")
    if summary.get("all_seeds_reported") is not True:
        raise ValueError(f"{variant} does not report every predetermined seed")
    paths = [directory / "manifest.json", directory / "summary.json"]
    records = []
    for run in runs:
        seed = run["seed"]
        run_dir = directory / "runs" / f"seed-{seed}"
        filename = "head.pt" if variant == "frozen_head" else "last-block.pt"
        checkpoint_path = run_dir / filename
        checkpoint_sha = _sha(checkpoint_path)
        if checkpoint_sha != run.get("checkpoint_sha256"):
            raise ValueError(f"Selected {variant} checkpoint checksum mismatch")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model_name = "Linear(512,3)" if variant == "frozen_head" else FINE_MODEL
        best_epoch = run.get("best_epoch")
        minimum_epoch = 1 if variant == "frozen_head" else 0
        if (
            checkpoint.get("model") != model_name
            or checkpoint.get("class_names") != list(CLASS_NAMES)
            or checkpoint.get("seed") != seed
            or type(best_epoch) is not int
            or best_epoch < minimum_epoch
            or checkpoint.get("best_epoch") != best_epoch
        ):
            raise ValueError(
                "Selected checkpoint model, class order, seed or epoch is inconsistent"
            )
        state = checkpoint.get("state_dict", {})
        head_prefix = "" if variant == "frozen_head" else "head."
        if (
            state.get(head_prefix + "weight") is None
            or state.get(head_prefix + "bias") is None
            or tuple(state[head_prefix + "weight"].shape) != (3, FEATURE_SIZE)
            or tuple(state[head_prefix + "bias"].shape) != (3,)
        ):
            raise ValueError("Selected checkpoint has an incompatible classifier")
        if variant == "fine_tuned" and checkpoint.get("input_shape") != list(ACTIVATION_SHAPE):
            raise ValueError("Selected last-block checkpoint uses a different activation shape")
        history_path = run_dir / "history.json"
        history = json.loads(history_path.read_text())
        if not isinstance(history, list) or not history:
            raise ValueError("Selected checkpoint has no completed validation history")
        epochs = [row.get("epoch") for row in history]
        expected_epochs = list(range(minimum_epoch, minimum_epoch + len(history)))
        if any(type(epoch) is not int for epoch in epochs) or epochs != expected_epochs:
            raise ValueError("Validation history epochs must be unique, consecutive and ordered")
        scores = [row["validation_macro_f1"] for row in history]
        if any(
            not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1
            for score in scores
        ):
            raise ValueError("Validation history contains an invalid selection score")
        best_score = max(scores)
        first_epoch = next(
            row["epoch"] for row in history if row["validation_macro_f1"] == best_score
        )
        if first_epoch != best_epoch or abs(run["validation_macro_f1"] - best_score) > 1e-12:
            raise ValueError("Selected checkpoint is not the earliest best validation epoch")
        if variant == "fine_tuned" and history[0].get("epoch") != 0:
            raise ValueError("Fine-tuning must include the initial warm-start as epoch zero")
        parity = run.get("initialization_parity") if variant == "fine_tuned" else None
        if variant == "fine_tuned":
            if (
                not isinstance(parity, dict)
                or type(parity.get("n_prediction_mismatches")) is not int
                or parity["n_prediction_mismatches"] != 0
                or parity.get("atol") != 1e-4
                or not isinstance(parity.get("max_probability_abs_error"), (int, float))
                or not 0 <= parity["max_probability_abs_error"] <= 1e-4
                or not isinstance(parity.get("epoch0_macro_f1"), (int, float))
                or not math.isfinite(parity["epoch0_macro_f1"])
                or abs(parity["epoch0_macro_f1"] - history[0]["validation_macro_f1"]) > 1e-12
            ):
                raise ValueError(
                    "Fine-tuning needs a successful numerical initialization parity audit"
                )
        metrics_path = run_dir / "metrics.json"
        metrics = _json(metrics_path)
        if metrics != run.get("metrics"):
            raise ValueError("Completed summary disagrees with its selected validation metrics")
        if abs(metrics["conditional"]["macro_f1"] - best_score) > 1e-12:
            raise ValueError("Selected validation metrics disagree with checkpoint selection")
        paths.extend((checkpoint_path, history_path, metrics_path))
        records.append(
            {
                "seed": seed,
                "path": str(checkpoint_path.resolve()),
                "sha256": checkpoint_sha,
                "best_epoch": best_epoch,
                "validation_macro_f1": best_score,
                "initial_validation_macro_f1": history[0]["validation_macro_f1"],
                "initialization_parity": parity,
                "initial_head_sha256": checkpoint.get("initial_head_sha256"),
                "feature_cache_sha256": checkpoint.get("feature_cache_sha256"),
                "original_weights_sha256": checkpoint.get("original_weights_sha256"),
                "protocol_sha256": checkpoint.get("protocol_sha256"),
                "batchnorm_running_statistics_frozen": checkpoint.get(
                    "batchnorm_running_statistics_frozen"
                ),
                "earlier_encoder_updated": checkpoint.get("earlier_encoder_updated"),
            }
        )
    return records, manifest, _files(paths)


def freeze_selection(
    protocol_path: str | Path,
    manifest_dir: str | Path,
    feature_dir: str | Path,
    head_dir: str | Path,
    finetune_dir: str | Path,
    out_file: str | Path,
) -> dict:
    """Exclusively create the seal after all six validation-selected models exist."""
    protocol_path, manifest_dir, feature_dir, head_dir, finetune_dir, out_file = map(
        Path, (protocol_path, manifest_dir, feature_dir, head_dir, finetune_dir, out_file)
    )
    if out_file.exists():
        raise ValueError("A selection seal already exists; it must never be overwritten")
    protocol = _json(protocol_path)
    data = load_feature_data(feature_dir)
    if (
        protocol.get("seeds") != list(SEEDS)
        or protocol.get("classes") != list(CLASS_NAMES)
        or protocol.get("levels") != list(LEVELS)
        or protocol.get("source_feature_cache_id") != data.metadata.get("cache_id")
        or protocol.get("heldout_test", {}).get("no_test_tuning") is not True
    ):
        raise ValueError("Protocol does not match the fixed class, seed or training feature cohort")
    manifests = {
        name: _sha(manifest_dir / name) for name in ("studies.csv", "series.csv", "labels.csv")
    }
    cohort = _json(manifest_dir / "summary.json")
    if any(cohort["output_sha256"][name] != digest for name, digest in manifests.items()):
        raise ValueError("Fixed cohort manifest checksum mismatch")
    if protocol.get("study_manifest_sha256") != manifests["studies.csv"]:
        raise ValueError("Protocol uses a different patient grouping")
    source_config = data.metadata.get("config", {})
    if source_config.get("manifests") != manifests:
        raise ValueError("Training features and held-out grouping refer to different manifests")
    original_weights = {
        name: record["sha256"] for name, record in locked_assets()["weights"].items()
    }
    if source_config.get("weights_sha256") != original_weights:
        raise ValueError("Training features do not use the reviewed original SpineNet weights")
    frozen, frozen_manifest, frozen_files = _completed_runs(head_dir, "frozen_head")
    fine, fine_manifest, fine_files = _completed_runs(finetune_dir, "fine_tuned")
    for manifest in (frozen_manifest, fine_manifest):
        if manifest.get("feature_cache_sha256") != data.file_sha256:
            raise ValueError(
                "All selected models must share the identical training/validation cache"
            )
    for record in frozen + fine:
        if record["feature_cache_sha256"] != data.file_sha256:
            raise ValueError("A selected checkpoint belongs to a different training feature cache")
    for frozen_record, fine_record in zip(frozen, fine):
        if fine_record["initial_head_sha256"] != frozen_record["sha256"]:
            raise ValueError("Fine-tuning must initialize from the matching seed's frozen head")
        if fine_record["original_weights_sha256"] != original_weights:
            raise ValueError("Fine-tuning does not retain the reviewed original prefix weights")
        if (
            fine_record["protocol_sha256"] != _sha(protocol_path)
            or fine_record["batchnorm_running_statistics_frozen"] is not True
            or fine_record["earlier_encoder_updated"] is not False
        ):
            raise ValueError(
                "Fine-tuned checkpoint does not certify the specified protocol and frozen prefix/BN statistics"
            )
        if (
            abs(fine_record["initial_validation_macro_f1"] - frozen_record["validation_macro_f1"])
            > 1e-12
        ):
            raise ValueError(
                "Fine-tuning epoch zero does not reproduce its matching frozen-head validation score"
            )
        if (
            fine_record["initialization_parity"].get("frozen_head_macro_f1")
            != frozen_record["validation_macro_f1"]
        ):
            raise ValueError("Initialization parity refers to a different frozen-head result")
    fine_config = fine_manifest["configuration"]
    if fine_config.get("protocol_sha256") != _sha(protocol_path):
        raise ValueError("Fine-tuning ran under a different written protocol")
    for name in (
        "optimizer",
        "learning_rate_block",
        "learning_rate_head",
        "weight_decay",
        "batch_size",
        "max_epochs",
        "patience",
    ):
        if fine_config.get(name) != protocol.get(name):
            raise ValueError(f"Fine-tuning configuration differs from the sealed protocol: {name}")
    implementations = _implementation_paths()
    for manifest, name in ((frozen_manifest, "training.py"), (fine_manifest, "finetuning.py")):
        declared = manifest["configuration"].get("implementation_sha256")
        if declared != _sha(implementations[name]):
            raise ValueError(f"Training source changed after its completed run: {name}")
    seal = {
        "schema_version": 1,
        "scope": "heldout_test",
        "frozen": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_dir": str(manifest_dir.resolve()),
        "manifests_sha256": manifests,
        "protocol": {"path": str(protocol_path.resolve()), "sha256": _sha(protocol_path)},
        "original_weights_sha256": original_weights,
        "checkpoints": {"frozen_head": frozen, "fine_tuned": fine},
        "trainval_feature_dir": str(feature_dir.resolve()),
        "trainval_feature_cache_sha256": data.file_sha256,
        "trainval_cache_id": data.metadata["cache_id"],
        "trainval_config": source_config,
        "training_configurations": {
            "frozen_head": frozen_manifest["configuration"],
            "fine_tuned": fine_config,
        },
        "training_artifacts": frozen_files + fine_files,
        "source_files": _files(list(implementations.values())),
        "test_policy": "All seven models, all three predetermined seeds; no training, selection or threshold tuning on test",
    }
    seal["seal_sha256"] = hashlib.sha256(_canonical(seal)).hexdigest()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("x", encoding="utf-8") as handle:
        handle.write(_canonical(seal).decode())
    return seal


def validate_selection(selection_file: str | Path) -> tuple[dict, str]:
    """Verify the seal before any held-out arrays or ROI pixels are opened."""
    from .rsna_bulk import validate_test_selection

    selection_file = Path(selection_file)
    selection = _json(selection_file)
    content = dict(selection)
    declared = content.pop("seal_sha256", None)
    if declared != hashlib.sha256(_canonical(content)).hexdigest():
        raise ValueError("Selection seal content changed")
    selection, file_sha = validate_test_selection(selection["manifest_dir"], selection_file)
    for item in selection.get("source_files", []) + selection.get("training_artifacts", []):
        if _sha(item["path"]) != item["sha256"]:
            raise ValueError("A sealed implementation or completed training artifact changed")
    return selection, file_sha


def _expected_test_targets(selection: dict) -> dict[tuple[str, str, str], int | None]:
    root = Path(selection["manifest_dir"])
    labels = {
        (row["study_id"], row["level"]): int(row["label"])
        for row in _csv(root / "labels.csv")
        if row["split"] == "test" and row["level"] in LEVELS
    }
    expected = {}
    studies = set()
    for row in _csv(root / "series.csv"):
        if row["split"] != "test":
            continue
        if row["series_description"] != "Sagittal T2/STIR" or row["study_id"] in studies:
            raise ValueError("Held-out cohort needs exactly one sagittal T2-like series per study")
        studies.add(row["study_id"])
        for level in LEVELS:
            expected[(row["study_id"], row["series_id"], level)] = labels.get(
                (row["study_id"], level)
            )
    if not expected:
        raise ValueError("No held-out target levels in the fixed cohort")
    return expected


def load_test_feature_data(feature_dir: str | Path, selection_file: str | Path) -> FeatureData:
    """Own strict test-only reader; the train/validation reader remains unchanged."""
    selection, selection_sha = validate_selection(selection_file)
    root = Path(feature_dir)
    metadata = _json(root / "metadata.json")
    config = metadata.get("config", {})
    if (
        metadata.get("extraction_complete") is not True
        or config.get("scope") != "heldout_test"
        or config.get("allowed_splits") != ["test"]
        or config.get("selection_sha256") != selection_sha
        or config.get("weights_sha256") != selection["original_weights_sha256"]
        or config.get("manifests") != selection["manifests_sha256"]
    ):
        raise ValueError("Test cache is incomplete or belongs to a different sealed selection")
    frozen_config = selection.get("trainval_config", {})
    if any(
        name not in config or name not in frozen_config or config[name] != frozen_config[name]
        for name in SOURCE_FIELDS
    ) or any(
        config.get("implementation_sha256", {}).get(name) is None
        or config["implementation_sha256"][name]
        != frozen_config.get("implementation_sha256", {}).get(name)
        for name in PREPROCESSING_FIELDS
    ):
        raise ValueError("Held-out source or preprocessing differs from the sealed training cache")
    hashes = {name: _sha(root / name) for name in ("features.npz", "target_manifest.csv")}
    if metadata.get("outputs_sha256") != hashes:
        raise ValueError("Held-out cache output checksum mismatch")
    hashes["metadata.json"] = _sha(root / "metadata.json")
    required = ("features", "original_logits", "labels", "study_id", "series_id", "level", "split")
    with np.load(root / "features.npz", allow_pickle=False) as archive:
        if not set(required).issubset(archive.files):
            raise ValueError("Held-out cache is missing required arrays")
        arrays = {name: archive[name].copy() for name in required}
    count = len(arrays["features"])
    if arrays["features"].shape != (count, FEATURE_SIZE) or arrays["features"].dtype != np.float32:
        raise ValueError("Held-out features must be float32[N,512]")
    if (
        arrays["original_logits"].shape != (count, 4)
        or not np.isfinite(arrays["original_logits"]).all()
    ):
        raise ValueError("Held-out original logits must be finite[N,4]")
    if not np.isfinite(arrays["features"]).all():
        raise ValueError("Held-out features contain nonfinite values")
    if (
        arrays["labels"].shape != (count,)
        or arrays["labels"].dtype.kind not in "iu"
        or not np.isin(arrays["labels"], [0, 1, 2]).all()
    ):
        raise ValueError("Held-out feature labels must be integer categories 0..2")
    for name in ("study_id", "series_id", "level", "split"):
        if arrays[name].shape != (count,) or arrays[name].dtype.kind != "U":
            raise ValueError("Held-out identifiers must be Unicode vectors aligned with features")
    if not set(arrays["split"].tolist()).issubset({"test"}):
        raise ValueError("Held-out evaluation accepts test features only")
    feature_keys = [
        tuple(str(arrays[name][i]) for name in ("study_id", "series_id", "level"))
        for i in range(count)
    ]
    if len(set(feature_keys)) != len(feature_keys):
        raise ValueError("Duplicate held-out feature identifier")
    feature_by_key = dict(zip(feature_keys, range(count)))
    expected = _expected_test_targets(selection)
    targets, seen, matched = [], set(), set()
    for raw in _csv(root / "target_manifest.csv"):
        row = dict(raw)
        key = _key(row)
        if row.get("split") != "test" or key in seen or key not in expected:
            raise ValueError("Held-out target identifiers or splits do not match the sealed cohort")
        seen.add(key)
        try:
            row["label"] = None if row["label"] == "" else int(row["label"])
        except (ValueError, TypeError):
            raise ValueError("Invalid held-out reference category") from None
        if row["label"] != expected[key]:
            raise ValueError("Held-out target label disagrees with the sealed reference")
        if not row.get("status") or row["status"] in {"pending", "not_downloaded", "not_processed"}:
            raise ValueError("Held-out extraction still has unfinished targets")
        index = feature_by_key.get(key)
        if row["status"] == "ok":
            if (
                index is None
                or row["label"] is None
                or int(arrays["labels"][index]) != row["label"]
            ):
                raise ValueError("Held-out feature/reference key alignment failed")
            matched.add(key)
        elif index is not None:
            raise ValueError("An unassessed held-out target unexpectedly has features")
        row["feature_index"] = index
        targets.append(row)
    if seen != set(expected) or matched != set(feature_keys):
        raise ValueError("Held-out cache omits requested targets or contains unmatched features")
    return FeatureData(arrays, targets, metadata, hashes)


def _load_test_activations(directory: Path, data: FeatureData, selection_sha: str) -> np.ndarray:
    metadata = _json(directory / "metadata.json")
    if (
        metadata.get("scope") != "heldout_inference"
        or metadata.get("selection_sha256") != selection_sha
        or metadata.get("extraction_complete") is not True
        or metadata.get("feature_cache_sha256") != data.file_sha256
    ):
        raise ValueError("Last-block activations do not belong to this sealed test cache")
    declared = metadata.get("outputs_sha256", {})
    if (
        not isinstance(declared, dict)
        or not {"activations.npy", "index.npz"}.issubset(declared)
        or set(declared) - {"activations.npy", "index.npz", "completed.npy", "row_sha256.npy"}
    ):
        raise ValueError("Held-out activation checksums are missing or unexpected")
    actual = {name: _sha(directory / name) for name in declared}
    if declared != actual:
        raise ValueError("Held-out activation cache checksum mismatch")
    activations = np.load(directory / "activations.npy", mmap_mode="r", allow_pickle=False)
    count = len(data.arrays["features"])
    if activations.shape != (count, *ACTIVATION_SHAPE) or activations.dtype != np.float32:
        raise ValueError("Expected held-out activations float32[N,512,9,7,14]")
    with np.load(directory / "index.npz", allow_pickle=False) as index:
        for name in ("study_id", "series_id", "level", "split", "labels"):
            if name not in index.files or not np.array_equal(index[name], data.arrays[name]):
                raise ValueError("Held-out activation keys, labels or order do not match features")
        if "feature_index" not in index.files or not np.array_equal(
            index["feature_index"], np.arange(count)
        ):
            raise ValueError("Held-out activation feature indices are inconsistent")
    for offset in range(0, count, 8):
        if not np.isfinite(activations[offset : offset + 8]).all():
            raise ValueError("Held-out activations contain nonfinite values")
    return activations


def _test_rows(data: FeatureData, probabilities: np.ndarray) -> list[dict]:
    expected_shape = (len(data.arrays["features"]), 3)
    if (
        probabilities.shape != expected_shape
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    ):
        raise ValueError("A fixed model returned invalid held-out category probabilities")
    rows = []
    for target in data.targets:
        row = {
            name: target[name]
            for name in ("study_id", "series_id", "level", "split", "label", "status", "reason")
        }
        row.update(prediction=None, p_normal_mild=None, p_moderate=None, p_severe=None)
        index = target["feature_index"]
        if index is not None:
            row["prediction"] = int(probabilities[index].argmax())
            row.update(
                {
                    f"p_{name}": float(value)
                    for name, value in zip(CLASS_NAMES, probabilities[index])
                }
            )
        rows.append(row)
    return rows


def _test_metrics(rows: list[dict]) -> dict:
    metrics = evaluate_predictions(rows)
    metrics["scope"] = (
        "Held-out test; all seven fixed models use the identical automatically assessed ROIs. No selection or training on test."
    )
    metrics["by_status"] = dict(Counter(row["status"] for row in rows if row["label"] is not None))
    metrics["not_assessed_by_reason"] = dict(
        Counter(
            row["reason"] or row["status"]
            for row in rows
            if row["label"] is not None and row["status"] != "ok"
        )
    )
    for group in [metrics, *metrics["by_level"].values()]:
        severe = group["severe"]
        severe["missed_all_requested"] = severe["n_total"] - severe["correctly_predicted_severe"]
    return metrics


def _prediction_csv(path: Path, rows: list[dict]) -> None:
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
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def _variant_stats(entries: list[dict]) -> dict:
    output = {}
    for variant in ("frozen_head", "fine_tuned"):
        subset = [entry for entry in entries if entry["variant"] == variant]
        if [entry["seed"] for entry in subset] != list(SEEDS):
            raise ValueError("Every predetermined seed must be reported")
        values = {
            "macro_f1": [entry["metrics"]["conditional"]["macro_f1"] for entry in subset],
            "severe_detection_rate": [
                entry["metrics"]["severe"]["detection_rate_all_requested"] for entry in subset
            ],
        }
        result = {
            "seeds": list(SEEDS),
            "std_definition": "Sample standard deviation across all three fixed seeds; not a confidence interval",
        }
        for name, samples in values.items():
            result[name + "_mean"] = (
                float(np.mean(samples)) if all(value is not None for value in samples) else None
            )
            result[name + "_std"] = (
                float(np.std(samples, ddof=1))
                if all(value is not None for value in samples)
                else None
            )
        output[variant] = result
    return output


def _synchronize(device: str) -> None:
    import torch

    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize(device)


def _device(requested: str) -> str:
    import torch

    if requested == "auto":
        requested = (
            "cuda:0"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; choose a supported device before held-out evaluation")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if requested not in {"cpu", "mps"} and not requested.startswith("cuda"):
        raise ValueError("Expected cpu, mps, cuda[:N] or auto")
    return requested


def _frozen_probabilities(
    features: np.ndarray, checkpoint_path: Path, device: str, batch_size: int
) -> np.ndarray:
    import torch
    from torch import nn

    if device == "cpu":
        return predict_head(features, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model") != "Linear(512,3)" or checkpoint.get("class_names") != list(
        CLASS_NAMES
    ):
        raise ValueError("Invalid sealed frozen classifier")
    head = nn.Linear(FEATURE_SIZE, 3).to(device).eval()
    head.load_state_dict(checkpoint["state_dict"], strict=True)
    batches = []
    with torch.inference_mode():
        for offset in range(0, len(features), batch_size):
            tensor = torch.from_numpy(features[offset : offset + batch_size]).to(device)
            batches.append(head(tensor).softmax(dim=1).cpu().numpy())
    return np.concatenate(batches) if batches else np.empty((0, 3), np.float32)


def _training_rois(selection: dict, maximum: int = 8) -> np.ndarray:
    """Timing samples come only from the already frozen training cache."""
    root = Path(selection["trainval_feature_dir"])
    data = load_feature_data(root)
    if data.file_sha256 != selection["trainval_feature_cache_sha256"]:
        raise ValueError("Training timing cache changed after selection was sealed")
    rois = []
    checked = {}
    for target in data.targets:
        if target["split"] != "train" or target["status"] != "ok":
            continue
        path = (root / target["roi_path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Timing ROI path escapes the training cache")
        if path not in checked:
            record = _json(root / "records" / f"{target['study_id']}_{target['series_id']}.json")
            if record["cache_id"] != selection["trainval_cache_id"] or record[
                "arrays_sha256"
            ] != _sha(path):
                raise ValueError("A training timing ROI cache is inconsistent")
            checked[path] = True
        with np.load(path, allow_pickle=False) as archive:
            index = int(target["roi_index"])
            if str(archive["levels"][index]) != target["level"]:
                raise ValueError("Training timing ROI/level alignment failed")
            roi = archive["rois"][index].astype(np.float32)
        if roi.shape != (9, 112, 224) or not np.isfinite(roi).all():
            raise ValueError("Training timing ROI has invalid dimensions or values")
        rois.append(roi)
        if len(rois) == maximum:
            break
    if not rois:
        raise ValueError("No previously processed training ROI is available for timing")
    return np.stack(rois)


def benchmark_classifiers(
    selection: dict, cache_dir: str | Path, device: str, *, warmup: int = 8, draws: int = 32
) -> dict:
    """Benchmark complete grading forward passes, not cached suffix inference."""
    import copy

    import torch
    from torch import nn

    from .finetuning import _load_original_grading, load_last_block_head

    if warmup < 1 or not 8 <= draws <= 256:
        raise ValueError("Use a positive warm-up and 8..256 timed classifier draws")
    device = _device(device)
    rois = _training_rois(selection)
    inputs = [torch.from_numpy(roi)[None, None].to(device) for roi in rois]
    original = _load_original_grading(Path(cache_dir), device=device).eval()
    models = [("original", None, None)]
    models.extend(
        (variant, record["seed"], Path(record["path"]))
        for variant in ("frozen_head", "fine_tuned")
        for record in selection["checkpoints"][variant]
    )
    results = []
    for variant, seed, checkpoint_path in models:
        model = copy.deepcopy(original)
        if variant == "frozen_head":
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model.fc_ccs = nn.Linear(FEATURE_SIZE, 3).to(device)
            model.fc_ccs.load_state_dict(checkpoint["state_dict"], strict=True)
        elif variant == "fine_tuned":
            suffix = load_last_block_head(checkpoint_path, cache_dir, device=device)
            model.layer4[-1] = suffix.block
            model.fc_ccs = suffix.head
        model.eval().requires_grad_(False)
        elapsed = []
        with torch.inference_mode():
            for index in range(warmup):
                model(inputs[index % len(inputs)])
                _synchronize(device)
            for index in range(draws):
                _synchronize(device)
                started = time.perf_counter()
                model(inputs[index % len(inputs)])
                _synchronize(device)
                elapsed.append(time.perf_counter() - started)
        results.append(
            {
                "variant": variant,
                "seed": seed,
                "median_seconds": float(np.median(elapsed)),
                "p90_seconds": float(np.percentile(elapsed, 90)),
                "draws": draws,
                "warmup": warmup,
                "batch_size": 1,
            }
        )
        del model
    return {
        "scope": "Complete grading network forward, including the unchanged prefix and selected head/block. Identical cached training ROI samples; model load, disk IO, host-to-device transfer, detector and ROI preparation excluded.",
        "hardware": {
            "device": device,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "torch": str(torch.__version__),
            "threads": torch.get_num_threads(),
            "synchronized_each_draw": True,
        },
        "roi_sample_split": "train",
        "n_roi_samples": len(rois),
        "roi_samples_sha256": hashlib.sha256(rois.tobytes()).hexdigest(),
        "classifier_batch1": results,
    }


def _preprocessing_timings(feature_dir: Path, data: FeatureData, selection_sha: str) -> dict:
    """Use optional extraction timings only when their full aggregate is current."""
    path = feature_dir / "summary.json"
    unavailable = {
        "available": False,
        "reason": "Missing or stale held-out extraction summary; common preprocessing time is not reported",
    }
    if not path.is_file():
        return unavailable
    try:
        summary = _json(path)
        labeled = [row for row in data.targets if row["label"] is not None]
        series_count = len({(row["study_id"], row["series_id"]) for row in data.targets})
        coverage = summary.get("coverage", {}).get("test", {})
        if (
            summary.get("scope") != "heldout_test"
            or summary.get("cache_id") != data.metadata["cache_id"]
            or summary.get("selection_sha256") != selection_sha
            or summary.get("extraction_complete") is not True
            or summary.get("n_features") != len(data.arrays["features"])
            or summary.get("n_requested_series") != series_count
            or coverage.get("requested_labeled_targets") != len(labeled)
            or coverage.get("assessed") != len(data.arrays["features"])
        ):
            return unavailable
        aggregates = summary.get("timing_aggregates", {})
        expected = {
            "full_original_seconds",
            "measured_grading_forward_seconds",
            "common_DICOM_localization_ROI_plus_tensorIO_seconds",
            "dicom_hash_seconds",
        }
        if set(aggregates) != expected:
            return unavailable
        for values in aggregates.values():
            count = values.get("count")
            if type(count) is not int or not 0 <= count <= series_count:
                return unavailable
            for name in ("median_seconds", "total_seconds"):
                value = values.get(name)
                if value is None and count == 0 and name == "median_seconds":
                    continue
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    return unavailable
        return {"available": True, **aggregates, "scope": summary.get("timing_definition", "")}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return unavailable


def evaluate_test(
    feature_dir: str | Path,
    selection_file: str | Path,
    head_dir: str | Path,
    finetune_dir: str | Path,
    activation_dir: str | Path,
    out: str | Path,
    *,
    cache_dir: str | Path = ".cache",
    device: str = "mps",
    batch_size: int = 16,
    benchmark: bool = True,
    warmup: int = 8,
    draws: int = 32,
) -> dict:
    """Compare original + three frozen + three fine-tuned models without selection."""
    if batch_size < 1:
        raise ValueError("Inference batch size must be positive")
    device = _device(device)
    out, feature_dir, activation_dir = map(Path, (out, feature_dir, activation_dir))
    if out.exists():
        raise ValueError("Held-out output already exists; results must not be overwritten")
    selection, selection_sha = validate_selection(selection_file)
    for variant, directory, filename in (
        ("frozen_head", Path(head_dir), "head.pt"),
        ("fine_tuned", Path(finetune_dir), "last-block.pt"),
    ):
        for record in selection["checkpoints"][variant]:
            supplied = (directory / "runs" / f"seed-{record['seed']}" / filename).resolve()
            if supplied != Path(record["path"]).resolve() or _sha(supplied) != record["sha256"]:
                raise ValueError("Evaluation model directory differs from the sealed checkpoints")
    data = load_test_feature_data(feature_dir, selection_file)
    count = len(data.arrays["features"])
    if count:
        from .finetuning import cache_inference_activations, predict_last_block

        if (
            not (activation_dir / "metadata.json").is_file()
            or _json(activation_dir / "metadata.json").get("extraction_complete") is not True
        ):
            cache_inference_activations(
                data,
                feature_dir,
                cache_dir,
                activation_dir,
                selection_sha256=selection_sha,
                device=device,
                batch_size=batch_size,
            )
        activations = _load_test_activations(activation_dir, data, selection_sha)
    else:
        activations = None
    predictions = []
    probabilities = collapse_stenosis_probabilities(data.arrays["original_logits"])
    predictions.append(("original", None, None, _test_rows(data, probabilities)))
    for variant in ("frozen_head", "fine_tuned"):
        for record in selection["checkpoints"][variant]:
            print(f"Held-out inference: {variant} seed={record['seed']}", flush=True)
            if not count:
                probabilities = np.empty((0, 3), np.float32)
            elif variant == "frozen_head":
                probabilities = _frozen_probabilities(
                    data.arrays["features"], Path(record["path"]), device, batch_size
                )
            else:
                probabilities = predict_last_block(
                    activations, record["path"], cache_dir, device=device, batch_size=batch_size
                )
            predictions.append(
                (variant, record["seed"], record["sha256"], _test_rows(data, probabilities))
            )
    entries = [
        {
            "variant": variant,
            "seed": seed,
            "checkpoint_sha256": digest,
            "metrics": _test_metrics(rows),
        }
        for variant, seed, digest, rows in predictions
    ]
    common = entries[0]["metrics"]
    for entry in entries[1:]:
        if any(
            entry["metrics"][name] != common[name]
            for name in (
                "n_requested",
                "n_labeled",
                "n_assessed",
                "n_not_assessed",
                "coverage",
                "class_counts",
                "assessed_class_counts",
            )
        ):
            raise ValueError("Model comparison does not use the identical held-out ROI subset")
    timings = (
        benchmark_classifiers(selection, cache_dir, device, warmup=warmup, draws=draws)
        if benchmark
        else {
            "scope": "Classifier timing was explicitly disabled; no inference-speed comparison is reported",
            "classifier_batch1": [],
        }
    )
    timings["common_preprocessing"] = _preprocessing_timings(feature_dir, data, selection_sha)
    timings["common_preprocessing_scope"] = (
        "Held-out extraction separately measures common DICOM/localization/ROI and tensor IO; cache construction and model loading are not ordinary classifier inference. Residual timing is not a pure detector benchmark."
    )
    # Recheck after the complete forward pass so a concurrent checkpoint/code
    # replacement cannot silently produce a mixed sealed comparison.
    validate_selection(selection_file)
    out.mkdir(parents=True)
    for variant, seed, digest, rows in predictions:
        directory = out / "entries" / variant
        if seed is not None:
            directory /= f"seed-{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        _prediction_csv(directory / "predictions.csv", rows)
        write_json(directory / "metrics.json", _test_metrics(rows))
    summary = {
        "schema_version": 1,
        "scope": "test",
        "entries": entries,
        "variants": _variant_stats(entries),
        "coverage": {
            name: common[name]
            for name in (
                "n_requested",
                "n_labeled",
                "n_assessed",
                "n_not_assessed",
                "coverage",
                "by_status",
                "not_assessed_by_reason",
            )
        },
        "timings": timings,
        "selection_sha256": selection_sha,
        "test_feature_cache_sha256": data.file_sha256,
        "prediction_device": device,
        "inference_batch_size": batch_size,
        "test_tuning": False,
        "all_seed_results_reported": True,
        "clinical_scale_equivalence_verified": False,
    }
    summary["coverage"]["by_level"] = {
        level: {
            name: metrics[name]
            for name in ("n_labeled", "n_assessed", "n_not_assessed", "coverage")
        }
        for level, metrics in common["by_level"].items()
    }
    write_json(out / "summary.json", summary)
    write_json(
        out / "manifest.json",
        {
            "scope": "heldout_test",
            "selection_sha256": selection_sha,
            "test_feature_cache_sha256": data.file_sha256,
            "implementation_sha256": _sha(Path(__file__)),
            "entries": [
                {
                    "variant": entry["variant"],
                    "seed": entry["seed"],
                    "checkpoint_sha256": entry["checkpoint_sha256"],
                }
                for entry in entries
            ],
            "summary_sha256": _sha(out / "summary.json"),
            "scope_note": "Predictions/ROI data stay local; aggregate reporting must not redistribute dataset identifiers or MRI pixels",
        },
    )
    return summary
