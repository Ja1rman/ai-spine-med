"""Apply a verified frozen SpineNet encoder and exported head to one MRI series.

This is an independent research inference entry point: no labels, cohort
manifest, downloaded-series marker or feature cache are needed for the input.
The selected head's training manifest/summary are model provenance, not patient
references. Keep the original v1 ``runs/seed-N/head.pt`` or v2
``validation/candidate/seed-N/head.pt`` model-directory layout.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import io
import json
import re
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from . import baseline, dicom, features
from .assets import sha256_file, write_json
from .data import CLASS_NAMES, LEVELS

_HASH = re.compile(r"^[0-9a-f]{64}$")
_SAFE_REASON = re.compile(r"^missing_vertebra:(?:L4|L5|S1)(?:,(?:L4|L5|S1))*$")
_REASONS = {
    "target_pair_not_detected",
    "duplicate_target_roi",
    "invalid_roi_shape_or_values",
}
_CSV_FIELDS = (
    "case_id",
    "level",
    "status",
    "reason",
    "prediction",
    "category",
    "p_normal_mild",
    "p_moderate",
    "p_severe",
)


@dataclass
class _Head:
    model: torch.nn.Module
    provenance: dict
    declared_seeds: tuple[int, ...]
    candidate: str = "linear"
    version: str = "v1"


def _canonical_sha(value: object) -> str:
    content = (
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode()
    return hashlib.sha256(content).hexdigest()


def _json_file(path: Path) -> tuple[dict, str]:
    try:
        content = path.read_bytes()
        value = json.loads(content)
    except (OSError, ValueError):
        raise ValueError("The head's training manifest and summary must accompany it") from None
    if not isinstance(value, dict):
        raise ValueError("Invalid head training provenance")
    return value, hashlib.sha256(content).hexdigest()


def _preprocessing_signature() -> dict[str, str]:
    return {
        "features.py": sha256_file(Path(features.__file__)),
        "dicom.py": sha256_file(Path(dicom.__file__)),
        "isotropic_inplane": hashlib.sha256(
            inspect.getsource(baseline.isotropic_inplane).encode()
        ).hexdigest(),
    }


def _verify_encoder_metadata(metadata, cache_sha, source: dict, preprocessing: dict) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("Invalid frozen-head training/cache provenance")
    config = metadata.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("Invalid frozen encoder configuration")
    if (
        metadata.get("extraction_complete") is not True
        or not isinstance(cache_sha, dict)
        or set(cache_sha) != {"features.npz", "target_manifest.csv", "metadata.json"}
        or any(not isinstance(v, str) or not _HASH.fullmatch(v) for v in cache_sha.values())
        or metadata.get("outputs_sha256")
        != {name: cache_sha[name] for name in ("features.npz", "target_manifest.csv")}
        or _canonical_sha(metadata) != cache_sha["metadata.json"]
        or metadata.get("cache_id") != _canonical_sha(config)
    ):
        raise ValueError("Frozen-head training/cache provenance does not match its checkpoint")
    if (
        any(config.get(name) != value for name, value in source.items())
        or config.get("implementation_sha256") != preprocessing
        or config.get("classes") != list(CLASS_NAMES)
        or config.get("levels") != list(LEVELS)
        or config.get("original_class_order") != ["none", "mild", "moderate", "severe"]
    ):
        raise ValueError("Current encoder or preprocessing differs from the head's frozen cache")


def _load_experiment_head(
    path: Path, checkpoint: dict, checkpoint_sha: str, source: dict, preprocessing: dict
) -> _Head:
    from . import head_experiments as experiments

    name = checkpoint.get("candidate")
    if (
        name not in experiments.CANDIDATES
        or checkpoint.get("schema_version") != 1
        or checkpoint.get("architecture") != experiments.architecture(name)
        or checkpoint.get("feature_size") != 512
        or checkpoint.get("class_names") != list(CLASS_NAMES)
        or type(checkpoint.get("seed")) is not int
        or checkpoint["seed"] not in experiments.SEEDS
        or type(checkpoint.get("best_epoch")) is not int
        or checkpoint["best_epoch"] < 1
        or checkpoint.get("encoder_updated") is not False
        or "fold" not in checkpoint
        or checkpoint["fold"] is not None
        or checkpoint.get("scope")
        != "Original-validation single development check; epochs selected by TRAIN CV"
        or any(
            not isinstance(checkpoint.get(field), str) or not _HASH.fullmatch(checkpoint[field])
            for field in (
                "implementation_sha256",
                "protocol_sha256",
                "folds_sha256",
                "selection_sha256",
            )
        )
        or len(path.parents) < 4
        or path.parents[2].name != "validation"
        or path.parents[1].name != name
    ):
        raise ValueError(
            "Only complete v2 final heads are supported; CV heads are not inference models"
        )
    root = path.parents[3]
    manifest, manifest_sha = _json_file(root / "manifest.json")
    summary, summary_sha = _json_file(root / "summary.json")
    selection, selection_sha = _json_file(root / "selection.json")
    configuration = manifest.get("configuration", {})
    metadata = manifest.get("feature_cache_metadata", {})
    cache_sha = checkpoint.get("feature_cache_sha256")
    _verify_encoder_metadata(metadata, cache_sha, source, preprocessing)
    if not isinstance(configuration, dict):
        raise ValueError("Invalid v2 head training configuration")
    provenance_fields = (
        "implementation_sha256",
        "protocol_sha256",
        "folds_sha256",
        "feature_cache_sha256",
        "feature_cache_id",
    )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("feature_cache_sha256") != cache_sha
        or configuration != summary.get("configuration")
        or configuration.get("class_names") != list(CLASS_NAMES)
        or configuration.get("architectures", {}).get(name) != experiments.architecture(name)
        or checkpoint.get("feature_cache_id") != metadata["cache_id"]
        or any(checkpoint.get(field) != configuration.get(field) for field in provenance_fields)
        or any(selection.get(field) != checkpoint.get(field) for field in provenance_fields)
        or checkpoint.get("implementation_sha256") != sha256_file(Path(experiments.__file__))
        or checkpoint.get("folds_sha256") != sha256_file(root / "folds.json")
        or checkpoint.get("selection_sha256") != selection_sha
        or summary.get("selection_sha256") != selection_sha
        or selection.get("created_before_original_validation_evaluation") is not True
        or selection.get("validation_used_for_epoch_or_candidate_selection") is not False
        or selection.get("test_accessed") is not False
        or summary.get("all_seeds_reported") is not True
        or summary.get("test_evaluated") is not False
        or summary.get("new_version_independently_tested") is not False
        or summary.get("selected_candidate") != selection.get("selected_candidate")
        or summary.get("final_epochs") != selection.get("final_epochs")
    ):
        raise ValueError("V2 head selection or encoder/training provenance does not match")
    try:
        runs = summary["candidates"][name]["validation"]["runs"]
        selected = [run for run in runs if run["seed"] == checkpoint["seed"]]
        epochs = selection["final_epochs"][name][str(checkpoint["seed"])]
    except (KeyError, TypeError):
        raise ValueError("Missing v2 final head training results") from None
    if (
        len(runs) != 3
        or {run.get("seed") for run in runs} != set(experiments.SEEDS)
        or len(selected) != 1
        or selected[0].get("fold") is not None
        or selected[0].get("best_epoch") != checkpoint["best_epoch"]
        or selected[0].get("checkpoint_sha256") != checkpoint_sha
        or checkpoint["best_epoch"] != epochs
    ):
        raise ValueError("V2 checkpoint is not a recorded final fit from its fixed epoch plan")
    model = experiments.build_head(name)
    expected = model.state_dict()
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("V2 head state dictionary does not match its declared architecture")
    for field, tensor in state.items():
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.shape != expected[field].shape
            or tensor.dtype != torch.float32
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError("Invalid or nonfinite v2 head tensors")
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    return _Head(
        model,
        {
            "sha256": checkpoint_sha,
            "candidate": name,
            "architecture": experiments.architecture(name),
            "seed": checkpoint["seed"],
            "best_epoch": checkpoint["best_epoch"],
            "training_manifest_sha256": manifest_sha,
            "training_summary_sha256": summary_sha,
            "training_implementation_sha256": checkpoint["implementation_sha256"],
            "selection_sha256": selection_sha,
            "protocol_sha256": checkpoint["protocol_sha256"],
            "folds_sha256": checkpoint["folds_sha256"],
            "feature_cache_sha256": cache_sha,
            "selected_by_training_cv": name == selection["selected_candidate"],
        },
        tuple(experiments.SEEDS),
        name,
        "v2",
    )


def _load_head(path: Path, source: dict, preprocessing: dict) -> _Head:
    try:
        content = path.read_bytes()
        checkpoint = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    except Exception:
        raise ValueError("Cannot load the restricted frozen-head checkpoint") from None
    checkpoint_sha = hashlib.sha256(content).hexdigest()
    if isinstance(checkpoint, dict) and "candidate" in checkpoint:
        return _load_experiment_head(path, checkpoint, checkpoint_sha, source, preprocessing)
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("schema_version") != 1
        or checkpoint.get("model") != "Linear(512,3)"
        or checkpoint.get("feature_size") != 512
        or checkpoint.get("class_names") != list(CLASS_NAMES)
        or type(checkpoint.get("seed")) is not int
        or checkpoint["seed"] < 0
        or type(checkpoint.get("best_epoch")) is not int
        or checkpoint["best_epoch"] < 1
    ):
        raise ValueError("Checkpoint architecture, category order or selection metadata differ")
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict) or set(state) != {"weight", "bias"}:
        raise ValueError("Expected only the frozen head's weight and bias tensors")
    for name, shape in (("weight", (3, 512)), ("bias", (3,))):
        value = state[name]
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or value.dtype != torch.float32
            or not torch.isfinite(value).all()
        ):
            raise ValueError("Invalid or nonfinite frozen-head tensors")
    if len(path.parents) < 3:
        raise ValueError("The head's training manifest and summary must accompany it")
    manifest, manifest_sha = _json_file(path.parents[2] / "manifest.json")
    summary, summary_sha = _json_file(path.parents[2] / "summary.json")
    configuration = manifest.get("configuration", {})
    cache_sha = checkpoint.get("feature_cache_sha256")
    metadata = manifest.get("feature_cache_metadata", {})
    if not isinstance(configuration, dict):
        raise ValueError("Invalid frozen-head training/cache provenance")
    _verify_encoder_metadata(metadata, cache_sha, source, preprocessing)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("class_names") != list(CLASS_NAMES)
        or configuration.get("model") != "Linear(512,3)"
        or configuration.get("encoder_updated") is not False
        or not isinstance(configuration.get("implementation_sha256"), str)
        or not _HASH.fullmatch(configuration["implementation_sha256"])
        or configuration != summary.get("configuration")
        or summary.get("all_seeds_reported") is not True
        or summary.get("test_evaluated") is not False
        or cache_sha != manifest.get("feature_cache_sha256")
    ):
        raise ValueError("Frozen-head training/cache provenance does not match its checkpoint")
    declared = configuration.get("seeds", [])
    runs = summary.get("runs", [])
    if (
        not isinstance(declared, list)
        or not isinstance(runs, list)
        or any(not isinstance(run, dict) for run in runs)
    ):
        raise ValueError("Invalid selected-head training run list")
    selected = [run for run in runs if run.get("seed") == checkpoint["seed"]]
    if (
        not declared
        or any(type(seed) is not int or seed < 0 for seed in declared)
        or len(set(declared)) != len(declared)
        or len(runs) != len(declared)
        or {run.get("seed") for run in runs} != set(declared)
        or len(selected) != 1
        or selected[0].get("best_epoch") != checkpoint["best_epoch"]
        or selected[0].get("checkpoint_sha256") != checkpoint_sha
    ):
        raise ValueError("The checkpoint is not the selected head recorded by its training run")
    head = torch.nn.Linear(512, 3)
    head.load_state_dict(state, strict=True)
    head.eval().requires_grad_(False)
    return _Head(
        head,
        {
            "sha256": checkpoint_sha,
            "seed": checkpoint["seed"],
            "best_epoch": checkpoint["best_epoch"],
            "training_manifest_sha256": manifest_sha,
            "training_summary_sha256": summary_sha,
            "training_implementation_sha256": configuration.get("implementation_sha256"),
            "feature_cache_sha256": cache_sha,
        },
        tuple(declared),
    )


def _heads(checkpoint, source: dict, preprocessing: dict) -> list[_Head]:
    if isinstance(checkpoint, (str, Path)):
        paths = [Path(checkpoint)]
    elif isinstance(checkpoint, Sequence):
        if any(not isinstance(path, (str, Path)) for path in checkpoint):
            raise ValueError("Checkpoint must be one path or three distinct head paths")
        paths = list(map(Path, checkpoint))
    else:
        raise ValueError("Checkpoint must be one path or three distinct head paths")
    if len(paths) not in {1, 3} or len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Use one head or the complete ensemble of three distinct heads")
    heads = [_load_head(path, source, preprocessing) for path in paths]
    if len(heads) == 3 and (
        len({head.provenance["training_manifest_sha256"] for head in heads}) != 1
        or len({head.provenance["training_summary_sha256"] for head in heads}) != 1
        or {head.provenance["seed"] for head in heads} != set(heads[0].declared_seeds)
        or len({(head.version, head.candidate) for head in heads}) != 1
    ):
        raise ValueError("An ensemble must contain all three selected seeds from one training run")
    return sorted(heads, key=lambda head: head.provenance["seed"])


def _empty_targets(reason: str) -> list[dict]:
    return [
        {
            "case_id": "case_1",
            "level": level,
            "status": "not_assessed",
            "reason": reason,
            "prediction": None,
            "category": None,
            **{f"p_{name}": None for name in CLASS_NAMES},
            "original_baseline": None,
        }
        for level in LEVELS
    ]


def _predictions(result: dict, arrays: dict, heads: list[_Head]) -> list[dict]:
    targets = result["targets"]
    if (
        len(targets) != 2
        or [row["level"] for row in targets] != list(LEVELS)
        or any(row["status"] not in {"ok", "not_assessed"} for row in targets)
    ):
        raise ValueError("Unexpected automatically assessed target set")
    assessed = [row for row in targets if row["status"] == "ok"]
    count = len(assessed)
    values = np.asarray(arrays["features"])
    if (
        values.shape != (count, 512)
        or values.dtype != np.float32
        or not np.isfinite(values).all()
        or np.asarray(arrays["levels"]).tolist() != [row["level"] for row in assessed]
        or [row["roi_index"] for row in assessed] != list(range(count))
    ):
        raise ValueError("Frozen features and automatically extracted levels do not align")
    original = features.merge_original_logits(arrays["original_logits"])
    if original.shape != (count, 3) or not np.allclose(
        original, arrays["original_probs"], rtol=1e-6, atol=1e-7
    ):
        raise ValueError("Original baseline probabilities do not match the four-grade logits")
    with torch.inference_mode():
        tensor = torch.from_numpy(values)
        distributions = []
        for head in heads:
            logits = head.model(tensor)
            if head.version == "v2":
                from .head_experiments import head_probabilities

                probability = head_probabilities(head.candidate, logits)
            else:
                probability = logits.softmax(dim=1)
            distributions.append(probability.numpy())
        probabilities = np.mean(distributions, axis=0, dtype=np.float64)
    if (
        probabilities.shape != (count, 3)
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-6)
    ):
        raise ValueError("The head returned invalid category probabilities")
    rows = _empty_targets("target_pair_not_detected")
    for row, target in zip(rows, targets):
        if target["status"] != "ok":
            reason = target["reason"]
            row["reason"] = (
                reason
                if reason in _REASONS or _SAFE_REASON.fullmatch(reason)
                else "target_pair_not_detected"
            )
            continue
        index = target["roi_index"]
        prediction = int(probabilities[index].argmax())
        original_prediction = int(original[index].argmax())
        row.update(
            status="ok",
            reason="",
            prediction=prediction,
            category=CLASS_NAMES[prediction],
            original_baseline={
                "prediction": original_prediction,
                "category": CLASS_NAMES[original_prediction],
                **{f"p_{name}": float(value) for name, value in zip(CLASS_NAMES, original[index])},
            },
            **{f"p_{name}": float(value) for name, value in zip(CLASS_NAMES, probabilities[index])},
        )
    return rows


def run_inference(
    dicom_dir: str | Path,
    out: str | Path,
    *,
    checkpoint: str | Path | Sequence[str | Path],
    cache_dir: str | Path = ".cache",
    device: str = "auto",
    threads: int = 4,
) -> dict:
    """Write two-level predictions.csv and predictions.json for one DICOM series.

    ``checkpoint`` accepts one exported v1/v2 final frozen head or all three
    heads of one architecture from the same run. V2 CV heads are refused. An
    ensemble averages category probabilities. Scores are
    uncalibrated research outputs, not a diagnosis. A failed/absent ROI remains
    ``not_assessed`` with empty scores. No input path, DICOM UID/header text,
    reference label, MRI pixels or intermediate ROI/features are exported.

    Configuration/provenance errors raise before creating ``out``. Processing
    failures produce an explicit failed report; an existing output is refused.
    """
    out, cache_dir, dicom_dir = map(Path, (out, cache_dir, dicom_dir))
    if out.exists():
        raise ValueError("Output already exists; choose a new inference directory")
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    started = time.perf_counter()
    selected_device = features._device(device)
    source = features._source_signature(cache_dir)
    preprocessing = _preprocessing_signature()
    heads = _heads(checkpoint, source, preprocessing)
    verified_at = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(threads)
    targets = _empty_targets("processing_failed")
    status, failure, stage = "failed", None, "model_loading"
    times = {"verify_model": verified_at - started}
    try:
        model = features._load_model(cache_dir, selected_device)
        loaded_at = time.perf_counter()
        times["load_original_model"] = loaded_at - verified_at
        stage = "dicom_localization_roi_and_encoder"
        # A present key with a None value requests the two automatic ROIs while
        # explicitly supplying no reference label. Empty keys would trigger the
        # research extractor's missing_label filter and silently skip both ROIs.
        row = {
            "study_id": "case_1",
            "series_id": "case_1",
            "split": "inference",
            "series_path": ".",
        }
        no_references = {("case_1", level): None for level in LEVELS}
        with warnings.catch_warnings(record=True):
            # Decoder warnings may include arbitrary DICOM header text. Keep
            # those local instead of printing them as part of the command.
            result, arrays = features._process_series(
                model, dicom_dir, row, no_references, selected_device
            )
        encoded_at = time.perf_counter()
        times["dicom_localization_roi_and_encoder"] = encoded_at - loaded_at
        stage = "frozen_head"
        targets = _predictions(result, arrays, heads)
        times["frozen_head"] = time.perf_counter() - encoded_at
        count = sum(target["status"] == "ok" for target in targets)
        status = "ok" if count == 2 else "partial" if count else "failed"
        if not count:
            failure = {"stage": "automatic_roi", "reason": "no_supported_target_roi"}
    except Exception as error:
        reason = (
            "invalid_dicom_series"
            if isinstance(error, dicom.DicomGeometryError)
            else "model_loading_failed"
            if stage == "model_loading"
            else "processing_failed"
        )
        targets = _empty_targets(reason)
        failure = {"stage": stage, "reason": reason, "error_type": type(error).__name__}
        # Never serialize/print str(error): decoder/library errors can contain
        # patient names, DICOM identifiers or the input filesystem path.
    times["total_before_output"] = time.perf_counter() - started
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": "case_1",
        "status": status,
        "n_requested": 2,
        "n_assessed": sum(target["status"] == "ok" for target in targets),
        "targets": targets,
        "failure": failure,
        "model": {
            "variant": ("frozen_head" if heads[0].version == "v1" else "frozen_head_v2")
            + ("" if len(heads) == 1 else "_probability_ensemble"),
            "candidate": heads[0].candidate,
            "architecture": "Original SpineNet frozen encoder + "
            + ("Linear(512,3)" if heads[0].version == "v1" else heads[0].candidate),
            "class_names": list(CLASS_NAMES),
            "levels": list(LEVELS),
            "aggregation": "single head"
            if len(heads) == 1
            else "mean of three category probability vectors",
            "checkpoints": [head.provenance for head in heads],
            "encoder": source,
            "preprocessing_sha256": preprocessing,
            "inference_implementation_sha256": sha256_file(Path(__file__)),
            "original_baseline_mapping": "softmax4 -> [p_none+p_mild,p_moderate,p_severe] -> argmax3",
        },
        "runtime": {
            "encoder_device": selected_device,
            "head_device": "cpu",
            "threads": threads,
            "torch": str(torch.__version__),
            "numpy": np.__version__,
        },
        "seconds": times,
        "scope": "Research inference on one sagittal T2-like lumbar MRI series. Sequence subtype and anatomical numbering require external review; this output is not a clinical diagnosis or a certified medical device.",
        "probabilities_calibrated": False,
        "reference_labels_used": False,
        "clinical_scale_equivalence_verified": False,
    }
    with (out / "predictions.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(targets)
    write_json(out / "predictions.json", report)
    return report
