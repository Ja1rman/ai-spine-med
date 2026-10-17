"""Cache unmodified SpineNet ROIs, frozen features and baseline predictions.

Only training and validation are accepted. No reference coordinates are passed
to the detector, and a failed/missing ROI is never converted into normal.
"""

from __future__ import annotations

import csv
import fcntl
import hashlib
import importlib.metadata
import inspect
import json
import re
import tempfile
import time
import zipfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from .assets import (
    COMPATIBILITY_PATCH,
    SOURCE_URL,
    SPINENET_COMMIT,
    expected_digest,
    sha256_file,
    source_dir,
    verify_assets,
)
from .baseline import import_upstream, isotropic_inplane
from .data import CLASS_NAMES, LEVELS, METADATA_FILES
from .dicom import load_sagittal_series
from .rsna_bulk import COMPLETE_MARKER, is_complete_series

SCHEMA_VERSION = 1
ALLOWED_SPLITS = ("train", "validation")
TARGET_FIELDS = (
    "study_id",
    "series_id",
    "split",
    "level",
    "label",
    "status",
    "reason",
    "roi_path",
    "roi_index",
)
ARRAY_KEYS = (
    "features",
    "original_logits",
    "original_probs",
    "labels",
    "study_id",
    "series_id",
    "level",
    "split",
)
_ID = re.compile(r"^[0-9]+$")


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.name + ".", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    temporary.replace(path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.name + ".", delete=False
    ) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


@contextmanager
def _output_lock(output_dir: Path):
    # Disjoint workers may finish series independently; serialize only aggregate
    # replacement, never the slow inference. flock releases on process exit.
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".aggregate.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def merge_original_logits(logits: np.ndarray) -> np.ndarray:
    """Merge none+mild probabilities BEFORE argmax, preserving row order."""
    values = np.asarray(logits)
    if values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all():
        raise ValueError("Original CCS logits must be a finite [N,4] array")
    stable = values.astype(np.float64) - values.max(axis=1, keepdims=True)
    probabilities = np.exp(stable)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return np.column_stack(
        (probabilities[:, 0] + probabilities[:, 1], probabilities[:, 2:])
    ).astype(np.float32)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _cohort(data_root: Path, manifest_dir: Path) -> tuple[list[dict], dict, dict]:
    summary = json.loads((manifest_dir / "summary.json").read_text())
    manifest_hashes = {}
    for name in ("studies.csv", "series.csv", "labels.csv"):
        manifest_hashes[name] = sha256_file(manifest_dir / name)
        if manifest_hashes[name] != summary["output_sha256"][name]:
            raise ValueError(f"Manifest checksum mismatch: {name}")
    for name in METADATA_FILES:
        if sha256_file(data_root / name) != summary["input_sha256"][name]:
            raise ValueError(f"RSNA source metadata changed: {name}")
    studies = _read_csv(manifest_dir / "studies.csv")
    study_splits = {item["study_id"]: item["split"] for item in studies}
    if len(studies) != len(study_splits):
        raise ValueError("Duplicate study groups in split manifest")
    labels = {}
    for row in _read_csv(manifest_dir / "labels.csv"):
        key = row["study_id"], row["level"]
        if (
            key in labels
            or row["level"] not in LEVELS
            or row["split"] != study_splits.get(row["study_id"])
        ):
            raise ValueError("Invalid/duplicate label or inconsistent study split")
        if row["label"] not in {"0", "1", "2"}:
            raise ValueError("Unexpected label outside the fixed three classes")
        labels[key] = int(row["label"])
    rows, seen = [], set()
    for row in _read_csv(manifest_dir / "series.csv"):
        study, series = row["study_id"], row["series_id"]
        if not _ID.fullmatch(study) or not _ID.fullmatch(series):
            raise ValueError("Non-numeric series/study identifier")
        if row["split"] != study_splits.get(study):
            raise ValueError("Series/study split disagreement")
        key = study, series
        if key in seen:
            raise ValueError("Duplicate series in manifest")
        seen.add(key)
        if row["series_path"] != f"train_images/{study}/{series}":
            raise ValueError("Unexpected series path in manifest")
        if row["series_description"] != "Sagittal T2/STIR":
            raise ValueError("The fixed cohort contains an unsupported sequence description")
        rows.append(row)
    return rows, labels, {"manifests": manifest_hashes, "source_metadata": summary["input_sha256"]}


def _source_signature(cache_dir: Path) -> dict:
    """Verify Python source as well as checkpoint bytes before model loading."""
    assets = verify_assets(cache_dir)
    archive = cache_dir / "downloads" / f"SpineNet-{SPINENET_COMMIT}.zip"
    if sha256_file(archive) != expected_digest(SOURCE_URL):
        raise ValueError("Pinned upstream source archive failed its checksum")
    upstream = source_dir(cache_dir)
    expected = {}
    with zipfile.ZipFile(archive) as zipped:
        prefix = upstream.name + "/"
        for name in zipped.namelist():
            if name.startswith(prefix + "spinenet/") and name.endswith(".py"):
                relative = name[len(prefix) :]
                content = zipped.read(name)
                if relative == COMPATIBILITY_PATCH["path"]:
                    content = content.replace(
                        COMPATIBILITY_PATCH["before"].encode(),
                        COMPATIBILITY_PATCH["after"].encode(),
                    )
                expected[relative] = hashlib.sha256(content).hexdigest()
    actual = {
        str(p.relative_to(upstream)): sha256_file(p) for p in (upstream / "spinenet").rglob("*.py")
    }
    if actual != expected:
        raise ValueError(
            "Upstream Python source differs from the reviewed source and compatibility fix"
        )
    return {
        "upstream_commit": SPINENET_COMMIT,
        "source_archive_sha256": assets["source"]["sha256"],
        "source_python_sha256": expected,
        "weights_sha256": {name: item["sha256"] for name, item in assets["weights"].items()},
    }


def _device(requested: str) -> str:
    if requested == "auto":
        requested = (
            "cuda:0"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError(
            "MPS is unavailable in this process; on macOS use a process with GPU access or explicitly choose cpu"
        )
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if requested not in {"cpu", "mps"} and not requested.startswith("cuda"):
        raise ValueError("Expected auto, cpu, mps, or cuda[:N]")
    return requested


def _load_model(cache_dir: Path, device: str):
    upstream = import_upstream(cache_dir)
    with torch.serialization.safe_globals(
        [np.core.multiarray.scalar, np.dtype, type(np.dtype(np.float64))]
    ):
        model = upstream.SpineNet(device=device, verbose=False, scan_type="lumbar")
    for name in ("detection_model", "appearance_model", "context_model", "grading_model"):
        getattr(model, name).eval().requires_grad_(False)
    return model


def _available_source(data_root: Path, row: dict, manifest_receipt: str) -> tuple[dict | None, str]:
    directory = data_root / row["series_path"]
    if not is_complete_series(directory):
        return None, "series_not_fully_downloaded_or_verified"
    receipt = json.loads((directory / COMPLETE_MARKER).read_text())
    if receipt["manifest_sha256"] != manifest_receipt:
        raise ValueError("Download completeness receipt does not match the fixed cohort")
    paths = [directory / name for name in sorted(receipt["files"])]
    if len(paths) < 2:
        raise ValueError("A spatial MRI series requires at least two slices")
    # Ignore neither a partially written slice nor a concurrent replacement.
    if any(p.suffix == ".part" for p in directory.iterdir()):
        return None, "download_in_progress"
    files = [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in paths]
    source = {"files": files, "download_receipt": receipt}
    return {**source, "sha256": _digest(source)}, ""


def _targets(row: dict, labels: dict, status: str, reason: str) -> list[dict]:
    return [
        {
            "study_id": row["study_id"],
            "series_id": row["series_id"],
            "split": row["split"],
            "level": level,
            "label": labels.get((row["study_id"], level)),
            "status": status if (row["study_id"], level) in labels else "missing_label",
            "reason": reason if (row["study_id"], level) in labels else "no_reference_label",
            "roi_path": "",
            "roi_index": "",
        }
        for level in LEVELS
    ]


def _record_path(output_dir: Path, row: dict) -> Path:
    return output_dir / "records" / f"{row['study_id']}_{row['series_id']}.json"


def _capture_grading(grading_model, rois: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    """The head's exact input is the frozen 512-vector, captured in its forward."""
    captured = []
    handle = grading_model.fc_ccs.register_forward_pre_hook(
        lambda _module, arguments: captured.append(arguments[0].detach().cpu().numpy().copy())
    )
    try:
        with torch.inference_mode():
            tensor = torch.from_numpy(rois.astype(np.float32))[:, None].to(device)
            outputs = grading_model(tensor)
            logits = outputs[2].detach().cpu().numpy().copy()
    finally:
        handle.remove()
    features = np.concatenate(captured)
    if features.shape != (len(rois), 512) or logits.shape != (len(rois), 4):
        raise ValueError("Unexpected original grading model feature/head shapes")
    if not np.isfinite(features).all() or not np.isfinite(logits).all():
        raise ValueError("Nonfinite frozen features or original logits")
    return features.astype(np.float32), logits.astype(np.float32)


def _process_series(
    model, data_root: Path, row: dict, labels: dict, device: str
) -> tuple[dict, dict]:
    started = time.perf_counter()
    scan = load_sagittal_series(data_root / row["series_path"])
    volume, spacing, resampling = isotropic_inplane(scan.volume, scan.pixel_spacing)
    with torch.inference_mode():
        vertices = model.detect_vb(volume, spacing)
        ivds = model.get_ivds_from_vert_dicts(vertices, volume)
    targets = _targets(row, labels, "not_assessed", "target_pair_not_detected")
    arrays, saved_levels = [], []
    roi_qc = []
    for target in targets:
        if target["status"] == "missing_label":
            continue
        matches = [item for item in ivds if item["level_name"] == target["level"]]
        if len(matches) != 1:
            names = {item["predicted_label"] for item in vertices}
            missing = [name for name in target["level"].split("-") if name not in names]
            target["reason"] = (
                "duplicate_target_roi"
                if len(matches) > 1
                else "missing_vertebra:" + ",".join(missing)
                if missing
                else "target_pair_not_detected"
            )
            continue
        roi = np.asarray(matches[0]["volume"])
        if roi.shape != (9, 112, 224) or not np.isfinite(roi).all() or np.ptp(roi) <= 1e-12:
            target["reason"] = "invalid_roi_shape_or_values"
            continue
        target.update(status="ok", reason="", roi_index=len(arrays))
        arrays.append(roi.copy())
        saved_levels.append(target["level"])
        roi_qc.append(
            {
                "level": target["level"],
                "zero_fraction": float(np.mean(roi == 0)),
                "min": float(roi.min()),
                "max": float(roi.max()),
            }
        )
    if arrays:
        rois = np.stack(arrays)
        features, logits = _capture_grading(model.grading_model, rois, device)
    else:
        rois, features, logits = (
            np.empty((0, 9, 112, 224)),
            np.empty((0, 512), np.float32),
            np.empty((0, 4), np.float32),
        )
    result = {
        "targets": targets,
        "input": {**scan.metadata, **resampling},
        "vertices": [
            {
                "label": str(item["predicted_label"]),
                "slice_nos": [int(i) for i in item["slice_nos"]],
                "polys": np.asarray(item["polys"]).tolist(),
                "average_polygon": np.asarray(item["average_polygon"]).tolist(),
            }
            for item in vertices
        ],
        "roi_qc": roi_qc,
        "seconds": time.perf_counter() - started,
    }
    return result, {
        "rois": rois,
        "features": features,
        "original_logits": logits,
        "original_probs": merge_original_logits(logits),
        "levels": np.asarray(saved_levels, dtype="U8"),
    }


def _check_record(record: dict, config_id: str, source: dict | None, output_dir: Path) -> None:
    if record["cache_id"] != config_id:
        raise ValueError("A cached series belongs to different model/split/preprocessing settings")
    if source is not None and record["source"]["sha256"] != source["sha256"]:
        raise ValueError("Source DICOM bytes changed after extraction; use a new output directory")
    if record.get("arrays_path"):
        path = (output_dir / record["arrays_path"]).resolve()
        if not path.is_relative_to(output_dir.resolve()):
            raise ValueError("Cached arrays path escapes output directory")
        if not path.is_file() or sha256_file(path) != record["arrays_sha256"]:
            raise ValueError("Per-series feature/ROI cache is missing or corrupted")


def _aggregate(
    output_dir: Path, rows: list[dict], labels: dict, config: dict, config_id: str, runtime: dict
) -> dict:
    with _output_lock(output_dir):
        targets, groups, timings = [], [], []
        for row in rows:
            record_path = _record_path(output_dir, row)
            if not record_path.is_file():
                targets.extend(
                    _targets(row, labels, "pending", "not_extracted_or_download_incomplete")
                )
                continue
            record = json.loads(record_path.read_text())
            _check_record(record, config_id, None, output_dir)
            targets.extend(record["targets"])
            if record.get("arrays_path"):
                with np.load(output_dir / record["arrays_path"], allow_pickle=False) as arrays:
                    for target in record["targets"]:
                        if target["status"] != "ok":
                            continue
                        index = target["roi_index"]
                        if str(arrays["levels"][index]) != target["level"]:
                            raise ValueError("Cached ROI order/target level mismatch")
                        groups.append(
                            {
                                **target,
                                "features": arrays["features"][index],
                                "original_logits": arrays["original_logits"][index],
                                "original_probs": arrays["original_probs"][index],
                            }
                        )
            if "seconds" in record:
                timings.append(record["seconds"])
        arrays = {
            "features": np.stack([g["features"] for g in groups]).astype(np.float32)
            if groups
            else np.empty((0, 512), np.float32),
            "original_logits": np.stack([g["original_logits"] for g in groups]).astype(np.float32)
            if groups
            else np.empty((0, 4), np.float32),
            "original_probs": np.stack([g["original_probs"] for g in groups]).astype(np.float32)
            if groups
            else np.empty((0, 3), np.float32),
            "labels": np.asarray([g["label"] for g in groups], dtype=np.int64),
            **{
                key: np.asarray([g[key] for g in groups], dtype="U32")
                for key in ("study_id", "series_id", "level", "split")
            },
        }
        _atomic_npz(output_dir / "features.npz", arrays)
        import io

        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=TARGET_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(targets)
        _atomic_bytes(output_dir / "target_manifest.csv", stream.getvalue().encode())
        coverage = {}
        for split in ALLOWED_SPLITS:
            subset = [
                item for item in targets if item["split"] == split and item["label"] is not None
            ]
            coverage[split] = {
                "requested_labeled_targets": len(subset),
                "assessed": sum(item["status"] == "ok" for item in subset),
                "by_status": dict(Counter(item["status"] for item in subset)),
                "by_level": {
                    level: {
                        "requested": sum(item["level"] == level for item in subset),
                        "assessed": sum(
                            item["level"] == level and item["status"] == "ok" for item in subset
                        ),
                    }
                    for level in LEVELS
                },
                "by_label": {
                    str(label): {
                        "requested": sum(item["label"] == label for item in subset),
                        "assessed": sum(
                            item["label"] == label and item["status"] == "ok" for item in subset
                        ),
                    }
                    for label in range(3)
                },
            }
        complete = all(item["status"] != "pending" for item in targets)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "cache_id": config_id,
            "n_features": len(groups),
            "n_requested_series": len(rows),
            "coverage": coverage,
            "extraction_complete": complete,
            "median_series_seconds": float(np.median(timings)) if timings else None,
            "runtime": runtime,
            "scope": "Automatic research ROIs and label agreement, no clinical/anatomical expert validation. Test is untouched.",
        }
        outputs = {
            name: sha256_file(output_dir / name) for name in ("features.npz", "target_manifest.csv")
        }
        _atomic_bytes(
            output_dir / "metadata.json",
            _json_bytes(
                {
                    "schema_version": SCHEMA_VERSION,
                    "cache_id": config_id,
                    "config": config,
                    "outputs_sha256": outputs,
                    "extraction_complete": complete,
                }
            ),
        )
        _atomic_bytes(output_dir / "summary.json", _json_bytes(summary))
    return summary


def extract_features(
    data_root: str | Path,
    manifest_dir: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    device: str = "mps",
    splits: tuple[str, ...] = ALLOWED_SPLITS,
    limit: int | None = None,
    retry_failed: bool = False,
    *,
    worker_index: int = 0,
    num_workers: int = 1,
    threads: int = 4,
) -> dict:
    """Resume ready local series; rerun as downloading supplies more series.

    ``limit`` bounds newly attempted series, not already verified cached ones.
    Optional disjoint index-modulo workers share per-series records safely.
    Aggregate files are rebuilt on return; training must wait for all workers.
    """
    if not splits or len(set(splits)) != len(splits) or set(splits) - set(ALLOWED_SPLITS):
        raise ValueError("Only train and validation splits may be extracted; test is sealed")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if num_workers < 1 or not 0 <= worker_index < num_workers or threads < 1:
        raise ValueError("Invalid worker or thread configuration")
    data_root, manifest_dir, cache_dir, output_dir = map(
        Path, (data_root, manifest_dir, cache_dir, output_dir)
    )
    device = _device(device)
    torch.set_num_threads(threads)
    rows, labels, manifest_signature = _cohort(data_root, manifest_dir)
    manifest_receipt = hashlib.sha256(
        (
            json.dumps(
                {
                    name: manifest_signature["manifests"][name]
                    for name in ("studies.csv", "series.csv")
                },
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode()
    ).hexdigest()
    cohort_rows = [row for row in rows if row["split"] in ALLOWED_SPLITS]
    rows = [row for row in cohort_rows if row["split"] in splits]
    config = {
        "schema_version": SCHEMA_VERSION,
        **manifest_signature,
        **_source_signature(cache_dir),
        "device": device,
        "classes": list(CLASS_NAMES),
        "levels": list(LEVELS),
        "allowed_splits": list(ALLOWED_SPLITS),
        "original_class_order": ["none", "mild", "moderate", "severe"],
        "baseline_merge": "softmax4 -> [p_none+p_mild,p_moderate,p_severe] -> argmax3",
        "roi_storage": "exact upstream numpy dtype, model input is float32; [9,112,224]",
        "feature_definition": "original fc_ccs forward input after avgpool+flatten, 512 floats",
        "sequence_scope": "RSNA Sagittal T2/STIR T2-like; conventional T2 versus STIR/Dixon subtype remains unknown",
        "implementation_sha256": {
            "features.py": sha256_file(Path(__file__)),
            "dicom.py": sha256_file(Path(inspect.getfile(load_sagittal_series))),
            "isotropic_inplane": hashlib.sha256(
                inspect.getsource(isotropic_inplane).encode()
            ).hexdigest(),
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "torch", "torchvision", "pydicom", "scipy")
        },
    }
    config_id = _digest(config)
    with _output_lock(output_dir):
        metadata_path = output_dir / "metadata.json"
        if (
            metadata_path.is_file()
            and json.loads(metadata_path.read_text())["cache_id"] != config_id
        ):
            raise ValueError(
                "Existing cache has different model, split, source or preprocessing provenance; choose a new output directory"
            )
        if not metadata_path.exists():
            _atomic_bytes(
                metadata_path,
                _json_bytes(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "cache_id": config_id,
                        "config": config,
                        "outputs_sha256": {},
                        "extraction_complete": False,
                    }
                ),
            )
    model, attempted, reused, waiting = None, 0, 0, 0
    for index, row in enumerate(rows):
        if index % num_workers != worker_index:
            continue
        path = _record_path(output_dir, row)
        existing = json.loads(path.read_text()) if path.is_file() else None
        source, _reason = _available_source(data_root, row, manifest_receipt)
        if existing:
            _check_record(existing, config_id, source, output_dir)
            failed = any(
                target["status"] not in {"ok", "missing_label"} for target in existing["targets"]
            )
            if not (retry_failed and failed):
                reused += 1
                continue
        if source is None:
            waiting += 1
            # Do not persist a failure: an incomplete download becomes eligible
            # automatically next time, without requiring --retry-failed.
            continue
        if limit is not None and attempted >= limit:
            continue
        if model is None:
            model = _load_model(cache_dir, device)
        attempted += 1
        record = {"cache_id": config_id, "source": source, "arrays_path": "", "arrays_sha256": ""}
        try:
            result, arrays = _process_series(model, data_root, row, labels, device)
            arrays_path = path.with_suffix(".npz")
            _atomic_npz(arrays_path, arrays)
            record.update(
                result,
                arrays_path=str(arrays_path.relative_to(output_dir)),
                arrays_sha256=sha256_file(arrays_path),
            )
            for target in record["targets"]:
                if target["status"] == "ok":
                    target["roi_path"] = record["arrays_path"]
        except (ValueError, RuntimeError, OSError, IndexError, KeyError, AssertionError) as exc:
            record.update(
                targets=_targets(row, labels, "failed", type(exc).__name__),
                error_type=type(exc).__name__,
                error_message=str(exc)[:1000],
            )
            print(
                f"Feature extraction failed: {row['study_id']}/{row['series_id']} ({type(exc).__name__})",
                flush=True,
            )
        _atomic_bytes(path, _json_bytes(record))
        assessed = sum(target["status"] == "ok" for target in record["targets"])
        print(
            f"Extracted {attempted}: {row['study_id']}/{row['series_id']} {assessed}/2 target ROIs",
            flush=True,
        )
    return _aggregate(
        output_dir,
        cohort_rows,
        labels,
        config,
        config_id,
        {
            "attempted": attempted,
            "reused": reused,
            "waiting_download": waiting,
            "worker_index": worker_index,
            "num_workers": num_workers,
            "device": device,
            "threads": threads,
        },
    )
