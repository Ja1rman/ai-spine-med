"""Extract TEST-only original SpineNet ROIs/features after a frozen model seal.

This module never trains a model. Its cache is deliberately rejected by the
training reader; only the sealed evaluation reader may consume it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import io
import json
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from . import features as f
from .assets import sha256_file
from .data import CLASS_NAMES, LEVELS
from .rsna_bulk import ArchiveDownloadError, validate_test_selection


def _synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize(device)


@contextmanager
def _grading_timer(grading_model, device):
    measurement = {"seconds": 0.0, "calls": 0}
    starts = []

    def before(_module, _inputs):
        _synchronize(device)
        starts.append(time.perf_counter())

    def after(_module, _inputs, _output):
        _synchronize(device)
        if starts:
            measurement["seconds"] += time.perf_counter() - starts.pop()
            measurement["calls"] += 1

    first = grading_model.register_forward_pre_hook(before)
    second = grading_model.register_forward_hook(after, always_call=True)
    try:
        yield measurement
    finally:
        first.remove()
        second.remove()


def _aggregate_test(output_dir, rows, labels, config, config_id, runtime):
    """Keep the frozen cache schema while accepting exactly the held-out split."""
    with f._output_lock(output_dir):
        targets, groups, timings = [], [], []
        for row in rows:
            if row["split"] != "test":
                raise ValueError("A non-test series entered the held-out cache")
            path = f._record_path(output_dir, row)
            if not path.is_file():
                targets.extend(
                    f._targets(row, labels, "pending", "not_extracted_or_download_incomplete")
                )
                continue
            record = json.loads(path.read_text())
            f._check_record(record, config_id, None, output_dir)
            expected = {(row["study_id"], row["series_id"], level) for level in LEVELS}
            actual = {
                (target["study_id"], target["series_id"], target["level"])
                for target in record["targets"]
            }
            if actual != expected or len(record["targets"]) != len(LEVELS):
                raise ValueError("Held-out record target identifiers changed")
            if any(target["split"] != "test" for target in record["targets"]):
                raise ValueError("A non-test target entered the held-out cache")
            targets.extend(record["targets"])
            if record.get("arrays_path"):
                with np.load(output_dir / record["arrays_path"], allow_pickle=False) as arrays:
                    for target in record["targets"]:
                        if target["status"] != "ok":
                            continue
                        index = target["roi_index"]
                        if str(arrays["levels"][index]) != target["level"]:
                            raise ValueError("Cached ROI order/target level mismatch")
                        if target["label"] != labels.get((row["study_id"], target["level"])):
                            raise ValueError("Held-out target reference label changed")
                        groups.append(
                            {
                                **target,
                                "features": arrays["features"][index],
                                "original_logits": arrays["original_logits"][index],
                                "original_probs": arrays["original_probs"][index],
                            }
                        )
            if "timings" in record:
                timings.append(
                    {
                        "study_id": row["study_id"],
                        "series_id": row["series_id"],
                        **record["timings"],
                    }
                )
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
        f._atomic_npz(output_dir / "features.npz", arrays)
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=f.TARGET_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(targets)
        f._atomic_bytes(output_dir / "target_manifest.csv", stream.getvalue().encode())
        subset = [target for target in targets if target["label"] is not None]
        coverage = {
            "test": {
                "requested_labeled_targets": len(subset),
                "assessed": sum(target["status"] == "ok" for target in subset),
                "by_status": dict(Counter(target["status"] for target in subset)),
                "by_level": {
                    level: {
                        "requested": sum(t["level"] == level for t in subset),
                        "assessed": sum(
                            t["level"] == level and t["status"] == "ok" for t in subset
                        ),
                    }
                    for level in LEVELS
                },
                "by_label": {
                    str(label): {
                        "requested": sum(t["label"] == label for t in subset),
                        "assessed": sum(
                            t["label"] == label and t["status"] == "ok" for t in subset
                        ),
                    }
                    for label in range(3)
                },
            }
        }
        complete = all(target["status"] != "pending" for target in targets)
        full_seconds = [t["full_original_seconds"] for t in timings]
        summary = {
            "schema_version": f.SCHEMA_VERSION,
            "scope": "heldout_test",
            "selection_sha256": config["selection_sha256"],
            "cache_id": config_id,
            "n_features": len(groups),
            "n_requested_series": len(rows),
            "coverage": coverage,
            "extraction_complete": complete,
            "median_series_seconds": float(np.median(full_seconds)) if full_seconds else None,
            "timing_definition": "Synchronized original pipeline includes DICOM load, resampling, detector, exact ROI construction and original grading; model load and file hashing excluded. Grading forward hooks are synchronized. The residual includes localization, ROI and tensor/feature IO, and is not a pure detector benchmark. Failed attempts retained locally.",
            "timing_aggregates": {
                name: {
                    "count": len(timings),
                    "median_seconds": float(np.median([t[name] for t in timings]))
                    if timings
                    else None,
                    "total_seconds": float(sum(t[name] for t in timings)),
                }
                for name in (
                    "full_original_seconds",
                    "measured_grading_forward_seconds",
                    "common_DICOM_localization_ROI_plus_tensorIO_seconds",
                    "dicom_hash_seconds",
                )
            },
            "runtime": runtime,
        }
        f._atomic_bytes(
            output_dir / "metadata.json",
            f._json_bytes(
                {
                    "schema_version": f.SCHEMA_VERSION,
                    "cache_id": config_id,
                    "config": config,
                    "outputs_sha256": {
                        name: sha256_file(output_dir / name)
                        for name in ("features.npz", "target_manifest.csv")
                    },
                    "extraction_complete": complete,
                }
            ),
        )
        f._atomic_bytes(output_dir / "summary.json", f._json_bytes(summary))
    return summary


def extract_heldout_features(
    data_root: str | Path,
    manifest_dir: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    selection_manifest: str | Path,
    device: str = "mps",
    limit: int | None = None,
    retry_failed: bool = False,
    *,
    worker_index: int = 0,
    num_workers: int = 1,
    threads: int = 4,
) -> dict:
    """Resume only TEST series bound to one unchanged, fully frozen selection."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if num_workers < 1 or not 0 <= worker_index < num_workers or threads < 1:
        raise ValueError("Invalid worker or thread configuration")
    data_root, manifest_dir, cache_dir, output_dir = map(
        Path, (data_root, manifest_dir, cache_dir, output_dir)
    )
    # Gate precedes device/model loading and any inspection of held-out MRI.
    selection, selection_sha = validate_test_selection(manifest_dir, selection_manifest)
    rows, labels, manifest_signature = f._cohort(data_root, manifest_dir)
    rows = [row for row in rows if row["split"] == "test"]
    if not rows:
        raise ValueError("The fixed manifest has no held-out test series")
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
    source_signature = f._source_signature(cache_dir)
    if source_signature["weights_sha256"] != selection["original_weights_sha256"]:
        raise ValueError("Original model weights differ from the frozen selection")
    implementation = {
        "features.py": sha256_file(Path(f.__file__)),
        "dicom.py": sha256_file(Path(inspect.getfile(f.load_sagittal_series))),
        "isotropic_inplane": hashlib.sha256(
            inspect.getsource(f.isotropic_inplane).encode()
        ).hexdigest(),
    }
    frozen_config = selection.get("trainval_config", {})
    if any(
        source_signature.get(name) != frozen_config.get(name)
        for name in (
            "upstream_commit",
            "source_archive_sha256",
            "source_python_sha256",
            "weights_sha256",
        )
    ) or any(
        digest != frozen_config.get("implementation_sha256", {}).get(name)
        for name, digest in implementation.items()
    ):
        raise ValueError(
            "Source or preprocessing differs from the frozen training/validation cache"
        )
    device = f._device(device)
    torch.set_num_threads(threads)
    config = {
        "schema_version": f.SCHEMA_VERSION,
        "scope": "heldout_test",
        "selection_sha256": selection_sha,
        **manifest_signature,
        **source_signature,
        "device": device,
        "classes": list(CLASS_NAMES),
        "levels": list(LEVELS),
        "allowed_splits": ["test"],
        "original_class_order": ["none", "mild", "moderate", "severe"],
        "baseline_merge": "softmax4 -> [p_none+p_mild,p_moderate,p_severe] -> argmax3",
        "roi_storage": "exact upstream numpy dtype, model input is float32; [9,112,224]",
        "feature_definition": "original fc_ccs forward input after avgpool+flatten, 512 floats",
        "sequence_scope": "RSNA Sagittal T2/STIR T2-like; conventional T2 versus STIR/Dixon subtype remains unknown",
        "implementation_sha256": {
            **implementation,
            "heldout_features.py": sha256_file(Path(__file__)),
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "torch", "torchvision", "pydicom", "scipy")
        },
    }
    config_id = f._digest(config)
    with f._output_lock(output_dir):
        metadata_path = output_dir / "metadata.json"
        if metadata_path.is_file():
            previous = json.loads(metadata_path.read_text())
            if previous.get("cache_id") != config_id:
                raise ValueError(
                    "Existing cache has a different selection/model/source/device; choose a separate TEST output directory"
                )
        else:
            f._atomic_bytes(
                metadata_path,
                f._json_bytes(
                    {
                        "schema_version": f.SCHEMA_VERSION,
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
        path = f._record_path(output_dir, row)
        existing = json.loads(path.read_text()) if path.is_file() else None
        hashing_started = time.perf_counter()
        source, _reason = f._available_source(data_root, row, manifest_receipt)
        hash_seconds = time.perf_counter() - hashing_started
        if source is not None:
            receipt = source["download_receipt"]
            if (
                receipt.get("scope") != "heldout_test"
                or receipt.get("selection_sha256") != selection_sha
            ):
                raise ValueError("TEST download receipt is not bound to this frozen selection")
        if existing:
            f._check_record(existing, config_id, source, output_dir)
            failed = any(t["status"] not in {"ok", "missing_label"} for t in existing["targets"])
            if not (retry_failed and failed):
                reused += 1
                continue
        if source is None:
            waiting += 1
            continue
        if limit is not None and attempted >= limit:
            continue
        if model is None:
            model = f._load_model(cache_dir, device)
        attempted += 1
        record = {"cache_id": config_id, "source": source, "arrays_path": "", "arrays_sha256": ""}
        _synchronize(device)
        started = time.perf_counter()
        with _grading_timer(model.grading_model, device) as grading:
            try:
                result, arrays = f._process_series(model, data_root, row, labels, device)
                _synchronize(device)
                elapsed = time.perf_counter() - started
                arrays_path = path.with_suffix(".npz")
                f._atomic_npz(arrays_path, arrays)
                record.update(
                    result,
                    arrays_path=str(arrays_path.relative_to(output_dir)),
                    arrays_sha256=sha256_file(arrays_path),
                )
                for target in record["targets"]:
                    if target["status"] == "ok":
                        target["roi_path"] = record["arrays_path"]
            except (ValueError, RuntimeError, OSError, IndexError, KeyError, AssertionError) as exc:
                _synchronize(device)
                elapsed = time.perf_counter() - started
                record.update(
                    targets=f._targets(row, labels, "failed", type(exc).__name__),
                    error_type=type(exc).__name__,
                )
        record["timings"] = {
            "full_original_seconds": elapsed,
            "dicom_hash_seconds": hash_seconds,
            "measured_grading_forward_seconds": grading["seconds"],
            "grading_forward_calls": grading["calls"],
            "common_DICOM_localization_ROI_plus_tensorIO_seconds": max(
                0.0, elapsed - grading["seconds"]
            ),
        }
        record["seconds"] = elapsed
        f._atomic_bytes(path, f._json_bytes(record))
        print(
            f"Extracted held-out {attempted}; assessed {sum(t['status'] == 'ok' for t in record['targets'])}/2 targets",
            flush=True,
        )
    return _aggregate_test(
        output_dir,
        rows,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "manifest-dir", "cache-dir", "output-dir", "selection-manifest"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = extract_heldout_features(**vars(args))
    except (ArchiveDownloadError, ValueError) as exc:
        parser.exit(2, str(exc) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
