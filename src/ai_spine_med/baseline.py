"""Run the original SpineNet baseline; no training or accuracy claims."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .assets import EXAMPLES, SPINENET_COMMIT, source_dir, verify_assets, write_json

TARGET_LEVELS = ("L4-L5", "L5-S1")
CCS_NAMES = {1: "none", 2: "mild", 3: "moderate", 4: "severe"}


def environment() -> dict:
    import torch

    packages = {}
    for name in (
        "ai-spine-med",
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "pydicom",
        "scipy",
        "scikit-image",
        "opencv-python-headless",
        "shapely",
        "kaggle",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "kaggle_credentials_present": bool(
            os.getenv("KAGGLE_API_TOKEN")
            or os.getenv("KAGGLE_KEY")
            or (Path.home() / ".kaggle" / "kaggle.json").is_file()
            or (Path.home() / ".kaggle" / "access_token").is_file()
            or (Path.home() / ".kaggle" / "credentials.json").is_file()
        ),
    }


def import_upstream(cache: Path):
    verify_assets(cache)
    upstream = source_dir(cache).resolve()
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    # Matplotlib's default user cache is outside the isolated project.
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    import spinenet

    if not Path(spinenet.__file__).resolve().is_relative_to(upstream):
        raise ValueError("A different spinenet installation is already imported")
    return spinenet


def isotropic_inplane(volume: np.ndarray, spacing: tuple[float, float]):
    """The original detector actually requires a scalar, isotropic pixel spacing."""
    from scipy.ndimage import affine_transform

    row, col = map(float, spacing)
    target = min(row, col)
    if np.isclose(row, col, rtol=1e-4):
        return volume, target, {"resampled": False, "pixel_spacing_mm": target}
    shape = (
        round((volume.shape[0] - 1) * row / target) + 1,
        round((volume.shape[1] - 1) * col / target) + 1,
        volume.shape[2],
    )
    # Map output centres directly into input physical coordinates. Rounding the
    # field-of-view boundary never changes the requested output spacing.
    result = affine_transform(
        volume,
        np.diag([target / row, target / col, 1.0]),
        offset=0,
        output_shape=shape,
        order=1,
        mode="nearest",
        prefilter=False,
    )
    return (
        result.astype(np.float32),
        target,
        {
            "resampled": True,
            "original_pixel_spacing_mm": [row, col],
            "pixel_spacing_mm": target,
            "output_shape": list(result.shape),
            "last_center_extent_delta_mm": [
                (shape[0] - 1) * target - (volume.shape[0] - 1) * row,
                (shape[1] - 1) * target - (volume.shape[1] - 1) * col,
            ],
        },
    )


def run_smoke(
    *,
    cache: Path,
    out: Path,
    examples_root: Path | None = None,
    dicom_dir: Path | None = None,
    device: str = "cpu",
    threads: int = 4,
) -> dict:
    import torch

    if (examples_root is None) == (dicom_dir is None):
        raise ValueError("Choose exactly one of --examples-root and --dicom-dir")
    if threads < 1:
        raise ValueError("--threads must be positive")
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError("This baseline supports cpu or cuda[:N]; MPS is not verified")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is not available; use --device cpu")
    if out.exists():
        raise ValueError("Output directory already exists; choose a new run directory")
    out.mkdir(parents=True)
    torch.set_num_threads(threads)
    spinenet = import_upstream(cache)
    start = time.perf_counter()
    # The author checkpoints contain numpy.float64 loss/accuracy metadata.
    # Keep PyTorch's restricted unpickler and allow only these specific NumPy
    # scalar types, scoped to loading the four verified original checkpoints.
    with torch.serialization.safe_globals(
        [
            np.core.multiarray.scalar,
            np.dtype,
            type(np.dtype(np.float64)),
        ]
    ):
        model = spinenet.SpineNet(device=device, verbose=True, scan_type="lumbar")
    model_loading_seconds = time.perf_counter() - start
    sources = (
        [(name, examples_root / name) for name in EXAMPLES]
        if examples_root
        else [("case_1", dicom_dir)]
    )
    summaries = []
    target_rows = []
    for case_id, directory in sources:
        print(f"Running original SpineNet: {case_id}", flush=True)
        timing = time.perf_counter()
        details = {"case_id": case_id}
        case_rows = []
        try:
            if examples_root:
                # Exactly the authors' quickstart. These demo files have missing
                # metadata. Never use the fake geometry override on actual RSNA.
                from spinenet.io import load_dicoms_from_folder

                scan = load_dicoms_from_folder(
                    str(directory),
                    require_extensions=False,
                    metadata_overwrites={
                        "SliceThickness": [2],
                        "ImageOrientationPatient": [0, 1, 0, 0, 0, -1],
                    },
                )
                volume, spacing = scan.volume, float(scan.pixel_spacing)
                details["input"] = {
                    "mode": "official_author_demo_with_synthetic_geometry",
                    "shape": list(volume.shape),
                    "pixel_spacing_mm": spacing,
                    "geometry_override": "Authors' tutorial: SliceThickness=2, IOP=[0,1,0,0,0,-1]",
                }
            else:
                from .dicom import load_sagittal_series

                scan = load_sagittal_series(directory)
                volume, spacing, resampling = isotropic_inplane(scan.volume, scan.pixel_spacing)
                details["input"] = {**scan.metadata, **resampling}
            loaded_at = time.perf_counter()
            with torch.inference_mode():
                vertebrae = model.detect_vb(volume, spacing)
                detected_at = time.perf_counter()
                ivds = model.get_ivds_from_vert_dicts(vertebrae, volume)
                extracted_at = time.perf_counter()
                if not ivds:
                    raise ValueError("SpineNet produced no intervertebral disc volumes")
                grades = model.grade_ivds(ivds)
            grades.index.name = "level"
            grades.to_csv(out / f"{case_id}_all_original_grades.csv")
            write_json(
                out / f"{case_id}_detections.json",
                [
                    {
                        "label": str(item.get("predicted_label", "")),
                        "slice_count": len(item.get("slice_nos", [])),
                    }
                    for item in vertebrae
                ],
            )
            for level in TARGET_LEVELS:
                row = {
                    "case_id": case_id,
                    "level": level,
                    "status": "not_assessed",
                    "spinenet_ccs_4grade": None,
                    "spinenet_ccs_name": None,
                }
                if level in grades.index:
                    grade = int(grades.loc[level, "CentralCanalStenosis"])
                    if grade not in CCS_NAMES:
                        raise ValueError("Original central stenosis grade outside 1..4")
                    row.update(
                        status="ok", spinenet_ccs_4grade=grade, spinenet_ccs_name=CCS_NAMES[grade]
                    )
                case_rows.append(row)
            details.update(
                status="ok" if all(row["status"] == "ok" for row in case_rows) else "partial",
                vertebra_count=len(vertebrae),
                ivd_count=len(ivds),
                ivd_shapes={item["level_name"]: list(item["volume"].shape) for item in ivds},
                seconds={
                    "load": loaded_at - timing,
                    "detect_and_number": detected_at - loaded_at,
                    "extract_ivds": extracted_at - detected_at,
                    "grade_ivds": time.perf_counter() - extracted_at,
                },
            )
        except Exception as error:
            # Failures are retained as not_assessed, never converted into normal.
            details.update(status="failed", error_type=type(error).__name__)
            print(f"{case_id}: failed ({type(error).__name__}): {error}", file=sys.stderr)
            case_rows = [
                {
                    "case_id": case_id,
                    "level": level,
                    "status": "not_assessed",
                    "spinenet_ccs_4grade": None,
                    "spinenet_ccs_name": None,
                }
                for level in TARGET_LEVELS
            ]
        details["total_seconds"] = time.perf_counter() - timing
        summaries.append(details)
        target_rows.extend(case_rows)
        write_json(out / "cases.json", summaries)
    pd.DataFrame(target_rows).to_csv(out / "target_levels.csv", index=False)
    write_json(out / "target_levels.json", target_rows)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "upstream_commit": SPINENET_COMMIT,
        "environment": environment(),
        "device": device,
        "threads": threads,
        "model_loading_seconds": model_loading_seconds,
        "assets": json.loads((cache / "assets.json").read_text()),
        "cases": summaries,
        "successes": sum(case["status"] == "ok" for case in summaries),
        "target_levels_assessed": sum(row["status"] == "ok" for row in target_rows),
        "scope": "Startup smoke only, not validation. No reference labels or accuracy metrics.",
        "grade_scale": "Original SpineNet CCS 1=none,2=mild,3=moderate,4=severe; no RSNA scale conversion yet.",
    }
    write_json(out / "summary.json", summary)
    return summary
