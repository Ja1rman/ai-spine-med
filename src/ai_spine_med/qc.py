"""Private technical ROI contact sheets; never a clinical/anatomical sign-off."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np

from .assets import sha256_file, write_json
from .baseline import isotropic_inplane
from .data import CLASS_NAMES, LEVELS
from .dicom import load_sagittal_series


def select_qc_targets(records: list[dict], count: int = 24, seed: int = 42) -> list[dict]:
    """Round-robin severity/level groups; prefer distinct training studies."""
    if count < 1:
        raise ValueError("QC count must be positive")
    groups = {(label, level): [] for label in range(3) for level in LEVELS}
    for record in records:
        for target in record["targets"]:
            if target["split"] != "train" or target["status"] != "ok":
                continue
            key = target["label"], target["level"]
            if key not in groups:
                raise ValueError("Unexpected target label or level")
            groups[key].append({**target, "record": record})
    for candidates in groups.values():
        candidates.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}:{row['study_id']}:{row['series_id']}:{row['level']}".encode()
            ).hexdigest()
        )
    selected, used_keys, used_studies = [], set(), set()
    while len(selected) < count:
        added = False
        for candidates in groups.values():
            available = [
                row
                for row in candidates
                if (row["study_id"], row["series_id"], row["level"]) not in used_keys
            ]
            if not available:
                continue
            item = next(
                (row for row in available if row["study_id"] not in used_studies), available[0]
            )
            selected.append(item)
            used_keys.add((item["study_id"], item["series_id"], item["level"]))
            used_studies.add(item["study_id"])
            added = True
            if len(selected) == count:
                break
        if not added:
            break
    return selected


def _safe_array_path(feature_dir: Path, relative: str) -> Path:
    path = (feature_dir / relative).resolve()
    if not path.is_relative_to(feature_dir.resolve()):
        raise ValueError("ROI cache path escapes feature directory")
    return path


def _display(ax, image, title):
    low, high = np.percentile(image, [1, 99])
    ax.imshow(image, cmap="gray", vmin=low, vmax=high if high > low else low + 1)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def generate_qc(
    feature_dir: str | Path,
    data_root: str | Path,
    out: str | Path,
    *,
    count: int = 24,
    seed: int = 42,
    per_page: int = 6,
) -> dict:
    """Read committed per-series records, even while other series are running.

    Saved PNGs contain actual MRI pixels and stay in an ignored local directory.
    Rendering uses fixed display percentiles only; cached model ROIs never change.
    """
    if not 1 <= per_page <= 8:
        raise ValueError("per_page must be between 1 and 8")
    feature_dir, data_root, out = map(Path, (feature_dir, data_root, out))
    metadata = json.loads((feature_dir / "metadata.json").read_text())
    records = []
    for path in sorted((feature_dir / "records").glob("*.json")):
        record = json.loads(path.read_text())
        if record["cache_id"] != metadata["cache_id"]:
            raise ValueError("QC record belongs to another feature cache")
        records.append(record)
    selected = select_qc_targets(records, count, seed)
    if not selected:
        raise ValueError("No successful training ROIs are ready for technical QC")
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    review, pages = [], []
    for page_index in range(math.ceil(len(selected) / per_page)):
        subset = selected[page_index * per_page : (page_index + 1) * per_page]
        figure, axes = plt.subplots(
            len(subset),
            4,
            figsize=(16, 2.9 * len(subset)),
            squeeze=False,
            gridspec_kw={"width_ratios": [1.0, 1.35, 1.35, 1.35]},
        )
        for row_index, item in enumerate(subset):
            number = page_index * per_page + row_index + 1
            record = item["record"]
            array_path = _safe_array_path(feature_dir, item["roi_path"])
            if sha256_file(array_path) != record["arrays_sha256"]:
                raise ValueError("QC found a corrupted per-series ROI cache")
            with np.load(array_path, allow_pickle=False) as arrays:
                roi = arrays["rois"][item["roi_index"]]
                probabilities = arrays["original_probs"][item["roi_index"]]
                if str(arrays["levels"][item["roi_index"]]) != item["level"]:
                    raise ValueError("QC ROI index/level mismatch")
            valid = roi.shape == (9, 112, 224) and bool(np.isfinite(roi).all())
            if not valid:
                raise ValueError("QC found an invalid saved ROI")
            label_name = CLASS_NAMES[item["label"]]
            for column, slice_index in enumerate((2, 4, 6), start=1):
                _display(
                    axes[row_index, column], roi[slice_index], f"ROI slice {slice_index + 1}/9"
                )
            axes[row_index, 1].set_title(
                f"#{number:02d} {item['level']} | RSNA: {label_name}\nROI slice 3/9", fontsize=10
            )
            vertices = record["vertices"]
            pair_names = set(item["level"].split("-"))
            names = {vertex["label"] for vertex in vertices}
            target_vertices = [vertex for vertex in vertices if vertex["label"] in pair_names]
            medians = [np.median(vertex["slice_nos"]) for vertex in target_vertices]
            source_status, source_error, source_slice = "ok", "", None
            try:
                source_directory = data_root / "train_images" / item["study_id"] / item["series_id"]
                if any(
                    sha256_file(source_directory / file["name"]) != file["sha256"]
                    for file in record["source"]["files"]
                ):
                    raise ValueError("Source MRI changed after feature extraction")
                source = load_sagittal_series(source_directory)
                volume, _spacing, _resampling = isotropic_inplane(
                    source.volume, source.pixel_spacing
                )
                source_slice = (
                    int(round(float(np.mean(medians)))) if medians else volume.shape[-1] // 2
                )
                source_slice = min(max(source_slice, 0), volume.shape[-1] - 1)
                _display(
                    axes[row_index, 0],
                    volume[:, :, source_slice],
                    f"Source slice {source_slice + 1}/{volume.shape[-1]}",
                )
                for vertex in vertices:
                    polygon = np.asarray(vertex["average_polygon"])
                    color = "#44ff99" if vertex["label"] in pair_names else "#ffd54f"
                    axes[row_index, 0].add_patch(
                        Polygon(
                            polygon,
                            closed=True,
                            fill=False,
                            edgecolor=color,
                            linewidth=0.8,
                            linestyle="--",
                        )
                    )
                    axes[row_index, 0].text(
                        float(polygon[:, 0].mean()),
                        float(polygon[:, 1].mean()),
                        vertex["label"],
                        color=color,
                        fontsize=8,
                        ha="center",
                    )
            except (ValueError, OSError, IndexError) as exc:
                source_status, source_error = "failed", type(exc).__name__
                axes[row_index, 0].text(
                    0.5,
                    0.5,
                    f"Source display failed\n{type(exc).__name__}",
                    ha="center",
                    va="center",
                    transform=axes[row_index, 0].transAxes,
                )
                axes[row_index, 0].axis("off")
            review.append(
                {
                    "case": number,
                    "study_id": item["study_id"],
                    "series_id": item["series_id"],
                    "split": "train",
                    "level": item["level"],
                    "label": item["label"],
                    "label_name": label_name,
                    "roi_path": item["roi_path"],
                    "roi_index": item["roi_index"],
                    "roi_finite": valid,
                    "roi_zero_fraction": float(np.mean(roi == 0)),
                    "source_display_status": source_status,
                    "source_display_error": source_error,
                    "source_slice_index": source_slice,
                    "target_pair_in_predicted_chain": pair_names.issubset(names),
                    "original_prediction_3class": int(probabilities.argmax()),
                    "technical_review": "pending_visual_review",
                    "clinical_review": "not_performed",
                    "anatomical_numbering_confirmed_by_expert": False,
                }
            )
        figure.suptitle(
            "Automatic SpineNet ROIs — private TRAIN technical QC\nDashed boxes: mean detector polygons; green: target pair. No expert anatomical/clinical validation.",
            fontsize=12,
        )
        figure.tight_layout(rect=(0, 0.01, 1, 0.96), h_pad=2.5)
        path = out / f"page-{page_index + 1:02d}.png"
        figure.savefig(path, dpi=140)
        plt.close(figure)
        pages.append(path.name)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(review[0]))
    writer.writeheader()
    writer.writerows(review)
    (out / "technical_review.csv").write_text(stream.getvalue())
    failed = [
        target
        for record in records
        for target in record["targets"]
        if target["split"] == "train" and target["status"] not in {"ok", "missing_label"}
    ]
    summary = {
        "cache_id": metadata["cache_id"],
        "selection_seed": seed,
        "selected_targets": len(review),
        "available_successful_train_targets": sum(
            target["split"] == "train" and target["status"] == "ok"
            for record in records
            for target in record["targets"]
        ),
        "by_level_and_grade": dict(
            Counter(f"{row['level']}:{row['label_name']}" for row in review)
        ),
        "pages": pages,
        "train_failures_so_far": len(failed),
        "scope": "Technical image/ROI inspection only. Expert anatomical numbering and clinical correctness remain unverified. Images must remain local, never published with source code.",
    }
    write_json(out / "summary.json", summary)
    write_json(out / "train_failures.json", failed)
    return summary
