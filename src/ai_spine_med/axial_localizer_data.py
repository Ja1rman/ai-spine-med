"""Explicit slice/central-canal localization supervision, separate from grades.

Only AISSLab CCS boxes supervise central-canal x/y. RSNA subarticular points
supervise representative level planes only; they are never canal-center labels.
The slice target is a distance kernel around sparse representative annotations,
not a claim that unannotated slices are anatomically negative.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np

from .assets import sha256_file, write_json
from .axial_features import WEIGHTS_SHA256
from .axial_localization_audit import (
    LEVELS,
    _coordinate_rows,
    _distances,
    _make_anchor,
    _rsna_anchors,
    _xml_ccs_point,
    reconstruct_slice_bag,
)


def canonical_xy(header: dict, x: float, y: float) -> np.ndarray:
    """Mirror the axial decoder's transpose/flips before normalize/resize.

    x is the untransformed DICOM column, y is its row. The returned [x,y]
    indexes the anterior-up, patient-left-on-right image in [0,1].
    """
    rows, columns = header["image_shape"]
    orientation = np.asarray(header["orientation"], dtype=np.float64)
    column_axis, row_axis = orientation[:2].copy(), orientation[3:5].copy()
    norms = np.asarray([np.linalg.norm(column_axis), np.linalg.norm(row_axis)])
    if (
        min(rows, columns) < 2
        or not np.isfinite([x, y, *orientation]).all()
        or min(norms) < 1e-6
        or not (0 <= x <= columns - 1 and 0 <= y <= rows - 1)
    ):
        raise ValueError("Invalid source center coordinate or geometry")
    column_axis /= norms[0]
    row_axis /= norms[1]
    if abs(column_axis[1]) > abs(column_axis[0]):
        x, y = y, x
        rows, columns = columns, rows
        column_axis, row_axis = row_axis, column_axis
    if min(abs(column_axis[0]), abs(row_axis[1])) < math.cos(math.radians(20)):
        raise ValueError("Unsupported in-plane rotation for canonical point")
    if column_axis[0] < 0:
        x = columns - 1 - x
    if row_axis[1] < 0:
        y = rows - 1 - y
    return np.asarray([x / (columns - 1), y / (rows - 1)], dtype=np.float32)


def canonical_geometry(header: dict) -> dict:
    """Image shape/spacing after the unchanged decoder transpose; flips keep size."""
    rows, columns = header["image_shape"]
    row_spacing, column_spacing = header["spacing"]
    orientation = np.asarray(header["orientation"], dtype=np.float64)
    transpose = abs(orientation[1]) > abs(orientation[0])
    if transpose:
        rows, columns = columns, rows
        row_spacing, column_spacing = column_spacing, row_spacing
    return {
        "canonical_shape": [int(rows), int(columns)],
        "canonical_spacing": [float(row_spacing), float(column_spacing)],
        "canonical_scale_mm": [(columns - 1) * column_spacing, (rows - 1) * row_spacing],
    }


def xml_ccs_anchor(root: Path, reference: dict, item: dict) -> dict:
    """Exact annotated slice + physical center + canonical center, no grade target."""
    point = _xml_ccs_point(root, reference, item)
    annotation = ET.parse(root / reference["xml_path"]).getroot()
    obj = annotation.findall("object")[int(reference["object_index"])]
    box = obj.find("bndbox")
    xmin, ymin, xmax, ymax = (float(box.findtext(key)) for key in ("xmin", "ymin", "xmax", "ymax"))
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("A central-canal box must have positive area")
    anchor = _make_anchor(
        item,
        kind="aisslab_xml_ccs_exact_slice",
        level=reference["level"],
        point=point,
    )
    anchor.pop("grade", None)
    anchor["center_xy"] = canonical_xy(item, (xmin + xmax) / 2, (ymin + ymax) / 2).tolist()
    anchor["center_reference"] = "exact_xml_ccs_bbox_center"
    return anchor


def build_targets(slices: list[dict], anchors: list[dict], *, sigma_mm: float = 5.0) -> dict:
    """Return sparse-supervised level distributions and exact central-canal x/y.

    Unknown levels have an all-zero target and a false level_mask. Distances to
    unrelated series/frames remain infinity. The trainer must honor level_mask
    and center_mask; neither missing annotation nor a model output is truth.
    """
    if not slices or not math.isfinite(sigma_mm) or sigma_mm <= 0:
        raise ValueError("Require slices and a positive finite distance bandwidth")
    if len({(s["series_id"], s["instance"]) for s in slices}) != len(slices):
        raise ValueError("Duplicate selected source slices")
    n = len(slices)
    relevance = np.zeros((2, n), dtype=np.float32)
    level_mask = np.zeros(2, dtype=bool)
    centers = np.zeros((2, n, 2), dtype=np.float32)
    center_mask = np.zeros((2, n), dtype=bool)
    distances = np.full((2, n), np.inf, dtype=np.float32)
    support = np.zeros((2, n), dtype=bool)
    for index, level in enumerate(LEVELS):
        subset = [a for a in anchors if a["level"] == level]
        distances[index] = _distances(slices, subset)
        support[index] = np.isfinite(distances[index])
        if support[index].any():
            finite = distances[index, support[index]].astype(np.float64)
            # Subtract the minimum exponent, preserving a stable soft target
            # even when a sparse anchor was omitted by the original 96 cap.
            exponent = -(finite**2) / (2 * sigma_mm**2)
            weights = np.exp(exponent - exponent.max())
            relevance[index, support[index]] = (weights / weights.sum()).astype(np.float32)
            level_mask[index] = True
        for number, item in enumerate(slices):
            exact = [
                a
                for a in subset
                if a["kind"] == "aisslab_xml_ccs_exact_slice" and a["path"] == item["path"]
            ]
            values = [np.asarray(a["center_xy"], dtype=np.float32) for a in exact]
            if values:
                if any(
                    v.shape != (2,) or not np.isfinite(v).all() or np.any((v < 0) | (v > 1))
                    for v in values
                ):
                    raise ValueError("Invalid exact central-canal center")
                if not all(np.allclose(v, values[0], atol=1e-6, rtol=0) for v in values):
                    raise ValueError("Conflicting central-canal center annotations")
                centers[index, number] = values[0]
                center_mask[index, number] = True
    return {
        "relevance": relevance,
        "level_mask": level_mask,
        "center_xy": centers,
        "center_mask": center_mask,
        "anchor_distances_mm": distances,
        "series_support": support,
    }


def _read_json(path: Path):
    return json.loads(path.read_text())


def _safe_relative(root: Path, name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError("Unsafe cached array path")
    target = root / path
    if not target.resolve(strict=True).is_relative_to(root.resolve()) or target.is_symlink():
        raise ValueError("Unsafe cached array path")
    return target


def _aisslab_sources(prepared: Path, cache_index: Path, phase: str) -> list[dict]:
    """AISSLab train/validation only or explicit later test cache, never grades."""
    prepared_rows = _read_json(prepared / "records.json")
    cache = _read_json(cache_index)
    if cache["configuration"]["encoder"]["sha256"] != WEIGHTS_SHA256:
        raise ValueError("AISSLab prefix is not original ImageNet")
    lookup = {(r["dataset"], r["study_id"]): r for r in cache["records"]}
    if len(lookup) != len(cache["records"]):
        raise ValueError("Duplicate activation cache patient")
    allowed = {"train", "validation"} if phase == "fitting" else {"test"}
    result = []
    for row in prepared_rows:
        if row["split"] not in allowed:
            continue
        entry = lookup.get(("aisslab", row["study_id"]))
        if entry is None or entry["status"] != "ok" or entry["split"] != row["split"]:
            raise ValueError("Successful same-split AISSLab activation is missing")
        result.append(
            {
                "dataset": "aisslab",
                "study_id": row["study_id"],
                "split": row["split"],
                "source": row["source"],
                "source_root": str((prepared / "extracted").resolve()),
                "references": row["label_references"],
                "activation_path": str(
                    _safe_relative(cache_index.parent, entry["arrays_path"]).resolve()
                ),
                "activation_sha256": entry["arrays_sha256"],
                "n_slices": entry["n_slices"],
            }
        )
    return result


def _rsna_fitting_sources(cache: Path, data_root: Path) -> tuple[list[dict], list[dict]]:
    metadata = _read_json(cache / "metadata.json")
    if metadata["config"]["encoder"]["sha256"] != WEIGHTS_SHA256:
        raise ValueError("RSNA vectors are not original ImageNet")
    entries = metadata["study_records"]
    if any(entry["split"] not in {"train", "validation"} for entry in entries):
        raise ValueError("TEST cannot enter original RSNA fitting metadata")
    records, failures = [], []
    for entry in entries:
        if entry["status"] != "ok":
            failures.append(
                {
                    "dataset": "rsna",
                    "study_id": entry["study_id"],
                    "split": entry["split"],
                    "status": entry["status"],
                    "reason": entry["reason"],
                }
            )
            continue
        record_path = _safe_relative(cache, entry["record_path"])
        if (
            sha256_file(record_path) != entry["record_sha256"]
            or metadata["outputs_sha256"].get(entry["record_path"]) != entry["record_sha256"]
        ):
            raise ValueError("Original RSNA source receipt changed")
        record = _read_json(record_path)
        if any(
            record[key] != entry[key]
            for key in ("study_id", "split", "status", "arrays_path", "arrays_sha256")
        ):
            raise ValueError("Original RSNA receipt identity differs from metadata")
        if metadata["outputs_sha256"].get(record["arrays_path"]) != record["arrays_sha256"]:
            raise ValueError("Original RSNA feature signature differs from metadata")
        records.append(
            {
                "dataset": "rsna",
                "study_id": record["study_id"],
                "split": record["split"],
                "source": record["source"],
                "source_root": str(data_root.resolve()),
                "feature_path": str(_safe_relative(cache, record["arrays_path"]).resolve()),
                "feature_sha256": record["arrays_sha256"],
            }
        )
    return records, failures


def _original_block(weights: Path, device: str):
    from .axial_features import _load_encoder, _weight_signature

    _weight_signature(weights)
    return _load_encoder(weights, device).layer4[1].eval().requires_grad_(False)


def _features_and_centers(row, selected, full, targets, block, device):
    import torch

    center_indices, center_levels = [], []
    for level, number in zip(*np.nonzero(targets["center_mask"])):
        center_indices.append(int(number))
        center_levels.append(int(level))
    if row["dataset"] == "aisslab":
        path = Path(row["activation_path"])
        if sha256_file(path) != row["activation_sha256"]:
            raise ValueError("AISSLab frozen activation archive changed")
        with np.load(path, allow_pickle=False) as archive:
            positions = archive["positions"].copy()
            activations = archive["activations"].copy()
        np.testing.assert_array_equal(
            positions, np.asarray([s["positions"] for s in selected], dtype=np.float32)
        )
        if len(activations) != len(selected) or activations.shape[1:] != (512, 7, 7):
            raise ValueError("Unexpected original prefix activation shape")
        features, retained_maps = [], {}
        requested = set(center_indices)
        with torch.inference_mode():
            for start in range(0, len(activations), 16):
                maps = (
                    block(torch.from_numpy(activations[start : start + 16]).to(device))
                    .cpu()
                    .numpy()
                )
                features.append(maps.mean(axis=(2, 3)))
                for offset, value in enumerate(maps):
                    if start + offset in requested:
                        retained_maps[start + offset] = value.copy()
        features = np.concatenate(features)
        center_maps = np.asarray(
            [retained_maps[i] for i in center_indices], dtype=np.float32
        ).reshape(-1, 512, 7, 7)
    else:
        path = Path(row["feature_path"])
        if sha256_file(path) != row["feature_sha256"]:
            raise ValueError("Original RSNA frozen feature archive changed")
        with np.load(path, allow_pickle=False) as archive:
            expected = [(s["series_id"], s["instance"]) for s in full]
            actual = list(zip(archive["series_id"].astype(str), archive["instance"].astype(int)))
            if expected != actual:
                raise ValueError("Original RSNA instance/series order differs from header geometry")
            selected_indices = [s["raw_index"] for s in selected]
            features = archive["features"][selected_indices].copy()
            positions = archive["positions"][selected_indices].copy()
        np.testing.assert_array_equal(
            positions, np.asarray([s["positions"] for s in selected], dtype=np.float32)
        )
        center_maps = np.empty((0, 512, 7, 7), dtype=np.float32)
    centers = np.asarray(
        [targets["center_xy"][level, i] for level, i in zip(center_levels, center_indices)],
        dtype=np.float32,
    ).reshape(-1, 2)
    return {
        "features": features.astype(np.float32),
        "positions": positions.astype(np.float32),
        "relevance": targets["relevance"],
        "level_mask": targets["level_mask"],
        "center_maps": center_maps,
        "center_xy": centers,
        "center_indices": np.asarray(center_indices, dtype=np.int64),
        "center_levels": np.asarray(center_levels, dtype=np.int64),
        "center_scale_mm": np.asarray(
            [canonical_geometry(selected[i])["canonical_scale_mm"] for i in center_indices],
            dtype=np.float32,
        ).reshape(-1, 2),
        "anchor_distances_mm": targets["anchor_distances_mm"],
        "series_support": targets["series_support"],
    }


def prepare_localizer_data(
    out: str | Path = "artifacts/axial-localizer-v6",
    *,
    prepared: str | Path = "artifacts/aisslab-axial-v4",
    aisslab_index: str | Path = "artifacts/axial-transfer-fit-v4/index.json",
    rsna_cache: str | Path = "artifacts/axial-features-v3",
    rsna_root: str | Path = "data/rsna",
    weights: str | Path = ".cache/axial-v3/resnet18-f37072fd.pth",
    phase: str = "fitting",
    sigma_mm: float = 5.0,
    device: str = "cpu",
    threads: int = 4,
) -> dict:
    """Assemble original frozen image features and explicit localization truth.

    This entry point is fitting-only: it refuses TEST before medical input is
    opened. A separately sealed evaluation uses prepare_evaluation_data later.
    No stenosis degrees, AP distances or model-generated grades become targets.
    """
    import hashlib

    import torch

    out, prepared, aisslab_index, rsna_cache, rsna_root, weights = map(
        Path, (out, prepared, aisslab_index, rsna_cache, rsna_root, weights)
    )
    if phase != "fitting":
        raise ValueError("This entry point prepares fitting train/validation only")
    torch.set_num_threads(threads)
    if device not in {"cpu", "mps"}:
        raise ValueError("Use cpu or mps for original frozen feature completion")
    rows, failures = _rsna_fitting_sources(rsna_cache, rsna_root)
    rows += _aisslab_sources(prepared, aisslab_index, phase)
    keys = [(r["dataset"], r["study_id"]) for r in rows]
    if len(keys) != len(set(keys)) or any(r["split"] not in {"train", "validation"} for r in rows):
        raise ValueError("Duplicate/cross-split patient or TEST in localization fitting")
    configuration = {
        "schema_version": 1,
        "phase": phase,
        "levels": list(LEVELS),
        "sigma_mm": sigma_mm,
        "aisslab_index_sha256": sha256_file(aisslab_index),
        "aisslab_records_sha256": sha256_file(prepared / "records.json"),
        "rsna_metadata_sha256": sha256_file(rsna_cache / "metadata.json"),
        "rsna_coordinates_sha256": sha256_file(rsna_root / "train_label_coordinates.csv"),
        "original_weights_sha256": sha256_file(weights),
        "implementation_sha256": sha256_file(Path(__file__)),
        "geometry_implementation_sha256": sha256_file(
            Path(__file__).with_name("axial_localization_audit.py")
        ),
        "rsna_center_supervision": False,
        "slice_policy": "unchanged image cache order; original linear maximum 96 slices",
        "rsna_level_policy": "same-frame sagittal central plane when present, otherwise axial subarticular representative plane",
        "center_policy": "exact AISSLab XML CCS bounding-box center; original ImageNet last-block maps",
        "grade_supervision": False,
    }
    out.mkdir(parents=True, exist_ok=True)
    conf_path = out / "configuration.json"
    if conf_path.exists() and _read_json(conf_path) != configuration:
        raise ValueError("Localization data provenance changed; use a new directory")
    write_json(conf_path, configuration)
    coordinates = _coordinate_rows(rsna_root)
    block = _original_block(weights, device)
    results, counts = [], Counter()
    (out / "records").mkdir(exist_ok=True)
    for number, row in enumerate(rows):
        token = hashlib.sha256(f"{row['dataset']}:{row['study_id']}".encode()).hexdigest()[:24]
        target_path = out / "records" / f"{token}.npz"
        record_path = out / "records" / f"{token}.json"
        if record_path.exists():
            result = _read_json(record_path)
            if (
                sha256_file(target_path) != result["arrays_sha256"]
                or result["split"] != row["split"]
            ):
                raise ValueError("Localization cache differs while resuming")
        else:
            selected, full = reconstruct_slice_bag(row)
            for item in selected:
                item.update(canonical_geometry(item))
            if row["dataset"] == "aisslab":
                by_path = {s["path"]: s for s in full}
                anchors = [
                    xml_ccs_anchor(Path(row["source_root"]), ref, by_path[ref["dicom_path"]])
                    for ref in row["references"]
                ]
            else:
                anchors, _ = _rsna_anchors(row, full, coordinates, rsna_root)
                for level in LEVELS:
                    same_frame = [
                        a
                        for a in anchors
                        if a["level"] == level
                        and a["kind"] == "rsna_sagittal_central_same_frame_projection"
                    ]
                    if same_frame:
                        anchors = [a for a in anchors if a["level"] != level] + same_frame
            for anchor in anchors:
                anchor.pop("grade", None)
            targets = build_targets(selected, anchors, sigma_mm=sigma_mm)
            arrays = _features_and_centers(row, selected, full, targets, block, device)
            np.savez_compressed(target_path, **arrays)
            result = {
                "dataset": row["dataset"],
                "study_id": row["study_id"],
                "split": row["split"],
                "arrays_path": str(target_path.relative_to(out)),
                "arrays_sha256": sha256_file(target_path),
                "n_slices": len(selected),
                "level_mask": targets["level_mask"].tolist(),
                "center_target_count": targets["center_mask"].sum(axis=1).tolist(),
                "anchors": anchors,
                "selected_slices": selected,
                "source_root": row["source_root"],
                "source": row["source"],
                "status": "ok",
                "reason": "",
            }
            write_json(record_path, result)
        results.append(result)
        counts[f"{row['dataset']}/{row['split']}/studies"] += 1
        for j, level in enumerate(LEVELS):
            counts[f"{row['dataset']}/{row['split']}/{level}/level_targets"] += int(
                result["level_mask"][j]
            )
            counts[f"{row['dataset']}/{row['split']}/{level}/center_targets"] += int(
                result["center_target_count"][j]
            )
        if (number + 1) % 100 == 0 or number + 1 == len(rows):
            print(f"Explicit localization features/targets {number + 1}/{len(rows)}", flush=True)
    output = {
        "schema_version": 1,
        "configuration": configuration,
        "counts": dict(counts),
        "excluded_original_failures": failures,
        "records": results + failures,
    }
    write_json(out / "index.json", output)
    write_json(
        out / "summary.json",
        {
            "phase": phase,
            "patient_count": len(results),
            "counts": dict(counts),
            "index_sha256": sha256_file(out / "index.json"),
        },
    )
    return output
