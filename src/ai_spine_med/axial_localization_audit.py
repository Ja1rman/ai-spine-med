"""Private, post-hoc anatomy audit of fixed axial attention; never a training input.

XML CCS slices and RSNA axial subarticular coordinates are representative level
anchors, not an exhaustive list of all diagnostically valid slices. Attention is
an explanatory weight, not a validated anatomical segmentation/localizer.
"""

from __future__ import annotations

import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
import pydicom

from .axial_transfer_cache import select_slice_indices, validate_sources

LEVELS = ("L4-L5", "L5-S1")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _uid_hash(value):
    return hashlib.sha256(str(value).encode()).hexdigest() if value else None


def _header(path):
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    orientation = np.asarray(ds.ImageOrientationPatient, dtype=float)
    spacing = np.asarray(ds.PixelSpacing, dtype=float)
    origin = np.asarray(ds.ImagePositionPatient, dtype=float)
    normal = np.cross(orientation[:3], orientation[3:])
    normal /= np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal
    center = (
        origin
        + orientation[:3] * spacing[1] * (int(ds.Columns) - 1) / 2
        + orientation[3:] * spacing[0] * (int(ds.Rows) - 1) / 2
    )
    values = np.concatenate((orientation, spacing, origin, normal, center))
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite header geometry")
    return {
        "instance": int(ds.InstanceNumber),
        "center_lps": center.tolist(),
        "normal": normal.tolist(),
        "frame_sha256": _uid_hash(getattr(ds, "FrameOfReferenceUID", "")),
        "origin_lps": origin.tolist(),
        "orientation": orientation.tolist(),
        "spacing": spacing.tolist(),
        "image_shape": [int(ds.Rows), int(ds.Columns)],
    }


def _normalized_z(slices):
    z = np.asarray([s["center_lps"][2] for s in slices], dtype=np.float64)
    span = float(np.ptp(z))
    return (
        ((z.max() - z) / span).astype(np.float32)
        if span > 1e-6
        else np.full(len(z), 0.5, np.float32)
    )


def reconstruct_slice_bag(source_record):
    """Reproduce the extractor order/96 cap using DICOM headers only.

    The caller compares returned positions against the immutable cache. No
    PixelData is read or decoded. Source paths/checksums remain private.
    """
    root = validate_sources(source_record)
    all_slices = []
    series_positions = {}
    for source in source_record["source"]["series"]:
        slices = []
        for file in source["files"]:
            relative = str(Path(source["series_path"]) / file["name"])
            item = _header(root / relative)
            item.update(series_id=source["series_id"], path=relative)
            slices.append(item)
        slices.sort(key=lambda s: (-s["center_lps"][2], s["instance"]))
        if len({s["instance"] for s in slices}) != len(slices):
            raise ValueError("Duplicate per-series instances")
        for item, position in zip(slices, _normalized_z(slices)):
            series_positions[(item["series_id"], item["instance"])] = float(position)
        all_slices.extend(slices)
    frames = {}
    for item in all_slices:
        frame = item["frame_sha256"] or f"missing:{item['series_id']}"
        frames.setdefault(frame, []).append(item)
    order = sorted(frames, key=lambda frame: min(int(s["series_id"]) for s in frames[frame]))
    ordered = []
    for group, frame in enumerate(order):
        slices = sorted(
            frames[frame],
            key=lambda s: (-s["center_lps"][2], int(s["series_id"]), s["instance"]),
        )
        for item, position in zip(slices, _normalized_z(slices)):
            item["positions"] = [
                float(position),
                series_positions[(item["series_id"], item["instance"])],
            ]
            item["frame_group"] = group
            item["raw_index"] = len(ordered)
            ordered.append(item)
    selected = select_slice_indices(len(ordered))
    return [dict(ordered[int(i)], selected_index=j) for j, i in enumerate(selected)], ordered


def dicom_point(header, x, y):
    """DICOM x=column, y=row: each multiplies its matching direction/spacing."""
    orientation = np.asarray(header["orientation"])
    spacing = header["spacing"]
    return (
        np.asarray(header["origin_lps"])
        + float(x) * spacing[1] * orientation[:3]
        + float(y) * spacing[0] * orientation[3:]
    )


def plane_distance(item, point):
    return abs(float(np.dot(np.asarray(item["normal"]), np.asarray(point) - item["center_lps"])))


def _coordinate_rows(root):
    with (root / "train_label_coordinates.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for row in rows:
        level = row["level"].replace("/", "-")
        if level in LEVELS:
            result.setdefault(row["study_id"], []).append(dict(row, level=level))
    return result


def _make_anchor(item, *, kind, level, point=None, grade=None):
    return {
        "kind": kind,
        "level": level,
        "series_id": item["series_id"],
        "instance": item["instance"],
        "path": item["path"],
        "frame_sha256": item["frame_sha256"],
        "point_lps": item["center_lps"] if point is None else np.asarray(point).tolist(),
        "raw_index": item.get("raw_index"),
        "grade": grade,
    }


def _xml_ccs_point(root, reference, item):
    """Use the explicit central-canal box, not the image/FOV center.

    XML coordinates refer to the untransformed DICOM-sized image. They must be
    mapped before any display canonicalization, resize, or intensity transform.
    """
    annotation = ET.parse(root / reference["xml_path"]).getroot()
    size = annotation.find("size")
    if (
        size is None
        or [int(size.findtext("height")), int(size.findtext("width"))] != item["image_shape"]
    ):
        raise ValueError("XML and DICOM image dimensions differ")
    objects = annotation.findall("object")
    obj = objects[int(reference["object_index"])]
    if (
        obj.findtext("name") != "CCS_" + reference["source_grade"]
        or obj.findtext("level") != reference["level"]
    ):
        raise ValueError("XML object identity differs from prepared CCS reference")
    box = obj.find("bndbox")
    if box is None:
        raise ValueError("CCS anchor has no bounding box")
    xmin, ymin, xmax, ymax = [
        float(box.findtext(name)) for name in ("xmin", "ymin", "xmax", "ymax")
    ]
    rows, columns = item["image_shape"]
    if not (0 <= xmin <= xmax <= columns and 0 <= ymin <= ymax <= rows):
        raise ValueError("CCS bounding box is outside source image")
    return dicom_point(item, (xmin + xmax) / 2, (ymin + ymax) / 2)


def _rsna_anchors(row, slices, coordinates, root):
    by_key = {(s["series_id"], s["instance"]): s for s in slices}
    output = []
    missing = 0
    for coordinate in coordinates.get(row["study_id"], []):
        series, instance = coordinate["series_id"], int(coordinate["instance_number"])
        condition = coordinate["condition"]
        if "Subarticular" in condition:
            item = by_key.get((series, instance))
            if item is None:
                missing += 1
                continue
            output.append(
                _make_anchor(
                    item,
                    kind="rsna_axial_subarticular_approximate_level",
                    level=coordinate["level"],
                    point=dicom_point(item, coordinate["x"], coordinate["y"]),
                )
            )
        elif condition == "Spinal Canal Stenosis":
            path = root / "train_images" / row["study_id"] / series / f"{instance}.dcm"
            if not path.exists():
                missing += 1
                continue
            item = _header(path)
            item.update(series_id=series, path=str(path.relative_to(root)))
            if item["frame_sha256"] and any(
                s["frame_sha256"] == item["frame_sha256"] for s in slices
            ):
                output.append(
                    _make_anchor(
                        item,
                        kind="rsna_sagittal_central_same_frame_projection",
                        level=coordinate["level"],
                        point=dicom_point(item, coordinate["x"], coordinate["y"]),
                    )
                )
    return output, missing


def build_anchor_records(sources_path, index_path, prepared_dir, rsna_root):
    """Return private geometry/anchors aligned exactly to frozen evaluation bags."""
    sources_path, index_path = Path(sources_path), Path(index_path)
    prepared_dir, rsna_root = Path(prepared_dir), Path(rsna_root)
    sources = json.loads(sources_path.read_text())["records"]
    index = json.loads(index_path.read_text())["records"]
    by_key = {(r["dataset"], r["study_id"]): r for r in index}
    external = {r["study_id"]: r for r in json.loads((prepared_dir / "records.json").read_text())}
    coordinates = _coordinate_rows(rsna_root)
    records = []
    for number, row in enumerate(sources):
        selected, full = reconstruct_slice_bag(row)
        cached = by_key[(row["dataset"], row["study_id"])]
        with np.load(index_path.parent / cached["arrays_path"], allow_pickle=False) as archive:
            positions = archive["positions"]
        if cached["n_source_slices"] != len(full) or cached["n_slices"] != len(selected):
            raise ValueError("Header-only slice inventory differs from frozen cache")
        np.testing.assert_array_equal(
            np.asarray([s["positions"] for s in selected], dtype=np.float32), positions
        )
        missing = 0
        if row["dataset"] == "aisslab":
            by_path = {s["path"]: s for s in full}
            anchors = []
            for ref in external[row["study_id"]]["label_references"]:
                if ref["level"] not in LEVELS:
                    continue
                item = by_path.get(ref["dicom_path"])
                if item is None:
                    raise ValueError("Explicit XML anchor lacks exact source DICOM")
                anchors.append(
                    _make_anchor(
                        item,
                        kind="aisslab_xml_ccs_exact_slice",
                        level=ref["level"],
                        point=_xml_ccs_point(Path(row["source_root"]), ref, item),
                        grade=ref["grade"],
                    )
                )
        else:
            anchors, missing = _rsna_anchors(row, full, coordinates, rsna_root)
        records.append(
            {
                "dataset": row["dataset"],
                "study_id": row["study_id"],
                "labels": row["labels"],
                "selected_slices": selected,
                "n_source_slices": len(full),
                "selected_exact_anchors": sum(
                    any(s["path"] == a["path"] for s in selected) for a in anchors
                ),
                "anchors": anchors,
                "missing_coordinate_files_or_series": missing,
                "cache_positions_exact": True,
            }
        )
        if (number + 1) % 50 == 0 or number + 1 == len(sources):
            print(f"Header-only anatomical audit {number + 1}/{len(sources)}", flush=True)
    return {
        "schema_version": 1,
        "scope": "PRIVATE post-hoc frozen TEST anatomy audit; no retraining or selection",
        "sources_sha256": _sha(sources_path),
        "activation_index_sha256": _sha(index_path),
        "aisslab_records_sha256": _sha(prepared_dir / "records.json"),
        "rsna_coordinates_sha256": _sha(rsna_root / "train_label_coordinates.csv"),
        "pixels_read": False,
        "records": records,
    }


def _distances(slices, anchors):
    """Plane distances; direct anchors compare same series, projections same frame."""
    output = np.full(len(slices), np.inf)
    for anchor in anchors:
        for index, item in enumerate(slices):
            allowed = item["series_id"] == anchor["series_id"]
            if anchor["kind"] == "rsna_sagittal_central_same_frame_projection":
                allowed = (
                    anchor["frame_sha256"] is not None
                    and item["frame_sha256"] == anchor["frame_sha256"]
                )
            if allowed:
                output[index] = min(output[index], plane_distance(item, anchor["point_lps"]))
    return output


def localization_metrics(weights, anchor_record):
    """Describe proximity to representative anchors, never clinical correctness.

    5/10 mm windows acknowledge annotation sparsity. Uniform-window mass provides
    a baseline for windows that cover differing portions of the bag. COM z is
    reported only for a single frame, because coordinates cannot cross frames.
    """
    weights = np.asarray(weights, dtype=float)
    slices = anchor_record["selected_slices"]
    if (
        weights.shape != (2, len(slices))
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or not np.allclose(weights.sum(axis=-1), 1, atol=1e-5, rtol=0)
    ):
        raise ValueError("Invalid attention weights")
    output = {}
    for level_index, level in enumerate(LEVELS):
        anchors = [a for a in anchor_record["anchors"] if a["level"] == level]
        kinds = sorted({a["kind"] for a in anchors})
        groups = {kind: [a for a in anchors if a["kind"] == kind] for kind in kinds}
        per_kind = {}
        for kind, subset in groups.items():
            distances = _distances(slices, subset)
            top = int(np.argmax(weights[level_index]))
            top3 = np.argsort(-weights[level_index], kind="stable")[:3]
            finite = np.isfinite(distances)
            per_kind[kind] = {
                "anchor_count": len(subset),
                "top1_distance_mm": float(distances[top]) if finite[top] else None,
                "top1_other_series_or_frame": not bool(finite[top]),
                "top3_min_distance_mm": (
                    float(distances[top3].min()) if np.isfinite(distances[top3]).any() else None
                ),
                **{
                    f"{key}_{window}mm": value
                    for window in (5, 10)
                    for key, value in (
                        ("top1_near", bool(distances[top] <= window)),
                        ("top3_any_near", bool(np.any(distances[top3] <= window))),
                        ("mass_near", float(weights[level_index, distances <= window].sum())),
                        ("uniform_mass_near", float((distances <= window).mean())),
                    )
                },
            }
        w = weights[level_index]
        entropy = -float(np.sum(w * np.log(np.clip(w, 1e-300, None))))
        output[level] = {
            "by_anchor_kind": per_kind,
            "top1_index": int(w.argmax()),
            "max_weight": float(w.max()),
            "effective_slice_count": float(np.exp(entropy)),
            "normalized_entropy": entropy / np.log(len(w)) if len(w) > 1 else 0.0,
            "com_normalized_frame_position": float(
                sum(value * s["positions"][0] for value, s in zip(w, slices))
            )
            if len({s["frame_group"] for s in slices}) == 1
            else None,
        }
    output["head_overlap_mass"] = float(np.minimum(weights[0], weights[1]).sum())
    output["same_top1"] = bool(weights[0].argmax() == weights[1].argmax())
    return output


def run_localization_audit(anchors_path, attention_index_path, out):
    """Audit all six fixed runs and their fixed means; export private details.

    The first three rows must be original seed 42/2024/3407, followed by the
    three corresponding transferred runs. This function never fits parameters.
    """
    anchors_path, attention_index_path, out = (
        Path(anchors_path),
        Path(attention_index_path),
        Path(out),
    )
    anchors = json.loads(anchors_path.read_text())
    attention = json.loads(attention_index_path.read_text())
    lookup = {(r["dataset"], r["study_id"]): r for r in attention["records"]}
    phases = [
        "before_seed_42",
        "before_seed_2024",
        "before_seed_3407",
        "after_seed_42",
        "after_seed_2024",
        "after_seed_3407",
    ]
    if attention.get("entries") != phases:
        raise ValueError("Fixed six-seed attention order differs from audit protocol")
    if len(lookup) != len(attention["records"]) or set(lookup) != {
        (r["dataset"], r["study_id"]) for r in anchors["records"]
    }:
        raise ValueError("Attention and anatomy inventories differ or contain duplicates")
    groups, head_groups, details = {}, {}, []
    for record in anchors["records"]:
        entry = lookup[(record["dataset"], record["study_id"])]
        path = attention_index_path.parent / entry["arrays_path"]
        if _sha(path) != entry["arrays_sha256"]:
            raise ValueError("Fixed attention archive changed")
        with np.load(path, allow_pickle=False) as archive:
            weights, positions = archive["attention"], archive["positions"]
        if weights.shape != (6, 2, len(record["selected_slices"])):
            raise ValueError("Attention inventory differs from anatomical descriptors")
        np.testing.assert_array_equal(
            positions,
            np.asarray([s["positions"] for s in record["selected_slices"]], dtype=np.float32),
        )
        variants = dict(zip(phases, weights))
        variants.update(before_mean=weights[:3].mean(0), after_mean=weights[3:].mean(0))
        for phase, value in variants.items():
            metrics = localization_metrics(value, record)
            details.append(
                {
                    "dataset": record["dataset"],
                    "study_id": record["study_id"],
                    "phase": phase,
                    "metrics": metrics,
                }
            )
            hk = record["dataset"], phase
            head_groups.setdefault(hk, []).append(metrics)
            for level in LEVELS:
                for kind, proximity in metrics[level]["by_anchor_kind"].items():
                    key = record["dataset"], phase, level, kind
                    groups.setdefault(key, []).append(
                        dict(
                            proximity,
                            normalized_entropy=metrics[level]["normalized_entropy"],
                            effective_slice_fraction=metrics[level]["effective_slice_count"]
                            / len(record["selected_slices"]),
                        )
                    )
    aggregated = []
    for (dataset, phase, level, kind), rows in sorted(groups.items()):
        finite = [r["top1_distance_mm"] for r in rows if r["top1_distance_mm"] is not None]
        aggregated.append(
            {
                "dataset": dataset,
                "phase": phase,
                "level": level,
                "anchor_kind": kind,
                "n_anchored_targets": len(rows),
                "top1_other_series_or_frame_unverifiable": sum(
                    r["top1_other_series_or_frame"] for r in rows
                ),
                "median_top1_distance_mm_when_verifiable": float(np.median(finite))
                if finite
                else None,
                **{
                    name + "_count": sum(bool(r[name]) for r in rows)
                    for name in (
                        "top1_near_5mm",
                        "top1_near_10mm",
                        "top3_any_near_5mm",
                        "top3_any_near_10mm",
                    )
                },
                **{
                    name + "_mean": float(np.mean([r[name] for r in rows]))
                    for name in (
                        "mass_near_5mm",
                        "mass_near_10mm",
                        "uniform_mass_near_5mm",
                        "uniform_mass_near_10mm",
                        "normalized_entropy",
                        "effective_slice_fraction",
                    )
                },
            }
        )
    heads = []
    for (dataset, phase), rows in sorted(head_groups.items()):
        orderable = [
            r
            for r in rows
            if all(r[level]["com_normalized_frame_position"] is not None for level in LEVELS)
        ]
        heads.append(
            {
                "dataset": dataset,
                "phase": phase,
                "n_studies": len(rows),
                "head_overlap_mass_mean": float(np.mean([r["head_overlap_mass"] for r in rows])),
                "same_top1_count": sum(r["same_top1"] for r in rows),
                "single_frame_studies": len(orderable),
                "com_L5S1_more_inferior_than_L4L5_count": sum(
                    r["L5-S1"]["com_normalized_frame_position"]
                    > r["L4-L5"]["com_normalized_frame_position"]
                    for r in orderable
                ),
            }
        )
    summary = {
        "schema_version": 1,
        "scope": "PRIVATE post-hoc fixed TEST attention proximity, not clinical accuracy; sparse representative anchors",
        "anchors_sha256": _sha(anchors_path),
        "attention_index_sha256": _sha(attention_index_path),
        "groups": aggregated,
        "head_relationships": heads,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "attention-localization-details.json").write_text(
        json.dumps(details, indent=2, allow_nan=False) + "\n"
    )
    (out / "attention-localization-summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    return summary


def summarize_geometry(records):
    """Aggregate anatomical order without comparing coordinates across series."""
    summary = {
        "n_studies": len(records),
        "exact_cache_positions_all": all(r["cache_positions_exact"] for r in records),
        "anchors_by_kind": dict(Counter(a["kind"] for r in records for a in r["anchors"])),
        "studies_by_dataset": dict(Counter(r["dataset"] for r in records)),
        "unavailable_coordinate_files_or_series": sum(
            r["missing_coordinate_files_or_series"] for r in records
        ),
    }
    order, exceptions = {}, []
    for dataset in sorted({r["dataset"] for r in records}):
        rows = [r for r in records if r["dataset"] == dataset]
        positions = {level: [] for level in LEVELS}
        deltas, failed = [], []
        unpaired = removed = 0
        for record in rows:
            anchors = record["anchors"]
            selected = {(s["series_id"], s["instance"]): s for s in record["selected_slices"]}
            for anchor in anchors:
                item = selected.get((anchor["series_id"], anchor["instance"]))
                if item is None:
                    removed += 1
                else:
                    positions[anchor["level"]].append(item["positions"][0])
            paired = False
            for series in sorted({a["series_id"] for a in anchors}):
                levels = {
                    level: [a for a in anchors if a["series_id"] == series and a["level"] == level]
                    for level in LEVELS
                }
                if not all(levels.values()):
                    continue
                paired = True
                mean_z = {
                    level: float(np.mean([a["point_lps"][2] for a in values]))
                    for level, values in levels.items()
                }
                delta = mean_z["L4-L5"] - mean_z["L5-S1"]
                deltas.append(delta)
                if delta <= 0:
                    failed.append(
                        {
                            "study_id": record["study_id"],
                            "series_id": series,
                            "delta_mm": delta,
                        }
                    )
            unpaired += not paired
        order[dataset] = {
            "n_studies": len(rows),
            "same_series_level_pairs": len(deltas),
            "L4_L5_above_L5_S1": sum(z > 0 for z in deltas),
            "L4_L5_not_above_L5_S1": len(failed),
            "unpaired_studies": unpaired,
            "median_L4minusL5_z_mm": float(np.median(deltas)) if deltas else None,
            "normalized_frame_position_by_level": {
                level: {
                    "n": len(values),
                    "median": float(np.median(values)) if values else None,
                    "range": [float(np.min(values)), float(np.max(values))] if values else None,
                }
                for level, values in positions.items()
            },
            "anchor_slices_removed_by_96cap": removed,
        }
        exceptions.extend(dict(dataset=dataset, **row) for row in failed)
    return summary, order, exceptions


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        type=Path,
        default=Path("artifacts/axial-transfer-sources-v4/evaluation-sources.json"),
    )
    parser.add_argument(
        "--index", type=Path, default=Path("artifacts/axial-transfer-test-v4/index.json")
    )
    parser.add_argument("--prepared", type=Path, default=Path("artifacts/aisslab-axial-v4"))
    parser.add_argument("--rsna-root", type=Path, default=Path("data/rsna"))
    parser.add_argument(
        "--attention-index",
        type=Path,
        default=Path("reports/runs/axial-diagnosis-v5/attention/index.json"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("reports/runs/axial-diagnosis-v5/localization")
    )
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    anchors = build_anchor_records(args.sources, args.index, args.prepared, args.rsna_root)
    outputs = {
        "anchors.json": anchors,
        **dict(
            zip(
                (
                    "geometry-summary.json",
                    "anatomical-order-summary.json",
                    "anatomical-order-exceptions.json",
                ),
                summarize_geometry(anchors["records"]),
            )
        ),
    }
    for name, value in outputs.items():
        (args.out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    run_localization_audit(args.out / "anchors.json", args.attention_index, args.out)
    print(
        json.dumps(
            {"status": "complete", "n_studies": len(anchors["records"]), "out": str(args.out)}
        )
    )


if __name__ == "__main__":
    main()
