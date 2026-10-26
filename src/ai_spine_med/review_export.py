"""Export the full fixed RSNA validation cohort for private physician review.

All predictions are the frozen-head v1 ensemble; no model fitting or TEST access.
Original DICOM bytes remain exact, with a resumable allowlisted ZIP export.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image, ImageDraw, ImageFont

from .assets import sha256_file, write_json
from .rsna_bulk import COMPLETE_MARKER, is_complete_series

CLASSES = ("normal_mild", "moderate", "severe")
GRADES = ("Норма / лёгкий", "Умеренный", "Тяжёлый")
SEEDS = (42, 2024, 3407)


def digest_bytes(content):
    return hashlib.sha256(content).hexdigest()


def key(prefix, text):
    return prefix + "-" + digest_bytes(text.encode())[:16]


atomic_json = write_json


def hardlink_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != sha256_file(source):
            raise ValueError("existing_export_asset_changed")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def vector(ds, name, length):
    try:
        values = np.asarray(getattr(ds, name), dtype=float)
        if values.shape == (length,) and np.isfinite(values).all():
            return values.tolist()
    except (AttributeError, TypeError, ValueError):
        pass
    return None


def canonical_display(image, orientation, sequence, spacing):
    """Display supported flips/transposes; preserve native raster otherwise."""
    receipt = {
        "orientation_status": "native_acquisition",
        "transpose": False,
        "flip_rows": False,
        "flip_columns": False,
    }
    if orientation is None:
        return image, spacing, receipt
    col, row = np.asarray(orientation[:3]), np.asarray(orientation[3:])
    horizontal, vertical = (0, 1) if sequence == "Axial T2" else (1, 2)
    col2, row2 = col[[horizontal, vertical]].copy(), row[[horizontal, vertical]].copy()
    norms = np.linalg.norm(col2), np.linalg.norm(row2)
    if min(norms) < 1e-6:
        return image, spacing, receipt
    col2 /= norms[0]
    row2 /= norms[1]
    transpose = abs(col2[1]) > abs(col2[0])
    if transpose:
        col2, row2 = row2, col2
    if abs(col2[0]) < math.cos(math.radians(20)) or abs(row2[1]) < math.cos(math.radians(20)):
        return image, spacing, receipt
    if transpose:
        image = image.T
        spacing = spacing[::-1] if spacing else None
    flip_columns = col2[0] < 0
    flip_rows = row2[1] < 0 if sequence == "Axial T2" else row2[1] > 0
    if flip_columns:
        image = image[:, ::-1]
    if flip_rows:
        image = image[::-1, :]
    receipt.update(
        orientation_status="canonical_flips_transpose_only",
        transpose=bool(transpose),
        flip_rows=bool(flip_rows),
        flip_columns=bool(flip_columns),
    )
    return np.ascontiguousarray(image), spacing, receipt


def decode_preview(content, sequence):
    ds = pydicom.dcmread(io.BytesIO(content))
    if int(getattr(ds, "SamplesPerPixel", 1)) != 1 or int(getattr(ds, "NumberOfFrames", 1)) != 1:
        raise ValueError("unsupported_color_or_multiframe")
    if "Modality" in ds and str(ds.Modality) != "MR":
        raise ValueError("non_MR")
    photo = str(ds.PhotometricInterpretation)
    if photo not in {"MONOCHROME1", "MONOCHROME2"}:
        raise ValueError("unsupported_photometric")
    slope, intercept = (
        float(getattr(ds, "RescaleSlope", 1)),
        float(getattr(ds, "RescaleIntercept", 0)),
    )
    if not np.isfinite([slope, intercept]).all() or slope == 0:
        raise ValueError("invalid_rescale")
    image = ds.pixel_array.astype(np.float32) * slope + intercept
    rows, columns = int(ds.Rows), int(ds.Columns)
    if image.ndim != 2 or image.shape != (rows, columns) or not np.isfinite(image).all():
        raise ValueError("invalid_pixel_array")
    orientation, position, spacing = (
        vector(ds, "ImageOrientationPatient", 6),
        vector(ds, "ImagePositionPatient", 3),
        vector(ds, "PixelSpacing", 2),
    )
    if spacing is not None and min(spacing) <= 0:
        spacing = None
    center = None
    if orientation and position and spacing:
        center = (
            np.asarray(position)
            + np.asarray(orientation[:3]) * spacing[1] * (columns - 1) / 2
            + np.asarray(orientation[3:]) * spacing[0] * (rows - 1) / 2
        ).tolist()
    image, display_spacing, transform = canonical_display(image, orientation, sequence, spacing)
    return image, {
        "instance_number": int(ds.InstanceNumber),
        "rows": rows,
        "columns": columns,
        "photometric_interpretation": photo,
        "rescale_slope": slope,
        "rescale_intercept": intercept,
        "pixel_spacing_mm": spacing,
        "image_orientation_patient": orientation,
        "image_position_patient": position,
        "image_center_lps_mm": center,
        "display_pixel_spacing_mm": display_spacing,
        "display_transform": transform,
        "missing_Modality": "Modality" not in ds,
    }


def image_from_float(image, lower, upper, spacing=None, edge=512):
    raster = Image.fromarray(
        np.rint(np.clip((image - lower) / max(upper - lower, 1e-12), 0, 1) * 255).astype(np.uint8)
    )
    height, width = image.shape
    physical_h, physical_w = (
        (height * spacing[0], width * spacing[1]) if spacing else (height, width)
    )
    target = min(edge, max(height, width))
    shape = (
        max(1, round(target * physical_w / max(physical_h, physical_w))),
        max(1, round(target * physical_h / max(physical_h, physical_w))),
    )
    return raster.resize(shape, Image.Resampling.LANCZOS) if raster.size != shape else raster


def render_series(patient, series, *, source_root, output_root, selection_sha, previous_root=None):
    ROOT, BASE = Path(source_root), Path(output_root)
    SITE = BASE / "site"
    EXPECTED_SELECTION = selection_sha
    study, sid = patient["study_id"], series["series_id"]
    series_key = key("series", f"{study}/{sid}")
    receipt_path = BASE / "export-records" / f"{series_key}.json"
    directory = (ROOT / series["relative_directory"]).resolve()
    expected_directory = ROOT / "data/rsna/train_images" / study / sid
    if directory != expected_directory.resolve() or not study.isdigit() or not sid.isdigit():
        raise ValueError("source_path_escape")
    if not is_complete_series(directory):
        return None
    marker_sha = sha256_file(directory / COMPLETE_MARKER)
    if (
        series.get("completion_receipt_sha256")
        and marker_sha != series["completion_receipt_sha256"]
    ):
        raise ValueError("source_completion_receipt_changed")
    files = sorted(directory.glob("*.dcm"), key=lambda p: int(p.stem))
    source = []
    expected = {row["name"]: row for row in series.get("files", [])}
    for path in files:
        content = path.read_bytes()
        digest = digest_bytes(content)
        old = expected.get(path.name, {})
        if old.get("expected_sha256_from_feature_cache") not in {None, digest} or old.get(
            "sha256"
        ) not in {None, digest}:
            raise ValueError("frozen_source_SHA_changed")
        if old.get("bytes") not in {None, len(content)}:
            raise ValueError("source_size_changed")
        source.append({"name": path.name, "sha256": digest, "bytes": len(content)})
    if expected and {row["name"] for row in source} != set(expected):
        raise ValueError("frozen_source_inventory_changed")
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text())
        if old["source_files"] != source or old["selection_file_sha256"] != EXPECTED_SELECTION:
            raise ValueError("existing_export_source_changed")
        if all(
            (SITE / relative).is_file() and sha256_file(SITE / relative) == digest
            for relative, digest in old["output_sha256"].items()
        ):
            return old
        raise ValueError("existing_export_corrupted")
    if previous_root is not None:
        old_path = Path(previous_root) / "export-records" / f"{series_key}.json"
        if old_path.is_file():
            old = json.loads(old_path.read_text())
            if old["source_files"] != source:
                raise ValueError("reused_export_source_changed")
            for relative, expected_sha in old["output_sha256"].items():
                src = Path(previous_root) / "site" / relative
                if not src.is_file() or sha256_file(src) != expected_sha:
                    raise ValueError("reused_export_corrupted")
                hardlink_or_copy(src, SITE / relative)
            old = {**old, "selection_file_sha256": EXPECTED_SELECTION}
            atomic_json(receipt_path, old)
            return old
    image_directory = SITE / "images" / series_key
    image_directory.mkdir(parents=True, exist_ok=True)
    zip_path = SITE / "originals" / f"{series_key}.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_zip = zip_path.with_suffix(".zip.tmp")
    decoded, slices, routes, failures = [], [], {}, []
    with zipfile.ZipFile(
        temporary_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True
    ) as archive:
        for file in source:
            path = directory / file["name"]
            content = path.read_bytes()
            if digest_bytes(content) != file["sha256"]:
                raise ValueError("source_changed_during_export")
            relative_source = str(path.relative_to(ROOT))
            slice_key = key("slice", relative_source)
            member = f"{slice_key}.dcm"
            info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
            route = f"dicom/{series_key}/{slice_key}.dcm"
            routes[route] = {
                "zip_path": f"site/originals/{series_key}.zip",
                "member": member,
                "sha256": file["sha256"],
                "bytes": file["bytes"],
            }
            item = {
                "slice_key": slice_key,
                "sha256": file["sha256"],
                "bytes": file["bytes"],
                "source_filename": file["name"],
                "dicom_url": route,
                "image_url": None,
                "preview_status": "failed",
                "instance_number": int(path.stem),
            }
            try:
                image, header = decode_preview(content, series["series_description"])
                item.update(header, preview_status="ok", preview_error="")
                decoded.append((item, image))
            except Exception as error:
                item["preview_error"] = type(error).__name__
                failures.append({"slice_key": slice_key, "reason": type(error).__name__})
            slices.append(item)
    temporary_zip.replace(zip_path)
    photos = {item["photometric_interpretation"] for item, _ in decoded}
    if len(photos) > 1:
        raise ValueError("inconsistent_series_photometric")
    inverted = photos == {"MONOCHROME1"}
    if decoded:
        if inverted:
            lo = min(float(image.min()) for _, image in decoded)
            hi = max(float(image.max()) for _, image in decoded)
            decoded = [(item, lo + hi - image) for item, image in decoded]
        lower, upper = np.percentile(
            np.concatenate([image.ravel() for _, image in decoded]), [1, 99]
        ).tolist()
    else:
        lower, upper = 0.0, 1.0
    output_sha = {f"originals/{series_key}.zip": sha256_file(zip_path)}
    for item, image in decoded:
        raster = image_from_float(image, lower, upper, item["display_pixel_spacing_mm"])
        relative = f"images/{series_key}/{item['slice_key']}.jpg"
        raster.save(SITE / relative, "JPEG", quality=88, optimize=True)
        item.update(
            image_url=relative,
            image_sha256=sha256_file(SITE / relative),
            preview_rows=raster.height,
            preview_columns=raster.width,
        )
        output_sha[relative] = item["image_sha256"]

    def sort_slice(item):
        center = item.get("image_center_lps_mm")
        spatial = (
            (center[2] * -1 if series["series_description"] == "Axial T2" else center[0])
            if center
            else 0
        )
        return (0, spatial, item["instance_number"]) if center else (1, item["instance_number"], 0)

    slices.sort(key=sort_slice)
    instance = [item["instance_number"] for item in slices]
    if len(set(instance)) != len(instance):
        raise ValueError("duplicate_instance_within_export_series")
    public_series = {
        "series_key": series_key,
        "series_description": series["series_description"],
        "series_id": sid,
        "study_id": study,
        "n_slices": len(slices),
        "n_preview_slices": len(decoded),
        "zip_url": f"originals/{series_key}.zip",
        "zip_sha256": output_sha[f"originals/{series_key}.zip"],
        "slices": slices,
        "display_window": {
            "method": "one 1st/99th percentile window for whole series after rescale and common MONOCHROME1 inversion",
            "lower": lower,
            "upper": upper,
            "monochrome1_inverted": inverted,
        },
        "preview_note": "Lossy full-FOV preview; use original DICOM for diagnostic review. Acquisition-plane obliquity does not filter slices.",
    }
    record = {
        "schema_version": 1,
        "selection_file_sha256": EXPECTED_SELECTION,
        "patient_id": patient["patient_id"],
        "series": public_series,
        "source_files": source,
        "completion_receipt_sha256": marker_sha,
        "routes": routes,
        "output_sha256": output_sha,
        "failures": failures,
    }
    atomic_json(receipt_path, record)
    return record


def font(size):
    for path in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def target_key(row):
    return row["study_id"], row["series_id"], row["level"]


def verified_predictions(root):
    """Verify the exported mean against every fixed seed and cached reference."""
    run = root / "reports/runs/ensemble-v1-validation"
    summary = json.loads((run / "summary.json").read_text())
    if summary["seeds"] != list(SEEDS):
        raise ValueError("unexpected_ensemble_seeds")
    seed_rows = {}
    for seed in SEEDS:
        path = root / f"reports/runs/head-v1/runs/seed-{seed}/predictions.csv"
        if sha256_file(path) != summary["input_prediction_sha256"][str(seed)]:
            raise ValueError("seed_predictions_changed")
        seed_rows[seed] = {target_key(row): row for row in read_csv(path)}
    metadata_path = root / "artifacts/features-v1/metadata.json"
    metadata = json.loads(metadata_path.read_text())
    manifest_path = root / "artifacts/features-v1/target_manifest.csv"
    if sha256_file(manifest_path) != metadata["outputs_sha256"]["target_manifest.csv"]:
        raise ValueError("feature_manifest_changed")
    official = root / "data/rsna/train_series_descriptions.csv"
    if sha256_file(official) != metadata["config"]["source_metadata"][official.name]:
        raise ValueError("official_series_metadata_changed")
    expected = {
        target_key(row): row for row in read_csv(manifest_path) if row["split"] == "validation"
    }
    rows = read_csv(run / "predictions.csv")
    if len(rows) != len(expected) or {target_key(row) for row in rows} != set(expected):
        raise ValueError("ensemble_cohort_changed")
    for row in rows:
        k = target_key(row)
        if row["split"] != "validation":
            raise ValueError("validation_only_export_required")
        for field in ("label", "status", "reason"):
            if any(row[field] != seed_rows[seed][k][field] for seed in SEEDS):
                raise ValueError("seed_target_metadata_changed")
            if row[field] != expected[k][field]:
                raise ValueError("cached_target_metadata_changed")
        if row["status"] == "ok":
            mean = np.mean(
                [[float(seed_rows[seed][k]["p_" + c]) for c in CLASSES] for seed in SEEDS],
                axis=0,
            )
            actual = np.array([float(row["p_" + c]) for c in CLASSES])
            if (
                not np.isfinite(actual).all()
                or not np.allclose(mean, actual, rtol=0, atol=1e-7)
                or int(row["prediction"]) != int(mean.argmax())
            ):
                raise ValueError("ensemble_predictions_changed")
    return rows, {
        "predictions_sha256": sha256_file(run / "predictions.csv"),
        "summary_sha256": sha256_file(run / "summary.json"),
        "feature_manifest_sha256": sha256_file(manifest_path),
        "feature_metadata_sha256": sha256_file(metadata_path),
        "seed_predictions_sha256": summary["input_prediction_sha256"],
    }


def build_cases(rows, previous):
    """Preserve all existing opaque case/patient IDs and visible case numbers."""
    old_cases = {row["case_id"]: row for row in previous.get("cases", [])}
    old_order = {row["case_id"]: row["review_order"] for row in old_cases.values()}
    new_rows = []
    for row in rows:
        k = target_key(row)
        label = int(row["label"]) if row["label"] else None
        prediction = int(row["prediction"]) if row["prediction"] else None
        assessed = row["status"] == "ok"
        wrong = assessed and label is not None and prediction != label
        case_id = "case-mm-" + hashlib.sha256(f"private:{k}".encode()).hexdigest()[:14]
        patient_id = (
            "patient-" + hashlib.sha256(f"private-full-review-v3:{k[0]}".encode()).hexdigest()[:14]
        )
        if case_id in old_cases and old_cases[case_id]["patient_id"] != patient_id:
            raise ValueError("existing_patient_identity_changed")
        new_rows.append(
            {
                "case_id": case_id,
                "patient_id": patient_id,
                "study_id": k[0],
                "sagittal_series_id": k[1],
                "level": k[2],
                "split": "validation",
                "label": label,
                "reference": None if label is None else CLASSES[label],
                "status": row["status"],
                "processing_reason": row["reason"],
                "correct_control": assessed and label is not None and prediction == label,
                "selection_flags": {
                    "any_wrong": wrong,
                    "sagittal_wrong": wrong,
                    "disagreement": False,
                    "not_assessed": not assessed,
                },
                "predictions": {
                    "sagittal": {
                        "status": row["status"],
                        "prediction": prediction,
                        "probabilities": [float(row["p_" + c]) for c in CLASSES]
                        if assessed
                        else None,
                    }
                },
                "case_url": f"?case={case_id}",
            }
        )
    if not set(old_cases).issubset({row["case_id"] for row in new_rows}):
        raise ValueError("existing_cases_missing_from_full_cohort")
    new_rows.sort(
        key=lambda row: (
            row["case_id"] not in old_order,
            old_order.get(row["case_id"], 0),
            row["case_id"],
        )
    )
    for number, row in enumerate(new_rows, 1):
        row["review_order"] = number
    return new_rows


def source_inventory(root, cases, previous):
    studies = {row["study_id"] for row in cases}
    descriptions = [
        row
        for row in read_csv(root / "data/rsna/train_series_descriptions.csv")
        if row["study_id"] in studies
    ]
    expected = {}
    feature_metadata = json.loads((root / "artifacts/features-v1/metadata.json").read_text())
    for study in sorted(studies):
        sid = next(row["sagittal_series_id"] for row in cases if row["study_id"] == study)
        record = json.loads(
            (root / f"artifacts/features-v1/records/{study}_{sid}.json").read_text()
        )
        if record.get("cache_id") != feature_metadata["cache_id"]:
            raise ValueError("feature_record_cache_changed")
        expected[(study, sid)] = record["source"]["files"]
        axial = json.loads((root / f"artifacts/axial-features-v3/records/{study}.json").read_text())
        for entry in axial["source"]["series"]:
            expected[(study, entry["series_id"])] = entry["files"]
    old_order = [row["patient_id"] for row in previous.get("patients", [])]
    patients = []
    for study in sorted(studies):
        selected = [row for row in cases if row["study_id"] == study]
        patient_id = selected[0]["patient_id"]
        series = []
        for row in sorted(
            (s for s in descriptions if s["study_id"] == study), key=lambda s: int(s["series_id"])
        ):
            sid = row["series_id"]
            directory = root / "data/rsna/train_images" / study / sid
            complete = is_complete_series(directory)
            if row["series_description"] != "Sagittal T1" and not complete:
                raise ValueError("required_T2_series_not_complete")
            marker = directory / COMPLETE_MARKER
            series.append(
                {
                    "series_id": sid,
                    "series_description": row["series_description"],
                    "relative_directory": str(directory.relative_to(root)),
                    "available": complete,
                    "completion_receipt_sha256": sha256_file(marker) if complete else None,
                    "files": [
                        {
                            "name": f["name"],
                            "bytes": f["bytes"],
                            "expected_sha256_from_feature_cache": f["sha256"],
                        }
                        for f in expected.get((study, sid), [])
                    ],
                }
            )
        patients.append(
            {
                "patient_id": patient_id,
                "study_id": study,
                "case_ids": [row["case_id"] for row in selected],
                "source_series": series,
            }
        )
    positions = {patient_id: i for i, patient_id in enumerate(old_order)}
    patients.sort(
        key=lambda p: (
            p["patient_id"] not in positions,
            positions.get(p["patient_id"], 0),
            p["patient_id"],
        )
    )
    return patients


def model_input_layers(record, sagittal, level):
    """Mirror SpineNet's round(median) per vertebra, then round(mean) rule."""
    pair = set(level.split("-"))
    positions = [
        int(np.round(np.median(v["slice_nos"])))
        for v in record.get("vertices", [])
        if v["label"] in pair and v.get("slice_nos")
    ]
    if len(positions) != 2:
        raise ValueError("model_input_level_pair_missing")
    middle = int(np.round(np.mean(positions)))
    slices = list(sagittal["slices"])
    orientation = slices[0].get("image_orientation_patient")
    if orientation is None or any(s.get("image_position_patient") is None for s in slices):
        raise ValueError("model_input_source_geometry_missing")
    normal = np.cross(orientation[:3], orientation[3:])
    normal = normal if normal[0] > 0 else -normal
    slices.sort(key=lambda s: np.dot(s["image_position_patient"], normal))
    result = []
    for layer in range(9):
        offset = middle + layer - 4
        item = slices[offset] if 0 <= offset < len(slices) else None
        result.append(
            {
                "roi_layer": layer + 1,
                "slice_key": item["slice_key"] if item else None,
                "padding": item is None,
            }
        )
    return result


def render_overview(root, output, case, series):
    """Show every actual model-input layer, not a single illustrative slice."""
    record = json.loads(
        (
            root
            / "artifacts/features-v1/records"
            / f"{case['study_id']}_{case['sagittal_series_id']}.json"
        ).read_text()
    )
    matching = next(t for t in record["targets"] if t["level"] == case["level"])
    links = []
    roi = None
    if case["status"] == "ok":
        feature_root = (root / "artifacts/features-v1").resolve()
        arrays_path = (feature_root / record["arrays_path"]).resolve()
        if (
            not arrays_path.is_relative_to(feature_root)
            or sha256_file(arrays_path) != record["arrays_sha256"]
        ):
            raise ValueError("feature_ROI_integrity_changed")
        with np.load(arrays_path, allow_pickle=False) as data:
            roi = data["rois"][int(matching["roi_index"])].copy()
        if roi.shape != (9, 112, 224):
            raise ValueError("unexpected_model_ROI_shape")
    sagittal = next(s for s in series if s["series_id"] == case["sagittal_series_id"])
    layer_sources = model_input_layers(record, sagittal, case["level"]) if roi is not None else []
    source_map = {s["slice_key"]: s for s in sagittal["slices"]}
    for layer, layer_source in enumerate(layer_sources):
        item = source_map.get(layer_source["slice_key"])
        if item is not None:
            links.append(
                {
                    "label": f"Вход модели {layer + 1}/9 · исходный срез {item['instance_number']}",
                    "url": f"?case={case['case_id']}&series={sagittal['series_key']}&slice={item['slice_key']}",
                    "series_key": sagittal["series_key"],
                    "slice_key": item["slice_key"],
                    "source_sha256": item["sha256"],
                }
            )
    canvas = Image.new("RGB", (744, 555), "#161919")
    draw = ImageDraw.Draw(canvas)
    answer = case["predictions"]["sagittal"]["prediction"]
    draw.text(
        (15, 10), f"{case['level']} · RSNA: {GRADES[case['label']]}", fill="white", font=font(18)
    )
    draw.text(
        (15, 37),
        "Сагиттальный ансамбль v1: " + (GRADES[answer] if answer is not None else "не оценено"),
        fill="white",
        font=font(17),
    )
    if roi is not None:
        lower, upper = float(roi.min()), float(roi.max())
        for i in range(9):
            x, y = 15 + (i % 3) * 245, 85 + (i // 3) * 150
            canvas.paste(image_from_float(roi[i], lower, upper, edge=224), (x, y))
            draw.text(
                (x, y + 115),
                f"Слой {i + 1}/9" + (" · дополнение нулями" if layer_sources[i]["padding"] else ""),
                fill="white",
                font=font(14),
            )
    else:
        draw.text(
            (15, 100),
            "Область уровня не получена. Откройте полную серию.",
            fill="white",
            font=font(17),
        )
        item = sagittal["slices"][len(sagittal["slices"]) // 2]
        if item.get("image_url"):
            image = Image.open(output / "site" / item["image_url"]).convert("RGB")
            image.thumbnail((700, 360))
            canvas.paste(image, (15, 150))
        links = [
            {
                "label": "Сагиттальная T2 · контекст, модель не оценила уровень",
                "url": f"?case={case['case_id']}&series={sagittal['series_key']}&slice={item['slice_key']}",
                "series_key": sagittal["series_key"],
                "slice_key": item["slice_key"],
            }
        ]
        layer_sources = []
    relative = f"overviews/{case['case_id']}.jpg"
    path = output / "site" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, "JPEG", quality=86, optimize=True)
    return {
        **case,
        "overview_url": relative,
        "overview_sha256": sha256_file(path),
        "view_links": links,
        "model_input": {
            "series_key": sagittal["series_key"],
            "roi_shape": [9, 112, 224],
            "layers": layer_sources,
            "all_series_slices_used_for_severity": False,
        },
    }


def export_review(root, output, previous_root):
    root, output, previous_root = (
        Path(root).resolve(),
        Path(output).resolve(),
        Path(previous_root).resolve(),
    )
    previous = json.loads((previous_root / "site/data/index.json").read_text())
    rows, provenance = verified_predictions(root)
    cases = build_cases(rows, previous)
    patients = source_inventory(root, cases, previous)
    selection = {
        "schema_version": 1,
        "scope": "full_fixed_validation_cohort",
        "model": "frozen-head-v1-ensemble",
        "seeds": list(SEEDS),
        "provenance": provenance,
        "cases": cases,
        "patients": patients,
        "test_accessed": False,
        "clinical_review_performed": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    selection_path = output / "selection.json"
    atomic_json(selection_path, selection)
    selection_sha = sha256_file(selection_path)
    records, pending = {}, []
    for patient in patients:
        for series in patient["source_series"]:
            if series["available"]:
                pending.append((patient, series))
    for number, (patient, series) in enumerate(pending, 1):
        record = render_series(
            patient,
            series,
            source_root=root,
            output_root=output,
            selection_sha=selection_sha,
            previous_root=previous_root,
        )
        records[record["series"]["series_key"]] = record
        if number % 20 == 0 or number == len(pending):
            print(f"Exported {number}/{len(pending)} series", flush=True)
    public_patients, final_cases = [], []
    for patient in patients:
        all_series = [
            records[key("series", patient["study_id"] + "/" + s["series_id"])]["series"]
            for s in patient["source_series"]
            if s["available"]
        ]
        missing = [
            {"series_description": s["series_description"], "reason": "not_downloaded"}
            for s in patient["source_series"]
            if not s["available"]
        ]
        atomic_json(
            output / "site/data/patients" / f"{patient['patient_id']}.json",
            {
                "schema_version": 1,
                "patient_id": patient["patient_id"],
                "case_ids": patient["case_ids"],
                "series": all_series,
                "unavailable_series": missing,
            },
        )
        public_patients.append(
            {
                "patient_id": patient["patient_id"],
                "case_ids": patient["case_ids"],
                "series_ids": [s["series_key"] for s in all_series],
            }
        )
        for case in cases:
            if case["patient_id"] == patient["patient_id"]:
                final_cases.append(render_overview(root, output, case, all_series))
    final_cases.sort(key=lambda case: case["review_order"])
    aggregate = {
        "n_validation_targets": len(cases),
        "n_selected_cases": len(cases),
        "n_selected_patients": len(patients),
        "n_correct": sum(c["correct_control"] for c in cases),
        "n_errors": sum(c["selection_flags"]["any_wrong"] for c in cases),
        "n_unassessed": sum(c["status"] != "ok" for c in cases),
        "n_available_series": len(records),
        "n_unavailable_series": sum(
            not s["available"] for p in patients for s in p["source_series"]
        ),
    }
    index = {
        "schema_version": 1,
        "scope": "full_fixed_validation_cohort",
        "aggregate": aggregate,
        "clinical_review_performed": False,
        "test_accessed": False,
        "selection_file_sha256": selection_sha,
        "model": {
            "id": "frozen-head-v1-ensemble",
            "title": "Сагиттальный ансамбль · SpineNet + голова v1",
            "seeds": list(SEEDS),
            "input_policy": "Сагиттальная T2-серия → область из 9 соседних срезов на уровень",
            "levels": ["L4-L5", "L5-S1"],
            "provenance": provenance,
        },
        "cases": final_cases,
        "patients": public_patients,
        "preview_note": "Полные доступные поясничные серии. Вход классификатора — ROI9, а не все срезы серии.",
    }
    atomic_json(output / "site/data/index.json", index)
    atomic_json(
        output / "private-source-index.json",
        {
            "schema_version": 1,
            "selection_file_sha256": selection_sha,
            "routes": {
                route: value for r in records.values() for route, value in r["routes"].items()
            },
        },
    )
    files = [p for p in (output / "site").rglob("*") if p.is_file()]
    old_asset_names = {
        str(p.relative_to(previous_root / "site"))
        for p in (previous_root / "site").rglob("*")
        if p.is_file()
    }
    result = {
        **aggregate,
        "status": "complete",
        "source_bytes": sum(f["bytes"] for r in records.values() for f in r["source_files"]),
        "output_bytes": sum(p.stat().st_size for p in files),
        "new_asset_bytes": sum(
            p.stat().st_size
            for p in files
            if str(p.relative_to(output / "site")) not in old_asset_names
        ),
        "replaced_asset_bytes": sum(
            p.stat().st_size
            for p in files
            if str(p.relative_to(output / "site")).startswith(("overviews/", "data/"))
        ),
        "n_original_DICOM": sum(r["series"]["n_slices"] for r in records.values()),
        "n_previews": sum(r["series"]["n_preview_slices"] for r in records.values()),
        "decode_failures": [f for r in records.values() for f in r["failures"]],
        "provenance": provenance,
        "test_accessed": False,
    }
    atomic_json(output / "export-summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path, default=Path("reports/runs/physician-review-all-v13")
    )
    parser.add_argument(
        "--previous", type=Path, default=Path("reports/runs/physician-review-full-v3")
    )
    args = parser.parse_args()
    result = export_review(args.root, args.output, args.previous)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "provenance"}, ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
