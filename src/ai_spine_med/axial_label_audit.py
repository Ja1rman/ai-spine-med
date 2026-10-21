"""Read-only audit of AISSLab CCS labels, anatomy order and frozen predictions.

The public dataset describes three classes and an AP measurement, but does not
provide a validated RSNA crosswalk: https://data.mendeley.com/datasets/wjxkrkxhb8/1
RSNA uses modified Lee grades, then joins normal and mild (Appendix S3):
https://pubs.rsna.org/doi/suppl/10.1148/ryai.250480/suppl_file/ryai250480suppa1.pdf

Outputs contain private study identifiers. The default destination is ignored by
Git. This audit never changes annotations, training, checkpoints or primary scores.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pydicom
from sklearn.metrics import confusion_matrix, f1_score

from .aisslab import ALL_LEVELS, GRADE_MAP, LEVELS, MAX_XML_BYTES


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(root: Path, relative: str) -> Path:
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Unsafe prepared-data path")
    return path


def verified_prepared_file(root: Path, relative: str, receipts: dict) -> Path:
    """Read only the pinned extracted bytes, rather than trusting an edited copy."""
    path = _inside(root, relative)
    receipt = receipts.get(relative)
    if (
        receipt is None
        or path.stat().st_size != receipt["bytes"]
        or _sha(path) != receipt["sha256"]
    ):
        raise ValueError("Prepared XML/DICOM bytes differ from extraction receipt")
    return path


def parse_ccs_xml(content: bytes) -> list[dict]:
    """Read explicit CCS only; keep AP separate from the anatomical ROI box."""
    if len(content) > MAX_XML_BYTES or b"<!" in content or b"\x00" in content:
        raise ValueError("Unsafe annotation XML")
    root = ET.fromstring(content)
    if root.tag != "annotation":
        raise ValueError("Expected annotation root")
    result = []
    for index, obj in enumerate(root.findall("object")):
        name = (obj.findtext("name") or "").strip()
        if not name.startswith("CCS_"):
            continue
        suffix = name.removeprefix("CCS_")
        level = (obj.findtext("level") or "").strip()
        if suffix not in GRADE_MAP or level not in ALL_LEVELS:
            raise ValueError("Unknown CCS class or level")
        ap = float((obj.findtext("ap_diameter") or "").removesuffix("mm"))
        shape = [int(root.findtext("size/height")), int(root.findtext("size/width"))]
        box = [float(obj.findtext(f"bndbox/{k}")) for k in ("xmin", "ymin", "xmax", "ymax")]
        if not math.isfinite(ap) or ap < 0:
            raise ValueError("Invalid AP measurement")
        if not (0 <= box[0] < box[2] <= shape[1] and 0 <= box[1] < box[3] <= shape[0]):
            raise ValueError("Invalid CCS bounding box")
        filename = root.findtext("filename") or ""
        if "/" in filename or "\\" in filename or not filename:
            raise ValueError("Invalid source image filename")
        result.append(
            {
                "object_index": index,
                "source_class": name,
                "source_level": level,
                "grade": GRADE_MAP[suffix],
                "ap_mm": ap,
                "bbox_xyxy": box,
                "xml_shape": shape,
                "image_filename": filename,
            }
        )
    return result


def empirical_ap_grade(ap_mm: float) -> int:
    """Observed partition, not a clinical rule; exact boundaries need clarification."""
    if not math.isfinite(ap_mm) or ap_mm < 0:
        raise ValueError("Invalid AP measurement")
    return 2 if ap_mm < 10 else (1 if ap_mm < 12 else 0)


def aggregate_targets(references: list[dict]) -> list[int | None]:
    """Mirror the importer: absence is unknown, never an implicit normal grade."""
    return [
        max((row["grade"] for row in references if row["level"] == level), default=None)
        for level in LEVELS
    ]


def bbox_center_lps(box: list[float], ds) -> list[float]:
    """DICOM plane point at the ROI center, including obliquity and pixel spacing."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    orientation = list(map(float, ds.ImageOrientationPatient))
    spacing = list(map(float, ds.PixelSpacing))
    origin = list(map(float, ds.ImagePositionPatient))
    return [
        origin[i] + orientation[i] * spacing[1] * cx + orientation[i + 3] * spacing[0] * cy
        for i in range(3)
    ]


def ap_summary(rows: list[dict]) -> dict:
    result = {}
    for suffix in GRADE_MAP:
        name = "CCS_" + suffix
        values = [r["ap_mm"] for r in rows if r["source_class"] == name]
        result[name] = {
            "n": len(values),
            "min": min(values, default=None),
            "median": statistics.median(values) if values else None,
            "max": max(values, default=None),
            "zero_values": values.count(0),
        }
    return result


def _scores(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No scored frozen predictions")
    targets = [int(r["label"]) for r in rows]
    predictions = [int(r["prediction"]) for r in rows]
    matrix = confusion_matrix(targets, predictions, labels=[0, 1, 2])
    return {
        "n": len(rows),
        "macro_f1": f1_score(targets, predictions, labels=[0, 1, 2], average="macro"),
        "severe_detected": int(matrix[2, 2]),
        "severe_references": int(matrix[2].sum()),
        "confusion_matrix": matrix.tolist(),
    }


def _sensitivity(run: Path, zero_rows: list[dict], records: list[dict]) -> dict:
    zero_keys = {(r["study_id"], r["source_level"]) for r in zero_rows}
    expected = {
        (r["study_id"], level): r["labels"][i]
        for r in records
        if r["split"] == "test"
        for i, level in enumerate(LEVELS)
    }
    result = {}
    for variant in ("before", "after"):
        path = run / "evaluation" / f"{variant}_ensemble.csv"
        with path.open(newline="") as stream:
            rows = [r for r in csv.DictReader(stream) if r["dataset"] == "aisslab"]
        keys = [(r["study_id"], r["level"]) for r in rows]
        if len(keys) != len(set(keys)) or set(keys) != set(expected):
            raise ValueError("Frozen predictions differ from prepared TEST targets")
        for row, key in zip(rows, keys, strict=True):
            label = int(row["label"]) if row["label"] else None
            if label != expected[key]:
                raise ValueError("Frozen reference labels differ")
        scored = [r for r in rows if r["label"] and r["status"] == "ok"]
        subset = [r for r in scored if (r["study_id"], r["level"]) not in zero_keys]
        result[variant] = {
            "file_sha256": _sha(path),
            "metrics": {
                "all_known": _scores(scored),
                "excluding_ap_zero_exploratory": _scores(subset),
            },
            "ap_zero_predictions": [r for r in scored if (r["study_id"], r["level"]) in zero_keys],
        }
    return result


def audit_labels(prepared: Path, archive: Path, run: Path) -> dict:
    """Check saved reference labels and header geometry without decoding pixels."""
    manifest = json.loads((prepared / "manifest.json").read_text())
    records_path = prepared / "records.json"
    if _sha(records_path) != manifest["output_sha256"]["records.json"]:
        raise ValueError("Prepared records checksum differs")
    files_path = prepared / "extracted_files.json"
    if _sha(files_path) != manifest["output_sha256"]["extracted_files.json"]:
        raise ValueError("Extraction receipts checksum differs")
    receipts = {r["path"]: r for r in json.loads(files_path.read_text())}
    archive_sha = _sha(archive)
    if archive_sha != manifest["configuration"]["archive_sha256"]:
        raise ValueError("Pinned source archive checksum differs")
    records = json.loads(records_path.read_text())
    extracted = prepared / "extracted"
    with zipfile.ZipFile(archive) as source:
        members = {n.replace("\\", "/").rstrip("/") for n in source.namelist()}
    roots = {n.split("/", 1)[0] for n in members}
    if len(roots) != 1:
        raise ValueError("Ambiguous archive root")
    archive_root = roots.pop()
    all_rows, target_rows, mismatches, duplicates, missing = [], [], [], [], []
    shape_mismatches, order_pairs, xml_ccs_counts, bbox_ap = [], [], Counter(), []
    for record in records:
        study = record["study_id"]
        patient = study.removeprefix("aisslab-")
        series = record["source"]["series"]
        prefix = Path(series[0]["series_path"]).parent.parent
        files = {f["name"]: Path(s["series_path"]) / f["name"] for s in series for f in s["files"]}
        annotations = []
        for path in sorted((extracted / prefix / "annotations").glob("*.xml")):
            verified_prepared_file(extracted, str(path.relative_to(extracted)), receipts)
            parsed = parse_ccs_xml(path.read_bytes())
            xml_ccs_counts[len(parsed)] += 1
            for annotation in parsed:
                member = (
                    f"{archive_root}/DICOM/{patient}/{Path(annotation['image_filename']).stem}.dcm"
                )
                name = hashlib.sha256(member.encode()).hexdigest() + ".dcm"
                if member not in members or name not in files:
                    raise ValueError("Source XML/DICOM link cannot be restored")
                dicom_path = verified_prepared_file(extracted, str(files[name]), receipts)
                ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
                if annotation["xml_shape"] != [int(ds.Rows), int(ds.Columns)]:
                    shape_mismatches.append(
                        {"study_id": study, "level": annotation["source_level"]}
                    )
                box = annotation["bbox_xyxy"]
                height_mm = (box[3] - box[1]) * float(ds.PixelSpacing[0])
                row = {
                    **annotation,
                    "xml_path": str(path.relative_to(extracted)),
                    "study_id": study,
                    "split": record["split"],
                    "physical_ccs_center_lps": bbox_center_lps(box, ds),
                    "bbox_height_mm": height_mm,
                }
                annotations.append(row)
                all_rows.append(row)
        if aggregate_targets(record["label_references"]) != record["labels"]:
            mismatches.append({"study_id": study, "reason": "record_aggregation_differs"})
        references = {(r["xml_path"], r["object_index"]): r for r in annotations}
        for i, level in enumerate(LEVELS):
            refs = [r for r in record["label_references"] if r["level"] == level]
            if not refs:
                missing.append({"study_id": study, "level": level, "split": record["split"]})
            if len(refs) > 1:
                duplicates.append({"study_id": study, "level": level})
            for ref in refs:
                original = references.get((ref["xml_path"], ref["object_index"]))
                if (
                    original is None
                    or original["source_level"] != level
                    or original["grade"] != ref["grade"]
                ):
                    mismatches.append(
                        {"study_id": study, "level": level, "reason": "ref_xml_differs"}
                    )
                    continue
                target_rows.append(original)
                bbox_ap.append(abs(original["ap_mm"] - original["bbox_height_mm"]))
            actual_grades = [r["grade"] for r in annotations if r["source_level"] == level]
            if record["labels"][i] != max(actual_grades, default=None):
                mismatches.append(
                    {"study_id": study, "level": level, "reason": "xml_grade_differs"}
                )
        by_level = defaultdict(list)
        for row in annotations:
            by_level[row["source_level"]].append(row)
        present = [level for level in ALL_LEVELS if level in by_level]
        for superior, inferior in zip(present[:-1], present[1:], strict=True):
            # This release has one CCS per level. Median stays defined for other releases.
            z_high = statistics.median(r["physical_ccs_center_lps"][2] for r in by_level[superior])
            z_low = statistics.median(r["physical_ccs_center_lps"][2] for r in by_level[inferior])
            order_pairs.append(
                {
                    "study_id": study,
                    "superior_level": superior,
                    "inferior_level": inferior,
                    "z_diff_mm": z_high - z_low,
                }
            )
    zero_rows = [r for r in target_rows if r["ap_mm"] == 0]
    threshold_errors = [r for r in all_rows if empirical_ap_grade(r["ap_mm"]) != r["grade"]]
    return {
        "schema_version": 1,
        "purpose": "Post-hoc diagnostic audit; no fit, selection or changed primary metrics",
        "input_sha256": {
            "manifest.json": _sha(prepared / "manifest.json"),
            "records.json": _sha(records_path),
            "extracted_files.json": _sha(files_path),
            "archive": archive_sha,
        },
        "counts": {
            "patients": len(records),
            "xml": sum(xml_ccs_counts.values()),
            "ccs_by_level": dict(Counter(r["source_level"] for r in all_rows)),
            "target_annotations": len(target_rows),
        },
        "import_integrity": {
            "record_label_mismatches": mismatches,
            "duplicate_target_annotations": duplicates,
            "ccs_per_xml": dict(xml_ccs_counts),
            "xml_dicom_shape_mismatches": shape_mismatches,
            "missing_target_labels": missing,
        },
        "ap_mm_grade_correspondence": {
            "all_levels": ap_summary(all_rows),
            "target_levels": ap_summary(target_rows),
            "all_levels_by_split": {
                split: ap_summary([r for r in all_rows if r["split"] == split])
                for split in ("train", "validation", "test")
            },
            "target_by_split_level": {
                split: {
                    level: ap_summary(
                        [
                            r
                            for r in target_rows
                            if r["split"] == split and r["source_level"] == level
                        ]
                    )
                    for level in LEVELS
                }
                for split in ("train", "validation", "test")
            },
            "matches": len(all_rows) - len(threshold_errors),
            "mismatches": threshold_errors,
            "exact_boundary_counts": {
                f"{boundary}mm": sum(r["ap_mm"] == boundary for r in all_rows)
                for boundary in (10, 12)
            },
            "interpretation": "Observed AP partition, not source-proven derivation or RSNA equivalence. Boundary inclusion cannot be inferred if exact10/12 values are absent.",
        },
        "bbox_ap_are_distinct": {
            "target_count": len(target_rows),
            "height_times_spacing_close_to_ap_within_0_05mm": sum(v <= 0.05 for v in bbox_ap),
            "median_absolute_difference_mm": statistics.median(bbox_ap) if bbox_ap else None,
        },
        "ap_zero_cases_private": zero_rows,
        "all_ap_zero_cases_private": [r for r in all_rows if r["ap_mm"] == 0],
        "sensitivity_only": _sensitivity(run, zero_rows, records),
        "all_five_levels_internal_order": {
            "linked_ccs": len(all_rows),
            "ordered_pairs": len(order_pairs),
            "reversed_or_equal_pairs": [r for r in order_pairs if r["z_diff_mm"] <= 0],
            "minimum_superior_offset_mm": min((r["z_diff_mm"] for r in order_pairs), default=None),
            "median_superior_offset_mm": statistics.median(r["z_diff_mm"] for r in order_pairs)
            if order_pairs
            else None,
            "pair_counts_by_level_pair": dict(
                Counter(r["superior_level"] + " > " + r["inferior_level"] for r in order_pairs)
            ),
            "interpretation": "Internal consistency of physical ROI centers, not absolute clinical numbering.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("artifacts/aisslab-axial-v4"))
    parser.add_argument(
        "--archive", type=Path, default=Path("data/external/aisslab-axial-v1/source.zip")
    )
    parser.add_argument("--run", type=Path, default=Path("reports/runs/axial-transfer-v4"))
    parser.add_argument(
        "--out", type=Path, default=Path("reports/runs/axial-diagnosis-v5/label-reproduce")
    )
    args = parser.parse_args()
    result = audit_labels(args.prepared, args.archive, args.run)
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / "label-audit.json"
    with destination.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "out": str(destination),
                "counts": result["counts"],
                "label_mismatches": len(result["import_integrity"]["record_label_mismatches"]),
            }
        )
    )


if __name__ == "__main__":
    main()
