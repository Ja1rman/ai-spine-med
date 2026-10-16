"""Prepare a reproducible, study-grouped research cohort from official RSNA CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
from pathlib import Path
from typing import Any

LEVELS = {"L4-L5": "spinal_canal_stenosis_l4_l5", "L5-S1": "spinal_canal_stenosis_l5_s1"}
CLASS_NAMES = ("normal_mild", "moderate", "severe")
METADATA_FILES = ("train.csv", "train_series_descriptions.csv", "train_label_coordinates.csv")
SPLITS = ("train", "validation", "test")
_ID = re.compile(r"^[0-9]+$")


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(
            f"Missing metadata file: {path.name}. Download all three official CSVs first."
        )
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path.name}: missing columns: {', '.join(sorted(missing))}")
        return [
            {key: (value or "").strip() for key, value in row.items() if key is not None}
            for row in reader
        ]


def _identifier(value: str, field: str) -> str:
    if not _ID.fullmatch(value):
        raise ValueError(
            f"Invalid {field}: expected a numeric identifier, not a path or decimal number."
        )
    return value


def _grade(value: str) -> int | None:
    if value.strip().lower() in ("", "nan", "na", "null", "none"):
        return None
    name = value.strip().lower().replace("/", "_").replace(" ", "_")
    try:
        return CLASS_NAMES.index(name)
    except ValueError as exc:
        raise ValueError(
            "Unexpected severity label; expected Normal/Mild, Moderate, Severe, or an empty value."
        ) from exc


def _level(value: str) -> str:
    return value.upper().replace("/", "-").replace("_", "-")


def _allocation(size: int, ratios: tuple[float, float, float]) -> list[int]:
    """Largest-remainder allocation, retaining each split when a stratum permits it."""
    ideal = [size * ratio for ratio in ratios]
    result = [math.floor(value) for value in ideal]
    for _ in range(size - sum(result)):
        index = max(range(3), key=lambda i: (ideal[i] - result[i], ratios[i], -i))
        result[index] += 1
    positive = [i for i, ratio in enumerate(ratios) if ratio > 0]
    if size >= len(positive):
        for index in positive:
            if result[index] == 0:
                donor = max(
                    (i for i in positive if result[i] > 1),
                    key=lambda i: (result[i] - ideal[i], result[i], -i),
                )
                result[donor] -= 1
                result[index] += 1
    return result


def _csv_bytes(fields: list[str], rows: list[dict[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare_rsna(
    data_root: str | Path,
    out: str | Path,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, Any]:
    """Write manifests without requiring images, without using coordinates to select test ROIs.

    The official LumbarDISC supplement says no patient was to be represented more
    than once. We therefore group by study_id; the released CSVs do not include a
    separate patient linkage table. All levels and series of a study stay together.
    """
    data_root, out = Path(data_root), Path(out)
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(not math.isfinite(value) or value < 0 for value in ratios) or not math.isclose(
        sum(ratios), 1.0, abs_tol=1e-9
    ):
        raise ValueError("Split ratios must be finite, non-negative, and sum to 1.")
    if train_ratio <= 0:
        raise ValueError("The training ratio must be positive.")
    grades = _read_csv(data_root / "train.csv", {"study_id", *LEVELS.values()})
    raw_series = _read_csv(
        data_root / "train_series_descriptions.csv", {"study_id", "series_id", "series_description"}
    )
    coordinates = _read_csv(
        data_root / "train_label_coordinates.csv",
        {"study_id", "series_id", "instance_number", "condition", "level", "x", "y"},
    )

    grade_by_study: dict[str, dict[str, int | None]] = {}
    missing_labels = {level: 0 for level in LEVELS}
    for row in grades:
        study = _identifier(row["study_id"], "study_id")
        if study in grade_by_study:
            raise ValueError(
                "train.csv contains duplicate study_id values; resolve them before splitting."
            )
        grade_by_study[study] = {level: _grade(row[column]) for level, column in LEVELS.items()}
        for level, grade in grade_by_study[study].items():
            missing_labels[level] += grade is None

    series_by_study: dict[str, list[dict[str, str]]] = {}
    seen_series: dict[tuple[str, str], str] = {}
    for row in raw_series:
        study = _identifier(row["study_id"], "study_id")
        series = _identifier(row["series_id"], "series_id")
        key = study, series
        description = row["series_description"]
        if key in seen_series:
            if seen_series[key] != description:
                raise ValueError(
                    "Conflicting descriptions for the same study/series in series metadata."
                )
            continue
        seen_series[key] = description
        if description == "Sagittal T2/STIR" and study in grade_by_study:
            series_by_study.setdefault(study, []).append(row)

    eligible = sorted(
        study
        for study, labels in grade_by_study.items()
        if study in series_by_study and any(value is not None for value in labels.values())
    )
    if not eligible:
        raise ValueError(
            "No studies with Sagittal T2/STIR series and non-missing target labels were found."
        )
    strata: dict[int, list[str]] = {grade: [] for grade in range(3)}
    worst: dict[str, int] = {}
    for study in eligible:
        worst[study] = max(grade for grade in grade_by_study[study].values() if grade is not None)
        strata[worst[study]].append(study)

    split_by_study: dict[str, str] = {}
    stratum_allocation: dict[str, dict[str, int]] = {}
    for grade, studies in strata.items():
        # Each stratum has its own generator so unrelated strata cannot perturb it.
        shuffled = studies.copy()
        random.Random(f"{seed}:{grade}").shuffle(shuffled)
        sizes = _allocation(len(shuffled), ratios)
        start = 0
        for split, size in zip(SPLITS, sizes):
            for study in shuffled[start : start + size]:
                split_by_study[study] = split
            start += size
        stratum_allocation[CLASS_NAMES[grade]] = dict(zip(SPLITS, sizes))

    study_rows = [
        {
            "study_id": study,
            "split": split_by_study[study],
            "worst_grade": worst[study],
            "n_series": len(series_by_study[study]),
            "n_labeled_levels": sum(value is not None for value in grade_by_study[study].values()),
        }
        for study in eligible
    ]
    series_rows: list[dict[str, Any]] = []
    selected_series: set[tuple[str, str]] = set()
    for study in eligible:
        for row in sorted(series_by_study[study], key=lambda item: item["series_id"]):
            series = row["series_id"]
            selected_series.add((study, series))
            relative_path = f"train_images/{study}/{series}"
            count = sum(path.is_file() for path in (data_root / relative_path).glob("*.dcm"))
            series_rows.append(
                {
                    "study_id": study,
                    "series_id": series,
                    "split": split_by_study[study],
                    "series_description": row["series_description"],
                    "series_path": relative_path,
                    "dicom_count": count,
                    "downloaded": count > 0,
                    "download_status": "present_unverified" if count else "not_downloaded",
                    "sequence_family": "T2_like",
                    "sequence_subtype": "unknown",
                    "geometry_qc_status": "pending",
                }
            )

    coordinate_rows: list[dict[str, Any]] = []
    invalid_coordinates = 0
    duplicate_coordinates = 0
    seen_coordinates: set[tuple[Any, ...]] = set()
    coordinate_counts: dict[tuple[str, str], int] = {}
    for row in coordinates:
        study, series = row["study_id"], row["series_id"]
        level = _level(row["level"])
        condition = row["condition"].lower().replace(" ", "_")
        if (
            (study, series) not in selected_series
            or condition != "spinal_canal_stenosis"
            or level not in LEVELS
        ):
            continue
        try:
            instance = int(row["instance_number"])
            x, y = float(row["x"]), float(row["y"])
            if instance < 0 or not all(math.isfinite(value) and value >= 0 for value in (x, y)):
                raise ValueError
        except (ValueError, OverflowError):
            invalid_coordinates += 1
            continue
        key = study, series, level, instance, x, y
        if key in seen_coordinates:
            duplicate_coordinates += 1
            continue
        seen_coordinates.add(key)
        coordinate_counts[(study, level)] = coordinate_counts.get((study, level), 0) + 1
        coordinate_rows.append(
            {
                "study_id": study,
                "series_id": series,
                "split": split_by_study[study],
                "level": level,
                "instance_number": instance,
                "x": x,
                "y": y,
                "coordinate_use": "reference_qc_only",
            }
        )
    coordinate_rows.sort(
        key=lambda row: (
            row["study_id"],
            row["series_id"],
            row["level"],
            row["instance_number"],
            row["x"],
            row["y"],
        )
    )

    label_rows = [
        {
            "study_id": study,
            "split": split_by_study[study],
            "level": level,
            "label": grade,
            "label_name": CLASS_NAMES[grade],
            "source_column": LEVELS[level],
            "coordinate_count": coordinate_counts.get((study, level), 0),
        }
        for study in eligible
        for level, grade in grade_by_study[study].items()
        if grade is not None
    ]
    output_bytes = {
        "studies.csv": _csv_bytes(
            ["study_id", "split", "worst_grade", "n_series", "n_labeled_levels"], study_rows
        ),
        "series.csv": _csv_bytes(
            [
                "study_id",
                "series_id",
                "split",
                "series_description",
                "series_path",
                "dicom_count",
                "downloaded",
                "download_status",
                "sequence_family",
                "sequence_subtype",
                "geometry_qc_status",
            ],
            series_rows,
        ),
        "labels.csv": _csv_bytes(
            [
                "study_id",
                "split",
                "level",
                "label",
                "label_name",
                "source_column",
                "coordinate_count",
            ],
            label_rows,
        ),
        "coordinates.csv": _csv_bytes(
            [
                "study_id",
                "series_id",
                "split",
                "level",
                "instance_number",
                "x",
                "y",
                "coordinate_use",
            ],
            coordinate_rows,
        ),
    }
    fixed_split = out / "studies.csv"
    if fixed_split.exists() and fixed_split.read_bytes() != output_bytes["studies.csv"]:
        raise ValueError(
            "The existing study split differs. Use a new output directory; an established test split must not be overwritten."
        )

    warnings = []
    for split, ratio in zip(SPLITS, ratios):
        present = {row["label"] for row in label_rows if row["split"] == split}
        if ratio > 0 and present != {0, 1, 2}:
            warnings.append(
                f"{split} does not contain every severity class; sample size limits stratification."
            )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "seed": seed,
        "requested_ratios": dict(zip(SPLITS, ratios)),
        "split_method": "study-grouped; stratified by worst available grade across L4-L5 and L5-S1; largest remainder; each positive split retained per stratum if large enough",
        "grouping": {
            "key": "study_id",
            "patient_linkage_available": False,
            "basis": "The official LumbarDISC Appendix S1 states no patient was to be represented more than once in submitted studies.",
            "source": "https://pubs.rsna.org/doi/suppl/10.1148/ryai.250480/suppl_file/ryai250480suppa1.pdf",
        },
        "class_mapping": dict(enumerate(CLASS_NAMES)),
        "levels": list(LEVELS),
        "input_sha256": {name: _sha((data_root / name).read_bytes()) for name in METADATA_FILES},
        "output_sha256": {name: _sha(content) for name, content in output_bytes.items()},
        "n_source_studies": len(grade_by_study),
        "n_eligible_studies": len(eligible),
        "n_excluded_no_sagittal_t2_stir": sum(
            study not in series_by_study for study in grade_by_study
        ),
        "n_excluded_all_target_labels_missing_with_series": sum(
            study in series_by_study and all(value is None for value in labels.values())
            for study, labels in grade_by_study.items()
        ),
        "missing_target_labels_in_source": missing_labels,
        "n_series": len(series_rows),
        "n_series_with_local_dicoms": sum(row["downloaded"] for row in series_rows),
        "n_local_dicoms": sum(row["dicom_count"] for row in series_rows),
        "n_invalid_target_coordinates": invalid_coordinates,
        "n_duplicate_target_coordinates": duplicate_coordinates,
        "sequence_note": "Sagittal T2/STIR is T2-like and can include STIR, conventional T2, fat-saturated T2, or Dixon. Subtypes remain unknown; Kaggle's DICOM metadata whitelist does not preserve reliable TE/TR/TI/sequence descriptions.",
        "geometry_qc": "pending; local file counts do not prove complete series or valid geometry",
        "coordinates_note": "Reference annotations are preserved for QC; a deployment-style test must use model-localized ROIs, not test ground-truth coordinates.",
        "stratum_allocation": stratum_allocation,
        "splits": {
            split: {
                "studies": sum(row["split"] == split for row in study_rows),
                "series": sum(row["split"] == split for row in series_rows),
                "labels": sum(row["split"] == split for row in label_rows),
                "class_counts": {
                    name: sum(row["split"] == split and row["label"] == grade for row in label_rows)
                    for grade, name in enumerate(CLASS_NAMES)
                },
                "by_level": {
                    level: {
                        name: sum(
                            row["split"] == split
                            and row["level"] == level
                            and row["label"] == grade
                            for row in label_rows
                        )
                        for grade, name in enumerate(CLASS_NAMES)
                    }
                    for level in LEVELS
                },
            }
            for split in SPLITS
        },
        "warnings": warnings,
    }
    output_bytes["summary.json"] = (
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    out.mkdir(parents=True, exist_ok=True)
    for name, content in output_bytes.items():
        temporary = out / (name + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(out / name)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args(argv)
    try:
        summary = prepare_rsna(
            args.data_root, args.out, args.seed, args.train_ratio, args.val_ratio, args.test_ratio
        )
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Data preparation failed: {exc}\n")
    print(
        json.dumps(
            {
                "studies": summary["n_eligible_studies"],
                "series": summary["n_series"],
                "splits": {key: value["studies"] for key, value in summary["splits"].items()},
                "warnings": summary["warnings"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
