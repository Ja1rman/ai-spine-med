"""Prepare and download every Axial T2 series in the fixed TRAIN/VALIDATION.

No reference grades, localizer coordinates or held-out MRI are read. Download
uses the authorized official Kaggle archive, ranges, CRC and atomic receipts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from .rsna_bulk import (
    COMPLETE_MARKER,
    ArchiveDownloadError,
    _archive_plan,
    _download_selected,
    _json_bytes,
    _open_official_archive,
)

FIELDS = ("study_id", "series_id", "split", "series_description", "series_path")
SPLITS = ("train", "validation")
SCOPE = "axial_v3_train_validation"
MAX_UNCOMPRESSED_BYTES = 50_000_000_000


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_bytes(rows: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def _source_rows(data_root: Path, manifest_dir: Path):
    """Read only the fixed study grouping and official series descriptions."""
    studies_path = manifest_dir / "studies.csv"
    original_summary = manifest_dir / "summary.json"
    descriptions = data_root / "train_series_descriptions.csv"
    summary = json.loads(original_summary.read_text())
    hashes = {
        "studies.csv": _sha(studies_path),
        "train_series_descriptions.csv": _sha(descriptions),
        "preparation_summary.json": _sha(original_summary),
    }
    if (
        hashes["studies.csv"] != summary["output_sha256"]["studies.csv"]
        or hashes["train_series_descriptions.csv"]
        != summary["input_sha256"]["train_series_descriptions.csv"]
    ):
        raise ArchiveDownloadError(
            "Fixed study grouping or official series metadata checksum changed"
        )
    with studies_path.open(newline="", encoding="utf-8") as stream:
        studies = list(csv.DictReader(stream))
    groups = {}
    for row in studies:
        if (
            not re.fullmatch(r"[0-9]+", row["study_id"])
            or row["split"] not in (*SPLITS, "test")
            or row["study_id"] in groups
        ):
            raise ArchiveDownloadError("Invalid or duplicate fixed study grouping")
        groups[row["study_id"]] = row["split"]
    selected_groups = {key: value for key, value in groups.items() if value in SPLITS}
    rows, seen = [], set()
    with descriptions.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            study, series = row["study_id"], row["series_id"]
            if study not in selected_groups or row["series_description"] != "Axial T2":
                continue
            if not re.fullmatch(r"[0-9]+", series) or (study, series) in seen:
                raise ArchiveDownloadError("Unsafe or duplicate axial series identifier")
            seen.add((study, series))
            rows.append(
                {
                    "study_id": study,
                    "series_id": series,
                    "split": selected_groups[study],
                    "series_description": "Axial T2",
                    "series_path": f"train_images/{study}/{series}",
                }
            )
    rows.sort(key=lambda row: (int(row["study_id"]), int(row["series_id"])))
    if not rows:
        raise ArchiveDownloadError("The fixed train/validation cohort has no Axial T2 series")
    return rows, hashes, selected_groups


def _summary(rows, hashes, groups, data_root, manifest_dir):
    series_sha = hashlib.sha256(_csv_bytes(rows)).hexdigest()
    receipt = {
        "studies.csv": hashes["studies.csv"],
        "series.csv": series_sha,
        "train_series_descriptions.csv": hashes["train_series_descriptions.csv"],
    }
    per_study = Counter(row["study_id"] for row in rows)
    return {
        "schema_version": 1,
        "scope": SCOPE,
        "allowed_splits": list(SPLITS),
        "sequence": "Axial T2",
        "selection_uses_reference_coordinates": False,
        "test_images_accessed": False,
        "input_sha256": hashes,
        "input_paths": {
            "studies.csv": str((manifest_dir / "studies.csv").resolve()),
            "train_series_descriptions.csv": str(
                (data_root / "train_series_descriptions.csv").resolve()
            ),
            "preparation_summary.json": str((manifest_dir / "summary.json").resolve()),
        },
        "output_sha256": {"series.csv": series_sha},
        "manifest_sha256": hashlib.sha256(_json_bytes(receipt)).hexdigest(),
        "n_series": len(rows),
        "n_studies_with_axial": len(per_study),
        "n_fixed_train_validation_studies": len(groups),
        "n_studies_without_axial": len(set(groups) - set(per_study)),
        "series_by_split": dict(Counter(row["split"] for row in rows)),
        "studies_by_split": dict(Counter(groups[study] for study in per_study)),
        "series_per_study_histogram": {
            str(number): count for number, count in sorted(Counter(per_study.values()).items())
        },
        "scope_note": "All official Axial T2 series for fixed TRAIN/VALIDATION studies; no grades or localizer coordinates are used; no test MRI.",
    }


def prepare_axial_manifest(
    data_root: str | Path, manifest_dir: str | Path, out: str | Path
) -> dict:
    """Save a separate immutable series.csv/summary.json; never refresh the old split."""
    data_root, manifest_dir, out = map(Path, (data_root, manifest_dir, out))
    try:
        rows, hashes, groups = _source_rows(data_root, manifest_dir)
        summary = _summary(rows, hashes, groups, data_root, manifest_dir)
        contents = {"series.csv": _csv_bytes(rows), "summary.json": _json_bytes(summary)}
        if out.exists() and any(out.iterdir()):
            if set(path.name for path in out.iterdir()) != set(contents) or any(
                not (out / name).is_file() or (out / name).read_bytes() != value
                for name, value in contents.items()
            ):
                raise ArchiveDownloadError(
                    "Axial manifest already exists with different contents; choose a new directory"
                )
            return summary
        out.mkdir(parents=True, exist_ok=True)
        for name, content in contents.items():
            temporary = out / (name + ".tmp")
            temporary.write_bytes(content)
            temporary.replace(out / name)
        return summary
    except ArchiveDownloadError:
        raise
    except (OSError, ValueError, KeyError, TypeError):
        raise ArchiveDownloadError(
            "Prepare axial data from valid fixed study and official series metadata"
        ) from None


def validate_axial_manifest(
    data_root: str | Path, axial_manifest: str | Path
) -> tuple[list[dict], dict]:
    """Validate complete axial selection before any MRI pixels or remote archive."""
    data_root, root = Path(data_root), Path(axial_manifest)
    root = root.parent if root.is_file() else root
    try:
        summary = json.loads((root / "summary.json").read_text())
        study_path = Path(summary["input_paths"]["studies.csv"])
        if study_path.name != "studies.csv":
            raise ArchiveDownloadError("Unexpected fixed study manifest path")
        rows, hashes, groups = _source_rows(data_root, study_path.parent)
        expected = _summary(rows, hashes, groups, data_root, study_path.parent)
        if summary != expected or (root / "series.csv").read_bytes() != _csv_bytes(rows):
            raise ArchiveDownloadError(
                "Axial manifest checksum, source metadata or complete train/validation selection changed"
            )
        return rows, summary
    except ArchiveDownloadError:
        raise
    except (OSError, ValueError, KeyError, TypeError):
        raise ArchiveDownloadError(
            "Invalid axial manifest; prepare it from the fixed train/validation grouping first"
        ) from None


def download_axial(data_root: str | Path, axial_manifest: str | Path, workers: int = 8) -> dict:
    """Range-download only prepared TRAIN/VALIDATION axial series, at most 50 GB."""
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ArchiveDownloadError("Axial downloader requires 1..8 workers")
    root = Path(data_root)
    rows, summary = validate_axial_manifest(root, axial_manifest)
    selected = {row["series_path"] for row in rows}
    for row in rows:
        path = root / row["series_path"]
        if not path.resolve().is_relative_to(root.resolve()):
            raise ArchiveDownloadError("Axial series path escapes the local data directory")
        marker = path / COMPLETE_MARKER
        if marker.is_file():
            existing = json.loads(marker.read_text())
            if (
                existing.get("modality") != "axial_v3"
                or existing.get("manifest_sha256") != summary["manifest_sha256"]
            ):
                raise ArchiveDownloadError(
                    "A selected directory has another pipeline's receipt; refusing to replace it"
                )
    source = _open_official_archive()
    plans = _archive_plan(source, selected)  # ZIP central directory only, no pixel payload.
    uncompressed = sum(member.file_size for plan in plans for member in plan.members)
    if uncompressed > MAX_UNCOMPRESSED_BYTES:
        raise ArchiveDownloadError(
            "Selected axial MRI exceeds the 50 GB safety limit; no image downloaded"
        )
    largest_series = max(sum(member.file_size for member in plan.members) for plan in plans)
    required_free = uncompressed + workers * largest_series + 512 * 1024**2
    if shutil.disk_usage(root).free < required_free:
        raise ArchiveDownloadError(
            "Insufficient local free space for axial data and atomic staging; no image downloaded"
        )
    return _download_selected(
        root,
        selected,
        summary["manifest_sha256"],
        workers,
        source=source,
        status_name="axial_v3_download.json",
        context={"modality": "axial_v3", "scope": SCOPE, "allowed_splits": list(SPLITS)},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--data-root", type=Path, required=True)
    prepare.add_argument("--manifest-dir", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    download = commands.add_parser("download")
    download.add_argument("--data-root", type=Path, required=True)
    download.add_argument("--axial-manifest", type=Path, required=True)
    download.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_axial_manifest(args.data_root, args.manifest_dir, args.out)
        else:
            result = download_axial(args.data_root, args.axial_manifest, args.workers)
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted; completed axial series remain reusable.\n")
    except ArchiveDownloadError as exc:
        parser.exit(2, str(exc) + "\n")
    except Exception as exc:
        parser.exit(
            2,
            f"Axial download failed ({type(exc).__name__}); no server URLs or credentials are logged.\n",
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
