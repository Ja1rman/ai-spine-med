"""Explicit, sealed RSNA axial TEST access for the external-data transfer study.

Preparation reads study/series metadata only. Download and source inventories
require all three AFTER models and their BEFORE counterparts to be frozen.
The original axial TRAIN/validation pipeline is deliberately not modified.
"""

from __future__ import annotations

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
    is_complete_series,
)

SCOPE = "external_axial_v4_rsna_test"
MODALITY = "axial_v4_rsna_test"
SEEDS = (42, 2024, 3407)
EXPECTED_STUDIES = 295
EXPECTED_SERIES = 346
MAX_UNCOMPRESSED_BYTES = 25_000_000_000
FIELDS = ("study_id", "series_id", "split", "series_description", "series_path")
_HEX = re.compile(r"[0-9a-f]{64}")
_ID = re.compile(r"[0-9]+")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv_bytes(rows: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ArchiveDownloadError("Expected an object in the sealed metadata")
    return value


def _source_rows(data_root: Path, fixed: Path):
    """No grade/coordinate file is needed or opened here."""
    paths = {
        "studies.csv": fixed / "studies.csv",
        "train_series_descriptions.csv": data_root / "train_series_descriptions.csv",
        "preparation_summary.json": fixed / "summary.json",
    }
    hashes = {name: _sha(path) for name, path in paths.items()}
    prepared = _json(paths["preparation_summary.json"])
    if (
        hashes["studies.csv"] != prepared["output_sha256"]["studies.csv"]
        or hashes["train_series_descriptions.csv"]
        != prepared["input_sha256"]["train_series_descriptions.csv"]
    ):
        raise ArchiveDownloadError("Fixed study grouping or official series metadata changed")
    groups = {}
    with paths["studies.csv"].open(newline="") as stream:
        for row in csv.DictReader(stream):
            study = row["study_id"]
            if (
                not _ID.fullmatch(study)
                or study in groups
                or row["split"] not in {"train", "validation", "test"}
            ):
                raise ArchiveDownloadError("Invalid or duplicate fixed study grouping")
            groups[study] = row["split"]
    studies = {study for study, split in groups.items() if split == "test"}
    if len(studies) != EXPECTED_STUDIES:
        raise ArchiveDownloadError("Require exactly the original 295 held-out RSNA studies")
    rows, seen = [], set()
    with paths["train_series_descriptions.csv"].open(newline="") as stream:
        for row in csv.DictReader(stream):
            study, series = row["study_id"], row["series_id"]
            if study not in studies or row["series_description"] != "Axial T2":
                continue
            if not _ID.fullmatch(series) or (study, series) in seen:
                raise ArchiveDownloadError("Unsafe or duplicate TEST axial series identifier")
            seen.add((study, series))
            rows.append(
                dict(
                    study_id=study,
                    series_id=series,
                    split="test",
                    series_description="Axial T2",
                    series_path=f"train_images/{study}/{series}",
                )
            )
    rows.sort(key=lambda row: (int(row["study_id"]), int(row["series_id"])))
    if len(rows) != EXPECTED_SERIES or {row["study_id"] for row in rows} != studies:
        raise ArchiveDownloadError(
            "Require all 346 Axial T2 series from the 295 fixed TEST studies"
        )
    return rows, hashes, paths


def _summary(rows, hashes, paths):
    series_sha = hashlib.sha256(_csv_bytes(rows)).hexdigest()
    fingerprint = {**hashes, "series.csv": series_sha}
    per_study = Counter(row["study_id"] for row in rows)
    return {
        "schema_version": 1,
        "scope": SCOPE,
        "allowed_splits": ["test"],
        "sequence": "Axial T2",
        "n_studies": len(per_study),
        "n_series": len(rows),
        "input_sha256": hashes,
        "input_paths": {name: str(path.resolve()) for name, path in paths.items()},
        "output_sha256": {"series.csv": series_sha},
        "manifest_sha256": hashlib.sha256(_json_bytes(fingerprint)).hexdigest(),
        "series_per_study_histogram": {
            str(n): count for n, count in sorted(Counter(per_study.values()).items())
        },
        "preparation_reads_pixels": False,
        "selection_uses_labels_or_reference_coordinates": False,
        "download_requires_completed_transfer_selection": True,
    }


def _save_immutable(path: Path, contents: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != contents:
            raise ArchiveDownloadError("Output already exists with different contents")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(contents)
    temporary.replace(path)


def prepare_rsna_axial_test_manifest(
    data_root: str | Path, manifest_dir: str | Path, out: str | Path
) -> dict:
    """Prepare only metadata; this does not authorize reading/downloading TEST MRI."""
    try:
        rows, hashes, paths = _source_rows(Path(data_root), Path(manifest_dir))
        summary = _summary(rows, hashes, paths)
        out = Path(out)
        contents = {"series.csv": _csv_bytes(rows), "summary.json": _json_bytes(summary)}
        if out.exists() and set(path.name for path in out.iterdir()) - set(contents):
            raise ArchiveDownloadError("TEST manifest directory contains unrelated files")
        # Check both existing outputs before writing either.
        for name, value in contents.items():
            if (out / name).exists() and (out / name).read_bytes() != value:
                raise ArchiveDownloadError("TEST manifest already exists with different contents")
        for name, value in contents.items():
            _save_immutable(out / name, value)
        return summary
    except ArchiveDownloadError:
        raise
    except (OSError, ValueError, KeyError, TypeError):
        raise ArchiveDownloadError(
            "Invalid fixed TEST grouping or official series metadata"
        ) from None


def validate_rsna_axial_test_manifest(
    data_root: str | Path, test_manifest: str | Path
) -> tuple[list[dict], dict]:
    """Reconstruct the complete selection from the fixed metadata, without pixels."""
    try:
        root = Path(test_manifest)
        root = root.parent if root.is_file() else root
        summary = _json(root / "summary.json")
        source = Path(summary["input_paths"]["studies.csv"])
        if source.name != "studies.csv":
            raise ArchiveDownloadError("Unexpected fixed study grouping path")
        rows, hashes, paths = _source_rows(Path(data_root), source.parent)
        if summary != _summary(rows, hashes, paths) or (
            root / "series.csv"
        ).read_bytes() != _csv_bytes(rows):
            raise ArchiveDownloadError("TEST axial manifest, hashes or complete cohort changed")
        return rows, summary
    except ArchiveDownloadError:
        raise
    except (OSError, ValueError, KeyError, TypeError):
        raise ArchiveDownloadError(
            "Invalid TEST axial manifest; prepare its metadata first"
        ) from None


def _checked_file(spec: dict, anchor: Path, *, confined: bool = False) -> Path:
    if (
        not isinstance(spec, dict)
        or not isinstance(spec.get("path"), str)
        or not spec["path"]
        or not isinstance(spec.get("sha256"), str)
        or not _HEX.fullmatch(spec["sha256"])
    ):
        raise ArchiveDownloadError("Malformed sealed file reference")
    path = Path(spec["path"])
    path = path if path.is_absolute() else anchor / path
    if confined and not path.resolve().is_relative_to(anchor.resolve()):
        raise ArchiveDownloadError("Training checkpoint escapes its completed run directory")
    if not path.is_file() or _sha(path) != spec["sha256"]:
        raise ArchiveDownloadError("Sealed file is missing or its checksum changed")
    return path


def _validate_completed_training(selection_path: Path, protocol_sha: str, test_studies: set[str]):
    selected = _json(selection_path)
    if (
        selected.get("schema_version") != 1
        or selected.get("model_family") != "axial_transfer_v1"
        or selected.get("frozen") is not True
        or selected.get("protocol_sha256") != protocol_sha
        or not isinstance(selected.get("training_index_sha256"), str)
        or not _HEX.fullmatch(selected["training_index_sha256"])
    ):
        raise ArchiveDownloadError("Require a frozen completed axial transfer selection")
    root = selection_path.parent
    summary_path = _checked_file(
        {"path": "summary.json", "sha256": selected.get("training_summary_sha256")}, root
    )
    summary = _json(summary_path)
    if (
        summary.get("schema_version") != 1
        or summary.get("status") != "complete"
        or summary.get("model_family") != "axial_transfer_v1"
        or summary.get("protocol_sha256") != protocol_sha
        or summary.get("index_sha256") != selected["training_index_sha256"]
        or summary.get("seeds") != list(SEEDS)
    ):
        raise ArchiveDownloadError(
            "All three transfer training runs must be complete before TEST access"
        )
    keys = []
    for name in ("fit_keys", "validation_keys"):
        values = selected.get(name)
        if not isinstance(values, list) or not values:
            raise ArchiveDownloadError(
                "Transfer selection must record its fitting and validation patients"
            )
        parsed = set()
        for pair in values:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not all(isinstance(value, str) and value for value in pair)
                or tuple(pair) in parsed
            ):
                raise ArchiveDownloadError("Invalid or duplicate transfer patient keys")
            parsed.add(tuple(pair))
            if pair[0].lower().startswith("rsna") and pair[1] in test_studies:
                raise ArchiveDownloadError(
                    "Original RSNA TEST patients cannot fit or select transfer models"
                )
        keys.append(parsed)
    if keys[0] & keys[1]:
        raise ArchiveDownloadError("Transfer fitting and validation patients overlap")
    runs, summary_runs = selected.get("runs"), summary.get("runs")
    if (
        not isinstance(runs, list)
        or not isinstance(summary_runs, list)
        or len(runs) != 3
        or len(summary_runs) != 3
    ):
        raise ArchiveDownloadError("Require exactly three completed transfer seeds")
    seen, all_paths = set(), set()
    for run in runs:
        if (
            not isinstance(run, dict)
            or type(run.get("seed")) is not int
            or run["seed"] not in SEEDS
            or run["seed"] in seen
        ):
            raise ArchiveDownloadError("Invalid or duplicate transfer seed")
        if type(run.get("best_epoch")) is not int or run["best_epoch"] < 0:
            raise ArchiveDownloadError(
                "Selected transfer epoch must be fixed, including optional warm-start epoch zero"
            )
        seen.add(run["seed"])
        matching = [
            entry
            for entry in summary_runs
            if isinstance(entry, dict) and entry.get("seed") == run["seed"]
        ]
        fields = (
            "seed",
            "best_epoch",
            "checkpoint",
            "checkpoint_sha256",
            "before_checkpoint",
            "before_checkpoint_sha256",
        )
        if len(matching) != 1 or any(matching[0].get(key) != run.get(key) for key in fields):
            raise ArchiveDownloadError("Completed summary and frozen checkpoint selection disagree")
        for prefix in ("checkpoint", "before_checkpoint"):
            path = _checked_file(
                {"path": run.get(prefix), "sha256": run.get(prefix + "_sha256")},
                root,
                confined=True,
            ).resolve()
            if path in all_paths:
                raise ArchiveDownloadError("BEFORE/AFTER seed checkpoints must have distinct paths")
            all_paths.add(path)


def validate_transfer_selection(
    data_root: str | Path, test_manifest: str | Path, selection_file: str | Path
) -> tuple[dict, str]:
    """Gate every TEST download/read on the exact completed three-seed selection."""
    try:
        rows, manifest = validate_rsna_axial_test_manifest(data_root, test_manifest)
        path = Path(selection_file)
        selected = _json(path)
        if (
            selected.get("schema_version") != 1
            or selected.get("scope") != SCOPE
            or selected.get("frozen") is not True
            or selected.get("test_manifest_sha256") != manifest["manifest_sha256"]
        ):
            raise ArchiveDownloadError("TEST access requires its frozen external-transfer seal")
        protocol = _checked_file(selected.get("protocol"), path.parent)
        training = _checked_file(selected.get("training_selection"), path.parent)
        _validate_completed_training(training, _sha(protocol), {row["study_id"] for row in rows})
        return selected, _sha(path)
    except ArchiveDownloadError:
        raise
    except (OSError, ValueError, KeyError, TypeError):
        raise ArchiveDownloadError(
            "TEST remains sealed until all transfer models are complete and verified"
        ) from None


def _series_directories(root, rows, manifest_sha, selection_sha):
    for row in rows:
        directory = root / row["series_path"]
        if not directory.resolve().is_relative_to(root.resolve()):
            raise ArchiveDownloadError("TEST axial series path escapes its data directory")
        marker = directory / COMPLETE_MARKER
        if marker.exists():
            saved = _json(marker)
            expected = {
                "scope": SCOPE,
                "modality": MODALITY,
                "allowed_splits": ["test"],
                "manifest_sha256": manifest_sha,
                "selection_sha256": selection_sha,
            }
            if any(saved.get(key) != value for key, value in expected.items()):
                raise ArchiveDownloadError(
                    "A TEST directory has another pipeline's completion receipt"
                )
        yield row, directory


def download_rsna_axial_test(
    data_root: str | Path,
    test_manifest: str | Path,
    selection_file: str | Path,
    workers: int = 4,
    *,
    source=None,
) -> dict:
    """Download all fixed TEST axial series only after all transfer models are sealed."""
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ArchiveDownloadError("TEST axial download requires 1..8 workers")
    try:
        _selected, seal_sha = validate_transfer_selection(data_root, test_manifest, selection_file)
        rows, manifest = validate_rsna_axial_test_manifest(data_root, test_manifest)
        root = Path(data_root)
        list(_series_directories(root, rows, manifest["manifest_sha256"], seal_sha))
        selected = {row["series_path"] for row in rows}
        source = source if source is not None else _open_official_archive()
        plans = _archive_plan(source, selected)
        size = sum(member.file_size for plan in plans for member in plan.members)
        if size > MAX_UNCOMPRESSED_BYTES:
            raise ArchiveDownloadError("TEST axial series exceed the fixed 25 GB download limit")
        largest = max(sum(member.file_size for member in plan.members) for plan in plans)
        if shutil.disk_usage(root).free < size + workers * largest + 512 * 1024**2:
            raise ArchiveDownloadError("Insufficient free space for TEST MRI and atomic staging")
        return _download_selected(
            root,
            selected,
            manifest["manifest_sha256"],
            workers,
            source=source,
            status_name="external_axial_v4_rsna_test_download.json",
            context={
                "scope": SCOPE,
                "modality": MODALITY,
                "allowed_splits": ["test"],
                "selection_sha256": seal_sha,
            },
        )
    except ArchiveDownloadError:
        raise
    except (OSError, ValueError, KeyError, TypeError):
        raise ArchiveDownloadError("Invalid TEST download metadata or completion receipt") from None


def build_source_inventory(
    data_root: str | Path,
    test_manifest: str | Path,
    selection_file: str | Path,
    out: str | Path,
) -> dict:
    """Hash every completed source file without decoding pixels or reading grades."""
    try:
        _selected, seal_sha = validate_transfer_selection(data_root, test_manifest, selection_file)
        rows, manifest = validate_rsna_axial_test_manifest(data_root, test_manifest)
        root = Path(data_root)
        sources = []
        for row, directory in _series_directories(
            root, rows, manifest["manifest_sha256"], seal_sha
        ):
            if not is_complete_series(directory):
                raise ArchiveDownloadError(
                    "All TEST axial series must be complete before source inventory"
                )
            receipt = directory / COMPLETE_MARKER
            saved = _json(receipt)
            files = []
            for name in sorted(saved["files"], key=lambda value: int(value.removesuffix(".dcm"))):
                path = directory / name
                if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
                    raise ArchiveDownloadError("Source DICOM cannot escape its verified series")
                files.append({"name": name, "bytes": path.stat().st_size, "sha256": _sha(path)})
            sources.append(
                {
                    **row,
                    "completion_receipt_sha256": _sha(receipt),
                    "n_files": len(files),
                    "total_bytes": sum(item["bytes"] for item in files),
                    "files": files,
                }
            )
        result = {
            "schema_version": 1,
            "scope": SCOPE,
            "complete": True,
            "allowed_splits": ["test"],
            "manifest_sha256": manifest["manifest_sha256"],
            "selection_sha256": seal_sha,
            "data_root": str(root.resolve()),
            "n_studies": EXPECTED_STUDIES,
            "n_series": len(sources),
            "n_files": sum(row["n_files"] for row in sources),
            "total_bytes": sum(row["total_bytes"] for row in sources),
            "source_series": sources,
        }
        result["sources_sha256"] = hashlib.sha256(_json_bytes(result)).hexdigest()
        _save_immutable(Path(out), _json_bytes(result))
        return result
    except ArchiveDownloadError:
        raise
    except (OSError, ValueError, KeyError, TypeError):
        raise ArchiveDownloadError("Invalid or incomplete sealed TEST source inventory") from None
