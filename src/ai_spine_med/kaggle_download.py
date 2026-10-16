"""Download official RSNA metadata or a bounded smoke sample through Kaggle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import tempfile
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path, PurePosixPath
from typing import Any

from .data import METADATA_FILES

COMPETITION = "rsna-2024-lumbar-spine-degenerative-classification"
RULES_URL = f"https://www.kaggle.com/competitions/{COMPETITION}/rules"
_DICOM = re.compile(r"^train_images/([0-9]+)/([0-9]+)/([0-9]+)\.dcm$")


class DownloadError(RuntimeError):
    """An actionable download error whose message does not expose credentials."""


def _api() -> Any:
    try:
        # Kaggle imports authenticate a global client and may print auth URLs or
        # call sys.exit(). Keep this narrowly scoped; never intercept Ctrl-C.
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise DownloadError(
            "Install the optional download dependency: pip install '.[data]'."
        ) from None
    except (Exception, SystemExit):
        raise DownloadError(
            "Configure your Kaggle credentials locally (kaggle auth login or ~/.kaggle/kaggle.json); never paste them into chat or Git."
        ) from None
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            api = KaggleApi()
            api.authenticate()
        return api
    except (Exception, SystemExit):
        raise DownloadError(
            "Kaggle authentication failed. Configure your own local credentials and accept the competition rules in your browser."
        ) from None


def _service_error() -> DownloadError:
    return DownloadError(
        f"Kaggle could not provide the requested file/list. Check network access, local credentials, and accepted rules: {RULES_URL}. Server details are omitted to protect tokens."
    )


def _download_file(api: Any, name: str, data_root: Path, overwrite: bool = False) -> bool:
    """Handle both old flattened and current nested Kaggle download layouts."""
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise DownloadError("Unsafe remote file path was refused.")
    destination = data_root.joinpath(*relative.parts)
    if destination.is_file() and destination.stat().st_size > 0 and not overwrite:
        return False
    data_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".kaggle-file-", dir=data_root) as directory:
        temporary = Path(directory)
        try:
            api.competition_download_file(
                COMPETITION, name, path=str(temporary), force=True, quiet=True
            )
        except Exception:
            raise _service_error() from None
        candidates = [
            path for path in temporary.rglob("*") if path.is_file() and path.name == relative.name
        ]
        archives = [
            path
            for path in temporary.rglob("*")
            if path.is_file() and path.name == relative.name + ".zip"
        ]
        source = temporary / "unwrapped-content"
        if len(candidates) == 1:
            source = candidates[0]
        elif len(archives) == 1 and not candidates:
            try:
                with zipfile.ZipFile(archives[0]) as archive:
                    members = [
                        item
                        for item in archive.infolist()
                        if not item.is_dir() and PurePosixPath(item.filename).name == relative.name
                    ]
                    if len(members) != 1:
                        raise DownloadError(
                            "Kaggle archive did not contain exactly the requested file."
                        )
                    # Read a single requested member; do not extract archive paths.
                    with archive.open(members[0]) as compressed, source.open("wb") as output:
                        shutil.copyfileobj(compressed, output)
            except zipfile.BadZipFile:
                raise DownloadError(
                    "The downloaded Kaggle archive is invalid; retry the download."
                ) from None
        else:
            raise DownloadError("Kaggle did not save the requested file in a recognized layout.")
        if source.stat().st_size == 0:
            raise DownloadError("Kaggle returned an empty file; retry the download.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staged = destination.with_name(destination.name + ".part")
        shutil.copyfile(source, staged)
        staged.replace(destination)
    return True


def download_metadata(
    data_root: str | Path, *, api: Any = None, overwrite: bool = False
) -> dict[str, Any]:
    """Download exactly three CSVs; never request the whole competition archive."""
    root = Path(data_root)
    client = api
    downloaded, existing = 0, 0
    for name in METADATA_FILES:
        destination = root / name
        if destination.is_file() and destination.stat().st_size > 0 and not overwrite:
            existing += 1
            continue
        if client is None:
            client = _api()
        downloaded += _download_file(client, name, root, overwrite)
    return {
        "competition": COMPETITION,
        "metadata_files": len(METADATA_FILES),
        "downloaded": downloaded,
        "already_present": existing,
    }


def _attribute(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _file_index(api: Any, data_root: Path, *, refresh: bool, max_pages: int = 2000) -> list[str]:
    cache = data_root / "_download_cache" / "kaggle_file_index.json"
    if cache.is_file() and not refresh:
        try:
            index = json.loads(cache.read_text())
            if (
                index["competition"] == COMPETITION
                and index["complete"] is True
                and isinstance(index["files"], list)
                and all(isinstance(name, str) for name in index["files"])
            ):
                return index["files"]
        except (ValueError, KeyError, TypeError):
            pass
    names: set[str] = set()
    token = None
    seen_tokens: set[str] = set()
    for page_number in range(max_pages):
        try:
            response = api.competition_list_files(COMPETITION, page_token=token, page_size=200)
        except Exception:
            raise _service_error() from None
        files = _attribute(response, "files", default=[])
        for item in files:
            name = _attribute(item, "name", "ref")
            if isinstance(name, str):
                names.add(name)
        if page_number == 0 or (page_number + 1) % 50 == 0:
            print(
                f"Indexed {len(names)} official file names (no images downloaded yet)", flush=True
            )
        token = _attribute(response, "next_page_token", "nextPageToken")
        if not token:
            result = sorted(names)
            cache.parent.mkdir(parents=True, exist_ok=True)
            staged = cache.with_suffix(".tmp")
            staged.write_text(
                json.dumps(
                    {"competition": COMPETITION, "complete": True, "files": result}, indent=2
                )
                + "\n"
            )
            staged.replace(cache)
            return result
        if token in seen_tokens:
            raise DownloadError(
                "Kaggle repeated a pagination token; the incomplete index was not saved."
            )
        seen_tokens.add(token)
    raise DownloadError(
        "Kaggle file listing exceeded its page limit; the incomplete index was not saved."
    )


def _manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _training_series(manifest_dir: Path, count: int) -> list[tuple[str, str]]:
    try:
        summary = json.loads((manifest_dir / "summary.json").read_text())
        for name in ("studies.csv", "series.csv"):
            actual = hashlib.sha256((manifest_dir / name).read_bytes()).hexdigest()
            if actual != summary["output_sha256"][name]:
                raise DownloadError(
                    "A split manifest checksum does not match summary.json; prepare a consistent manifest first."
                )
        studies = _manifest_rows(manifest_dir / "studies.csv")
        series = _manifest_rows(manifest_dir / "series.csv")
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, DownloadError):
            raise
        raise DownloadError(
            "Prepare studies.csv, series.csv, and summary.json before downloading a smoke sample."
        ) from None
    study_split = {row["study_id"]: row["split"] for row in studies}
    if len(study_split) != len(studies):
        raise DownloadError("The study manifest contains duplicate groups.")
    by_study: dict[str, list[str]] = {}
    for row in series:
        study, item = row["study_id"], row["series_id"]
        if not study.isascii() or not study.isdigit() or not item.isascii() or not item.isdigit():
            raise DownloadError("Non-numeric study/series identifiers in manifest were refused.")
        if row["split"] != study_split.get(study):
            raise DownloadError("Series and study split assignments disagree.")
        if row["split"] == "train" and row["series_description"] == "Sagittal T2/STIR":
            by_study.setdefault(study, []).append(item)
    # Prefer distinct studies, with an example from each training severity stratum.
    eligible = sorted(by_study)
    selected_studies: list[str] = []
    for grade in ("0", "1", "2"):
        candidates = sorted(
            row["study_id"]
            for row in studies
            if row["study_id"] in by_study and row["worst_grade"] == grade
        )
        if candidates and len(selected_studies) < count:
            selected_studies.append(candidates[0])
    selected_studies.extend(study for study in eligible if study not in selected_studies)
    if len(selected_studies) < count:
        raise DownloadError("Not enough distinct training studies for the requested smoke sample.")
    return [(study, sorted(by_study[study])[0]) for study in selected_studies[:count]]


def download_sample(
    data_root: str | Path,
    manifest_dir: str | Path,
    count: int = 3,
    *,
    api: Any = None,
    max_files: int = 500,
    refresh_index: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Download all indexed slices of a few training series, with a hard file bound."""
    if not 1 <= count <= 20:
        raise DownloadError("Smoke sample count must be between 1 and 20 series.")
    if max_files < 1:
        raise DownloadError("max_files must be positive.")
    root = Path(data_root)
    selected = _training_series(Path(manifest_dir), count)
    client = api if api is not None else _api()
    index = _file_index(client, root, refresh=refresh_index)
    selected_set = set(selected)
    files_by_series: dict[tuple[str, str], list[str]] = {key: [] for key in selected}
    for name in index:
        match = _DICOM.fullmatch(name)
        if match and (match[1], match[2]) in selected_set:
            files_by_series[(match[1], match[2])].append(name)
    if any(not files for files in files_by_series.values()):
        raise DownloadError(
            "The official Kaggle file index lacks a selected series. Try --refresh-index or check dataset availability."
        )
    total = sum(len(files) for files in files_by_series.values())
    if total > max_files:
        raise DownloadError(
            "The selected series exceed --max-files. Nothing was downloaded; reduce --count or deliberately raise the bound."
        )
    downloaded, existing = 0, 0
    for key in selected:
        for name in sorted(files_by_series[key]):
            changed = _download_file(client, name, root, overwrite)
            downloaded += changed
            existing += not changed
    report = {
        "competition": COMPETITION,
        "selection": "training split only; one Sagittal T2/STIR series per distinct study",
        "n_series": len(selected),
        "n_files": total,
        "downloaded": downloaded,
        "already_present": existing,
        "series": [
            {
                "study_id": study,
                "series_id": series,
                "path": f"train_images/{study}/{series}",
                "n_files": len(files_by_series[(study, series)]),
            }
            for study, series in selected
        ],
        "note": "Full series according to the official file index; geometry and pixel decoding still need QC. Rerun preparation to refresh local availability counts.",
    }
    root.mkdir(parents=True, exist_ok=True)
    staged = root / "sample_download.json.tmp"
    staged.write_text(json.dumps(report, indent=2) + "\n")
    staged.replace(root / "sample_download.json")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    metadata = commands.add_parser("metadata", help="Download only the three official CSVs.")
    metadata.add_argument("--data-root", type=Path, required=True)
    metadata.add_argument("--overwrite", action="store_true")
    sample = commands.add_parser(
        "sample",
        help="Download several complete training series; initially indexes the official file list.",
    )
    sample.add_argument("--data-root", type=Path, required=True)
    sample.add_argument("--manifest-dir", type=Path, required=True)
    sample.add_argument("--count", type=int, default=3)
    sample.add_argument("--max-files", type=int, default=500)
    sample.add_argument("--refresh-index", action="store_true")
    sample.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "metadata":
            report = download_metadata(args.data_root, overwrite=args.overwrite)
        else:
            report = download_sample(
                args.data_root,
                args.manifest_dir,
                args.count,
                max_files=args.max_files,
                refresh_index=args.refresh_index,
                overwrite=args.overwrite,
            )
    except (DownloadError, OSError) as exc:
        parser.exit(2, f"Download failed: {exc}\n")
    print(json.dumps({key: value for key, value in report.items() if key != "series"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
