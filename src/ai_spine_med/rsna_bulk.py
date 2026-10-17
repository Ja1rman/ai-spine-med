"""Fetch fixed train/validation MRI series from the official ZIP with HTTP ranges."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import stat
import struct
import tempfile
import threading
import time
import uuid
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .kaggle_download import COMPETITION, _api, _service_error

COMPLETE_MARKER = "_series_complete.json"
_DICOM = re.compile(r"^train_images/([0-9]+)/([0-9]+)/([0-9]+)\.dcm$")
_MAX_RANGE = 64 * 1024 * 1024


class ArchiveDownloadError(RuntimeError):
    """Safe diagnostic text: never contain remote URLs or server bodies."""


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name + ".tmp")
    staged.write_bytes(_json_bytes(value))
    staged.replace(path)


class OfficialArchive:
    """A short-lived authorized URL stays in memory, never in logs or resume files."""

    def __init__(self, url: str, size: int, etag: str | None, modified: str | None):
        self._url = url
        self.size = size
        self._validator = etag if etag and not etag.startswith("W/") else modified
        if self.size <= 0 or not self._validator:
            raise ArchiveDownloadError(
                "The official archive has no reliable size/version validator."
            )
        self.identity = {"size": size, "etag": etag, "last_modified": modified}
        self._local = threading.local()

    def read_range(self, start: int, end: int) -> bytes:
        import requests

        if not 0 <= start <= end < self.size or end - start + 1 > _MAX_RANGE:
            raise ArchiveDownloadError("An archive byte range exceeded its safety bounds.")
        if not hasattr(self._local, "session"):
            self._local.session = requests.Session()
        for attempt in range(4):
            try:
                with self._local.session.get(
                    self._url,
                    headers={
                        "Range": f"bytes={start}-{end}",
                        "If-Range": self._validator,
                        "Accept-Encoding": "identity",
                    },
                    stream=True,
                    timeout=(20, 90),
                ) as response:
                    if response.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                        time.sleep(2**attempt)
                        continue
                    if response.status_code != 206:
                        raise ArchiveDownloadError(
                            f"The official archive range request returned HTTP {response.status_code}; retry to refresh access."
                        )
                    expected = f"bytes {start}-{end}/{self.size}"
                    if response.headers.get("Content-Range") != expected:
                        raise ArchiveDownloadError(
                            "The archive range or version changed; retry from the official API."
                        )
                    if (
                        self.identity["etag"]
                        and response.headers.get("ETag") != self.identity["etag"]
                    ):
                        raise ArchiveDownloadError(
                            "The archive ETag changed during download; retry."
                        )
                    chunks = []
                    received = 0
                    for chunk in response.iter_content(1024 * 1024):
                        received += len(chunk)
                        if received > end - start + 1:
                            raise ArchiveDownloadError(
                                "The archive server returned too many bytes."
                            )
                        chunks.append(chunk)
                    if received != end - start + 1:
                        raise ArchiveDownloadError("An archive byte range was incomplete; retry.")
                    return b"".join(chunks)
            except ArchiveDownloadError:
                raise
            except requests.RequestException:
                if attempt == 3:
                    raise ArchiveDownloadError(
                        "Archive network retries were exhausted; completed series can be reused on retry."
                    ) from None
                time.sleep(2**attempt)
        raise ArchiveDownloadError("Archive network retries were exhausted.")


def _open_official_archive() -> OfficialArchive:
    from kagglesdk.competitions.types.competition_api_service import ApiDownloadDataFilesRequest

    api = _api()
    try:
        with api.build_kaggle_client() as client:
            request = ApiDownloadDataFilesRequest()
            request.competition_name = COMPETITION
            response = client.competitions.competition_api_client.download_data_files(request)
            try:
                if response.status_code != 200 or response.headers.get("Accept-Ranges") != "bytes":
                    raise ArchiveDownloadError(
                        "The official archive is not available with byte-range support."
                    )
                return OfficialArchive(
                    response.url,
                    int(response.headers["Content-Length"]),
                    response.headers.get("ETag"),
                    response.headers.get("Last-Modified"),
                )
            finally:
                response.close()
    except ArchiveDownloadError:
        raise
    except Exception as exc:
        raise ArchiveDownloadError(str(_service_error(exc))) from None


class _ZipMetadataReader(io.RawIOBase):
    def __init__(self, source: Any):
        self.source = source
        self.position = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        self.position = (
            offset
            if whence == 0
            else self.position + offset
            if whence == 1
            else self.source.size + offset
        )
        if not 0 <= self.position <= self.source.size:
            raise ArchiveDownloadError("Invalid ZIP metadata offset.")
        return self.position

    def read(self, size: int = -1) -> bytes:
        size = (
            self.source.size - self.position
            if size < 0
            else min(size, self.source.size - self.position)
        )
        if size == 0:
            return b""
        data = self.source.read_range(self.position, self.position + size - 1)
        self.position += len(data)
        return data


@dataclass
class SeriesPlan:
    path: str
    members: list[zipfile.ZipInfo]
    spans: list[tuple[int, int, list[zipfile.ZipInfo]]]
    metadata_sha256: str


def _selected_series(
    manifest_dir: Path, *, splits: tuple[str, ...] = ("train", "validation")
) -> tuple[set[str], str]:
    try:
        summary = json.loads((manifest_dir / "summary.json").read_text())
        contents = {
            name: (manifest_dir / name).read_bytes() for name in ("studies.csv", "series.csv")
        }
        for name, content in contents.items():
            if _sha(content) != summary["output_sha256"][name]:
                raise ArchiveDownloadError("Fixed split manifest checksum mismatch.")
        studies = list(csv.DictReader(io.StringIO(contents["studies.csv"].decode())))
        series = list(csv.DictReader(io.StringIO(contents["series.csv"].decode())))
        groups = {row["study_id"]: row["split"] for row in studies}
        if len(groups) != len(studies):
            raise ArchiveDownloadError("Duplicate study groups in the fixed manifest.")
        selected = set()
        for row in series:
            study, number = row["study_id"], row["series_id"]
            if row["split"] != groups.get(study):
                raise ArchiveDownloadError("Study and series split assignments disagree.")
            if not re.fullmatch(r"[0-9]+", study) or not re.fullmatch(r"[0-9]+", number):
                raise ArchiveDownloadError(
                    "Unsafe study or series identifier in the fixed manifest."
                )
            if row["split"] in splits and row["series_description"] == "Sagittal T2/STIR":
                path = f"train_images/{study}/{number}"
                if row["series_path"] != path or path in selected:
                    raise ArchiveDownloadError(
                        "Unsafe or duplicate series path in the fixed manifest."
                    )
                selected.add(path)
        if not selected:
            raise ArchiveDownloadError(
                "The fixed manifest has no training/validation T2-like series."
            )
        manifest_sha = _sha(_json_bytes({name: _sha(value) for name, value in contents.items()}))
        return selected, manifest_sha
    except ArchiveDownloadError:
        raise
    except (OSError, ValueError, KeyError):
        raise ArchiveDownloadError(
            "Prepare valid fixed study/series manifests before downloading."
        ) from None


def validate_test_selection(
    manifest_dir: str | Path, selection_manifest: str | Path
) -> tuple[dict[str, Any], str]:
    """Verify the frozen six-checkpoint decision before accessing held-out MRI.

    Relative protocol/checkpoint paths are resolved beside the selection file.
    Extra audit fields from the evaluation seal are preserved, not discarded.
    The returned digest covers the exact selection-file bytes.
    """
    manifest_dir, selection_manifest = Path(manifest_dir), Path(selection_manifest)
    try:
        content = selection_manifest.read_bytes()
        if len(content) > 2 * 1024 * 1024:
            raise ValueError("oversized selection")
        selection = json.loads(content)
        if (
            type(selection.get("schema_version")) is not int
            or selection["schema_version"] != 1
            or selection.get("scope") != "heldout_test"
            or selection.get("frozen") is not True
        ):
            raise ValueError("not a frozen held-out selection")
        summary = json.loads((manifest_dir / "summary.json").read_text())
        for name in ("studies.csv", "series.csv", "labels.csv"):
            digest = _sha((manifest_dir / name).read_bytes())
            if (
                digest != selection["manifests_sha256"][name]
                or digest != summary["output_sha256"][name]
            ):
                raise ValueError("fixed manifest changed")
        lock = json.loads(Path(__file__).with_name("assets.lock.json").read_text())
        weights = {name: item["sha256"] for name, item in lock["weights"].items()}
        if selection["original_weights_sha256"] != weights:
            raise ValueError("original model weights are not the pinned weights")

        def verify_file(item: dict) -> None:
            expected = item["sha256"]
            if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError("invalid artifact checksum")
            value = item["path"]
            if not isinstance(value, str) or not value or "://" in value:
                raise ValueError("artifact must be a local file")
            path = Path(value)
            if not path.is_absolute():
                path = selection_manifest.parent / path
            if not path.is_file() or path.is_symlink():
                raise ValueError("frozen artifact is missing or a symlink")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError("frozen artifact changed")

        verify_file(selection["protocol"])
        for key in ("source_files", "training_artifacts"):
            if key in selection:
                if not isinstance(selection[key], list) or not selection[key]:
                    raise ValueError("frozen audit artifacts must be a nonempty list")
                for item in selection[key]:
                    verify_file(item)
        seeds = {}
        paths = set()
        for variant in ("frozen_head", "fine_tuned"):
            items = selection["checkpoints"][variant]
            if not isinstance(items, list) or len(items) != 3:
                raise ValueError("three completed checkpoints per variant are required")
            seeds[variant] = set()
            for item in items:
                seed = item["seed"]
                if type(seed) is not int or seed in seeds[variant]:
                    raise ValueError("checkpoint seeds must be distinct integers")
                seeds[variant].add(seed)
                path = Path(item["path"])
                path = path if path.is_absolute() else selection_manifest.parent / path
                identity = str(path.resolve())
                if identity in paths:
                    raise ValueError("each run needs its own checkpoint file")
                paths.add(identity)
                verify_file(item)
        if seeds["frozen_head"] != seeds["fine_tuned"]:
            raise ValueError("the two model variants must use the same three seeds")
        return selection, _sha(content)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        raise ArchiveDownloadError(
            "Held-out test is sealed: provide a valid frozen selection with unchanged fixed manifests, protocol, pinned original weights and all six checkpoints."
        ) from None


def _archive_plan(source: Any, selected: set[str]) -> list[SeriesPlan]:
    try:
        with zipfile.ZipFile(_ZipMetadataReader(source)) as archive:
            infos = sorted(archive.infolist(), key=lambda item: item.header_offset)
            central_start = archive.start_dir
    except ArchiveDownloadError:
        raise
    except (ValueError, zipfile.BadZipFile):
        raise ArchiveDownloadError("The official archive ZIP metadata is invalid.") from None
    seen = set()
    grouped: dict[str, list[zipfile.ZipInfo]] = {path: [] for path in selected}
    positions = {}
    for position, member in enumerate(infos):
        path = PurePosixPath(member.filename)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in member.filename
            or "\x00" in member.filename
        ):
            raise ArchiveDownloadError("Unsafe member path in the official archive.")
        if member.filename in seen or stat.S_ISLNK(member.external_attr >> 16):
            raise ArchiveDownloadError("Duplicate or symbolic-link member in the official archive.")
        seen.add(member.filename)
        positions[member.filename] = position
        parent = member.filename.rsplit("/", 1)[0]
        if parent in selected and not member.is_dir():
            if (
                not _DICOM.fullmatch(member.filename)
                or member.flag_bits & 1
                or member.compress_type not in (0, 8)
            ):
                raise ArchiveDownloadError("An MRI archive member has an unsupported format.")
            if not 0 < member.file_size <= _MAX_RANGE or member.compress_size > _MAX_RANGE:
                raise ArchiveDownloadError("An MRI archive member exceeds its size bound.")
            grouped[parent].append(member)
    if any(not members for members in grouped.values()):
        raise ArchiveDownloadError("The archive lacks a selected training/validation series.")
    plans = []
    for path, members in sorted(grouped.items()):
        spans: list[tuple[int, int, list[zipfile.ZipInfo]]] = []
        for member in members:
            index = positions[member.filename]
            end = (infos[index + 1].header_offset if index + 1 < len(infos) else central_start) - 1
            if (
                not 0 <= member.header_offset <= end < central_start
                or end - member.header_offset + 1 > _MAX_RANGE
            ):
                raise ArchiveDownloadError("A local MRI ZIP member has invalid bounds.")
            if (
                spans
                and member.header_offset == spans[-1][1] + 1
                and end - spans[-1][0] + 1 <= _MAX_RANGE
            ):
                start, _, run = spans[-1]
                spans[-1] = (start, end, run + [member])
            else:
                spans.append((member.header_offset, end, [member]))
        metadata = [
            {
                "name": member.filename,
                "offset": member.header_offset,
                "compressed_size": member.compress_size,
                "size": member.file_size,
                "crc32": member.CRC,
                "compression": member.compress_type,
            }
            for member in members
        ]
        plans.append(SeriesPlan(path, members, spans, _sha(_json_bytes(metadata))))
    return plans


def _decode_member(block: bytes, start: int, member: zipfile.ZipInfo) -> bytes:
    offset = member.header_offset - start
    try:
        header = struct.unpack_from("<4s5H3I2H", block, offset)
        if header[0] != b"PK\x03\x04" or header[2] & 1 or header[3] != member.compress_type:
            raise ValueError
        name_size, extra_size = header[-2:]
        name_start = offset + 30
        encoding = "utf-8" if header[2] & 0x800 else "cp437"
        if block[name_start : name_start + name_size].decode(encoding) != member.filename:
            raise ValueError
        data_start = name_start + name_size + extra_size
        data_end = data_start + member.compress_size
        if data_end > len(block):
            raise ValueError
        compressed = block[data_start:data_end]
        if member.compress_type == 8:
            decoder = zlib.decompressobj(-15)
            data = decoder.decompress(compressed, member.file_size + 1)
            if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                raise ValueError
        else:
            data = compressed
        if len(data) != member.file_size or zlib.crc32(data) & 0xFFFFFFFF != member.CRC:
            raise ValueError
        return data
    except (ValueError, UnicodeError, struct.error, zlib.error):
        raise ArchiveDownloadError(
            "MRI ZIP member failed its header, size, or CRC verification."
        ) from None


def _existing_matches(directory: Path, members: list[zipfile.ZipInfo]) -> bool:
    if not directory.is_dir():
        return False
    if {path.name for path in directory.glob("*.dcm")} != {
        PurePosixPath(member.filename).name for member in members
    }:
        return False
    for member in members:
        path = directory / PurePosixPath(member.filename).name
        if not path.is_file() or path.stat().st_size != member.file_size:
            return False
        crc = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                crc = zlib.crc32(chunk, crc)
        if crc & 0xFFFFFFFF != member.CRC:
            return False
    return True


def is_complete_series(directory: str | Path) -> bool:
    """A cheap startup gate; CRC is checked by the downloader before publishing."""
    directory = Path(directory)
    try:
        marker = json.loads((directory / COMPLETE_MARKER).read_text())
        expected = marker["expected_files"]
        return (
            marker["complete"] is True
            and type(expected) is int
            and expected > 0
            and len(marker["files"]) == expected
            and len(set(marker["files"])) == expected
            and set(marker["files"])
            == {path.name for path in directory.glob("*.dcm") if path.is_file()}
            and all(re.fullmatch(r"[0-9]+\.dcm", name) for name in marker["files"])
            and all(
                re.fullmatch(r"[0-9a-f]{64}", marker[name])
                for name in ("metadata_sha256", "manifest_sha256", "source_identity_sha256")
            )
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


class _Progress:
    def __init__(
        self,
        path: Path,
        plans: list[SeriesPlan],
        source_sha: str,
        manifest_sha: str,
        context: dict | None = None,
    ):
        self.path, self.lock, self.started = path, threading.Lock(), time.monotonic()
        self.value = {
            "stage": "downloading",
            "total_series": len(plans),
            "total_files": sum(len(plan.members) for plan in plans),
            "completed_series": 0,
            "completed_files": 0,
            "reused_series": 0,
            "range_bytes_received": 0,
            "selected_compressed_bytes": sum(
                member.compress_size for plan in plans for member in plan.members
            ),
            "source_identity_sha256": source_sha,
            "manifest_sha256": manifest_sha,
            **(context or {}),
        }
        self.update()

    def update(
        self,
        *,
        stage: str | None = None,
        received: int = 0,
        plan: SeriesPlan | None = None,
        reused: bool = False,
    ) -> None:
        with self.lock:
            self.value["range_bytes_received"] += received
            if stage:
                self.value["stage"] = stage
            if plan:
                self.value["completed_series"] += 1
                self.value["completed_files"] += len(plan.members)
                self.value["reused_series"] += reused
            self.value["elapsed_seconds"] = round(time.monotonic() - self.started, 1)
            _atomic_json(self.path, self.value)
            if plan and (
                self.value["completed_series"] % 25 == 0
                or self.value["completed_series"] == self.value["total_series"]
            ):
                print(
                    f"Completed {self.value['completed_series']}/{self.value['total_series']} series; received {self.value['range_bytes_received'] / 1024**3:.2f} GiB",
                    flush=True,
                )


def _process_series(
    plan: SeriesPlan,
    source: Any,
    root: Path,
    manifest_sha: str,
    source_sha: str,
    progress: _Progress,
    stop: threading.Event,
    marker_context: dict | None = None,
) -> None:
    if stop.is_set():
        return
    final = root.joinpath(*PurePosixPath(plan.path).parts)
    marker = {
        "schema_version": 1,
        "complete": True,
        "expected_files": len(plan.members),
        "files": sorted(PurePosixPath(member.filename).name for member in plan.members),
        "metadata_sha256": plan.metadata_sha256,
        "manifest_sha256": manifest_sha,
        "source_identity_sha256": source_sha,
        "verification": "ZIP central metadata, local header, decompressed size, CRC32",
        **(marker_context or {}),
    }
    if _existing_matches(final, plan.members):
        _atomic_json(final / COMPLETE_MARKER, marker)
        progress.update(plan=plan, reused=True)
        return
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".bulk-{final.name}-", dir=final.parent))
    try:
        for start, end, members in plan.spans:
            if stop.is_set():
                return
            block = source.read_range(start, end)
            if len(block) != end - start + 1:
                raise ArchiveDownloadError("An MRI series byte range was truncated.")
            progress.update(received=len(block))
            for member in members:
                destination = staging / PurePosixPath(member.filename).name
                temporary = destination.with_suffix(".dcm.part")
                temporary.write_bytes(_decode_member(block, start, member))
                temporary.replace(destination)
        _atomic_json(staging / COMPLETE_MARKER, marker)
        if not is_complete_series(staging):
            raise ArchiveDownloadError("The staged MRI series did not pass its completeness check.")
        if final.exists():
            # Preserve an older partial directory; never silently delete user files.
            backup = final.with_name(f".bulk-previous-{final.name}-{uuid.uuid4().hex[:8]}")
            final.rename(backup)
        staging.rename(final)
        progress.update(plan=plan)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def download_train_validation(
    data_root: str | Path,
    manifest_dir: str | Path,
    workers: int = 4,
    *,
    source: Any = None,
) -> dict[str, Any]:
    """Only fixed training/validation Sagittal T2/STIR series become visible locally."""
    if not 1 <= workers <= 8:
        raise ArchiveDownloadError("Worker count must be between 1 and 8.")
    root = Path(data_root)
    selected, manifest_sha = _selected_series(Path(manifest_dir))
    return _download_selected(root, selected, manifest_sha, workers, source=source)


def download_test(
    data_root: str | Path,
    manifest_dir: str | Path,
    selection_manifest: str | Path,
    workers: int = 4,
    *,
    source: Any = None,
) -> dict[str, Any]:
    """Download only the sealed test cohort after all six models are fixed."""
    if not 1 <= workers <= 8:
        raise ArchiveDownloadError("Worker count must be between 1 and 8.")
    _selection, selection_sha = validate_test_selection(manifest_dir, selection_manifest)
    selected, manifest_sha = _selected_series(Path(manifest_dir), splits=("test",))
    return _download_selected(
        Path(data_root),
        selected,
        manifest_sha,
        workers,
        source=source,
        status_name="test_download.json",
        context={"scope": "heldout_test", "selection_sha256": selection_sha},
    )


def _download_selected(
    root: Path,
    selected: set[str],
    manifest_sha: str,
    workers: int,
    *,
    source: Any,
    status_name: str = "train_validation_download.json",
    context: dict | None = None,
) -> dict[str, Any]:
    status_path = root / "_download_cache" / status_name
    _atomic_json(
        status_path,
        {
            "stage": "indexing",
            "total_series": len(selected),
            "manifest_sha256": manifest_sha,
            **(context or {}),
        },
    )
    try:
        source = source if source is not None else _open_official_archive()
        source_sha = _sha(_json_bytes(source.identity))
        plans = _archive_plan(source, selected)
    except BaseException as exc:
        _atomic_json(
            status_path,
            {
                "stage": "failed",
                "phase": "indexing",
                "error_type": type(exc).__name__,
                "total_series": len(selected),
                "manifest_sha256": manifest_sha,
                **(context or {}),
            },
        )
        raise
    progress = _Progress(status_path, plans, source_sha, manifest_sha, context)
    stop = threading.Event()
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _process_series,
                    plan,
                    source,
                    root,
                    manifest_sha,
                    source_sha,
                    progress,
                    stop,
                    context,
                )
                for plan in plans
            ]
            try:
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                stop.set()
                for future in futures:
                    future.cancel()
                raise
    except BaseException:
        progress.update(stage="failed")
        raise
    progress.update(stage="complete")
    return progress.value.copy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--split", choices=("train-validation", "test"), default="train-validation")
    parser.add_argument("--selection-manifest", type=Path)
    args = parser.parse_args(argv)
    if args.split == "test" and args.selection_manifest is None:
        parser.error("--split test requires --selection-manifest; test remains sealed")
    try:
        result = (
            download_test(args.data_root, args.manifest_dir, args.selection_manifest, args.workers)
            if args.split == "test"
            else download_train_validation(args.data_root, args.manifest_dir, args.workers)
        )
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted. Completed MRI series remain reusable.\n")
    except ArchiveDownloadError as exc:
        parser.exit(2, f"Download failed: {exc}\n")
    except Exception as exc:
        parser.exit(
            2,
            f"Download failed ({type(exc).__name__}). Completed MRI series remain reusable; retry.\n",
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
