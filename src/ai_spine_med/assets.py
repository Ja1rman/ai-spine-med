"""Fetch the pinned original source, weights and author-provided example scans.

No source is vendored into Git. Each download is recorded with its SHA-256, URL,
and byte count. Hashes record the fetched artifact, not an author-signed checksum.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import requests

SPINENET_COMMIT = "5e96ac54b1c1845a37d12e40bfd16f647d40acd8"
SOURCE_URL = f"https://codeload.github.com/rwindsor1/SpineNet/zip/{SPINENET_COMMIT}"
BASE_URL = "https://www.robots.ox.ac.uk/~vgg/research/spinenet"
WEIGHTS = {
    "grading/ckpt1.pt": f"{BASE_URL}/weights/grading/ckpt1.pt",
    "appearance/ckpt187.pt": f"{BASE_URL}/weights/appearance/ckpt187.pt",
    "detect-vfr/ckpt435.pt": f"{BASE_URL}/weights/detect-vfr/ckpt435.pt",
    "context/ckpt16.pt": f"{BASE_URL}/weights/context/ckpt16.pt",
}
EXAMPLES = ("t2_lumbar_scan_1", "t2_lumbar_scan_2")
COMPATIBILITY_PATCH = {
    "path": "spinenet/utils/gen_utils.py",
    "before": "mask = np.zeros(shape, dtype=np.bool)",
    "after": "mask = np.zeros(shape, dtype=bool)",
    "reason": "np.bool was removed in NumPy 1.24; built-in bool is the identical dtype",
}


def locked_assets() -> dict:
    return json.loads(Path(__file__).with_name("assets.lock.json").read_text())


def expected_digest(url: str) -> str:
    lock = locked_assets()
    entries = [lock["source"], *lock["weights"].values(), *lock["examples"].values()]
    for entry in entries:
        if entry["url"] == url:
            return entry["sha256"]
    raise ValueError("URL is not in the reviewed asset lock")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def download(url: str, destination: Path) -> dict:
    """Reuse a verified cached file; an incomplete or changed file is re-fetched."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = destination.with_name(destination.name + ".download.json")
    if destination.is_file() and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if (
            receipt["url"] == url
            and receipt["sha256"] == expected_digest(url)
            and receipt["sha256"] == sha256_file(destination)
        ):
            return receipt
    partial = destination.with_name(destination.name + ".part")
    print(f"Downloading {destination.name}", flush=True)
    try:
        with requests.get(url, stream=True, timeout=(20, 60)) as response:
            response.raise_for_status()
            if "text/html" in response.headers.get("content-type", "").lower():
                raise ValueError(f"Expected a binary asset, received HTML: {url}")
            with partial.open("wb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    stream.write(chunk)
            expected = response.headers.get("content-length")
            if expected and not response.headers.get("content-encoding"):
                if partial.stat().st_size != int(expected):
                    raise ValueError(f"Incomplete asset: {destination.name}")
        if partial.stat().st_size == 0:
            raise ValueError(f"Empty asset: {destination.name}")
        receipt = {
            "url": url,
            "bytes": partial.stat().st_size,
            "sha256": sha256_file(partial),
        }
        if receipt["sha256"] != expected_digest(url):
            raise ValueError(
                f"Downloaded asset differs from the reviewed SHA-256: {destination.name}"
            )
        partial.replace(destination)
        write_json(receipt_path, receipt)
        return receipt
    finally:
        partial.unlink(missing_ok=True)


def extract_zip(archive: Path, destination: Path) -> None:
    """Reject traversal and links before extracting an archive into a fresh directory."""
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for info in zipped.infolist():
            name = PurePosixPath(info.filename)
            mode = info.external_attr >> 16
            if (
                name.is_absolute()
                or ".." in name.parts
                or "\\" in info.filename
                or stat.S_ISLNK(mode)
            ):
                raise ValueError(f"Unsafe ZIP entry: {info.filename}")
            if not (destination / info.filename).resolve().is_relative_to(destination):
                raise ValueError("ZIP entry escapes destination")
        zipped.extractall(destination)


def source_dir(cache: Path) -> Path:
    return cache / "vendor" / f"SpineNet-{SPINENET_COMMIT}"


def apply_compatibility_patch(upstream: Path) -> None:
    """Apply one audited, semantics-preserving old-NumPy compatibility fix."""
    patch = COMPATIBILITY_PATCH
    path = upstream / patch["path"]
    text = path.read_text()
    if text.count(patch["before"]) == 1:
        path.write_text(text.replace(patch["before"], patch["after"]))
    elif text.count(patch["after"]) != 1:
        raise ValueError("Pinned upstream does not match the expected compatibility patch")


def fetch_assets(cache: Path, examples_root: Path, *, with_examples: bool = True) -> dict:
    cache = cache.resolve()
    archive = cache / "downloads" / f"SpineNet-{SPINENET_COMMIT}.zip"
    source_receipt = download(SOURCE_URL, archive)
    upstream = source_dir(cache)
    if not upstream.exists():
        upstream.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=upstream.parent) as staging:
            extract_zip(archive, Path(staging))
            unpacked = Path(staging) / upstream.name
            if not (unpacked / "spinenet" / "main.py").is_file():
                raise ValueError("Pinned source archive has an unexpected layout")
            shutil.move(str(unpacked), upstream)
    apply_compatibility_patch(upstream)
    records = {}
    for relative_path, url in WEIGHTS.items():
        records[relative_path] = download(url, upstream / "spinenet" / "weights" / relative_path)
    examples = {}
    if with_examples:
        for name in EXAMPLES:
            zip_path = cache / "downloads" / f"{name}.zip"
            examples[name] = download(f"{BASE_URL}/example_scans/{name}.zip", zip_path)
            target = examples_root / name
            if not target.exists():
                examples_root.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=examples_root) as staging:
                    extract_zip(zip_path, Path(staging))
                    unpacked = Path(staging) / name
                    if not unpacked.is_dir():
                        raise ValueError(f"Unexpected example archive layout: {name}")
                    shutil.move(str(unpacked), target)
    manifest = {
        "upstream_commit": SPINENET_COMMIT,
        "source": source_receipt,
        "weights": records,
        "examples": examples,
        "compatibility_patch": COMPATIBILITY_PATCH,
        "license": "SpineNet non-commercial research license; see upstream LICENCE.md",
    }
    write_json(cache / "assets.json", manifest)
    return manifest


def verify_assets(cache: Path) -> dict:
    """Check all downloaded weights again before torch deserialisation."""
    manifest = json.loads((cache / "assets.json").read_text())
    if manifest["upstream_commit"] != SPINENET_COMMIT:
        raise ValueError("Cached source version differs from the pinned commit")
    upstream = source_dir(cache)
    for relative, url in WEIGHTS.items():
        record = manifest["weights"][relative]
        path = upstream / "spinenet" / "weights" / relative
        if (
            record["url"] != url
            or record["sha256"] != expected_digest(url)
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"Weight integrity check failed: {relative}; run fetch-assets")
    return manifest
