"""Prepare and explicitly launch a private, free T4 research extraction job."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import inspect
import io
import json
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .assets import sha256_file
from .kaggle_download import COMPETITION, _api
from .rsna_bulk import COMPLETE_MARKER, _selected_series, is_complete_series

KERNEL_ID = "jairman/ai-spine-med-frozen-features-v1"
_MANIFESTS = ("studies.csv", "series.csv", "labels.csv", "coordinates.csv", "summary.json")


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _cloud_bootstrap(payload_b64, payload_sha256):
    """Standalone cloud code: all imports are inside to keep the launch auditable."""
    import base64
    import csv
    import gzip
    import hashlib
    import io
    import json
    import os
    import re
    import shutil
    import signal
    import subprocess
    import sys
    import tarfile
    import time
    import tomllib
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path, PurePosixPath

    started = time.monotonic()
    temporary = Path("/kaggle/temp/ai-spine-med-frozen-v1")
    working = Path("/kaggle/working")
    temporary.mkdir(parents=True, exist_ok=False)
    working.mkdir(exist_ok=True)
    logs = temporary / "logs"
    logs.mkdir()
    processes = []
    report = {"stage": "starting", "payload_sha256": payload_sha256, "private": True}

    def sha_file(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def canonical(value):
        return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()

    def run(arguments, name):
        with (logs / name).open("w") as stream:
            subprocess.run(arguments, check=True, stdout=stream, stderr=subprocess.STDOUT)

    def timeout(_signum, _frame):
        raise TimeoutError("Research job reached its internal 85 minute limit")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(5100)
    try:
        blob = base64.b64decode(payload_b64, validate=True)
        if hashlib.sha256(blob).hexdigest() != payload_sha256:
            raise RuntimeError("Embedded payload checksum mismatch")
        with tarfile.open(fileobj=io.BytesIO(gzip.decompress(blob)), mode="r:") as archive:
            for member in archive.getmembers():
                name = PurePosixPath(member.name)
                if (
                    name.is_absolute()
                    or ".." in name.parts
                    or "\\" in member.name
                    or not member.isfile()
                ):
                    raise RuntimeError("Unsafe payload member")
                target = temporary.joinpath(*name.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
        bundle = json.loads((temporary / "bundle.json").read_text())
        for relative, expected in bundle["files_sha256"].items():
            if sha_file(temporary / relative) != expected:
                raise RuntimeError("Unchanged source/fixed manifest verification failed")

        import torch

        gpu_names = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
        if len(gpu_names) != 2 or not all("T4" in name for name in gpu_names):
            raise RuntimeError("This job requires exactly the free T4 x2 accelerator")
        report["gpu_names"] = gpu_names
        report["initial_torch"] = torch.__version__
        app = temporary / "app"
        environment = temporary / "venv"
        run([sys.executable, "-m", "venv", "--system-site-packages", str(environment)], "venv.log")
        python = environment / "bin/python"
        dependencies = tomllib.loads((app / "pyproject.toml").read_text())["project"][
            "dependencies"
        ]
        dependencies = [
            item
            for item in dependencies
            if re.match(r"[A-Za-z0-9_-]+", item).group(0).lower() not in ("torch", "torchvision")
        ]
        run(
            [str(python), "-m", "pip", "install", "--disable-pip-version-check", *dependencies],
            "dependencies.log",
        )
        run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-deps",
                str(app),
            ],
            "install.log",
        )
        print(
            json.dumps(
                {
                    "stage": "dependencies_ready",
                    "elapsed_seconds": round(time.monotonic() - started, 1),
                }
            ),
            flush=True,
        )
        run(
            [
                str(python),
                "-c",
                "import torch,torchvision,numpy,cv2; assert torch.cuda.device_count()==2; assert int(numpy.__version__.split('.')[0])<2; print(torch.__version__,torchvision.__version__,numpy.__version__,cv2.__version__)",
            ],
            "versions.log",
        )

        # Only official mounted competition images are used, never an external mirror.
        candidates = list(Path("/kaggle/input").rglob("train_series_descriptions.csv"))
        mounted = None
        for candidate in candidates:
            if (
                sha_file(candidate)
                == bundle["source_metadata_sha256"]["train_series_descriptions.csv"]
            ):
                mounted = candidate.parent
                break
        if mounted is None:
            raise RuntimeError(
                "The official mounted RSNA source metadata differs from the fixed local data"
            )
        data = temporary / "data"
        data.mkdir()
        for name, expected in bundle["source_metadata_sha256"].items():
            source = mounted / name
            if sha_file(source) != expected:
                raise RuntimeError("The official mounted RSNA metadata checksum differs")
            shutil.copyfile(source, data / name)
        series_expected = json.loads((temporary / "series_digests.json").read_text())
        manifests = temporary / "manifests"
        with (manifests / "series.csv").open() as stream:
            rows = [
                row for row in csv.DictReader(stream) if row["split"] in ("train", "validation")
            ]
        if len(rows) != bundle["n_series"]:
            raise RuntimeError("Fixed training/validation cohort count mismatch")

        def stage_series(row):
            relative = row["series_path"]
            if not re.fullmatch(r"train_images/[0-9]+/[0-9]+", relative):
                raise RuntimeError("Unsafe fixed series path")
            receipt_path = (
                temporary / "receipts" / (row["study_id"] + "_" + row["series_id"] + ".json")
            )
            receipt = json.loads(receipt_path.read_text())
            if receipt["manifest_sha256"] != bundle["manifest_receipt_sha256"]:
                raise RuntimeError("Completeness receipt changed its fixed manifest binding")
            source = mounted / relative
            actual_names = sorted(path.name for path in source.glob("*.dcm") if path.is_file())
            if actual_names != sorted(receipt["files"]):
                raise RuntimeError("Mounted RSNA DICOM instance names differ")
            records = [
                {
                    "name": name,
                    "bytes": (source / name).stat().st_size,
                    "sha256": sha_file(source / name),
                }
                for name in actual_names
            ]
            if (
                hashlib.sha256(canonical(records)).hexdigest()
                != series_expected[relative]["sha256"]
            ):
                raise RuntimeError("Mounted RSNA DICOM bytes differ from local verified images")
            final = data / relative
            final.parent.mkdir(parents=True, exist_ok=True)
            stage = final.with_name("." + final.name + ".staging")
            stage.mkdir()
            for name in actual_names:
                (stage / name).symlink_to(source / name)
            shutil.copyfile(receipt_path, stage / "_series_complete.json")
            stage.rename(final)
            return len(actual_names)

        with ThreadPoolExecutor(max_workers=4) as executor:
            checked = sum(executor.map(stage_series, rows))
        if checked != bundle["n_dicom_files"]:
            raise RuntimeError("Verified DICOM count mismatch")
        report["verified_series"] = len(rows)
        report["verified_dicom_files"] = checked
        print(
            json.dumps({"stage": "mounted_images_verified", "series": len(rows), "files": checked}),
            flush=True,
        )
        run(
            [
                str(python),
                "-m",
                "ai_spine_med",
                "fetch-assets",
                "--cache",
                str(temporary / "cache"),
                "--no-examples",
            ],
            "assets.log",
        )
        print(
            json.dumps(
                {"stage": "assets_ready", "elapsed_seconds": round(time.monotonic() - started, 1)}
            ),
            flush=True,
        )
        output = temporary / "artifacts/features-kaggle-v1"
        handles = []
        for index in range(2):
            env = dict(os.environ)
            env.update(
                CUDA_VISIBLE_DEVICES=str(index),
                OMP_NUM_THREADS="2",
                MKL_NUM_THREADS="2",
                OPENBLAS_NUM_THREADS="2",
            )
            handle = (logs / f"worker-{index}.log").open("w")
            handles.append(handle)
            arguments = [
                str(python),
                "-m",
                "ai_spine_med",
                "extract-features",
                "--data-root",
                str(data),
                "--manifest-dir",
                str(manifests),
                "--cache",
                str(temporary / "cache"),
                "--out",
                str(output),
                "--device",
                "cuda:0",
                "--threads",
                "2",
                "--worker-index",
                str(index),
                "--num-workers",
                "2",
            ]
            processes.append(
                subprocess.Popen(
                    arguments,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        last_progress = 0.0
        while any(process.poll() is None for process in processes):
            if any(process.poll() not in (None, 0) for process in processes):
                raise RuntimeError("A CUDA feature worker failed")
            if time.monotonic() - started > 5050:
                raise TimeoutError("Extraction reached its bounded runtime")
            if time.monotonic() - last_progress > 30:
                print(
                    json.dumps(
                        {
                            "stage": "extracting",
                            "completed_series": len(list((output / "records").glob("*.json"))),
                            "total_series": len(rows),
                            "elapsed_seconds": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )
                last_progress = time.monotonic()
            time.sleep(5)
        if any(process.returncode != 0 for process in processes):
            raise RuntimeError("A CUDA feature worker failed")
        for handle in handles:
            handle.close()
        summary = json.loads((output / "summary.json").read_text())
        if (
            not summary["extraction_complete"]
            or summary["n_requested_series"] != bundle["n_series"]
        ):
            raise RuntimeError("The CUDA feature cache is incomplete")
        report["stage"] = "complete"
        report["feature_summary"] = summary
    except BaseException as error:
        report["stage"] = "failed"
        report["error_type"] = type(error).__name__
        # Only our bounded diagnostic, never environment variables or remote requests.
        report["error_message"] = (
            str(error)[:500]
            if isinstance(error, (RuntimeError, TimeoutError))
            else "See private bootstrap logs"
        )
    finally:
        signal.alarm(0)
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
        report["elapsed_seconds"] = round(time.monotonic() - started, 1)
        (temporary / "run_summary.json").write_text(json.dumps(report, indent=2) + "\n")
        export = temporary / "export"
        export.mkdir()
        for name in ("manifests", "logs"):
            source = temporary / name
            if source.is_dir():
                shutil.copytree(source, export / name)
        source = temporary / "artifacts/features-kaggle-v1"
        if source.is_dir():
            target = export / "artifacts/features-kaggle-v1"
            target.parent.mkdir()
            shutil.copytree(source, target)
            if (source / "features.npz").is_file():
                shutil.copyfile(source / "features.npz", working / "features.npz")
        shutil.copyfile(temporary / "run_summary.json", export / "run_summary.json")
        archive_path = working / "features-kaggle-v1.tar.gz"
        with tarfile.open(archive_path, "w:gz", compresslevel=1) as archive:
            for path in sorted(export.rglob("*")):
                if path.is_file():
                    if path.suffix in (".dcm", ".pt", ".pth") or path.is_symlink():
                        raise RuntimeError("Raw MRI/weights must not be exported")
                    archive.add(path, arcname=str(path.relative_to(export)), recursive=False)
        if archive_path.stat().st_size >= 18 * 1024**3:
            raise RuntimeError("Output exceeded the bounded private research artifact budget")
        (working / "run_summary.json").write_text(json.dumps(report, indent=2) + "\n")
        hashes = {path.name: sha_file(path) for path in working.iterdir() if path.is_file()}
        (working / "output_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    if report["stage"] != "complete":
        raise SystemExit(1)


def prepare_launch(
    project_root: str | Path,
    data_root: str | Path,
    manifest_dir: str | Path,
    out: str | Path,
) -> dict[str, Any]:
    project, data, manifests, out = map(Path, (project_root, data_root, manifest_dir, out))
    selected, manifest_sha = _selected_series(manifests)
    contents = {"app/pyproject.toml": (project / "pyproject.toml").read_bytes()}
    source_files = sorted((project / "src/ai_spine_med").glob("*.py"))
    source_files.append(project / "src/ai_spine_med/assets.lock.json")
    for path in source_files:
        contents["app/" + str(path.relative_to(project))] = path.read_bytes()
    for name in _MANIFESTS:
        contents["manifests/" + name] = (manifests / name).read_bytes()
    original_summary = json.loads(contents["manifests/summary.json"])
    for name, expected in original_summary["input_sha256"].items():
        if sha256_file(data / name) != expected:
            raise ValueError("Local source metadata differs from the frozen manifest")

    def series_digest(relative):
        directory = data / relative
        if not is_complete_series(directory):
            raise ValueError("Every selected MRI series must be complete before cloud packaging")
        receipt_bytes = (directory / COMPLETE_MARKER).read_bytes()
        receipt = json.loads(receipt_bytes)
        if receipt["manifest_sha256"] != manifest_sha:
            raise ValueError("An MRI completeness receipt is bound to another manifest")
        records = [
            {
                "name": name,
                "bytes": (directory / name).stat().st_size,
                "sha256": sha256_file(directory / name),
            }
            for name in sorted(receipt["files"])
        ]
        return (
            relative,
            {
                "sha256": _sha(_canonical(records)),
                "files": len(records),
                "bytes": sum(record["bytes"] for record in records),
            },
            receipt_bytes,
        )

    digests = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        for relative, digest, receipt in executor.map(series_digest, sorted(selected)):
            digests[relative] = digest
            _, study, series = relative.split("/")
            contents[f"receipts/{study}_{series}.json"] = receipt
    contents["series_digests.json"] = _canonical(digests)
    bundle = {
        "schema_version": 1,
        "kernel_id": KERNEL_ID,
        "n_series": len(selected),
        "n_dicom_files": sum(value["files"] for value in digests.values()),
        "dicom_bytes": sum(value["bytes"] for value in digests.values()),
        "manifest_receipt_sha256": manifest_sha,
        "source_metadata_sha256": original_summary["input_sha256"],
        "files_sha256": {name: _sha(content) for name, content in sorted(contents.items())},
    }
    contents["bundle.json"] = _canonical(bundle)
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:") as archive:
        for name, content in sorted(contents.items()):
            entry = tarfile.TarInfo(name)
            entry.size = len(content)
            entry.mode = 0o644
            archive.addfile(entry, io.BytesIO(content))
    payload = gzip.compress(stream.getvalue(), compresslevel=9, mtime=0)
    encoded = base64.b64encode(payload).decode()
    script = "# Private, bounded T4 research extraction; no credentials or MRI bytes embedded.\n"
    script += "PAYLOAD_SHA256 = " + repr(_sha(payload)) + "\nPAYLOAD_B64 = (\n"
    script += "\n".join(repr(encoded[index : index + 120]) for index in range(0, len(encoded), 120))
    script += (
        "\n)\n\n"
        + inspect.getsource(_cloud_bootstrap)
        + "\n\nif __name__ == '__main__':\n    _cloud_bootstrap(PAYLOAD_B64, PAYLOAD_SHA256)\n"
    )
    if len(script.encode()) >= 1_000_000:
        raise ValueError(
            f"Prepared kernel source is {len(script.encode())} bytes, above the 1 MB source bound"
        )
    compile(script, "bootstrap.py", "exec")
    metadata = {
        "id": KERNEL_ID,
        "title": "Ai Spine Med Frozen Features V1",
        "code_file": "bootstrap.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "competition_sources": [COMPETITION],
        "dataset_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "bootstrap.py").write_text(script)
    metadata_content = (json.dumps(metadata, indent=2) + "\n").encode()
    (out / "kernel-metadata.json").write_bytes(metadata_content)
    receipt = {
        **{key: value for key, value in bundle.items() if key != "files_sha256"},
        "script_bytes": len(script.encode()),
        "payload_bytes": len(payload),
        "script_sha256": _sha(script.encode()),
        "payload_sha256": _sha(payload),
        "source_sha256": {
            str(path.relative_to(project)): sha256_file(path) for path in source_files
        },
        "metadata_sha256": _sha(metadata_content),
        "max_session_seconds": 5400,
        "internal_timeout_seconds": 5100,
        "launch_state": "prepared_only_not_launched",
    }
    (out / "preparation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def launch_prepared(folder: str | Path, *, api: Any = None) -> dict[str, Any]:
    folder = Path(folder)
    receipt = json.loads((folder / "preparation.json").read_text())
    metadata = json.loads((folder / "kernel-metadata.json").read_text())
    if (
        metadata["id"] != KERNEL_ID
        or metadata["is_private"] is not True
        or metadata["machine_shape"] != "NvidiaTeslaT4"
        or metadata["competition_sources"] != [COMPETITION]
        or any(metadata[name] for name in ("dataset_sources", "kernel_sources", "model_sources"))
    ):
        raise ValueError("Only the reviewed private T4 research configuration may be launched")
    if (
        sha256_file(folder / "bootstrap.py") != receipt["script_sha256"]
        or sha256_file(folder / "kernel-metadata.json") != receipt["metadata_sha256"]
        or receipt["max_session_seconds"] != 5400
    ):
        raise ValueError(
            "Prepared launch code or runtime budget changed; prepare and review it again"
        )
    client = api if api is not None else _api()
    quota = client.quota_view().gpu_quota
    if (
        quota is None
        or quota.is_pay_to_scale_enabled
        or (quota.total_time_allowed - quota.time_used - quota.time_reserved).total_seconds() < 5400
    ):
        raise ValueError(
            "A verified free GPU quota of at least 90 minutes is required; paid compute is excluded"
        )
    existing = client.kernels_list(
        mine=True, search="ai-spine-med-frozen-features-v1", page_size=100
    )
    if any(getattr(kernel, "ref", "") == KERNEL_ID for kernel in existing or []):
        previous = folder / "launch.json"
        if not previous.is_file() or json.loads(previous.read_text()).get("kernel") != KERNEL_ID:
            raise ValueError(
                "The kernel ID already exists without this launch's provenance; refusing to overwrite prior work"
            )
    result = client.kernels_push(str(folder), timeout="5400", acc="NvidiaTeslaT4")
    if result.error or result.invalid_competition_sources or result.invalid_dataset_sources:
        raise RuntimeError(
            "Kaggle rejected the reviewed private kernel or its official data source"
        )
    response = {
        "kernel": KERNEL_ID,
        "version": result.version_number,
        "kernel_id": result.kernel_id,
        "private": True,
        "max_session_seconds": 5400,
    }
    (folder / "launch.json").write_text(json.dumps(response, indent=2) + "\n")
    return response


def status(*, api: Any = None) -> dict[str, Any]:
    value = (api if api is not None else _api()).kernels_status(KERNEL_ID)
    return {"kernel": KERNEL_ID, "status": str(value.status)}


def download_outputs(
    destination: str | Path, *, small_only: bool = False, api: Any = None
) -> dict[str, Any]:
    """Stream only allowlisted private outputs; signed URLs never enter reports."""
    import requests
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    allowed = {"features.npz", "run_summary.json", "output_sha256.json"}
    if not small_only:
        allowed.add("features-kaggle-v1.tar.gz")
    client = api if api is not None else _api()
    links = {}
    token = ""
    seen = set()
    with client.build_kaggle_client() as sdk:
        while True:
            request = ApiListKernelSessionOutputRequest()
            request.user_name, request.kernel_slug = KERNEL_ID.split("/")
            request.page_size, request.page_token = 100, token
            result = sdk.kernels.kernels_api_client.list_kernel_session_output(request)
            for item in result.files or []:
                if item.file_name in allowed:
                    links[item.file_name] = item.url
            token = result.next_page_token
            if not token:
                break
            if token in seen:
                raise RuntimeError("Private output listing repeated a page token")
            seen.add(token)
    if not links or "output_sha256.json" not in links:
        raise ValueError("The private kernel has no finalized research outputs yet")
    try:
        for name in sorted(links, key=lambda name: name != "output_sha256.json"):
            target = destination / name
            partial = target.with_name(target.name + ".part")
            with requests.get(links[name], stream=True, timeout=(20, 120)) as response:
                response.raise_for_status()
                received = 0
                with partial.open("wb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        received += len(chunk)
                        if received > 18 * 1024**3:
                            raise RuntimeError("A private output exceeded its size budget")
                        stream.write(chunk)
                expected = response.headers.get("Content-Length")
                if expected and received != int(expected):
                    raise RuntimeError("A private output download was truncated")
            partial.replace(target)
        expected_hashes = json.loads((destination / "output_sha256.json").read_text())
        for name in links:
            if (
                name != "output_sha256.json"
                and sha256_file(destination / name) != expected_hashes[name]
            ):
                raise RuntimeError("A private research output failed its SHA256 check")
    except requests.RequestException:
        raise RuntimeError(
            "Private output download failed; signed response details were omitted"
        ) from None
    return {"kernel": KERNEL_ID, "files": sorted(links), "sha256_verified": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--project-root", type=Path, default=Path.cwd())
    prepare.add_argument("--data-root", type=Path, default=Path("data/rsna"))
    prepare.add_argument("--manifest-dir", type=Path, default=Path("artifacts/rsna-v1"))
    prepare.add_argument("--out", type=Path, default=Path("reports/runs/kaggle-launch-v1"))
    launch = sub.add_parser("launch")
    launch.add_argument("--folder", type=Path, default=Path("reports/runs/kaggle-launch-v1"))
    sub.add_parser("status")
    download = sub.add_parser("download")
    download.add_argument("--out", type=Path, required=True)
    download.add_argument("--small-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            value = prepare_launch(args.project_root, args.data_root, args.manifest_dir, args.out)
        elif args.command == "launch":
            value = launch_prepared(args.folder)
        elif args.command == "status":
            value = status()
        else:
            value = download_outputs(args.out, small_only=args.small_only)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(
            2,
            f"Kaggle research job failed ({type(error).__name__}); no credentials were printed.\n",
        )
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
