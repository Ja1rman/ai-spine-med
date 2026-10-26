"""Bounded private DICOM-series uploads and isolated, real model inference.

Original DICOM bytes may contain personal data. They stay under the private
upload root, never in a public static directory; browser JSON contains only an
allowlist of numeric image fields and opaque identifiers. No untrusted model
files are accepted. Each production job runs in a disposable Python process.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import threading
import uuid
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

MAX_UPLOAD_BYTES = 256 * 1024 * 1024
MAX_EXPANDED_BYTES = 256 * 1024 * 1024
MAX_MEMBER_BYTES = 32 * 1024 * 1024
MAX_FILES = 512
MAX_STORED_BYTES = 1024 * 1024 * 1024
MAX_UPLOADS = 100
MAX_PENDING = 4
MAX_PIXELS = 32 * 1024 * 1024
JOB_TIMEOUT_SECONDS = 900
UPLOAD_ID = re.compile(r"^upload-[0-9a-f]{32}$")
MODEL_NAME = "SpineNet · frozen-head v1 · ансамбль 3 запусков"
_ASSETS = re.compile(
    r"^(?:images/slice-[0-9]{4}\.jpg|dicom/slice-[0-9]{4}\.dcm|series\.zip|result\.json)$"
)


class UploadRejected(ValueError):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(code)
        self.code, self.message, self.status = code, message, status


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def _update(job: Path, **fields):
    record = json.loads((job / "record.json").read_text())
    record.update(fields, updated_at_utc=_now())
    _write_json(job / "record.json", record)
    return record


def _size(root: Path):
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def _rejected(
    code="invalid_series",
    message="Не удалось прочитать одну сагиттальную МР-серию. Загрузите полный набор DICOM одной серии.",
):
    return UploadRejected(code, message)


def _extract(job: Path):
    """Manually copy allowed members; never use ZipFile.extract/all."""
    dicom = job / "dicom"
    dicom.mkdir(mode=0o700)
    count, expanded = 0, 0
    try:
        with zipfile.ZipFile(job / "source.zip") as archive:
            infos = archive.infolist()
            if len(infos) > MAX_FILES * 3:
                raise _rejected(
                    "too_many_files", "В архиве слишком много файлов; максимум 512 DICOM-срезов."
                )
            for entry in infos:
                parts = PurePosixPath(entry.filename).parts
                mode = entry.external_attr >> 16
                if (
                    not parts
                    or entry.filename.startswith("/")
                    or "\\" in entry.filename
                    or ":" in parts[0]
                    or "\x00" in entry.filename
                    or any(p in {".", ".."} for p in parts)
                    or stat.S_ISLNK(mode)
                    or entry.flag_bits & 1
                    or entry.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                ):
                    raise _rejected(
                        "unsafe_archive", "Архив содержит неподдерживаемые или небезопасные записи."
                    )
                if entry.is_dir() or any(p == "__MACOSX" or p.startswith(".") for p in parts):
                    continue
                count += 1
                expanded += entry.file_size
                if (
                    count > MAX_FILES
                    or not 0 < entry.file_size <= MAX_MEMBER_BYTES
                    or expanded > MAX_EXPANDED_BYTES
                    or entry.file_size > max(entry.compress_size, 1) * 200
                ):
                    raise _rejected(
                        "archive_limit",
                        "Архив превышает допустимый объём распаковки или число срезов.",
                    )
                target = dicom / f"file-{count:04d}.dcm"
                with archive.open(entry) as source, target.open("xb") as destination:
                    remaining = entry.file_size
                    while remaining:
                        chunk = source.read(min(remaining, 128 * 1024))
                        if not chunk:
                            raise _rejected()
                        destination.write(chunk)
                        remaining -= len(chunk)
                    if source.read(1):
                        raise _rejected()
                target.chmod(0o600)
    except UploadRejected:
        raise
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError):
        raise _rejected(
            "invalid_zip", "Файл не является поддерживаемым ZIP-архивом DICOM."
        ) from None
    if count < 2:
        raise _rejected(
            "single_image",
            "Нужна полная сагиттальная T2-серия минимум из двух DICOM-срезов; одиночная картинка не поддерживается.",
        )
    return dicom


def _prepare_series(job: Path):
    import numpy as np
    import pydicom
    from PIL import Image
    from pydicom.uid import EnhancedMRImageStorage, MRImageStorage

    from .dicom import DicomGeometryError, load_sagittal_series

    directory = _extract(job)
    files = sorted(directory.iterdir())
    headers, pixels = [], 0
    try:
        for path in files:
            with warnings.catch_warnings(record=True):
                ds = pydicom.dcmread(path, stop_before_pixels=True)
            rows, columns = int(ds.Rows), int(ds.Columns)
            pixels += rows * columns
            if (
                min(rows, columns) < 2
                or max(rows, columns) > 2048
                or pixels > MAX_PIXELS
                or int(getattr(ds, "NumberOfFrames", 1)) != 1
                or int(getattr(ds, "SamplesPerPixel", 1)) != 1
                or int(getattr(ds, "BitsAllocated", 0)) not in {8, 16}
                or str(getattr(ds, "Modality", "MR")) != "MR"
                or str(
                    getattr(ds, "SOPClassUID", getattr(ds.file_meta, "MediaStorageSOPClassUID", ""))
                )
                not in {str(MRImageStorage), str(EnhancedMRImageStorage)}
            ):
                raise _rejected(
                    "unsupported_dicom",
                    "Поддерживаются однослойные монохромные МР-DICOM размером до 2048×2048.",
                )
            headers.append(ds)
        if len({str(ds.SeriesInstanceUID) for ds in headers}) != 1:
            raise _rejected(
                "multiple_series",
                "В архиве несколько серий. Выберите одну сагиттальную T2-серию и загрузите её отдельно.",
            )
        # No arbitrary description, dates or patient fields enter browser JSON.
        # Acquisition geometry is checked strictly before any model execution.
        with warnings.catch_warnings(record=True):
            scan = load_sagittal_series(directory)
    except UploadRejected:
        raise
    except DicomGeometryError:
        raise _rejected(
            "invalid_geometry",
            "Серия не прошла проверку сагиттальной геометрии: проверьте полноту DICOM, порядок и отсутствие смешанных серий.",
        ) from None
    except Exception:
        raise _rejected() from None
    orientation = np.asarray(headers[0].ImageOrientationPatient, float)
    normal = np.cross(orientation[:3], orientation[3:])
    if normal[0] < 0:
        normal = -normal
    order = np.argsort([np.asarray(ds.ImagePositionPatient, float) @ normal for ds in headers])
    image_dir = job / "images"
    image_dir.mkdir(mode=0o700)
    low, high = np.percentile(scan.volume, [1, 99])
    if high <= low:
        high = low + 1
    slices = []
    identifier = job.name
    for i, source_index in enumerate(order):
        key = f"slice-{i + 1:04d}"
        destination = directory / f"{key}.dcm"
        files[int(source_index)].rename(destination)
        image = np.clip((scan.volume[:, :, i] - low) / (high - low), 0, 1)
        preview = Image.fromarray((image * 255).astype("uint8"))
        preview.thumbnail((1024, 1024))
        preview.save(image_dir / f"{key}.jpg", quality=90)
        (image_dir / f"{key}.jpg").chmod(0o600)
        instance = getattr(headers[int(source_index)], "InstanceNumber", None)
        try:
            instance = int(instance) if instance is not None else i + 1
        except (ValueError, TypeError):
            instance = i + 1
        slices.append(
            {
                "slice_key": key,
                "instance_number": instance,
                "rows": scan.volume.shape[0],
                "columns": scan.volume.shape[1],
                "source_rows": int(headers[int(source_index)].Rows),
                "source_columns": int(headers[int(source_index)].Columns),
                "image_url": f"api/uploads/{identifier}/images/{key}.jpg",
                "dicom_url": f"api/uploads/{identifier}/dicom/{key}.dcm",
            }
        )
    with zipfile.ZipFile(job / "series.zip", "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(directory.iterdir()):
            archive.write(path, arcname=path.name)
    (job / "series.zip").chmod(0o600)
    (job / "source.zip").unlink()
    patient = {
        "patient_id": identifier,
        "series": [
            {
                "series_key": "uploaded-series",
                "series_description": "Sagittal MRI",
                "sequence_subtype_verified": False,
                "n_slices": len(slices),
                "slices": slices,
                "zip_url": f"api/uploads/{identifier}/series.zip",
            }
        ],
    }
    _update(
        job,
        n_slices=len(slices),
        patient=patient,
        input_scope="Полная выбранная серия используется для локализации. Оценка каждого уровня — по одному блоку из 9 соседних срезов. Нужна сагиттальная T2-серия; её подтип автоматически не подтверждён.",
        sequence_subtype_verified=False,
        message="Серия проверена. Модель определяет уровни и оценивает стеноз…",
    )
    del scan, headers
    gc.collect()
    return directory


def _process_job(job: Path, configuration: dict, runner=None):
    try:
        directory = _prepare_series(job)
        if runner is None:
            from .inference import run_inference

            runner = run_inference
        report = runner(
            directory,
            job / "inference",
            checkpoint=configuration["heads"],
            cache_dir=configuration["cache"],
            device=configuration.get("device", "cpu"),
            threads=configuration.get("threads", 1),
        )
        targets = []
        for target in report["targets"]:
            ok = target["status"] == "ok"
            targets.append(
                {
                    "level": target["level"],
                    "status": target["status"],
                    "prediction": target["prediction"] if ok else None,
                    "probabilities": [
                        target[f"p_{name}"] for name in ("normal_mild", "moderate", "severe")
                    ]
                    if ok
                    else [],
                    "reason": target.get("reason", ""),
                }
            )
        status = {"ok": "completed", "partial": "partial"}.get(report["status"], "failed")
        messages = {
            "completed": "Анализ завершён: доступны оценки L4–L5 и L5–S1.",
            "partial": "Анализ завершён частично: один из уровней не удалось оценить.",
            "failed": "Модель не смогла оценить уровни. Это не означает отсутствие стеноза.",
        }
        _write_json(job / "result.json", report)
        _update(
            job,
            status=status,
            targets=targets,
            message=messages[status],
            report_url=f"api/uploads/{job.name}/result.json",
            completed_at_utc=_now(),
        )
    except UploadRejected as error:
        _update(
            job,
            status="failed",
            error_code=error.code,
            message=error.message,
            completed_at_utc=_now(),
        )
    except Exception:
        _update(
            job,
            status="failed",
            error_code="processing_failed",
            message="Обработка не завершилась. Проверьте формат серии; технические подробности не публикуются.",
            completed_at_utc=_now(),
        )


class UploadStore:
    """One running CPU job; bounded queue and private, persistent disk storage."""

    def __init__(
        self, root: Path, *, heads=(), cache=Path(".cache"), device="cpu", threads=1, runner=None
    ):
        self.root = Path(root).resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.root.chmod(0o700)
        self.configuration = {
            "heads": [str(Path(p).resolve()) for p in heads],
            "cache": str(Path(cache).resolve()),
            "device": device,
            "threads": threads,
        }
        self.configured = (
            len(heads) == 3 and all(Path(p).is_file() for p in heads) and Path(cache).is_dir()
        )
        self.runner = runner
        self.lock = threading.Lock()
        self.reservations = {}
        self.slots = threading.BoundedSemaphore(MAX_PENDING)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="private-mri")
        self.config_path = self.root / "worker-config.json"
        _write_json(self.config_path, self.configuration)
        for path in self.root.glob("upload-*/record.json"):
            try:
                if json.loads(path.read_text()).get("status") in {
                    "queued",
                    "processing",
                    "receiving",
                }:
                    _update(
                        path.parent,
                        status="failed",
                        error_code="interrupted",
                        message="Обработка прервалась при перезапуске сервера. Загрузите серию повторно.",
                    )
            except (ValueError, OSError):
                pass

    def list(self):
        records = []
        for path in self.root.glob("upload-*/record.json"):
            try:
                record = json.loads(path.read_text())
                if record.get("status") != "receiving":
                    records.append({k: v for k, v in record.items() if k != "patient"})
            except (ValueError, OSError):
                pass
        return {
            "schema_version": 1,
            "uploads": sorted(records, key=lambda r: r["created_at_utc"], reverse=True),
            "limits": {"max_upload_bytes": MAX_UPLOAD_BYTES, "max_files": MAX_FILES},
            "model": {"name": MODEL_NAME, "configured": self.configured},
        }

    def get(self, identifier):
        if not UPLOAD_ID.fullmatch(identifier):
            return None
        path = self.root / identifier / "record.json"
        try:
            record = json.loads(path.read_text())
            return record if record.get("status") != "receiving" else None
        except (ValueError, OSError):
            return None

    def receive(self, stream, length: int):
        if not self.configured:
            raise UploadRejected(
                "model_unavailable",
                "Анализ временно недоступен: модель не настроена на сервере.",
                503,
            )
        if type(length) is not int or not 0 < length <= MAX_UPLOAD_BYTES:
            raise UploadRejected("upload_limit", "Максимальный размер ZIP — 256 МиБ.", 413)
        if not self.slots.acquire(blocking=False):
            raise UploadRejected("queue_full", "Очередь анализа заполнена. Попробуйте позже.", 429)
        job = None
        try:
            with self.lock:
                reservation = MAX_EXPANDED_BYTES * 2 + length + 32 * 1024 * 1024
                reserved_remaining = sum(
                    max(0, size - _size(self.root / identifier))
                    for identifier, size in self.reservations.items()
                )
                if (
                    len(list(self.root.glob("upload-*/record.json"))) >= MAX_UPLOADS
                    or _size(self.root) + reserved_remaining + reservation > MAX_STORED_BYTES
                    or shutil.disk_usage(self.root).free < reservation + 128 * 1024 * 1024
                ):
                    raise UploadRejected(
                        "storage_full", "На сервере недостаточно места для новой серии.", 507
                    )
                identifier = "upload-" + uuid.uuid4().hex
                job = self.root / identifier
                job.mkdir(mode=0o700)
                self.reservations[identifier] = reservation
                record = {
                    "schema_version": 1,
                    "upload_id": identifier,
                    "created_at_utc": _now(),
                    "status": "receiving",
                    "n_slices": None,
                    "targets": [],
                    "model_name": MODEL_NAME,
                    "message": "Файл загружается…",
                }
                _write_json(job / "record.json", record)
                # Reserve the bounded storage budget before another receive starts.
                with (job / "source.zip").open("xb") as destination:
                    destination.truncate(length)
            with (job / "source.zip").open("r+b") as destination:
                remaining = length
                while remaining:
                    chunk = stream.read(min(remaining, 128 * 1024))
                    if not chunk:
                        raise UploadRejected("incomplete_upload", "Передача файла прервалась.")
                    destination.write(chunk)
                    remaining -= len(chunk)
            (job / "source.zip").chmod(0o600)
            result = _update(job, status="queued", message="Серия в очереди на анализ.")
            self.executor.submit(self._work, job)
            return result
        except Exception:
            if job is not None:
                shutil.rmtree(job, ignore_errors=True)
                with self.lock:
                    self.reservations.pop(job.name, None)
            self.slots.release()
            raise

    def _work(self, job):
        try:
            _update(job, status="processing", message="Проверяю DICOM-серию…")
            if self.runner is not None:
                _process_job(job, self.configuration, self.runner)
                return
            command = [
                sys.executable,
                "-m",
                "ai_spine_med.review_uploads",
                "--job",
                str(job),
                "--config",
                str(self.config_path),
            ]
            environment = dict(
                os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
            )
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=environment,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=JOB_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                _update(
                    job,
                    status="failed",
                    error_code="timeout",
                    message="Анализ превысил лимит времени. Результат не получен.",
                )
                return
            record = self.get(job.name)
            if code != 0 or record is None or record["status"] in {"queued", "processing"}:
                _update(
                    job,
                    status="failed",
                    error_code="worker_failed",
                    message="Обработка остановлена сервером. Результат не получен.",
                )
        except Exception:
            _update(
                job,
                status="failed",
                error_code="worker_failed",
                message="Обработка остановлена сервером. Результат не получен.",
            )
        finally:
            with self.lock:
                self.reservations.pop(job.name, None)
            self.slots.release()

    def asset(self, identifier, relative):
        if not UPLOAD_ID.fullmatch(identifier) or not _ASSETS.fullmatch(relative):
            return None
        root = (self.root / identifier).resolve()
        path = (root / relative).resolve()
        return path if path.is_relative_to(root) and path.is_file() else None

    def close(self):
        self.executor.shutdown(wait=False, cancel_futures=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    try:
        _process_job(args.job.resolve(), json.loads(args.config.read_text()))
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
