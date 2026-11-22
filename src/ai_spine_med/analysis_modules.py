"""Optional study analyses, isolated from the primary stenosis model.

Module configuration is server-owned. Upstream output is reduced to an explicit
public schema and matched back to the uploaded DICOM by its full byte digest.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pydicom
from pydicom.uid import EnhancedMRImageStorage, MRImageStorage

DISC_COMMIT = "80d2aeff96b044aa9e0f1c585a6a0de73554c53c"
MODEL_COMMIT = "66fd963b4e7e4f9530e6cf7b983b23348712dcab"
LEVELS = ("L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1")
MAX_RESULT_BYTES = 96 * 1024 * 1024
MAX_WORK_BYTES = 768 * 1024 * 1024
MAX_LOG_BYTES = 2 * 1024 * 1024
MIN_HOST_AVAILABLE_BYTES = 384 * 1024 * 1024
_SLICE = re.compile(r"^slice-[0-9]{4}$")
_FLAG = re.compile(r"^[a-z][a-z0-9_]{0,95}$")
_CANDIDATES = {"extrusion", "protrusion", "indeterminate"}
_SOURCE_HASHES = {
    "__init__.py": "9c338723ebd327f372899af9d6e8f12ded56b85256e45ead4243075e432fed64",
    "__main__.py": "79ee6f134c5c5193377cf9b189cbdcc3e2095e49ba9cec310114441c15a917c4",
    "_measurements.py": "d115e84e65131c17dea9ccac920d164e19bc670f46fbb91458efa9bdb1fb732e",
    "_spatial.py": "982bf63445922f181d0ad2c229895187d4fc07288fd15d122431dfd6d66bd500",
    "backend.py": "e3bc6c49ac29d577804fc85e6f184bee61eee8912c4174a505e7538556fe365e",
    "cli.py": "7e2da35308d56d67bc6c6a4dd0149c618c62503dfee1edf23ae37128439c8d5d",
    "demo.py": "fd1bb12b289030410047a578d112f1b74e4620a5dd94f8bc8d33ec69b8fcb520",
    "dicom.py": "02b5f6ffea3e6830b1e8c5a5e2156f079ec1acbce20df9fba353cf1f69d945df",
    "geometry.py": "673b9cd13fbce16f68d8cbb9ed8f5d8709adcf0c1182d36f50548703d99ff25c",
    "mask_io.py": "db08a6db4033f2a936deefd376584d724b06e338d62cc70e529c42be5de7dc01",
    "model_worker.py": "ebc044deec325ad10be47ab5fed402177663e58728299616ea49421e0d3d909b",
    "pipeline.py": "098ab8976d183bce80b9c49e933eeb80363e043ed2d49b6864e0fbbada671cff",
    "reference.py": "e2a368682c681d4d72c5950d6ce69f68f58a2c8e4c576f6fb7de3f604aef3d2e",
}
_LAUNCHER = """
import hashlib, importlib.util, pathlib, sys
spec = importlib.util.find_spec('disc_morphology')
if spec is None or spec.origin is None:
    raise RuntimeError('Disc morphology is not installed')
root = pathlib.Path(spec.origin).parent
expected = __HASHES__
for name, digest in expected.items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
        raise RuntimeError('Disc morphology source differs from the pinned release')
from disc_morphology.cli import main
sys.exit(main(sys.argv[1:]))
""".replace("__HASHES__", repr(_SOURCE_HASHES))


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _process_parents():
    """Read process ancestry, including descendants which created new sessions."""
    proc = Path("/proc")
    parents = {}
    if proc.is_dir():
        for path in proc.glob("[0-9]*/stat"):
            try:
                # The command name can contain spaces or parentheses.
                parts = path.read_text().rsplit(")", 1)[1].split()
                parents[int(path.parent.name)] = int(parts[1])
            except (OSError, ValueError, IndexError):
                continue
        return parents
    result = subprocess.run(
        ["ps", "-axo", "pid=,ppid="], capture_output=True, text=True, timeout=5, check=True
    )
    for row in result.stdout.splitlines():
        pid, parent = map(int, row.split())
        parents[pid] = parent
    return parents


def terminate_process_tree(pid: int):
    """Stop a worker and its model subprocesses before removing temporary files."""
    if pid <= 1 or pid == os.getpid():
        raise ValueError("Refusing to terminate this process")
    try:
        os.kill(pid, signal.SIGSTOP)
    except ProcessLookupError:
        return
    descendants = {pid}
    try:
        previous = set()
        while descendants != previous:
            previous = descendants.copy()
            parents = _process_parents()
            new = {p for p, parent in parents.items() if parent in descendants} - descendants
            for child in new:
                try:
                    os.kill(child, signal.SIGSTOP)
                except ProcessLookupError:
                    pass
            descendants.update(new)
    finally:
        for child in sorted(descendants - {pid}) + [pid]:
            try:
                os.kill(child, signal.SIGKILL)
            except ProcessLookupError:
                pass


def _source_slices(directory):
    sources, series = {}, set()
    for path in sorted(directory.glob("*.dcm")):
        if not _SLICE.fullmatch(path.stem) or path.is_symlink():
            raise ValueError("Input must contain prepared upload slices")
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        missing_modality = not str(getattr(ds, "Modality", ""))
        if missing_modality and str(
            getattr(ds, "SOPClassUID", getattr(ds.file_meta, "MediaStorageSOPClassUID", ""))
        ) not in {str(MRImageStorage), str(EnhancedMRImageStorage)}:
            raise ValueError("Missing modality requires verified MR image storage")
        series.add(str(ds.SeriesInstanceUID))
        orientation = np.asarray(ds.ImageOrientationPatient, dtype=float)
        col, row = orientation[:3], orientation[3:]
        transform = []
        rows, columns = int(ds.Rows), int(ds.Columns)
        if abs(col[2]) > 0.8 and abs(row[1]) > 0.8:
            col, row = row, col
            rows, columns = columns, rows
            transform.append("transpose")
        if col[1] < 0:
            transform.append("flip_columns")
        if row[2] > 0:
            transform.append("flip_rows")
        digest = _digest(path)
        if digest in sources:
            raise ValueError("Duplicate source slice")
        sources[digest] = {
            "slice_key": path.stem,
            "width": columns,
            "height": rows,
            "display_transform": transform,
            "missing_modality": missing_modality,
        }
    if not 3 <= len(sources) <= 512 or len(series) != 1:
        raise ValueError("Expected one spatial sagittal series")
    return sources, next(iter(series))


def _number(value):
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1e6:
        raise ValueError("Invalid measurement")
    return value


def _flags(flags):
    if not isinstance(flags, list) or len(flags) > 64:
        raise ValueError("Invalid quality flags")
    if any(not isinstance(flag, str) or not _FLAG.fullmatch(flag) for flag in flags):
        raise ValueError("Invalid quality flag")
    return list(dict.fromkeys(flags))


def _points(points, width, height):
    if points is None:
        return None
    if not isinstance(points, list) or len(points) > 32768:
        raise ValueError("Invalid overlay")
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("Invalid overlay point")
        for n, limit in zip(point, (width, height)):
            if type(n) not in (int, float) or not math.isfinite(n) or abs(n) > 8 * limit:
                raise ValueError("Invalid overlay coordinate")
    return points


def normalize_disc_result(payload, sources):
    """Allowlist an actual pinned model result; exclude image bytes, UIDs and paths."""
    if (
        payload.get("schema_version") != "1.0"
        or payload.get("tool_version") != "0.1.0"
        or payload.get("synthetic") is not False
        or payload.get("model", {}).get("inference_performed") is not True
        or payload.get("model", {}).get("source_commit") != MODEL_COMMIT
        or payload.get("clinically_validated") is not False
    ):
        raise ValueError("Unexpected model result provenance")
    images = payload.get("images")
    cases = payload.get("cases")
    if not isinstance(images, dict) or len(images) != len(sources):
        raise ValueError("Source image inventory differs")
    if not isinstance(cases, list) or len(cases) != len(LEVELS):
        raise ValueError("Unexpected level inventory")
    levels, seen = [], set()
    point_budget = 200000
    for case in cases:
        level = case["level"]
        if level not in LEVELS or level in seen:
            raise ValueError("Invalid or duplicate level")
        seen.add(level)
        voxels = case["candidate_voxels"]
        if type(voxels) is not int or not 0 <= voxels <= 32 * 1024 * 1024:
            raise ValueError("Invalid mask volume")
        decision = case["decision"]
        if (
            decision.get("clinical_classification") != "indeterminate"
            or decision.get("geometric_candidate") not in _CANDIDATES
            or decision.get("absence_assessed") is not False
        ):
            raise ValueError("Unexpected classification scope")
        planes = case["planes"]
        sagittal = [p for p in planes if p.get("plane") == "sagittal"]
        if len(sagittal) != 1 or any(
            p.get("frames") for p in planes if p.get("plane") != "sagittal"
        ):
            raise ValueError("Unexpected output plane")
        plane = sagittal[0]
        raw_frames = plane["frames"]
        if len(raw_frames) != len(sources):
            raise ValueError("Incomplete frame inventory")
        selected = plane["selected_index"]
        if type(selected) is not int or not 0 <= selected < len(raw_frames):
            raise ValueError("Invalid representative frame")
        frames, frame_keys = [], set()
        for frame in raw_frames:
            image = images[frame["image_id"]]
            source = sources.get(image["sha256"])
            if source is None or any(
                image.get(k) != source[k] for k in ("width", "height", "display_transform")
            ):
                raise ValueError("Frame does not match the canonical upload preview")
            if source["slice_key"] in frame_keys:
                raise ValueError("Duplicate frame result")
            frame_keys.add(source["slice_key"])
            record = frame["record"]
            if record.get("geometric_candidate") not in _CANDIDATES:
                raise ValueError("Unexpected frame classification")
            measurable = record.get("geometry_measurable")
            if type(measurable) is not bool or frame.get("mask_source") != "sagittal_model_mask":
                raise ValueError("Unexpected measurement or mask source")
            polygons = frame["mask_polygons_xy"]
            if not isinstance(polygons, list) or len(polygons) > 4096:
                raise ValueError("Invalid mask contours")
            for polygon in polygons:
                _points(polygon, source["width"], source["height"])
                point_budget -= len(polygon)
            if point_budget < 0:
                raise ValueError("Overlay size limit exceeded")
            segments = record.get("segments_xy", {})
            frames.append(
                {
                    "slice_key": source["slice_key"],
                    "plane": "sagittal",
                    "width": source["width"],
                    "height": source["height"],
                    "B_mm": _number(record.get("candidate_base_width_mm")),
                    "D_mm": _number(record.get("candidate_displaced_width_mm")),
                    "H_mm": _number(record.get("candidate_depth_mm")),
                    "D_over_B": _number(record.get("candidate_ratio")),
                    "uncertainty_mm": _number(record.get("uncertainty_mm")),
                    "geometry_measurable": measurable,
                    "geometric_candidate": record["geometric_candidate"],
                    "quality_flags": _flags(record.get("quality_flags", [])),
                    "mask_polygons_xy": polygons,
                    "reference_line_xy": _points(
                        frame.get("reference_line_xy"), source["width"], source["height"]
                    ),
                    "segments_xy": {
                        name: _points(segments.get(name), source["width"], source["height"])
                        for name in ("base", "displaced", "depth")
                    },
                    "mask_source": "sagittal_model_mask",
                }
            )
        levels.append(
            {
                "level": level,
                "candidate_voxels": voxels,
                "status": "candidate_needs_review"
                if voxels
                else "no_candidate_mask_not_an_absence_diagnosis",
                "geometric_candidate": decision["geometric_candidate"],
                "clinical_status": "indeterminate",
                "quality_flags": _flags(decision.get("quality_flags", [])),
                "representative_slice_key": frames[selected]["slice_key"],
                "frames": frames,
            }
        )
    return {
        "module_id": "disc_morphology",
        "status": "completed",
        "message": "Рассчитана геометрия кандидатных масок дисков.",
        "tool_version": "0.1.0",
        "source_commit": DISC_COMMIT,
        "model_source_commit": MODEL_COMMIT,
        "clinical_status": "indeterminate",
        "classification_scope": "unvalidated_research_candidate_geometry",
        "levels": levels,
    }


def _failed(code, message):
    return {
        "module_id": "disc_morphology",
        "status": "failed",
        "error_code": code,
        "message": message,
        "clinical_status": "indeterminate",
        "levels": [],
    }


def _workspace_bytes(root):
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def _host_memory_bytes(meminfo=Path("/proc/meminfo")):
    """Combined host RAM/swap headroom; container limits remain separate."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        values = {}
        for line in meminfo.read_text().splitlines():
            parts = line.split()
            if parts and parts[0] in {"MemAvailable:", "SwapFree:"}:
                if len(parts) != 3 or parts[2] != "kB" or int(parts[1]) < 0:
                    return None
                values[parts[0]] = int(parts[1]) * 1024
        if set(values) == {"MemAvailable:", "SwapFree:"}:
            return sum(values.values())
    except (OSError, ValueError):
        pass
    return None


def _memory_unavailable():
    return _failed(
        "resource_unavailable",
        "На сервере недостаточно свободной памяти для морфологии. Попробуйте повторить анализ позже.",
    )


def run_disc_morphology(directory: Path, job: Path, config: dict):
    if not config.get("enabled", False):
        return {
            "module_id": "disc_morphology",
            "status": "disabled",
            "message": "Модуль морфологии дисков не подключён.",
            "levels": [],
        }
    work = job / (".disc-morphology-" + uuid.uuid4().hex)
    process = None
    try:
        python = Path(config["python"])
        home = Path(config["model_home"])
        timeout, threads = config.get("timeout_seconds", 3600), config.get("threads", 2)
        device = config.get("device", "cpu")
        if (
            not python.is_absolute()
            or not python.is_file()
            or not home.is_absolute()
            or not (home / "config.json").is_file()
            or type(timeout) is not int
            or not 1 <= timeout <= 7200
            or type(threads) is not int
            or not 1 <= threads <= 16
            or device not in {"cpu", "cuda"}
        ):
            raise ValueError("Invalid module configuration")
        available = _host_memory_bytes()
        if available is not None and available < MIN_HOST_AVAILABLE_BYTES:
            return _memory_unavailable()
        sources, series_uid = _source_slices(directory)
        command = [
            str(python),
            "-c",
            _LAUNCHER,
            "analyze",
            str(directory),
            "--output",
            str(work),
            "--model-home",
            str(home),
            "--sagittal-series",
            series_uid,
            "--device",
            device,
            "--threads",
            str(threads),
            "--timeout",
            str(timeout),
            "--no-open",
        ]
        if any(source.get("missing_modality") for source in sources.values()):
            command.append("--allow-missing-modality")
        log_path = job / "disc-morphology.log"
        with log_path.open("wb") as log:
            log_path.chmod(0o600)
            environment = dict(
                os.environ,
                OMP_NUM_THREADS=str(threads),
                OPENBLAS_NUM_THREADS="1",
                MKL_NUM_THREADS=str(threads),
            )
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
            deadline = time.monotonic() + timeout + 120
            while process.poll() is None:
                available = _host_memory_bytes()
                if available is not None and available < MIN_HOST_AVAILABLE_BYTES:
                    terminate_process_tree(process.pid)
                    process.wait()
                    return _memory_unavailable()
                if time.monotonic() > deadline:
                    terminate_process_tree(process.pid)
                    process.wait()
                    return _failed("timeout", "Модуль морфологии превысил время обработки.")
                if (
                    _workspace_bytes(work) > MAX_WORK_BYTES
                    or log_path.stat().st_size > MAX_LOG_BYTES
                ):
                    terminate_process_tree(process.pid)
                    process.wait()
                    return _failed(
                        "output_limit", "Модуль морфологии превысил допустимый объём результата."
                    )
                time.sleep(0.5)
        if process.returncode != 0:
            return _failed("inference_failed", "Модуль морфологии не смог обработать эту серию.")
        result = work / "result.json"
        if not result.is_file() or not 0 < result.stat().st_size <= MAX_RESULT_BYTES:
            raise ValueError("Missing or oversized result")

        def reject_constant(value):
            raise ValueError("Nonfinite JSON number")

        payload = json.loads(result.read_text(), parse_constant=reject_constant)
        return normalize_disc_result(payload, sources)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError):
        return _failed(
            "module_unavailable",
            "Результат морфологии недоступен. Требуется проверка модуля и серии.",
        )
    finally:
        if process is not None and process.poll() is None:
            terminate_process_tree(process.pid)
            process.wait()
        shutil.rmtree(work, ignore_errors=True)


def run_analysis_modules(directory: Path, job: Path, configuration: dict):
    """Run configured, known adapters; callers own model/storage resource budgets."""
    modules = configuration.get("modules", {})
    return {
        "disc_morphology": run_disc_morphology(directory, job, modules.get("disc_morphology", {}))
    }


def patient_summary(targets: list[dict], modules: dict):
    findings, unassessed = [], []
    grades = {0: "норма или лёгкая степень", 1: "умеренная степень", 2: "тяжёлая степень"}
    for level in ("L4-L5", "L5-S1"):
        candidates = [target for target in targets if target.get("level") == level]
        target = candidates[0] if len(candidates) == 1 else {}
        prediction = target.get("prediction")
        if target.get("status") == "ok" and type(prediction) is int and prediction in grades:
            findings.append(f"{level}: модель центрального стеноза — {grades[prediction]}.")
        else:
            unassessed.append(f"{level}: степень центрального стеноза не оценена.")
    morphology = modules.get("disc_morphology", {})
    if morphology.get("status") == "completed":
        for level in morphology.get("levels", []):
            name = level["level"]
            if not level["candidate_voxels"]:
                unassessed.append(
                    f"{name}: кандидатная маска диска не получена, наличие грыжи не оценено."
                )
                continue
            candidate = level["geometric_candidate"]
            hypothesis = {
                "extrusion": "геометрическая гипотеза экструзии",
                "protrusion": "геометрическая гипотеза протрузии",
                "indeterminate": "тип формы не определён",
            }[candidate]
            detail = ""
            frames = [
                frame
                for frame in level.get("frames", [])
                if frame.get("slice_key") == level.get("representative_slice_key")
            ]
            if len(frames) == 1:
                frame = frames[0]
                quantities = []
                for field, label, unit, precision in (
                    ("B_mm", "B", " мм", 1),
                    ("D_mm", "D", " мм", 1),
                    ("H_mm", "H", " мм", 1),
                    ("D_over_B", "D/B", "", 2),
                ):
                    value = frame.get(field)
                    if value is not None:
                        formatted = f"{value:.{precision}f}".replace(".", ",")
                        quantities.append(f"{label} = {formatted}{unit}")
                if quantities:
                    detail = " На представительном срезе " + ", ".join(quantities) + "."
                    uncertainty = frame.get("uncertainty_mm")
                    if uncertainty is not None:
                        floor = f"{uncertainty:.1f}".replace(".", ",")
                        detail += f" Нижняя оценка неопределённости длины {floor} мм."
                    if frame.get("geometry_measurable") is not True:
                        detail += " Пригодность геометрии не подтверждена."
                    detail += " Погрешность сегментации не оценена."
            findings.append(
                f"{name}: получена кандидатная маска, {hypothesis}. Морфологический статус требует проверки.{detail}"
            )
    else:
        unassessed.append("Морфология дисков не оценена.")
    return {
        "status": "research",
        "text": "Сводка автоматического анализа исследования. Степени стеноза и геометрия масок требуют проверки по исходным снимкам.",
        "findings": findings,
        "unassessed": unassessed,
    }
