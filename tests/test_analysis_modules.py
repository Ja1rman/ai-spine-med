"""Optional module boundary and privacy regressions using synthetic payloads."""

import copy
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ai_spine_med import analysis_modules as module


def payload_fixture():
    sources = {}
    images, frames = {}, []
    for i in range(3):
        digest = hashlib.sha256(f"synthetic-{i}".encode()).hexdigest()
        source = {
            "slice_key": f"slice-{i + 1:04d}",
            "width": 16,
            "height": 16,
            "display_transform": [],
        }
        sources[digest] = source
        image_id = digest[:16]
        images[image_id] = {**source, "sha256": digest, "png": "SECRET_IMAGE_BYTES"}
        frames.append(
            {
                "image_id": image_id,
                "mask_polygons_xy": [[[4, 4], [6, 4], [6, 8], [4, 8]]],
                "reference_line_xy": [[4, 0], [4, 15]],
                "mask_source": "sagittal_model_mask",
                "record": {
                    "image_ref": {"series_id": "SECRET_UID"},
                    "geometry_measurable": True,
                    "geometric_candidate": "extrusion",
                    "candidate_base_width_mm": 2.5,
                    "candidate_displaced_width_mm": 7.0,
                    "candidate_depth_mm": 3.0,
                    "candidate_ratio": 2.8,
                    "uncertainty_mm": 1.0,
                    "quality_flags": ["reference_unreviewed", "segmentation_needs_review"],
                    "detail": "SECRET_PATIENT_NAME",
                    "segments_xy": {"base": [[4, 4], [4, 8]]},
                },
            }
        )
    cases = []
    for level in module.LEVELS:
        cases.append(
            {
                "level": level,
                "candidate_voxels": 18,
                "decision": {
                    "clinical_classification": "indeterminate",
                    "geometric_candidate": "extrusion",
                    "absence_assessed": False,
                    "quality_flags": ["numbering_unreviewed"],
                },
                "planes": [
                    {"plane": "sagittal", "frames": copy.deepcopy(frames), "selected_index": 1},
                    {"plane": "axial", "frames": [], "selected_index": 0},
                ],
                "reference": {"source_path": "SECRET_PATH"},
            }
        )
    payload = {
        "schema_version": "1.0",
        "tool_version": "0.1.0",
        "synthetic": False,
        "model": {
            "inference_performed": True,
            "source_commit": module.MODEL_COMMIT,
            "stage_a": {"private": "SECRET_MODEL_PATH"},
        },
        "clinically_validated": False,
        "images": images,
        "cases": cases,
        "study_id": "SECRET_STUDY_UID",
    }
    return payload, sources


def test_normalizes_only_public_fields_and_matches_by_digest():
    payload, sources = payload_fixture()
    result = module.normalize_disc_result(payload, sources)
    assert result["status"] == "completed"
    assert "SECRET" not in json.dumps(result)
    assert result["clinical_status"] == "indeterminate"
    level = result["levels"][0]
    assert level["representative_slice_key"] == "slice-0002"
    assert level["clinical_status"] == "indeterminate"
    frame = level["frames"][0]
    assert frame["B_mm"] == 2.5 and frame["D_over_B"] == 2.8
    assert frame["reference_line_xy"] == [[4, 0], [4, 15]]
    assert "reference_unreviewed" in frame["quality_flags"]


@pytest.mark.parametrize(
    "alter",
    [
        lambda p: p.update(synthetic=True),
        lambda p: p["model"].update(inference_performed=False),
        lambda p: p["model"].update(source_commit="wrong"),
        lambda p: p["cases"][0]["decision"].update(clinical_classification="extrusion"),
        lambda p: p["cases"][0]["planes"][0]["frames"][0]["record"].update(
            candidate_ratio=float("nan")
        ),
        lambda p: next(iter(p["images"].values())).update(sha256="incorrect"),
        lambda p: next(iter(p["images"].values())).update(display_transform=["flip_rows"]),
        lambda p: p["cases"][0]["planes"][0].update(selected_index=12),
        lambda p: p["cases"][0]["planes"][0]["frames"][0]["record"].update(
            quality_flags=["SECRET_PERSON"]
        ),
    ],
)
def test_rejects_wrong_provenance_unmatched_frames_and_invalid_numbers(alter):
    payload, sources = payload_fixture()
    alter(payload)
    with pytest.raises(ValueError):
        module.normalize_disc_result(payload, sources)


def test_null_or_empty_geometry_remains_unknown():
    payload, sources = payload_fixture()
    case = payload["cases"][0]
    case["candidate_voxels"] = 0
    case["decision"]["geometric_candidate"] = "indeterminate"
    for frame in case["planes"][0]["frames"]:
        frame["record"] = {
            "geometry_measurable": False,
            "geometric_candidate": "indeterminate",
            "quality_flags": ["empty_mask"],
        }
        frame["mask_polygons_xy"] = []
        frame["reference_line_xy"] = None
    result = module.normalize_disc_result(payload, sources)
    assert result["levels"][0]["frames"][0]["B_mm"] is None
    summary = module.patient_summary([], {"disc_morphology": result})
    assert any(
        "L1-L2" in text and "наличие грыжи не оценено" in text for text in summary["unassessed"]
    )


def test_summary_uses_selected_frame_measurements_and_keeps_unknowns():
    payload, sources = payload_fixture()
    result = module.normalize_disc_result(payload, sources)
    level = result["levels"][0]
    selected = level["frames"][1]
    selected.update(
        B_mm=3.4, D_mm=None, H_mm=1.2, D_over_B=None, uncertainty_mm=0.8, geometry_measurable=False
    )
    level["frames"][0]["B_mm"] = 99
    summary = module.patient_summary([], {"disc_morphology": result})
    text = next(s for s in summary["findings"] if s.startswith("L1-L2"))
    assert "На представительном срезе B = 3,4 мм, H = 1,2 мм" in text
    assert "99" not in text and "D =" not in text and "D/B =" not in text
    assert "неопределённости длины 0,8 мм" in text
    assert "Пригодность геометрии не подтверждена" in text
    assert "Морфологический статус требует проверки" in text


def test_optional_module_failure_does_not_fabricate_findings(tmp_path):
    assert (
        module.run_analysis_modules(tmp_path, tmp_path, {})["disc_morphology"]["status"]
        == "disabled"
    )
    cfg = {"enabled": True, "python": "/nonexistent/python", "model_home": "/nonexistent/models"}
    result = module.run_disc_morphology(tmp_path, tmp_path, cfg)
    assert result["status"] == "failed" and result["levels"] == []
    summary = module.patient_summary(
        [
            {"level": "L4-L5", "status": "ok", "prediction": 1},
            {"level": "L5-S1", "status": "not_assessed", "prediction": None},
        ],
        {"disc_morphology": result},
    )
    assert any("умеренная степень" in s for s in summary["findings"])
    assert any("L5-S1" in s and "не оценена" in s for s in summary["unassessed"])
    assert any("Морфология дисков не оценена" in s for s in summary["unassessed"])


@pytest.mark.parametrize("missing_modality", [False, True])
def test_success_uses_pinned_isolated_cli_and_removes_raw_private_outputs(
    tmp_path, monkeypatch, missing_modality
):
    payload, sources = payload_fixture()
    for source in sources.values():
        source["missing_modality"] = missing_modality
    models = tmp_path / "models"
    models.mkdir()
    (models / "config.json").write_text("{}")
    monkeypatch.setattr(module, "_source_slices", lambda _: (sources, "PRIVATE_SERIES_UID"))
    command_seen = []

    class Completed:
        returncode = 0

        def poll(self):
            return 0

    def fake_process(command, **kwargs):
        command_seen.extend(command)
        output = Path(command[command.index("--output") + 1])
        output.mkdir()
        (output / "result.json").write_text(json.dumps(payload))
        (output / "private-mask.nii.gz").write_bytes(b"private-image")
        assert kwargs["start_new_session"] is True
        assert kwargs["stdin"] == subprocess.DEVNULL
        return Completed()

    monkeypatch.setattr(module.subprocess, "Popen", fake_process)
    result = module.run_disc_morphology(
        tmp_path,
        tmp_path,
        {
            "enabled": True,
            "python": sys.executable,
            "model_home": str(models),
            "timeout_seconds": 60,
            "threads": 1,
        },
    )
    assert result["status"] == "completed"
    assert "SECRET" not in json.dumps(result)
    assert command_seen[2] == module._LAUNCHER
    assert command_seen[command_seen.index("--sagittal-series") + 1] == "PRIVATE_SERIES_UID"
    assert "--no-open" in command_seen
    assert ("--allow-missing-modality" in command_seen) is missing_modality
    assert "--allow-unverified-registration" not in command_seen
    assert not list(tmp_path.glob(".disc-morphology-*"))
    assert (tmp_path / "disc-morphology.log").is_file()


def test_timeout_terminates_model_descendants_and_returns_unknown(tmp_path, monkeypatch):
    payload, sources = payload_fixture()
    models = tmp_path / "models"
    models.mkdir()
    (models / "config.json").write_text("{}")
    monkeypatch.setattr(module, "_source_slices", lambda _: (sources, "PRIVATE_SERIES_UID"))
    clock = iter([0, 4000])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
    killed = []

    class Running:
        pid = 12345
        done = False

        def poll(self):
            return 0 if self.done else None

        def wait(self):
            self.done = True

    process = Running()
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(module, "terminate_process_tree", killed.append)
    result = module.run_disc_morphology(
        tmp_path,
        tmp_path,
        {
            "enabled": True,
            "python": sys.executable,
            "model_home": str(models),
            "timeout_seconds": 60,
        },
    )
    assert result["error_code"] == "timeout"
    assert result["levels"] == [] and killed == [12345]
    assert not list(tmp_path.glob(".disc-morphology-*"))


def test_linux_host_memory_reads_combined_available_ram_and_swap(tmp_path, monkeypatch):
    monkeypatch.setattr(module.sys, "platform", "linux")
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 8000000 kB\nMemAvailable: 230428 kB\nSwapFree: 50200 kB\n")
    assert module._host_memory_bytes(meminfo) == (230428 + 50200) * 1024
    assert module._host_memory_bytes(meminfo) < module.MIN_HOST_AVAILABLE_BYTES
    meminfo.write_text("MemAvailable: 100 kB\n")
    assert module._host_memory_bytes(meminfo) is None
    monkeypatch.setattr(module.sys, "platform", "darwin")
    assert module._host_memory_bytes(meminfo) is None


@pytest.mark.parametrize("before_start", [True, False])
def test_low_host_memory_does_not_launch_or_terminates_only_own_worker(
    tmp_path, monkeypatch, before_start
):
    _, sources = payload_fixture()
    models = tmp_path / "models"
    models.mkdir()
    (models / "config.json").write_text("{}")
    monkeypatch.setattr(module, "_source_slices", lambda _: (sources, "PRIVATE_SERIES_UID"))
    memory = iter(
        [module.MIN_HOST_AVAILABLE_BYTES - 1]
        if before_start
        else [module.MIN_HOST_AVAILABLE_BYTES, module.MIN_HOST_AVAILABLE_BYTES - 1]
    )
    monkeypatch.setattr(module, "_host_memory_bytes", lambda: next(memory))
    killed, launched = [], []

    class Running:
        pid = 12345
        done = False

        def poll(self):
            return 0 if self.done else None

        def wait(self):
            self.done = True

    def launch(*args, **kwargs):
        launched.append(True)
        return Running()

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    monkeypatch.setattr(module, "terminate_process_tree", killed.append)
    result = module.run_disc_morphology(
        tmp_path,
        tmp_path,
        {
            "enabled": True,
            "python": sys.executable,
            "model_home": str(models),
            "timeout_seconds": 60,
        },
    )
    assert result["status"] == "failed" and result["error_code"] == "resource_unavailable"
    assert result["levels"] == [] and "недостаточно свободной памяти" in result["message"]
    assert launched == ([] if before_start else [True])
    assert killed == ([] if before_start else [12345])
    assert not list(tmp_path.glob(".disc-morphology-*"))


@pytest.mark.skipif(os.name != "posix", reason="POSIX model subprocesses")
def test_termination_includes_worker_in_another_session(tmp_path, monkeypatch):
    heartbeat = tmp_path / "heartbeat"
    child = (
        "import pathlib,time; p=pathlib.Path("
        + repr(str(heartbeat))
        + ");\nwhile True:\n p.write_text(str(time.monotonic())); time.sleep(.02)"
    )
    parent = (
        "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c',"
        + repr(child)
        + "],start_new_session=True); print(p.pid,flush=True); time.sleep(60)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", parent], stdout=subprocess.PIPE, text=True, start_new_session=True
    )
    worker = int(process.stdout.readline())
    try:
        # The sandbox may prohibit host-wide `ps`; process ancestry is known for
        # this real parent/child pair and the kill operates on the actual PIDs.
        monkeypatch.setattr(module, "_process_parents", lambda: {worker: process.pid})
        assert os.getpgid(worker) != os.getpgid(process.pid)
        deadline = time.monotonic() + 3
        while not heartbeat.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert heartbeat.exists()
        module.terminate_process_tree(process.pid)
        process.wait(timeout=3)
        time.sleep(0.05)
        value = heartbeat.read_text()
        time.sleep(0.1)
        assert heartbeat.read_text() == value
    finally:
        for pid in (worker, process.pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=3)
