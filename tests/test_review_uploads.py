"""Synthetic HTTP/DICOM tests: no real patient images or model credentials."""

import http.client
import io
import json
import socket
import threading
import time
import zipfile
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import EnhancedMRImageStorage, ExplicitVRLittleEndian, MRImageStorage, generate_uid

from ai_spine_med import review_uploads as module
from ai_spine_med.analysis_modules import _source_slices
from ai_spine_med.review_server import ReviewServer


def synthetic_zip(*, series_count=1, count=3, axial=False):
    output = io.BytesIO()
    series = [generate_uid() for _ in range(series_count)]
    study = generate_uid()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for i in range(count):
            meta = FileMetaDataset()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            meta.MediaStorageSOPClassUID = MRImageStorage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            ds = FileDataset("", {}, file_meta=meta, preamble=b"\0" * 128)
            ds.SOPClassUID = MRImageStorage
            ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
            ds.StudyInstanceUID = study
            ds.SeriesInstanceUID = series[i % len(series)]
            ds.PatientName = "SECRET_PERSON"
            ds.PatientID = "SECRET_ID"
            ds.StudyDescription = "SECRET_DIAGNOSIS"
            ds.Modality = "MR"
            ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0] if axial else [0, 1, 0, 0, 0, -1]
            ds.ImagePositionPatient = [i * 3, 0, 0]
            ds.PixelSpacing = [0.8, 0.8]
            ds.InstanceNumber = i + 1
            ds.Rows = ds.Columns = 16
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated = ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0
            ds.PixelData = (np.arange(256).reshape(16, 16) + i).astype("uint16").tobytes()
            content = io.BytesIO()
            ds.save_as(content, enforce_file_format=True)
            archive.writestr(f"SECRET_FOLDER/SECRET_PERSON_{i}.dcm", content.getvalue())
    return output.getvalue()


def fake_inference(directory, out, **kwargs):
    assert len(list(Path(directory).glob("*.dcm"))) == 3
    assert len(kwargs["checkpoint"]) == 3
    return {
        "status": "ok",
        "targets": [
            {
                "level": level,
                "status": "ok",
                "reason": "",
                "prediction": 1,
                "p_normal_mild": 0.1,
                "p_moderate": 0.8,
                "p_severe": 0.1,
            }
            for level in ("L4-L5", "L5-S1")
        ],
        "scope": "synthetic inference fixture",
    }


@pytest.fixture
def store(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    heads = []
    for i in range(3):
        path = tmp_path / f"head-{i}.pt"
        path.write_bytes(b"synthetic trusted fixture")
        heads.append(path)
    uploads = module.UploadStore(
        tmp_path / "uploads", heads=heads, cache=cache, runner=fake_inference
    )
    yield uploads
    uploads.executor.shutdown(wait=True)


def await_record(store, identifier):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        record = store.get(identifier)
        if record["status"] not in {"receiving", "queued", "processing"}:
            return record
        time.sleep(0.01)
    pytest.fail("Synthetic upload processing did not finish")


def test_complete_private_upload_preview_and_persistence(store):
    raw = synthetic_zip()
    accepted = store.receive(io.BytesIO(raw), len(raw))
    record = await_record(store, accepted["upload_id"])
    assert record["status"] == "completed"
    assert record["n_slices"] == 3
    assert len(record["targets"]) == 2
    assert record["targets"][0]["probabilities"] == [0.1, 0.8, 0.1]
    assert "SECRET" not in json.dumps(record)
    assert "SECRET" not in json.dumps(store.list())
    assert "patient" not in store.list()["uploads"][0]
    assert record["sequence_subtype_verified"] is False
    assert record["patient"]["series"][0]["series_description"] == "Sagittal MRI"
    assert record["patient"]["series"][0]["sequence_subtype_verified"] is False
    assert "9" in record["input_scope"]
    for item in record["patient"]["series"][0]["slices"]:
        assert (
            store.asset(record["upload_id"], f"images/{item['slice_key']}.jpg").read_bytes()[:2]
            == b"\xff\xd8"
        )
        assert store.asset(record["upload_id"], f"dicom/{item['slice_key']}.dcm").is_file()
    assert store.asset(record["upload_id"], "source.zip") is None
    assert store.asset(record["upload_id"], "record.json") is None
    assert store.asset(record["upload_id"], "inference/predictions.json") is None
    assert store.asset(record["upload_id"], "../worker-config.json") is None
    assert not store.reservations
    reopened = module.UploadStore(store.root)
    try:
        assert reopened.get(record["upload_id"])["status"] == "completed"
    finally:
        reopened.close()


@pytest.mark.parametrize("storage_class", [MRImageStorage, EnhancedMRImageStorage])
def test_rsna_style_missing_modality_and_sop_class_uses_mr_file_meta(store, storage_class):
    raw = synthetic_zip()
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(raw)) as original, zipfile.ZipFile(output, "w") as archive:
        for entry in original.infolist():
            dataset = pydicom.dcmread(io.BytesIO(original.read(entry)))
            del dataset.Modality
            del dataset.SOPClassUID
            dataset.file_meta.MediaStorageSOPClassUID = storage_class
            stream = io.BytesIO()
            dataset.save_as(stream, enforce_file_format=True)
            archive.writestr(entry.filename, stream.getvalue())
    record = store.receive(io.BytesIO(output.getvalue()), len(output.getvalue()))
    assert await_record(store, record["upload_id"])["status"] == "completed"
    sources, _ = _source_slices(store.root / record["upload_id"] / "dicom")
    assert len(sources) == 3 and all(s["missing_modality"] for s in sources.values())


def test_active_job_reservation_prevents_disk_overcommit(store):
    identifier = "upload-" + "f" * 32
    store.reservations[identifier] = module.MAX_STORED_BYTES
    try:
        with pytest.raises(module.UploadRejected, match="storage_full"):
            store.receive(io.BytesIO(b"x"), 1)
    finally:
        store.reservations.clear()


@pytest.mark.parametrize(
    "kwargs,code",
    [
        ({"series_count": 2}, "multiple_series"),
        ({"count": 1}, "single_image"),
        ({"axial": True}, "unsupported_plane_axial"),
    ],
)
def test_unsupported_series_fails_explicitly(store, kwargs, code):
    raw = synthetic_zip(**kwargs)
    record = store.receive(io.BytesIO(raw), len(raw))
    finished = await_record(store, record["upload_id"])
    assert finished["status"] == "failed"
    assert finished["error_code"] == code
    assert finished["targets"] == []
    assert "SECRET" not in json.dumps(finished)


def test_auxiliary_annotation_files_are_ignored_not_read_as_dicom(store):
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(synthetic_zip())) as original:
        with zipfile.ZipFile(output, "w") as archive:
            for entry in original.infolist():
                archive.writestr(entry.filename, original.read(entry))
            archive.writestr("annotations/labels.xml", "<annotation>SECRET</annotation>")
            archive.writestr("case.json", '{"name":"SECRET"}')
            archive.writestr("README.txt", "SECRET")
    raw = output.getvalue()
    accepted = store.receive(io.BytesIO(raw), len(raw))
    record = await_record(store, accepted["upload_id"])
    assert record["status"] == "completed"
    assert record["ignored_auxiliary_files"] == 3
    assert record["input_plane"] == "sagittal"
    assert record["n_dicom_files"] == 3
    assert "SECRET" not in json.dumps(record)


def test_axial_export_with_xml_gets_actionable_plane_error_without_inference(store):
    def unexpected_inference(*args, **kwargs):
        pytest.fail("Sagittal model must not be applied to axial input")

    store.runner = unexpected_inference
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(synthetic_zip(axial=True))) as original:
        with zipfile.ZipFile(output, "w") as archive:
            for entry in original.infolist():
                archive.writestr(entry.filename, original.read(entry))
            archive.writestr("annotations/labels.xml", "<xml>SECRET</xml>")
    raw = output.getvalue()
    record = store.receive(io.BytesIO(raw), len(raw))
    finished = await_record(store, record["upload_id"])
    assert finished["error_code"] == "unsupported_plane_axial"
    assert "аксиальные" in finished["message"]
    assert "сагиттальная T2-серия" in finished["message"]
    assert finished["input_plane"] == "axial"
    assert finished["ignored_auxiliary_files"] == 1
    assert finished["targets"] == []


@pytest.mark.parametrize(
    "orientations,expected",
    [
        ([[1, 0, 0, 0, 0, -1]], "coronal"),
        ([[0, 1, 0, 0, 0, -1], [1, 0, 0, 0, 1, 0]], "oblique_or_mixed"),
        ([[0, 0, 0, 0, 0, 0]], "unknown"),
        ([[float("nan"), 0, 0, 0, 1, 0]], "unknown"),
        ([[1, 0, 0]], "unknown"),
    ],
)
def test_input_plane_does_not_guess_from_invalid_or_mixed_geometry(orientations, expected):
    from types import SimpleNamespace

    headers = [SimpleNamespace(ImageOrientationPatient=value) for value in orientations]
    assert module._input_plane(headers) == expected


@pytest.mark.parametrize(
    "filename", ["../escape.dcm", "/escape.dcm", "folder\\escape.dcm", "C:/escape.dcm"]
)
def test_zip_path_traversal_rejected(store, filename):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(filename, b"not DICOM")
    record = store.receive(io.BytesIO(output.getvalue()), len(output.getvalue()))
    finished = await_record(store, record["upload_id"])
    assert finished["error_code"] == "unsafe_archive"
    assert not (store.root.parent / "escape.dcm").exists()


def test_zip_bomb_and_symlink_rejected(store):
    for symlink in (False, True):
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            if symlink:
                entry = zipfile.ZipInfo("link.dcm")
                entry.external_attr = 0o120777 << 16
                archive.writestr(entry, "/etc/passwd")
            else:
                archive.writestr("zeros.dcm", bytes(1024 * 1024))
        record = store.receive(io.BytesIO(output.getvalue()), len(output.getvalue()))
        assert await_record(store, record["upload_id"])["status"] == "failed"


def test_upload_limits_no_model_and_incomplete_transfer(store, tmp_path, monkeypatch):
    with pytest.raises(module.UploadRejected, match="upload_limit"):
        store.receive(io.BytesIO(b"x"), module.MAX_UPLOAD_BYTES + 1)
    with pytest.raises(module.UploadRejected, match="incomplete_upload"):
        store.receive(io.BytesIO(b"x"), 10)
    assert not list(store.root.glob("upload-*"))
    disabled = module.UploadStore(tmp_path / "disabled")
    try:
        with pytest.raises(module.UploadRejected, match="model_unavailable"):
            disabled.receive(io.BytesIO(b"x"), 1)
    finally:
        disabled.close()
    monkeypatch.setattr(module, "MAX_STORED_BYTES", 1)
    with pytest.raises(module.UploadRejected, match="storage_full"):
        store.receive(io.BytesIO(b"x"), 1)


@pytest.fixture
def server(store, tmp_path):
    package = tmp_path / "package"
    (package / "site/data").mkdir(parents=True)
    (package / "site/index.html").write_text("<title>Synthetic review</title>")
    (package / "site/data/index.json").write_text('{"synthetic":true}')
    (package / "private-source-index.json").write_text('{"schema_version":1,"routes":{}}')
    instance = ReviewServer(
        package,
        "synthetic_capability_123456789",
        port=0,
        uploads=store,
        public_origin="https://review.example",
        max_http_connections=2,
    )
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    yield instance
    instance.shutdown()
    instance.server_close()
    thread.join(timeout=2)


def request(server, path, method="GET", body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    result = response.status, dict(response.getheaders()), response.read()
    connection.close()
    return result


def test_http_upload_and_download_contract(server):
    route = server.prefix + "api/uploads"
    status, _, body = request(server, route)
    assert status == 200
    assert json.loads(body)["model"]["configured"] is True
    headers = {
        "Origin": "https://review.example",
        "Content-Type": "application/zip",
        "X-Upload-Intent": "dicom-series",
    }
    status, _, body = request(server, route, "POST", synthetic_zip(), headers)
    assert status == 202
    identifier = json.loads(body)["upload_id"]
    await_record(server.uploads, identifier)
    status, headers, body = request(server, route + "/" + identifier)
    assert status == 200
    assert headers["Referrer-Policy"] == "no-referrer"
    assert "no-store" in headers["Cache-Control"]
    assert "SECRET" not in body.decode()
    record = json.loads(body)
    for resource in (
        record["report_url"],
        record["patient"]["series"][0]["slices"][0]["image_url"],
        record["patient"]["series"][0]["slices"][0]["dicom_url"],
        record["patient"]["series"][0]["zip_url"],
    ):
        assert request(server, server.prefix + resource)[0] == 200
    assert request(server, route + "/" + identifier + "/record.json")[0] == 404
    assert request(server, route + "/" + identifier + "/source.zip")[0] == 404
    assert request(server, "/api/uploads")[0] == 404


@pytest.mark.parametrize(
    "headers,status",
    [
        ({}, 403),
        ({"Origin": "https://elsewhere.example", "X-Upload-Intent": "dicom-series"}, 403),
        (
            {
                "Origin": "https://review.example",
                "X-Upload-Intent": "dicom-series",
                "Content-Type": "image/png",
            },
            415,
        ),
        (
            {
                "Origin": "https://review.example",
                "X-Upload-Intent": "dicom-series",
                "Content-Type": "application/zip",
                "Sec-Fetch-Site": "cross-site",
            },
            403,
        ),
    ],
)
def test_http_upload_requires_same_origin_and_raw_zip(server, headers, status):
    assert (
        request(server, server.prefix + "api/uploads", "POST", b"synthetic", headers)[0] == status
    )
    assert not server.uploads.list()["uploads"]


def test_static_auth_and_path_boundaries_remain(server):
    assert request(server, server.prefix)[0] == 200
    assert request(server, server.prefix + "data/index.json")[0] == 200
    for path in (
        "/",
        "/r/wrong/",
        server.prefix + "../private-source-index.json",
        server.prefix + "private-source-index.json",
        server.prefix + "api/uploads/upload-" + "0" * 32,
    ):
        assert request(server, path)[0] == 404


def test_restart_marks_pending_interrupted(tmp_path):
    root = tmp_path / "uploads"
    job = root / ("upload-" + "1" * 32)
    job.mkdir(parents=True)
    (job / "record.json").write_text(
        json.dumps({"upload_id": job.name, "created_at_utc": "2026-10-26", "status": "processing"})
    )
    store = module.UploadStore(root)
    try:
        assert store.get(job.name)["error_code"] == "interrupted"
    finally:
        store.close()


def test_http_connections_are_bounded_before_headers(server):
    clients = []
    try:
        for _ in range(2):
            client = socket.create_connection(("127.0.0.1", server.server_port), timeout=2)
            client.sendall(f"GET {server.prefix} HTTP/1.1\r\n".encode())
            clients.append(client)
        deadline = time.monotonic() + 2
        while server.http_slots._value != 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.http_slots._value == 0
        excess = socket.create_connection(("127.0.0.1", server.server_port), timeout=2)
        try:
            try:
                excess.sendall(b"GET / HTTP/1.1\r\n\r\n")
                assert excess.recv(1) == b""
            except (ConnectionResetError, BrokenPipeError):
                pass
        finally:
            excess.close()
    finally:
        for client in clients:
            client.close()
    deadline = time.monotonic() + 2
    while server.http_slots._value != 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.http_slots._value == 2
    assert request(server, server.prefix)[0] == 200


def test_http_slot_released_when_thread_creation_fails(server, monkeypatch):
    def fail(*args):
        raise RuntimeError("synthetic thread creation failure")

    with monkeypatch.context() as patch:
        patch.setattr(ThreadingHTTPServer, "process_request", fail)
        with pytest.raises(RuntimeError, match="synthetic"):
            server.process_request(object(), ("127.0.0.1", 0))
    assert server.http_slots._value == 2
    assert request(server, server.prefix)[0] == 200


def test_http_slot_released_when_handler_fails(server, monkeypatch):
    def fail(*args):
        raise RuntimeError("synthetic handler failure")

    with monkeypatch.context() as patch:
        patch.setattr(server, "finish_request", fail)
        with pytest.raises((http.client.RemoteDisconnected, ConnectionResetError)):
            request(server, server.prefix)
    deadline = time.monotonic() + 2
    while server.http_slots._value != 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.http_slots._value == 2
    assert request(server, server.prefix)[0] == 200


def test_optional_module_failure_preserves_stenosis_and_summary(store, monkeypatch):
    result = {
        "disc_morphology": {"status": "failed", "levels": [], "message": "Серия не обработана."}
    }
    monkeypatch.setattr(module, "run_analysis_modules", lambda *args: result)
    raw = synthetic_zip()
    accepted = store.receive(io.BytesIO(raw), len(raw))
    record = await_record(store, accepted["upload_id"])
    assert record["status"] == "completed"
    assert record["modules"] == result
    assert len(record["targets"]) == 2
    assert len(record["patient_summary"]["findings"]) == 2
    assert "Морфология дисков не оценена." in record["patient_summary"]["unassessed"]
    report = json.loads((store.root / record["upload_id"] / "result.json").read_text())
    assert report["modules"] == record["modules"]
    assert "modules" not in store.list()["uploads"][0]


def test_modules_rerun_reuses_dicom_and_preserves_central_results(store, monkeypatch):
    raw = synthetic_zip()
    accepted = store.receive(io.BytesIO(raw), len(raw))
    original = await_record(store, accepted["upload_id"])
    store.configuration["modules"] = {"disc_morphology": {"enabled": True}}
    result = {"disc_morphology": {"status": "completed", "levels": []}}
    calls = []

    def run(directory, job, configuration):
        calls.append(directory)
        assert len(list(directory.glob("*.dcm"))) == 3
        return result

    def unexpected_inference(*args, **kwargs):
        pytest.fail("Existing stenosis estimates should be reused")

    monkeypatch.setattr(module, "run_analysis_modules", run)
    store.runner = unexpected_inference
    assert store.run_modules(original["upload_id"])["status"] == "queued"
    record = await_record(store, original["upload_id"])
    assert len(calls) == 1
    assert record["targets"] == original["targets"]
    assert record["modules"] == result
    assert record["status"] == "completed"
    assert record["modules_available"] is True


def test_modules_http_requires_origin_before_enqueuing(server):
    route = server.prefix + "api/uploads/upload-" + "1" * 32 + "/modules"
    assert request(server, route, "POST", b"")[0] == 403
    headers = {"Origin": "https://review.example", "X-Upload-Intent": "dicom-series"}
    assert request(server, route, "POST", b"", headers)[0] == 503
    headers["Origin"] = "https://elsewhere.example"
    assert request(server, route, "POST", b"", headers)[0] == 403


def test_old_records_get_summary_without_rewriting_sources(store):
    identifier = "upload-" + "a" * 32
    job = store.root / identifier
    job.mkdir()
    saved = {"upload_id": identifier, "status": "completed", "targets": []}
    path = job / "record.json"
    path.write_text(json.dumps(saved))
    before = path.read_bytes()
    record = store.get(identifier)
    assert record["modules"] == {}
    assert record["patient_summary"]["findings"] == []
    assert len(record["patient_summary"]["unassessed"]) == 3
    assert path.read_bytes() == before


def test_uncaught_module_error_keeps_primary_model_result(store, monkeypatch):
    def fail(*args):
        raise RuntimeError("PRIVATE_MODEL_ERROR")

    monkeypatch.setattr(module, "run_analysis_modules", fail)
    raw = synthetic_zip()
    accepted = store.receive(io.BytesIO(raw), len(raw))
    record = await_record(store, accepted["upload_id"])
    assert record["status"] == record["central_status"] == "completed"
    assert len(record["targets"]) == 2
    assert record["modules"]["disc_morphology"]["status"] == "failed"
    assert "PRIVATE_MODEL_ERROR" not in json.dumps(record)
    report = json.loads((store.root / record["upload_id"] / "result.json").read_text())
    assert len(report["targets"]) == 2
    assert report["modules"] == record["modules"]


def test_primary_result_is_visible_while_optional_module_is_running(store, monkeypatch):
    store.configuration["modules"] = {"disc_morphology": {"enabled": True}}
    started, release = threading.Event(), threading.Event()

    def pause(*args):
        started.set()
        assert release.wait(3)
        return {"disc_morphology": {"status": "completed", "levels": []}}

    monkeypatch.setattr(module, "run_analysis_modules", pause)
    raw = synthetic_zip()
    accepted = store.receive(io.BytesIO(raw), len(raw))
    try:
        assert started.wait(3)
        record = store.get(accepted["upload_id"])
        assert record["status"] == "processing"
        assert record["central_status"] == "completed"
        assert record["modules"]["disc_morphology"]["status"] == "processing"
        assert len(record["targets"]) == 2 and record["report_url"]
        assert len(record["patient_summary"]["findings"]) == 2
    finally:
        release.set()
    assert await_record(store, accepted["upload_id"])["status"] == "completed"


def test_restart_during_module_preserves_saved_primary_report(store):
    raw = synthetic_zip()
    accepted = store.receive(io.BytesIO(raw), len(raw))
    original = await_record(store, accepted["upload_id"])
    job = store.root / original["upload_id"]
    module._update(job, status="processing", central_status="completed")
    work = job / ".disc-morphology-incomplete"
    work.mkdir()
    (work / "raw-model.nii.gz").write_bytes(b"private-incomplete-output")
    reopened = module.UploadStore(store.root)
    try:
        record = reopened.get(original["upload_id"])
        assert record["status"] == "completed"
        assert record["targets"] == original["targets"]
        assert record["modules"]["disc_morphology"]["error_code"] == "interrupted"
        assert not work.exists()
        report = json.loads((job / "result.json").read_text())
        assert report["modules"] == record["modules"]
        assert len(report["targets"]) == 2
    finally:
        reopened.close()


def test_module_queue_submission_failure_releases_slot_and_preserves_record(store, monkeypatch):
    raw = synthetic_zip()
    accepted = store.receive(io.BytesIO(raw), len(raw))
    original = await_record(store, accepted["upload_id"])
    store.configuration["modules"] = {"disc_morphology": {"enabled": True}}

    def reject(*args, **kwargs):
        raise RuntimeError("executor is closing")

    monkeypatch.setattr(store.executor, "submit", reject)
    with pytest.raises(module.UploadRejected, match="queue_unavailable"):
        store.run_modules(original["upload_id"])
    assert store.slots._value == module.MAX_PENDING
    assert not store.reservations
    record = store.get(original["upload_id"])
    assert record["status"] == "completed" and record["targets"] == original["targets"]
    assert record["modules"] == original["modules"]


def test_module_run_requires_temporary_workspace_free_disk(store, monkeypatch):
    raw = synthetic_zip()
    accepted = store.receive(io.BytesIO(raw), len(raw))
    original = await_record(store, accepted["upload_id"])
    store.configuration["modules"] = {"disc_morphology": {"enabled": True}}
    monkeypatch.setattr(
        module.shutil, "disk_usage", lambda _: SimpleNamespace(free=module.MAX_WORK_BYTES - 1)
    )
    with pytest.raises(module.UploadRejected, match="storage_full"):
        store.run_modules(original["upload_id"])
    assert store.slots._value == module.MAX_PENDING
    assert not store.reservations
    assert store.get(original["upload_id"])["status"] == "completed"


def test_outer_timeout_preserves_primary_result_and_marks_only_module_failed(store, monkeypatch):
    raw = synthetic_zip()
    accepted = store.receive(io.BytesIO(raw), len(raw))
    original = await_record(store, accepted["upload_id"])
    job = store.root / original["upload_id"]
    module._update(job, status="queued", central_status="completed")
    store.runner = None
    assert store.slots.acquire(blocking=False)
    stopped = []

    class Running:
        pid = 12345
        done = False

        def poll(self):
            return -9 if self.done else None

        def wait(self, timeout=None):
            if timeout is not None:
                raise module.subprocess.TimeoutExpired("test-worker", timeout)
            self.done = True
            return -9

    process = Running()
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(module, "terminate_process_tree", stopped.append)
    monkeypatch.setattr(module.os, "killpg", lambda *args: None)
    store._work(job, modules_only=True)
    record = store.get(original["upload_id"])
    assert record["status"] == "completed"
    assert record["targets"] == original["targets"]
    assert record["modules"]["disc_morphology"]["error_code"] == "timeout"
    assert stopped == [12345] and store.slots._value == module.MAX_PENDING
    report = json.loads((job / "result.json").read_text())
    assert report["modules"] == record["modules"]


@pytest.mark.parametrize(
    "config",
    [
        [],
        {"unknown_module": {}},
        {"disc_morphology": "invalid"},
        {"disc_morphology": {"enabled": "yes"}},
        {"disc_morphology": {"enabled": True, "python": "relative", "model_home": "/models"}},
        {
            "disc_morphology": {
                "enabled": True,
                "python": "/python",
                "model_home": "/models",
                "timeout_seconds": -1,
            }
        },
        {"disc_morphology": {"unexpected": True}},
    ],
)
def test_invalid_module_configuration_fails_server_startup(tmp_path, config):
    with pytest.raises(ValueError):
        module.UploadStore(tmp_path / "uploads", modules=config)


def test_delete_removes_images_originals_reports_and_is_idempotent(store):
    raw = synthetic_zip()
    created = store.receive(io.BytesIO(raw), len(raw))
    record = await_record(store, created["upload_id"])
    identifier = record["upload_id"]
    job = store.root / identifier
    (job / "disc-morphology-inference.log").write_text("private diagnostic fixture")
    assert store.delete(identifier) == {"deleted": identifier}
    assert not job.exists()
    assert not (store.root / (".deleting-" + identifier)).exists()
    assert store.get(identifier) is None
    assert store.list()["uploads"] == []
    assert store.asset(identifier, "series.zip") is None
    assert store.delete(identifier) == {"deleted": identifier}
    assert store.config_path.is_file()


@pytest.mark.parametrize("status", ["receiving", "queued", "processing"])
def test_delete_rejects_inflight_jobs(store, status):
    identifier = "upload-" + "d" * 32
    job = store.root / identifier
    job.mkdir()
    (job / "record.json").write_text(json.dumps({"status": status}))
    with pytest.raises(module.UploadRejected) as exc:
        store.delete(identifier)
    assert exc.value.status == 409
    assert job.is_dir()


def test_delete_waits_for_worker_cleanup_reservation(store):
    identifier = "upload-" + "d" * 32
    store.reservations[identifier] = 10
    try:
        with pytest.raises(module.UploadRejected) as exc:
            store.delete(identifier)
        assert exc.value.status == 409
    finally:
        store.reservations.clear()


@pytest.mark.parametrize("identifier", ["../models", "worker-config.json", "upload-abc"])
def test_delete_rejects_non_upload_paths(store, identifier):
    with pytest.raises(module.UploadRejected) as exc:
        store.delete(identifier)
    assert exc.value.status == 404
    assert store.config_path.is_file()


def test_delete_refuses_symlink_to_other_storage(store, tmp_path):
    protected = tmp_path / "protected"
    protected.mkdir()
    (protected / "keep").write_text("keep")
    identifier = "upload-" + "e" * 32
    (store.root / identifier).symlink_to(protected, target_is_directory=True)
    with pytest.raises(module.UploadRejected):
        store.delete(identifier)
    assert (protected / "keep").read_text() == "keep"


def test_delete_failure_hides_files_and_retry_finishes_cleanup(store, monkeypatch):
    identifier = "upload-" + "e" * 32
    job = store.root / identifier
    job.mkdir()
    (job / "record.json").write_text(
        json.dumps({"status": "failed", "created_at_utc": "2026-11-24"})
    )
    (job / "source.zip").write_bytes(b"private fixture")
    real_remove = module.shutil.rmtree

    def fail_delete(*args, **kwargs):
        raise OSError("fixture")

    with monkeypatch.context() as patch:
        patch.setattr(module.shutil, "rmtree", fail_delete)
        with pytest.raises(module.UploadRejected) as exc:
            store.delete(identifier)
        assert exc.value.status == 503
    assert not job.exists()
    assert store.get(identifier) is None
    assert store.list()["uploads"] == []
    assert (store.root / (".deleting-" + identifier)).is_dir()
    store.delete(identifier)
    assert not (store.root / (".deleting-" + identifier)).exists()
    assert module.shutil.rmtree is real_remove


def test_restart_finishes_interrupted_deletion(tmp_path):
    root = tmp_path / "uploads"
    hidden = root / (".deleting-upload-" + "b" * 32)
    hidden.mkdir(parents=True)
    (hidden / "series.zip").write_bytes(b"private fixture")
    reopened = module.UploadStore(root)
    try:
        assert not hidden.exists()
    finally:
        reopened.close()


def test_http_delete_requires_origin_removes_assets_and_can_retry(server):
    raw = synthetic_zip()
    created = server.uploads.receive(io.BytesIO(raw), len(raw))
    record = await_record(server.uploads, created["upload_id"])
    route = server.prefix + "api/uploads/" + record["upload_id"]
    assert request(server, route, "DELETE", b"")[0] == 403
    headers = {"Origin": "https://review.example", "X-Upload-Intent": "dicom-series"}
    assert request(server, route, "DELETE", b"x", headers)[0] == 400
    assert server.uploads.get(record["upload_id"]) is not None
    status, _, content = request(server, route, "DELETE", b"", headers)
    assert status == 200
    assert json.loads(content) == {"deleted": record["upload_id"]}
    assert request(server, route)[0] == 404
    assert request(server, route + "/series.zip")[0] == 404
    assert request(server, route + "/result.json")[0] == 404
    assert request(server, route, "DELETE", b"", headers)[0] == 200


@pytest.mark.parametrize(
    "morphology_status,expected",
    [("completed", "partial"), ("disabled", "failed"), ("failed", "failed")],
)
def test_module_success_produces_partial_when_central_model_failed(
    store, morphology_status, expected
):
    identifier = "upload-" + "b" * 32
    job = store.root / identifier
    job.mkdir()
    targets = [
        {"level": level, "status": "not_assessed", "prediction": None}
        for level in ("L4-L5", "L5-S1")
    ]
    module._write_json(
        job / "record.json",
        {
            "upload_id": identifier,
            "status": "processing",
            "central_status": "failed",
            "targets": targets,
        },
    )
    module._write_json(job / "result.json", {"status": "failed", "targets": targets})
    modules = {"disc_morphology": {"status": morphology_status, "levels": []}}
    module._finish_modules(job, modules)
    record = store.get(identifier)
    assert record["status"] == expected and record["central_status"] == "failed"
    assert record["targets"] == targets and record["modules"] == modules
    assert not record["patient_summary"]["findings"]
