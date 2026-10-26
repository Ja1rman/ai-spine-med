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

import numpy as np
import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import EnhancedMRImageStorage, ExplicitVRLittleEndian, MRImageStorage, generate_uid

from ai_spine_med import review_uploads as module
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


def test_rsna_style_missing_modality_and_sop_class_uses_mr_file_meta(store):
    raw = synthetic_zip()
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(raw)) as original, zipfile.ZipFile(output, "w") as archive:
        for entry in original.infolist():
            dataset = pydicom.dcmread(io.BytesIO(original.read(entry)))
            del dataset.Modality
            del dataset.SOPClassUID
            dataset.file_meta.MediaStorageSOPClassUID = EnhancedMRImageStorage
            stream = io.BytesIO()
            dataset.save_as(stream, enforce_file_format=True)
            archive.writestr(entry.filename, stream.getvalue())
    record = store.receive(io.BytesIO(output.getvalue()), len(output.getvalue()))
    assert await_record(store, record["upload_id"])["status"] == "completed"


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
        ({"axial": True}, "invalid_geometry"),
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
