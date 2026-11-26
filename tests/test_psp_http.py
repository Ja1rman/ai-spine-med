"""Private PSP boundary and MRI-link tests; no models or real patient data."""

import http.client
import json
import threading
import uuid
from types import SimpleNamespace

import pytest

from ai_spine_med.psp import PSPStore
from ai_spine_med.review_server import ReviewServer


@pytest.fixture
def psp_server(tmp_path):
    package = tmp_path / "package"
    (package / "site").mkdir(parents=True)
    (package / "site/index.html").write_text("<title>Synthetic</title>")
    (package / "private-source-index.json").write_text('{"schema_version":1,"routes":{}}')
    uploads = SimpleNamespace(
        root=tmp_path / "uploads",
        close=lambda: None,
        get=lambda identifier: (
            {
                "status": "completed",
                "targets": [],
                "patient_summary": {"text": "Synthetic model result"},
            }
            if identifier == "upload-" + "a" * 32
            else None
        ),
    )
    server = ReviewServer(
        package,
        "synthetic_psp_capability_123456789",
        0,
        psp=PSPStore(tmp_path / "clinical"),
        uploads=uploads,
        public_origin="https://review.example",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(2)


def request(server, route, payload=None, *, headers=None, raw=None, method=None):
    write = payload is not None or raw is not None
    body = raw if raw is not None else json.dumps(payload) if write else None
    defaults = (
        {
            "Origin": "https://review.example",
            "X-PSP-Intent": "clinical-record",
            "Content-Type": "application/json",
        }
        if write
        else {}
    )
    defaults.update(headers or {})
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    connection.request(
        method or ("POST" if write else "GET"), server.prefix + route, body=body, headers=defaults
    )
    response = connection.getresponse()
    result = response.status, dict(response.getheaders()), response.read()
    connection.close()
    return result


def draft(**extra):
    return {
        "patient_code": "DEMO-001",
        "adult_confirmed": True,
        "consent": True,
        "identity_confirmed": True,
        "idempotency_key": "synthetic-create-001",
        **extra,
    }


def test_psp_http_roundtrip_mri_and_export(psp_server):
    status, headers, body = request(
        psp_server, "api/psp/episodes", draft(upload_id="upload-" + "a" * 32)
    )
    assert status == 201, body
    result = json.loads(body)
    identifier = result["episode_id"]
    assert result["mri"]["available"] is True
    assert "view=uploads" in result["mri"]["viewer_url"]
    assert "no-store" in headers["Cache-Control"]
    assert request(psp_server, "api/psp/episodes/" + identifier)[0] == 200
    assert request(psp_server, "api/psp/episodes/" + identifier + "/export")[0] == 200
    assert request(psp_server, "api/psp/catalog")[0] == 200
    assert request(psp_server, "api/psp/episodes")[0] == 200


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "https://elsewhere.example"},
        {"X-PSP-Intent": ""},
        {"Sec-Fetch-Site": "cross-site"},
    ],
)
def test_psp_writes_require_same_origin(psp_server, headers):
    assert request(psp_server, "api/psp/episodes", draft(), headers=headers)[0] == 403
    assert json.loads(request(psp_server, "api/psp/episodes")[2])["episodes"] == []


@pytest.mark.parametrize(
    "raw", ['{"consent":true,"consent":false}', '{"x":NaN}', "[]", '{"x":', "\ufffd"]
)
def test_psp_rejects_ambiguous_json(psp_server, raw):
    assert request(psp_server, "api/psp/episodes", raw=raw.encode())[0] == 400


def test_psp_size_type_routes_and_upload_binding(psp_server):
    assert (
        request(psp_server, "api/psp/episodes", draft(), headers={"Content-Type": "text/plain"})[0]
        == 415
    )
    assert request(psp_server, "api/psp/episodes", raw=b"x" * (512 * 1024 + 1))[0] == 413
    assert request(psp_server, "api/psp/catalog", draft())[0] == 405
    assert request(psp_server, "api/psp/episodes", draft(upload_id="upload-" + "b" * 32))[0] == 422
    assert request(psp_server, "api/psp/episodes/episode-" + "0" * 32)[0] == 404
    for route in ("api/psp/psp.sqlite3", "clinical/psp.sqlite3", "api/psp/../episodes"):
        assert request(psp_server, route)[0] == 404
    connection = http.client.HTTPConnection("127.0.0.1", psp_server.server_port, timeout=5)
    connection.request("GET", "/r/wrong-secret/api/psp/episodes")
    assert connection.getresponse().status == 404
    connection.close()


def test_psp_cannot_store_database_under_static_site(tmp_path):
    package = tmp_path / "package"
    (package / "site").mkdir(parents=True)
    with pytest.raises(ValueError, match="outside"):
        ReviewServer(
            package,
            "synthetic_psp_capability_123456789",
            0,
            psp=PSPStore(package / "site/clinical"),
            public_origin="https://review.example",
        )


def test_psp_collection_survives_deleted_linked_mri(psp_server):
    status, _, raw = request(psp_server, "api/psp/episodes", draft(upload_id="upload-" + "a" * 32))
    assert status == 201
    record = json.loads(raw)
    psp_server.uploads.get = lambda _: None
    status, _, raw = request(
        psp_server,
        "api/psp/episodes/" + record["episode_id"],
        {
            "revision": record["revision"],
            "idempotency_key": uuid.uuid4().hex,
            "upload_id": record["upload_id"],
            "answers": {"Q06": {"current": 0}},
        },
    )
    assert status == 200, raw
    assert json.loads(raw)["mri"]["available"] is False
    assert json.loads(raw)["answers"]["Q06"]["current"] == 0
