"""Private launch preparation tests never create remote resources."""

import ast
import base64
import builtins
import csv
import gzip
import hashlib
import inspect
import io
import json
import symtable
import tarfile
from datetime import timedelta
from types import SimpleNamespace

import pytest

from ai_spine_med.kaggle_run import KERNEL_ID, _cloud_bootstrap, launch_prepared, prepare_launch
from ai_spine_med.rsna_bulk import _selected_series


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepared(tmp_path):
    project, raw, manifests, out = [
        tmp_path / name for name in ("project", "raw", "manifests", "launch")
    ]
    project.mkdir()
    raw.mkdir()
    manifests.mkdir()
    source = project / "src/ai_spine_med"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("")
    (source / "features.py").write_text("# frozen synthetic implementation\n")
    (source / "assets.lock.json").write_text("{}")
    (project / "pyproject.toml").write_text('[project]\nname="synthetic-only"\nversion="0.0.1"\n')
    (project / ".env").write_text("TOPSECRET_MUST_NEVER_ENTER_PAYLOAD")
    studies = [
        {"study_id": str(index), "split": split}
        for index, split in [(1, "train"), (2, "validation"), (3, "test")]
    ]
    series = [
        {
            **row,
            "series_id": "10",
            "series_description": "Sagittal T2/STIR",
            "series_path": f"train_images/{row['study_id']}/10",
        }
        for row in studies
    ]
    write_csv(manifests / "studies.csv", studies)
    write_csv(manifests / "series.csv", series)
    (manifests / "labels.csv").write_text("study_id,split,level,label\n")
    (manifests / "coordinates.csv").write_text("study_id,x,y\n")
    for name in ("train.csv", "train_series_descriptions.csv", "train_label_coordinates.csv"):
        (raw / name).write_text("synthetic,metadata\n")
    summary = {
        "output_sha256": {
            name: hashlib.sha256((manifests / name).read_bytes()).hexdigest()
            for name in ("studies.csv", "series.csv", "labels.csv", "coordinates.csv")
        },
        "input_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in raw.iterdir()
        },
    }
    (manifests / "summary.json").write_text(json.dumps(summary))
    _, binding = _selected_series(manifests)
    for index in (1, 2):
        directory = raw / f"train_images/{index}/10"
        directory.mkdir(parents=True)
        for name in ("1.dcm", "2.dcm"):
            (directory / name).write_bytes(b"SYNTHETIC-DICOM-ONLY")
        marker = {
            "complete": True,
            "expected_files": 2,
            "files": ["1.dcm", "2.dcm"],
            "manifest_sha256": binding,
            "metadata_sha256": "a" * 64,
            "source_identity_sha256": "b" * 64,
        }
        (directory / "_series_complete.json").write_text(json.dumps(marker))
    receipt = prepare_launch(project, raw, manifests, out)
    return out, receipt


def test_bootstrap_is_standalone_and_has_no_unbound_module_globals():
    source = inspect.getsource(_cloud_bootstrap)
    table = symtable.symtable(source, "bootstrap", "exec")
    allowed = set(dir(builtins))

    def check(scope):
        missing = {
            symbol.get_name()
            for symbol in scope.get_symbols()
            if symbol.is_global() and symbol.is_referenced() and symbol.get_name() not in allowed
        }
        assert not missing
        for child in scope.get_children():
            check(child)

    check(table.get_children()[0])


def test_prepare_embeds_only_allowlisted_frozen_source_and_private_receipts(tmp_path):
    out, receipt = prepared(tmp_path)
    script = (out / "bootstrap.py").read_text()
    assert receipt["script_bytes"] < 1_000_000
    assert receipt["n_series"] == 2
    assert receipt["n_dicom_files"] == 4
    assert receipt["max_session_seconds"] == 5400
    assert receipt["launch_state"] == "prepared_only_not_launched"
    compile(script, "bootstrap.py", "exec")
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(script).body
        if isinstance(node, ast.Assign)
    }
    blob = base64.b64decode(assignments["PAYLOAD_B64"])
    assert hashlib.sha256(blob).hexdigest() == assignments["PAYLOAD_SHA256"]
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(blob))) as archive:
        names = archive.getnames()
        assert "app/src/ai_spine_med/features.py" in names
        assert "receipts/1_10.json" in names
        assert not any(
            name.endswith(".dcm") or ".env" in name or "kaggle.json" in name for name in names
        )
        assert b"TOPSECRET" not in b"".join(archive.extractfile(name).read() for name in names)
        assert (
            archive.extractfile("app/src/ai_spine_med/features.py").read()
            == b"# frozen synthetic implementation\n"
        )
    metadata = json.loads((out / "kernel-metadata.json").read_text())
    assert metadata["is_private"] is True
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == []


class FakeApi:
    def __init__(self, paid=False, existing=False):
        self.paid = paid
        self.existing = existing
        self.pushes = []

    def quota_view(self):
        return SimpleNamespace(
            gpu_quota=SimpleNamespace(
                is_pay_to_scale_enabled=self.paid,
                total_time_allowed=timedelta(hours=30),
                time_used=timedelta(),
                time_reserved=timedelta(),
            )
        )

    def kernels_list(self, **kwargs):
        assert kwargs["mine"] is True
        return [SimpleNamespace(ref=KERNEL_ID)] if self.existing else []

    def kernels_push(self, folder, timeout, acc):
        self.pushes.append((folder, timeout, acc))
        return SimpleNamespace(
            error="",
            invalid_competition_sources=[],
            invalid_dataset_sources=[],
            version_number=1,
            kernel_id=123,
        )


def test_launch_requires_the_reviewed_private_code_and_free_quota(tmp_path):
    out, _ = prepared(tmp_path)
    api = FakeApi()
    response = launch_prepared(out, api=api)
    assert response["private"] is True
    assert api.pushes[0][1:] == ("5400", "NvidiaTeslaT4")


@pytest.mark.parametrize("mode", ["paid", "public", "modified_code", "existing_unrelated"])
def test_launch_guard_rejects_unreviewed_or_paid_actions(tmp_path, mode):
    out, _ = prepared(tmp_path)
    api = FakeApi(paid=mode == "paid", existing=mode == "existing_unrelated")
    if mode == "public":
        path = out / "kernel-metadata.json"
        metadata = json.loads(path.read_text())
        metadata["is_private"] = False
        path.write_text(json.dumps(metadata))
    elif mode == "modified_code":
        path = out / "bootstrap.py"
        path.write_text(path.read_text() + "\n# changed since review\n")
    with pytest.raises(ValueError):
        launch_prepared(out, api=api)
    assert not api.pushes
