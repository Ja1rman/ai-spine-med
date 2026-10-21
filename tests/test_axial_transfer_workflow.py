"""Workflow wiring and phase boundaries use metadata and mocked downloads only."""

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_spine_med import axial_test
from ai_spine_med import axial_transfer as transfer
from ai_spine_med import axial_transfer_cache as cache
from ai_spine_med import axial_transfer_report as report
from ai_spine_med import axial_transfer_workflow as workflow


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def metadata_workflow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    paths = {
        "PREPARED": tmp_path / "prepared",
        "SOURCES": tmp_path / "sources",
        "FITTING": tmp_path / "fit",
        "TEST": tmp_path / "test-cache",
        "RUN": tmp_path / "run",
        "PROTOCOL": tmp_path / "protocol.json",
        "RSNA_TEST": tmp_path / "test-manifest",
    }
    for key, value in paths.items():
        monkeypatch.setattr(workflow, key, value)
    paths["RUN"].mkdir()
    write_json(paths["PROTOCOL"], {"seeds": [42, 2024, 3407]})
    write_json(paths["RUN"] / "selection.json", {"frozen": True, "seeds": [42, 2024, 3407]})
    fixed = tmp_path / "artifacts/rsna-v1"
    fixed.mkdir(parents=True)
    labels_path = fixed / "labels.csv"
    with labels_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["study_id", "level", "split", "label"])
        writer.writeheader()
        writer.writerows(
            [
                {"study_id": "1", "level": "L4-L5", "split": "train", "label": 0},
                {"study_id": "2", "level": "L5-S1", "split": "validation", "label": 1},
                *(
                    {"study_id": study, "level": level, "split": "test", "label": grade}
                    for study, grades in [("30", [2, 0]), ("31", [1, None])]
                    for level, grade in zip(("L4-L5", "L5-S1"), grades)
                ),
            ]
        )
    write_json(fixed / "summary.json", {"output_sha256": {"labels.csv": sha(labels_path)}})
    (fixed / "studies.csv").write_text("study_id,split\n30,test\n31,test\n")
    rows = [
        {
            "study_id": study,
            "series_id": series,
            "split": "test",
            "series_description": "Axial T2",
            "series_path": f"train_images/{study}/{series}",
        }
        for study, series in (("30", "300"), ("30", "301"), ("31", "310"))
    ]
    summary = {
        "manifest_sha256": "a" * 64,
        "n_studies": 2,
        "n_series": 3,
        "input_paths": {"studies.csv": str(fixed / "studies.csv")},
    }
    monkeypatch.setattr(axial_test, "validate_rsna_axial_test_manifest", lambda *_: (rows, summary))
    new_records = [
        {
            "dataset": "aisslab",
            "study_id": "external-heldout",
            "split": "test",
            "labels": [None, 2],
            "source_root": str(paths["PREPARED"] / "extracted"),
            "source": {
                "series": [
                    {
                        "series_id": "70",
                        "series_path": "external/70",
                        "files": [{"name": "1.dcm", "bytes": 1, "sha256": "b" * 64}],
                    }
                ]
            },
        }
    ]
    write_json(
        paths["SOURCES"] / "new-test-sources.json",
        {"records": new_records, "provenance": {"private_receipt": "synthetic"}},
    )
    write_json(
        paths["SOURCES"] / "summary.json",
        {"new_test_sources_sha256": sha(paths["SOURCES"] / "new-test-sources.json")},
    )
    events = []

    def validate_selection(data_root, manifest, seal_path):
        events.append("seal")
        seal = json.loads(seal_path.read_text())
        assert seal["training_selection"]["sha256"] == sha(paths["RUN"] / "selection.json")
        return seal, sha(seal_path)

    def download(data_root, manifest, seal_path, workers=4):
        events.append("download")
        seal = json.loads(seal_path.read_text())
        assert seal["scope"] == axial_test.SCOPE and seal["frozen"] is True
        assert seal["test_manifest_sha256"] == summary["manifest_sha256"]
        assert seal["training_selection"]["sha256"] == sha(paths["RUN"] / "selection.json")
        assert seal["protocol"]["sha256"] == sha(paths["PROTOCOL"])
        assert workers == 3

    def inventory(data_root, manifest, seal_path, out):
        events.append("inventory")
        source_rows = [
            {
                **row,
                "files": [{"name": "1.dcm", "bytes": 3, "sha256": "c" * 64}],
                "completion_receipt_sha256": "d" * 64,
            }
            for row in rows
        ]
        write_json(out, {"source_series": source_rows})

    monkeypatch.setattr(axial_test, "download_rsna_axial_test", download)
    monkeypatch.setattr(axial_test, "validate_transfer_selection", validate_selection)
    monkeypatch.setattr(axial_test, "build_source_inventory", inventory)
    return SimpleNamespace(
        paths=paths, labels_path=labels_path, fixed=fixed, events=events, new_records=new_records
    )


def test_missing_training_selection_prevents_all_test_downloads(metadata_workflow):
    e = metadata_workflow
    (e.paths["RUN"] / "selection.json").unlink()
    with pytest.raises(ValueError, match="seal|training"):
        workflow.prepare_evaluation_sources(workers=3)
    assert not e.events
    assert not (e.paths["SOURCES"] / "evaluation-sources.json").exists()


def test_rejected_seal_is_not_bypassed_by_source_preparation(metadata_workflow, monkeypatch):
    e = metadata_workflow

    def blocked(*args, **kwargs):
        e.events.append("guard_rejected")
        raise axial_test.ArchiveDownloadError("TEST remains sealed")

    monkeypatch.setattr(axial_test, "validate_transfer_selection", blocked)
    with pytest.raises(axial_test.ArchiveDownloadError, match="sealed"):
        workflow.prepare_evaluation_sources(workers=3)
    assert e.events == ["guard_rejected"]
    assert not (e.paths["SOURCES"] / "evaluation-sources.json").exists()


def test_combined_test_sources_preserve_labels_relative_file_paths_and_seal(metadata_workflow):
    e = metadata_workflow
    result_path = workflow.prepare_evaluation_sources(workers=3)
    source = json.loads(result_path.read_text())
    assert e.events == ["seal", "download", "inventory"]
    assert source["scope"] == axial_test.SCOPE
    rows = {(r["dataset"], r["study_id"]): r for r in source["records"]}
    assert set(rows) == {("aisslab", "external-heldout"), ("rsna", "30"), ("rsna", "31")}
    assert rows["aisslab", "external-heldout"] == e.new_records[0]
    assert rows["rsna", "30"]["labels"] == [2, 0]
    assert rows["rsna", "31"]["labels"] == [1, None]
    assert len(rows["rsna", "30"]["source"]["series"]) == 2
    assert all(r["split"] == "test" for r in rows.values())
    for row in rows.values():
        for series in row["source"]["series"]:
            assert not Path(series["series_path"]).is_absolute()
            assert series["files"][0]["name"] == "1.dcm"
    assert source["provenance"]["training_selection_sha256"] == sha(
        e.paths["RUN"] / "selection.json"
    )
    assert source["provenance"]["rsna_test_labels_sha256"] == sha(e.labels_path)
    before = result_path.read_bytes()
    assert workflow.prepare_evaluation_sources(workers=3) == result_path
    assert result_path.read_bytes() == before


def test_changed_original_rsna_labels_rejected_before_download(metadata_workflow):
    e = metadata_workflow
    e.labels_path.write_text(e.labels_path.read_text() + "\n")
    with pytest.raises(ValueError, match="label|Label|checksum"):
        workflow.prepare_evaluation_sources(workers=3)
    assert e.events == ["seal"]
    assert not (e.paths["SOURCES"] / "evaluation-sources.json").exists()


def test_changed_external_heldout_sources_cannot_be_published(metadata_workflow):
    e = metadata_workflow
    path = e.paths["SOURCES"] / "new-test-sources.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="held-out|changed"):
        workflow.prepare_evaluation_sources(workers=3)
    assert not (e.paths["SOURCES"] / "evaluation-sources.json").exists()


def test_evaluate_wires_test_only_cache_and_fixed_selection(tmp_path, monkeypatch):
    events = []
    test, run, sources = (tmp_path / name for name in ("test", "run", "combined-sources.json"))
    monkeypatch.setattr(workflow, "TEST", test)
    monkeypatch.setattr(workflow, "RUN", run)

    def prepare(*, workers):
        events.append("prepare")
        assert workers == 2
        return sources

    def build(path, out, *, device, selection):
        events.append("cache")
        assert (path, out, device, selection) == (sources, test, "cpu", run / "selection.json")

    def evaluate(index, run_dir, out, *, device):
        events.append("evaluate")
        assert (index, run_dir, out, device) == (
            test / "index.json",
            run,
            run / "evaluation",
            "cpu",
        )
        return {"complete": True}

    monkeypatch.setattr(workflow, "prepare_evaluation_sources", prepare)
    monkeypatch.setattr(cache, "build_transfer_cache", build)
    monkeypatch.setattr(transfer, "evaluate_axial_transfer", evaluate)
    assert workflow.evaluate("cpu", workers=2) == {"complete": True}
    assert events == ["prepare", "cache", "evaluate"]


def test_report_dispatch_uses_the_supported_public_api(tmp_path, monkeypatch, capsys):
    paths = [tmp_path / name for name in ("run", "prepared", "sources")]
    for name, path in zip(("RUN", "PREPARED", "SOURCES"), paths):
        monkeypatch.setattr(workflow, name, path)
    called = []

    def build_report(*args):
        called.append(args)
        return {"schema_version": 1, "aggregate_only": True}

    monkeypatch.setattr(report, "report_axial_transfer", build_report)
    workflow.main(["report"])
    assert called == [tuple(paths)]
    assert json.loads(capsys.readouterr().out) == {"schema_version": 1, "aggregate_only": True}
