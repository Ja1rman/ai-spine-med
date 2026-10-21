"""Synthetic fixed predictions verify reporting, pairing and public privacy."""

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ai_spine_med import axial_transfer_report as report
from ai_spine_med.training import evaluate_predictions


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_predictions(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def completed(tmp_path):
    run, prepared, sources = (tmp_path / name for name in ("run", "prepared", "sources"))
    (run / "evaluation").mkdir(parents=True)
    external = [
        {"dataset": "aisslab", "study_id": study, "split": split, "labels": labels}
        for study, split, labels in (
            ("PRIVATE_NEW_TRAIN", "train", [0, 1]),
            ("PRIVATE_NEW_VAL", "validation", [1, 2]),
            ("PRIVATE_NEW_ONE", "test", [0, 1]),
            ("PRIVATE_NEW_TWO", "test", [2, None]),
            ("PRIVATE_NEW_THREE", "test", [2, 0]),
        )
    ]
    write_json(prepared / "records.json", external)
    write_json(
        prepared / "manifest.json",
        {
            "dataset": "aisslab",
            "archive_verified": True,
            "patients": 5,
            "patients_by_split": {"train": 1, "validation": 1, "test": 3},
            "output_sha256": {"records.json": sha(prepared / "records.json")},
            "sensitive_unused_metadata": "PRIVATE_PATIENT_NAME_NOT_FOR_PUBLIC",
        },
    )
    fitting = [r for r in external if r["split"] != "test"]
    new_test = [r for r in external if r["split"] == "test"]
    rsna = [
        {"dataset": "rsna", "study_id": "PRIVATE_RSNA_ONE", "split": "test", "labels": [0, 2]},
        {"dataset": "rsna", "study_id": "PRIVATE_RSNA_TWO", "split": "test", "labels": [1, 2]},
    ]
    write_json(sources / "fitting-sources.json", {"records": fitting})
    write_json(sources / "new-test-sources.json", {"records": new_test})
    write_json(
        sources / "summary.json",
        {
            "fitting_sources_sha256": sha(sources / "fitting-sources.json"),
            "new_test_sources_sha256": sha(sources / "new-test-sources.json"),
            "replay_studies": 1,
        },
    )
    write_json(
        sources / "evaluation-sources.json",
        {
            "schema_version": 1,
            "scope": "external_axial_v4_rsna_test",
            "records": new_test + rsna,
            "private_url": "https://example.invalid/private?token=SYNTHETIC_CAPABILITY",
        },
    )
    config = {
        "label_mapping_authorized_by_user": True,
        "clinical_scale_equivalence_verified": False,
        "classes": list(report.CLASSES),
        "levels": list(report.LEVELS),
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 2026,
    }
    runs = []
    scores = {42: [0.5, 0.55, 0.53], 2024: [0.6, 0.6, 0.59], 3407: [0.55, 0.61, 0.62]}
    for seed in report.SEEDS:
        directory = run / "runs" / f"seed-{seed}"
        directory.mkdir(parents=True)
        for name in ("before.pt", "model.pt"):
            (directory / name).write_bytes(f"SYNTHETIC-{seed}-{name}".encode())
        history = [
            {
                "epoch": i,
                "training_loss": None if i == 0 else 0.7 / i,
                "validation_loss_unweighted": 0.7 - i * 0.05,
                "validation_macro_f1": value,
            }
            for i, value in enumerate(scores[seed])
        ]
        write_json(directory / "history.json", history)
        best = int(np.argmax(scores[seed]))
        runs.append(
            {
                "seed": seed,
                "best_epoch": best,
                "epochs_run": 2,
                "checkpoint": f"runs/seed-{seed}/model.pt",
                "checkpoint_sha256": sha(directory / "model.pt"),
                "before_checkpoint": f"runs/seed-{seed}/before.pt",
                "before_checkpoint_sha256": sha(directory / "before.pt"),
                "metrics": {"conditional": {"macro_f1": scores[seed][best]}},
            }
        )
    training = {
        "schema_version": 1,
        "status": "complete",
        "model_family": "axial_transfer_v1",
        "seeds": list(report.SEEDS),
        "runs": runs,
        "protocol_sha256": "a" * 64,
        "index_sha256": "b" * 64,
        "configuration": config,
        "n_fitting_studies": 2,
        "n_replay_studies": 1,
        "n_new_validation_studies": 1,
        "fitting_class_counts": [1, 1, 2],
        "class_weights": [1.1, 1.1, 0.8],
    }
    write_json(run / "summary.json", training)
    selected = {
        "schema_version": 1,
        "model_family": "axial_transfer_v1",
        "frozen": True,
        "seeds": list(report.SEEDS),
        "protocol_sha256": "a" * 64,
        "training_index_sha256": "b" * 64,
        "training_summary_sha256": sha(run / "summary.json"),
        "runs": runs,
    }
    write_json(run / "selection.json", selected)
    tables = {}
    before = [0, 0, 1, 0, None, 0, 0, 0, 0, 2]
    after = [0, 1, 2, 1, None, 1, 0, 2, 1, 2]
    for name in report.ENTRIES:
        predictions = before if name.startswith("before") else after
        rows = []
        for patient in new_test + rsna:
            for level, label in zip(report.LEVELS, patient["labels"]):
                prediction = predictions[len(rows)]
                row = {
                    "dataset": patient["dataset"],
                    "study_id": patient["study_id"],
                    "level": level,
                    "split": "test",
                    "label": label,
                    "reference_status": "missing" if label is None else "available",
                    "status": "ok" if prediction is not None else "not_assessed",
                    "reason": ""
                    if prediction is not None
                    else "missing_reference_label"
                    if label is None
                    else "not_axial_within_supported_obliquity",
                    "prediction": prediction,
                    **{
                        f"p_{c}": None if prediction is None else 0.8 if j == prediction else 0.1
                        for j, c in enumerate(report.CLASSES)
                    },
                }
                rows.append(row)
        tables[name] = rows
    evaluation = {
        "schema_version": 1,
        "model_family": "axial_transfer_v1",
        "training_performed": False,
        "selection_sha256": sha(run / "selection.json"),
        "training_index_sha256": "b" * 64,
        "entries": {},
    }
    for name, rows in tables.items():
        path = run / "evaluation" / f"{name}.csv"
        write_predictions(path, rows)
        evaluation["entries"][name] = {
            "metrics": evaluate_predictions(rows),
            "by_dataset": {
                d: evaluate_predictions([r for r in rows if r["dataset"] == d])
                for d in report.DATASETS
            },
            "predictions_sha256": sha(path),
        }
    write_json(run / "evaluation" / "summary.json", evaluation)
    return SimpleNamespace(
        run=run,
        prepared=prepared,
        sources=sources,
        evaluation=evaluation,
        tables=tables,
        document=tmp_path / "public" / "axial-transfer.md",
        figures=tmp_path / "public" / "images",
    )


def execute(e):
    return report.report_axial_transfer(
        e.run, e.prepared, e.sources, document=e.document, figures=e.figures
    )


def refresh_entry(e, name):
    rows = e.tables[name]
    path = e.run / "evaluation" / f"{name}.csv"
    write_predictions(path, rows)
    e.evaluation["entries"][name] = {
        "metrics": evaluate_predictions(rows),
        "by_dataset": {
            d: evaluate_predictions([r for r in rows if r["dataset"] == d]) for d in report.DATASETS
        },
        "predictions_sha256": sha(path),
    }
    write_json(e.run / "evaluation" / "summary.json", e.evaluation)


def test_report_pairs_patients_preserves_failure_denominators_and_contains_no_ids(completed):
    e = completed
    public = execute(e)
    assert public["training"]["epoch_zero_selected"] == 1
    assert [r["best_epoch"] for r in public["training"]["runs"]] == [1, 0, 2]
    aiss = public["datasets"]["aisslab"]
    metric = aiss["entries"]["after_ensemble"]["metrics"]
    assert (
        metric["n_requested"],
        metric["n_labeled"],
        metric["n_assessed"],
        metric["n_not_assessed"],
    ) == (6, 5, 4, 1)
    assert metric["severe"]["n_total"] == 2
    assert metric["severe"]["detection_rate_all_requested"] == 0.5
    assert metric["severe"]["recall_conditional"] == 1
    assert metric["severe"]["missed_all_requested"] == 1
    assert aiss["bootstrap"]["excluded_missing_reference_targets"] == 1
    assert aiss["bootstrap"]["n_labeled_targets"] == 5
    assert aiss["bootstrap"]["n_studies_with_known_reference"] == 3
    assert aiss["bootstrap"]["resamples"] == 2000
    assert aiss["failures_by_reason"] == {
        "not_axial_within_supported_obliquity": 1,
    }
    assert aiss["missing_reference"] == {"n_targets": 1, "n_predicted": 1, "n_unprocessed": 0}
    assert sum(aiss["transitions"].values()) == 4
    delta = aiss["bootstrap"]["contrasts"]["after_minus_before_ensemble"]["metrics"]["macro_f1"][
        "point"
    ]
    expected = (
        metric["conditional"]["macro_f1"]
        - aiss["entries"]["before_ensemble"]["metrics"]["conditional"]["macro_f1"]
    )
    assert delta == pytest.approx(expected)
    serialized = json.dumps(public) + e.document.read_text()
    for private in (
        "PRIVATE_",
        "SYNTHETIC_CAPABILITY",
        str(e.run),
        "study_id",
        "series_id",
        "source_root",
        "private_url",
    ):
        assert private not in serialized
    assert "Клиническая эквивалентность" in e.document.read_text()
    assert (e.document.parent / "axial-transfer-results.json").is_file()
    assert all(
        (e.figures / f"axial_transfer_{name}.png").stat().st_size > 1000
        for name in ("comparison", "confusion", "learning")
    )


@pytest.mark.parametrize(
    "cause",
    [
        "csv_bytes",
        "coherent_label",
        "cohort_status",
        "ensemble",
        "stored_metric",
        "unsafe_reason",
        "missing_entry",
        "checkpoint",
    ],
)
def test_tampered_or_misaligned_result_never_creates_public_outputs(completed, cause):
    e = completed
    name = "after_ensemble"
    if cause == "csv_bytes":
        path = e.run / "evaluation" / f"{name}.csv"
        path.write_text(path.read_text() + "\n")
    elif cause == "checkpoint":
        (e.run / "runs/seed-42/model.pt").write_bytes(b"CHANGED")
    elif cause == "missing_entry":
        del e.evaluation["entries"]["before_seed_42"]
        write_json(e.run / "evaluation" / "summary.json", e.evaluation)
    elif cause == "stored_metric":
        e.evaluation["entries"][name]["by_dataset"]["aisslab"]["conditional"]["macro_f1"] += 0.01
        write_json(e.run / "evaluation" / "summary.json", e.evaluation)
    else:
        row = e.tables[name][0]
        if cause == "coherent_label":
            row["label"] = 1
        elif cause == "cohort_status":
            row.update(
                status="not_assessed",
                reason="missing_or_invalid_geometry",
                prediction=None,
                **{f"p_{c}": None for c in report.CLASSES},
            )
        elif cause == "ensemble":
            row.update(p_normal_mild=0.7, p_moderate=0.2, p_severe=0.1)
        else:
            e.tables[name][4]["reason"] = "private_patient_12345"
        refresh_entry(e, name)
    with pytest.raises(ValueError):
        execute(e)
    assert not e.document.parent.exists()


def test_nonconsecutive_or_posthoc_selected_history_is_rejected(completed):
    e = completed
    path = e.run / "runs/seed-42/history.json"
    history = json.loads(path.read_text())
    history[2]["validation_macro_f1"] = 0.8
    write_json(path, history)
    with pytest.raises(ValueError, match="earliest validation improvement"):
        execute(e)
    assert not e.document.parent.exists()
