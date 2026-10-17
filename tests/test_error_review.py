"""Private gallery selection and rendering use synthetic pixels only."""

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ai_spine_med import error_review as e
from ai_spine_med.assets import sha256_file
from ai_spine_med.features import TARGET_FIELDS
from ai_spine_med.training import CLASS_NAMES, SEEDS, load_feature_data


def _write_csv(path, fields, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _candidates():
    result = []
    conditions = ((1, 0), (2, 0), (2, 1), (0, 2), (0, 1), (1, 1), (2, 2))
    for number in range(112):
        label, prediction = conditions[number % 7]
        result.append(
            {
                "study_id": str(number),
                "series_id": str(number + 100),
                "level": "L5-S1" if number % 2 else "L4-L5",
                "split": "train" if number % 3 == 0 else "validation",
                "label": label,
                "status": "ok",
                "ensemble_prediction": prediction,
                "head_predictions": [0, 1, 2] if number % 7 == 5 else [prediction] * 3,
            }
        )
    for number in range(112, 116):
        result.append(
            {
                "study_id": str(number),
                "series_id": str(number + 100),
                "level": "L5-S1",
                "split": "validation",
                "label": 2,
                "status": "not_assessed",
                "reason": "missing_pair",
                "ensemble_prediction": None,
                "head_predictions": [],
            }
        )
    return result


def test_deterministic_purposive_quotas_controls_and_no_test():
    candidates = _candidates()
    first = e.select_error_targets(candidates)
    second = e.select_error_targets(list(reversed(candidates)))
    assert first == second
    assert len(first) == 48
    assert len({(row["study_id"], row["level"]) for row in first}) == 48
    assert sum(row["sampling_reason"] == "correct_control" for row in first) == 8
    assert sum(row["split"] == "validation" for row in first) > 24
    assert {row["sampling_reason"] for row in first}.issuperset(
        {category for category, _ in e.QUOTAS}
    )
    assert all(row["case_id"].startswith("case-") for row in first)
    assert first != e.select_error_targets(candidates, seed=2027)
    bad = [{**candidates[0], "split": "test"}]
    with pytest.raises(ValueError, match="test is forbidden"):
        e.select_error_targets(bad)


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    monkeypatch.setattr(e, "_project_root", lambda: tmp_path)
    (tmp_path / ".gitignore").write_text("reports/runs/\n")
    features, heads, data = (tmp_path / name for name in ("features", "heads", "data"))
    (features / "records").mkdir(parents=True)
    targets = []
    for index, (split, label) in enumerate(
        (("train", 1), ("validation", 2), ("validation", 0), ("validation", 0), ("validation", 1)),
        1,
    ):
        assessed = index < 5
        target = {
            "study_id": str(index),
            "series_id": str(index + 100),
            "split": split,
            "level": "L4-L5" if index % 2 else "L5-S1",
            "label": label,
            "status": "ok" if assessed else "not_assessed",
            "reason": "" if assessed else "missing_pair",
            "roi_index": 0 if assessed else "",
            "roi_path": f"records/{index}_{index + 100}.npz" if assessed else "",
        }
        targets.append(target)
        source = data / "train_images" / str(index) / str(index + 100)
        source.mkdir(parents=True)
        files = []
        for name in ("1.dcm", "2.dcm"):
            (source / name).write_bytes(b"SYNTHETIC PIXELS PROVIDED BY TEST LOADER")
            files.append(
                {
                    "name": name,
                    "sha256": sha256_file(source / name),
                    "bytes": (source / name).stat().st_size,
                }
            )
        record = {
            "cache_id": "synthetic-cache",
            "source": {"files": files, "sha256": "0" * 64},
            "targets": [target],
            "arrays_path": target["roi_path"],
            "arrays_sha256": "",
            "vertices": [
                {
                    "label": name,
                    "slice_nos": [1, 2],
                    "average_polygon": [[1, 1], [4, 1], [4, 4], [1, 4]],
                }
                for name in target["level"].split("-")
            ],
        }
        if assessed:
            roi = np.arange(9 * 112 * 224, dtype=np.float64).reshape(1, 9, 112, 224) / 100000
            np.savez(features / target["roi_path"], rois=roi, levels=np.asarray([target["level"]]))
            record["arrays_sha256"] = sha256_file(features / target["roi_path"])
        (features / "records" / f"{index}_{index + 100}.json").write_text(json.dumps(record))
    assessed = targets[:4]
    np.savez(
        features / "features.npz",
        features=np.ones((4, 512), np.float32),
        original_logits=np.zeros((4, 4), np.float32),
        labels=np.array([row["label"] for row in assessed]),
        **{
            name: np.asarray([row[name] for row in assessed], dtype="U32")
            for name in ("study_id", "series_id", "level", "split")
        },
    )
    _write_csv(features / "target_manifest.csv", TARGET_FIELDS, targets)
    (features / "metadata.json").write_text(
        json.dumps(
            {
                "cache_id": "synthetic-cache",
                "extraction_complete": True,
                "outputs_sha256": {
                    name: sha256_file(features / name)
                    for name in ("features.npz", "target_manifest.csv")
                },
            }
        )
    )
    feature_data = load_feature_data(features)
    runs = []
    for seed in SEEDS:
        path = heads / "runs" / f"seed-{seed}" / "head.pt"
        path.parent.mkdir(parents=True)
        head = torch.nn.Linear(512, 3)
        head.weight.data.zero_()
        head.bias.data.copy_(torch.tensor([2, 0, -1]))
        torch.save(
            {
                "model": "Linear(512,3)",
                "feature_size": 512,
                "class_names": list(CLASS_NAMES),
                "seed": seed,
                "feature_cache_sha256": feature_data.file_sha256,
                "state_dict": head.state_dict(),
            },
            path,
        )
        runs.append({"seed": seed, "checkpoint_sha256": sha256_file(path)})
    (heads / "summary.json").write_text(json.dumps({"runs": runs, "test_evaluated": False}))
    (heads / "manifest.json").write_text(
        json.dumps({"feature_cache_sha256": feature_data.file_sha256})
    )
    monkeypatch.setattr(
        e,
        "load_sagittal_series",
        lambda _: SimpleNamespace(
            volume=np.arange(16 * 16 * 5, dtype=np.float32).reshape(16, 16, 5),
            pixel_spacing=(1.0, 1.0),
        ),
    )
    return features, heads, data, tmp_path / "reports/runs/review"


def test_private_offline_gallery_preserves_blank_clinician_form_and_source_failures(experiment):
    features, heads, data, out = experiment
    (data / "train_images/2/102/1.dcm").write_bytes(b"changed")
    summary = e.generate_error_review(features, heads, data, out, count=5)
    assert summary["selected_cases"] == 5
    assert summary["distinct_study_levels"] == 5
    assert summary["png_files"] == 6
    assert len(list(out.glob("*.png"))) == 6
    assert summary["technical"]["source_display_status"] == {"ok": 4, "failed": 1}
    assert summary["technical"]["roi_display_status"]["not_available"] == 1
    assert summary["clinical_review"] == "not_performed"
    assert summary["representative_accuracy_sample"] is False
    assert summary["test_accessed"] is False
    rows = list(csv.DictReader((out / "doctor_review.csv").open()))
    assert len(rows) == 5
    assert all(row[field] == "" for row in rows for field in e.CLINICIAN_FIELDS)
    assert all(
        {"study_id", "series_id", "sampling_reason", "reference", "ensemble_prediction"}.issubset(
            row
        )
        for row in rows
    )
    page = (out / "index.html").read_text()
    assert "http://" not in page and "https://" not in page
    assert "<script" not in page and "<form" not in page
    assert "not a representative accuracy sample" in page
    assert "mean-prob ensemble" in page
    assert (out / "files_sha256.json").is_file()
    with pytest.raises(ValueError, match="preserve existing physician answers"):
        e.generate_error_review(features, heads, data, out, count=5)


def test_output_in_docs_or_symlink_escape_is_rejected_before_pixel_access(experiment, monkeypatch):
    features, heads, data, out = experiment
    root = e._project_root()
    touched = []
    monkeypatch.setattr(e, "load_feature_data", lambda _: touched.append(True))
    with pytest.raises(ValueError, match="ignored reports/runs"):
        e.generate_error_review(features, heads, data, root / "docs/images")
    assert touched == []
    (root / "reports/runs").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "reports/runs/escape").symlink_to(root / "docs", target_is_directory=True)
    with pytest.raises(ValueError, match="ignored reports/runs"):
        e.generate_error_review(features, heads, data, root / "reports/runs/escape/private")
    assert touched == []


def test_changed_head_is_rejected_and_test_cache_is_never_accepted(experiment):
    features, heads, data, out = experiment
    path = heads / "runs/seed-42/head.pt"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        e.generate_error_review(features, heads, data, out)
    assert not out.exists()
    # A train/validation reader rejects this before any checkpoint or pixels.
    with np.load(features / "features.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["split"][0] = "test"
    np.savez(features / "features.npz", **arrays)
    metadata = json.loads((features / "metadata.json").read_text())
    metadata["outputs_sha256"]["features.npz"] = sha256_file(features / "features.npz")
    (features / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Only train and validation"):
        e.generate_error_review(features, heads, data, out)
    assert not out.exists()


def test_unsafe_source_name_or_roi_path_is_not_rendered(experiment):
    features, heads, data, out = experiment
    record_path = features / "records/1_101.json"
    record = json.loads(record_path.read_text())
    record["source"]["files"][0]["name"] = "../../outside.dcm"
    record_path.write_text(json.dumps(record))
    summary = e.generate_error_review(features, heads, data, out, count=5)
    assert summary["technical"]["source_display_status"]["failed"] == 1
    assert summary["technical"]["roi_display_status"]["ok"] == 4


def _development_experiment(review, directory):
    cases = json.loads((review / "cases.json").read_text())
    gallery = json.loads((review / "summary.json").read_text())
    candidates = {}
    fields = [
        "study_id",
        "series_id",
        "level",
        "split",
        "label",
        "status",
        "reason",
        "prediction",
        "p_normal_mild",
        "p_moderate",
        "p_severe",
    ]
    for variant in ("linear", "mlp", "coral"):
        rows = []
        for case in cases:
            if case["split"] != "validation":
                continue
            prediction = (
                (case["label"] if variant == "mlp" else 0) if case["status"] == "ok" else None
            )
            probabilities = [0.1, 0.1, 0.1] if prediction is not None else []
            if probabilities:
                probabilities[prediction] = 0.8
            rows.append(
                {
                    **{
                        name: case[name]
                        for name in (
                            "study_id",
                            "series_id",
                            "level",
                            "split",
                            "label",
                            "status",
                            "reason",
                        )
                    },
                    "prediction": prediction if prediction is not None else "",
                    **{
                        f"p_{name}": probabilities[i] if probabilities else ""
                        for i, name in enumerate(CLASS_NAMES)
                    },
                }
            )
        path = directory / "validation" / variant / "ensemble/predictions.csv"
        path.parent.mkdir(parents=True)
        _write_csv(path, fields, rows)
        candidates[variant] = {
            "validation": {"ensemble": {"predictions_sha256": sha256_file(path)}}
        }
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "test_evaluated": False,
                "configuration": {"feature_cache_sha256": gallery["feature_cache_sha256"]},
                "candidates": candidates,
            }
        )
    )


def test_comparison_preserves_cases_pixels_and_physician_answers(experiment):
    features, heads, data, review = experiment
    e.generate_error_review(features, heads, data, review, count=5)
    development = e._project_root() / "head-v2"
    _development_experiment(review, development)
    doctor = (review / "doctor_review.csv").read_bytes()
    cases = (review / "cases.json").read_bytes()
    images = {path.name: sha256_file(path) for path in review.glob("*.png")}
    result = e.attach_development_comparison(review, development)
    assert result["n_fixed_cases"] == 5
    assert result["n_matched_validation"] == 4
    assert result["n_paired_assessed_validation"] == 3
    assert result["n_train_not_evaluated"] == 1
    assert result["by_model"]["mlp"]["corrected"] == 1
    assert result["by_model"]["mlp"]["not_evaluated"] == 1
    assert result["by_model"]["mlp"]["not_assessed"] == 1
    assert (review / "doctor_review.csv").read_bytes() == doctor
    assert (review / "cases.json").read_bytes() == cases
    assert {path.name: sha256_file(path) for path in review.glob("*.png")} == images
    assert 'href="comparison.html"' in (review / "index.html").read_text()
    page = (review / "comparison.html").read_text()
    assert "http://" not in page and "https://" not in page
    assert "не представляет обычную точность" in page
    rows = list(csv.DictReader((review / "comparison.csv").open()))
    assert len(rows) == 5
    assert next(row for row in rows if row["split"] == "train")["mlp_comparison"] == "not_evaluated"


@pytest.mark.parametrize("corruption", ["wrong_key", "mixed_label", "test_split", "hash"])
def test_comparison_rejects_wrong_keys_mixed_references_and_unbound_csv(experiment, corruption):
    features, heads, data, review = experiment
    e.generate_error_review(features, heads, data, review, count=5)
    development = e._project_root() / "head-v2"
    _development_experiment(review, development)
    summary_path = development / "summary.json"
    summary = json.loads(summary_path.read_text())
    variants = ("linear", "mlp", "coral") if corruption == "wrong_key" else ("mlp",)
    for variant in variants:
        path = development / "validation" / variant / "ensemble/predictions.csv"
        if corruption == "hash":
            path.write_text(path.read_text() + "\n")
            continue
        with path.open() as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        if corruption == "wrong_key":
            rows[0]["series_id"] = "999999"
        elif corruption == "mixed_label":
            rows[0]["label"] = str((int(rows[0]["label"]) + 1) % 3)
        else:
            rows[0]["split"] = "test"
        _write_csv(path, fields, rows)
        summary["candidates"][variant]["validation"]["ensemble"]["predictions_sha256"] = (
            sha256_file(path)
        )
    summary_path.write_text(json.dumps(summary))
    doctor = (review / "doctor_review.csv").read_bytes()
    with pytest.raises(ValueError):
        e.attach_development_comparison(review, development)
    assert (review / "doctor_review.csv").read_bytes() == doctor
    assert not (review / "comparison.csv").exists()
