import csv
import json

import pytest

from ai_spine_med import localization_test as lt


def index(rows):
    return {
        "configuration": {
            "phase": "evaluation_image_inputs",
            "reference_fields_in_inputs": False,
            "input_fields": ["features", "positions"],
        },
        "records": rows,
    }


def row(study="t", split="test", dataset="rsna"):
    return {"dataset": dataset, "study_id": study, "split": split, "status": "ok"}


def test_rsna_test_only_and_no_fitting_patient_overlap():
    rows = [row(), row("a", dataset="aisslab")]
    assert lt.test_records(index(rows), {"records": [row("train", "train")]}) == rows[:1]
    with pytest.raises(ValueError, match="overlap"):
        lt.test_records(index(rows), {"records": [row("t", "validation")]})


@pytest.mark.parametrize(
    "field", ["anchors", "label", "labels", "relevance", "center_xy", "level_mask"]
)
def test_reference_fields_are_rejected_before_any_medical_file_is_opened(field):
    entry = row()
    entry[field] = [2]
    with pytest.raises(ValueError, match="Reference fields"):
        lt.test_records(index([entry]), {"records": []})


def test_non_test_split_rejected_even_in_excluded_dataset():
    with pytest.raises(ValueError, match="image-only TEST"):
        lt.test_records(index([row(), row("a", "validation", "aisslab")]), {"records": []})


def test_duplicate_test_identity_is_not_double_counted():
    with pytest.raises(ValueError, match="duplicate"):
        lt.test_records(index([row(), row()]), {"records": []})


def test_reference_join_rejects_validation_before_file_access():
    with pytest.raises(ValueError, match="RSNA TEST only"):
        lt.test_labels([row(split="validation")], "missing.csv")


def test_explicit_missing_grade_kept_and_training_rows_not_joined(tmp_path):
    path = tmp_path / "labels.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["study_id", "spinal_canal_stenosis_l4_l5", "spinal_canal_stenosis_l5_s1"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "study_id": "t",
                "spinal_canal_stenosis_l4_l5": "",
                "spinal_canal_stenosis_l5_s1": "Severe",
            }
        )
        writer.writerow(
            {
                "study_id": "train",
                "spinal_canal_stenosis_l4_l5": "Normal/Mild",
                "spinal_canal_stenosis_l5_s1": "Moderate",
            }
        )
    labels = lt.test_labels([row()], path)
    assert labels == {("rsna", "t", "L4-L5"): None, ("rsna", "t", "L5-S1"): 2}


def test_wrong_selection_scope_rejected_before_model_weights_opened(tmp_path, monkeypatch):
    monkeypatch.setattr(lt, "V9", tmp_path)
    (tmp_path / "selection.json").write_text(json.dumps({"frozen": False}))
    (tmp_path / "summary.json").write_text(
        json.dumps({"phase": "validation", "test_evaluated": False})
    )
    with pytest.raises(ValueError, match="validation-selected"):
        lt.load_fixed_classifiers("cpu")


def test_failure_export_has_no_imputed_normal_category(tmp_path):
    path = tmp_path / "predictions.csv"
    r = {
        "dataset": "rsna",
        "study_id": "t",
        "split": "test",
        "level": "L5-S1",
        "label": 1,
        "status": "not_assessed",
        "reason": "missing_vertebra:S1",
        "prediction": None,
        "p_normal_mild": None,
        "p_moderate": None,
        "p_severe": None,
    }
    lt._write(path, [r])
    with path.open() as stream:
        saved = list(csv.DictReader(stream))[0]
    assert saved["label"] == "1" and saved["reference_status"] == "known"
    assert saved["prediction"] == saved["p_normal_mild"] == ""
