import csv

import pytest

from ai_spine_med.localization_finetuning_data import rsna_train_records, train_labels


def test_only_rsna_train_is_selected_and_original_failures_remain():
    rows = [
        {"dataset": "rsna", "study_id": "1", "split": "train", "status": "failed"},
        {"dataset": "rsna", "study_id": "2", "split": "validation"},
        {"dataset": "aisslab", "study_id": "3", "split": "train"},
    ]
    assert rsna_train_records({"records": rows}) == rows[:1]


def test_test_rejected_before_any_training_inputs_are_read():
    with pytest.raises(ValueError, match="TEST is forbidden"):
        rsna_train_records({"records": [{"split": "test"}]})


@pytest.mark.parametrize(
    "dataset,split", [("rsna", "validation"), ("aisslab", "train"), ("rsna", "test")]
)
def test_non_training_grade_join_rejected_before_file_access(dataset, split):
    with pytest.raises(ValueError, match="RSNA TRAIN only"):
        train_labels([{"dataset": dataset, "split": split}], "does-not-exist.csv")


def test_training_label_mapping_preserves_unknown_grade(tmp_path):
    path = tmp_path / "labels.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["study_id", "spinal_canal_stenosis_l4_l5", "spinal_canal_stenosis_l5_s1"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "study_id": "1",
                "spinal_canal_stenosis_l4_l5": "",
                "spinal_canal_stenosis_l5_s1": "Severe",
            }
        )
    labels = train_labels([{"dataset": "rsna", "split": "train", "study_id": "1"}], path)
    assert labels[("rsna", "1", "L4-L5")] is None
    assert labels[("rsna", "1", "L5-S1")] == 2
