from collections import Counter

from ai_spine_med.qc import select_qc_targets


def test_qc_selection_is_balanced_reproducible_and_training_only():
    records = []
    for label in range(3):
        for level in ("L4-L5", "L5-S1"):
            for index in range(8):
                records.append(
                    {
                        "targets": [
                            {
                                "study_id": f"{label}{level}{index}",
                                "series_id": str(index),
                                "split": "train",
                                "level": level,
                                "label": label,
                                "status": "ok",
                            }
                        ]
                    }
                )
    records.append(
        {
            "targets": [
                {
                    "study_id": "validation",
                    "series_id": "0",
                    "split": "validation",
                    "level": "L4-L5",
                    "label": 0,
                    "status": "ok",
                }
            ]
        }
    )
    records.append(
        {
            "targets": [
                {
                    "study_id": "test",
                    "series_id": "0",
                    "split": "test",
                    "level": "L5-S1",
                    "label": 2,
                    "status": "ok",
                }
            ]
        }
    )
    first = select_qc_targets(records)
    second = select_qc_targets(list(reversed(records)))
    assert [(row["study_id"], row["level"]) for row in first] == [
        (row["study_id"], row["level"]) for row in second
    ]
    assert len(first) == 24
    assert set(row["split"] for row in first) == {"train"}
    assert set(Counter((row["label"], row["level"]) for row in first).values()) == {4}
