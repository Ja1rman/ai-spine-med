import csv
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ai_spine_med import multimodal_review as review


def _candidate(number, category):
    prediction = {
        "fusion_corrected": (0, 1, 1),
        "fusion_worsened": (1, 0, 0),
        "shared_error": (0, 0, 2),
        "correct_control": (1, 1, 1),
    }[category]
    return {
        "study_id": str(number),
        "series_id": "12345",
        "level": "L4-L5",
        "label": 1,
        "split": "validation",
        "status": "ok",
        "predictions": {
            variant: {"prediction": value, "probabilities": [0.2, 0.6, 0.2]}
            for variant, value in zip(review.VARIANTS, prediction)
        },
    }


def test_purposive_selection_is_deterministic_balanced_and_rejects_test():
    candidates = [
        _candidate(group * 100 + number, category)
        for group, category in enumerate(tuple(review.REASONS)[:4])
        for number in range(8)
    ]
    selected = review.select_multimodal_cases(candidates)
    assert selected == review.select_multimodal_cases(list(reversed(candidates)))
    assert len(selected) == len({(row["study_id"], row["level"]) for row in selected}) == 24
    assert {
        category: sum(row["sampling_reason"] == category for row in selected)
        for category in tuple(review.REASONS)[:4]
    } == {category: 6 for category in tuple(review.REASONS)[:4]}
    assert all(row["case_id"].startswith("case-mm-") for row in selected)
    candidates[0]["split"] = "test"
    with pytest.raises(ValueError, match="TEST is forbidden"):
        review.select_multimodal_cases(candidates)


def test_insufficient_quota_fills_without_duplicates():
    selected = review.select_multimodal_cases(
        [_candidate(i, "correct_control") for i in range(30)], count=24
    )
    assert len(selected) == 24
    assert sum(row["sampling_reason"] == "correct_control" for row in selected) == 6
    assert sum(row["sampling_reason"] == "additional_case" for row in selected) == 18


def _write_predictions(path, row):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prediction_hash_and_key_reference_alignment_are_required(tmp_path):
    row = {
        "study_id": "111",
        "series_id": "222",
        "level": "L4-L5",
        "split": "validation",
        "label": "1",
        "status": "ok",
        "reason": "",
        "prediction": "1",
        "p_normal_mild": "0.2",
        "p_moderate": "0.6",
        "p_severe": "0.2",
    }
    expected = {("111", "222", "L4-L5"): {**row, "label": 1}}
    path = tmp_path / "predictions.csv"
    digest = _write_predictions(path, row)
    assert (
        review._read_predictions(path, digest, expected)[("111", "222", "L4-L5")]["prediction"] == 1
    )
    with pytest.raises(ValueError, match="checksum"):
        review._read_predictions(path, "0" * 64, expected)
    for changed in ({"label": "2"}, {"study_id": "333"}, {"split": "test"}):
        digest = _write_predictions(path, {**row, **changed})
        with pytest.raises(ValueError):
            review._read_predictions(path, digest, expected)


def test_attention_must_point_to_exact_cached_slice():
    data = SimpleNamespace(
        study_ids=["111"],
        splits=np.asarray(["validation"]),
        paired=torch.tensor([[True, False]]),
        selected_indices=[np.asarray([2, 4, 6])],
        selected_series=[np.asarray(["222", "222", "333"])],
        selected_instances=[np.asarray([3, 5, 7])],
    )
    document = {
        "seeds": list(review.SEEDS),
        "records": [
            {
                "study_id": "111",
                "split": "validation",
                "selected_source_indices": [2, 4, 6],
                "levels": {
                    "L4-L5": {
                        "status": "ok",
                        "weights": [0.2, 0.6, 0.2],
                        "top3": [
                            {
                                "bag_index": 1,
                                "source_index": 4,
                                "series_id": "222",
                                "instance": 5,
                                "weight": 0.6,
                            },
                            {
                                "bag_index": 0,
                                "source_index": 2,
                                "series_id": "222",
                                "instance": 3,
                                "weight": 0.2,
                            },
                            {
                                "bag_index": 2,
                                "source_index": 6,
                                "series_id": "333",
                                "instance": 7,
                                "weight": 0.2,
                            },
                        ],
                    },
                    "L5-S1": {"status": "not_assessed", "weights": [], "top3": []},
                },
            }
        ],
    }
    assert review._validate_attention(document, data)["111"]["split"] == "validation"
    document["records"][0]["levels"]["L4-L5"]["top3"][0]["source_index"] = 5
    with pytest.raises(ValueError, match="different source slice"):
        review._validate_attention(document, data)


def test_public_output_rejected_before_cache_or_mri_reads(tmp_path, monkeypatch):
    def must_not_load(*args):
        raise AssertionError("cache should not be opened")

    monkeypatch.setattr(review, "_load_inputs", must_not_load)
    with pytest.raises(ValueError, match="ignored reports/runs"):
        review.generate_multimodal_review(
            "missing", "missing", "missing", "missing", tmp_path / "docs"
        )
    assert not (tmp_path / "docs").exists()


def test_html_has_no_patient_ids_and_no_network_or_forms():
    item = _candidate(887766554433, "correct_control")
    item["series_id"] = "998877665544"
    cases = review.select_multimodal_cases([item], count=1)
    page = review._html(cases)
    assert item["study_id"] not in page and item["series_id"] not in page
    assert cases[0]["case_id"] in page
    assert "Content-Security-Policy" in page and "form-action 'none'" in page
    assert "https://" not in page and "http://" not in page and "<form" not in page
    assert "не репрезентативная выборка" in page


def test_rendering_uses_opaque_ids_and_preserves_failures(tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    def fail(*args):
        raise ValueError("must not serialize patient name or UID: SECRET")

    item = review.select_multimodal_cases([_candidate(123456789, "correct_control")], 1)[0]
    (tmp_path / "images").mkdir()
    monkeypatch.setattr(review, "_roi_image", fail)
    monkeypatch.setattr(review, "_axial_images", fail)
    result = review._render_case(item, None, None, None, None, None, tmp_path, plt)
    assert result == {
        "sagittal": {"status": "failed", "reason": "ValueError"},
        "axial": {"status": "failed", "reason": "ValueError"},
    }
    image = tmp_path / "images" / (item["case_id"] + ".png")
    assert image.stat().st_size > 1000
    assert b"SECRET" not in image.read_bytes()
