from copy import deepcopy

import pytest

from ai_spine_med.uncertainty import bootstrap_v1_test, paired_study_bootstrap


def rows():
    return [
        {
            "study_id": str(i),
            "series_id": str(i),
            "level": level,
            "label": i % 3,
            "prediction": i % 3,
            "status": "ok",
        }
        for i in range(12)
        for level in ("L4-L5", "L5-S1")
    ]


def test_paired_identical_models_have_zero_difference_and_reproduce():
    predictions = {"a": rows(), "b": deepcopy(rows())}
    kwargs = dict(contrasts={"a_minus_b": (["a"], ["b"])}, resamples=100, seed=5)
    result = paired_study_bootstrap(predictions, **kwargs)
    assert result == paired_study_bootstrap(predictions, **kwargs)
    for metric in result["contrasts"]["a_minus_b"]["metrics"].values():
        assert metric["point"] == metric["lower_95"] == metric["upper_95"] == 0
    assert result["n_studies"] == 12
    assert result["n_targets"] == 24


def test_study_cluster_duplication_preserves_bootstrap_distribution():
    single = rows()[::2]
    both = rows()
    for rr in (single, both):
        for row in rr:
            if row["study_id"] in {"2", "5"}:
                row["prediction"] = 0
    a = paired_study_bootstrap({"m": single}, resamples=100, seed=7)
    b = paired_study_bootstrap({"m": both}, resamples=100, seed=7)
    assert a["models"] == b["models"]


def test_unassessed_severe_remains_in_detection_denominator():
    rr = rows()
    severe = next(row for row in rr if row["label"] == 2)
    severe.update(status="not_assessed", prediction=None)
    result = paired_study_bootstrap({"m": rr}, resamples=100)
    assert result["n_severe_all_requested"] == 8
    assert result["models"]["m"]["severe_rate"]["point"] == 7 / 8
    assert result["n_assessed"] == 23


@pytest.mark.parametrize("field,value", [("label", 1), ("status", "not_assessed")])
def test_models_must_have_same_cohort(field, value):
    a, b = rows(), rows()
    b[0][field] = value
    with pytest.raises(ValueError, match="different labels or assessed"):
        paired_study_bootstrap({"a": a, "b": b}, resamples=100)


def test_all_unassessed_has_undefined_classification_but_zero_severe_detection():
    rr = rows()
    for row in rr:
        row.update(status="not_assessed", prediction=None)
    result = paired_study_bootstrap({"m": rr}, resamples=100)
    assert result["models"]["m"]["macro_f1"]["point"] is None
    assert result["models"]["m"]["macro_f1"]["valid_resamples"] == 0
    assert result["models"]["m"]["severe_rate"]["point"] == 0


def test_changed_test_summary_is_rejected_before_csv_access(tmp_path):
    (tmp_path / "summary.json").write_text('{"scope":"test","selection_sha256":"fixed"}')
    (tmp_path / "manifest.json").write_text(
        '{"scope":"heldout_test","selection_sha256":"fixed","summary_sha256":"wrong"}'
    )
    with pytest.raises(ValueError, match="manifest disagree"):
        bootstrap_v1_test(tmp_path, tmp_path / "output", resamples=100)
    assert not (tmp_path / "output").exists()
