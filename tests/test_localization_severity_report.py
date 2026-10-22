"""Meaningful safeguards for paired VAL comparisons and public aggregates."""

import copy
import csv
import json

import matplotlib.image as mpimg
import numpy as np
import pytest

from ai_spine_med import localization_severity_report as report


def prediction(dataset, patient, level, label, category, *, failed=False):
    return {
        "dataset": dataset,
        "study_id": f"PRIVATE_{dataset}_{patient}",
        "split": "validation",
        "level": level,
        "label": label,
        "status": "not_assessed" if failed else "ok",
        "reason": "PRIVATE source unavailable" if failed else "",
        "prediction": None if failed else category,
        **{
            f"p_{name}": None if failed else 0.8 if index == category else 0.1
            for index, name in enumerate(report.CLASSES)
        },
    }


@pytest.fixture
def comparison():
    tables = {arm: [] for arm in report.ARMS}
    for dataset in report.DATASETS:
        for patient, label in enumerate((0, 0, 1, 1, 2, 2)):
            for level in report.LEVELS:
                known = not (dataset == "aisslab" and patient == 0 and level == "L4-L5")
                for arm in report.ARMS:
                    failed = arm == "xml_only" and patient == 4 and level == "L5-S1"
                    tables[arm].append(
                        prediction(
                            dataset,
                            patient,
                            level,
                            label if known else None,
                            0 if arm == "xml_only" else label,
                            failed=failed,
                        )
                    )
    summary = {
        "phase": "validation",
        "center_reused_exactly": True,
        "severity_retrained": False,
        "test_evaluated": False,
        "localizer_ensemble": "mean_probabilities_before_argmax",
        "grade_labels_joined_after_both_arms_image_inference": True,
        "severity_seeds": list(report.SEEDS),
        "bootstrap_resamples": 100,
        "bootstrap_seed": 83017,
        "private_note": "PRIVATE /Users/PRIVATE/image.dcm?token=SECRET",
    }
    return summary, tables


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_full_coverage_and_rare_severe_misses_are_not_hidden_by_common_subset(comparison):
    summary, tables = comparison
    public = report.public_snapshot(summary, tables, resamples=100)
    group = public["groups"]["rsna/L5-S1"]
    xml = group["models_full_cohort"]["xml_only"]
    mixed = group["models_full_cohort"]["mixed"]
    assert (group["n_requested"], group["n_common_labeled_assessed"]) == (6, 5)
    assert xml["severe"]["n_total"] == mixed["severe"]["n_total"] == 2
    assert xml["severe"]["predicted_normal_mild"] == 1
    assert xml["severe"]["not_assessed"] == 1
    assert xml["severe"]["missed_all_requested"] == 2
    assert mixed["severe"]["correctly_predicted_severe"] == 2
    assert group["paired_changes"]["severe_mixed_only_detected"] == 1
    assert group["models_common_cohort"]["mixed"]["severe"]["n_total"] == 1
    assert (
        public["groups"]["aisslab/all"]["models_full_cohort"]["mixed"]["n_missing_reference_label"]
        == 1
    )
    assert (
        public["groups"]["aisslab/all"]["models_full_cohort"]["mixed"][
            "n_predictions_including_missing_reference"
        ]
        == 12
    )


def test_paired_difference_has_protocol_orientation_and_patient_units(comparison):
    summary, tables = comparison
    public = report.public_snapshot(summary, tables, resamples=100)
    group = public["groups"]["rsna/all"]
    boot = group["bootstrap"]
    delta = boot["contrasts"]["mixed_minus_xml_only"]
    assert delta["left"] == ["mixed"]
    assert delta["right"] == ["xml_only"]
    assert delta["metrics"]["macro_f1"]["point"] > 0
    a, b = [group["models_common_cohort"][arm]["conditional"]["macro_f1"] for arm in report.ARMS]
    assert delta["metrics"]["macro_f1"]["point"] == pytest.approx(b - a)
    assert (boot["n_studies"], boot["n_targets"], boot["seed"], boot["resamples"]) == (
        6,
        11,
        83017,
        100,
    )


def test_equal_counts_cannot_substitute_for_same_patient_cohort(comparison):
    summary, tables = comparison
    tables["mixed"][0]["study_id"] = "PRIVATE_different_patient"
    with pytest.raises(ValueError, match="cohorts"):
        report.public_snapshot(summary, tables, resamples=100)


def test_reference_gap_is_not_imputed_as_normal(comparison):
    summary, tables = comparison
    tables["mixed"][12]["label"] = 0
    with pytest.raises(ValueError, match="missing-reference masks"):
        report.public_snapshot(summary, tables, resamples=100)


@pytest.mark.parametrize(
    "field,value",
    [
        ("phase", "test"),
        ("severity_retrained", True),
        ("center_reused_exactly", False),
        ("test_evaluated", True),
        ("severity_seeds", [42]),
        ("bootstrap_seed", 2026),
        ("bootstrap_resamples", 2000),
        ("localizer_ensemble", "majority_vote"),
        ("grade_labels_joined_after_both_arms_image_inference", False),
        ("severity_retrained", 0),
    ],
)
def test_scope_and_frozen_protocol_are_required(comparison, field, value):
    summary, tables = comparison
    summary[field] = value
    with pytest.raises(ValueError, match="fixed VAL-only"):
        report.public_snapshot(summary, tables, resamples=100)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda row: row.update(split="test"),
        lambda row: row.update(level="L3-L4"),
        lambda row: row.update(p_normal_mild=float("nan")),
        lambda row: row.update(p_normal_mild=1.2),
        lambda row: row.update(p_normal_mild=0.3),
        lambda row: row.update(prediction=2),
        lambda row: row.update(status="not_assessed", reason="failed"),
    ],
)
def test_reader_rejects_invalid_probabilities_nonval_and_imputed_failures(
    comparison, tmp_path, mutate
):
    _, tables = comparison
    mutate(tables["xml_only"][0])
    path = tmp_path / "predictions.csv"
    write_csv(path, tables["xml_only"])
    with pytest.raises(ValueError):
        report.read_validation_predictions(path)


def test_probability_ensemble_is_not_category_voting_and_order_is_irrelevant(comparison):
    _, tables = comparison
    base = [r for r in tables["mixed"] if r["dataset"] == "rsna"]
    ensemble = copy.deepcopy(base)
    seeds = {seed: copy.deepcopy(base) for seed in report.SEEDS}
    probabilities = ([0.49, 0.48, 0.03], [0.49, 0.48, 0.03], [0.01, 0.98, 0.01])
    for seed, p in zip(report.SEEDS, probabilities):
        seeds[seed][0].update(prediction=int(np.argmax(p)))
        seeds[seed][0].update({f"p_{c}": v for c, v in zip(report.CLASSES, p)})
    mean = np.mean(probabilities, axis=0)
    ensemble[0].update(prediction=1, **{f"p_{c}": v for c, v in zip(report.CLASSES, mean)})
    seeds[3407].reverse()
    report.verify_ensemble(ensemble, seeds)
    ensemble[0].update(prediction=0)
    with pytest.raises(ValueError, match="probability mean"):
        report.verify_ensemble(ensemble, seeds)


def test_report_recomputes_public_aggregates_and_excludes_arbitrary_private_metadata(
    comparison, tmp_path
):
    summary, tables = comparison
    summary["arms"] = {}
    for arm in report.ARMS:
        path = tmp_path / arm / "ensemble.csv"
        write_csv(path, tables[arm])
        seeds = {}
        for seed in report.SEEDS:
            p = path.parent / f"seed_{seed}.csv"
            write_csv(p, list(reversed(tables[arm])))
            seeds[str(seed)] = str(p.relative_to(tmp_path))
        summary["arms"][arm] = {
            "predictions": str(path.relative_to(tmp_path)),
            "seed_predictions": seeds,
            "metrics": {"macro_f1": 1.0, "private_id": "PRIVATE"},
        }
    source = tmp_path / "summary.json"
    source.write_text(json.dumps(summary))
    out_json, out_md = tmp_path / "public.json", tmp_path / "public.md"
    image, confusion = tmp_path / "comparison.png", tmp_path / "confusion.png"
    public = report.report_localization_severity(
        source,
        out_json=out_json,
        out_markdown=out_md,
        comparison_image=image,
        confusion_image=confusion,
        resamples=100,
    )
    text = out_json.read_text() + out_md.read_text()
    for secret in ("PRIVATE", "/Users/", "SECRET", "study_id", "private_id"):
        assert secret not in text
    assert (
        public["groups"]["rsna/all"]["models_full_cohort"]["xml_only"]["conditional"]["macro_f1"]
        < 0.3
    )
    assert "повторно используемая валидация" in out_md.read_text().lower()
    assert "классификаторы v6 обучались на roi прежнего локализатора" in out_md.read_text().lower()
    assert "Смешанная − XML" in out_md.read_text()
    assert "100 парными" in out_md.read_text()
    assert mpimg.imread(image).shape[:2] == (736, 2000)
    assert mpimg.imread(confusion).shape[:2] == (1280, 1600)
