"""Meaningful four-way pairing, severe-error and validation-scope safeguards."""

import copy
import csv
import json
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pytest

from ai_spine_med import localization_finetuning_report as report
from ai_spine_med.assets import sha256_file


def row(dataset, patient, level, label, predicted, *, failed=False):
    return {
        "dataset": dataset,
        "study_id": f"PRIVATE_{dataset}_{patient}",
        "level": level,
        "split": "validation",
        "label": label,
        "status": "not_assessed" if failed else "ok",
        "reason": "PRIVATE /Users/PRIVATE/image.dcm" if failed else "",
        "prediction": None if failed else predicted,
        **{
            f"p_{name}": None if failed else 0.8 if i == predicted else 0.1
            for i, name in enumerate(report.CLASSES)
        },
    }


@pytest.fixture
def comparison():
    tables = {name: [] for name in report.MODELS}
    for dataset in report.DATASETS:
        for patient, label in enumerate((0, 0, 1, 1, 2, 2)):
            for level in report.LEVELS:
                for name in report.MODELS:
                    failed = name.startswith("xml_only") and patient == 4 and level == "L5-S1"
                    target_label = (
                        None
                        if dataset == "aisslab" and patient == 0 and level == "L4-L5"
                        else label
                    )
                    category = 0 if name.endswith("before") else label
                    tables[name].append(
                        row(dataset, patient, level, target_label, category, failed=failed)
                    )
    summary = {
        "phase": "validation",
        "center_reused_exactly": True,
        "severity_retrained": True,
        "test_evaluated": False,
        "localizers_frozen": True,
        "warm_start": "v6_same_seed",
        "normalization": "frozen_v6_train_statistics",
        "localizer_ensemble": "mean_probabilities_before_argmax",
        "severity_training_source": "rsna_train_only",
        "severity_seeds": list(report.SEEDS),
        "bootstrap_resamples": 100,
        "bootstrap_seed": 93017,
        "private_note": "PRIVATE token=SECRET /Users/PRIVATE/case",
    }
    return summary, tables


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def training_receipt(summary):
    summary["training_config"] = {
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "max_epochs": 15,
        "patience": 4,
        "epoch0_eligible": True,
    }
    counts = [6, 4, 2]
    weights = 1 / np.sqrt(counts)
    summary["class_weights"] = (weights / weights.mean()).tolist()
    summary["arms"] = {
        arm: {
            "training": {
                "n_patients": 6,
                "n_targets": 12,
                "n_requested_targets": 14,
                "class_counts": counts,
                "by_level_class_counts": {level: [3, 2, 1] for level in report.LEVELS},
                "selected_epochs": {str(seed): i for i, seed in enumerate(report.SEEDS)},
                "epoch0_scores": {str(seed): 0.4 for seed in report.SEEDS},
                "selected_scores": {
                    str(seed): 0.4 + i * 0.01 for i, seed in enumerate(report.SEEDS)
                },
            }
        }
        for arm in report.ARMS
    }
    return summary


def test_preserves_full_denominators_and_distinguishes_common_cohort(comparison):
    summary, tables = comparison
    public = report.public_snapshot(summary, tables, resamples=100)
    group = public["groups"]["rsna/L5-S1"]
    xml = group["models_full_cohort"]["xml_only_after"]
    assert group["n_requested"] == 6
    assert group["n_common_labeled_assessed"] == 5
    assert xml["severe"]["n_total"] == 2
    assert xml["severe"]["correctly_predicted_severe"] == 1
    assert xml["severe"]["not_assessed"] == 1
    assert xml["severe"]["missed_all_requested"] == 1
    assert group["models_full_cohort"]["mixed_after"]["severe"]["n_total"] == 2
    assert group["models_common_cohort"]["mixed_after"]["severe"]["n_total"] == 1
    assert group["paired_changes"]["xml_only_after_minus_before"]["severe_improved"] == 1
    assert (
        public["groups"]["aisslab/all"]["models_full_cohort"]["mixed_after"][
            "n_missing_reference_label"
        ]
        == 1
    )


def test_paired_patient_bootstrap_orientation_and_all_four_models(comparison):
    summary, tables = comparison
    group = report.public_snapshot(summary, tables, resamples=100)["groups"]["rsna/all"]
    bootstrap = group["bootstrap"]
    assert (bootstrap["n_studies"], bootstrap["n_targets"], bootstrap["seed"]) == (6, 11, 93017)
    assert set(bootstrap["models"]) == set(report.MODELS)
    for arm in report.ARMS:
        contrast = bootstrap["contrasts"][f"{arm}_after_minus_before"]
        metrics = group["models_common_cohort"]
        delta = (
            metrics[f"{arm}_after"]["conditional"]["macro_f1"]
            - metrics[f"{arm}_before"]["conditional"]["macro_f1"]
        )
        assert contrast["metrics"]["macro_f1"]["point"] == pytest.approx(delta)
        assert contrast["left"] == [f"{arm}_after"]
        assert contrast["right"] == [f"{arm}_before"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("phase", "test"),
        ("severity_training_source", "rsna_and_aisslab"),
        ("severity_retrained", False),
        ("localizers_frozen", False),
        ("test_evaluated", True),
        ("warm_start", "random"),
        ("normalization", "refitted_validation"),
        ("center_reused_exactly", 1),
        ("severity_seeds", [42]),
        ("bootstrap_seed", 83017),
        ("localizer_ensemble", "vote"),
    ],
)
def test_rejects_changed_scope_or_unfrozen_components(comparison, field, value):
    summary, tables = comparison
    summary[field] = value
    with pytest.raises(ValueError, match="fixed validation-only"):
        report.public_snapshot(summary, tables, resamples=100)


def test_before_after_must_have_the_same_roi_failure_mask(comparison):
    summary, tables = comparison
    tables["xml_only_after"][0]["status"] = "not_assessed"
    with pytest.raises(ValueError, match="coverage mask"):
        report.public_snapshot(summary, tables, resamples=100)


def test_equal_count_does_not_substitute_for_same_patient_or_grade(comparison):
    summary, tables = comparison
    changed = copy.deepcopy(tables)
    changed["mixed_after"][0]["study_id"] = "PRIVATE_other_patient"
    with pytest.raises(ValueError, match="patient/level"):
        report.public_snapshot(summary, changed, resamples=100)
    changed = copy.deepcopy(tables)
    changed["mixed_after"][0]["label"] = 2
    with pytest.raises(ValueError, match="matching validation grades"):
        report.public_snapshot(summary, changed, resamples=100)


def test_new_severe_find_cannot_hide_lost_moderate_or_false_severe(comparison):
    summary, tables = comparison
    # One moderate is pushed into severe, one true severe is found.
    for name in ("mixed_before", "mixed_after"):
        for item in tables[name]:
            if item["dataset"] == "rsna" and item["level"] == "L5-S1" and item["label"] == 1:
                item["prediction"] = 1 if name.endswith("before") else 2
    group = report.public_snapshot(summary, tables, resamples=100)["groups"]["rsna/L5-S1"]
    before, after = [group["models_full_cohort"][f"mixed_{stage}"] for stage in report.STAGES]
    assert (before["moderate_found"], after["moderate_found"]) == (2, 0)
    assert (before["false_severe"], after["false_severe"]) == (0, 2)
    assert group["paired_changes"]["mixed_after_minus_before"]["worsened"] == 2


def test_public_outputs_have_only_aggregates_and_clear_selection_bias(comparison, tmp_path):
    summary, tables = comparison
    public = report.public_snapshot(summary, tables, resamples=100)
    encoded = json.dumps(public)
    assert "PRIVATE" not in encoded and "SECRET" not in encoded and "/Users/" not in encoded
    text = report.render_markdown(
        public, comparison_image="comparison.png", confusion_image="l5.png"
    )
    assert "эпоху 0" in text
    assert "повторно используемой валидации" in text
    assert "не исправляют смещение" in text
    assert "Тяжёлый → норма" in text and "Ложный тяжёлый" in text
    assert "TEST в этом эксперименте не оценивал" in text
    for function, name in (
        (report.draw_comparison, "comparison.png"),
        (report.draw_confusion, "confusion.png"),
    ):
        path = tmp_path / name
        function(public, path)
        image = mpimg.imread(path)
        assert image.shape[0] > 500 and image.shape[1] > 1000


def test_actual_report_checks_all_three_seeds_and_sealed_csvs(comparison, tmp_path):
    summary, tables = comparison
    training_receipt(summary)
    for arm in report.ARMS:
        for stage, field in (("before", "epoch0_scores"), ("after", "selected_scores")):
            value = report._metrics(
                [row for row in tables[f"{arm}_{stage}"] if row["dataset"] == "rsna"]
            )
            summary["arms"][arm]["training"][field] = {
                str(seed): value["conditional"]["macro_f1"] for seed in report.SEEDS
            }
        summary["arms"][arm]["training"]["selected_epochs"] = {
            str(seed): 1 for seed in report.SEEDS
        }
    for arm in report.ARMS:
        for stage in report.STAGES:
            name = f"{arm}_{stage}"
            base = tmp_path / name
            path = base / "ensemble.csv"
            write_csv(path, tables[name])
            seeds = {}
            for seed in report.SEEDS:
                seed_path = base / f"seed_{seed}.csv"
                write_csv(seed_path, list(reversed(tables[name])))
                seeds[str(seed)] = str(seed_path)
            summary["arms"][arm][stage] = {
                "predictions": str(path),
                "predictions_sha256": sha256_file(path),
                "seed_predictions": seeds,
            }
    source = tmp_path / "summary.json"
    source.write_text(json.dumps(summary))
    args = dict(
        out_json=tmp_path / "results.json",
        out_markdown=tmp_path / "report.md",
        comparison_image=tmp_path / "comparison.png",
        confusion_image=tmp_path / "l5.png",
        resamples=100,
    )
    report.report_localization_finetuning(source, **args)
    assert "PRIVATE" not in (tmp_path / "results.json").read_text()
    changed = copy.deepcopy(summary)
    changed["arms"]["mixed"]["training"]["selected_scores"]["42"] -= 0.01
    source.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="scores disagree"):
        report.report_localization_finetuning(source, **args)
    changed = copy.deepcopy(summary)
    changed["arms"]["mixed"]["training"]["selected_epochs"]["42"] = 0
    source.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Epoch-zero selection changed"):
        report.report_localization_finetuning(source, **args)
    source.write_text(json.dumps(summary))
    path = Path(summary["arms"]["mixed"]["after"]["predictions"])
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed after"):
        report.report_localization_finetuning(source, **args)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s["training_config"].update(learning_rate=0.001),
        lambda s: s["arms"]["mixed"]["training"].update(n_targets=13),
        lambda s: s["arms"]["mixed"]["training"]["selected_epochs"].update({"42": 16}),
        lambda s: s["arms"]["mixed"]["training"]["selected_scores"].update({"42": 0.3}),
        lambda s: s.update(class_weights=[1, 1, 1]),
    ],
)
def test_train_metadata_cannot_claim_a_different_protocol_or_hidden_cohort(comparison, mutation):
    summary, _ = comparison
    training_receipt(summary)
    report.public_training(summary)
    mutation(summary)
    with pytest.raises(ValueError):
        report.public_training(summary)
