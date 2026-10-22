"""Paired TEST-only reporting, coverage, L5 safeguards and private aggregate output."""

import copy
import csv
import json

import matplotlib.image as mpimg
import pytest

from ai_spine_med import localization_test_report as report
from ai_spine_med.assets import sha256_file


def row(patient, level, label, category, *, failed=False):
    return {
        "dataset": "rsna",
        "study_id": f"PRIVATE_{patient}",
        "level": level,
        "split": "test",
        "label": label,
        "status": "not_assessed" if failed else "ok",
        "reason": "PRIVATE /Users/PRIVATE/image.dcm" if failed else "",
        "prediction": None if failed else category,
        **{
            f"p_{name}": None if failed else 0.8 if index == category else 0.1
            for index, name in enumerate(report.CLASSES)
        },
    }


@pytest.fixture
def comparison():
    tables = {name: [] for name in report.MODELS}
    for patient, label in enumerate((0, 0, 1, 1, 2, 2)):
        for level in report.LEVELS:
            for name in report.MODELS:
                failed = name == "sagittal" and level == "L5-S1" and patient in (0, 2)
                category = 0 if name == "xml_only" else label
                if name == "mixed" and label == 1:
                    category = 2  # New severe finds must not hide false severe calls.
                tables[name].append(row(patient, level, label, category, failed=failed))
    summary = {
        "phase": "test",
        "scope": "historically_reused_test_exploratory",
        "severity_seeds": list(report.SEEDS),
        "fixed_before_test": True,
        "no_test_selection": True,
        "labels_joined_after_image_inference": True,
        "bootstrap_resamples": 100,
        "bootstrap_seed": report.BOOTSTRAP_SEED,
        "private_comment": "PRIVATE /Users/PRIVATE/path SECRET",
    }
    return summary, tables


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_pairs_actual_patients_levels_preserves_failures_and_complete_cases(comparison):
    summary, tables = comparison
    public = report.public_snapshot(summary, tables, resamples=100)
    all_group = public["groups"]["rsna/all"]
    assert all_group["n_requested"] == 12
    assert all_group["n_requested_patients"] == 6
    assert all_group["n_common_labeled_assessed"] == 10
    assert all_group["n_common_patients"] == 6
    assert all_group["n_excluded_from_common"] == 2
    assert all_group["models_full_cohort"]["sagittal"]["n_not_assessed"] == 2
    assert all_group["models_full_cohort"]["xml_only"]["n_not_assessed"] == 0
    assert public["groups"]["rsna/L4-L5"]["n_common_patients"] == 6
    l5 = public["groups"]["rsna/L5-S1"]
    assert l5["n_common_patients"] == 4
    assert l5["models_common_cohort"]["xml_only"]["class_counts"] == {
        "normal_mild": 1,
        "moderate": 1,
        "severe": 2,
    }
    complete = public["complete_case_groups"]["rsna/all"]
    assert complete["n_common_patients"] == 4
    assert complete["n_common_labeled_assessed"] == 8
    assert complete["n_requested"] == 8
    assert complete["bootstrap"]["n_studies"] == 4


def test_bootstrap_is_paired_by_patient_with_delta_orientation(comparison):
    summary, tables = comparison
    group = report.public_snapshot(summary, tables, resamples=100)["groups"]["rsna/all"]
    bootstrap = group["bootstrap"]
    assert (bootstrap["n_studies"], bootstrap["n_targets"], bootstrap["seed"]) == (6, 10, 103017)
    assert set(bootstrap["models"]) == set(report.MODELS)
    assert set(bootstrap["contrasts"]) == set(report.CONTRASTS)
    for name, (left, right) in report.CONTRASTS.items():
        actual = bootstrap["contrasts"][name]
        metrics = group["models_common_cohort"]
        delta = (
            metrics[left[0]]["conditional"]["macro_f1"]
            - metrics[right[0]]["conditional"]["macro_f1"]
        )
        assert actual["metrics"]["macro_f1"]["point"] == pytest.approx(delta)
        assert actual["left"] == left and actual["right"] == right


def test_l5_severe_found_and_false_severe_counts_are_separate(comparison):
    summary, tables = comparison
    group = report.public_snapshot(summary, tables, resamples=100)["groups"]["rsna/L5-S1"]
    xml, mixed = (group["models_common_cohort"][name] for name in ("xml_only", "mixed"))
    assert xml["severe"]["correctly_predicted_severe"] == 0
    assert mixed["severe"]["correctly_predicted_severe"] == 2
    assert mixed["moderate_found"] == 0
    assert mixed["false_severe"] == 1
    changes = group["paired_changes"]["mixed_minus_xml_only"]
    assert changes["severe_improved"] == 2
    assert changes["severe_both_missed"] == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("phase", "validation"),
        ("scope", "independent_test"),
        ("severity_seeds", [3407]),
        ("fixed_before_test", False),
        ("no_test_selection", False),
        ("labels_joined_after_image_inference", 1),
    ],
)
def test_rejects_test_tuning_or_unfrozen_protocol(comparison, field, value):
    summary, tables = comparison
    summary[field] = value
    with pytest.raises(ValueError, match="sealed, inference-only"):
        report.public_snapshot(summary, tables, resamples=100)


@pytest.mark.parametrize(
    "mutation", ("patient", "grade", "split", "dataset", "duplicate", "missing")
)
def test_equal_number_of_targets_does_not_substitute_for_actual_pairing(comparison, mutation):
    summary, tables = comparison
    changed = copy.deepcopy(tables)
    target = changed["mixed"][0]
    if mutation == "patient":
        target["study_id"] = "PRIVATE_other_patient"
    elif mutation == "grade":
        target["label"] = 2
    elif mutation == "split":
        target["split"] = "validation"
    elif mutation == "dataset":
        target["dataset"] = "aisslab"
    elif mutation == "duplicate":
        changed["mixed"].append(dict(target))
    else:
        changed["mixed"].pop()
    with pytest.raises(ValueError, match="cohorts|matching RSNA"):
        report.public_snapshot(summary, changed, resamples=100)


def test_missing_labels_not_imputed_or_confused_with_failed_predictions(comparison):
    summary, tables = comparison
    for name in report.MODELS:
        tables[name][-1]["label"] = None
    public = report.public_snapshot(summary, tables, resamples=100)
    full = public["groups"]["rsna/all"]["models_full_cohort"]["xml_only"]
    assert full["n_missing_reference_label"] == 1
    assert full["n_not_assessed"] == 0
    assert full["n_predictions_including_missing_reference"] == 12
    assert public["groups"]["rsna/all"]["n_common_labeled_assessed"] == 9


def test_public_output_is_private_aggregate_only_and_explains_scope(comparison, tmp_path):
    summary, tables = comparison
    public = report.public_snapshot(summary, tables, resamples=100)
    encoded = json.dumps(public)
    assert "PRIVATE" not in encoded and "SECRET" not in encoded and "/Users/" not in encoded
    text = report.render_markdown(
        public, comparison_image="comparison.png", confusion_image="l5.png"
    )
    assert "не является новым независимым подтверждением" in text
    assert "не выбирал" in text and "TRAIN" in text and "VAL" in text
    assert "Тяжёлый → норма" in text and "Ложный тяжёлый" in text
    assert "обработаны оба уровня" in text
    assert "только конкретный необработанный уровень" in text
    for function, filename in (
        (report.draw_comparison, "comparison.png"),
        (report.draw_confusion, "l5.png"),
    ):
        path = tmp_path / filename
        function(public, path)
        image = mpimg.imread(path)
        assert image.shape[0] > 500 and image.shape[1] > 1000


def sealed_summary(tmp_path, summary, tables):
    summary["models"] = {}
    for name in report.MODELS:
        ensemble = tmp_path / name / "ensemble.csv"
        write_csv(ensemble, tables[name])
        seed_paths = {}
        for seed in report.SEEDS:
            path = tmp_path / name / f"seed_{seed}.csv"
            write_csv(path, list(reversed(tables[name])))
            seed_paths[str(seed)] = str(path.relative_to(tmp_path))
        summary["models"][name] = {
            "predictions": str(ensemble.relative_to(tmp_path)),
            "predictions_sha256": sha256_file(ensemble),
            "seed_predictions": seed_paths,
        }
    source = tmp_path / "summary.json"
    source.write_text(json.dumps(summary))
    return source


def report_paths(tmp_path):
    return dict(
        out_json=tmp_path / "public.json",
        out_markdown=tmp_path / "report.md",
        comparison_image=tmp_path / "comparison.png",
        confusion_image=tmp_path / "l5.png",
        resamples=100,
    )


def test_loader_verifies_all_seeds_equal_mean_and_sealed_ensemble(comparison, tmp_path):
    summary, tables = comparison
    source = sealed_summary(tmp_path, summary, tables)
    public = report.report_localization_test(source, **report_paths(tmp_path))
    assert set(public["private_input_sha256"]) == set(report.MODELS)
    seed_metrics = public["groups"]["rsna/all"]["individual_seed_models_full_cohort"]
    assert all(
        set(seed_metrics[name]) == {str(seed) for seed in report.SEEDS} for name in report.MODELS
    )
    assert "PRIVATE" not in (tmp_path / "public.json").read_text()
    path = tmp_path / "mixed/ensemble.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed after inference"):
        report.report_localization_test(source, **report_paths(tmp_path))


def test_loader_cannot_publish_chosen_seed_as_ensemble(comparison, tmp_path):
    summary, tables = comparison
    source = sealed_summary(tmp_path, summary, tables)
    changed = copy.deepcopy(tables["mixed"])
    changed[0] = row(0, "L4-L5", 0, 2)
    write_csv(tmp_path / "mixed/seed_42.csv", changed)
    with pytest.raises(ValueError, match="equal seed probability mean"):
        report.report_localization_test(source, **report_paths(tmp_path))


def test_loader_rejects_incomplete_seed_set(comparison, tmp_path):
    summary, tables = comparison
    source = sealed_summary(tmp_path, summary, tables)
    del summary["models"]["xml_only"]["seed_predictions"]["42"]
    source.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="all three predetermined"):
        report.report_localization_test(source, **report_paths(tmp_path))
