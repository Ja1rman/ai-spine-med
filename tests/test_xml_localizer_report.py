"""Public report must retain reference denominators and exclude private identities."""

import copy
import json

import matplotlib.image as mpimg
import pytest

from ai_spine_med import xml_localizer_report as report
from ai_spine_med.xml_localizer_ablation import aggregate_cases


@pytest.fixture
def comparison():
    cases = []
    for dataset in report.DATASETS:
        for patient in range(2):
            for level in report.LEVELS:
                known = not (dataset == "aisslab" and patient == 0 and level == "L4-L5")
                distance = float(4 + 8 * patient) if known else None
                cases.append(
                    {
                        "dataset": dataset,
                        "study_id": f"PRIVATE_{dataset}_{patient}",
                        "split": "validation",
                        "level": level,
                        "reference_present": known,
                        "predicted": True,
                        "selected_index": 1,
                        "distance_mm": distance,
                        "hit5": known and distance <= 5,
                        "hit10": known and distance <= 10,
                        "private_path": "/Users/PRIVATE_MRI.dcm",
                    }
                )
    snapshot = aggregate_cases(cases)
    summary = {
        "phase": "validation",
        "center_reused_exactly": True,
        "original_validation_failures": 3,
        "arms": {},
        "private_url": "https://PRIVATE.invalid/?token=SECRET",
    }
    private = {"seeds": {}, "ensembles": {}}
    for arm in report.ARMS:
        datasets = ["aisslab"] if arm == "xml_only" else ["aisslab", "rsna"]
        runs = []
        for seed in report.SEEDS:
            runs.append(
                {
                    "seed": seed,
                    "selected_epoch": 2,
                    "fit_counts": {dataset: 2 for dataset in datasets},
                    "validation_levels": copy.deepcopy(snapshot),
                    "positional_baseline": copy.deepcopy(snapshot),
                    "study_id": "PRIVATE_SELECTED_ID",
                }
            )
            private["seeds"][f"{arm}/seed_{seed}"] = copy.deepcopy(cases)
        summary["arms"][arm] = {
            "training_datasets": datasets,
            "seeds": runs,
            "ensemble": copy.deepcopy(snapshot),
            "private_weights": "/Users/PRIVATE_WEIGHTS.pt",
        }
        private["ensembles"][arm] = copy.deepcopy(cases)
    summary["paired_ensemble_comparison"] = {
        "unit": "PRIVATE arbitrary source field",
        "resamples": 20,
        "groups": {
            group: {
                "n_patients": snapshot["groups"][group]["n"],
                "delta_hit10": 0.0,
                "ci95": [0.0, 0.0],
            }
            for group in report.GROUPS
        },
    }
    return summary, private


def test_public_report_excludes_arbitrary_metadata_and_keeps_missing_reference(
    comparison, tmp_path
):
    summary, private = comparison
    source = tmp_path / "private-run"
    source.mkdir()
    (source / "summary.json").write_text(json.dumps(summary))
    (source / "cases-private.json").write_text(json.dumps(private))
    output_json, output_md, image = (
        tmp_path / "public.json",
        tmp_path / "report.md",
        tmp_path / "image.png",
    )
    public = report.report_xml_localizer(
        source / "summary.json", out_json=output_json, out_markdown=output_md, out_image=image
    )
    serialized = output_json.read_text() + output_md.read_text()
    for secret in ("PRIVATE", "/Users/", "SECRET", "study_id", "private_path", "private_url"):
        assert secret not in serialized
    group = public["arms"]["xml_only"]["ensemble"]["groups"]["aisslab/L4-L5"]
    assert (group["requested"], group["predicted"], group["n"], group["missing_reference"]) == (
        2,
        2,
        1,
        1,
    )
    assert group["hit10_count"] == 0
    assert group["hit10"] == 0
    assert public["test_evaluated"] is False
    assert public["severity_evaluated"] is False
    assert public["center_reused_exactly"] is True
    assert "не маска сегментации" in serialized
    assert "не выравнивал" in serialized
    assert mpimg.imread(image).shape[:2] == (736, 1728)


def test_equal_counts_do_not_substitute_for_equal_patient_cohorts(comparison):
    summary, private = comparison
    private["seeds"]["mixed/seed_42"][0]["study_id"] = "DIFFERENT_PRIVATE_PATIENT"
    with pytest.raises(ValueError, match="cohorts"):
        report.public_snapshot(summary, private)


def test_changed_reference_mask_is_rejected_even_on_same_patients(comparison):
    summary, private = comparison
    row = private["seeds"]["mixed/seed_42"][0]
    row.update(reference_present=True, distance_mm=20.0)
    summary["arms"]["mixed"]["seeds"][0]["validation_levels"] = aggregate_cases(
        private["seeds"]["mixed/seed_42"]
    )
    with pytest.raises(ValueError, match="reference masks"):
        report.public_snapshot(summary, private)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.update(n=2),
        lambda row: row.update(hit10=0.7),
        lambda row: row.update(hit10_count=1),
        lambda row: row.update(missing_reference=0),
        lambda row: row.update(predicted=3),
        lambda row: row.update(unverifiable_top_series=3),
    ],
)
def test_invalid_reference_denominators_and_hit_counts_are_rejected(comparison, mutation):
    summary, _ = comparison
    value = summary["arms"]["xml_only"]["seeds"][0]["validation_levels"]
    mutation(value["groups"]["aisslab/L4-L5"])
    with pytest.raises(ValueError):
        report.level_snapshot(value)


def test_unverifiable_source_series_cannot_be_counted_as_a_hit(comparison):
    summary, _ = comparison
    value = summary["arms"]["xml_only"]["seeds"][0]["validation_levels"]
    value["groups"]["rsna/L4-L5"]["unverifiable_top_series"] = 2
    with pytest.raises(ValueError, match="unverifiable"):
        report.level_snapshot(value)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda source: source.update(phase="test"),
        lambda source: source.update(center_reused_exactly=False),
        lambda source: source["arms"]["xml_only"]["seeds"].pop(),
        lambda source: source["arms"]["xml_only"].update(training_datasets=["rsna"]),
    ],
)
def test_fixed_scope_and_all_predetermined_seeds_are_required(comparison, mutation):
    summary, private = comparison
    mutation(summary)
    with pytest.raises(ValueError):
        report.public_snapshot(summary, private)


def test_private_test_row_is_rejected_independently_of_summary_scope(comparison):
    summary, private = comparison
    private["seeds"]["xml_only/seed_42"][0]["split"] = "test"
    with pytest.raises(ValueError, match="validation identity"):
        report.public_snapshot(summary, private)


def test_bootstrap_uses_reference_patients_and_matches_reported_ensemble(comparison):
    summary, private = comparison
    summary["paired_ensemble_comparison"]["groups"]["aisslab/L4-L5"]["n_patients"] = 2
    with pytest.raises(ValueError, match="bootstrap"):
        report.public_snapshot(summary, private)
    summary["paired_ensemble_comparison"]["groups"]["aisslab/L4-L5"].update(
        n_patients=1, delta_hit10=0.5
    )
    with pytest.raises(ValueError, match="difference"):
        report.public_snapshot(summary, private)


def test_summary_must_be_recomputed_from_actual_private_predictions(comparison):
    summary, private = comparison
    row = summary["arms"]["xml_only"]["seeds"][0]["validation_levels"]["groups"]["rsna/L4-L5"]
    row.update(hit10_count=2, hit10=1.0)
    with pytest.raises(ValueError, match="private predictions"):
        report.public_snapshot(summary, private)
