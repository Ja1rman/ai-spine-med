import copy
import json

import pytest

from ai_spine_med.comparison_report import draw_final_comparison
from ai_spine_med.training import evaluate_predictions


@pytest.fixture
def comparison_inputs(tmp_path):
    directories = [tmp_path / name for name in ("head", "fine", "test")]
    for directory in directories:
        directory.mkdir()
    metrics = evaluate_predictions(
        [
            {"level": "L4-L5", "label": 0, "status": "ok", "prediction": 0},
            {"level": "L4-L5", "label": 1, "status": "ok", "prediction": 1},
            {"level": "L5-S1", "label": 2, "status": "ok", "prediction": 1},
            {"level": "L5-S1", "label": 2, "status": "not_assessed", "prediction": None},
        ]
    )
    metrics["scope"] = "test"
    runs = [
        {
            "seed": seed,
            "checkpoint_sha256": str(seed),
            "best_epoch": 0,
            "epochs_run": 1,
            "validation_macro_f1": 0.6,
            "metrics": metrics,
        }
        for seed in (42, 2024, 3407)
    ]
    for directory in directories[:2]:
        (directory / "summary.json").write_text(
            json.dumps(
                {
                    "runs": runs,
                    "validation_macro_f1_mean": 0.6,
                    "validation_macro_f1_std": 0.0,
                }
            )
        )
    for run in runs:
        run_dir = directories[1] / "runs" / f"seed-{run['seed']}"
        run_dir.mkdir(parents=True)
        (run_dir / "history.json").write_text(
            json.dumps(
                [
                    {"epoch": 0, "training_loss": None, "validation_macro_f1": 0.6},
                    {"epoch": 1, "training_loss": 0.5, "validation_macro_f1": 0.59},
                ]
            )
        )
    test = {
        "scope": "test",
        "selection_sha256": "fixed",
        "entries": [
            {
                "variant": "original",
                "seed": None,
                "metrics": metrics,
                "checkpoint_sha256": "original",
            }
        ]
        + [
            {
                "variant": variant,
                "seed": run["seed"],
                "metrics": metrics,
                "checkpoint_sha256": run["checkpoint_sha256"],
            }
            for variant in ("frozen_head", "fine_tuned")
            for run in runs
        ],
        "coverage": {"n_requested": 4, "n_labeled": 4, "n_assessed": 3, "n_not_assessed": 1},
        "variants": {},
        "timings": {},
    }
    (directories[2] / "summary.json").write_text(json.dumps(test))
    return directories, test


def test_final_report_keeps_all_seeds_and_abstained_severe_denominator(comparison_inputs, tmp_path):
    directories, _ = comparison_inputs
    readme = tmp_path / "README.md"
    readme.write_text(
        "Before\n<!-- FINAL_RESULTS_START -->\nPending\n<!-- FINAL_RESULTS_END -->\nAfter"
    )
    out = tmp_path / "docs" / "images"
    public = draw_final_comparison(*directories, out, readme=readme)
    assert len(public["entries"]) == 7
    assert public["entries"][0]["metrics"]["severe"]["n_total"] == 2
    assert public["entries"][0]["metrics"]["severe"]["not_assessed"] == 1
    assert len(list(out.glob("*.png"))) == 5
    assert (out.parent / "test-results.json").is_file()
    assert "0 / 2" in readme.read_text()
    assert "Before" in readme.read_text() and "After" in readme.read_text()


@pytest.mark.parametrize(
    "corruption", ["validation", "checkpoint", "duplicate", "cohort", "patient_field"]
)
def test_final_report_rejects_unsealed_or_incomparable_results(
    comparison_inputs, tmp_path, corruption
):
    directories, source = comparison_inputs
    test = copy.deepcopy(source)
    if corruption == "validation":
        test["scope"] = "validation"
    elif corruption == "checkpoint":
        test["entries"][1]["checkpoint_sha256"] = "changed"
    elif corruption == "duplicate":
        test["entries"].append(test["entries"][1])
    elif corruption == "cohort":
        test["entries"][1]["metrics"] = copy.deepcopy(test["entries"][1]["metrics"])
        test["entries"][1]["metrics"]["n_assessed"] = 2
    else:
        test["timings"]["study_id"] = "private"
    (directories[2] / "summary.json").write_text(json.dumps(test))
    out = tmp_path / "public" / "images"
    with pytest.raises(ValueError):
        draw_final_comparison(*directories, out)
    assert not out.exists()
