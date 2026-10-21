from unittest.mock import patch

import pytest

from ai_spine_med.axial_localizer_workflow import run_stage


def test_test_extraction_uses_separate_inputs_and_includes_both_datasets():
    with patch("ai_spine_med.axial_localizer_extract.extract_localized_features") as extract:
        run_stage("test-features", device="cpu")
    assert "axial-localizer-test-v6" in str(extract.call_args.args[0])
    assert "axial-localized-test-features-v6" in str(extract.call_args.args[2])
    assert extract.call_args.kwargs["datasets"] == ("rsna", "aisslab")


def test_fitting_extraction_keeps_severity_labels_rsna_only():
    with patch("ai_spine_med.axial_localizer_extract.extract_localized_features") as extract:
        run_stage("features", device="cpu")
    assert extract.call_args.kwargs["datasets"] == ("rsna",)


def test_unknown_stage_fails_without_any_data_access():
    with pytest.raises(ValueError, match="Unknown"):
        run_stage("invalid")


def test_references_are_passed_to_scorer_separately_from_image_inputs():
    with patch("ai_spine_med.axial_localizer_scoring.score_localizer_predictions") as score:
        run_stage("score")
    assert score.call_args.args[0].name == "index.json"
    assert score.call_args.kwargs["references"].name == "references.json"


def test_report_uses_its_separate_validation_and_test_sources():
    with patch("ai_spine_med.localized_stenosis_report.report_localized_stenosis") as report:
        run_stage("report")
    report.assert_called_once_with()
