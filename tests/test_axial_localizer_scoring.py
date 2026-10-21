"""Post-hoc anatomy metrics use predictions unchanged, never clinical grades."""

import hashlib
import json

import numpy as np
import pytest

from ai_spine_med import axial_localizer_scoring as scoring
from ai_spine_med.assets import sha256_file


def item(z, instance, series="1", frame="shared"):
    return {
        "series_id": series,
        "instance": instance,
        "path": f"{series}/{instance}.dcm",
        "image_shape": [9, 9],
        "orientation": [1, 0, 0, 0, 1, 0],
        "spacing": [1, 1],
        "origin_lps": [0, 0, z],
        "center_lps": [4, 4, z],
        "normal": [0, 0, 1],
        "frame_sha256": frame,
        "positions": [0.0 if instance == 1 else 1.0] * 2,
    }


def anchor(s, point=None, kind="aisslab_xml_ccs_exact_slice", level="L5-S1"):
    return {
        "kind": kind,
        "level": level,
        "series_id": s["series_id"],
        "path": s["path"],
        "frame_sha256": s["frame_sha256"],
        "point_lps": s["center_lps"] if point is None else point,
    }


def source(slices, dataset="aisslab"):
    return {
        "dataset": dataset,
        "study_id": "private-patient-id",
        "split": "validation",
        "status": "ok",
        "selected_slices": slices,
    }


def prediction(slices, index=0, center=(0.5, 0.5)):
    selected = slices[index]
    return {
        "provenance": {
            "prediction": {
                "levels": {
                    "L5-S1": {
                        "status": "ok",
                        "series_id": selected["series_id"],
                        "instance": selected["instance"],
                        "anchor_selected_index": index,
                        "center_xy_norm": list(center),
                    }
                }
            },
            "crops": {
                "L5-S1": {
                    "metadata": {
                        "anchor_lps": scoring.predicted_point_lps(selected, center).tolist()
                    }
                }
            },
        }
    }


def l5(cases):
    return [c for c in cases if c["level"] == "L5-S1"][0]


def test_end_to_end_center_distinguishes_wrong_plane_from_inplane_accuracy():
    slices = [item(12, 1), item(0, 2)]
    case = l5(
        scoring.score_case(source(slices), prediction(slices), [anchor(slices[1], [7, 8, 0])])
    )
    assert case["hit10"] is False
    assert case["center"]["plane_error_mm"] == 12
    assert case["center"]["inplane_error_mm"] == 5
    assert case["center"]["distance3d_mm"] == 13


def test_subarticular_reference_never_becomes_true_canal_xy():
    slices = [item(0, 1)]
    case = l5(
        scoring.score_case(
            source(slices, "rsna"),
            prediction(slices),
            [anchor(slices[0], kind="rsna_axial_subarticular_approximate_level")],
        )
    )
    assert case["hit5"]
    assert case["center"]["status"] == "no_true_central_canal_reference"


def test_other_series_unverifiable_is_explicit_and_stays_a_miss_in_hit_denominator():
    slices = [item(0, 1, "1", "frame1"), item(0, 2, "2", "frame2")]
    case = l5(scoring.score_case(source(slices), prediction(slices), [anchor(slices[1])]))
    summary = scoring.summarize_cases([case])["groups"]["aisslab/validation/L5-S1"]
    assert summary["reference_levels"] == 1
    assert summary["hit10"] == 0
    assert summary["unverifiable_other_series_or_frame"] == 1
    assert summary["verifiable_plane_predictions"] == 0
    assert summary["plane_distance_mm_when_verifiable"]["median"] is None
    assert case["center"]["status"] == "unverifiable_other_series_or_frame"


def test_same_verified_frame_permits_physical_center_comparison_but_not_exact_series_plane_claim():
    slices = [item(0, 1, "1"), item(0, 2, "2")]
    case = l5(scoring.score_case(source(slices), prediction(slices), [anchor(slices[1])]))
    assert case["plane_status"] == "unverifiable_other_series_or_frame"
    assert case["center"]["status"] == "verifiable_exact_ccs"
    assert case["center"]["distance3d_mm"] == 0


def test_missing_reference_is_not_normal_nor_a_wrong_localization():
    slices = [item(0, 1)]
    cases = scoring.score_case(source(slices), prediction(slices), [])
    summary = scoring.summarize_cases(cases)["groups"]["aisslab/validation/L5-S1"]
    assert summary["predicted"] == 1 and summary["missing_reference"] == 1
    assert summary["reference_levels"] == 0 and summary["hit10"] is None


def test_missing_prediction_with_reference_remains_coverage_failure_and_miss():
    slices = [item(0, 1)]
    cases = scoring.score_case(source(slices), None, [anchor(slices[0])])
    summary = scoring.summarize_cases(cases)["groups"]["aisslab/validation/L5-S1"]
    assert summary["missing_prediction_with_reference"] == 1
    assert summary["predicted"] == 0 and summary["hit10"] == 0


def test_frozen_positional_and_uniform_baselines_use_same_bag_and_reference():
    slices = [item(20, 1), item(0, 2)]
    case = l5(
        scoring.score_case(
            source(slices),
            prediction(slices),
            [anchor(slices[1])],
            prior_mean=[0, 1],
            prior_std=[0.1, 0.1],
        )
    )
    assert case["hit10"] is False
    assert case["uniform_expected_hit10"] == 0.5
    assert case["positional_baseline"]["hit10"] is True


def test_mismatched_actual_crop_lps_fails_instead_of_reporting_wrong_physical_metric():
    slices = [item(0, 1)]
    record = prediction(slices)
    record["provenance"]["crops"]["L5-S1"]["metadata"]["anchor_lps"] = [100, 100, 100]
    with pytest.raises(ValueError, match="actual extracted crop"):
        scoring.score_case(source(slices), record, [anchor(slices[0])])


def test_recorded_probabilities_must_match_selected_anchor():
    slices = [item(20, 1), item(0, 2)]
    record = prediction(slices)
    record["provenance"]["slice_weights"] = [[0.9, 0.1], [0.1, 0.9]]
    with pytest.raises(ValueError, match="probabilities disagree"):
        scoring.score_case(source(slices), record, [anchor(slices[1])])


def test_severity_grade_and_ap_perturbations_leave_anatomical_metrics_unchanged():
    slices = [item(0, 1)]
    a = scoring.score_case(
        {**source(slices), "labels": [0, 0]},
        prediction(slices),
        [{**anchor(slices[0]), "grade": 0, "AP_mm": 14}],
    )
    b = scoring.score_case(
        {**source(slices), "labels": [2, 2]},
        prediction(slices),
        [{**anchor(slices[0]), "grade": 2, "AP_mm": 4}],
    )
    assert a == b
    assert "private-patient-id" not in json.dumps(scoring.summarize_cases(a))


@pytest.mark.parametrize(
    "orientation",
    [
        (1, 0, 0, 0, 1, 0),
        (-1, 0, 0, 0, 1, 0),
        (1, 0, 0, 0, -1, 0),
        (0, 1, 0, 1, 0, 0),
        (0, -1, 0, -1, 0, 0),
    ],
)
def test_normalized_predicted_center_inverts_canonical_flips_and_transpose(orientation):
    from ai_spine_med.axial_localization_audit import dicom_point
    from ai_spine_med.axial_localizer_data import canonical_xy

    header = {**item(0, 1), "orientation": orientation, "image_shape": [5, 9], "spacing": [2, 3]}
    xy = canonical_xy(header, 6, 1)
    np.testing.assert_allclose(
        scoring.predicted_point_lps(header, xy), dicom_point(header, 6, 1), atol=1e-6
    )


def test_test_scoring_refuses_unsealed_references_before_reading_predictions(tmp_path):
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"configuration": {}, "records": []}))
    with pytest.raises(ValueError, match="both frozen"):
        scoring.score_localization(
            index,
            tmp_path / "predictions",
            tmp_path / "out",
            splits=("test",),
            references=tmp_path / "missing.json",
        )
    assert not (tmp_path / "out").exists()


def test_intentionally_unextracted_domain_is_excluded_not_failed(tmp_path):
    slices = [item(0, 1)]
    rsna = source(slices, "rsna")
    rsna["anchors"] = [anchor(slices[0], kind="rsna_axial_subarticular_approximate_level")]
    aisslab = {**source(slices), "anchors": [anchor(slices[0])]}
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"records": [rsna, aisslab]}))
    root = tmp_path / "features"
    (root / "records").mkdir(parents=True)
    (root / "configuration.json").write_text(
        json.dumps(
            {"input_index_sha256": sha256_file(index), "datasets": ["rsna"], "phase": "fitting"}
        )
    )
    record = {**{key: rsna[key] for key in ("dataset", "study_id", "split")}, **prediction(slices)}
    token = hashlib.sha256(f"rsna:{rsna['study_id']}".encode()).hexdigest()[:24]
    (root / "records" / f"{token}.json").write_text(json.dumps(record))
    summary = scoring.score_localizer_predictions(index, root, None, tmp_path / "scores")
    assert summary["domains_intentionally_not_extracted"] == ["aisslab"]
    assert all(key.startswith("rsna/") for key in summary["groups"])
    assert summary["missing_prediction_files"] == 0
