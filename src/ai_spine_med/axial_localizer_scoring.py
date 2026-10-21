"""Post-hoc anatomical scoring; reference truth is never an inference input.

Sparse RSNA subarticular points define approximate representative level planes,
not central-canal x/y truth. Only exact AISSLab XML CCS centers support center
errors. In-plane, plane and 3D distances are separate, since an accurate x/y on
an incorrect vertebral level is not a correct end-to-end localization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .assets import sha256_file, write_json
from .axial_localization_audit import LEVELS, _distances, dicom_point
from .axial_localizer_data import _safe_relative


def predicted_point_lps(item: dict, center_xy_norm) -> np.ndarray:
    """Invert the image-only canonical transform using source DICOM geometry."""
    xy = np.asarray(center_xy_norm, dtype=np.float64)
    if xy.shape != (2,) or not np.isfinite(xy).all() or np.any((xy < 0) | (xy > 1)):
        raise ValueError("Invalid predicted normalized center")
    orientation = np.asarray(item["orientation"], dtype=np.float64)
    column, row = orientation[:3], orientation[3:]
    rows, columns = item["image_shape"]
    transpose = abs(column[1]) > abs(column[0])
    if transpose:
        rows, columns = columns, rows
        column, row = row, column
    x, y = xy * np.asarray([columns - 1, rows - 1])
    if column[0] < 0:
        x = columns - 1 - x
    if row[1] < 0:
        y = rows - 1 - y
    if transpose:
        x, y = y, x
    return dicom_point(item, x, y)


def _center_error(item, prediction, crop, anchors):
    exact = [a for a in anchors if a["kind"] == "aisslab_xml_ccs_exact_slice"]
    if not exact:
        return {"status": "no_true_central_canal_reference"}
    allowed = [
        a
        for a in exact
        if a["series_id"] == item["series_id"]
        or (a.get("frame_sha256") and a["frame_sha256"] == item.get("frame_sha256"))
    ]
    if not allowed:
        return {"status": "unverifiable_other_series_or_frame"}
    point = predicted_point_lps(item, prediction["center_xy_norm"])
    recorded = crop.get("metadata", {}).get("anchor_lps")
    if recorded is not None and not np.allclose(point, recorded, atol=1e-3, rtol=0):
        raise ValueError("Predicted center geometry differs from actual extracted crop")
    normal = np.asarray(item["normal"], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    # Sparse annotations may contain more than one accepted representative
    # slice. Choose its nearest plane, then nearest point, never another level.
    comparisons = []
    for anchor in allowed:
        delta = point - np.asarray(anchor["point_lps"], dtype=np.float64)
        plane = abs(float(delta @ normal))
        distance3d = float(np.linalg.norm(delta))
        inplane = float(np.sqrt(max(0.0, distance3d**2 - plane**2)))
        comparisons.append((plane, distance3d, inplane))
    plane, distance3d, inplane = min(comparisons)
    return {
        "status": "verifiable_exact_ccs",
        "plane_error_mm": plane,
        "inplane_error_mm": inplane,
        "distance3d_mm": distance3d,
        "reference_policy": "nearest same-level exact CCS plane, same series or verified shared frame",
    }


def score_case(
    source: dict, record: dict | None, anchors: list[dict], prior_mean=None, prior_std=None
) -> list[dict]:
    """Assess already-produced image predictions without calling any model."""
    slices = source.get("selected_slices", [])
    provenance = {} if record is None else record.get("provenance", {})
    predictions = provenance.get("prediction", {}).get("levels", {})
    crops = provenance.get("crops", {})
    weights = np.asarray(provenance.get("slice_weights", []), dtype=np.float64)
    if weights.size and (
        weights.shape != (2, len(slices))
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or not np.allclose(weights.sum(-1), 1, atol=1e-5, rtol=0)
    ):
        raise ValueError("Invalid recorded image-only slice probabilities")
    results = []
    for j, level in enumerate(LEVELS):
        subset = [a for a in anchors if a["level"] == level]
        distances = _distances(slices, subset)
        result = {
            "dataset": source["dataset"],
            "study_id": source["study_id"],
            "split": source["split"],
            "level": level,
            "reference_status": "present" if subset else "missing",
            "reference_kinds": sorted({a["kind"] for a in subset}),
            "prediction_status": "missing",
            "plane_status": "missing_prediction",
            "top1_plane_distance_mm": None,
            "hit5": False,
            "hit10": False,
            "uniform_expected_hit10": float(np.mean(distances <= 10))
            if subset and len(slices)
            else None,
            "positional_baseline": None,
            "center": {"status": "missing_prediction"},
        }
        if prior_mean is not None and prior_std is not None and slices:
            mean, std = np.asarray(prior_mean), np.asarray(prior_std)
            if (
                mean.shape != (2,)
                or std.shape != (2,)
                or not np.isfinite([*mean, *std]).all()
                or np.any(std <= 0)
            ):
                raise ValueError("Invalid frozen TRAIN positional prior")
            position = np.asarray([s["positions"][0] for s in slices])
            top = int(np.argmax(-0.5 * ((position - mean[j]) / std[j]) ** 2))
            result["positional_baseline"] = {
                "top1_index": top,
                "hit5": bool(distances[top] <= 5),
                "hit10": bool(distances[top] <= 10),
                "plane_distance_mm": float(distances[top]) if np.isfinite(distances[top]) else None,
            }
        prediction = predictions.get(level)
        if not isinstance(prediction, dict) or prediction.get("status") != "ok":
            if not subset:
                result["plane_status"] = "missing_reference"
            results.append(result)
            continue
        result["prediction_status"] = "ok"
        key = str(prediction["series_id"]), int(prediction["instance"])
        matching = [
            i
            for i, item in enumerate(slices)
            if (str(item["series_id"]), int(item["instance"])) == key
        ]
        if len(matching) != 1:
            raise ValueError("Predicted anchor missing or duplicated in frozen image descriptors")
        top = matching[0]
        if (
            "anchor_selected_index" in prediction
            and int(prediction["anchor_selected_index"]) != top
        ):
            raise ValueError("Predicted selected index and source identity disagree")
        if weights.size and int(weights[j].argmax()) != top:
            raise ValueError("Recorded probabilities disagree with selected prediction")
        result["top1_index"] = top
        if subset:
            distance = distances[top]
            result.update(
                plane_status="verifiable"
                if np.isfinite(distance)
                else "unverifiable_other_series_or_frame",
                top1_plane_distance_mm=float(distance) if np.isfinite(distance) else None,
                hit5=bool(distance <= 5),
                hit10=bool(distance <= 10),
            )
        else:
            result["plane_status"] = "missing_reference"
        result["center"] = _center_error(slices[top], prediction, crops.get(level, {}), subset)
        results.append(result)
    return results


def _distribution(values):
    return {
        "n": len(values),
        "mean": float(np.mean(values)) if values else None,
        "median": float(np.median(values)) if values else None,
        "p90": float(np.quantile(values, 0.9)) if values else None,
    }


def summarize_cases(cases: list[dict]) -> dict:
    """Only fixed strings/counts/numbers enter the public-safe aggregate."""
    grouped = defaultdict(list)
    for case in cases:
        if (
            case["dataset"] not in {"aisslab", "rsna"}
            or case["level"] not in LEVELS
            or case["split"] not in {"train", "validation", "test"}
        ):
            raise ValueError("Unknown aggregate grouping label")
        grouped[(case["dataset"], case["split"], case["level"])].append(case)
    output = {}
    for (dataset, split, level), rows in grouped.items():
        known = [r for r in rows if r["reference_status"] == "present"]
        centers = [r["center"] for r in rows if r["center"]["status"] == "verifiable_exact_ccs"]
        baseline = [r["positional_baseline"] for r in known if r["positional_baseline"] is not None]
        distances = [
            r["top1_plane_distance_mm"] for r in known if r["top1_plane_distance_mm"] is not None
        ]
        uniform = [
            r["uniform_expected_hit10"] for r in known if r["uniform_expected_hit10"] is not None
        ]
        output[f"{dataset}/{split}/{level}"] = {
            "requested": len(rows),
            "predicted": sum(r["prediction_status"] == "ok" for r in rows),
            "reference_levels": len(known),
            "missing_reference": len(rows) - len(known),
            "missing_prediction_with_reference": sum(r["prediction_status"] != "ok" for r in known),
            "unverifiable_other_series_or_frame": sum(
                r["plane_status"] == "unverifiable_other_series_or_frame" for r in known
            ),
            "verifiable_plane_predictions": len(distances),
            "hit5_count": sum(r["hit5"] for r in known),
            "hit10_count": sum(r["hit10"] for r in known),
            "hit5": float(np.mean([r["hit5"] for r in known])) if known else None,
            "hit10": float(np.mean([r["hit10"] for r in known])) if known else None,
            "plane_distance_mm_when_verifiable": _distribution(distances),
            "uniform_expected_hit10": float(np.mean(uniform)) if uniform else None,
            "frozen_train_positional_baseline_hit5": float(np.mean([r["hit5"] for r in baseline]))
            if baseline
            else None,
            "frozen_train_positional_baseline_hit10": float(np.mean([r["hit10"] for r in baseline]))
            if baseline
            else None,
            "true_ccs_center": {
                "n": len(centers),
                **{
                    key: _distribution([r[key] for r in centers])
                    for key in ("plane_error_mm", "inplane_error_mm", "distance3d_mm")
                },
                "inplane_within10mm": float(np.mean([r["inplane_error_mm"] <= 10 for r in centers]))
                if centers
                else None,
                "three_d_within10mm": float(np.mean([r["distance3d_mm"] <= 10 for r in centers]))
                if centers
                else None,
            },
            "reference_interpretation": "exact CCS representative slice and center"
            if dataset == "aisslab"
            else "approximate level planes; subarticular points are not canal-center truth",
        }
    return {
        "schema_version": 1,
        "groups": output,
        "hit_denominator": "all levels with anatomical references, including missing predictions and unverifiable series as misses; missing references excluded",
        "median_distance_denominator": "verifiable same-series/shared-frame predictions only; unverifiable counts reported separately",
        "center_interpretation": "end-to-end predicted-anchor center against AISSLab exact CCS only; plane, in-plane and 3D errors separate; not center-on-GT-slice validation",
        "grade_labels_used": False,
    }


def score_localization(
    input_index: str | Path,
    prediction_root: str | Path,
    out: str | Path,
    *,
    references: str | Path | None = None,
    localizer_selection: str | Path | None = None,
    splits=("validation",),
    datasets=("rsna", "aisslab"),
) -> dict:
    """Join already frozen predictions with separate anatomy, never rerun inference."""
    input_index, prediction_root, out = map(Path, (input_index, prediction_root, out))
    index = json.loads(input_index.read_text())
    if (
        set(splits) - {"train", "validation", "test"}
        or not splits
        or set(datasets) - {"rsna", "aisslab"}
    ):
        raise ValueError("Invalid explicit post-hoc scoring scope")
    if "test" in splits and (
        references is None
        or index.get("configuration", {}).get("seals", {}).get("both_selections_frozen_before_test")
        is not True
    ):
        raise ValueError(
            "TEST scoring requires separate references and both frozen model selections"
        )
    rows = [r for r in index["records"] if r["split"] in splits and r["dataset"] in datasets]
    lookup = None
    if references is not None:
        reference_index = json.loads(Path(references).read_text())
        if reference_index.get("input_index_sha256") != sha256_file(input_index):
            raise ValueError("Anatomical references belong to a different frozen input index")
        lookup = {(r["dataset"], r["study_id"]): r for r in reference_index["records"]}
        if len(lookup) != len(reference_index["records"]):
            raise ValueError("Duplicate anatomical reference patient")
    prior_mean = prior_std = None
    selection_sha = None
    if localizer_selection is not None:
        import torch

        selection_path = Path(localizer_selection)
        selection = json.loads(selection_path.read_text())
        if (
            selection.get("frozen") is not True
            or selection.get("model_family") != "explicit_axial_localizer_v1"
        ):
            raise ValueError("Positional comparison requires frozen localizer selection")
        checkpoint_path = _safe_relative(selection_path.parent, selection["checkpoint"])
        if sha256_file(checkpoint_path) != selection["checkpoint_sha256"]:
            raise ValueError("Frozen localizer checkpoint changed")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        prior_mean = checkpoint["level_state"]["prior_mean"].numpy()
        prior_std = checkpoint["level_state"]["prior_std"].numpy()
        selection_sha = sha256_file(selection_path)
    cases, missing_prediction_files = [], 0
    for row in rows:
        token = hashlib.sha256(f"{row['dataset']}:{row['study_id']}".encode()).hexdigest()[:24]
        path = prediction_root / "records" / f"{token}.json"
        record = json.loads(path.read_text()) if path.exists() else None
        if record is None:
            missing_prediction_files += 1
        elif any(record.get(key) != row[key] for key in ("dataset", "study_id", "split")):
            raise ValueError("Prediction patient/split identity differs from frozen inputs")
        anchors = (
            row.get("anchors", [])
            if lookup is None
            else lookup[(row["dataset"], row["study_id"])]["anchors"]
        )
        cases.extend(score_case(row, record, anchors, prior_mean, prior_std))
    summary = summarize_cases(cases)
    summary.update(
        {
            "scored_splits": list(splits),
            "scored_datasets": list(datasets),
            "missing_prediction_files": missing_prediction_files,
        }
    )
    out.mkdir(parents=True, exist_ok=True)
    write_json(
        out / "cases-private.json",
        {
            "scope": "private post-hoc anatomy; no fitting/selection/inference",
            "input_index_sha256": sha256_file(input_index),
            "localizer_selection_sha256": selection_sha,
            "records": cases,
        },
    )
    write_json(out / "summary.json", summary)
    return summary


def score_localizer_predictions(
    inputs_index, features_dir, localizer_selection, out, references=None
):
    """Stable workflow entry: validation during fit, sealed TEST afterward.

    The extractor configuration determines which domains were actually run.
    Domains intentionally absent from extraction are omitted, not failed cases.
    Missing records inside an extracted domain remain explicit failures.
    """
    inputs_index, features_dir = Path(inputs_index), Path(features_dir)
    index = json.loads(inputs_index.read_text())
    config_path = features_dir / "configuration.json"
    if not config_path.is_file():
        raise ValueError("Scoring requires the frozen extraction configuration")
    extraction = json.loads(config_path.read_text())
    if extraction.get("input_index_sha256") != sha256_file(inputs_index):
        raise ValueError("Predictions refer to a different frozen image input index")
    datasets = tuple(extraction.get("datasets", []))
    phase = extraction.get("phase")
    if phase not in {"fitting", "evaluation_image_inputs"} or not datasets:
        raise ValueError("Scoring requires an explicit extraction phase and domains")
    # Evaluation extraction uses the same image-input phase name as its source
    # index; the source determines whether it contains held-out TEST patients.
    splits = ("validation",) if phase == "fitting" else ("test",)
    summary = score_localization(
        inputs_index,
        features_dir,
        out,
        references=references,
        localizer_selection=localizer_selection,
        splits=splits,
        datasets=datasets,
    )
    present = sorted({r["dataset"] for r in index["records"]})
    summary["domains_intentionally_not_extracted"] = sorted(set(present) - set(datasets))
    write_json(Path(out) / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-index", required=True)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--localizer-selection", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--references")
    args = parser.parse_args(argv)
    score_localizer_predictions(
        args.inputs_index, args.features_dir, args.localizer_selection, args.out, args.references
    )


if __name__ == "__main__":
    main()
