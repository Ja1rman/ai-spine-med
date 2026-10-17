"""Paired study-level bootstrap for fixed predictions, without model selection."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np

from .assets import write_json


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_prediction_csv(path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["label"] = None if row["label"] == "" else int(row["label"])
        row["prediction"] = None if row["prediction"] == "" else int(row["prediction"])
    return rows


def _matrix_metrics(matrices: np.ndarray, severe_total: np.ndarray) -> dict[str, np.ndarray]:
    diagonal = np.diagonal(matrices, axis1=-2, axis2=-1)
    denominators = matrices.sum(axis=-1) + matrices.sum(axis=-2)
    f1 = np.divide(
        2 * diagonal,
        denominators,
        out=np.zeros_like(diagonal, dtype=np.float64),
        where=denominators != 0,
    )
    severe = np.divide(
        diagonal[..., 2],
        severe_total,
        out=np.full(diagonal.shape[:-1], np.nan),
        where=severe_total != 0,
    )
    assessed = matrices.sum(axis=(-2, -1)) > 0
    return {
        "macro_f1": np.where(assessed, f1.mean(axis=-1), np.nan),
        "moderate_f1": np.where(assessed, f1[..., 1], np.nan),
        "severe_rate": severe,
    }


def _interval(draws: np.ndarray, point: float) -> dict:
    values = np.asarray(draws)[np.isfinite(draws)]
    return {
        "point": float(point) if np.isfinite(point) else None,
        "lower_95": float(np.percentile(values, 2.5)) if len(values) else None,
        "upper_95": float(np.percentile(values, 97.5)) if len(values) else None,
        "valid_resamples": int(len(values)),
    }


def paired_study_bootstrap(
    predictions: dict[str, list[dict]],
    *,
    contrasts: dict[str, tuple[list[str], list[str]]] | None = None,
    resamples: int = 2000,
    seed: int = 2026,
) -> dict:
    """Keep both levels and failures together, using identical draws for every model.

    A family contrast is the mean of its individual model metrics, not a
    probability ensemble. All labels/statuses must agree before any resampling.
    Percentile intervals describe case sampling for fixed fitted models only.
    """
    if not predictions or resamples < 100 or seed < 0:
        raise ValueError("Require models, at least 100 resamples and a nonnegative seed")
    names = list(predictions)
    first = predictions[names[0]]
    if not first:
        raise ValueError("Predictions are empty")

    def key(row):
        return str(row["study_id"]), str(row["series_id"]), row["level"]

    reference = {key(row): row for row in first}
    if len(reference) != len(first):
        raise ValueError("Duplicate prediction target")
    if len({(k[0], k[2]) for k in reference}) != len(reference):
        raise ValueError("Duplicate study/level")
    studies = sorted({item[0] for item in reference})
    study_index = {name: index for index, name in enumerate(studies)}
    matrices = np.zeros((len(names), len(studies), 3, 3), dtype=np.int64)
    severe_total = np.zeros(len(studies), dtype=np.int64)
    for row in reference.values():
        if row["label"] not in (0, 1, 2):
            raise ValueError("Every bootstrap target requires a valid reference label")
        severe_total[study_index[str(row["study_id"])]] += row["label"] == 2
    for model_index, name in enumerate(names):
        rows = predictions[name]
        indexed = {key(row): row for row in rows}
        if len(indexed) != len(rows) or indexed.keys() != reference.keys():
            raise ValueError("Models have different prediction target sets")
        for target, row in indexed.items():
            ref = reference[target]
            if (row["label"], row["status"]) != (ref["label"], ref["status"]):
                raise ValueError("Models have different labels or assessed subsets")
            if row["status"] == "ok":
                if row["prediction"] not in (0, 1, 2):
                    raise ValueError("Assessed prediction has an invalid category")
                matrices[model_index, study_index[target[0]], row["label"], row["prediction"]] += 1
            elif row.get("prediction") is not None:
                raise ValueError("An unassessed target cannot carry a category")
    contrasts = contrasts or {}
    for left, right in contrasts.values():
        if not left or not right or not set(left + right).issubset(names):
            raise ValueError("Contrast references an empty or unknown model family")
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(studies), np.full(len(studies), 1 / len(studies)), resamples)
    sampled_matrices = np.einsum("bs,msij->bmij", weights, matrices, optimize=True)
    sampled_severe = weights @ severe_total
    draws = _matrix_metrics(sampled_matrices, sampled_severe[:, None])
    points = _matrix_metrics(matrices.sum(axis=1), np.asarray(severe_total.sum()))
    entries = {
        name: {
            metric: _interval(values[:, index], points[metric][index])
            for metric, values in draws.items()
        }
        for index, name in enumerate(names)
    }
    differences = {}
    for name, (left, right) in contrasts.items():
        left_indices, right_indices = (
            [names.index(k) for k in left],
            [names.index(k) for k in right],
        )
        differences[name] = {
            "left": left,
            "right": right,
            "metrics": {
                metric: _interval(
                    values[:, left_indices].mean(axis=1) - values[:, right_indices].mean(axis=1),
                    points[metric][left_indices].mean() - points[metric][right_indices].mean(),
                )
                for metric, values in draws.items()
            },
        }
    return {
        "method": "Paired nonparametric percentile bootstrap, unit=study_id; all levels and failures stay together",
        "resamples": resamples,
        "seed": seed,
        "n_studies": len(studies),
        "n_targets": len(first),
        "n_assessed": sum(row["status"] == "ok" for row in first),
        "n_severe_all_requested": int(severe_total.sum()),
        "models": entries,
        "contrasts": differences,
        "limitations": "Intervals condition on these fitted models and this cohort. They do not measure label uncertainty, training-data variation, new-site generalization, or clinical validity. A family mean is not an ensemble. Macro-F1 uses fixed classes [0,1,2], zero F1 for an absent class.",
    }


def bootstrap_v1_test(
    test_dir: str | Path,
    output_dir: str | Path,
    *,
    resamples: int = 2000,
    seed: int = 2026,
) -> dict:
    """Analyze the already completed v1 comparison; never load new models."""
    import json

    root, out = Path(test_dir), Path(output_dir)
    if out.exists():
        raise ValueError("Output directory already exists")
    summary = json.loads((root / "summary.json").read_text())
    if summary.get("scope") != "test" or not (root / "manifest.json").is_file():
        raise ValueError("Require a completed fixed test evaluation")
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest.get("scope") != "heldout_test"
        or manifest.get("summary_sha256") != _sha(root / "summary.json")
        or manifest.get("selection_sha256") != summary.get("selection_sha256")
    ):
        raise ValueError("Test summary and completed manifest disagree")
    predictions, hashes = {}, {}
    for entry in summary["entries"]:
        name = entry["variant"] + (f"_{entry['seed']}" if entry["seed"] is not None else "")
        path = root / "entries" / entry["variant"]
        if entry["seed"] is not None:
            path /= f"seed-{entry['seed']}"
        path /= "predictions.csv"
        rows = load_prediction_csv(path)
        if any(row.get("split") != "test" for row in rows):
            raise ValueError("A v1 test CSV contains a non-test row")
        matrix = np.zeros((3, 3), dtype=int)
        for row in rows:
            if row["status"] == "ok":
                if row["label"] not in (0, 1, 2) or row["prediction"] not in (0, 1, 2):
                    raise ValueError("Invalid category in an assessed test prediction")
                matrix[row["label"], row["prediction"]] += 1
        if (
            matrix.tolist() != entry["metrics"]["conditional"]["confusion_matrix"]
            or len(rows) != entry["metrics"]["n_labeled"]
        ):
            raise ValueError("A test prediction table disagrees with the fixed summary")
        predictions[name], hashes[name] = rows, _sha(path)
    expected = {"original"} | {
        f"{variant}_{seed}"
        for variant in ("frozen_head", "fine_tuned")
        for seed in (42, 2024, 3407)
    }
    if set(predictions) != expected:
        raise ValueError("Bootstrap requires all seven fixed v1 models")
    frozen = [f"frozen_head_{seed}" for seed in (42, 2024, 3407)]
    fine = [f"fine_tuned_{seed}" for seed in (42, 2024, 3407)]
    result = paired_study_bootstrap(
        predictions,
        contrasts={
            "frozen_mean_minus_original": (frozen, ["original"]),
            "fine_mean_minus_original": (fine, ["original"]),
            "fine_mean_minus_frozen_mean": (fine, frozen),
        },
        resamples=resamples,
        seed=seed,
    )
    result.update(
        scope="Fixed v1 test models only; no retraining or selection from these intervals",
        selection_sha256=summary["selection_sha256"],
        source_summary_sha256=_sha(root / "summary.json"),
        prediction_sha256=hashes,
    )
    out.mkdir(parents=True)
    write_json(out / "summary.json", result)
    return result


def legacy_validation_ensemble(head_dir: str | Path, output_dir: str | Path) -> dict:
    """Average the three existing v1 heads on development validation only."""
    from .data import CLASS_NAMES
    from .training import _write_predictions, evaluate_predictions

    root, out = Path(head_dir), Path(output_dir)
    if out.exists():
        raise ValueError("Output directory already exists")
    paths = [root / "runs" / f"seed-{seed}" / "predictions.csv" for seed in (42, 2024, 3407)]
    inputs = [load_prediction_csv(path) for path in paths]
    index = [
        {(row["study_id"], row["series_id"], row["level"]): row for row in rr} for rr in inputs
    ]
    if any(len(table) != len(rr) for table, rr in zip(index, inputs)):
        raise ValueError("Duplicate legacy validation target")
    for table in index[1:]:
        if table.keys() != index[0].keys() or any(
            (row["label"], row["status"]) != (index[0][key]["label"], index[0][key]["status"])
            for key, row in table.items()
        ):
            raise ValueError("Legacy heads have different validation cohorts")
    rows = []
    for item in inputs[0]:
        if item["split"] != "validation":
            raise ValueError("The legacy ensemble accepts validation predictions only")
        row = dict(item)
        if row["status"] == "ok":
            key = item["study_id"], item["series_id"], item["level"]
            probabilities = np.asarray(
                [[float(table[key][f"p_{name}"]) for name in CLASS_NAMES] for table in index]
            )
            if (
                not np.isfinite(probabilities).all()
                or np.any(probabilities < 0)
                or np.any(probabilities > 1)
                or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6)
            ):
                raise ValueError("Invalid probability vector in legacy validation predictions")
            mean = probabilities.mean(axis=0)
            row["prediction"] = int(mean.argmax())
            for name, value in zip(CLASS_NAMES, mean):
                row[f"p_{name}"] = float(value)
        rows.append(row)
    metrics = evaluate_predictions(rows)
    result = {
        "scope": "Development validation; v1 heads selected epochs on this same validation, not an independent test",
        "ensemble": "Uniform arithmetic mean of all three v1 heads; encoder evaluated once",
        "seeds": [42, 2024, 3407],
        "metrics": metrics,
        "input_prediction_sha256": {str(s): _sha(p) for s, p in zip((42, 2024, 3407), paths)},
    }
    out.mkdir(parents=True)
    _write_predictions(out / "predictions.csv", rows)
    write_json(out / "summary.json", result)
    return result


def bootstrap_v2_validation(
    experiment_dir: str | Path,
    legacy_ensemble_dir: str | Path,
    output_dir: str | Path,
    *,
    resamples: int = 2000,
    seed: int = 2026,
) -> dict:
    """Case-sampling intervals for the completed development check, never a new test."""
    import json

    root, legacy, out = map(Path, (experiment_dir, legacy_ensemble_dir, output_dir))
    if out.exists():
        raise ValueError("Output directory already exists")
    summary = json.loads((root / "summary.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        summary.get("test_evaluated") is not False
        or summary.get("validation_used_for_epoch_or_candidate_selection") is not False
        or summary.get("configuration") != manifest.get("configuration")
        or set(summary.get("candidates", {})) != {"linear", "mlp", "coral"}
    ):
        raise ValueError("Require a completed train-CV-selected development experiment")
    predictions, hashes = {}, {}
    expected_scores = {}
    for name, candidate in summary["candidates"].items():
        for run in candidate["validation"]["runs"]:
            key = f"{name}_{run['seed']}"
            path = root / "validation" / name / f"seed-{run['seed']}" / "predictions.csv"
            if _sha(path) != run["predictions_sha256"]:
                raise ValueError("A development prediction table changed")
            predictions[key], hashes[key] = load_prediction_csv(path), _sha(path)
            expected_scores[key] = run["metrics"]["conditional"]["macro_f1"]
        key = f"{name}_ensemble"
        path = root / "validation" / name / "ensemble" / "predictions.csv"
        if _sha(path) != candidate["validation"]["ensemble"]["predictions_sha256"]:
            raise ValueError("A development ensemble table changed")
        predictions[key], hashes[key] = load_prediction_csv(path), _sha(path)
        expected_scores[key] = candidate["validation"]["ensemble"]["metrics"]["conditional"][
            "macro_f1"
        ]
    predictions["legacy_ensemble"] = load_prediction_csv(legacy / "predictions.csv")
    hashes["legacy_ensemble"] = _sha(legacy / "predictions.csv")
    if any(row.get("split") != "validation" for rr in predictions.values() for row in rr):
        raise ValueError("Development bootstrap accepts original validation rows only")
    result = paired_study_bootstrap(
        predictions,
        contrasts={
            "mlp_ensemble_minus_linear_ensemble": (["mlp_ensemble"], ["linear_ensemble"]),
            "mlp_ensemble_minus_legacy_ensemble": (["mlp_ensemble"], ["legacy_ensemble"]),
            "coral_ensemble_minus_linear_ensemble": (["coral_ensemble"], ["linear_ensemble"]),
        },
        resamples=resamples,
        seed=seed,
    )
    for key, expected in expected_scores.items():
        if not np.isclose(result["models"][key]["macro_f1"]["point"], expected, atol=1e-12):
            raise ValueError("Independent matrix metric disagrees with development summary")
    result.update(
        scope="Original validation development cohort only, not a new independent test",
        source_summary_sha256=_sha(root / "summary.json"),
        source_selection_sha256=_sha(root / "selection.json"),
        prediction_sha256=hashes,
        legacy_caveat="Legacy v1 heads selected their epochs on this validation. The v2 heads selected epochs by TRAIN CV. This historical comparison does not isolate architecture alone.",
    )
    out.mkdir(parents=True)
    write_json(out / "summary.json", result)
    return result
