"""Fixed 50/50 probability fusion of completed validation ensembles.

This post-hoc development comparison reads prediction tables only: no training,
MRI, feature extraction, test evaluation, calibration or weight selection.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .assets import sha256_file, write_json
from .data import CLASS_NAMES
from .training import LEVELS, SEEDS, evaluate_predictions
from .uncertainty import paired_study_bootstrap

VARIANTS = ("sagittal", "axial", "fusion")
MODELS = (*VARIANTS, "probability_mean")
FIELDS = (
    "study_id",
    "series_id",
    "level",
    "split",
    "label",
    "status",
    "reason",
    "sagittal_available",
    "axial_available",
    "prediction",
    *(f"p_{name}" for name in CLASS_NAMES),
)
_IDENTIFIER = re.compile(r"^[0-9]+$")
_HASH = re.compile(r"^[0-9a-f]{64}$")


def _key(row):
    return str(row["study_id"]), str(row["series_id"]), row["level"]


def _boolean(value):
    if type(value) is bool:
        return value
    if value in ("True", "False"):
        return value == "True"
    raise ValueError("Availability must be an explicit boolean")


def _indexed(rows):
    if not rows:
        raise ValueError("Prediction table is empty")
    result, study_levels = {}, set()
    for original in rows:
        if not set(FIELDS).issubset(original):
            raise ValueError("Prediction table is missing required fields")
        row = {field: original[field] for field in FIELDS}
        key = _key(row)
        if (
            not all(_IDENTIFIER.fullmatch(x) for x in key[:2])
            or key[2] not in LEVELS
            or row["split"] != "validation"
            or type(row["label"]) is not int
            or row["label"] not in (0, 1, 2)
            or row["status"] not in {"ok", "not_assessed"}
            or not isinstance(row["reason"], str)
        ):
            raise ValueError("Require labeled validation targets with known levels and statuses")
        if key in result or (key[0], key[2]) in study_levels:
            raise ValueError("Duplicate prediction study/level or target key")
        row["sagittal_available"] = _boolean(row["sagittal_available"])
        row["axial_available"] = _boolean(row["axial_available"])
        if row["status"] == "ok":
            try:
                values = np.asarray([row[f"p_{name}"] for name in CLASS_NAMES], dtype=float)
            except (TypeError, ValueError) as error:
                raise ValueError("Invalid probability vector") from error
            if (
                not np.isfinite(values).all()
                or np.any(values < 0)
                or np.any(values > 1)
                or not np.isclose(values.sum(), 1, rtol=0, atol=1e-6)
                or type(row["prediction"]) is not int
                or row["prediction"] != int(values.argmax())
                or row["reason"]
                or not row["sagittal_available"]
                or not row["axial_available"]
            ):
                raise ValueError("Invalid assessed probabilities, argmax or availability")
        elif (
            not row["reason"]
            or row["prediction"] is not None
            or any(row[f"p_{name}"] is not None for name in CLASS_NAMES)
        ):
            raise ValueError("Not-assessed targets must retain a reason and no predictions")
        result[key] = row
        study_levels.add((key[0], key[2]))
    return result


def read_predictions(path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not set(FIELDS).issubset(reader.fieldnames or ()):
            raise ValueError("Prediction CSV is missing required columns")
        rows = []
        for raw in reader:
            try:
                row = {field: raw[field] for field in FIELDS}
                for field in ("label", "prediction"):
                    row[field] = None if row[field] == "" else int(row[field])
                for name in CLASS_NAMES:
                    field = f"p_{name}"
                    row[field] = None if row[field] == "" else float(row[field])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Prediction CSV has invalid category or probability fields"
                ) from error
            rows.append(row)
    indexed = _indexed(rows)
    return [indexed[k] for k in sorted(indexed)]


def _aligned(*tables):
    mappings = [_indexed(rows) for rows in tables]
    reference = mappings[0]
    for mapping in mappings[1:]:
        if mapping.keys() != reference.keys():
            raise ValueError("Models have different prediction target keys")
        for key, row in reference.items():
            if any(
                mapping[key][field] != row[field]
                for field in (
                    "split",
                    "label",
                    "status",
                    "reason",
                    "sagittal_available",
                    "axial_available",
                )
            ):
                raise ValueError("Models have different labels, coverage or source availability")
    return mappings


def mean_probability_rows(sagittal, axial, fusion) -> list[dict]:
    """Join full keys, then average two probability vectors without consulting labels."""
    sag, ax, _ = _aligned(sagittal, axial, fusion)
    output = []
    for key in sorted(sag):
        row = dict(sag[key])
        if row["status"] == "ok":
            values = np.asarray(
                [[table[key][f"p_{name}"] for name in CLASS_NAMES] for table in (sag, ax)],
                dtype=np.float64,
            ).mean(axis=0)
            row.update({f"p_{name}": float(p) for name, p in zip(CLASS_NAMES, values)})
            row["prediction"] = int(values.argmax())
        output.append(row)
    return output


def metrics_for_rows(rows) -> dict:
    metrics = evaluate_predictions(rows)
    metrics["by_status"] = dict(Counter(row["status"] for row in rows))
    metrics["not_assessed_by_reason"] = dict(
        Counter(row["reason"] for row in rows if row["status"] != "ok")
    )
    for group in [metrics, *metrics["by_level"].values()]:
        group["severe"]["missed_all_requested"] = (
            group["severe"]["n_total"] - group["severe"]["correctly_predicted_severe"]
        )
    return metrics


def _validate_protocol(protocol):
    expected = {
        "schema_version": 1,
        "experiment": "probability_mean_v3",
        "weights": {"sagittal": 0.5, "axial": 0.5},
        "seeds": list(SEEDS),
        "classes": list(CLASS_NAMES),
        "levels": list(LEVELS),
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 2026,
        "training_performed": False,
        "weight_or_threshold_tuning": False,
    }
    if any(protocol.get(k) != v for k, v in expected.items()):
        raise ValueError("Require the fixed equal-probability mean development protocol")
    cohort = protocol.get("expected_validation", {})
    if (
        not isinstance(cohort, dict)
        or set(cohort) != {"n_studies", "n_targets", "n_assessed"}
        or any(type(v) is not int or v < 1 for v in cohort.values())
        or cohort["n_targets"] != len(LEVELS) * cohort["n_studies"]
        or cohort["n_assessed"] > cohort["n_targets"]
    ):
        raise ValueError("Protocol must specify the complete original validation cohort")
    hashes = protocol.get("source_prediction_sha256", {})
    if set(hashes) != set(VARIANTS) or not all(
        isinstance(x, str) and _HASH.fullmatch(x)
        for x in [protocol.get("source_summary_sha256"), *hashes.values()]
    ):
        raise ValueError("Protocol must pin source summary and all three prediction SHA-256")


def load_sources(experiment: Path, protocol: dict):
    _validate_protocol(protocol)
    root = Path(experiment)
    summary_path = root / "summary.json"
    digest = sha256_file(summary_path)
    if digest != protocol["source_summary_sha256"]:
        raise ValueError("Source experiment summary checksum differs from the protocol")
    source = json.loads(summary_path.read_text())
    config = source.get("configuration", {}).get("protocol", {})
    if (
        source.get("test_evaluated") is not False
        or source.get("all_seeds_reported") is not True
        or source.get("all_variants_share_paired_cohort") is not True
        or any(config.get(k) != protocol[k] for k in ("classes", "levels", "seeds"))
    ):
        raise ValueError("Require the completed original three-seed paired development experiment")
    sources, inputs = {}, {"summary.json": digest}
    for variant in VARIANTS:
        relative = f"validation/{variant}/ensemble/predictions.csv"
        path = root / relative
        digest = sha256_file(path)
        ensemble = source["variants"][variant]["validation"]["ensemble"]
        if (
            digest != protocol["source_prediction_sha256"][variant]
            or digest != ensemble["predictions_sha256"]
            or ensemble.get("seeds") != list(SEEDS)
            or ensemble.get("weights") != [1 / 3] * 3
        ):
            raise ValueError("Source prediction checksum, seed order or ensemble weights differ")
        sources[variant] = read_predictions(path)
        inputs[relative] = digest
        metrics = metrics_for_rows(sources[variant])
        old = ensemble["metrics"]
        if (
            metrics["n_requested"] != old["n_requested"]
            or metrics["n_assessed"] != old["n_assessed"]
            or metrics["conditional"]["confusion_matrix"] != old["conditional"]["confusion_matrix"]
            or not np.isclose(
                metrics["conditional"]["macro_f1"],
                old["conditional"]["macro_f1"],
                atol=1e-12,
                rtol=0,
            )
        ):
            raise ValueError("Recalculated baseline metrics differ from the frozen experiment")
    _aligned(*(sources[name] for name in VARIANTS))
    expected = protocol["expected_validation"]
    first = sources["sagittal"]
    actual = {
        "n_studies": len({r["study_id"] for r in first}),
        "n_targets": len(first),
        "n_assessed": sum(r["status"] == "ok" for r in first),
    }
    if actual != expected or any(
        {r["level"] for r in first if r["study_id"] == study} != set(LEVELS)
        for study in {r["study_id"] for r in first}
    ):
        raise ValueError("Validation must retain both requested levels for every original study")
    return sources, inputs


def _transitions(baseline, candidate):
    old, new = _aligned(baseline, candidate)

    def count(keys):
        result = dict.fromkeys(("corrected", "worsened", "both_correct", "both_wrong"), 0)
        for k in keys:
            a, b = old[k], new[k]
            if a["status"] != "ok":
                continue
            ca, cb = a["prediction"] == a["label"], b["prediction"] == b["label"]
            category = (
                "both_correct"
                if ca and cb
                else "worsened"
                if ca
                else "corrected"
                if cb
                else "both_wrong"
            )
            result[category] += 1
        return result

    return {
        **count(old),
        "by_level": {level: count(k for k in old if k[2] == level) for level in LEVELS},
    }


def run_probability_mean(experiment, protocol_path, out):
    protocol_path, out = Path(protocol_path), Path(out).resolve()
    protocol = json.loads(protocol_path.read_text())
    sources, inputs = load_sources(Path(experiment), protocol)
    project = Path(__file__).resolve().parents[2]
    private = (project / "reports" / "runs").resolve()
    if not private.is_relative_to(project) or out == private or not out.is_relative_to(private):
        raise ValueError(
            "Individual predictions must stay inside a separate ignored reports/runs directory"
        )
    if "reports/runs/" not in (project / ".gitignore").read_text().splitlines():
        raise ValueError("Private prediction output must remain excluded from Git")
    if out.exists():
        raise ValueError("Output already exists; preserve completed experiments")
    configuration = {
        "protocol_sha256": sha256_file(protocol_path),
        "implementation_sha256": sha256_file(Path(__file__)),
        "inputs_sha256": inputs,
    }
    out.mkdir(parents=True)
    write_json(out / "manifest.json", {"protocol": protocol, "configuration": configuration})
    mean = mean_probability_rows(*(sources[name] for name in VARIANTS))
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(mean)
    models = {
        name: {"metrics": metrics_for_rows(rows)}
        for name, rows in {**sources, "probability_mean": mean}.items()
    }
    models["probability_mean"]["predictions_sha256"] = sha256_file(out / "predictions.csv")
    transitions = {f"vs_{name}": _transitions(rows, mean) for name, rows in sources.items()}
    with (out / "changes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "study_id",
                "series_id",
                "level",
                "label",
                "status",
                "reason",
                *(f"{name}_prediction" for name in MODELS),
            ),
        )
        writer.writeheader()
        tables = {name: {_key(r): r for r in rows} for name, rows in sources.items()}
        for row in mean:
            item = {
                k: row[k] for k in ("study_id", "series_id", "level", "label", "status", "reason")
            }
            item.update(
                {
                    f"{name}_prediction": table[_key(row)]["prediction"]
                    for name, table in tables.items()
                }
            )
            item["probability_mean_prediction"] = row["prediction"]
            writer.writerow(item)
    bootstrap = paired_study_bootstrap(
        {**sources, "probability_mean": mean},
        contrasts={f"mean_minus_{name}": (["probability_mean"], [name]) for name in VARIANTS},
        resamples=protocol["bootstrap_resamples"],
        seed=protocol["bootstrap_seed"],
    )
    summary = {
        "schema_version": 1,
        "experiment": "probability_mean_v3",
        "scope": "Post-hoc development comparison on previously used original validation; no independent test",
        "protocol": protocol,
        "configuration": configuration,
        **protocol["expected_validation"],
        "models": models,
        "bootstrap": bootstrap,
        "transitions": transitions,
        "training_performed": False,
        "test_evaluated": False,
        "new_version_independently_tested": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if any(sha256_file(Path(experiment) / rel) != digest for rel, digest in inputs.items()):
        raise RuntimeError("A frozen source changed during the comparison")
    write_json(out / "summary.json", summary)
    return summary
