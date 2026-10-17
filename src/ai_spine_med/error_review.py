"""Private, purposively sampled train/validation error gallery for a physician.

This is a review worksheet, not a representative accuracy estimate or an expert
validation. Images and patient-level tables may only be written to reports/runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .assets import sha256_file, write_json
from .baseline import isotropic_inplane
from .data import CLASS_NAMES, LEVELS
from .dicom import load_sagittal_series
from .training import SEEDS, load_feature_data, predict_head

CLINICIAN_FIELDS = (
    "level_correct",
    "roi_adequate",
    "reference_agreement",
    "suggested_grade",
    "comment",
)
QUOTAS = (
    ("moderate_to_normal", 8),
    ("severe_to_normal", 6),
    ("severe_to_moderate", 6),
    ("normal_to_higher", 6),
    ("l5s1_error", 6),
    ("head_disagreement", 6),
    ("missing_roi", 2),
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _private_output(out: Path) -> Path:
    root = (_project_root() / "reports" / "runs").resolve()
    out = out.resolve()
    if (
        not root.is_relative_to(_project_root().resolve())
        or out == root
        or not out.is_relative_to(root)
    ):
        raise ValueError("MRI review output must be a separate ignored reports/runs directory")
    ignore = (_project_root() / ".gitignore").read_text().splitlines()
    if "reports/runs/" not in ignore:
        raise ValueError("reports/runs must remain excluded from Git before rendering MRI")
    if out.exists() and any(out.iterdir()):
        raise ValueError(
            "Review output already contains files; preserve existing physician answers"
        )
    return out


def _key(item: dict) -> tuple[str, str]:
    return str(item["study_id"]), str(item["level"])


def _order(item: dict, seed: int) -> tuple:
    digest = hashlib.sha256(
        f"{seed}:{item['study_id']}:{item['series_id']}:{item['level']}".encode()
    ).hexdigest()
    return item["split"] != "validation", digest


def _tags(item: dict) -> list[str]:
    if item["status"] != "ok":
        return ["missing_roi"]
    truth, prediction = item["label"], item["ensemble_prediction"]
    tags = []
    if truth == 1 and prediction == 0:
        tags.append("moderate_to_normal")
    if truth == 2 and prediction == 0:
        tags.append("severe_to_normal")
    if truth == 2 and prediction == 1:
        tags.append("severe_to_moderate")
    if truth == 0 and prediction > 0:
        tags.append("normal_to_higher")
    if truth != prediction and item["level"] == "L5-S1":
        tags.append("l5s1_error")
    if len(set(item["head_predictions"])) > 1:
        tags.append("head_disagreement")
    if truth != prediction:
        tags.append("ensemble_error")
    if all(prediction == truth for prediction in item["head_predictions"]):
        tags.append("correct_control")
    return tags


def select_error_targets(candidates: list[dict], count: int = 48, seed: int = 2026) -> list[dict]:
    """Deterministic nonrepresentative quotas; always prefer validation in a pool."""
    if type(count) is not int or count < 1 or type(seed) is not int:
        raise ValueError("Use a positive integer count and an integer seed")
    seen = set()
    ordered = []
    for item in candidates:
        if item["split"] not in {"train", "validation"}:
            raise ValueError("Error review accepts train/validation only; test is forbidden")
        if item["level"] not in LEVELS or item["label"] not in (0, 1, 2):
            raise ValueError("Error review requires fixed target levels and reference labels")
        if _key(item) in seen:
            raise ValueError("Choose one series per study/level before error sampling")
        seen.add(_key(item))
        ordered.append({**item, "sampling_tags": _tags(item)})
    ordered.sort(key=lambda item: _order(item, seed))
    selected, used = [], set()

    def take(pool, maximum, reason):
        added = 0
        for item in pool:
            if len(selected) >= count or added >= maximum:
                break
            if _key(item) in used:
                continue
            selected.append({**item, "sampling_reason": reason})
            used.add(_key(item))
            added += 1

    controls = min(8, count // 6)
    take(
        [item for item in ordered if "correct_control" in item["sampling_tags"]],
        controls,
        "correct_control",
    )
    error_slots = count - len(selected)
    quota_total = sum(quota for _, quota in QUOTAS)
    for category, quota in QUOTAS:
        limit = max(1, int(round(quota * error_slots / quota_total)))
        take([item for item in ordered if category in item["sampling_tags"]], limit, category)
    take(
        [item for item in ordered if set(item["sampling_tags"]) - {"correct_control"}],
        count,
        "additional_error_or_disagreement",
    )
    take(ordered, count, "additional_control_if_error_pools_exhausted")
    selected.sort(
        key=lambda item: hashlib.sha256(
            f"gallery:{seed}:{item['study_id']}:{item['level']}".encode()
        ).hexdigest()
    )
    for item in selected:
        item["case_id"] = (
            "case-"
            + hashlib.sha256(
                f"private-error-review:{seed}:{item['study_id']}:{item['series_id']}:{item['level']}".encode()
            ).hexdigest()[:14]
        )
    return selected


def _predictions(data, head_dir: Path):
    summary = json.loads((head_dir / "summary.json").read_text())
    manifest = json.loads((head_dir / "manifest.json").read_text())
    if (
        [run.get("seed") for run in summary["runs"]] != list(SEEDS)
        or summary.get("test_evaluated") is not False
        or manifest.get("feature_cache_sha256") != data.file_sha256
    ):
        raise ValueError("All three frozen heads must belong to this train/validation cache")
    probabilities, checkpoint_hashes = [], {}
    for run in summary["runs"]:
        path = head_dir / "runs" / f"seed-{run['seed']}" / "head.pt"
        digest = sha256_file(path)
        if digest != run["checkpoint_sha256"]:
            raise ValueError("A selected frozen-head checkpoint changed")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (
            checkpoint.get("seed") != run["seed"]
            or checkpoint.get("feature_cache_sha256") != data.file_sha256
        ):
            raise ValueError("Frozen-head checkpoint seed/source provenance differs")
        probability = predict_head(data.arrays["features"], path)
        if (
            probability.shape != (len(data.arrays["features"]), 3)
            or not np.isfinite(probability).all()
        ):
            raise ValueError("Frozen head returned invalid probabilities")
        probabilities.append(probability)
        checkpoint_hashes[str(run["seed"])] = digest
    values = np.stack(probabilities)
    return values, values.mean(axis=0), checkpoint_hashes


def _safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Cached review path escapes its local data directory")
    return path


def _show(ax, image, title):
    low, high = np.percentile(image, [1, 99])
    ax.imshow(image, cmap="gray", vmin=low, vmax=high if high > low else low + 1)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def _placeholder(ax, text):
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, fontsize=10)
    ax.axis("off")


def _safe_failure(exc):
    known = {
        "invalid_local_identifiers",
        "invalid_source_receipt",
        "source_hash_mismatch",
        "source_nonfinite_or_blank",
        "roi_hash_mismatch",
        "roi_level_mismatch",
        "roi_shape_nonfinite_or_blank",
    }
    return str(exc) if isinstance(exc, ValueError) and str(exc) in known else type(exc).__name__


def _render_case(item, feature_dir, data_root, out, cache_id, plt):
    from matplotlib.patches import Polygon

    record_path = feature_dir / "records" / f"{item['study_id']}_{item['series_id']}.json"
    record = json.loads(record_path.read_text())
    if record["cache_id"] != cache_id:
        raise ValueError("Per-series error record belongs to a different feature cache")
    stored = [target for target in record["targets"] if target["level"] == item["level"]]
    if len(stored) != 1 or any(
        stored[0][name] != item[name]
        for name in ("study_id", "series_id", "split", "status", "label")
    ):
        raise ValueError("Per-series review target does not match the frozen target manifest")
    figure, axes = plt.subplots(
        1, 4, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.1, 1.25, 1.25, 1.25]}
    )
    review = {
        "case_id": item["case_id"],
        "source_display_status": "not_available",
        "source_display_reason": "no_source_record",
        "roi_display_status": "not_available",
        "roi_display_reason": item["reason"],
        "source_hash_verified": False,
        "roi_hash_verified": False,
        "roi_finite": None,
        "roi_blank": None,
        "roi_zero_fraction": None,
        "source_finite": None,
        "source_blank": None,
        "polygons_valid": None,
        "polygons_in_bounds": None,
        "target_pair_in_predicted_chain": None,
        "geometry_check": "not_checked",
        "source_slice_index": None,
    }
    try:
        if not all(re.fullmatch(r"[0-9]+", item[name]) for name in ("study_id", "series_id")):
            raise ValueError("invalid_local_identifiers")
        directory = _safe_path(data_root, f"train_images/{item['study_id']}/{item['series_id']}")
        files = record["source"]["files"]
        names = [entry["name"] for entry in files]
        if (
            not names
            or len(set(names)) != len(names)
            or any(not re.fullmatch(r"[0-9]+\.dcm", name) for name in names)
        ):
            raise ValueError("invalid_source_receipt")
        actual = {p.name for p in directory.glob("*.dcm") if p.is_file()}
        if set(names) != actual or any(
            sha256_file(_safe_path(directory, entry["name"])) != entry["sha256"] for entry in files
        ):
            raise ValueError("source_hash_mismatch")
        review["source_hash_verified"] = True
        scan = load_sagittal_series(directory)
        volume, _, _ = isotropic_inplane(scan.volume, scan.pixel_spacing)
        review["geometry_check"] = "strict_sagittal_loader_passed"
        review["source_finite"] = bool(np.isfinite(volume).all())
        review["source_blank"] = bool(np.ptp(volume) <= 1e-12)
        if not review["source_finite"] or review["source_blank"]:
            raise ValueError("source_nonfinite_or_blank")
        vertices = record.get("vertices", [])
        pair = set(item["level"].split("-"))
        review["target_pair_in_predicted_chain"] = pair.issubset({v["label"] for v in vertices})
        medians = [
            np.median(v["slice_nos"])
            for v in vertices
            if v["label"] in pair and len(v["slice_nos"])
        ]
        index = int(round(float(np.mean(medians)))) if medians else volume.shape[-1] // 2
        index = min(max(index, 0), volume.shape[-1] - 1)
        review["source_slice_index"] = index
        _show(axes[0], volume[:, :, index], f"Source sagittal {index + 1}/{volume.shape[-1]}")
        valid_polygons, in_bounds = True, True
        for vertex in vertices:
            polygon = np.asarray(vertex["average_polygon"])
            if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
                valid_polygons = False
                continue
            in_bounds &= bool(
                (polygon[:, 0] >= 0).all()
                and (polygon[:, 0] < volume.shape[1]).all()
                and (polygon[:, 1] >= 0).all()
                and (polygon[:, 1] < volume.shape[0]).all()
            )
            color = "#44ff99" if vertex["label"] in pair else "#ffd54f"
            axes[0].add_patch(
                Polygon(
                    polygon, closed=True, fill=False, edgecolor=color, linestyle="--", linewidth=1
                )
            )
            axes[0].text(
                float(polygon[:, 0].mean()),
                float(polygon[:, 1].mean()),
                str(vertex["label"]),
                color=color,
                ha="center",
                fontsize=9,
            )
        review["polygons_valid"] = valid_polygons
        review["polygons_in_bounds"] = in_bounds
        review.update(source_display_status="ok", source_display_reason="")
    except (ValueError, OSError, IndexError, KeyError, TypeError) as exc:
        review.update(source_display_status="failed", source_display_reason=_safe_failure(exc))
        _placeholder(axes[0], "Source unavailable / verification failed\n" + type(exc).__name__)
    if item["status"] == "ok":
        try:
            path = _safe_path(feature_dir, item["roi_path"])
            if sha256_file(path) != record["arrays_sha256"]:
                raise ValueError("roi_hash_mismatch")
            review["roi_hash_verified"] = True
            with np.load(path, allow_pickle=False) as arrays:
                index = int(item["roi_index"])
                if str(arrays["levels"][index]) != item["level"]:
                    raise ValueError("roi_level_mismatch")
                roi = arrays["rois"][index]
            review["roi_finite"] = bool(np.isfinite(roi).all())
            review["roi_blank"] = bool(np.ptp(roi) <= 1e-12)
            review["roi_zero_fraction"] = float(np.mean(roi == 0))
            if roi.shape != (9, 112, 224) or not review["roi_finite"] or review["roi_blank"]:
                raise ValueError("roi_shape_nonfinite_or_blank")
            for ax, index in zip(axes[1:], (2, 4, 6)):
                _show(ax, roi[index], f"ROI index {index} ({index + 1}/9)")
            review.update(roi_display_status="ok", roi_display_reason="")
        except (ValueError, OSError, IndexError, KeyError, TypeError) as exc:
            review.update(roi_display_status="failed", roi_display_reason=_safe_failure(exc))
            for ax in axes[1:]:
                _placeholder(ax, "ROI verification failed\n" + type(exc).__name__)
    else:
        for ax in axes[1:]:
            _placeholder(ax, "No assessed ROI\n" + item["reason"])
    predictions = (
        "not assessed"
        if item["status"] != "ok"
        else (
            " | ".join(
                f"{seed}: {CLASS_NAMES[prediction]}"
                for seed, prediction in zip(SEEDS, item["head_predictions"])
            )
            + f" | mean-prob ensemble: {CLASS_NAMES[item['ensemble_prediction']]}"
        )
    )
    figure.suptitle(
        f"{item['case_id']} | {item['split']} | {item['level']} | RSNA reference: {CLASS_NAMES[item['label']]}\n{predictions}\nPurposive review sample. Dashed mean detector boxes; green target pair. No physician review yet.",
        fontsize=10,
    )
    if item["status"] == "ok":
        probability_text = "P [normal/mild, moderate, severe]  " + " | ".join(
            f"{name}: [{', '.join(f'{value:.3f}' for value in values)}]"
            for name, values in zip(
                [str(seed) for seed in SEEDS] + ["ensemble"],
                item["head_probabilities"] + [item["ensemble_probabilities"]],
            )
        )
        figure.text(0.5, 0.015, probability_text, ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.05, 1, 0.82))
    image = item["case_id"] + ".png"
    figure.savefig(out / image, dpi=120)
    plt.close(figure)
    return {
        **review,
        "image": image,
        "record_sha256": sha256_file(record_path),
        "source_signature_sha256": record.get("source", {}).get("sha256"),
    }


def generate_error_review(
    feature_dir: str | Path,
    head_dir: str | Path,
    data_root: str | Path,
    out: str | Path,
    *,
    count: int = 48,
    seed: int = 2026,
) -> dict:
    """Render a local worksheet; blank clinician fields are never auto-approved."""
    feature_dir, head_dir, data_root = map(Path, (feature_dir, head_dir, data_root))
    out = _private_output(Path(out))
    data = load_feature_data(feature_dir)
    probabilities, ensemble, checkpoint_hashes = _predictions(data, head_dir)
    candidates = []
    for target in data.targets:
        if target["label"] is None:
            continue
        index = target["feature_index"]
        candidates.append(
            {
                **target,
                "head_predictions": [int(p[index].argmax()) for p in probabilities]
                if index is not None
                else [],
                "head_probabilities": probabilities[:, index, :].tolist()
                if index is not None
                else [],
                "ensemble_prediction": int(ensemble[index].argmax()) if index is not None else None,
                "ensemble_probabilities": ensemble[index].tolist() if index is not None else [],
            }
        )
    selected = select_error_targets(candidates, count, seed)
    if not selected:
        raise ValueError("No labeled train/validation targets are available for review")
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rendered = []
    for number, item in enumerate(selected, 1):
        rendered.append(
            _render_case(item, feature_dir, data_root, out, data.metadata["cache_id"], plt)
        )
        print(f"Rendered private review case {number}/{len(selected)}", flush=True)
    contacts = []
    for offset in range(0, len(selected), 8):
        subset = rendered[offset : offset + 8]
        fig, axes = plt.subplots(
            math.ceil(len(subset) / 2),
            2,
            figsize=(16, 2.7 * math.ceil(len(subset) / 2)),
            squeeze=False,
        )
        for ax, row in zip(axes.flat, subset):
            ax.imshow(plt.imread(out / row["image"]))
            ax.axis("off")
        for ax in list(axes.flat)[len(subset) :]:
            ax.axis("off")
        name = f"contact-sheet-{offset // 8 + 1:02d}.png"
        fig.tight_layout()
        fig.savefig(out / name, dpi=110)
        plt.close(fig)
        contacts.append(name)
    audit = []
    doctor = []
    cards = []
    for item, technical in zip(selected, rendered):
        row = {
            "case_id": item["case_id"],
            "study_id": item["study_id"],
            "series_id": item["series_id"],
            "split": item["split"],
            "level": item["level"],
            "sampling_reason": item["sampling_reason"],
            "sampling_tags": "|".join(item["sampling_tags"]),
            "reference": CLASS_NAMES[item["label"]],
            **{
                f"prediction_seed_{s}": CLASS_NAMES[item["head_predictions"][i]]
                if item["status"] == "ok"
                else "not_assessed"
                for i, s in enumerate(SEEDS)
            },
            "ensemble_prediction": CLASS_NAMES[item["ensemble_prediction"]]
            if item["status"] == "ok"
            else "not_assessed",
            "roi_status": item["status"],
            "roi_reason": item["reason"],
            **{field: "" for field in CLINICIAN_FIELDS},
        }
        doctor.append(row)
        audit.append({**item, "technical": technical, "clinical_review": "not_performed"})
        probability_rows = ""
        if item["status"] == "ok":
            for label, values in zip(
                [f"seed {s}" for s in SEEDS] + ["mean-prob ensemble"],
                item["head_probabilities"] + [item["ensemble_probabilities"]],
            ):
                probability_rows += (
                    "<tr><th>"
                    + html.escape(label)
                    + "</th>"
                    + "".join(f"<td>{p:.4f}</td>" for p in values)
                    + "</tr>"
                )
        cards.append(
            f'<section id="{item["case_id"]}"><h2>{item["case_id"]} · {html.escape(item["level"])} · {item["split"]}</h2><p>RSNA reference: <strong>{html.escape(row["reference"])}</strong>. Sampling: {html.escape(row["sampling_tags"])}. ROI: {html.escape(item["status"])} {html.escape(item["reason"])}.</p><img src="{technical["image"]}" alt="Local MRI and three ROI slices"><table><tr><th>Model</th><th>normal/mild</th><th>moderate</th><th>severe</th></tr>{probability_rows}</table><p>Doctor answers are blank in doctor_review.csv. No expert validation is implied.</p></section>'
        )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(doctor[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(doctor)
    (out / "doctor_review.csv").write_text(stream.getvalue())
    write_json(out / "cases.json", audit)
    (out / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src \'self\'; style-src \'unsafe-inline\'; base-uri \'none\'; form-action \'none\'"><title>Private spine error review</title><style>body{max-width:1400px;margin:auto;padding:24px;font:16px system-ui;background:#f4f6f8;color:#17232c}section{background:white;padding:18px;margin:24px 0;break-inside:avoid}img{width:100%}table{border-collapse:collapse}td,th{padding:6px 12px;border:1px solid #b8c1c8}a{color:#164d73}</style><h1>Private train/validation error review</h1><p><strong>Purposively selected errors/disagreements with correct controls; this is not a representative accuracy sample. Reference and predictions are visible, so this review is not blinded. No physician review has been performed.</strong></p><p>Everything stays local. No scripts, online assets or form submission. Open <a href="doctor_review.csv">doctor_review.csv</a> in a spreadsheet editor and complete the blank clinician columns. Keep all MRI images and individual records outside Git.</p>'
        + "".join(cards)
        + "</html>"
    )
    summary = {
        "scope": "purposive_train_validation_error_review",
        "representative_accuracy_sample": False,
        "clinical_review": "not_performed",
        "selection_seed": seed,
        "requested_cases": count,
        "selected_cases": len(selected),
        "distinct_study_levels": len({_key(item) for item in selected}),
        "by_split": dict(Counter(item["split"] for item in selected)),
        "by_sampling_reason": dict(Counter(item["sampling_reason"] for item in selected)),
        "by_level_and_reference": dict(
            Counter(f"{item['level']}:{CLASS_NAMES[item['label']]}" for item in selected)
        ),
        "technical": {
            name: dict(Counter(str(row[name]) for row in rendered))
            for name in (
                "source_display_status",
                "source_display_reason",
                "roi_display_status",
                "roi_display_reason",
                "source_hash_verified",
                "roi_hash_verified",
                "roi_finite",
                "roi_blank",
                "source_finite",
                "source_blank",
                "polygons_valid",
                "polygons_in_bounds",
                "target_pair_in_predicted_chain",
                "geometry_check",
            )
        },
        "missing_roi_reasons": dict(
            Counter(item["reason"] for item in selected if item["status"] != "ok")
        ),
        "feature_cache_sha256": data.file_sha256,
        "checkpoint_sha256": checkpoint_hashes,
        "contact_sheets": contacts,
        "png_files": len(rendered) + len(contacts),
        "test_accessed": False,
        "clinician_fields_prefilled": False,
    }
    write_json(out / "summary.json", summary)
    (out / "README.txt").write_text(
        "PRIVATE LOCAL REVIEW — contains MRI pixels and individual study identifiers.\nOpen index.html locally; no web server or network is needed.\nFill doctor_review.csv in a spreadsheet editor. The level_correct, roi_adequate, reference_agreement, suggested_grade and comment columns are intentionally blank.\nReference categories and predictions are visible: the review is not blinded. The sample intentionally overrepresents errors, rare severe cases, L5-S1 errors and head disagreement; do not use it to estimate accuracy. Correct controls are included.\nSource sagittal views use dashed mean detector polygons, not expert contours. Green boxes mark the requested pair. ROI indices 2/4/6 correspond to slices 3/5/7 of 9. Rendering failures remain visible rather than being replaced by normal.\nNo physician has reviewed these cases yet. Keep this directory local and outside Git/docs. Regenerate into a new output directory to preserve answers.\n"
    )
    write_json(
        out / "files_sha256.json",
        {path.name: sha256_file(path) for path in sorted(out.iterdir()) if path.is_file()},
    )
    return summary


def _development_predictions(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    indexed = {}
    for row in rows:
        key = row["study_id"], row["series_id"], row["level"]
        if row["split"] != "validation" or key in indexed:
            raise ValueError("Development comparison requires unique validation-only keys; no test")
        if any(not re.fullmatch(r"[0-9]+", value) for value in key[:2]) or key[2] not in LEVELS:
            raise ValueError("Invalid development prediction key")
        label = None if row["label"] == "" else int(row["label"])
        if label is not None and label not in (0, 1, 2):
            raise ValueError("Invalid development reference label")
        if row["status"] == "ok":
            probabilities = np.asarray([float(row[f"p_{name}"]) for name in CLASS_NAMES])
            prediction = int(row["prediction"])
            if (
                not np.isfinite(probabilities).all()
                or np.any(probabilities < 0)
                or np.any(probabilities > 1)
                or not np.isclose(probabilities.sum(), 1, atol=1e-6)
                or prediction != int(probabilities.argmax())
            ):
                raise ValueError("Invalid development prediction/probability vector")
        else:
            if row["prediction"] or any(row[f"p_{name}"] for name in CLASS_NAMES):
                raise ValueError("An unassessed development target cannot carry predictions")
            prediction, probabilities = None, None
        indexed[key] = {
            **row,
            "label": label,
            "prediction": prediction,
            "probabilities": probabilities.tolist() if probabilities is not None else [],
        }
    if not indexed:
        raise ValueError("Development predictions are empty")
    return indexed


def attach_development_comparison(review_dir: str | Path, experiment_dir: str | Path) -> dict:
    """Attach new development predictions to the original, unchanged review sample.

    No inference, resampling, pixel rewrite or edits of doctor_review.csv occur.
    Counts describe only the deliberately selected cases, never accuracy.
    """
    review_dir, experiment_dir = Path(review_dir).resolve(), Path(experiment_dir).resolve()
    allowed = (_project_root() / "reports" / "runs").resolve()
    if (
        review_dir == allowed
        or not review_dir.is_relative_to(allowed)
        or not allowed.is_relative_to(_project_root().resolve())
        or "reports/runs/" not in (_project_root() / ".gitignore").read_text().splitlines()
    ):
        raise ValueError("Development comparison must stay inside ignored reports/runs")
    for name in (
        "comparison.html",
        "comparison.csv",
        "comparison-summary.json",
        "comparison-inputs.json",
    ):
        if (review_dir / name).exists():
            raise ValueError(
                "A comparison attachment already exists; preserve this review revision"
            )
    original_hashes = json.loads((review_dir / "files_sha256.json").read_text())
    if sha256_file(review_dir / "cases.json") != original_hashes["cases.json"]:
        raise ValueError("The original gallery case selection changed")
    if sha256_file(review_dir / "summary.json") != original_hashes["summary.json"]:
        raise ValueError("The original gallery aggregate changed")
    cases = json.loads((review_dir / "cases.json").read_text())
    old_summary = json.loads((review_dir / "summary.json").read_text())
    if len(cases) != old_summary["selected_cases"]:
        raise ValueError("The saved gallery selection count changed")
    seen, png_hashes = set(), {}
    for case in cases:
        key = case["study_id"], case["series_id"], case["level"]
        if key in seen or case["split"] not in {"train", "validation"}:
            raise ValueError("The original review has duplicate or forbidden split keys")
        seen.add(key)
        if not re.fullmatch(r"case-[0-9a-f]{14}", case["case_id"]):
            raise ValueError("The saved gallery has an unsafe case identifier")
        image = case["technical"]["image"]
        if image != case["case_id"] + ".png":
            raise ValueError("An existing review image has an unsafe filename")
        digest = sha256_file(_safe_path(review_dir, image))
        if digest != original_hashes.get(image):
            raise ValueError("An original gallery image changed")
        png_hashes[image] = digest
    summary_path = experiment_dir / "summary.json"
    experiment = json.loads(summary_path.read_text())
    if (
        experiment.get("test_evaluated") is not False
        or experiment["configuration"]["feature_cache_sha256"]
        != old_summary["feature_cache_sha256"]
    ):
        raise ValueError(
            "Development comparison must use the same train/validation cache and no test"
        )
    models, prediction_hashes = {}, {}
    for variant in ("linear", "mlp", "coral"):
        relative = f"validation/{variant}/ensemble/predictions.csv"
        path = _safe_path(experiment_dir, relative)
        digest = sha256_file(path)
        if (
            digest
            != experiment["candidates"][variant]["validation"]["ensemble"]["predictions_sha256"]
        ):
            raise ValueError("Development predictions do not match their experiment summary SHA")
        models[variant] = _development_predictions(path)
        prediction_hashes[relative] = digest
    reference = models["linear"]
    for table in models.values():
        if table.keys() != reference.keys() or any(
            (row["label"], row["status"]) != (reference[key]["label"], reference[key]["status"])
            for key, row in table.items()
        ):
            raise ValueError(
                "Development models disagree about keys, reference labels or ROI status"
            )
    for case in cases:
        if case["split"] == "validation":
            key = case["study_id"], case["series_id"], case["level"]
            if key not in reference or (reference[key]["label"], reference[key]["status"]) != (
                case["label"],
                case["status"],
            ):
                raise ValueError(
                    "Gallery and development predictions have wrong keys or mixed reference labels/status"
                )
    doctor_hash = sha256_file(review_dir / "doctor_review.csv")
    ru = {"normal_mild": "Норма / лёгкий", "moderate": "Умеренный", "severe": "Тяжёлый"}
    model_names = {
        "old_ensemble": "Старые 3 головы: ансамбль",
        "linear": "v2 Linear: ансамбль",
        "mlp": "v2 MLP: ансамбль",
        "coral": "v2 CORAL: ансамбль",
    }

    def grade(value):
        return "Не оценено" if value is None else ru[CLASS_NAMES[value]]

    rows, cards = [], []
    outcomes = {variant: Counter() for variant in models}
    for case in cases:
        key = case["study_id"], case["series_id"], case["level"]
        old = case["ensemble_prediction"]
        old_probs = case["ensemble_probabilities"]
        if case["status"] == "ok" and (
            old not in (0, 1, 2)
            or len(old_probs) != 3
            or not np.isfinite(old_probs).all()
            or not np.isclose(sum(old_probs), 1, atol=1e-6)
            or old != int(np.argmax(old_probs))
        ):
            raise ValueError("The saved old ensemble prediction is invalid")
        row = {
            "case_id": case["case_id"],
            "study_id": case["study_id"],
            "series_id": case["series_id"],
            "split": case["split"],
            "level": case["level"],
            "sampling_reason": case["sampling_reason"],
            "reference": case["label"],
            "reference_name": grade(case["label"]),
            "roi_status": case["status"],
            "old_ensemble_prediction": old if old is not None else "",
            **{
                f"old_ensemble_p_{name}": old_probs[i] if old_probs else ""
                for i, name in enumerate(CLASS_NAMES)
            },
        }
        table_rows = [(model_names["old_ensemble"], grade(old), old_probs)]
        for variant, table in models.items():
            if case["split"] != "validation":
                prediction, probabilities, outcome = None, [], "not_evaluated"
            elif case["status"] != "ok":
                prediction, probabilities, outcome = None, [], "not_assessed"
            else:
                prediction, probabilities = table[key]["prediction"], table[key]["probabilities"]
                old_correct, new_correct = old == case["label"], prediction == case["label"]
                outcome = (
                    "corrected"
                    if not old_correct and new_correct
                    else "worsened"
                    if old_correct and not new_correct
                    else "same_correct"
                    if old_correct
                    else "same_wrong"
                    if old == prediction
                    else "changed_wrong"
                )
            outcomes[variant][outcome] += 1
            row.update(
                {
                    f"{variant}_prediction": prediction if prediction is not None else "",
                    f"{variant}_comparison": outcome,
                    **{
                        f"{variant}_p_{name}": probabilities[i] if probabilities else ""
                        for i, name in enumerate(CLASS_NAMES)
                    },
                }
            )
            table_rows.append(
                (
                    model_names[variant],
                    "Нет validation-оценки" if outcome == "not_evaluated" else grade(prediction),
                    probabilities,
                )
            )
        rows.append(row)
        table_html = "".join(
            "<tr><th>"
            + html.escape(name)
            + "</th><td>"
            + html.escape(prediction)
            + "</td>"
            + "".join(f"<td>{value:.4f}</td>" for value in probability)
            + ("<td>—</td>" * 3 if not probability else "")
            + "</tr>"
            for name, prediction, probability in table_rows
        )
        image = case["technical"]["image"]
        cards.append(
            f'<section><h2>{case["case_id"]} · {html.escape(case["level"])} · {case["split"]}</h2><p>Метка RSNA: <strong>{html.escape(grade(case["label"]))}</strong>. Причина отбора: {html.escape(case["sampling_reason"])}. ROI: {html.escape(case["status"])}.</p><table><tr><th>Модель</th><th>Оценка</th><th>P(норма/лёгкий)</th><th>P(умеренный)</th><th>P(тяжёлый)</th></tr>{table_html}</table><p>Новые оценки приведены в таблице выше. На неизменённой картинке остались подписи старых голов.</p><a href="{image}"><img src="{image}" alt="Сохранённый исходный случай и ROI"></a></section>'
        )
    aggregate = {
        "scope": "purposive_unblinded_development_error_analysis_only",
        "representative_accuracy_sample": False,
        "clinical_review": "not_performed",
        "n_fixed_cases": len(cases),
        "n_matched_validation": sum(c["split"] == "validation" for c in cases),
        "n_paired_assessed_validation": sum(
            c["split"] == "validation" and c["status"] == "ok" for c in cases
        ),
        "n_train_not_evaluated": sum(c["split"] == "train" for c in cases),
        "by_model": {
            variant: {
                name: int(counts[name])
                for name in (
                    "corrected",
                    "worsened",
                    "same_correct",
                    "same_wrong",
                    "changed_wrong",
                    "not_assessed",
                    "not_evaluated",
                )
            }
            for variant, counts in outcomes.items()
        },
        "comparison_definition": "Corrected/worsened mean agreement changes with the unchanged RSNA reference, not physician confirmation. same_correct/same_wrong preserve prediction; changed_wrong changes category but remains wrong. Missing ROI and train cases are counted separately.",
        "experiment_summary_sha256": sha256_file(summary_path),
        "prediction_sha256": prediction_hashes,
        "images_reselected_or_modified": False,
        "doctor_csv_modified": False,
        "test_accessed": False,
    }
    for counts in aggregate["by_model"].values():
        counts["same_prediction"] = counts["same_correct"] + counts["same_wrong"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    (review_dir / "comparison.csv").write_text(stream.getvalue())
    write_json(review_dir / "comparison-summary.json", aggregate)
    write_json(
        review_dir / "comparison-inputs.json",
        {
            "original_cases_sha256": original_hashes["cases.json"],
            "unchanged_images_sha256": png_hashes,
            "unchanged_doctor_csv_sha256": doctor_hash,
            "experiment_summary_sha256": aggregate["experiment_summary_sha256"],
            "prediction_sha256": prediction_hashes,
        },
    )
    (review_dir / "comparison.html").write_text(
        '<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src \'self\'; style-src \'unsafe-inline\'; base-uri \'none\'; form-action \'none\'"><title>Сравнение моделей на фиксированных случаях</title><style>body{max-width:1400px;margin:auto;padding:24px;font:16px system-ui;background:#f4f6f8}section{background:white;padding:20px;margin:24px 0}img{width:100%}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #b8c1c8}</style><h1>Новые оценки на тех же 48 случаях</h1><p><strong>Это намеренно отобранные ошибки и контрольные случаи train/validation. Выборка не представляет обычную точность модели. Проверка не слепая, врачебной проверки пока нет.</strong></p><p>Старый ансамбль сравнивается с новыми ансамблями Linear, MLP и CORAL. Картинки и отбор не изменены: на картинках сохранены старые подписи, новые оценки находятся над ними. Для train-случая новых validation-оценок нет. Изменение совпадения с меткой RSNA не подтверждает правильность диагноза.</p><p><a href="index.html">Исходная галерея и врачебная форма</a> · <a href="comparison.csv">Таблица сравнения</a>. doctor_review.csv не изменён. Никакой отправки данных в сеть нет.</p>'
        + "".join(cards)
        + "</html>"
    )
    index_path = review_dir / "index.html"
    index = index_path.read_text()
    link = '<p><a href="comparison.html">Сравнить новые ансамбли MLP/CORAL/Linear на тех же случаях</a> — картинки и врачебная форма не менялись.</p>'
    index_path.write_text(index.replace("<h1>", link + "<h1>", 1))
    if sha256_file(review_dir / "doctor_review.csv") != doctor_hash:
        raise RuntimeError("The physician worksheet unexpectedly changed")
    if any(sha256_file(review_dir / name) != digest for name, digest in png_hashes.items()):
        raise RuntimeError("An existing gallery image unexpectedly changed")
    for name in (
        "comparison.html",
        "comparison.csv",
        "comparison-summary.json",
        "comparison-inputs.json",
        "index.html",
    ):
        original_hashes[name] = sha256_file(review_dir / name)
    write_json(review_dir / "files_sha256.json", original_hashes)
    return aggregate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("features", "heads", "data-root", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    try:
        summary = generate_error_review(
            args.features, args.heads, args.data_root, args.out, count=args.count, seed=args.seed
        )
    except (ValueError, OSError) as exc:
        parser.exit(
            2,
            f"Private error review failed ({type(exc).__name__}); no clinical approval was recorded.\n",
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
