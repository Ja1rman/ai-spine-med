"""Post-hoc diagnostics on fixed models: anatomy weights and pooling interventions.

Reference-based windows are explanatory oracle interventions, never deployable
models or validation results. Existing predictions/weights remain unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from .assets import sha256_file, write_json
from .axial_attention_diagnosis import ENTRIES
from .axial_localization_audit import LEVELS, _distances
from .axial_transfer import _sealed_models

CONTEXTS = {
    "original": "probabilities",
    "uniform_attention": "uniform_attention_probabilities",
    "swapped_level_identity": "swapped_level_probabilities",
    "swapped_attention_heads": "swapped_attention_probabilities",
}


def summarize_targets(rows):
    """Only known references enter classification metrics; preserve coverage."""
    known = [r for r in rows if r["label"] is not None]
    assessed = [r for r in known if r["probabilities"] is not None]
    targets = np.array([r["label"] for r in assessed], dtype=int)
    predictions = np.array([np.argmax(r["probabilities"]) for r in assessed], dtype=int)
    severe = [r for r in known if r["label"] == 2]
    return {
        "n_requested": len(rows),
        "n_labeled": len(known),
        "n_assessed_labeled": len(assessed),
        "n_missing_reference": len(rows) - len(known),
        "n_missing_prediction": sum(r["probabilities"] is None for r in rows),
        "macro_f1": float(
            f1_score(targets, predictions, labels=[0, 1, 2], average="macro", zero_division=0)
        )
        if assessed
        else None,
        "confusion_matrix": confusion_matrix(targets, predictions, labels=[0, 1, 2]).tolist(),
        "prediction_counts_known_references": np.bincount(predictions, minlength=3).tolist(),
        "severe_total": len(severe),
        "severe_found": sum(
            r["probabilities"] is not None and int(np.argmax(r["probabilities"])) == 2
            for r in severe
        ),
        "mean_p_normal_in_true_severe": float(
            np.mean([r["probabilities"][0] for r in severe if r["probabilities"] is not None])
        )
        if any(r["probabilities"] is not None for r in severe)
        else None,
    }


def oracle_window_probabilities(model, features, weights, masks):
    """Restrict each level to its own reference window, retaining attention ratios.

    A missing window remains unavailable. No inferred normal or reference grade
    enters this computation. Windows require expert anatomical anchors.
    """
    features, weights = np.asarray(features), np.asarray(weights)
    masks = np.asarray(masks, dtype=bool)
    if (
        features.shape != (weights.shape[1], 512)
        or weights.shape != masks.shape
        or len(weights) != 2
    ):
        raise ValueError("Mismatched oracle feature/attention/window arrays")
    head = model.head
    if model.training or head.training:
        raise ValueError("Only fixed evaluation models may be diagnosed")
    with torch.inference_mode():
        x = torch.from_numpy(features.copy()).float()
        normalized = (x - head.axial_mean) / head.axial_std
        results = []
        for level_index, mask in enumerate(masks):
            restricted = weights[level_index] * mask
            total = float(restricted.sum())
            if not mask.any() or total <= 0:
                results.append(None)
                continue
            w = torch.from_numpy((restricted / total).astype(np.float32))
            pooled = w @ normalized
            content = torch.cat((pooled, head.level_one_hot[level_index]), dim=-1)
            results.append(head.head(content).softmax(-1).numpy().tolist())
    return results


def diagnose_interventions(attention_dir, anchors_path, run_dir, out):
    attention_dir, anchors_path, run_dir, out = map(
        Path, (attention_dir, anchors_path, run_dir, out)
    )
    torch.set_num_threads(4)
    index = json.loads((attention_dir / "index.json").read_text())
    anchors = json.loads(anchors_path.read_text())
    if index.get("selection_sha256") != sha256_file(run_dir / "selection.json"):
        raise ValueError("Attention and diagnostic heads have different checkpoint selections")
    if (
        index["entries"] != list(ENTRIES)
        or index["index_sha256"] != anchors["activation_index_sha256"]
    ):
        raise ValueError("Attention and anatomical references have different activation provenance")
    anchor_map = {(r["dataset"], r["study_id"]): r for r in anchors["records"]}
    if set(anchor_map) != {(r["dataset"], r["study_id"]) for r in index["records"]}:
        raise ValueError("Anatomical and attention cohorts differ")
    _, models = _sealed_models(run_dir, "cpu")
    all_rows = []
    for number, record in enumerate(index["records"]):
        anchor = anchor_map[record["dataset"], record["study_id"]]
        path = attention_dir / record["arrays_path"]
        if sha256_file(path) != record["arrays_sha256"]:
            raise ValueError("Attention explanations changed")
        with np.load(path, allow_pickle=False) as archive:
            arrays = dict(archive)
        np.testing.assert_array_equal(
            arrays["positions"],
            np.array([s["positions"] for s in anchor["selected_slices"]], dtype=np.float32),
        )
        masks = []
        for level in LEVELS:
            references = [
                a
                for a in anchor["anchors"]
                if a["level"] == level
                and a["kind"] != "rsna_sagittal_central_same_frame_projection"
            ]
            masks.append(_distances(anchor["selected_slices"], references) <= 10)
        masks = np.asarray(masks)
        oracle, opposite = [], []
        for i, name in enumerate(ENTRIES):
            phase, _, seed = name.partition("_seed_")
            model = models[phase, int(seed)]
            oracle.append(
                oracle_window_probabilities(
                    model, arrays["features"][i], arrays["attention"][i], masks
                )
            )
            opposite.append(
                oracle_window_probabilities(
                    model, arrays["features"][i], arrays["attention"][i], masks[::-1]
                )
            )
        for phase_index, phase in enumerate(("before", "after")):
            selection = slice(phase_index * 3, phase_index * 3 + 3)
            probabilities = {
                context: arrays[field][selection].astype(np.float64).mean(0)
                for context, field in CONTEXTS.items()
            }
            for name, results in (
                ("oracle_own_level_10mm", oracle),
                ("oracle_other_level_10mm", opposite),
            ):
                probabilities[name] = [
                    None
                    if any(r[level] is None for r in results[selection])
                    else np.mean([r[level] for r in results[selection]], axis=0).tolist()
                    for level in (0, 1)
                ]
            for context, values in probabilities.items():
                for level_index, level in enumerate(LEVELS):
                    p = values[level_index]
                    all_rows.append(
                        {
                            "dataset": record["dataset"],
                            "study_id": record["study_id"],
                            "level": level,
                            "label": anchor["labels"][level_index],
                            "phase": phase,
                            "context": context,
                            "n_window_slices": int(masks[level_index].sum()),
                            "probabilities": None if p is None else np.asarray(p).tolist(),
                        }
                    )
        if (number + 1) % 50 == 0:
            print(f"Intervention audit {number + 1}/{len(index['records'])}", flush=True)
    summary = {
        "schema_version": 1,
        "scope": "Post-hoc explanatory interventions on previously inspected TEST; not new validation or a usable oracle model",
        "training_performed": False,
        "selection_performed": False,
        "attention_index_sha256": sha256_file(attention_dir / "index.json"),
        "anchors_sha256": sha256_file(anchors_path),
        "datasets": {},
    }
    for dataset in ("aisslab", "rsna"):
        summary["datasets"][dataset] = {}
        for phase in ("before", "after"):
            phase_rows = [r for r in all_rows if r["dataset"] == dataset and r["phase"] == phase]
            summary["datasets"][dataset][phase] = {}
            for context in sorted({r["context"] for r in phase_rows}):
                rows = [r for r in phase_rows if r["context"] == context]
                summary["datasets"][dataset][phase][context] = {
                    "combined": summarize_targets(rows),
                    "by_level": {
                        level: summarize_targets([r for r in rows if r["level"] == level])
                        for level in LEVELS
                    },
                }
            available = {
                (r["study_id"], r["level"])
                for r in phase_rows
                if r["context"] == "oracle_own_level_10mm" and r["probabilities"] is not None
            }
            matched = [
                r
                for r in phase_rows
                if r["context"] == "original" and (r["study_id"], r["level"]) in available
            ]
            summary["datasets"][dataset][phase]["original_on_oracle_coverage"] = {
                "combined": summarize_targets(matched),
                "by_level": {
                    level: summarize_targets([r for r in matched if r["level"] == level])
                    for level in LEVELS
                },
            }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "intervention-targets.json", {"records": all_rows})
    write_json(out / "interventions.json", summary)
    return summary


def export_public_summary(private_dir, destination):
    """Explicit aggregate whitelist: never copy private target/anchor inventories."""
    root, destination = Path(private_dir), Path(destination)
    labels = json.loads((root / "label-audit.json").read_text())
    localization = json.loads(
        (root / "localization/attention-localization-summary.json").read_text()
    )
    geometry = json.loads((root / "localization/anatomical-order-summary.json").read_text())
    interventions = json.loads((root / "interventions.json").read_text())
    public = {
        "schema_version": 1,
        "scope": "Post-hoc diagnostic audit of fixed models and previously inspected TEST. No training, selection or revised primary scores.",
        "labels": {
            "n_xml": labels["counts"]["xml"],
            "n_target_annotations": labels["counts"]["target_annotations"],
            "target_import_mismatches": len(labels["import_integrity"]["record_label_mismatches"]),
            "target_shape_mismatches": len(
                labels["import_integrity"]["xml_dicom_shape_mismatches"]
            ),
            "ap_by_source_class_all_levels": labels["ap_mm_grade_correspondence"]["all_levels"],
            "empirical_ap_rule_matches": labels["ap_mm_grade_correspondence"]["matches"],
            "boundary_counts": labels["ap_mm_grade_correspondence"]["exact_boundary_counts"],
            "all_levels_ordinal_pairs": labels["all_five_levels_internal_order"]["ordered_pairs"],
            "all_levels_ordinal_inversions": len(
                labels["all_five_levels_internal_order"]["reversed_or_equal_pairs"]
            ),
            "target_ap_zero_count": len(labels["ap_zero_cases_private"]),
            "zero_exclusion_sensitivity": {
                phase: values["metrics"] for phase, values in labels["sensitivity_only"].items()
            },
            "clinical_scale_equivalence_verified": False,
        },
        "anatomical_order": geometry,
        "localization_groups": localization["groups"],
        "interventions": interventions["datasets"],
        "limitations": [
            "XML slices are representative sparse anchors. Proximity is not clinical localization accuracy.",
            "RSNA axial subarticular slices provide approximate level anchors, not central stenosis grades.",
            "Unmatched FrameOfReferenceUIDs do not establish shared sagittal/axial registration.",
            "A mean attention map is a descriptive average; probability ensembles still average each model's predictions.",
            "Oracle windows use reference locations only for explanation; they are unavailable at ordinary inference and not a deployable improvement.",
            "Wrong-level windows can cover different target subsets. Their metrics are diagnostic, not a fair new comparison.",
            "The observed AP thresholds are empirical XML correspondence, not source-proven grade derivation or a validated crosswalk.",
        ],
    }
    write_json(destination, public)
    return public


def draw_public_figures(public, out):
    """Scientific result figures using aggregate counts only, without MRI pixels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    colors = ("#345c83", "#c96745")
    groups = public["localization_groups"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, dataset in zip(axes, ("aisslab", "rsna")):
        for phase_index, phase in enumerate(("before_mean", "after_mean")):
            rows = [
                next(
                    r
                    for r in groups
                    if r["dataset"] == dataset and r["phase"] == phase and r["level"] == level
                )
                for level in LEVELS
            ]
            x = np.arange(2) + (phase_index - 0.5) * 0.32
            y = [100 * r["top1_near_10mm_count"] / r["n_anchored_targets"] for r in rows]
            bars = ax.bar(x, y, 0.30, color=colors[phase_index], label=("До", "После")[phase_index])
            ax.bar_label(
                bars,
                labels=[f"{r['top1_near_10mm_count']}/{r['n_anchored_targets']}" for r in rows],
                padding=4,
                fontsize=10,
            )
        ax.set(
            xticks=np.arange(2),
            xticklabels=LEVELS,
            ylim=(0, 40),
            ylabel="Пик среднего внимания в ±10 мм, %",
            title=dataset.upper(),
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    fig.suptitle(
        "Близость пика внимания к анатомическому ориентиру\nРазреженные ориентиры; это не клиническая точность локализации",
        fontsize=13,
    )
    fig.savefig(out / "axial_diagnosis_localization.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, dataset in zip(axes, ("aisslab", "rsna")):
        for phase_index, phase in enumerate(("before_mean", "after_mean")):
            rows = [
                next(
                    r
                    for r in groups
                    if r["dataset"] == dataset and r["phase"] == phase and r["level"] == level
                )
                for level in LEVELS
            ]
            bars = ax.bar(
                np.arange(2) + (phase_index - 0.5) * 0.32,
                [100 * r["mass_near_10mm_mean"] for r in rows],
                0.30,
                color=colors[phase_index],
                label=("До", "После")[phase_index],
            )
            ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=10)
        reference = [100 * r["uniform_mass_near_10mm_mean"] for r in rows]
        ax.scatter(
            np.arange(2),
            reference,
            marker="_",
            s=1100,
            linewidths=2,
            color="#222",
            label="Равномерные веса",
        )
        ax.set(
            xticks=np.arange(2),
            xticklabels=LEVELS,
            ylim=(0, 30),
            ylabel="Средняя масса внимания в ±10 мм, %",
            title=dataset.upper(),
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=10, loc="upper left")
    fig.suptitle(
        "Доля внимания около нужного уровня\nСреднее по моделям и пациентам; чёрная отметка — равномерное внимание",
        fontsize=13,
    )
    fig.savefig(out / "axial_diagnosis_mass.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    settings = ("original", "uniform_attention", "oracle_own_level_10mm")
    names = ("Исходное\nвнимание", "Равномерное\nвнимание", "Окно по\nэталонному уровню")
    for ax, dataset in zip(axes, ("aisslab", "rsna")):
        for phase_index, phase in enumerate(("before", "after")):
            rows = [
                public["interventions"][dataset][phase][context]["by_level"]["L5-S1"]
                for context in settings
            ]
            values = [100 * r["severe_found"] / r["severe_total"] for r in rows]
            bars = ax.bar(
                np.arange(3) + (phase_index - 0.5) * 0.32,
                values,
                0.30,
                color=colors[phase_index],
                label=("До", "После")[phase_index],
            )
            ax.bar_label(
                bars,
                labels=[f"{r['severe_found']}/{r['severe_total']}" for r in rows],
                padding=4,
                fontsize=10,
            )
        ax.set(
            xticks=np.arange(3),
            xticklabels=names,
            ylim=(0, 48),
            ylabel="Найдены тяжёлые L5–S1, %",
            title=dataset.upper(),
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    fig.suptitle(
        "Замена внимания сама по себе не решает задачу L5–S1\nФиксированные веса; окно по эталону — диагностический опыт, не новая модель",
        fontsize=13,
    )
    fig.savefig(out / "axial_diagnosis_interventions.png", dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attention", type=Path, default=Path("reports/runs/axial-diagnosis-v5/attention")
    )
    parser.add_argument(
        "--anchors",
        type=Path,
        default=Path("reports/runs/axial-diagnosis-v5/localization/anchors.json"),
    )
    parser.add_argument("--run", type=Path, default=Path("reports/runs/axial-transfer-v4"))
    parser.add_argument("--out", type=Path, default=Path("reports/runs/axial-diagnosis-v5"))
    parser.add_argument("--public-summary", type=Path)
    parser.add_argument("--figures", type=Path, default=Path("docs/images"))
    args = parser.parse_args(argv)
    if args.public_summary:
        summary = export_public_summary(args.out, args.public_summary)
        draw_public_figures(summary, args.figures)
    else:
        diagnose_interventions(args.attention, args.anchors, args.run, args.out)


if __name__ == "__main__":
    main()
