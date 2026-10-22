"""Evaluate the fixed v9 axial classifiers against the historical sagittal TEST.

No training or model selection occurs. TEST labels and sagittal prediction
tables are joined only after both axial image-inference passes are complete.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from . import axial_features as af
from . import localization_severity_validation as v8
from . import localized_stenosis as severity
from .assets import sha256_file, write_json

PROTOCOL = Path("configs/localization-test-v10.json")
RUN = Path("reports/runs/localization-test-v10")
V9 = Path("reports/runs/localization-finetuning-v9")
TEST_INDEX = Path("artifacts/axial-localizer-test-v6/index.json")
SAGITTAL = Path("reports/runs/test-v1")


def verify_inputs(protocol_path):
    config = json.loads(Path(protocol_path).read_text())
    if (
        config.get("phase") != "test"
        or config.get("models") != ["xml_only", "mixed", "sagittal"]
        or config.get("seeds") != list(v8.SEEDS)
    ):
        raise ValueError("Require the fixed three-model TEST comparison")
    observed = {name: sha256_file(Path(name)) for name in config["frozen_files"]}
    if observed != config["frozen_files"]:
        raise ValueError("A frozen model, source, TEST input or sagittal result changed")
    return config, observed


def load_fixed_classifiers(device):
    seal = json.loads((V9 / "selection.json").read_text())
    summary = json.loads((V9 / "summary.json").read_text())
    if (
        seal.get("frozen") is not True
        or seal.get("status") != "all_six_fixed_before_AISSLab_prediction"
        or summary.get("phase") != "validation"
        or summary.get("test_evaluated") is not False
    ):
        raise ValueError("Require the unchanged completed validation-selected v9 classifiers")
    result = {}
    for arm in v8.ARMS:
        result[arm] = []
        if set(seal["checkpoints"][arm]) != {str(s) for s in v8.SEEDS}:
            raise ValueError("Missing a prespecified classifier seed")
        for seed in v8.SEEDS:
            path = V9 / "arms" / arm / f"seed_{seed}" / "model.pt"
            if sha256_file(path) != seal["checkpoints"][arm][str(seed)]:
                raise ValueError("A selected v9 checkpoint changed")
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if (
                checkpoint.get("family") != "paired_localization_finetuning_v9"
                or checkpoint.get("seed") != seed
                or checkpoint.get("classes") != list(severity.CLASSES)
                or checkpoint.get("levels") != list(severity.LEVELS)
                or checkpoint.get("selected_epoch")
                != summary["arms"][arm]["training"]["selected_epochs"][str(seed)]
                or checkpoint.get("provenance") != seal["provenance"]
            ):
                raise ValueError("Classifier classes, levels or pre-TEST selection differ")
            state = checkpoint["state_dict"]
            model = severity.LocalizedSeverityHead(state["mean"], state["std"])
            model.load_state_dict(state, strict=True)
            result[arm].append(model.eval().requires_grad_(False).to(device))
    return result


def test_records(index, fitting_index):
    configuration, entries = index["configuration"], index["records"]
    if (
        configuration.get("phase") != "evaluation_image_inputs"
        or configuration.get("reference_fields_in_inputs") is not False
        or configuration.get("input_fields") != ["features", "positions"]
        or any(r["split"] != "test" for r in entries)
    ):
        raise ValueError("Require the original image-only TEST inputs")
    rows = [r for r in entries if r["dataset"] == "rsna"]
    keys = [(r["dataset"], r["study_id"]) for r in rows]
    fitting = {(r["dataset"], r["study_id"]) for r in fitting_index["records"]}
    if not rows or len(keys) != len(set(keys)) or set(keys) & fitting:
        raise ValueError("TEST patients duplicate or overlap fitting/validation patients")
    forbidden = {"anchors", "label", "labels", "relevance", "center_xy", "level_mask"}
    if any(forbidden.intersection(r) for r in rows):
        raise ValueError("Reference fields cannot enter TEST inference records")
    return rows


def test_labels(rows, path):
    if any(r["dataset"] != "rsna" or r["split"] != "test" for r in rows):
        raise ValueError("Reference join accepts RSNA TEST only")
    patients = {r["study_id"] for r in rows}
    mapping = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2, "": None}
    labels = {}
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["study_id"] not in patients:
                continue
            for level, column in zip(
                severity.LEVELS, ("spinal_canal_stenosis_l4_l5", "spinal_canal_stenosis_l5_s1")
            ):
                key = "rsna", row["study_id"], level
                if key in labels or row[column] not in mapping:
                    raise ValueError("Invalid or duplicate TEST reference grade")
                labels[key] = mapping[row[column]]
    if labels.keys() != {("rsna", p, level) for p in patients for level in severity.LEVELS}:
        raise ValueError("TEST reference identity missing; absent grades must be explicit")
    return labels


def _write(path, rows):
    normalized = [
        {
            k: ("known" if row["label"] is not None else "missing")
            if k == "reference_status"
            else row[k]
            for k in severity.PREDICTION_FIELDS
        }
        for row in rows
    ]
    severity._write_predictions(path, normalized)


def evaluate_test(protocol_path=PROTOCOL, out=RUN, device="cpu"):
    protocol_path, out = Path(protocol_path), Path(out)
    config, files = verify_inputs(protocol_path)
    device = af._device(device)
    torch.set_num_threads(4)
    classifiers = load_fixed_classifiers(device)
    localizers, center = v8.load_ensembles(device)
    provenance = {
        "frozen_files": files,
        "protocol_sha256": sha256_file(protocol_path),
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    manifest = {
        "all_model_choices_fixed_before_test_inference": True,
        "provenance": provenance,
        "config": config,
    }
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists() and json.loads(
        (out / "manifest.json").read_text()
    ) != manifest:
        raise ValueError("Changed resumable TEST comparison; use a new directory")
    if (out / "summary.json").exists():
        raise ValueError("Completed TEST comparison is immutable")
    write_json(out / "manifest.json", manifest)
    rows = test_records(json.loads(TEST_INDEX.read_text()), json.loads(v8.INDEX.read_text()))
    encoder = af._load_encoder(v8.extract.WEIGHTS, device)
    crop_config = json.loads(Path("configs/axial-localizer-v6.json").read_text())
    records = {}
    for arm in v8.ARMS:
        records[arm] = v8.extract_arm(
            rows,
            localizers[arm],
            center,
            encoder,
            crop_config,
            TEST_INDEX,
            out / "models" / arm,
            device,
        )
    write_json(
        out / "images-complete.json",
        {"both_arms_complete": True, "labels_joined": False, "n_patients_per_arm": len(rows)},
    )
    labels = test_labels(rows, "data/rsna/train.csv")
    summary = {
        "phase": "test",
        "scope": "historically_reused_test_exploratory",
        "fixed_before_test": True,
        "no_test_selection": True,
        "labels_joined_after_image_inference": True,
        "severity_seeds": list(v8.SEEDS),
        "n_requested_patients": len(rows),
        "bootstrap_resamples": config["bootstrap_resamples"],
        "bootstrap_seed": config["bootstrap_seed"],
        "provenance": provenance,
        "models": {},
    }
    for arm in v8.ARMS:
        directory = out / "models" / arm
        targets, features = v8.pack_features(records[arm], labels, directory)
        all_predictions, paths = [], {}
        for seed, model in zip(v8.SEEDS, classifiers[arm], strict=True):
            predictions = severity.prediction_rows(targets, model, features)
            path = directory / f"seed_{seed}.csv"
            _write(path, predictions)
            paths[str(seed)] = str(path.relative_to(out))
            all_predictions.append(predictions)
        _write(directory / "ensemble.csv", severity._ensemble(all_predictions))
        summary["models"][arm] = {
            "predictions": str((directory / "ensemble.csv").relative_to(out)),
            "predictions_sha256": sha256_file(directory / "ensemble.csv"),
            "seed_predictions": paths,
            "features_sha256": sha256_file(directory / "features.npz"),
            "targets_sha256": sha256_file(directory / "targets.csv"),
        }
    # Historical sagittal probabilities are used unchanged, after axial image
    # inference. This verifier fixes the frozen-head family and all three seeds.
    from .localized_stenosis_report import _verified_sagittal, read_predictions

    ensemble, hashes = _verified_sagittal(SAGITTAL)
    expected = {(d, p, level) for d, p, level in labels}
    if {(r["dataset"], r["study_id"], r["level"]) for r in ensemble} != expected or any(
        r["label"] != labels[(r["dataset"], r["study_id"], r["level"])] for r in ensemble
    ):
        raise ValueError("Sagittal TEST patient/level references differ from axial")
    directory = out / "models" / "sagittal"
    directory.mkdir(parents=True)
    paths = {}
    for seed in v8.SEEDS:
        source = SAGITTAL / "entries" / "frozen_head" / f"seed-{seed}" / "predictions.csv"
        path = directory / f"seed_{seed}.csv"
        _write(path, read_predictions(source, default_dataset="rsna"))
        paths[str(seed)] = str(path.relative_to(out))
    _write(directory / "ensemble.csv", ensemble)
    summary["models"]["sagittal"] = {
        "predictions": str((directory / "ensemble.csv").relative_to(out)),
        "predictions_sha256": sha256_file(directory / "ensemble.csv"),
        "seed_predictions": paths,
        "historical_source_hashes": hashes,
        "source": "unchanged verified v1 frozen-head TEST probabilities",
    }
    verify_inputs(protocol_path)
    if sha256_file(Path(__file__)) != provenance["implementation_sha256"]:
        raise RuntimeError("TEST inference implementation changed during the run")
    write_json(out / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--out", type=Path, default=RUN)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args(argv)
    evaluate_test(args.protocol, args.out, args.device)


if __name__ == "__main__":
    main()
