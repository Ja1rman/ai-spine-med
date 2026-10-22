"""VAL-only substitution of XML/mixed localization under a frozen severity head.

This separate wrapper preserves the old TEST/fitting guards. It neither trains
models nor reads annotations or grade labels in the image inference function.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import axial_features as af
from . import axial_localizer as loc
from . import axial_localizer_extract as extract
from . import localized_stenosis as severity
from .assets import sha256_file, write_json

ARMS = ("xml_only", "mixed")
SEEDS = (42, 2024, 3407)
INDEX = Path("artifacts/axial-localizer-v6/index.json")
LOCALIZERS = Path("reports/runs/xml-localizer-v7")
GRADERS = Path("reports/runs/explicit-localization-v6/severity")
PROTOCOL = Path("configs/localization-severity-v8.json")
RUN = Path("reports/runs/localization-severity-v8")


class ProbabilitySliceEnsemble(nn.Module):
    """Expose mean probability as log-probability to the existing inference API."""

    def __init__(self, models):
        super().__init__()
        if len(models) != 3:
            raise ValueError("Use all three prespecified slice models")
        self.models = nn.ModuleList(models)

    def forward(self, features, positions, mask):
        logp = torch.stack([m(features, positions, mask).log_softmax(-1) for m in self.models])
        return torch.logsumexp(logp, 0) - math.log(3)


def validation_records(index):
    rows = index["records"]
    if any(r["split"] not in {"train", "validation"} for r in rows):
        raise ValueError("TEST is forbidden in the validation substitution experiment")
    val = [r for r in rows if r["split"] == "validation"]
    keys = [(r["dataset"], r["study_id"]) for r in val]
    if not val or len(keys) != len(set(keys)) or any(k[0] not in {"rsna", "aisslab"} for k in keys):
        raise ValueError("Invalid validation patient cohort")
    return val


def image_descriptor(row):
    """Drop level/XY references before inference, even when present in input JSON."""
    return {
        k: row[k]
        for k in (
            "dataset",
            "study_id",
            "split",
            "status",
            "reason",
            "arrays_path",
            "arrays_sha256",
            "selected_slices",
            "source_root",
            "source",
        )
        if k in row
    }


def load_ensembles(device):
    seal = json.loads((LOCALIZERS / "selection.json").read_text())
    if (
        seal.get("frozen") is not True
        or seal.get("status") != "all_six_sealed_before_rsna_validation"
    ):
        raise ValueError("Require the unchanged complete v7 model seal")
    result, frozen_center = {}, None
    for arm in ARMS:
        models = []
        for seed, expected in zip(SEEDS, seal["checkpoints"][arm], strict=True):
            path = LOCALIZERS / arm / f"seed_{seed}" / "model.pt"
            if sha256_file(path) != expected:
                raise ValueError("A v7 localizer checkpoint changed")
            model, center, checkpoint = loc.load_localizer_checkpoint(path, "cpu")
            if checkpoint.get("seed") != seed:
                raise ValueError("Localizer seed ordering differs")
            if frozen_center is None:
                frozen_center = checkpoint["center_state"]
                center_model = center
            elif frozen_center.keys() != checkpoint["center_state"].keys() or any(
                not torch.equal(frozen_center[k], checkpoint["center_state"][k])
                for k in frozen_center
            ):
                raise ValueError("Center model differs between localization arms")
            models.append(model.eval().requires_grad_(False).to(device))
        result[arm] = ProbabilitySliceEnsemble(models).eval()
    return result, center_model.eval().requires_grad_(False).to(device)


def read_validation_labels(rows, rsna_path, aisslab_path):
    """Explicit reference-only VAL join; called after all image inference."""
    keys = {(r["dataset"], r["study_id"]) for r in rows}
    result = {}
    with Path(rsna_path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            if ("rsna", row["study_id"]) not in keys:
                continue
            for level, column in zip(
                loc.LEVELS, ("spinal_canal_stenosis_l4_l5", "spinal_canal_stenosis_l5_s1")
            ):
                key = "rsna", row["study_id"], level
                value = row[column]
                if key in result or value not in {"Normal/Mild", "Moderate", "Severe", ""}:
                    raise ValueError("Invalid or duplicate RSNA validation grade")
                result[key] = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2, "": None}[value]
    for row in json.loads(Path(aisslab_path).read_text()):
        if row["split"] != "validation" or ("aisslab", row["study_id"]) not in keys:
            continue
        for level, label in zip(loc.LEVELS, row["labels"], strict=True):
            key = "aisslab", row["study_id"], level
            if key in result or (
                label is not None and (type(label) is not int or label not in (0, 1, 2))
            ):
                raise ValueError("Invalid or duplicate AISSLab validation grade")
            result[key] = label
    expected = {(d, s, level) for d, s in keys for level in loc.LEVELS}
    if result.keys() != expected:
        raise ValueError(
            "Grade source lacks a validation patient/level; missing grade must be explicit"
        )
    return result


def extract_arm(rows, model, center, encoder, config, index_path, directory, device):
    """Image-only per-study resumable cache. Labels are not an argument."""
    (directory / "records").mkdir(parents=True, exist_ok=True)
    records = []
    for number, source in enumerate(rows):
        row = image_descriptor(source)
        token = hashlib.sha256(f"{row['dataset']}:{row['study_id']}".encode()).hexdigest()[:24]
        receipt, feature_path = (
            directory / "records" / f"{token}.json",
            directory / "records" / f"{token}.npz",
        )
        signature = af._hash_json(row)
        if receipt.exists():
            record = json.loads(receipt.read_text())
            if record["image_input_sha256"] != signature or record[
                "features_sha256"
            ] != sha256_file(feature_path):
                raise ValueError("Resumed image cache differs")
        else:
            if row["status"] != "ok":
                features, targets, provenance = (
                    np.empty((0, 512), np.float32),
                    [
                        {
                            "level": level,
                            "status": "failed",
                            "reason": row["reason"] or "source_not_processed",
                            "feature_index": None,
                        }
                        for level in loc.LEVELS
                    ],
                    {},
                )
            else:
                inputs, positions = extract.read_image_inputs(index_path, row)
                try:
                    features, targets, provenance = extract.infer_localized_study(
                        row, inputs, positions, model, center, encoder, config, device
                    )
                except af.AxialDataError as error:
                    features, targets, provenance = (
                        np.empty((0, 512), np.float32),
                        [
                            {
                                "level": level,
                                "status": "failed",
                                "reason": error.reason,
                                "feature_index": None,
                            }
                            for level in loc.LEVELS
                        ],
                        {},
                    )
            with feature_path.with_suffix(".part").open("wb") as stream:
                np.savez_compressed(stream, features=features)
            feature_path.with_suffix(".part").replace(feature_path)
            record = {k: row[k] for k in ("dataset", "study_id", "split")}
            record.update(
                image_input_sha256=signature,
                features_sha256=sha256_file(feature_path),
                features_path=str(feature_path.relative_to(directory)),
                targets=targets,
                provenance=provenance,
            )
            write_json(receipt, record)
        records.append(record)
        if (number + 1) % 50 == 0 or number + 1 == len(rows):
            print(
                json.dumps(
                    {"arm": directory.name, "image_studies": number + 1, "total": len(rows)}
                ),
                flush=True,
            )
    return records


def pack_features(records, labels, directory):
    vectors, targets = [], []
    for record in records:
        with np.load(directory / record["features_path"], allow_pickle=False) as archive:
            features = archive["features"].copy()
        offset = len(vectors)
        vectors.extend(features)
        for item in record["targets"]:
            row = {k: record[k] for k in ("dataset", "study_id", "split")}
            row.update(item, label=labels[(row["dataset"], row["study_id"], item["level"])])
            if row["feature_index"] is not None:
                row["feature_index"] += offset
            targets.append(row)
    features = np.stack(vectors).astype(np.float32) if vectors else np.empty((0, 512), np.float32)
    np.savez_compressed(directory / "features.npz", features=features)
    with (directory / "targets.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=severity.FIELDS)
        writer.writeheader()
        writer.writerows(targets)
    return severity.read_targets(directory / "targets.csv"), features


def run_comparison(protocol_path=PROTOCOL, out=RUN, device="cpu"):
    protocol_path, out = Path(protocol_path), Path(out)
    config = json.loads(protocol_path.read_text())
    if config.get("phase") != "validation" or config.get("arms") != list(ARMS):
        raise ValueError("Require the fixed two-arm validation protocol")
    checks = {name: sha256_file(Path(name)) for name in config["frozen_files"]}
    if checks != config["frozen_files"]:
        raise ValueError("Frozen inputs differ from protocol")
    checks.update(
        {
            "protocol": sha256_file(protocol_path),
            "implementation": sha256_file(Path(__file__)),
            "decoder": sha256_file(Path(af.__file__)),
            "encoder": af.WEIGHTS_SHA256,
        }
    )
    device = af._device(device)
    torch.set_num_threads(4)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"phase": "validation", "frozen_inputs": checks, "protocol": config}
    if (out / "manifest.json").exists() and json.loads(
        (out / "manifest.json").read_text()
    ) != manifest:
        raise ValueError("Changed resumable comparison; use a new directory")
    if (out / "summary.json").exists():
        raise ValueError("Complete comparison is immutable; use a new directory")
    write_json(out / "manifest.json", manifest)
    rows = validation_records(json.loads(INDEX.read_text()))
    localizers, center = load_ensembles(device)
    graders, _, _ = severity.load_sealed_models(GRADERS, device=device)
    encoder = af._load_encoder(extract.WEIGHTS, device)
    crop_config = json.loads(Path("configs/axial-localizer-v6.json").read_text())
    records = {}
    for arm in ARMS:
        records[arm] = extract_arm(
            rows, localizers[arm], center, encoder, crop_config, INDEX, out / "arms" / arm, device
        )
    write_json(
        out / "images-complete.json",
        {
            "both_arms_complete": True,
            "labels_joined": False,
            "studies_per_arm": len(rows),
            "frozen_inputs": checks,
        },
    )
    labels = read_validation_labels(
        rows, "data/rsna/train.csv", "artifacts/aisslab-axial-v4/records.json"
    )
    summary = {
        "phase": "validation",
        "center_reused_exactly": True,
        "severity_retrained": False,
        "test_evaluated": False,
        "localizer_ensemble": "mean_probabilities_before_argmax",
        "severity_seeds": list(SEEDS),
        "arms": {},
        "frozen_inputs": checks,
        "patient_counts": dict(Counter(r["dataset"] for r in rows)),
        "grade_labels_joined_after_both_arms_image_inference": True,
        "bootstrap_resamples": config["bootstrap_resamples"],
        "bootstrap_seed": config["bootstrap_seed"],
    }
    for arm in ARMS:
        directory = out / "arms" / arm
        targets, features = pack_features(records[arm], labels, directory)
        outputs, all_predictions = {}, []
        for seed, model in zip(SEEDS, graders, strict=True):
            predictions = severity.prediction_rows(targets, model, features)
            path = directory / f"seed_{seed}.csv"
            severity._write_predictions(path, predictions)
            outputs[str(seed)] = str(path.relative_to(out))
            all_predictions.append(predictions)
        ensemble = severity._ensemble(all_predictions)
        path = directory / "ensemble.csv"
        severity._write_predictions(path, ensemble)
        summary["arms"][arm] = {
            "predictions": str(path.relative_to(out)),
            "seed_predictions": outputs,
            "predictions_sha256": sha256_file(path),
            "features_sha256": sha256_file(directory / "features.npz"),
            "targets_sha256": sha256_file(directory / "targets.csv"),
            "metrics": severity._metrics(
                ensemble, "Exploratory VAL substitution under the unchanged v6 severity ensemble"
            ),
        }
    if any(
        sha256_file(Path(name)) != expected for name, expected in config["frozen_files"].items()
    ) or (
        sha256_file(Path(__file__)) != checks["implementation"]
        or sha256_file(protocol_path) != checks["protocol"]
    ):
        raise RuntimeError("Comparison inputs changed during inference")
    write_json(out / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--out", type=Path, default=RUN)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args(argv)
    run_comparison(args.protocol, args.out, args.device)


if __name__ == "__main__":
    main()
