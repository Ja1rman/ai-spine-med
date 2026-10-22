"""Arm-specific RSNA TRAIN ROIs for the fixed v9 classifier adaptation.

The old v8 image-only extractor is reused without changing its guards. Labels
are joined only after both localization arms complete image inference.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import torch

from . import axial_features as af
from . import localization_severity_validation as v8
from .assets import sha256_file, write_json

PROTOCOL = Path("configs/localization-finetuning-v9.json")
OUT = Path("artifacts/localization-finetuning-v9")


def verify_protocol(path):
    path = Path(path)
    config = json.loads(path.read_text())
    if config.get("phase") != "validation" or config.get("test_evaluated") is not False:
        raise ValueError("Require a development-only v9 protocol")
    files = {name: sha256_file(Path(name)) for name in config["frozen_files"]}
    if files != config["frozen_files"]:
        raise ValueError("A frozen input or v8 validation result changed")
    return config, files


def rsna_train_records(index):
    if any(r["split"] not in {"train", "validation"} for r in index["records"]):
        raise ValueError("TEST is forbidden in adaptation preparation")
    rows = [r for r in index["records"] if r["dataset"] == "rsna" and r["split"] == "train"]
    if not rows or len({r["study_id"] for r in rows}) != len(rows):
        raise ValueError("Require distinct RSNA TRAIN patients")
    return rows


def train_labels(rows, path):
    if any(r["dataset"] != "rsna" or r["split"] != "train" for r in rows):
        raise ValueError("Severity gradients accept RSNA TRAIN only")
    patients = {r["study_id"] for r in rows}
    result = {}
    mapping = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2, "": None}
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["study_id"] not in patients:
                continue
            for level, column in zip(
                v8.loc.LEVELS, ("spinal_canal_stenosis_l4_l5", "spinal_canal_stenosis_l5_s1")
            ):
                key = "rsna", row["study_id"], level
                if key in result or row[column] not in mapping:
                    raise ValueError("Invalid or duplicate RSNA TRAIN reference")
                result[key] = mapping[row[column]]
    if result.keys() != {("rsna", p, level) for p in patients for level in v8.loc.LEVELS}:
        raise ValueError("Missing TRAIN identity; absent grades must be explicit")
    return result


def prepare_data(protocol_path=PROTOCOL, out=OUT, device="cpu"):
    protocol_path, out = Path(protocol_path), Path(out)
    config, files = verify_protocol(protocol_path)
    device = af._device(device)
    provenance = {
        "frozen_files": files,
        "protocol_sha256": sha256_file(protocol_path),
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"phase": "rsna_train_images", "provenance": provenance}
    if (out / "manifest.json").exists() and json.loads(
        (out / "manifest.json").read_text()
    ) != manifest:
        raise ValueError("Prepared TRAIN inputs changed; use a new cache")
    if (out / "summary.json").exists():
        raise ValueError("Completed TRAIN cache is immutable")
    write_json(out / "manifest.json", manifest)
    rows = rsna_train_records(json.loads(v8.INDEX.read_text()))
    torch.set_num_threads(4)
    localizers, center = v8.load_ensembles(device)
    encoder = af._load_encoder(v8.extract.WEIGHTS, device)
    crop_config = json.loads(Path("configs/axial-localizer-v6.json").read_text())
    records = {}
    for arm in v8.ARMS:
        records[arm] = v8.extract_arm(
            rows, localizers[arm], center, encoder, crop_config, v8.INDEX, out / arm, device
        )
    write_json(
        out / "images-complete.json",
        {"both_arms_complete": True, "grades_joined": False, "studies_per_arm": len(rows)},
    )
    labels = train_labels(rows, "data/rsna/train.csv")
    summary = {
        "phase": "rsna_train",
        "provenance": provenance,
        "n_patients_requested": len(rows),
        "grades_joined_after_both_arms_images": True,
        "arms": {},
    }
    for arm in v8.ARMS:
        directory = out / arm
        targets, features = v8.pack_features(records[arm], labels, directory)
        summary["arms"][arm] = {
            "features": str((directory / "features.npz").relative_to(out)),
            "targets": str((directory / "targets.csv").relative_to(out)),
            "features_sha256": sha256_file(directory / "features.npz"),
            "targets_sha256": sha256_file(directory / "targets.csv"),
            "n_targets": len(targets),
            "n_features": len(features),
            "statuses": dict(Counter(r["status"] for r in targets)),
        }
    verify_protocol(protocol_path)
    if sha256_file(Path(__file__)) != provenance["implementation_sha256"]:
        raise RuntimeError("TRAIN extractor changed during inference")
    write_json(out / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args(argv)
    prepare_data(args.protocol, args.out, args.device)


if __name__ == "__main__":
    main()
