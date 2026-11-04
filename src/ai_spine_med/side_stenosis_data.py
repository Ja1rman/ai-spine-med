"""Image-only frozen caches for the first bilateral RSNA stenosis experiment.

Foramina reuse SpineNet sagittal T2-like vectors. Recesses reuse the sealed
mixed-localizer axial 50 mm ROI layer3 maps, followed by unchanged ResNet18
layer4. No reference coordinates or grades choose images. This is deliberately
not a sagittal T1 experiment and not a model of individual nerve roots.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from . import axial_features as af
from .assets import sha256_file, write_json

PROTOCOL = Path("configs/side-stenosis-v15.json")
OUT = Path("artifacts/side-stenosis-v15")
LEVELS = ("L4-L5", "L5-S1")
SIDES = ("left", "right")
SPLITS = ("train", "validation")
CONDITIONS = ("foraminal", "subarticular")
GRADE = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2, "": None}
SAG = Path("artifacts/features-v1")
AXIAL = Path("artifacts/roi-layer4-v11/mixed")


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def verify_protocol(path=PROTOCOL):
    config = json.loads(Path(path).read_text())
    if config["test_evaluated"] is not False or config["splits"] != list(SPLITS):
        raise ValueError("Only the original TRAIN/validation partitions are permitted")
    for file, digest in config["frozen_files"].items():
        if sha256_file(Path(file)) != digest:
            raise ValueError(f"Frozen source changed: {file}")
    return config


def spatial_pool(maps):
    """Canonical x increases toward patient LEFT; exclude the middle column.

    Return global, anatomical LEFT and anatomical RIGHT, never image left/right.
    This relies on _canonical_image's verified DICOM-LPS transformation, retained
    unchanged by the physical crop sampler and spatial encoder.
    """
    if maps.ndim != 4 or maps.shape[-2:] != (7, 7) or maps.shape[1] != 512:
        raise ValueError("Expected original ResNet18 layer4 maps (N,512,7,7)")
    return torch.cat(
        (maps.mean((-2, -1)), maps[..., 4:].mean((-2, -1)), maps[..., :3].mean((-2, -1))),
        dim=1,
    )


def task_index(level, side):
    return LEVELS.index(level) * 2 + SIDES.index(side)


def labels_for(rows, condition, label_rows):
    """Join independent severity references only after image extraction."""
    by_id = {row["study_id"]: row for row in label_rows}
    if len(by_id) != len(label_rows):
        raise ValueError("Duplicate severity study")
    prefix = "neural_foraminal_narrowing" if condition == "foraminal" else "subarticular_stenosis"
    output = []
    for row in rows:
        for side in SIDES:
            column = f"{side}_{prefix}_{row['level'].lower().replace('-', '_')}"
            value = by_id[row["study_id"]][column]
            if value not in GRADE:
                raise ValueError("Unknown RSNA reference grade")
            output.append(
                {
                    **row,
                    "condition": condition,
                    "side": side,
                    "task_index": task_index(row["level"], side),
                    "label": GRADE[value],
                    "reference_status": "known" if GRADE[value] is not None else "missing",
                }
            )
    return output


def _sagittal_cache():
    cache = np.load(SAG / "features.npz", allow_pickle=False)
    if set(str(x) for x in cache["split"]) - set(SPLITS):
        raise ValueError("TEST cannot enter sagittal fitting data")
    keys = list(zip(cache["study_id"].astype(str), cache["level"].astype(str)))
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate sagittal feature target")
    key_to_index = {key: i for i, key in enumerate(keys)}
    rows = []
    for source in read_csv(SAG / "target_manifest.csv"):
        if source["split"] not in SPLITS:
            raise ValueError("TEST cannot enter sagittal targets")
        key = source["study_id"], source["level"]
        index = key_to_index.get(key)
        if (index is not None) != (source["status"] == "ok"):
            raise ValueError("Sagittal feature/target coverage mismatch")
        if index is not None and str(cache["split"][index]) != source["split"]:
            raise ValueError("Sagittal feature split mismatch")
        rows.append(
            {k: source[k] for k in ("study_id", "split", "level", "status", "reason")}
            | {"feature_index": index}
        )
    return rows, np.asarray(cache["features"], np.float32)


def _axial_partition(split, directory, encoder):
    source = AXIAL / split
    rows = json.loads((source / "rows.json").read_text())
    if any(r["dataset"] != "rsna" or r["split"] != split for r in rows):
        raise ValueError("Only the sealed matching RSNA partition is permitted")
    maps = np.load(source / "prefix.npy", mmap_mode="r", allow_pickle=False)
    offsets = np.load(source / "offsets.npy", allow_pickle=False)
    mean_global = np.load(source / "frozen_features.npy", allow_pickle=False)
    if maps.dtype != np.float32 or maps.shape[1:] != (256, 14, 14):
        raise ValueError("Invalid original layer3 maps")
    tail = encoder.layer4.eval().requires_grad_(False)
    per_center = []
    with torch.inference_mode():
        for start in range(0, len(maps), 128):
            batch = torch.from_numpy(np.array(maps[start : start + 128], copy=True))
            if not torch.isfinite(batch).all():
                raise ValueError("Nonfinite frozen maps")
            per_center.append(spatial_pool(tail(batch)).numpy())
            if start % 1024 == 0:
                print(
                    json.dumps({"axial_split": split, "encoded": start, "total": len(maps)}),
                    flush=True,
                )
    centers = np.concatenate(per_center).astype(np.float32)
    values = np.stack([centers[start:end].mean(0) for start, end in offsets])
    error = float(np.max(np.abs(values[:, :512] - mean_global)))
    if error > 1e-4:
        raise ValueError("Original global vectors do not reproduce from spatial maps")
    np.save(directory / f"axial-{split}.npy", values, allow_pickle=False)
    output = [
        {k: row[k] for k in ("study_id", "split", "level", "status", "reason", "feature_index")}
        for row in rows
    ]
    write_json(directory / f"axial-{split}-images.json", output)
    return output, values, error


def prepare(protocol=PROTOCOL, out=OUT):
    protocol, out = Path(protocol), Path(out)
    verify_protocol(protocol)
    if (out / "summary.json").exists():
        summary = json.loads((out / "summary.json").read_text())
        if summary["protocol_sha256"] != sha256_file(protocol):
            raise ValueError("Prepared data belongs to another protocol")
        if any(sha256_file(out / file) != digest for file, digest in summary["sha256"].items()):
            raise ValueError("Prepared cache bytes changed")
        return summary
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    sag_rows, sag_features = _sagittal_cache()
    images = {"foraminal": {}, "subarticular": {}}
    for split in SPLITS:
        split_rows, split_features = [], []
        for row in sag_rows:
            if row["split"] != split:
                continue
            row = dict(row)
            if row["feature_index"] is not None:
                split_features.append(sag_features[row["feature_index"]])
                row["feature_index"] = len(split_features) - 1
            split_rows.append(row)
        images["foraminal"][split] = split_rows
        np.save(
            out / f"foraminal-{split}.npy",
            np.asarray(split_features, np.float32),
            allow_pickle=False,
        )
    encoder = af._load_encoder(Path(".cache/axial-v3/resnet18-f37072fd.pth"), "cpu")
    errors = {}
    for split in SPLITS:
        rows, _, error = _axial_partition(split, out, encoder)
        images["subarticular"][split] = rows
        errors[split] = error
    write_json(out / "images-complete.json", {"grades_joined": False, "test_evaluated": False})
    # Severity tables are not used by either image extractor.
    label_rows = read_csv("data/rsna/train.csv")
    split_rows = read_csv("artifacts/rsna-v1/studies.csv")
    split_map = {r["study_id"]: r["split"] for r in split_rows}
    summary = {
        "protocol_sha256": sha256_file(protocol),
        "implementation_sha256": sha256_file(Path(__file__)),
        "test_evaluated": False,
        "grades_joined_after_images": True,
        "axial_original_feature_parity_max_abs": errors,
        "conditions": {},
        "sha256": {},
    }
    for condition in CONDITIONS:
        summary["conditions"][condition] = {}
        for split in SPLITS:
            rows = images[condition][split]
            if any(split_map[r["study_id"]] != split for r in rows):
                raise ValueError("Feature cache differs from original study split")
            expected = {r["study_id"] for r in split_rows if r["split"] == split}
            if len(rows) != len(expected) * 2 or {(r["study_id"], r["level"]) for r in rows} != {
                (study, level) for study in expected for level in LEVELS
            }:
                raise ValueError("Incomplete requested cohort")
            targets = labels_for(rows, condition, label_rows)
            write_json(out / f"{condition}-{split}-targets.json", targets)
            name = f"{condition}-{split}.npy" if condition == "foraminal" else f"axial-{split}.npy"
            summary["conditions"][condition][split] = {
                "features": name,
                "targets": f"{condition}-{split}-targets.json",
                "requested": len(targets),
                "assessed": sum(r["status"] == "ok" for r in targets),
                "known_references": sum(r["label"] is not None for r in targets),
                "studies": len(expected),
            }
    for path in sorted(out.iterdir()):
        if path.suffix in {".json", ".npy"} and path.name != "summary.json":
            summary["sha256"][path.name] = sha256_file(path)
    verify_protocol(protocol)
    write_json(out / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.protocol, args.out), indent=2))


if __name__ == "__main__":
    main()
