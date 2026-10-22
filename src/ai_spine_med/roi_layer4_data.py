"""Exact v9/v8 predicted-ROI prefix maps for a development-only layer4 study.

Only original RSNA TRAIN/validation images enter this module. Existing image-only
prediction receipts fix all levels, slice centres, physical crops and neighbours.
No localizer is retrained or rerun. Labels are joined after all four image-cache
partitions finish; they never choose a crop. The original frozen 512-vectors are
retained verbatim for the paired frozen-encoder controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from . import axial_features as af
from . import localization_severity_validation as v8
from . import localized_stenosis as severity
from .assets import sha256_file, write_json
from .axial_localizer_crops import crop_level_inputs

PROTOCOL = Path("configs/roi-layer4-v11.json")
OUT = Path("artifacts/roi-layer4-v11")
TRAIN = Path("artifacts/localization-finetuning-v9")
VALIDATION = Path("reports/runs/localization-severity-v8/arms")
SHAPE = (256, 14, 14)
SPLITS = ("train", "validation")
ARMS = ("xml_only", "mixed")


def verify_protocol(path):
    config = json.loads(Path(path).read_text())
    if (
        config.get("phase") != "validation"
        or config.get("test_evaluated") is not False
        or config.get("arms") != list(ARMS)
    ):
        raise ValueError("Require the fixed development-only two-localizer protocol")
    frozen = {name: sha256_file(Path(name)) for name in config["frozen_files"]}
    if frozen != config["frozen_files"]:
        raise ValueError("A frozen ROI input, source or encoder changed")
    return config, frozen


def fitting_records(index):
    records = index["records"]
    if any(row["split"] not in SPLITS for row in records):
        raise ValueError("TEST is forbidden in layer4 image preparation")
    result = [v8.image_descriptor(row) for row in records if row["dataset"] == "rsna"]
    keys = [(row["study_id"], row["split"]) for row in result]
    patients = [row["study_id"] for row in result]
    if not result or len(keys) != len(set(keys)) or len(patients) != len(set(patients)):
        raise ValueError("Require disjoint distinct RSNA fitting patients")
    if set(row["split"] for row in result) != set(SPLITS):
        raise ValueError("Both TRAIN and validation must be present")
    return result


def prefix_and_tail(encoder):
    """The prefix ends at layer3; layer4/avgpool follow without approximation."""
    prefix = torch.nn.Sequential(*list(encoder.children())[:-3])
    tail = torch.nn.Sequential(encoder.layer4, encoder.avgpool, torch.nn.Flatten(1))
    return prefix.eval().requires_grad_(False), tail.eval().requires_grad_(False)


def _safe(root, relative):
    root, relative = Path(root), Path(relative)
    path = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root.resolve()):
        raise ValueError("Unsafe ROI source cache path")
    return path


def _token(row):
    return hashlib.sha256(f"{row['dataset']}:{row['study_id']}".encode()).hexdigest()[:24]


def image_plan(records, directory):
    """Allocate offsets from frozen image receipts without reading grade tables."""
    plans, targets, offsets, offset = [], [], [], 0
    directory = Path(directory)
    for row in records:
        receipt_path = directory / "records" / f"{_token(row)}.json"
        receipt = json.loads(receipt_path.read_text())
        if any(receipt[k] != row[k] for k in ("dataset", "study_id", "split")) or (
            receipt["image_input_sha256"] != af._hash_json(row)
        ):
            raise ValueError("Original ROI image descriptor changed")
        feature_path = _safe(directory, receipt["features_path"])
        if sha256_file(feature_path) != receipt["features_sha256"]:
            raise ValueError("Original per-study ROI features changed")
        if [r["level"] for r in receipt["targets"]] != list(severity.LEVELS):
            raise ValueError("Original image receipt has invalid target levels")
        plan = {
            "source": row,
            "receipt": receipt,
            "receipt_sha256": sha256_file(receipt_path),
            "feature_path": feature_path,
            "targets": [],
        }
        for source_target in receipt["targets"]:
            target = {k: row[k] for k in ("dataset", "study_id", "split")}
            target.update(
                level=source_target["level"],
                status=source_target["status"],
                reason=source_target["reason"],
                feature_index=None,
            )
            if target["status"] == "ok":
                level = target["level"]
                crop = receipt["provenance"]["crops"][level]
                count = len(crop["metadata"]["centers"])
                if crop["status"] != "ok" or not 1 <= count <= 3:
                    raise ValueError("Original predicted ROI bag is invalid")
                target["feature_index"] = len(offsets)
                offsets.append((offset, offset + count))
                offset += count
            elif source_target["feature_index"] is not None or not target["reason"]:
                raise ValueError("Failed image target must retain explicit reason")
            plan["targets"].append(target)
            targets.append(target)
        plans.append(plan)
    return plans, targets, np.asarray(offsets, dtype=np.int64).reshape(-1, 2), offset


def _array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _open_array(path, shape):
    path = Path(path)
    if path.exists():
        array = np.load(path, mmap_mode="r+", allow_pickle=False)
        if array.shape != shape or array.dtype != np.float32:
            raise ValueError("Resumed prefix cache shape/dtype changed")
        return array
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)


def extract_partition(records, source_directory, directory, prefix, tail, config, device, atol):
    """Reconstruct the same predicted crops, with verified per-patient resumption."""
    directory = Path(directory)
    plans, targets, offsets, n_crops = image_plan(records, source_directory)
    if not offsets.size:
        raise ValueError("No assessed ROI targets in image partition")
    manifest = {
        "phase": "image_only",
        "source_receipts_sha256": af._hash_json([p["receipt_sha256"] for p in plans]),
        "n_crops": n_crops,
        "offsets_sha256": _array_digest(offsets),
        "parameters": {
            k: config[k] for k in ("roi_mm", "roi_max_centers", "roi_window_mm")
        },
    }
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "images-manifest.json").exists() and json.loads(
        (directory / "images-manifest.json").read_text()
    ) != manifest:
        raise ValueError("Resumed prefix image partition changed")
    write_json(directory / "images-manifest.json", manifest)
    maps = _open_array(directory / "prefix.npy", (n_crops, *SHAPE))
    reconstructed = _open_array(directory / "reconstructed_features.npy", (len(offsets), 512))
    receipts = directory / "records"
    receipts.mkdir(exist_ok=True)
    errors = []
    for number, plan in enumerate(plans):
        row, original = plan["source"], plan["receipt"]
        done_path = receipts / f"{_token(row)}.json"
        assessed = [t for t in plan["targets"] if t["status"] == "ok"]
        if done_path.exists():
            done = json.loads(done_path.read_text())
            actual = [
                _array_digest(maps[slice(*offsets[t["feature_index"]])]) for t in assessed
            ]
            if (
                done["source_receipt_sha256"] != plan["receipt_sha256"]
                or done["prefix_sha256"] != actual
                or done["reconstructed_sha256"]
                != [_array_digest(reconstructed[t["feature_index"]]) for t in assessed]
            ):
                raise ValueError("Resumed per-study prefix cache changed")
            errors.append(done["max_abs_feature_error"])
            continue
        errors.append(0.0)
        if assessed:
            provenance = original["provenance"]
            crops = crop_level_inputs(
                row,
                provenance["prediction"],
                roi_mm=config["roi_mm"],
                max_centers=config["roi_max_centers"],
                window_mm=config["roi_window_mm"],
            )
            if crops["parameters"] != provenance["parameters"]:
                raise ValueError("Reconstructed ROI physical parameters changed")
            with np.load(plan["feature_path"], allow_pickle=False) as archive:
                old_features = archive["features"]
                for target in assessed:
                    level, index = target["level"], target["feature_index"]
                    crop = crops["levels"][level]
                    old_crop = provenance["crops"][level]
                    if (
                        crop["status"] != "ok"
                        or crop["metadata"] != old_crop["metadata"]
                        or len(crop["images"]) != int(offsets[index, 1] - offsets[index, 0])
                    ):
                        raise ValueError("Reconstructed automatic ROI identity or geometry changed")
                    with torch.inference_mode():
                        z = prefix(crop["images"].to(device))
                        feature = tail(z).mean(0).detach().cpu().numpy().astype(np.float32)
                        z = z.detach().cpu().numpy().astype(np.float32)
                    old_index = next(
                        t["feature_index"] for t in original["targets"] if t["level"] == level
                    )
                    error = float(np.abs(feature - old_features[old_index]).max())
                    if not np.isfinite(z).all() or not np.isfinite(feature).all() or error > atol:
                        raise ValueError(f"Frozen ROI reconstruction exceeds tolerance ({error:.8g})")
                    maps[slice(*offsets[index])] = z
                    reconstructed[index] = feature
                    errors[-1] = max(errors[-1], error)
        maps.flush()
        reconstructed.flush()
        write_json(
            done_path,
            {
                "source_receipt_sha256": plan["receipt_sha256"],
                "prefix_sha256": [
                    _array_digest(maps[slice(*offsets[t["feature_index"]])]) for t in assessed
                ],
                "reconstructed_sha256": [
                    _array_digest(reconstructed[t["feature_index"]]) for t in assessed
                ],
                "max_abs_feature_error": errors[-1],
            },
        )
        if (number + 1) % 20 == 0 or number + 1 == len(plans):
            print(
                json.dumps(
                    {
                        "partition": str(directory),
                        "image_patients": number + 1,
                        "total": len(plans),
                        "max_abs_feature_error": max(errors),
                    }
                ),
                flush=True,
            )
    np.save(directory / "offsets.npy", offsets, allow_pickle=False)
    write_json(directory / "image-targets.json", targets)
    return {
        "n_patients": len(plans),
        "n_targets": len(targets),
        "n_assessed": len(offsets),
        "n_crops": n_crops,
        "max_abs_feature_error": max(errors),
    }


def join_targets(image_targets, source_targets, source_features):
    """After image completion, restore exact prior grades/features by target key."""
    indexed = {
        (r["dataset"], r["study_id"], r["level"]): r
        for r in source_targets
        if r["dataset"] == "rsna"
    }
    if len(indexed) != sum(r["dataset"] == "rsna" for r in source_targets):
        raise ValueError("Duplicate original RSNA target")
    if set(indexed) != {(r["dataset"], r["study_id"], r["level"]) for r in image_targets}:
        raise ValueError("Reconstructed and original requested targets differ")
    result, features = [], []
    for target in image_targets:
        source = indexed[(target["dataset"], target["study_id"], target["level"])]
        if any(target[k] != source[k] for k in ("split", "status", "reason")):
            raise ValueError("Reconstructed target status or split changed")
        result.append(dict(target, label=source["label"]))
        if target["status"] == "ok":
            if target["feature_index"] != len(features):
                raise ValueError("Image targets lost contiguous feature order")
            features.append(source_features[source["feature_index"]])
    return result, np.asarray(features, dtype=np.float32).reshape(-1, 512)


def prepare_cache(protocol_path=PROTOCOL, out=OUT, device="cpu"):
    protocol_path, out = Path(protocol_path), Path(out)
    config, frozen = verify_protocol(protocol_path)
    device = af._device(device)
    torch.set_num_threads(4)
    provenance = {
        "frozen_files": frozen,
        "protocol_sha256": sha256_file(protocol_path),
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    manifest = {"phase": "train_validation_images", "provenance": provenance}
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists() and json.loads((out / "manifest.json").read_text()) != manifest:
        raise ValueError("Changed resumable layer4 cache; use a new directory")
    if (out / "summary.json").exists():
        raise ValueError("Completed layer4 cache is immutable")
    write_json(out / "manifest.json", manifest)
    records = fitting_records(json.loads(v8.INDEX.read_text()))
    crop_config = json.loads(Path("configs/axial-localizer-v6.json").read_text())
    af._weight_signature(v8.extract.WEIGHTS)
    encoder = af._load_encoder(v8.extract.WEIGHTS, device)
    prefix, tail = prefix_and_tail(encoder)
    partitions = {}
    for arm in ARMS:
        for split in SPLITS:
            source_directory = (TRAIN if split == "train" else VALIDATION) / arm
            partitions[f"{arm}/{split}"] = extract_partition(
                [r for r in records if r["split"] == split],
                source_directory,
                out / arm / split,
                prefix,
                tail,
                crop_config,
                device,
                config.get("reconstruction_atol", 1e-4),
            )
    write_json(out / "images-complete.json", {"all_four_complete": True, "grades_joined": False})
    for key, partition in partitions.items():
        arm, split = key.split("/")
        source_directory = (TRAIN if split == "train" else VALIDATION) / arm
        original_rows = severity.read_targets(source_directory / "targets.csv", fitting=True)
        original_features = severity.load_features(source_directory / "features.npz", original_rows)
        directory = out / arm / split
        image_targets = json.loads((directory / "image-targets.json").read_text())
        targets, features = join_targets(image_targets, original_rows, original_features)
        reconstructed = np.load(directory / "reconstructed_features.npy", mmap_mode="r")
        if not np.allclose(reconstructed, features, atol=config.get("reconstruction_atol", 1e-4), rtol=0):
            raise ValueError("Aggregate frozen ROI features differ from original cache")
        write_json(directory / "rows.json", targets)
        np.save(directory / "frozen_features.npy", features, allow_pickle=False)
        partition["sha256"] = {
            name: sha256_file(directory / name)
            for name in ("prefix.npy", "offsets.npy", "rows.json", "frozen_features.npy")
        }
    verify_protocol(protocol_path)
    if sha256_file(Path(__file__)) != provenance["implementation_sha256"]:
        raise RuntimeError("Prefix-cache implementation changed during extraction")
    summary = {
        "phase": "train_validation",
        "test_evaluated": False,
        "provenance": provenance,
        "grades_joined_after_all_four_image_partitions": True,
        "partitions": partitions,
    }
    write_json(out / "summary.json", summary)
    return summary


def validate_arrays(rows, maps, offsets, features):
    """Check prefix-bag alignment, failures, finite features and split guards."""
    if any(r["dataset"] != "rsna" or r["split"] not in SPLITS for r in rows):
        raise ValueError("Only RSNA TRAIN/validation may enter the layer4 cache")
    keys = [(r["study_id"], r["level"]) for r in rows]
    if len(keys) != len(set(keys)) or len({r["split"] for r in rows}) != 1:
        raise ValueError("Duplicate target or mixed split in layer4 cache")
    indices = [r["feature_index"] for r in rows if r["status"] == "ok"]
    if indices != list(range(len(indices))):
        raise ValueError("Layer4 targets require contiguous assessed feature indices")
    if any(r["feature_index"] is not None or not r["reason"] for r in rows if r["status"] != "ok"):
        raise ValueError("Failed layer4 target requires explicit reason and no feature")
    if (
        offsets.dtype != np.int64
        or offsets.shape != (len(indices), 2)
        or not len(offsets)
        or offsets[0, 0] != 0
        or not np.array_equal(offsets[1:, 0], offsets[:-1, 1])
        or ((offsets[:, 1] - offsets[:, 0]) < 1).any()
        or ((offsets[:, 1] - offsets[:, 0]) > 3).any()
        or maps.dtype != np.float32
        or maps.shape != (int(offsets[-1, 1]), *SHAPE)
        or features.dtype != np.float32
        or features.shape != (len(indices), 512)
        or not np.isfinite(features).all()
    ):
        raise ValueError("Invalid aligned layer3 maps, offsets or frozen features")


def load_cache(directory=OUT, arm="mixed", split="train", verify=True):
    if arm not in ARMS or split not in SPLITS:
        raise ValueError("Use only the two fixed localizers and TRAIN/validation")
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text())
    if (
        summary.get("phase") != "train_validation"
        or summary.get("test_evaluated") is not False
        or summary.get("grades_joined_after_all_four_image_partitions") is not True
    ):
        raise ValueError("Require completed development-only prefix cache")
    root = directory / arm / split
    receipt = summary["partitions"][f"{arm}/{split}"]
    if verify and any(sha256_file(root / name) != digest for name, digest in receipt["sha256"].items()):
        raise ValueError("Prepared layer4 cache changed")
    rows = json.loads((root / "rows.json").read_text())
    maps = np.load(root / "prefix.npy", mmap_mode="r", allow_pickle=False)
    offsets = np.load(root / "offsets.npy", allow_pickle=False)
    features = np.load(root / "frozen_features.npy", allow_pickle=False)
    validate_arrays(rows, maps, offsets, features)
    if any(row["split"] != split for row in rows):
        raise ValueError("Prepared layer4 cache split differs from its partition")
    return rows, maps, offsets, features, receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args(argv)
    prepare_cache(args.protocol, args.out, args.device)


if __name__ == "__main__":
    main()
