"""Original frozen 512-vectors per ROI centre, before any bag pooling.

Reuse the unchanged v11 float32 layer3 maps for RSNA TRAIN/validation only.
The original ImageNet layer4 and average pool process each centre separately.
Existing labels are copied unchanged, never passed into feature extraction or
used to choose a centre. No DICOM, localizer, or TEST inference occurs here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import axial_features as af
from . import localization_severity_validation as v8
from . import roi_layer4_data as previous
from .assets import sha256_file, write_json

PROTOCOL = Path("configs/roi-attention-v12.json")
OUT = Path("artifacts/roi-attention-v12")
SOURCE = previous.OUT
ARMS = previous.ARMS
SPLITS = previous.SPLITS
FILES = ("center_features.npy", "offsets.npy", "rows.json", "frozen_features.npy")


def verify_protocol(path):
    config = json.loads(Path(path).read_text())
    if (
        config.get("phase") != "validation"
        or config.get("test_evaluated") is not False
        or config.get("arms") != list(ARMS)
    ):
        raise ValueError("Require the fixed development-only attention protocol")
    frozen = {name: sha256_file(Path(name)) for name in config["frozen_files"]}
    if frozen != config["frozen_files"]:
        raise ValueError("A frozen prefix input, original encoder or source changed")
    return config, frozen


def verify_source(directory=SOURCE):
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text())
    if (
        summary.get("phase") != "train_validation"
        or summary.get("test_evaluated") is not False
        or summary.get("grades_joined_after_all_four_image_partitions") is not True
        or summary["provenance"]["implementation_sha256"] != sha256_file(Path(previous.__file__))
    ):
        raise ValueError("Require the completed unchanged TRAIN/validation prefix cache")
    return summary


def extract_centers(maps, tail, device, *, batch_size=128):
    """Return one original vector per actual centre; no labels or averaging."""
    if (
        maps.dtype != np.float32
        or maps.ndim != 4
        or maps.shape[1:] != previous.SHAPE
        or not len(maps)
        or type(batch_size) is not int
        or batch_size < 1
    ):
        raise ValueError("Require float32 original per-centre layer3 maps")
    tail.eval().requires_grad_(False)
    immutable = {k: v.detach().cpu().clone() for k, v in tail.state_dict().items()}
    values = []
    with torch.inference_mode():
        for start in range(0, len(maps), batch_size):
            batch = np.array(maps[start : start + batch_size], dtype=np.float32, copy=True)
            if not np.isfinite(batch).all():
                raise ValueError("Nonfinite source prefix maps")
            encoded = tail(torch.from_numpy(batch).to(device)).detach().cpu().numpy()
            if encoded.shape != (len(batch), 512) or not np.isfinite(encoded).all():
                raise ValueError("Invalid frozen per-centre feature vectors")
            values.append(encoded.astype(np.float32))
            if (start // batch_size + 1) % 20 == 0 or start + batch_size >= len(maps):
                print(
                    json.dumps(
                        {"centres_encoded": min(start + batch_size, len(maps)), "total": len(maps)}
                    ),
                    flush=True,
                )
    if any(not torch.equal(v.detach().cpu(), immutable[k]) for k, v in tail.state_dict().items()):
        raise RuntimeError("Frozen original encoder weights or BatchNorm buffers changed")
    return np.concatenate(values)


def mean_features(center_features, offsets):
    """Historical equal average is retained only as a parity check/control."""
    return np.stack([center_features[start:end].mean(0) for start, end in offsets]).astype(
        np.float32
    )


def validate_arrays(rows, center_features, offsets, frozen_features):
    if not rows or any(r["dataset"] != "rsna" or r["split"] not in SPLITS for r in rows):
        raise ValueError("Only RSNA TRAIN/validation may enter attention fitting caches")
    keys = [(r["study_id"], r["level"]) for r in rows]
    if len(keys) != len(set(keys)) or len({r["split"] for r in rows}) != 1:
        raise ValueError("Duplicate target or mixed split in attention cache")
    if any(r["label"] is not None and r["label"] not in (0, 1, 2) for r in rows):
        raise ValueError("Invalid inherited reference grade")
    indices = [r["feature_index"] for r in rows if r["status"] == "ok"]
    if indices != list(range(len(indices))):
        raise ValueError("Attention targets require contiguous assessed feature indices")
    if any(r["feature_index"] is not None or not r["reason"] for r in rows if r["status"] != "ok"):
        raise ValueError("Failed target requires explicit reason and no fabricated features")
    if (
        offsets.dtype != np.int64
        or offsets.shape != (len(indices), 2)
        or not len(offsets)
        or offsets[0, 0] != 0
        or not np.array_equal(offsets[1:, 0], offsets[:-1, 1])
        or ((offsets[:, 1] - offsets[:, 0]) < 1).any()
        or ((offsets[:, 1] - offsets[:, 0]) > 3).any()
        or center_features.dtype != np.float32
        or center_features.shape != (int(offsets[-1, 1]), 512)
        or not np.isfinite(center_features).all()
        or frozen_features.dtype != np.float32
        or frozen_features.shape != (len(indices), 512)
        or not np.isfinite(frozen_features).all()
    ):
        raise ValueError("Invalid aligned per-centre features, offsets or frozen vectors")


def extract_partition(source, arm, split, directory, tail, device, atol=1e-4, *, batch_size=128):
    if arm not in ARMS or split not in SPLITS:
        raise ValueError("Use only the fixed localizers and TRAIN/validation")
    rows, maps, offsets, frozen_features, source_receipt = previous.load_cache(source, arm, split)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = directory / "receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["source_partition_sha256"] != source_receipt["sha256"] or any(
            sha256_file(directory / name) != digest for name, digest in receipt["sha256"].items()
        ):
            raise ValueError("Resumed per-centre feature partition changed")
        features = np.load(directory / "center_features.npy", allow_pickle=False)
        validate_arrays(rows, features, offsets, frozen_features)
        if float(np.max(np.abs(mean_features(features, offsets) - frozen_features))) > atol:
            raise ValueError("Resumed centre features no longer reproduce the frozen mean")
        return receipt
    features = extract_centers(maps, tail, device, batch_size=batch_size)
    validate_arrays(rows, features, offsets, frozen_features)
    error = float(np.max(np.abs(mean_features(features, offsets) - frozen_features)))
    if error > atol:
        raise ValueError(f"Per-centre mean exceeds historical feature tolerance ({error:.8g})")
    np.save(directory / "center_features.npy", features, allow_pickle=False)
    np.save(directory / "offsets.npy", offsets, allow_pickle=False)
    np.save(directory / "frozen_features.npy", frozen_features, allow_pickle=False)
    write_json(directory / "rows.json", rows)
    receipt = {
        "n_patients": len({r["study_id"] for r in rows}),
        "n_targets": len(rows),
        "n_assessed": len(offsets),
        "n_centers": len(features),
        "max_abs_mean_feature_error": error,
        "source_partition_sha256": source_receipt["sha256"],
        "sha256": {name: sha256_file(directory / name) for name in FILES},
    }
    write_json(receipt_path, receipt)
    return receipt


def prepare_cache(protocol_path=PROTOCOL, out=OUT, device="cpu", *, source=SOURCE):
    protocol_path, out, source = Path(protocol_path), Path(out), Path(source)
    config, frozen = verify_protocol(protocol_path)
    verify_source(source)
    if frozen.get(str(SOURCE / "summary.json")) != sha256_file(source / "summary.json"):
        raise ValueError("Prefix source differs from the protocol-pinned original cache")
    device = af._device(device)
    torch.set_num_threads(4)
    provenance = {
        "frozen_files": frozen,
        "protocol_sha256": sha256_file(protocol_path),
        "implementation_sha256": sha256_file(Path(__file__)),
        "source_summary_sha256": sha256_file(source / "summary.json"),
    }
    manifest = {"phase": "original_per_center_features", "provenance": provenance}
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists() and json.loads(
        (out / "manifest.json").read_text()
    ) != manifest:
        raise ValueError("Changed attention cache provenance; use a new directory")
    if (out / "summary.json").exists():
        raise ValueError("Completed attention cache is immutable")
    write_json(out / "manifest.json", manifest)
    af._weight_signature(v8.extract.WEIGHTS)
    encoder = af._load_encoder(v8.extract.WEIGHTS, device)
    _, tail = previous.prefix_and_tail(encoder)
    partitions = {}
    for arm in ARMS:
        for split in SPLITS:
            print(
                json.dumps({"arm": arm, "split": split, "status": "encoding_centers"}), flush=True
            )
            partitions[f"{arm}/{split}"] = extract_partition(
                source,
                arm,
                split,
                out / arm / split,
                tail,
                device,
                config.get("reconstruction_atol", 1e-4),
                batch_size=config.get("feature_batch_size", 128),
            )
    verify_protocol(protocol_path)
    verify_source(source)
    if (
        sha256_file(Path(__file__)) != provenance["implementation_sha256"]
        or sha256_file(source / "summary.json") != provenance["source_summary_sha256"]
    ):
        raise RuntimeError("Attention cache source or implementation changed during extraction")
    summary = {
        "phase": "train_validation",
        "test_evaluated": False,
        "encoder": "original_frozen_imagenet_resnet18",
        "per_center_features_preserved": True,
        "labels_used_for_feature_extraction": False,
        "provenance": provenance,
        "partitions": partitions,
    }
    write_json(out / "summary.json", summary)
    return summary


def load_cache(directory=OUT, arm="mixed", split="train", verify=True):
    if arm not in ARMS or split not in SPLITS:
        raise ValueError("Use only fixed localizers and TRAIN/validation")
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text())
    if (
        summary.get("phase") != "train_validation"
        or summary.get("test_evaluated") is not False
        or summary.get("encoder") != "original_frozen_imagenet_resnet18"
        or summary.get("per_center_features_preserved") is not True
        or summary.get("labels_used_for_feature_extraction") is not False
        or summary.get("provenance", {}).get("implementation_sha256") != sha256_file(Path(__file__))
    ):
        raise ValueError("Require completed original frozen per-centre features")
    root = directory / arm / split
    receipt = summary["partitions"][f"{arm}/{split}"]
    if set(receipt["sha256"]) != set(FILES) or (
        verify
        and any(sha256_file(root / name) != digest for name, digest in receipt["sha256"].items())
    ):
        raise ValueError("Prepared per-centre feature cache changed")
    rows = json.loads((root / "rows.json").read_text())
    features = np.load(root / "center_features.npy", allow_pickle=False)
    offsets = np.load(root / "offsets.npy", allow_pickle=False)
    frozen_features = np.load(root / "frozen_features.npy", allow_pickle=False)
    validate_arrays(rows, features, offsets, frozen_features)
    if any(row["split"] != split for row in rows):
        raise ValueError("Attention cache split differs from its partition")
    return rows, features, offsets, frozen_features, receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args(argv)
    prepare_cache(args.protocol, args.out, args.device, source=args.source)


if __name__ == "__main__":
    main()
