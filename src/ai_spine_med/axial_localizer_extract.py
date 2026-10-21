"""Frozen ROI features selected by the sealed predicted axial localizer only.

Inference opens precisely features/positions from localization NPZ inputs.
Annotations, relevance kernels and reference centers are not inference inputs.
RSNA grades are joined only after feature extraction; AISSLab grades are allowed
only in the separately sealed TEST phase, never severity fitting or selection.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from . import axial_features as af
from . import axial_localizer as localizer
from .assets import sha256_file, write_json
from .axial_localizer_crops import crop_level_inputs, prepare_crop_source
from .axial_transfer_cache import _inputs

LEVELS = ("L4-L5", "L5-S1")
FIELDS = ("dataset", "study_id", "split", "level", "label", "status", "reason", "feature_index")
WEIGHTS = Path(".cache/axial-v3/resnet18-f37072fd.pth")


def _json(path):
    return json.loads(Path(path).read_text())


def _safe(root, relative):
    relative = Path(relative)
    path = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root.resolve()):
        raise ValueError("Unsafe localized feature cache path")
    return path


def verify_localizer_selection(index_path, selection_path, protocol_path):
    """Verify model choices and image-input scope before opening any DICOM."""
    index_path, selection_path, protocol_path = map(
        Path, (index_path, selection_path, protocol_path)
    )
    index, seal = _json(index_path), _json(selection_path)
    protocol_sha = sha256_file(protocol_path)
    if (
        seal.get("model_family") != localizer.FAMILY
        or seal.get("schema_version") != 1
        or seal.get("frozen") is not True
        or seal.get("localization_gate_passed") is not True
        or seal.get("protocol_sha256") != protocol_sha
        or seal.get("implementation_sha256") != sha256_file(Path(localizer.__file__))
    ):
        raise ValueError(
            "ROI features require an unchanged sealed localizer and passed validation gate"
        )
    checkpoint = _safe(selection_path.parent, seal["checkpoint"])
    if sha256_file(checkpoint) != seal.get("checkpoint_sha256"):
        raise ValueError("Frozen localizer checkpoint changed")
    summary_path = selection_path.parent / "summary.json"
    if (
        sha256_file(summary_path) != seal.get("summary_sha256")
        or _json(summary_path).get("status") != "complete"
    ):
        raise ValueError("Frozen localizer summary changed or incomplete")
    phase = index.get("configuration", {}).get("phase")
    rows = index["records"]
    if phase == "fitting":
        if sha256_file(index_path) != seal.get("training_index_sha256") or any(
            r["split"] not in {"train", "validation"} for r in rows
        ):
            raise ValueError(
                "Severity fitting inputs differ from original localizer training index"
            )
    elif phase == "evaluation_image_inputs":
        receipt = index["configuration"].get("seals", {})
        if (
            any(r["split"] != "test" for r in rows)
            or receipt.get("both_selections_frozen_before_test") is not True
            or receipt.get("localization_gate_passed") is not True
            or receipt.get("localizer_selection_sha256") != sha256_file(selection_path)
            or receipt.get("protocol_file_sha256") != protocol_sha
            or receipt.get("fitting_index_sha256") != seal.get("training_index_sha256")
            or not receipt.get("severity_selection_sha256")
        ):
            raise ValueError(
                "TEST ROI extraction requires image inputs prepared after both model seals"
            )
    else:
        raise ValueError("Unsupported localized image-input phase")
    keys = [(r["dataset"], r["study_id"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate localization input patient")
    return index, seal, checkpoint, phase


def read_image_inputs(path, row):
    """Do not enumerate/read other NPZ keys, even when supervision is present."""
    path = _safe(Path(path).parent, row["arrays_path"])
    if sha256_file(path) != row["arrays_sha256"]:
        raise ValueError("Localization image inputs changed")
    with np.load(path, allow_pickle=False) as archive:
        features, positions = archive["features"].copy(), archive["positions"].copy()
    if (
        features.dtype != np.float32
        or features.shape != (len(row["selected_slices"]), 512)
        or positions.shape != (len(features), 2)
        or not len(features)
        or not np.isfinite(features).all()
        or not np.isfinite(positions).all()
    ):
        raise ValueError("Invalid frozen localization image features/positions")
    return features, positions.astype(np.float32)


def _full_fov_maps(encoder, inputs):
    return torch.nn.Sequential(*list(encoder.children())[:-2])(inputs)


def infer_localized_study(
    row, features, positions, slice_model, center_model, encoder, protocol, device
):
    """The complete image-only inference path; accepts no reference arguments."""
    weights = localizer.predict_slice_weights(slice_model, features, positions)
    if weights.shape != (2, len(features)) or not np.isfinite(weights).all():
        raise ValueError("Invalid predicted slice weights")
    selected = [int(weights[i].argmax()) for i in range(2)]
    prepared = prepare_crop_source(row)
    images, contexts = prepared["images"], prepared["contexts"]
    by_key = {(image.series_id, image.instance): i for i, image in enumerate(images)}
    keys = [
        (str(row["selected_slices"][i]["series_id"]), int(row["selected_slices"][i]["instance"]))
        for i in selected
    ]
    if any(key not in by_key for key in keys):
        raise ValueError("Predicted anchor missing from verified source")
    indices = np.asarray([by_key[key] for key in keys])
    with torch.inference_mode():
        maps = _full_fov_maps(encoder, _inputs(images, contexts, indices).to(device))
        xy, _ = center_model(maps)
        xy = xy.detach().cpu().numpy()
    if xy.shape != (2, 2) or not np.isfinite(xy).all():
        raise ValueError("Invalid predicted central-canal positions")
    prediction = {
        "levels": {
            level: {
                "status": "ok",
                "series_id": key[0],
                "instance": key[1],
                "center_xy_norm": xy[i].tolist(),
                "anchor_selected_index": selected[i],
                "anchor_weight": float(weights[i, selected[i]]),
            }
            for i, (level, key) in enumerate(zip(LEVELS, keys))
        }
    }
    crops = crop_level_inputs(
        row,
        prediction,
        roi_mm=protocol["roi_mm"],
        max_centers=protocol["roi_max_centers"],
        window_mm=protocol["roi_window_mm"],
        _prepared=prepared,
    )
    vectors, targets = [], []
    with torch.inference_mode():
        for level in LEVELS:
            result = crops["levels"][level]
            item = {
                "level": level,
                "status": "not_assessed",
                "reason": result["reason"],
                "feature_index": None,
            }
            if result["status"] == "ok":
                encoded = (
                    encoder(result["images"].to(device))
                    .mean(0)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                if encoded.shape != (512,) or not np.isfinite(encoded).all():
                    raise ValueError("Invalid localized frozen ROI feature")
                item.update(status="ok", reason="", feature_index=len(vectors))
                vectors.append(encoded)
            targets.append(item)
    provenance = {
        "prediction": prediction,
        "slice_weights": weights.tolist(),
        "crops": {
            level: {key: value for key, value in result.items() if key != "images"}
            for level, result in crops["levels"].items()
        },
        "parameters": crops["parameters"],
        "source_qc": crops["source_qc"],
    }
    return (
        np.stack(vectors).astype(np.float32) if vectors else np.empty((0, 512), np.float32),
        targets,
        provenance,
    )


def _reference_labels(rows, phase, rsna_labels, aisslab_records):
    """Reference-only join after image features; never used in inference."""
    result, receipts = {}, {}
    if any(r["dataset"] == "rsna" for r in rows):
        mapping = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2, "": None}
        with Path(rsna_labels).open(newline="") as stream:
            for row in csv.DictReader(stream):
                study = row["study_id"]
                for level, column in zip(
                    LEVELS, ("spinal_canal_stenosis_l4_l5", "spinal_canal_stenosis_l5_s1")
                ):
                    value = row[column]
                    key = "rsna", study, level
                    if key in result or value not in mapping:
                        raise ValueError("Duplicate RSNA reference or invalid grade")
                    result[key] = mapping[value]
        receipts["rsna_labels_sha256"] = sha256_file(Path(rsna_labels))
    if any(r["dataset"] == "aisslab" for r in rows):
        if phase != "evaluation_image_inputs":
            raise ValueError("AISSLab grades cannot enter severity fitting/selection")
        for row in _json(aisslab_records):
            if row["split"] != "test":
                continue
            for level, label in zip(LEVELS, row["labels"]):
                key = "aisslab", row["study_id"], level
                if key in result or (
                    label is not None and (type(label) is not int or label not in (0, 1, 2))
                ):
                    raise ValueError("Invalid external TEST grade")
                result[key] = label
        receipts["aisslab_records_sha256"] = sha256_file(Path(aisslab_records))
    return result, receipts


def _image_record_signature(row):
    return af._hash_json(
        {
            key: row.get(key)
            for key in (
                "dataset",
                "study_id",
                "split",
                "status",
                "reason",
                "arrays_path",
                "arrays_sha256",
                "source_root",
                "source",
            )
        }
        | {
            "selected_slices": [
                {"series_id": item["series_id"], "instance": item["instance"]}
                for item in row.get("selected_slices", [])
            ]
        }
    )


def extract_localized_features(
    index_path,
    localizer_selection,
    out,
    protocol,
    device="mps",
    datasets=("rsna",),
    rsna_labels="data/rsna/train.csv",
    aisslab_records="artifacts/aisslab-axial-v4/records.json",
):
    index_path, localizer_selection, out, protocol = map(
        Path, (index_path, localizer_selection, out, protocol)
    )
    index_sha, selection_sha = sha256_file(index_path), sha256_file(localizer_selection)
    index, seal, checkpoint, phase = verify_localizer_selection(
        index_path, localizer_selection, protocol
    )
    if (
        not datasets
        or set(datasets) - {"rsna", "aisslab"}
        or (phase == "fitting" and "aisslab" in datasets)
    ):
        raise ValueError("Severity fitting accepts RSNA only; external grades require sealed TEST")
    rows = [r for r in index["records"] if r["dataset"] in datasets]
    if not rows:
        raise ValueError("No selected ROI extraction studies")
    device = af._device(device)
    torch.set_num_threads(4)
    config = _json(protocol)
    configuration = {
        "schema_version": 1,
        "phase": phase,
        "input_index_sha256": index_sha,
        "localizer_selection_sha256": selection_sha,
        "localizer_checkpoint_sha256": seal["checkpoint_sha256"],
        "protocol_sha256": sha256_file(protocol),
        "datasets": list(datasets),
        "original_encoder": af._weight_signature(WEIGHTS),
        "implementation_sha256": sha256_file(Path(__file__)),
        "crop_implementation_sha256": sha256_file(
            Path(__file__).with_name("axial_localizer_crops.py")
        ),
        "decoder_implementation_sha256": sha256_file(Path(af.__file__)),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "records").mkdir(exist_ok=True)
    configuration_path = out / "configuration.json"
    if configuration_path.exists() and _json(configuration_path) != configuration:
        raise ValueError("Localized ROI cache provenance changed; use a new directory")
    write_json(configuration_path, configuration)
    slice_model, center_model, header = localizer.load_localizer_checkpoint(checkpoint, device)
    for key in ("protocol_sha256", "training_index_sha256", "implementation_sha256"):
        if header.get(key) != seal.get(key):
            raise ValueError("Localizer checkpoint provenance differs from selection")
    encoder = af._load_encoder(WEIGHTS, device)
    started, records = time.monotonic(), []
    for number, row in enumerate(rows):
        token = hashlib.sha256(f"{row['dataset']}:{row['study_id']}".encode()).hexdigest()[:24]
        record_path, feature_path = (
            out / "records" / f"{token}.json",
            out / "records" / f"{token}.npz",
        )
        signature = _image_record_signature(row)
        if record_path.exists():
            record = _json(record_path)
            if (
                record["image_record_sha256"] != signature
                or sha256_file(feature_path) != record["features_sha256"]
            ):
                raise ValueError("Resumed ROI image inputs/features changed")
        else:
            record = {key: row[key] for key in ("dataset", "study_id", "split")}
            record.update(
                image_record_sha256=signature, features_path=str(feature_path.relative_to(out))
            )
            if row["status"] != "ok":
                vectors, targets, provenance = (
                    np.empty((0, 512), np.float32),
                    [
                        {
                            "level": level,
                            "status": "failed",
                            "reason": row["reason"] or "source_not_processed",
                            "feature_index": None,
                        }
                        for level in LEVELS
                    ],
                    {},
                )
            else:
                features, positions = read_image_inputs(index_path, row)
                try:
                    vectors, targets, provenance = infer_localized_study(
                        row, features, positions, slice_model, center_model, encoder, config, device
                    )
                except af.AxialDataError as error:
                    vectors, targets, provenance = (
                        np.empty((0, 512), np.float32),
                        [
                            {
                                "level": level,
                                "status": "failed",
                                "reason": error.reason,
                                "feature_index": None,
                            }
                            for level in LEVELS
                        ],
                        {},
                    )
            temporary = feature_path.with_suffix(".npz.part")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, features=vectors)
            temporary.replace(feature_path)
            record.update(
                features_sha256=sha256_file(feature_path), targets=targets, provenance=provenance
            )
            write_json(record_path, record)
        records.append(record)
        if (number + 1) % 50 == 0 or number + 1 == len(rows):
            print(f"Predicted axial ROI features {number + 1}/{len(rows)}", flush=True)
    # Open grades only after the complete image-only inference/cache stage.
    labels, label_receipts = _reference_labels(rows, phase, rsna_labels, aisslab_records)
    aggregate, targets = [], []
    for record in records:
        with np.load(out / record["features_path"], allow_pickle=False) as archive:
            features = archive["features"].copy()
        base = len(aggregate)
        aggregate.extend(features)
        for item in record["targets"]:
            key = record["dataset"], record["study_id"], item["level"]
            target = (
                {key: record[key] for key in ("dataset", "study_id", "split")}
                | item
                | {"label": labels.get(key)}
            )
            if target["feature_index"] is not None:
                target["feature_index"] += base
            targets.append(target)
    features = (
        np.stack(aggregate).astype(np.float32) if aggregate else np.empty((0, 512), np.float32)
    )
    np.savez_compressed(out / "features.npz", features=features)
    with (out / "targets.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(targets)
    if sha256_file(index_path) != index_sha or sha256_file(localizer_selection) != selection_sha:
        raise ValueError("Image inputs or localizer selection changed during ROI extraction")
    summary = {
        "schema_version": 1,
        "configuration": configuration,
        "label_provenance": label_receipts,
        "n_studies": len(records),
        "n_targets": len(targets),
        "n_features": len(features),
        "by_status": dict(Counter(r["status"] for r in targets)),
        "features_sha256": sha256_file(out / "features.npz"),
        "targets_sha256": sha256_file(out / "targets.csv"),
        "seconds": time.monotonic() - started,
        "grade_labels_joined_after_image_inference": True,
    }
    write_json(out / "summary.json", summary)
    return summary
