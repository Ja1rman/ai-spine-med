"""Sealed evaluation image inputs and separate reference-only localization truth.

Image inputs contain only original ImageNet features/positions and source
geometry. Grades, localization annotations and centers are separate reference
files. Fitting and selection must finish before opening TEST medical files.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .assets import sha256_file, write_json
from .axial_features import WEIGHTS_SHA256
from .axial_localization_audit import LEVELS, _coordinate_rows, _rsna_anchors, reconstruct_slice_bag
from .axial_localizer_data import (
    _original_block,
    _safe_relative,
    build_targets,
    canonical_geometry,
    xml_ccs_anchor,
)


def _json(path):
    return json.loads(Path(path).read_text())


def verify_evaluation_seals(
    localizer_selection: str | Path,
    severity_selection: str | Path,
    protocol: str | Path,
    fitting_index: str | Path,
    severity_training_index: str | Path,
) -> dict:
    """Verify both frozen selections and all checkpoint bytes, without TEST I/O."""
    localizer_selection, severity_selection, protocol, fitting_index, severity_training_index = map(
        Path,
        (localizer_selection, severity_selection, protocol, fitting_index, severity_training_index),
    )
    localizer, severity = _json(localizer_selection), _json(severity_selection)
    protocol_sha = sha256_file(protocol)
    protocol_canonical_sha = hashlib.sha256(
        json.dumps(_json(protocol), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if (
        localizer.get("schema_version") != 1
        or localizer.get("model_family") != "explicit_axial_localizer_v1"
        or localizer.get("frozen") is not True
        or localizer.get("localization_gate_passed") is not True
        or localizer.get("protocol_sha256") != protocol_sha
        or localizer.get("training_index_sha256") != sha256_file(fitting_index)
    ):
        raise ValueError("TEST requires an unchanged, frozen localizer with passed validation gate")
    checkpoint = _safe_relative(localizer_selection.parent, localizer["checkpoint"])
    if sha256_file(checkpoint) != localizer.get("checkpoint_sha256"):
        raise ValueError("Selected localizer checkpoint changed before TEST")
    if (
        severity.get("schema_version") != 1
        or severity.get("family") != "localized_stenosis_v1"
        or severity.get("status") != "sealed_before_test"
        or severity.get("frozen") is not True
        or severity.get("seeds") != [42, 2024, 3407]
        or severity.get("protocol_sha256") != protocol_canonical_sha
        or severity.get("training_index_sha256") != sha256_file(severity_training_index)
        or [r.get("seed") for r in severity.get("runs", [])] != [42, 2024, 3407]
    ):
        raise ValueError("TEST requires all three unchanged frozen severity models")
    for run in severity["runs"]:
        path = _safe_relative(severity_selection.parent, run["checkpoint"])
        if sha256_file(path) != run.get("checkpoint_sha256"):
            raise ValueError("Selected severity checkpoint changed before TEST")
    for filename, field in (
        ("manifest.json", "manifest_sha256"),
        ("summary.json", "summary_sha256"),
    ):
        if (
            not severity.get(field)
            or sha256_file(severity_selection.parent / filename) != severity[field]
        ):
            raise ValueError("Severity training summary/manifest changed before TEST")
    return {
        "localizer_selection_sha256": sha256_file(localizer_selection),
        "severity_selection_sha256": sha256_file(severity_selection),
        "protocol_file_sha256": protocol_sha,
        "protocol_canonical_sha256": protocol_canonical_sha,
        "fitting_index_sha256": sha256_file(fitting_index),
        "severity_training_index_sha256": sha256_file(severity_training_index),
        "both_selections_frozen_before_test": True,
        "localization_gate_passed": True,
    }


def image_features(activations: np.ndarray, positions: np.ndarray, block, device: str) -> dict:
    """Complete the original encoder. This API accepts no labels/annotations."""
    if (
        activations.ndim != 4
        or activations.shape[1:] != (512, 7, 7)
        or positions.shape != (len(activations), 2)
        or not len(activations)
        or not np.isfinite(activations).all()
        or not np.isfinite(positions).all()
    ):
        raise ValueError("Invalid frozen TEST image activation geometry")
    parts = []
    with torch.inference_mode():
        for start in range(0, len(activations), 16):
            maps = block(torch.from_numpy(activations[start : start + 16]).to(device))
            parts.append(maps.mean(dim=(2, 3)).cpu().numpy())
    return {
        "features": np.concatenate(parts).astype(np.float32),
        "positions": positions.astype(np.float32),
    }


def prepare_evaluation_data(
    localizer_selection: str | Path,
    severity_selection: str | Path,
    protocol: str | Path,
    severity_training_index: str | Path,
    *,
    fitting_index: str | Path = "artifacts/axial-localizer-v6/index.json",
    out: str | Path = "artifacts/axial-localizer-test-v6",
    sources: str | Path = "artifacts/axial-transfer-sources-v4/evaluation-sources.json",
    activation_index: str | Path = "artifacts/axial-transfer-test-v4/index.json",
    prepared: str | Path = "artifacts/aisslab-axial-v4",
    rsna_root: str | Path = "data/rsna",
    weights: str | Path = ".cache/axial-v3/resnet18-f37072fd.pth",
    device: str = "cpu",
    sigma_mm: float = 5.0,
    threads: int = 4,
) -> dict:
    """Prepare image-only TEST inputs after both model choices are immutable.

    Reference-only NPZ/JSON exports are written in references/ and never included
    in an inference input record. They are for evaluation, not image selection.
    """
    receipt = verify_evaluation_seals(
        localizer_selection, severity_selection, protocol, fitting_index, severity_training_index
    )
    out, sources, activation_index, prepared, rsna_root, weights = map(
        Path, (out, sources, activation_index, prepared, rsna_root, weights)
    )
    source_data, cache = _json(sources), _json(activation_index)
    rows = source_data["records"]
    if not rows or any(r["split"] != "test" for r in rows):
        raise ValueError("Explicit TEST preparation accepts only held-out TEST patients")
    by_key = {(r["dataset"], r["study_id"]): r for r in cache["records"]}
    keys = [(r["dataset"], r["study_id"]) for r in rows]
    if (
        len(keys) != len(set(keys))
        or set(keys) != set(by_key)
        or len(by_key) != len(cache["records"])
    ):
        raise ValueError("Duplicate or mismatched TEST patient inventories")
    if cache["configuration"]["encoder"]["sha256"] != WEIGHTS_SHA256 or cache["configuration"][
        "source_inventory_sha256"
    ] != sha256_file(sources):
        raise ValueError("TEST activations differ from original frozen image sources")
    if device not in {"cpu", "mps"}:
        raise ValueError("Use cpu or mps for original encoder completion")
    torch.set_num_threads(threads)
    configuration = {
        "schema_version": 1,
        "phase": "evaluation_image_inputs",
        "levels": list(LEVELS),
        "seals": receipt,
        "sources_sha256": sha256_file(sources),
        "activation_index_sha256": sha256_file(activation_index),
        "original_weights_sha256": sha256_file(weights),
        "implementation_sha256": sha256_file(Path(__file__)),
        "fitting_implementation_sha256": sha256_file(
            Path(__file__).with_name("axial_localizer_data.py")
        ),
        "input_fields": ["features", "positions"],
        "reference_fields_in_inputs": False,
    }
    out.mkdir(parents=True, exist_ok=True)
    conf_path = out / "configuration.json"
    if conf_path.exists() and _json(conf_path) != configuration:
        raise ValueError("Evaluation input provenance changed; use a new directory")
    write_json(conf_path, configuration)
    block = _original_block(weights, device)
    coordinates = _coordinate_rows(rsna_root)
    external = {r["study_id"]: r for r in _json(prepared / "records.json")}
    (out / "records").mkdir(exist_ok=True)
    (out / "references").mkdir(exist_ok=True)
    inputs, references, counts = [], [], Counter()
    for number, row in enumerate(rows):
        entry = by_key[(row["dataset"], row["study_id"])]
        if entry["status"] != "ok" or entry["split"] != "test":
            raise ValueError("TEST inputs need a successful same-split frozen activation")
        selected, full = reconstruct_slice_bag(row)
        for item in selected:
            item.update(canonical_geometry(item))
        prefix_path = _safe_relative(activation_index.parent, entry["arrays_path"])
        if sha256_file(prefix_path) != entry["arrays_sha256"]:
            raise ValueError("Frozen TEST activation archive changed")
        with np.load(prefix_path, allow_pickle=False) as archive:
            positions = archive["positions"].copy()
            activations = archive["activations"].copy()
        np.testing.assert_array_equal(
            positions, np.asarray([s["positions"] for s in selected], dtype=np.float32)
        )
        arrays = image_features(activations, positions, block, device)
        token = hashlib.sha256(f"{row['dataset']}:{row['study_id']}".encode()).hexdigest()[:24]
        path = out / "records" / f"{token}.npz"
        np.savez_compressed(path, **arrays)
        inputs.append(
            {
                "dataset": row["dataset"],
                "study_id": row["study_id"],
                "split": "test",
                "status": "ok",
                "reason": "",
                "arrays_path": str(path.relative_to(out)),
                "arrays_sha256": sha256_file(path),
                "n_slices": len(selected),
                "selected_slices": selected,
                "source_root": row["source_root"],
                "source": row["source"],
            }
        )
        # References below this boundary cannot affect image features or descriptors.
        if row["dataset"] == "aisslab":
            original = external[row["study_id"]]
            if original["split"] != "test" or original["source"] != row["source"]:
                raise ValueError("External TEST identity differs from immutable preparation")
            by_path = {s["path"]: s for s in full}
            anchors = [
                xml_ccs_anchor(Path(row["source_root"]), ref, by_path[ref["dicom_path"]])
                for ref in original["label_references"]
            ]
        else:
            anchors, _ = _rsna_anchors(row, full, coordinates, rsna_root)
        for anchor in anchors:
            anchor.pop("grade", None)
        target_arrays = build_targets(selected, anchors, sigma_mm=sigma_mm)
        reference_path = out / "references" / f"{token}.npz"
        np.savez_compressed(reference_path, **target_arrays)
        references.append(
            {
                "dataset": row["dataset"],
                "study_id": row["study_id"],
                "split": "test",
                "labels": row["labels"],
                "anchors": anchors,
                "arrays_path": str(reference_path.relative_to(out)),
                "arrays_sha256": sha256_file(reference_path),
            }
        )
        counts[f"{row['dataset']}/studies"] += 1
        if (number + 1) % 50 == 0 or number + 1 == len(rows):
            print(f"Sealed TEST image inputs {number + 1}/{len(rows)}", flush=True)
    result = {
        "schema_version": 1,
        "configuration": configuration,
        "records": inputs,
        "counts": dict(counts),
    }
    write_json(out / "index.json", result)
    write_json(
        out / "references.json",
        {
            "schema_version": 1,
            "scope": "TEST reference-only scoring; never an inference input",
            "input_index_sha256": sha256_file(out / "index.json"),
            "sigma_mm": sigma_mm,
            "aisslab_records_sha256": sha256_file(prepared / "records.json"),
            "rsna_coordinates_sha256": sha256_file(rsna_root / "train_label_coordinates.csv"),
            "records": references,
        },
    )
    return result
