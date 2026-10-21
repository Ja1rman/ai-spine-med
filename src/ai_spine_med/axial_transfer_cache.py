"""Cache the unchanged ImageNet prefix for the separate axial transfer experiment.

Source inventories are prepared before this module is called. Test inventories
are separate from fitting inventories; no reference coordinate selects an image.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from torchvision.transforms.functional import resize

from .assets import sha256_file, write_json
from .axial_features import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    AxialDataError,
    _load_encoder,
    _study_inputs,
    _weight_signature,
)


def _relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError("Unsafe source inventory path")
    return path


def validate_sources(record: dict) -> Path:
    """Reject escaped paths before opening a single medical file."""
    root = Path(record["source_root"]).resolve(strict=True)
    series_ids, sources = set(), set()
    for series in record["source"]["series"]:
        sid = str(series["series_id"])
        if not sid.isdigit() or sid in series_ids:
            raise ValueError("Require distinct numeric series identifiers")
        series_ids.add(sid)
        directory = root.joinpath(*_relative(series["series_path"]).parts)
        for entry in series["files"]:
            name = _relative(entry["name"])
            if len(name.parts) != 1:
                raise ValueError("DICOM filename must be a basename")
            target = directory / name.name
            if (
                target.is_symlink()
                or not target.resolve(strict=True).is_relative_to(root)
                or target.resolve() in sources
                or not target.is_file()
            ):
                raise ValueError("Unsafe or duplicate DICOM source")
            sources.add(target.resolve())
            if target.stat().st_size != entry["bytes"]:
                raise ValueError("DICOM source size differs from sealed inventory")
            digest = entry["sha256"]
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError("DICOM source checksum is absent")
    if not sources:
        raise ValueError("Empty DICOM source inventory")
    return root


def select_slice_indices(count: int, maximum: int = 96) -> np.ndarray:
    if count < 1 or maximum != 96:
        raise ValueError("Require a nonempty bag and the frozen 96-slice policy")
    return (
        np.arange(count, dtype=np.int64)
        if count <= maximum
        else np.linspace(0, count - 1, maximum).astype(np.int64)
    )


def _inputs(images, contexts, selected):
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    tensors = []
    for number in selected:
        image = images[int(number)]
        channels = [
            resize(
                torch.from_numpy(neighbor.image.copy())[None],
                [IMAGE_SIZE, IMAGE_SIZE],
                antialias=True,
            )[0]
            for neighbor in contexts[(image.series_id, image.instance)]
        ]
        tensors.append((torch.stack(channels) - mean) / std)
    return torch.stack(tensors)


def build_transfer_cache(
    sources: str | Path,
    out: str | Path,
    *,
    weights: str | Path = ".cache/axial-v3/resnet18-f37072fd.pth",
    device: str = "auto",
    batch_size: int = 16,
    threads: int = 4,
    selection: str | Path | None = None,
) -> dict:
    from .axial_transfer import build_frozen_prefix

    sources, out, weights = Path(sources), Path(out), Path(weights)
    source_hash = sha256_file(sources)
    inventory = json.loads(sources.read_text())
    rows = inventory["records"]
    if not rows or batch_size < 1:
        raise ValueError("Empty source inventory or invalid batch size")
    keys = [(row["dataset"], row["study_id"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("A patient occurs more than once in the source inventory")
    for row in rows:
        if (
            row["dataset"] not in {"aisslab", "rsna"}
            or row["split"] not in {"train", "validation", "test"}
            or len(row["labels"]) != 2
            or any(
                x is not None and (type(x) is not int or x not in (0, 1, 2)) for x in row["labels"]
            )
        ):
            raise ValueError("Invalid patient/level target inventory")
    scope = inventory.get("scope")
    if scope == "external_axial_v4_fitting":
        if any(
            row["split"] == "test" or (row["dataset"] == "rsna" and row["split"] != "train")
            for row in rows
        ):
            raise ValueError("TEST or original validation cannot enter a fitting cache")
    elif scope in {"external_axial_v4_new_test", "external_axial_v4_rsna_test"}:
        if any(row["split"] != "test" for row in rows) or selection is None:
            raise ValueError("A test cache requires only test patients and sealed trained models")
        selection = Path(selection)
        seal = json.loads(selection.read_text())
        summary_path = selection.parent / "summary.json"
        if (
            seal.get("frozen") is not True
            or seal.get("model_family") != "axial_transfer_v1"
            or seal.get("seeds") != [42, 2024, 3407]
            or not summary_path.is_file()
            or sha256_file(summary_path) != seal.get("training_summary_sha256")
            or json.loads(summary_path.read_text()).get("status") != "complete"
        ):
            raise ValueError("Test cache requires a complete unchanged training selection")
        runs = seal.get("runs", [])
        if [x.get("seed") for x in runs] != [42, 2024, 3407]:
            raise ValueError("Test cache requires all three selected seeds")
        for run in runs:
            for key in ("checkpoint", "before_checkpoint"):
                candidate = selection.parent.joinpath(*_relative(run[key]).parts)
                if sha256_file(candidate) != run[f"{key}_sha256"]:
                    raise ValueError("Selected checkpoint changed before test preparation")
    else:
        raise ValueError("Unknown transfer cache phase")
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device not in {"cpu", "mps"}:
        raise ValueError("Use cpu, mps or auto for the local transfer cache")
    torch.set_num_threads(threads)
    signature = _weight_signature(weights)
    configuration = {
        "schema_version": 1,
        "source_inventory_sha256": source_hash,
        "encoder": signature,
        "max_slices": 96,
        "activation_shape": [512, 7, 7],
        "activation_dtype": "float32",
        "implementation_sha256": sha256_file(Path(__file__)),
        "decoder_sha256": sha256_file(Path(__file__).with_name("axial_features.py")),
        "prefix_helper_sha256": hashlib.sha256(
            inspect.getsource(build_frozen_prefix).encode()
        ).hexdigest(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "records").mkdir(exist_ok=True)
    config_path = out / "configuration.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != configuration:
            raise ValueError("Cache provenance changed; use a new directory")
    else:
        write_json(config_path, configuration)
    prefix = build_frozen_prefix(weights, device=device)
    # A separately loaded complete encoder checks the prefix boundary once.
    full_encoder = _load_encoder(weights, device)
    boundary_checked = False
    started = time.monotonic()
    results = []
    for number, original in enumerate(rows):
        identity = hashlib.sha256(
            f"{original['dataset']}:{original['study_id']}".encode()
        ).hexdigest()[:24]
        receipt_path = out / "records" / f"{identity}.json"
        arrays_path = out / "records" / f"{identity}.npz"
        record_hash = hashlib.sha256(
            json.dumps(original, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        if receipt_path.exists():
            saved = json.loads(receipt_path.read_text())
            if saved.get("source_record_sha256") != record_hash:
                raise ValueError("Patient source inventory changed while resuming")
            if saved["status"] == "ok" and (
                not arrays_path.is_file() or sha256_file(arrays_path) != saved["arrays_sha256"]
            ):
                raise ValueError("Previously cached activations changed")
            results.append(saved)
            continue
        record = {key: original[key] for key in ("dataset", "study_id", "split", "labels")}
        record.update(
            status="not_assessed",
            reason="",
            arrays_path="",
            arrays_sha256="",
            source_record_sha256=record_hash,
            n_slices=0,
        )
        try:
            root = validate_sources(original)
            images, positions, contexts, qc = _study_inputs(root, original["source"])
            selected = select_slice_indices(len(images))
            activations = []
            with torch.inference_mode():
                for offset in range(0, len(selected), batch_size):
                    inputs = _inputs(images, contexts, selected[offset : offset + batch_size]).to(
                        device
                    )
                    output = prefix(inputs)
                    if not boundary_checked:
                        # layer4[1] and pooling are still the original weights here.
                        reconstructed = full_encoder.layer4[1](output)
                        reconstructed = full_encoder.avgpool(reconstructed).flatten(1)
                        complete = full_encoder(inputs)
                        if not torch.allclose(reconstructed, complete, rtol=1e-4, atol=1e-4):
                            raise RuntimeError("Frozen ResNet prefix boundary parity failed")
                        boundary_checked = True
                    activations.append(output.detach().cpu().numpy().astype(np.float32))
            values = np.concatenate(activations)
            if values.shape != (len(selected), 512, 7, 7) or not np.isfinite(values).all():
                raise RuntimeError("Invalid frozen prefix activations")
            temporary = arrays_path.with_suffix(".npz.part")
            with temporary.open("wb") as handle:
                np.savez(
                    handle, activations=values, positions=positions[selected].astype(np.float32)
                )
            temporary.replace(arrays_path)
            record.update(
                status="ok",
                arrays_path=str(arrays_path.relative_to(out)),
                arrays_sha256=sha256_file(arrays_path),
                n_slices=len(selected),
                n_source_slices=len(images),
                source_dicom_study_uid_sha256=qc["series"][0]["study_uid_sha256"],
                source_dicom_series_uid_sha256=[x["series_uid_sha256"] for x in qc["series"]],
            )
        except AxialDataError as error:
            record["reason"] = error.reason
        write_json(receipt_path, record)
        results.append(record)
        if (number + 1) % 20 == 0 or number + 1 == len(rows):
            print(
                json.dumps(
                    {
                        "stage": "activation_cache",
                        "completed": number + 1,
                        "total": len(rows),
                        "seconds": round(time.monotonic() - started, 1),
                    }
                ),
                flush=True,
            )
    if sha256_file(sources) != source_hash or _weight_signature(weights) != signature:
        raise RuntimeError("Source inventory or encoder changed during extraction")
    index = {
        "schema_version": 1,
        "configuration": configuration,
        "provenance": inventory.get("provenance", {}),
        "records": results,
    }
    write_json(out / "index.json", index)
    summary = {
        "n_studies": len(results),
        "assessed_studies": sum(row["status"] == "ok" for row in results),
        "by_dataset_split": dict(Counter(f"{x['dataset']}/{x['split']}" for x in results)),
        "failures": dict(Counter(x["reason"] for x in results if x["status"] != "ok")),
        "selected_slices": sum(x["n_slices"] for x in results),
        "source_inventory_sha256": source_hash,
        "index_sha256": sha256_file(out / "index.json"),
        "prefix_boundary_parity_checked": boundary_checked
        or any(x["status"] == "ok" for x in results),
        "seconds": time.monotonic() - started,
    }
    write_json(out / "summary.json", summary)
    return summary
