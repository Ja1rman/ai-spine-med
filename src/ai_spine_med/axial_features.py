"""Frozen full-FOV axial T2 slice bags, without labels or sagittal guidance.

Every requested series contributes every decoded slice. An incomplete study
remains pending; any invalid downloaded series fails its entire study. Failures
are explicit. This module does not read grade or coordinate CSV files.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import time
import uuid
import warnings
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
import torch
import torchvision
from torchvision.models import resnet18
from torchvision.transforms.functional import resize

from . import rsna_bulk

WEIGHTS_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
WEIGHTS_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
ALLOWED_SPLITS = ("train", "validation")
MAX_AXIAL_OBLIQUITY = 45.0
MAX_INPLANE_ROTATION = 20.0
CONTEXT_ANGLE_DEGREES = 15.0
CONTEXT_DISTANCE_MM = 10.0
BATCH_SIZE = 32
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class AxialDataError(ValueError):
    """A stable, safe failure code; do not serialize decoder exception text."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class SourceNotReady(AxialDataError):
    pass


@dataclass
class _Slice:
    image: np.ndarray
    center_lps: np.ndarray
    normal: np.ndarray
    instance: int
    series_id: str
    orientation_action: str
    sop_uid_sha256: str = ""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _hash_json(value: Any) -> str:
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _lock(out: Path):
    import fcntl

    with (out / ".extract.lock").open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("An axial extractor already owns this output directory") from None
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _manifest(data_root: Path, manifest_path: Path) -> tuple[list[dict], dict]:
    from .axial_data import validate_axial_manifest

    rows, summary = validate_axial_manifest(data_root, manifest_path)
    if any(row["split"] not in ALLOWED_SPLITS for row in rows):
        raise ValueError("Axial extraction accepts only original train and validation studies")
    groups: dict[str, str] = {}
    seen = set()
    for row in rows:
        if row["study_id"] in groups and groups[row["study_id"]] != row["split"]:
            raise ValueError("An axial study crosses original train/validation splits")
        groups[row["study_id"]] = row["split"]
        key = row["study_id"], row["series_id"]
        if key in seen or row["series_description"] != "Axial T2":
            raise ValueError("Duplicate series or non-axial description in axial manifest")
        seen.add(key)
        expected = f"train_images/{key[0]}/{key[1]}"
        if row["series_path"] != expected or not all(str(value).isdigit() for value in key):
            raise ValueError("Unsafe axial study/series path")
    return rows, summary


def _device(device: str) -> str:
    if device == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if device not in {"cpu", "mps"}:
        raise ValueError("Axial feature extraction supports cpu, mps or auto")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable in this execution environment")
    return device


def _weight_signature(weights_path: Path) -> dict:
    digest = _sha(weights_path)
    if digest != WEIGHTS_SHA256:
        raise ValueError("ResNet18 weights differ from official IMAGENET1K_V1 bytes")
    return {
        "architecture": "torchvision.resnet18",
        "weights": "IMAGENET1K_V1",
        "url": WEIGHTS_URL,
        "sha256": digest,
    }


def _load_encoder(weights_path: Path, device: str) -> torch.nn.Module:
    model = resnet18(weights=None)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.fc = torch.nn.Identity()
    return model.eval().requires_grad_(False).to(device)


def _implementation_signature() -> dict:
    from . import axial_data

    return {
        "axial_features.py": _sha(Path(__file__)),
        "axial_data.py": _sha(Path(axial_data.__file__)),
        "rsna_bulk.py": _sha(Path(rsna_bulk.__file__)),
    }


def _numbers(ds: pydicom.Dataset, name: str, count: int) -> np.ndarray:
    try:
        values = np.asarray(getattr(ds, name), dtype=np.float64)
    except (AttributeError, TypeError, ValueError):
        raise AxialDataError("missing_or_invalid_geometry") from None
    if values.shape != (count,) or not np.isfinite(values).all():
        raise AxialDataError("missing_or_invalid_geometry")
    return values


def _canonical_image(
    image: np.ndarray, column_axis: np.ndarray, row_axis: np.ndarray
) -> tuple[np.ndarray, str]:
    """Anterior up, patient left on image right; retain supported oblique plane."""
    column_xy = np.asarray(column_axis[:2], dtype=np.float64).copy()
    row_xy = np.asarray(row_axis[:2], dtype=np.float64).copy()
    column_norm, row_norm = np.linalg.norm(column_xy), np.linalg.norm(row_xy)
    if min(column_norm, row_norm) < 1e-6:
        raise AxialDataError("unsupported_inplane_rotation")
    column_xy /= column_norm
    row_xy /= row_norm
    transpose = abs(column_xy[1]) > abs(column_xy[0])
    if transpose:
        image = image.T
        column_xy, row_xy = row_xy, column_xy
    cosine = math.cos(math.radians(MAX_INPLANE_ROTATION))
    if abs(column_xy[0]) < cosine or abs(row_xy[1]) < cosine:
        raise AxialDataError("unsupported_inplane_rotation")
    flip_columns, flip_rows = column_xy[0] < 0, row_xy[1] < 0
    if flip_columns:
        image = image[:, ::-1]
    if flip_rows:
        image = image[::-1, :]
    action = (
        f"transpose={int(transpose)},flip_rows={int(flip_rows)},flip_columns={int(flip_columns)}"
    )
    return np.ascontiguousarray(image, dtype=np.float32), action


def _study_source(data_root: Path, rows: list[dict], manifest_sha: str) -> dict:
    receipt = []
    for row in sorted(rows, key=lambda item: item["series_id"]):
        directory = data_root / row["series_path"]
        if not directory.resolve().is_relative_to(data_root.resolve()):
            raise AxialDataError("unsafe_source_path")
        if not rsna_bulk.is_complete_series(directory):
            raise SourceNotReady("incomplete_series_download")
        marker_path = directory / rsna_bulk.COMPLETE_MARKER
        marker = json.loads(marker_path.read_text())
        if (
            marker.get("manifest_sha256") != manifest_sha
            or marker.get("modality") != "axial_v3"
            or marker.get("scope") != "axial_v3_train_validation"
            or marker.get("allowed_splits") != list(ALLOWED_SPLITS)
        ):
            raise SourceNotReady("incompatible_series_receipt")
        if any(path.suffix == ".part" for path in directory.iterdir()):
            raise SourceNotReady("series_download_in_progress")
        files = []
        for name in marker["files"]:
            path = directory / name
            if path.is_symlink():
                raise AxialDataError("unsafe_source_path")
            files.append({"name": name, "bytes": path.stat().st_size, "sha256": _sha(path)})
        receipt.append(
            {
                "series_id": row["series_id"],
                "series_path": row["series_path"],
                "completion_receipt_sha256": _sha(marker_path),
                "files": files,
            }
        )
    return {"series": receipt, "sha256": _hash_json(receipt)}


def _read_series(data_root: Path, source: dict) -> tuple[list[_Slice], dict]:
    images: list[_Slice] = []
    study_uid = series_uid = frame_uid = photometric = None
    sop_uids, instances = set(), set()
    missing_modality = 0
    for file in source["files"]:
        content = (data_root / source["series_path"] / file["name"]).read_bytes()
        if len(content) != file["bytes"] or hashlib.sha256(content).hexdigest() != file["sha256"]:
            raise SourceNotReady("source_changed_during_extraction")
        try:
            with warnings.catch_warnings(record=True):
                ds = pydicom.dcmread(io.BytesIO(content))
            uid = str(getattr(ds, "SeriesInstanceUID", ""))
            study = str(getattr(ds, "StudyInstanceUID", ""))
            frame = str(getattr(ds, "FrameOfReferenceUID", ""))
            sop = str(getattr(ds, "SOPInstanceUID", ""))
            if not uid or not study or not sop:
                raise AxialDataError("missing_dicom_identity")
            # RSNA deidentification can omit Modality. The extractor has already
            # verified the official Axial T2 series manifest before any pixels.
            # A present conflicting modality is never inferred or overwritten.
            if "Modality" not in ds:
                missing_modality += 1
            elif str(ds.Modality) != "MR":
                raise AxialDataError("unsupported_dicom_modality_or_multiframe")
            if int(getattr(ds, "NumberOfFrames", 1)) != 1:
                raise AxialDataError("unsupported_dicom_modality_or_multiframe")
            if int(getattr(ds, "SamplesPerPixel", 1)) != 1:
                raise AxialDataError("unsupported_color_image")
            current_photometric = str(getattr(ds, "PhotometricInterpretation", ""))
            if current_photometric not in {"MONOCHROME1", "MONOCHROME2"}:
                raise AxialDataError("unsupported_photometric_interpretation")
            if series_uid is None:
                series_uid, study_uid, frame_uid, photometric = (
                    uid,
                    study,
                    frame,
                    current_photometric,
                )
            if uid != series_uid or study != study_uid or frame != frame_uid:
                raise AxialDataError("mixed_dicom_identity_or_frame")
            if current_photometric != photometric:
                raise AxialDataError("inconsistent_photometric_interpretation")
            if sop in sop_uids:
                raise AxialDataError("duplicate_sop_instance_uid")
            sop_uids.add(sop)
            instance = int(ds.InstanceNumber)
            if instance < 0 or instance in instances:
                raise AxialDataError("missing_or_duplicate_instance_number")
            instances.add(instance)
            orientation = _numbers(ds, "ImageOrientationPatient", 6)
            column_axis, row_axis = orientation[:3], orientation[3:]
            if (
                not np.isclose(np.linalg.norm(column_axis), 1, atol=1e-3)
                or not np.isclose(np.linalg.norm(row_axis), 1, atol=1e-3)
                or abs(float(column_axis @ row_axis)) > 1e-3
            ):
                raise AxialDataError("nonorthonormal_image_orientation")
            normal = np.cross(column_axis, row_axis)
            normal /= np.linalg.norm(normal)
            if abs(normal[2]) < math.cos(math.radians(MAX_AXIAL_OBLIQUITY)):
                raise AxialDataError("not_axial_within_supported_obliquity")
            if normal[2] < 0:
                normal = -normal
            position = _numbers(ds, "ImagePositionPatient", 3)
            spacing = _numbers(ds, "PixelSpacing", 2)
            if np.any(spacing <= 0):
                raise AxialDataError("invalid_pixel_spacing")
            shape = (int(ds.Rows), int(ds.Columns))
            if min(shape) < 2:
                raise AxialDataError("invalid_image_shape")
            slope, intercept = (
                float(getattr(ds, "RescaleSlope", 1)),
                float(getattr(ds, "RescaleIntercept", 0)),
            )
            if not np.isfinite([slope, intercept]).all() or slope == 0:
                raise AxialDataError("invalid_rescale_parameters")
            with warnings.catch_warnings(record=True):
                image = ds.pixel_array.astype(np.float32) * slope + intercept
            if image.shape != shape or not np.isfinite(image).all():
                raise AxialDataError("invalid_pixel_array")
            center = (
                position
                + column_axis * spacing[1] * (shape[1] - 1) / 2
                + row_axis * spacing[0] * (shape[0] - 1) / 2
            )
            image, action = _canonical_image(image, column_axis, row_axis)
            images.append(
                _Slice(
                    image,
                    center,
                    normal,
                    instance,
                    source["series_id"],
                    action,
                    hashlib.sha256(sop.encode()).hexdigest(),
                )
            )
        except AxialDataError:
            raise
        except Exception:
            raise AxialDataError("dicom_decode_or_header_failed") from None
    if not images:
        raise AxialDataError("empty_axial_series")
    if photometric == "MONOCHROME1":
        lower = min(float(item.image.min()) for item in images)
        upper = max(float(item.image.max()) for item in images)
        for item in images:
            item.image = np.ascontiguousarray(lower + upper - item.image, dtype=np.float32)
    images.sort(key=lambda item: (-float(item.center_lps[2]), item.instance))
    pixels = np.concatenate([item.image.ravel() for item in images])
    lower, upper = np.percentile(pixels, [1, 99]).tolist()
    if not math.isfinite(lower + upper) or upper - lower <= 1e-6:
        raise AxialDataError("flat_series_intensity_distribution")
    for item in images:
        item.image = np.clip((item.image - lower) / (upper - lower), 0, 1).astype(np.float32)
    return images, {
        "series_id": source["series_id"],
        "n_slices": len(images),
        "study_uid_sha256": hashlib.sha256(study_uid.encode()).hexdigest(),
        "series_uid_sha256": hashlib.sha256(series_uid.encode()).hexdigest(),
        "frame_uid_sha256": hashlib.sha256(frame_uid.encode()).hexdigest() if frame_uid else None,
        "robust_intensity_percentiles": [lower, upper],
        "monochrome1_inverted": photometric == "MONOCHROME1",
        "missing_Modality_slice_count": missing_modality,
        "orientation_actions": dict(Counter(item.orientation_action for item in images)),
        "z_center_range_mm": [float(images[-1].center_lps[2]), float(images[0].center_lps[2])],
        "minimum_abs_normal_z": min(float(abs(item.normal[2])) for item in images),
    }


def _normalized_z(images: list[_Slice]) -> np.ndarray:
    z = np.asarray([item.center_lps[2] for item in images], dtype=np.float64)
    span = float(np.ptp(z))
    return (
        ((z.max() - z) / span).astype(np.float32)
        if span > 1e-6
        else np.full(len(z), 0.5, np.float32)
    )


def _neighbor_allowed(current: _Slice, other: _Slice) -> bool:
    distance = float(np.linalg.norm(current.center_lps - other.center_lps))
    return (
        current.series_id == other.series_id
        and 1e-3 < distance <= CONTEXT_DISTANCE_MM
        and float(current.normal @ other.normal) >= math.cos(math.radians(CONTEXT_ANGLE_DEGREES))
    )


def _study_inputs(data_root: Path, source: dict) -> tuple[list[_Slice], np.ndarray, dict, dict]:
    all_images: list[_Slice] = []
    series_qc, contexts, series_positions = [], {}, {}
    replicated = 0
    for series in source["series"]:
        images, qc = _read_series(data_root, series)
        series_qc.append(qc)
        for item, position in zip(images, _normalized_z(images)):
            series_positions[(item.series_id, item.instance)] = float(position)
        for index, image in enumerate(images):
            previous, following = image, image
            if index > 0 and _neighbor_allowed(image, images[index - 1]):
                previous = images[index - 1]
            else:
                replicated += 1
            if index + 1 < len(images) and _neighbor_allowed(image, images[index + 1]):
                following = images[index + 1]
            else:
                replicated += 1
            contexts[(image.series_id, image.instance)] = (previous, image, following)
        all_images.extend(images)
    if len({qc["study_uid_sha256"] for qc in series_qc}) != 1:
        raise AxialDataError("mixed_study_instance_uid")
    if len({(image.series_id, image.instance) for image in all_images}) != len(all_images):
        raise AxialDataError("duplicate_series_instance_key")
    # Verified RSNA deidentification may restart SOP UID counters per series.
    # Within-series SOP uniqueness is checked by _read_series; the bag identity
    # is the manifest series_id plus InstanceNumber, never a global SOP UID.
    sop_counts = Counter(image.sop_uid_sha256 for image in all_images)
    # Coordinate systems from different FrameOfReferenceUIDs are not registered.
    # Keep each frame as its own set; only compare physical z within that set.
    series_frames = {
        qc["series_id"]: qc["frame_uid_sha256"] or f"missing:{qc['series_id']}" for qc in series_qc
    }
    frame_groups: dict[str, list[_Slice]] = {}
    for image in all_images:
        frame_groups.setdefault(series_frames[image.series_id], []).append(image)
    ordered_frames = sorted(
        frame_groups,
        key=lambda frame: min(int(image.series_id) for image in frame_groups[frame]),
    )
    all_images, frame_positions, group_mapping = [], [], {}
    for group, frame in enumerate(ordered_frames):
        images = sorted(
            frame_groups[frame],
            key=lambda image: (-float(image.center_lps[2]), int(image.series_id), image.instance),
        )
        all_images.extend(images)
        frame_positions.extend(_normalized_z(images))
        for image in images:
            group_mapping[image.series_id] = group
    positions = np.asarray(
        [
            [frame_position, series_positions[(image.series_id, image.instance)]]
            for image, frame_position in zip(all_images, frame_positions)
        ],
        dtype=np.float32,
    )
    return (
        all_images,
        positions,
        contexts,
        {
            "series": series_qc,
            "context_channels_replicated": replicated,
            "n_context_channels": 2 * len(all_images),
            "all_series_share_frame_of_reference": len(ordered_frames) == 1
            and series_qc[0]["frame_uid_sha256"] is not None,
            "frame_group_count": len(ordered_frames),
            "series_frame_groups": group_mapping,
            "cross_series_reused_sop_uid_count": sum(count > 1 for count in sop_counts.values()),
            "cross_series_reused_sop_extra_slice_count": sum(
                count - 1 for count in sop_counts.values()
            ),
        },
    )


def _embed(
    images: list[_Slice], contexts: dict, encoder: torch.nn.Module, device: str
) -> np.ndarray:
    output = []
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32)[:, None, None]
    with torch.inference_mode():
        for start in range(0, len(images), BATCH_SIZE):
            tensors = []
            for image in images[start : start + BATCH_SIZE]:
                channels = []
                for neighbor in contexts[(image.series_id, image.instance)]:
                    single = torch.from_numpy(neighbor.image.copy())[None]
                    channels.append(resize(single, [IMAGE_SIZE, IMAGE_SIZE], antialias=True)[0])
                tensors.append((torch.stack(channels) - mean) / std)
            result = encoder(torch.stack(tensors).to(device)).detach().cpu().numpy()
            if result.shape != (len(tensors), 512) or not np.isfinite(result).all():
                raise RuntimeError(
                    "Frozen encoder returned invalid feature shape or nonfinite values"
                )
            output.append(result.astype(np.float32))
    return np.concatenate(output)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _aggregate(
    out: Path,
    groups: dict[str, list[dict]],
    config: dict,
    cache_id: str,
    *,
    record_index: dict | None = None,
    verify_arrays: bool = True,
) -> dict:
    records, outputs = [], {}
    complete = True
    for study, rows in sorted(groups.items()):
        record_path = f"records/{study}.json"
        path = out / record_path
        if not path.is_file():
            complete = False
            records.append(
                {
                    "study_id": study,
                    "split": rows[0]["split"],
                    "status": "pending",
                    "reason": "not_attempted",
                    "record_path": "",
                    "record_sha256": "",
                    "arrays_path": "",
                    "arrays_sha256": "",
                    "n_slices": 0,
                    "n_series": len(rows),
                    "source_ready": False,
                }
            )
            continue
        record, record_sha = (
            record_index[study]
            if record_index is not None
            else (json.loads(path.read_text()), _sha(path))
        )
        if (
            record.get("cache_id") != cache_id
            or record.get("study_id") != study
            or record.get("split") != rows[0]["split"]
        ):
            raise ValueError("Axial per-study record belongs to a different experiment")
        complete = (
            complete
            and record.get("source_ready") is True
            and record.get("status") in {"ok", "failed"}
        )
        outputs[record_path] = record_sha
        fields = (
            "study_id",
            "split",
            "status",
            "reason",
            "arrays_path",
            "arrays_sha256",
            "n_slices",
            "n_series",
            "source_ready",
        )
        entry = {field: record[field] for field in fields}
        entry.update(record_path=record_path, record_sha256=outputs[record_path])
        if record["status"] == "ok":
            if record["arrays_path"] != f"records/{study}.npz":
                raise ValueError("Unsafe axial bag path")
            if verify_arrays and _sha(out / record["arrays_path"]) != record["arrays_sha256"]:
                raise ValueError("Axial bag checksum changed after extraction")
            outputs[record["arrays_path"]] = record["arrays_sha256"]
        elif (
            record["status"] not in {"failed", "pending"}
            or record["arrays_path"]
            or record["arrays_sha256"]
        ):
            raise ValueError("Invalid axial failure record")
        records.append(entry)
    metadata = {
        "schema_version": 1,
        "scope": "train_validation",
        "cache_id": cache_id,
        "config": config,
        "extraction_complete": complete,
        "study_records": records,
        "outputs_sha256": outputs,
    }
    _atomic_json(out / "metadata.json", metadata)
    summary = {
        "schema_version": 1,
        "scope": "train_validation",
        "cache_id": cache_id,
        "extraction_complete": complete,
        "n_requested_studies": len(groups),
        "n_requested_series": sum(len(rows) for rows in groups.values()),
        "n_successful_studies": sum(record["status"] == "ok" for record in records),
        "n_slices": sum(record["n_slices"] for record in records if record["status"] == "ok"),
        "by_split_and_status": dict(
            Counter(record["split"] + ":" + record["status"] for record in records)
        ),
        "failed_by_reason": dict(
            Counter(record["reason"] for record in records if record["status"] == "failed")
        ),
        "metadata_sha256": _sha(out / "metadata.json"),
        "labels_or_coordinates_read": False,
        "sagittal_guidance_used": False,
    }
    _atomic_json(out / "summary.json", summary)
    return summary


def extract_axial_features(
    data_root: str | Path,
    manifest_path: str | Path,
    out: str | Path,
    weights_path: str | Path,
    device: str = "auto",
    limit: int | None = None,
    threads: int = 4,
) -> dict:
    if (
        type(threads) is not int
        or threads < 1
        or (limit is not None and (type(limit) is not int or limit < 1))
    ):
        raise ValueError("Threads and an optional debug limit must be positive integers")
    data_root, manifest_path, out, weights_path = map(
        Path, (data_root, manifest_path, out, weights_path)
    )
    rows, preparation = _manifest(data_root, manifest_path)
    if not rows:
        raise ValueError("Axial manifest has no original train/validation series")
    selected_device = _device(device)
    weight_signature = _weight_signature(weights_path)
    config = {
        "schema_version": 1,
        "allowed_splits": list(ALLOWED_SPLITS),
        "manifest_sha256": preparation["manifest_sha256"],
        "manifest_output_sha256": preparation["output_sha256"],
        "manifest_input_sha256": preparation["input_sha256"],
        "encoder": weight_signature,
        "feature_size": 512,
        "device": selected_device,
        "batch_size": BATCH_SIZE,
        "preprocessing": {
            "missing_modality": "infer MR only when the Modality tag is absent, from the verified official RSNA Axial T2 manifest; reject any present non-MR value",
            "sop_identity": "require unique SOPInstanceUID within each series; verified deidentified RSNA may reuse SOP UIDs across series; bag slice identity is series_id plus InstanceNumber; audit cross-series reuse counts",
            "intensity_percentiles_per_series": [1, 99],
            "monochrome1_inversion": "one common min+max after rescale for the whole series",
            "canonical_orientation": "LPS X increases to image right, LPS Y increases down; transpose/flips; native supported oblique plane retained",
            "maximum_axial_obliquity_degrees": MAX_AXIAL_OBLIQUITY,
            "maximum_projected_inplane_rotation_degrees": MAX_INPLANE_ROTATION,
            "position_reference": "DICOM physical image center, not upper-left IPP",
            "cross_series_linkage": "nonmissing shared FrameOfReferenceUID defines one group; a missing frame UID defines a separate group for its series; no physical z comparison across groups",
            "sort": "frame groups ordered by minimum numeric series_id; within each group descending physical center LPS z, then numeric series_id, InstanceNumber",
            "position_columns": ["frame_group_cranial_to_caudal_z", "series_cranial_to_caudal_z"],
            "position_range": [0, 1],
            "zero_z_span_position": 0.5,
            "context": "previous/current/next in same series; otherwise replicate current",
            "maximum_context_normal_angle_degrees": CONTEXT_ANGLE_DEGREES,
            "maximum_context_center_distance_mm": CONTEXT_DISTANCE_MM,
            "resize": "full FOV bilinear 224x224, antialias=True; no center crop",
            "imagenet_mean": list(IMAGENET_MEAN),
            "imagenet_std": list(IMAGENET_STD),
            "encoder_updated": False,
        },
        "implementation_sha256": _implementation_signature(),
        "packages": {
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
            "numpy": str(np.__version__),
            "pydicom": str(pydicom.__version__),
        },
    }
    cache_id = _hash_json(config)
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["study_id"], []).append(row)
    out.mkdir(parents=True, exist_ok=True)
    (out / "records").mkdir(exist_ok=True)
    with _lock(out):
        if (out / "metadata.json").is_file():
            existing = json.loads((out / "metadata.json").read_text())
            if existing.get("cache_id") != cache_id or existing.get("config") != config:
                raise ValueError("Axial extraction settings changed; use a new cache directory")
        torch.set_num_threads(threads)
        encoder = _load_encoder(weights_path, selected_device)
        record_index = {
            study: (json.loads(path.read_text()), _sha(path))
            for study in groups
            if (path := out / "records" / f"{study}.json").is_file()
        }
        _aggregate(out, groups, config, cache_id, record_index=record_index)
        selected = sorted(groups)[:limit] if limit is not None else sorted(groups)
        for number, study in enumerate(selected, 1):
            study_rows, record_path = groups[study], out / "records" / f"{study}.json"
            source, source_ready = None, False
            try:
                source = _study_source(data_root, study_rows, preparation["manifest_sha256"])
                source_ready = True
                if study in record_index:
                    record = record_index[study][0]
                    if record.get("cache_id") != cache_id:
                        raise ValueError("An axial record belongs to another extraction")
                    if record.get("source_ready") is True:
                        if record.get("source") != source:
                            raise ValueError("Source axial DICOM bytes changed; use a new cache")
                        if record.get("status") == "ok" and _sha(
                            out / record["arrays_path"]
                        ) != record.get("arrays_sha256"):
                            raise ValueError("Cached axial features are missing or corrupted")
                        print(
                            f"Axial features {number}/{len(selected)}: reuse {record['status']}",
                            flush=True,
                        )
                        continue
                started = time.perf_counter()
                images, positions, contexts, qc = _study_inputs(data_root, source)
                decoded_at = time.perf_counter()
                embeddings = _embed(images, contexts, encoder, selected_device)
                arrays_path = f"records/{study}.npz"
                _atomic_npz(
                    out / arrays_path,
                    features=embeddings,
                    positions=positions,
                    instance=np.asarray([image.instance for image in images], dtype=np.int64),
                    series_id=np.asarray([image.series_id for image in images], dtype=str),
                )
                record = {
                    "schema_version": 1,
                    "cache_id": cache_id,
                    "study_id": study,
                    "split": study_rows[0]["split"],
                    "status": "ok",
                    "reason": "",
                    "source_ready": True,
                    "source": source,
                    "n_series": len(study_rows),
                    "n_slices": len(images),
                    "arrays_path": arrays_path,
                    "arrays_sha256": _sha(out / arrays_path),
                    "qc": qc,
                    "seconds": {
                        "decode_and_prepare": decoded_at - started,
                        "encoder_and_array_io": time.perf_counter() - decoded_at,
                        "total": time.perf_counter() - started,
                    },
                }
            except AxialDataError as error:
                source_ready = source_ready and not isinstance(error, SourceNotReady)
                record = {
                    "schema_version": 1,
                    "cache_id": cache_id,
                    "study_id": study,
                    "split": study_rows[0]["split"],
                    "status": "failed" if source_ready else "pending",
                    "reason": error.reason,
                    "source_ready": source_ready,
                    "source": source,
                    "n_series": len(study_rows),
                    "n_slices": 0,
                    "arrays_path": "",
                    "arrays_sha256": "",
                    "error_type": type(error).__name__,
                }
            _atomic_json(record_path, record)
            record_index[study] = record, _sha(record_path)
            if number % 25 == 0:
                _aggregate(
                    out, groups, config, cache_id, record_index=record_index, verify_arrays=False
                )
            print(
                f"Axial features {number}/{len(selected)}: {record['status']} {record['reason']}",
                flush=True,
            )
        if config[
            "implementation_sha256"
        ] != _implementation_signature() or weight_signature != _weight_signature(weights_path):
            raise ValueError("Axial encoder or extractor implementation changed during extraction")
        final_rows, final_preparation = _manifest(data_root, manifest_path)
        if final_rows != rows or final_preparation != preparation:
            raise ValueError("Axial manifest changed during extraction")
        return _aggregate(out, groups, config, cache_id, record_index=record_index)
