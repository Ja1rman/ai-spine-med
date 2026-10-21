"""Physical canal crops from predicted axial landmarks only.

This module has no annotation/grade-file reader. It reuses the verified original
axial decoder, robust per-series intensities and 2.5-D context policy. A predicted
canonical full-image point is mapped to LPS and independently projected onto
every context slice; it is never copied as a pixel coordinate across slices.

Normalized coordinates use 0=first pixel center, 1=last pixel center. A 224-pixel
heatmap decoder must therefore divide its expected pixel coordinates by 223.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pydicom
import torch
from torch.nn.functional import grid_sample
from torchvision.transforms.functional import resize

from . import axial_features as af
from .axial_transfer_cache import validate_sources

LEVELS = ("L4-L5", "L5-S1")
_ACTION = re.compile(r"transpose=([01]),flip_rows=([01]),flip_columns=([01])")


def _action(value):
    match = _ACTION.fullmatch(value)
    if match is None:
        raise af.AxialDataError("invalid_canonical_orientation_action")
    return tuple(bool(int(v)) for v in match.groups())


def canonical_to_source_xy(xy, source_shape, action):
    """Invert transpose then flips, preserving fractional pixel centers."""
    point = np.asarray(xy, dtype=np.float64).copy()
    if point.shape != (2,) or not np.isfinite(point).all():
        raise af.AxialDataError("invalid_predicted_center")
    transpose, flip_rows, flip_columns = _action(action)
    shape = np.asarray(source_shape)[::-1] if transpose else np.asarray(source_shape)
    if flip_columns:
        point[0] = shape[1] - 1 - point[0]
    if flip_rows:
        point[1] = shape[0] - 1 - point[1]
    return point[::-1] if transpose else point


def source_to_canonical_xy(xy, source_shape, action):
    transpose, flip_rows, flip_columns = _action(action)
    point = (
        np.asarray(xy, dtype=np.float64)[::-1].copy()
        if transpose
        else np.asarray(xy, dtype=np.float64).copy()
    )
    shape = np.asarray(source_shape)[::-1] if transpose else np.asarray(source_shape)
    if flip_columns:
        point[0] = shape[1] - 1 - point[0]
    if flip_rows:
        point[1] = shape[0] - 1 - point[1]
    return point


def _source_xy_to_lps(point, header):
    orientation = header["orientation"]
    return (
        header["origin"]
        + point[0] * header["spacing"][1] * orientation[:3]
        + point[1] * header["spacing"][0] * orientation[3:]
    )


def _lps_to_source_xy(point, header):
    delta = np.asarray(point) - header["origin"]
    orientation = header["orientation"]
    return np.asarray(
        [
            delta @ orientation[:3] / header["spacing"][1],
            delta @ orientation[3:] / header["spacing"][0],
        ]
    )


def _headers(root, source, images):
    """Recover original geometry discarded by _Slice, after byte verification."""
    by_key = {(image.series_id, image.instance): image for image in images}
    result = {}
    for series in source["series"]:
        sid = str(series["series_id"])
        for file in series["files"]:
            path = root / series["series_path"] / file["name"]
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            if af._sha(path) != file["sha256"]:
                raise af.SourceNotReady("source_changed_during_crop_geometry")
            key = sid, int(ds.InstanceNumber)
            if key in result or key not in by_key:
                raise af.AxialDataError("crop_geometry_instance_mismatch")
            header = {
                "shape": np.asarray([int(ds.Rows), int(ds.Columns)]),
                "orientation": af._numbers(ds, "ImageOrientationPatient", 6),
                "origin": af._numbers(ds, "ImagePositionPatient", 3),
                "spacing": af._numbers(ds, "PixelSpacing", 2),
                "source_file": file["name"],
                "source_sha256": file["sha256"],
            }
            image = by_key[key]
            center = _source_xy_to_lps((header["shape"][::-1] - 1) / 2, header)
            if not np.allclose(center, image.center_lps, atol=1e-5, rtol=0):
                raise af.AxialDataError("crop_geometry_center_mismatch")
            transpose = _action(image.orientation_action)[0]
            canonical_shape = header["shape"][::-1] if transpose else header["shape"]
            if tuple(canonical_shape) != image.image.shape:
                raise af.AxialDataError("crop_geometry_shape_mismatch")
            header["canonical_spacing"] = (
                header["spacing"][::-1] if transpose else header["spacing"]
            )
            result[key] = header
    if set(result) != set(by_key):
        raise af.AxialDataError("incomplete_crop_geometry")
    return result


def _physical_crop(image, center_xy, spacing, roi_mm):
    """Fractional, center-preserving physical square, zero-padded outside FOV."""
    height, width = image.shape
    native_height = int(math.ceil(roi_mm / float(spacing[0]))) + 1
    native_width = int(math.ceil(roi_mm / float(spacing[1]))) + 1
    if max(native_height, native_width) > 8192:
        raise af.AxialDataError("excessive_physical_crop_resolution")
    x = torch.linspace(-roi_mm / (2 * spacing[1]), roi_mm / (2 * spacing[1]), native_width) + float(
        center_xy[0]
    )
    y = torch.linspace(
        -roi_mm / (2 * spacing[0]), roi_mm / (2 * spacing[0]), native_height
    ) + float(center_xy[1])
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack((grid_x * (2 / (width - 1)) - 1, grid_y * (2 / (height - 1)) - 1), dim=-1)[
        None
    ].to(torch.float32)
    values = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))[None, None]
    patch = grid_sample(values, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]
    patch = resize(patch, [af.IMAGE_SIZE, af.IMAGE_SIZE], antialias=True)[0]
    padded_fraction = float(
        ((grid_x < 0) | (grid_x > width - 1) | (grid_y < 0) | (grid_y > height - 1)).float().mean()
    )
    return patch, padded_fraction


def _centers(images, anchor, maximum, window_mm):
    sides = {-1: [], 1: []}
    cosine = math.cos(math.radians(af.CONTEXT_ANGLE_DEGREES))
    for image in images:
        if image.series_id != anchor.series_id or image.instance == anchor.instance:
            continue
        distance = float((image.center_lps - anchor.center_lps) @ anchor.normal)
        if 1e-3 < abs(distance) <= window_mm and float(image.normal @ anchor.normal) >= cosine:
            sides[1 if distance > 0 else -1].append((abs(distance), image))
    nearest = [
        min(items, key=lambda pair: (pair[0], pair[1].instance))
        for items in sides.values()
        if items
    ]
    nearest.sort(key=lambda pair: (pair[0], -float(pair[1].center_lps[2]), pair[1].instance))
    return [anchor, *(image for _, image in nearest[: maximum - 1])]


def _empty(reason):
    return {
        "status": "not_localized",
        "reason": reason,
        "images": torch.empty((0, 3, af.IMAGE_SIZE, af.IMAGE_SIZE), dtype=torch.float32),
        "metadata": {},
    }


def prepare_crop_source(source_record):
    """Decode once for internal localization/ROI inference; no annotations read."""
    root = validate_sources(source_record)
    images, _, contexts, qc = af._study_inputs(root, source_record["source"])
    headers = _headers(root, source_record["source"], images)
    return {
        "source_sha256": af._hash_json(source_record["source"]),
        "source_root": str(root),
        "images": images,
        "contexts": contexts,
        "qc": qc,
        "headers": headers,
    }


def crop_level_inputs(
    source_record, prediction, roi_mm=50, max_centers=3, window_mm=10, *, _prepared=None
):
    """Build independent L4–L5/L5–S1 ROI bags without reading ground truth.

    ``prediction['levels'][level]`` requires status='ok', series_id, instance,
    and canonical full-FOV ``center_xy_norm``. Missing/invalid predictions keep
    explicit failures. Returned provenance is private research metadata.
    """
    if (
        not isinstance(roi_mm, (int, float))
        or not math.isfinite(roi_mm)
        or not 0 < roi_mm <= 200
        or type(max_centers) is not int
        or not 1 <= max_centers <= 3
        or not isinstance(window_mm, (int, float))
        or not math.isfinite(window_mm)
        or not 0 < window_mm <= 30
    ):
        raise ValueError("Invalid physical crop parameters")
    prepared = prepare_crop_source(source_record) if _prepared is None else _prepared
    if prepared["source_sha256"] != af._hash_json(source_record["source"]):
        raise ValueError("Prepared crop images belong to a different source inventory")
    images, contexts, qc, headers = (
        prepared[key] for key in ("images", "contexts", "qc", "headers")
    )
    by_key = {(image.series_id, image.instance): image for image in images}
    mean = torch.tensor(af.IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(af.IMAGENET_STD)[:, None, None]
    output = {}
    for level in LEVELS:
        landmark = prediction.get("levels", {}).get(level)
        if not isinstance(landmark, dict) or landmark.get("status") != "ok":
            output[level] = _empty("missing_predicted_landmark")
            continue
        try:
            series_id = str(landmark["series_id"])
            instance = int(landmark["instance"])
            normalized = np.asarray(landmark["center_xy_norm"], dtype=np.float64)
        except (KeyError, TypeError, ValueError, OverflowError):
            output[level] = _empty("invalid_predicted_landmark")
            continue
        if (
            normalized.shape != (2,)
            or not np.isfinite(normalized).all()
            or np.any((normalized < 0) | (normalized > 1))
        ):
            output[level] = _empty("invalid_predicted_center")
            continue
        key = series_id, instance
        if key not in by_key:
            output[level] = _empty("predicted_slice_not_in_source")
            continue
        anchor, header = by_key[key], headers[key]
        canonical_xy = normalized * (np.asarray(anchor.image.shape[::-1]) - 1)
        source_xy = canonical_to_source_xy(canonical_xy, header["shape"], anchor.orientation_action)
        point_lps = _source_xy_to_lps(source_xy, header)
        selected = _centers(images, anchor, max_centers, window_mm)
        tensors, crop_metadata = [], []
        for center in selected:
            channels, channel_metadata = [], []
            for neighbor in contexts[(center.series_id, center.instance)]:
                if neighbor.series_id != series_id:
                    raise af.AxialDataError("cross_series_crop_context")
                neighbor_header = headers[(neighbor.series_id, neighbor.instance)]
                xy = _lps_to_source_xy(point_lps, neighbor_header)
                canonical = source_to_canonical_xy(
                    xy, neighbor_header["shape"], neighbor.orientation_action
                )
                patch, padding = _physical_crop(
                    neighbor.image, canonical, neighbor_header["canonical_spacing"], float(roi_mm)
                )
                channels.append(patch)
                channel_metadata.append(
                    {
                        "series_id": neighbor.series_id,
                        "instance": neighbor.instance,
                        "source_file": neighbor_header["source_file"],
                        "source_sha256": neighbor_header["source_sha256"],
                        "source_xy": xy.tolist(),
                        "canonical_xy": canonical.tolist(),
                        "canonical_spacing_mm": neighbor_header["canonical_spacing"].tolist(),
                        "outside_fov_fraction": padding,
                    }
                )
            tensors.append((torch.stack(channels) - mean) / std)
            crop_metadata.append(
                {
                    "series_id": center.series_id,
                    "instance": center.instance,
                    "channels": channel_metadata,
                }
            )
        values = torch.stack(tensors).to(torch.float32)
        if not torch.isfinite(values).all():
            raise af.AxialDataError("nonfinite_physical_crop")
        output[level] = {
            "status": "ok",
            "reason": "",
            "images": values,
            "metadata": {
                "anchor_series_id": series_id,
                "anchor_instance": instance,
                "anchor_canonical_xy": canonical_xy.tolist(),
                "anchor_source_xy": source_xy.tolist(),
                "anchor_lps": point_lps.tolist(),
                "centers": crop_metadata,
                "coordinate_source": "predicted_axial_localizer_only",
            },
        }
    return {
        "levels": output,
        "parameters": {
            "roi_mm": float(roi_mm),
            "max_centers": max_centers,
            "window_mm": float(window_mm),
            "image_size": af.IMAGE_SIZE,
            "normalized_coordinate_convention": "pixel_center_divided_by_dimension_minus_one",
            "resize_antialias": True,
            "padding": "zero_before_imagenet_normalization",
        },
        "source_qc": qc,
    }
