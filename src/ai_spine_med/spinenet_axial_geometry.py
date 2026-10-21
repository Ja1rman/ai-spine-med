"""Recover physical SpineNet disc landmarks without guessing registration.

Cached SpineNet polygons describe vertebral bodies, not segmented canal boxes.
The posterior disc midpoint is therefore an anatomical level landmark only.
Projection into an axial series requires a shared, nonempty DICOM frame. A
different frame is an explicit refusal until image registration is implemented.
"""

from __future__ import annotations

import hashlib

import numpy as np

LEVELS = ("L4-L5", "L5-S1")


def _array(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"Invalid {name}")
    return result


def detector_to_source_xy(point, metadata, source_shape, source_spacing):
    """Invert canonicalization and center-preserving in-plane resampling.

    The point is (column, row) in the detector's isotropic image. Original
    DICOM spacing is (row, column). Transpose occurs before canonical flips.
    Coordinates are never rounded to integer pixels before LPS conversion.
    """
    point = _array(point, (2,), "detector point")
    shape = _array(source_shape, (2,), "source shape")
    spacing = _array(source_spacing, (2,), "source spacing")
    if np.any(shape < 2) or np.any(spacing <= 0):
        raise ValueError("Invalid source shape/spacing")
    transposed = bool(metadata["in_plane_transposed"])
    canonical_shape = shape[::-1] if transposed else shape
    canonical_spacing = spacing[::-1] if transposed else spacing
    target = float(metadata["pixel_spacing_mm"])
    if not np.isfinite(target) or target <= 0:
        raise ValueError("Invalid detector pixel spacing")
    # Exact inverse of baseline.isotropic_inplane's output-center transform.
    xy = (
        point * target / canonical_spacing[::-1]
        if metadata.get("resampled", False)
        else point.copy()
    )
    if metadata["columns_flipped"]:
        xy[0] = canonical_shape[1] - 1 - xy[0]
    if metadata["rows_flipped"]:
        xy[1] = canonical_shape[0] - 1 - xy[1]
    return xy[::-1] if transposed else xy


def source_xy_to_lps(point, header):
    xy = _array(point, (2,), "source point")
    orientation = _array(header["orientation"], (6,), "orientation")
    spacing = _array(header["spacing"], (2,), "spacing")
    origin = _array(header["origin_lps"], (3,), "origin")
    if (
        np.any(spacing <= 0)
        or not np.isclose(np.linalg.norm(orientation[:3]), 1, atol=1e-3)
        or not np.isclose(np.linalg.norm(orientation[3:]), 1, atol=1e-3)
        or abs(orientation[:3] @ orientation[3:]) > 1e-3
    ):
        raise ValueError("Invalid DICOM geometry")
    return origin + xy[0] * spacing[1] * orientation[:3] + xy[1] * spacing[0] * orientation[3:]


def _posterior_corner(polygon, *, inferior):
    polygon = _array(polygon, (4, 2), "vertebral polygon")
    # Canonical sagittal images have anterior at left and superior at top.
    # Selecting geometrically is robust to a changed upstream corner order.
    posterior = polygon[np.argsort(polygon[:, 0])[-2:]]
    return posterior[np.argmax(posterior[:, 1]) if inferior else np.argmin(posterior[:, 1])]


def disc_landmark(vertices, level):
    """Return a posterior disc landmark on an actually shared sagittal slice.

    Missing/duplicate vertebrae and disjoint detections return an explicit
    reason. There is no guessed sacral level and no average-polygon fallback.
    """
    if level not in LEVELS:
        raise ValueError("Unsupported disc level")
    names = level.split("-")
    found = [[v for v in vertices if v["label"] == name] for name in names]
    if any(len(items) != 1 for items in found):
        return {"status": "not_localized", "reason": "missing_or_duplicate_vertebra", "level": level}
    superior, inferior = (items[0] for items in found)
    maps = []
    for vertex in (superior, inferior):
        if len(vertex["slice_nos"]) != len(vertex["polys"]):
            raise ValueError("Polygon/slice count mismatch")
        mapping = {int(s): p for s, p in zip(vertex["slice_nos"], vertex["polys"])}
        if len(mapping) != len(vertex["slice_nos"]) or any(s < 0 for s in mapping):
            raise ValueError("Invalid or duplicate sagittal slice index")
        maps.append(mapping)
    shared = sorted(maps[0].keys() & maps[1].keys())
    if not shared:
        return {"status": "not_localized", "reason": "no_shared_sagittal_slice", "level": level}
    # A real central shared slice avoids inventing an unobserved cross-section.
    slice_index = shared[(len(shared) - 1) // 2]
    first = _posterior_corner(maps[0][slice_index], inferior=True)
    second = _posterior_corner(maps[1][slice_index], inferior=False)
    if second[1] <= first[1]:
        return {"status": "not_localized", "reason": "reversed_or_overlapping_endplates", "level": level}
    return {
        "status": "ok",
        "reason": "",
        "level": level,
        "slice_index": slice_index,
        "detector_xy": ((first + second) / 2).tolist(),
        "landmark_kind": "spinenet_posterior_disc_endplate_midpoint_not_canal_segmentation",
    }


def project_registered_landmark(point_lps, sagittal_frame_uid, axial_headers):
    """Select nearest source plane only inside one registered frame.

    Returns axial pixel coordinates for the landmark rather than claiming a
    detected canal center. Rejection is propagated per slice and per study.
    No API for bypassing the registration gate is provided.
    """
    point = _array(point_lps, (3,), "landmark LPS")
    if not sagittal_frame_uid:
        return {"status": "not_localized", "reason": "missing_sagittal_frame", "candidates": []}
    frame_hash = hashlib.sha256(str(sagittal_frame_uid).encode()).hexdigest()
    candidates = []
    for item in axial_headers:
        if item.get("frame_sha256") != frame_hash:
            continue
        orientation = _array(item["orientation"], (6,), "axial orientation")
        origin = _array(item["origin_lps"], (3,), "axial origin")
        spacing = _array(item["spacing"], (2,), "axial spacing")
        normal = np.cross(orientation[:3], orientation[3:])
        norm = np.linalg.norm(normal)
        if norm < 0.99 or abs(orientation[:3] @ orientation[3:]) > 1e-3 or np.any(spacing <= 0):
            raise ValueError("Invalid axial geometry")
        normal /= norm
        delta = point - origin
        xy = [float(delta @ orientation[:3] / spacing[1]), float(delta @ orientation[3:] / spacing[0])]
        rows, columns = _array(item["image_shape"], (2,), "axial shape")
        if not (0 <= xy[0] < columns and 0 <= xy[1] < rows):
            continue
        candidates.append({
            "series_id": item["series_id"],
            "instance": item["instance"],
            "distance_mm": abs(float(delta @ normal)),
            "source_xy": xy,
        })
    if not candidates:
        any_same = any(s.get("frame_sha256") == frame_hash for s in axial_headers)
        return {"status": "not_localized", "reason": "landmark_outside_axial_fov" if any_same else "unregistered_frame", "candidates": []}
    candidates.sort(key=lambda s: (s["distance_mm"], str(s["series_id"]), s["instance"]))
    return {"status": "ok", "reason": "", "candidates": candidates}
