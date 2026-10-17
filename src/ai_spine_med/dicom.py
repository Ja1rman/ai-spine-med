"""Strict, geometry-aware loading of a single-frame sagittal MRI series.

Images are canonicalized to superior at the top, anterior at the left. Slices
run from patient right to left (increasing LPS x). Oblique series within 20
degrees are retained in their acquisition plane; this is not a 3-D resampler.
No patient name, patient ID or arbitrary DICOM text is copied into metadata.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydicom


class DicomGeometryError(ValueError):
    """The input is unsuitable for the supported sagittal-series pipeline."""


@dataclass(frozen=True)
class DicomSeries:
    volume: np.ndarray
    pixel_spacing: tuple[float, float]
    slice_spacing_mm: float
    metadata: dict[str, Any]


def _numbers(ds: pydicom.dataset.Dataset, tag: str, length: int) -> np.ndarray:
    try:
        values = np.asarray(getattr(ds, tag), dtype=np.float64)
    except (AttributeError, TypeError, ValueError) as exc:
        raise DicomGeometryError(f"Missing or invalid {tag}") from exc
    if values.shape != (length,) or not np.isfinite(values).all():
        raise DicomGeometryError(f"Invalid {tag}: expected {length} finite values")
    return values


def load_sagittal_series(
    directory: str | Path,
    *,
    max_obliquity_degrees: float = 20.0,
) -> DicomSeries:
    """Load one DICOM series without guessing missing acquisition geometry.

    ``pixel_spacing`` remains a (row, column) pair, including anisotropy.
    SpineNet's implementation expects an isotropic scalar: a caller must first
    resample anisotropic images rather than silently average their spacing.
    The directory must contain only the series' DICOM files (any extension).
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise DicomGeometryError("DICOM series directory does not exist")
    paths = sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.name != "_series_complete.json"
    )
    if len(paths) < 2:
        raise DicomGeometryError("A spatial series requires at least two DICOM slices")
    if not 0 < max_obliquity_degrees < 45:
        raise ValueError("max_obliquity_degrees must be between 0 and 45")

    datasets = []
    for index, path in enumerate(paths):
        try:
            datasets.append(pydicom.dcmread(path))
        except Exception as exc:
            raise DicomGeometryError(f"Cannot read DICOM slice {index}") from exc

    first = datasets[0]
    series_uid = str(getattr(first, "SeriesInstanceUID", ""))
    if not series_uid:
        raise DicomGeometryError("Missing SeriesInstanceUID")
    study_uid = str(getattr(first, "StudyInstanceUID", ""))
    orientation = _numbers(first, "ImageOrientationPatient", 6)
    column_axis, row_axis = orientation[:3], orientation[3:]
    if (
        not np.isclose(np.linalg.norm(column_axis), 1, atol=1e-3)
        or not np.isclose(np.linalg.norm(row_axis), 1, atol=1e-3)
        or abs(float(column_axis @ row_axis)) > 1e-3
    ):
        raise DicomGeometryError("ImageOrientationPatient axes are not orthonormal")
    normal = np.cross(column_axis, row_axis)
    normal /= np.linalg.norm(normal)
    cosine = np.cos(np.deg2rad(max_obliquity_degrees))
    if abs(normal[0]) < cosine:
        raise DicomGeometryError("Series is not sagittal within the supported obliquity")
    if normal[0] < 0:
        normal = -normal

    # DICOM's first IOP triplet describes the direction of increasing columns.
    transpose = abs(row_axis[1]) >= cosine and abs(column_axis[2]) >= cosine
    if transpose:
        column_axis, row_axis = row_axis.copy(), column_axis.copy()
    elif not (abs(column_axis[1]) >= cosine and abs(row_axis[2]) >= cosine):
        raise DicomGeometryError("Unsupported in-plane rotation of the sagittal images")
    flip_columns = column_axis[1] < 0
    flip_rows = row_axis[2] > 0

    original_spacing = _numbers(first, "PixelSpacing", 2)
    if np.any(original_spacing <= 0):
        raise DicomGeometryError("PixelSpacing must be positive")
    spacing = original_spacing[::-1] if transpose else original_spacing
    expected_shape = (int(getattr(first, "Rows", 0)), int(getattr(first, "Columns", 0)))
    if min(expected_shape) < 2:
        raise DicomGeometryError("Invalid Rows or Columns")

    expected_photometric = str(getattr(first, "PhotometricInterpretation", ""))
    images, positions, sop_uids = [], [], []
    for index, ds in enumerate(datasets):
        if str(getattr(ds, "SeriesInstanceUID", "")) != series_uid:
            raise DicomGeometryError("Mixed SeriesInstanceUID values")
        if str(getattr(ds, "StudyInstanceUID", "")) != study_uid:
            raise DicomGeometryError("Mixed StudyInstanceUID values")
        if int(getattr(ds, "NumberOfFrames", 1)) != 1:
            raise DicomGeometryError("Multiframe DICOM is not supported")
        if str(getattr(ds, "Modality", "MR")) != "MR":
            raise DicomGeometryError("Expected MRI (Modality MR)")
        if not np.allclose(
            _numbers(ds, "ImageOrientationPatient", 6), orientation, rtol=0, atol=1e-4
        ):
            raise DicomGeometryError("Image orientation changes within the series")
        if not np.allclose(_numbers(ds, "PixelSpacing", 2), original_spacing, rtol=0, atol=1e-4):
            raise DicomGeometryError("PixelSpacing changes within the series")
        position = _numbers(ds, "ImagePositionPatient", 3)
        if int(getattr(ds, "SamplesPerPixel", 1)) != 1:
            raise DicomGeometryError("Expected a single-channel MRI image")
        photometric = str(getattr(ds, "PhotometricInterpretation", ""))
        if photometric not in {"MONOCHROME1", "MONOCHROME2"}:
            raise DicomGeometryError("Unsupported PhotometricInterpretation")
        if photometric != expected_photometric:
            raise DicomGeometryError("PhotometricInterpretation changes within the series")
        try:
            slope = float(getattr(ds, "RescaleSlope", 1))
            intercept = float(getattr(ds, "RescaleIntercept", 0))
            if not np.isfinite([slope, intercept]).all() or slope == 0:
                raise ValueError("Invalid rescale parameters")
            image = ds.pixel_array.astype(np.float32) * slope + intercept
        except Exception as exc:
            raise DicomGeometryError(
                f"Cannot decode/rescale pixels in slice {index}; check DICOM codecs"
            ) from exc
        if image.shape != expected_shape or not np.isfinite(image).all():
            raise DicomGeometryError(
                "Image shape changes or nonfinite pixels occur within the series"
            )
        if transpose:
            image = image.T
        if flip_columns:
            image = image[:, ::-1]
        if flip_rows:
            image = image[::-1, :]
        images.append(image)
        positions.append(position)
        sop_uid = str(getattr(ds, "SOPInstanceUID", ""))
        if sop_uid:
            sop_uids.append(sop_uid)
    if len(sop_uids) != len(set(sop_uids)):
        raise DicomGeometryError("Duplicate SOPInstanceUID values")

    positions = np.asarray(positions)
    coordinates = positions @ normal
    order = np.argsort(coordinates)
    sorted_coordinates = coordinates[order]
    distances = np.diff(sorted_coordinates)
    if np.any(distances < 1e-3):
        raise DicomGeometryError("Duplicate or coincident slice positions")
    slice_spacing = float(np.median(distances))
    if not np.allclose(distances, slice_spacing, rtol=0.05, atol=0.1):
        raise DicomGeometryError("Nonuniform slice spacing; missing or irregular slices")
    displacement = positions[order] - positions[order][0]
    in_plane_drift = displacement - np.outer(displacement @ normal, normal)
    if np.max(np.linalg.norm(in_plane_drift, axis=1)) > 0.5:
        raise DicomGeometryError(
            "In-plane slice positions drift; explicit 3-D resampling is required"
        )

    volume = np.ascontiguousarray(np.stack([images[i] for i in order], axis=-1), dtype=np.float32)
    if expected_photometric == "MONOCHROME1":
        # One inversion range for the whole series preserves between-slice
        # intensity relationships; per-slice minima/maxima would alter them.
        volume = volume.max() + volume.min() - volume
    return DicomSeries(
        volume=volume,
        pixel_spacing=(float(spacing[0]), float(spacing[1])),
        slice_spacing_mm=slice_spacing,
        metadata={
            "study_instance_uid": study_uid,
            "series_instance_uid": series_uid,
            "shape_hws": list(volume.shape),
            "pixel_spacing_mm": [float(spacing[0]), float(spacing[1])],
            "slice_spacing_mm": slice_spacing,
            "canonical_orientation": "superior_up_anterior_left_right_to_left_slices",
            "source_image_orientation_patient": orientation.tolist(),
            "in_plane_transposed": bool(transpose),
            "rows_flipped": bool(flip_rows),
            "columns_flipped": bool(flip_columns),
            "rescale_applied": True,
        },
    )
