import numpy as np
import pytest
from pydicom import dcmread
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, RLELossless, generate_uid

from ai_spine_med.dicom import DicomGeometryError, load_sagittal_series

SERIES_UID = generate_uid()
STUDY_UID = generate_uid()


def write_slice(path, *, x=0, pixels=None, orientation=None, spacing=(0.6, 0.9), **tags):
    pixels = np.asarray(pixels if pixels is not None else [[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = STUDY_UID
    ds.SeriesInstanceUID = SERIES_UID
    ds.Modality = "MR"
    ds.PatientID = "PRIVATE_ID_MUST_NOT_BE_EXPORTED"
    ds.PatientName = "PRIVATE_NAME_MUST_NOT_BE_EXPORTED"
    ds.ImageOrientationPatient = orientation or [0, 1, 0, 0, 0, -1]
    ds.ImagePositionPatient = [x, 0, 0]
    ds.PixelSpacing = list(spacing)
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    for key, value in tags.items():
        setattr(ds, key, value)
    ds.PixelData = pixels.tobytes()
    ds.save_as(path, enforce_file_format=True)
    return ds


def make_pair(tmp_path, **kwargs):
    write_slice(tmp_path / "a.dcm", x=3, InstanceNumber=1, **kwargs)
    write_slice(tmp_path / "b.dcm", x=0, InstanceNumber=100, **kwargs)


def test_physical_sort_spacing_and_private_metadata(tmp_path):
    write_slice(tmp_path / "a.dcm", x=3, pixels=[[10, 11, 12], [13, 14, 15]], InstanceNumber=1)
    write_slice(tmp_path / "b.dcm", x=0, InstanceNumber=100)
    result = load_sagittal_series(tmp_path)
    assert result.volume.shape == (2, 3, 2)
    np.testing.assert_array_equal(result.volume[:, :, 0], [[1, 2, 3], [4, 5, 6]])
    assert result.pixel_spacing == (0.6, 0.9)
    assert result.slice_spacing_mm == 3
    assert result.volume.dtype == np.float32
    assert "PRIVATE" not in str(result.metadata)


def test_load_rle_lossless_series_without_external_codecs(tmp_path):
    """Exercise the compressed pixel path used by RSNA, not a mock decoder."""
    pixels = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    for name, x, values in [("a", 3, pixels + 17), ("b", 0, pixels)]:
        path = tmp_path / name
        ds = write_slice(path, x=x, pixels=values, RescaleSlope=2, RescaleIntercept=-5)
        ds.compress(RLELossless, encoding_plugin="pydicom")
        ds.save_as(path, enforce_file_format=True)
        compressed = dcmread(path)
        assert compressed.file_meta.TransferSyntaxUID == RLELossless
        compressed.pixel_array_options(decoding_plugin="pydicom")
        np.testing.assert_array_equal(compressed.pixel_array, values)
    result = load_sagittal_series(tmp_path)
    assert result.volume.shape == (64, 64, 2)
    np.testing.assert_array_equal(result.volume[:, :, 0], pixels.astype(np.float32) * 2 - 5)
    np.testing.assert_array_equal(result.volume[:, :, 1], (pixels.astype(np.float32) + 17) * 2 - 5)


@pytest.mark.parametrize(
    "orientation, expected, spacing",
    [
        ([0, -1, 0, 0, 0, -1], [[3, 2, 1], [6, 5, 4]], (0.6, 0.9)),
        ([0, 1, 0, 0, 0, 1], [[4, 5, 6], [1, 2, 3]], (0.6, 0.9)),
        ([0, 0, -1, 0, 1, 0], [[1, 4], [2, 5], [3, 6]], (0.9, 0.6)),
    ],
)
def test_anatomical_axes_and_transpose(tmp_path, orientation, expected, spacing):
    make_pair(tmp_path, orientation=orientation)
    result = load_sagittal_series(tmp_path)
    np.testing.assert_array_equal(result.volume[:, :, 0], expected)
    assert result.pixel_spacing == spacing


def test_rescale_then_monochrome_inversion(tmp_path):
    make_pair(
        tmp_path, RescaleSlope=2, RescaleIntercept=-5, PhotometricInterpretation="MONOCHROME1"
    )
    result = load_sagittal_series(tmp_path)
    np.testing.assert_array_equal(result.volume[:, :, 0], [[7, 5, 3], [1, -1, -3]])


def test_monochrome_inversion_preserves_between_slice_intensities(tmp_path):
    write_slice(tmp_path / "a", x=0, PhotometricInterpretation="MONOCHROME1")
    write_slice(
        tmp_path / "b",
        x=3,
        pixels=[[11, 12, 13], [14, 15, 16]],
        PhotometricInterpretation="MONOCHROME1",
    )
    result = load_sagittal_series(tmp_path)
    assert result.volume[0, 0, 0] == 16
    assert result.volume[0, 0, 1] == 6


def test_moderate_obliquity_uses_plane_normal_for_distances(tmp_path):
    angle = np.deg2rad(12)
    orientation = [np.sin(angle), np.cos(angle), 0, 0, 0, -1]
    normal = np.array([np.cos(angle), -np.sin(angle), 0])
    write_slice(tmp_path / "a", x=0, orientation=orientation)
    write_slice(
        tmp_path / "b",
        x=0,
        orientation=orientation,
        ImagePositionPatient=(3 * normal).tolist(),
    )
    result = load_sagittal_series(tmp_path)
    assert result.slice_spacing_mm == pytest.approx(3)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"SeriesInstanceUID": generate_uid()}, "Mixed Series"),
        ({"ImageOrientationPatient": [1, 0, 0, 0, 1, 0]}, "orientation changes"),
        ({"PixelSpacing": [0.7, 0.9]}, "PixelSpacing changes"),
        ({"NumberOfFrames": 2}, "Multiframe"),
        ({"ImagePositionPatient": [3, 0, 0]}, "coincident"),
        ({"ImagePositionPatient": [0, 1, 0]}, "drift"),
    ],
)
def test_reject_inconsistent_series(tmp_path, changes, message):
    make_pair(tmp_path)
    ds = dcmread(tmp_path / "b.dcm")
    for key, value in changes.items():
        setattr(ds, key, value)
    ds.save_as(tmp_path / "b.dcm", enforce_file_format=True)
    with pytest.raises(DicomGeometryError, match=message):
        load_sagittal_series(tmp_path)


def test_reject_axial_series(tmp_path):
    make_pair(tmp_path, orientation=[1, 0, 0, 0, 1, 0])
    with pytest.raises(DicomGeometryError, match="not sagittal"):
        load_sagittal_series(tmp_path)


def test_reject_missing_position_and_irregular_spacing(tmp_path):
    make_pair(tmp_path)
    ds = dcmread(tmp_path / "a.dcm")
    del ds.ImagePositionPatient
    ds.save_as(tmp_path / "a.dcm", enforce_file_format=True)
    with pytest.raises(DicomGeometryError, match="ImagePositionPatient"):
        load_sagittal_series(tmp_path)
    write_slice(tmp_path / "a.dcm", x=3)
    write_slice(tmp_path / "c.dcm", x=8)
    with pytest.raises(DicomGeometryError, match="Nonuniform"):
        load_sagittal_series(tmp_path)
