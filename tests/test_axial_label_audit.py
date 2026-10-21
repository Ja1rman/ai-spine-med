"""Synthetic label semantics and DICOM geometry checks; no patient data needed."""

import hashlib
import math
from types import SimpleNamespace

import pytest

from ai_spine_med.axial_label_audit import (
    aggregate_targets,
    ap_summary,
    bbox_center_lps,
    empirical_ap_grade,
    parse_ccs_xml,
    verified_prepared_file,
)


def annotation(name="CCS_stenosis", level="L5-S1", ap="11.2mm"):
    return (
        "<annotation><filename>slice.png</filename>"
        "<size><width>64</width><height>64</height></size>"
        f"<object><name>{name}</name><level>{level}</level>"
        f"<ap_diameter>{ap}</ap_diameter>"
        "<bndbox><xmin>20</xmin><ymin>10</ymin>"
        "<xmax>30</xmax><ymax>50</ymax></bndbox>"
        "</object></annotation>"
    ).encode()


@pytest.mark.parametrize(
    ("name", "ap", "grade"),
    [
        ("CCS_normal", "13.1mm", 0),
        ("CCS_stenosis", "11.2mm", 1),
        ("CCS_severe_stenosis", "8.5mm", 2),
    ],
)
def test_ccs_class_order_and_explicit_level(name, ap, grade):
    rows = parse_ccs_xml(annotation(name, "L5-S1", ap))
    assert len(rows) == 1
    assert rows[0]["source_level"] == "L5-S1"
    assert rows[0]["grade"] == grade
    assert rows[0]["ap_mm"] == float(ap.removesuffix("mm"))


def test_missing_ccs_is_unknown_not_normal():
    assert parse_ccs_xml(annotation("LFS_normal")) == []
    assert aggregate_targets([]) == [None, None]
    assert aggregate_targets([{"level": "L4-L5", "grade": 0}]) == [0, None]
    assert aggregate_targets([{"level": "L5-S1", "grade": 2}, {"level": "L5-S1", "grade": 0}]) == [
        None,
        2,
    ]


@pytest.mark.parametrize(
    ("ap", "expected"), [(0, 2), (9.97, 2), (10.01, 1), (11.96, 1), (12.01, 0)]
)
def test_observed_ap_partition(ap, expected):
    assert empirical_ap_grade(ap) == expected


@pytest.mark.parametrize("ap", [-1, math.nan, math.inf])
def test_invalid_ap_not_silently_classified(ap):
    with pytest.raises(ValueError, match="AP"):
        empirical_ap_grade(ap)


def test_unknown_ccs_grade_and_bad_level_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        parse_ccs_xml(annotation("CCS_unknown"))
    with pytest.raises(ValueError, match="Unknown"):
        parse_ccs_xml(annotation(level="L5-L6"))


def test_source_ap_measurement_is_not_derived_from_bbox():
    row = parse_ccs_xml(annotation(ap="11.2mm"))[0]
    # ROI height would be 40px * 0.5mm = 20mm. The source AP stays 11.2mm.
    assert row["bbox_xyxy"][3] - row["bbox_xyxy"][1] == 40
    assert row["ap_mm"] == 11.2
    assert row["grade"] == 1
    summary = ap_summary([row])
    assert summary["CCS_stenosis"]["median"] == 11.2
    assert summary["CCS_normal"]["n"] == 0


def test_zero_ap_preserved_as_anomaly_not_changed_to_missing_or_normal():
    row = parse_ccs_xml(annotation("CCS_severe_stenosis", ap="0mm"))[0]
    assert row["ap_mm"] == 0
    assert row["grade"] == 2
    assert ap_summary([row])["CCS_severe_stenosis"]["zero_values"] == 1


def test_physical_center_uses_roi_spacing_and_obliquity():
    ds = SimpleNamespace(
        ImagePositionPatient=[100, 200, 300],
        ImageOrientationPatient=[1, 0, 0, 0, 0.8, 0.6],
        PixelSpacing=[2, 3],
    )
    assert bbox_center_lps([10, 20, 30, 40], ds) == pytest.approx([160, 248, 336])


def test_unsafe_xml_entities_are_rejected():
    with pytest.raises(ValueError, match="Unsafe"):
        parse_ccs_xml(b'<!DOCTYPE annotation [<!ENTITY x SYSTEM "file:///secret">]><annotation/>')


def test_modified_extraction_is_rejected_before_reading_headers(tmp_path):
    path = tmp_path / "source.dcm"
    original = b"original-header"
    path.write_bytes(original)
    receipts = {path.name: {"bytes": len(original), "sha256": hashlib.sha256(original).hexdigest()}}
    assert verified_prepared_file(tmp_path, path.name, receipts) == path
    path.write_bytes(b"modified-header")
    with pytest.raises(ValueError, match="bytes differ"):
        verified_prepared_file(tmp_path, path.name, receipts)
