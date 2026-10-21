"""Geometry/anchor tests use synthetic headers; no patient images or weights."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from ai_spine_med import axial_localization_audit as audit


def slice_descriptor(z, *, series="1", instance=1, frame="shared", group=0, position=0.5):
    return {
        "series_id": series,
        "instance": instance,
        "path": f"series/{series}/{instance}.dcm",
        "center_lps": [0.0, 0.0, z],
        "normal": [0.0, 0.0, 1.0],
        "frame_sha256": frame,
        "frame_group": group,
        "positions": [position, position],
        "raw_index": instance - 1,
    }


def anchor(s, *, level="L5-S1", kind="aisslab_xml_ccs_exact_slice"):
    return audit._make_anchor(s, kind=kind, level=level)


def test_dicom_xy_conversion_respects_oblique_axes_and_asymmetric_spacing():
    header = {
        "origin_lps": [10.0, 20.0, 30.0],
        "orientation": [0.0, 1.0, 0.0, 0.6, 0.0, 0.8],
        "spacing": [2.0, 3.0],
    }
    np.testing.assert_allclose(audit.dicom_point(header, 4, 5), [16.0, 32.0, 38.0])


def test_plane_distance_uses_normal_projection_not_center_euclidean_distance():
    item = slice_descriptor(10)
    assert audit.plane_distance(item, [1000.0, -999.0, 14.0]) == 4.0


def test_xml_anchor_uses_ccs_bbox_center_and_original_dicom_coordinates(tmp_path):
    path = tmp_path / "anchor.xml"
    path.write_text(
        "<annotation><size><height>100</height><width>200</width></size>"
        "<object><name>CCS_stenosis</name><level>L5-S1</level>"
        "<bndbox><xmin>30</xmin><xmax>50</xmax><ymin>10</ymin><ymax>20</ymax>"
        "</bndbox></object></annotation>"
    )
    ref = {"xml_path": path.name, "object_index": 0, "source_grade": "stenosis", "level": "L5-S1"}
    item = {
        "image_shape": [100, 200],
        "origin_lps": [10.0, 20.0, 30.0],
        "orientation": [0.0, 1.0, 0.0, 0.6, 0.0, 0.8],
        "spacing": [2.0, 3.0],
    }
    np.testing.assert_allclose(audit._xml_ccs_point(tmp_path, ref, item), [28.0, 140.0, 54.0])
    item["image_shape"] = [200, 100]
    with pytest.raises(ValueError, match="dimensions"):
        audit._xml_ccs_point(tmp_path, ref, item)


def test_header_only_reader_does_not_request_pixels(monkeypatch):
    header = SimpleNamespace(
        ImageOrientationPatient=[1, 0, 0, 0, 1, 0],
        PixelSpacing=[2, 3],
        ImagePositionPatient=[10, 20, 30],
        Rows=5,
        Columns=7,
        InstanceNumber=2,
        FrameOfReferenceUID="frame",
    )

    def reader(path, **kwargs):
        assert kwargs == {"stop_before_pixels": True}
        return header

    monkeypatch.setattr(audit.pydicom, "dcmread", reader)
    item = audit._header("unused.dcm")
    np.testing.assert_allclose(item["center_lps"], [19, 24, 30])
    assert item["frame_sha256"]


def test_reconstruction_exact_order_frame_split_and_96_cap(monkeypatch, tmp_path):
    source = {
        "source": {
            "series": [
                {"series_id": "9", "series_path": "9", "files": [{"name": "a"}]},
                {
                    "series_id": "2",
                    "series_path": "2",
                    "files": [{"name": str(i)} for i in range(100)],
                },
            ]
        }
    }
    monkeypatch.setattr(audit, "validate_sources", lambda row: tmp_path)

    def header(path):
        if path.parent.name == "9":
            return slice_descriptor(1000, frame="other", series="9")
        instance = int(path.name)
        return slice_descriptor(float(instance), instance=instance, frame="shared", series="2")

    monkeypatch.setattr(audit, "_header", header)
    selected, full = audit.reconstruct_slice_bag(source)
    assert len(selected) == 96 and len(full) == 101
    assert full[0]["instance"] == 99 and full[99]["instance"] == 0
    assert full[-1]["series_id"] == "9"  # separate frame is not ordered by physical z
    np.testing.assert_allclose(full[0]["positions"], [0, 0])
    np.testing.assert_allclose(full[99]["positions"], [1, 1])
    np.testing.assert_allclose(full[100]["positions"], [0.5, 0.5])
    assert [s["raw_index"] for s in selected] == np.linspace(0, 100, 96).astype(int).tolist()


def test_direct_anchor_does_not_match_unrelated_series_even_same_z():
    target = slice_descriptor(0, series="1")
    other = slice_descriptor(0, series="2")
    distances = audit._distances([target, other], [anchor(target)])
    assert distances[0] == 0 and np.isinf(distances[1])


def test_sagittal_projection_requires_shared_nonempty_frame():
    target = slice_descriptor(0, series="1")
    same = slice_descriptor(0, series="2")
    different = slice_descriptor(0, series="3", frame="other")
    kind = "rsna_sagittal_central_same_frame_projection"
    distances = audit._distances([same, different], [anchor(target, kind=kind)])
    assert distances[0] == 0 and np.isinf(distances[1])
    target["frame_sha256"] = None
    assert np.isinf(audit._distances([same], [anchor(target, kind=kind)])[0])


def test_metrics_report_localization_separately_from_prediction_and_window_chance():
    slices = [
        slice_descriptor(z, instance=i + 1, position=i / 3) for i, z in enumerate([30, 20, 10, 0])
    ]
    record = {"selected_slices": slices, "anchors": [anchor(slices[-1])]}
    weights = np.asarray([[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.7]])
    results = audit.localization_metrics(weights, record)
    metrics = results["L5-S1"]["by_anchor_kind"]["aisslab_xml_ccs_exact_slice"]
    assert metrics["top1_distance_mm"] == 0
    assert metrics["top1_near_5mm"]
    assert metrics["mass_near_5mm"] == pytest.approx(0.7)
    assert metrics["uniform_mass_near_5mm"] == 0.25
    assert not results["same_top1"]
    assert results["head_overlap_mass"] == pytest.approx(0.4)
    assert results["L4-L5"]["by_anchor_kind"] == {}


def test_invalid_attention_rejected():
    record = {"selected_slices": [slice_descriptor(0)], "anchors": []}
    with pytest.raises(ValueError, match="attention"):
        audit.localization_metrics(np.asarray([[1.0], [-1.0]]), record)


def test_fixed_run_aggregate_preserves_order_and_uniform_effective_slice_fraction(tmp_path):
    slices = [
        slice_descriptor(float(30 - i * 10), instance=i + 1, position=i / 3) for i in range(4)
    ]
    record = {
        "dataset": "aisslab",
        "study_id": "synthetic",
        "selected_slices": slices,
        "anchors": [anchor(slices[1], level="L4-L5"), anchor(slices[3])],
    }
    anchors_path = tmp_path / "anchors.json"
    anchors_path.write_text(json.dumps({"records": [record]}))
    archive = tmp_path / "weights.npz"
    np.savez(
        archive,
        attention=np.full((6, 2, 4), 0.25, dtype=np.float32),
        positions=np.asarray([s["positions"] for s in slices], dtype=np.float32),
    )
    entries = [f"{phase}_seed_{seed}" for phase in ("before", "after") for seed in (42, 2024, 3407)]
    index = {
        "entries": entries,
        "records": [
            {
                "dataset": "aisslab",
                "study_id": "synthetic",
                "arrays_path": archive.name,
                "arrays_sha256": audit._sha(archive),
            }
        ],
    }
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index))
    result = audit.run_localization_audit(anchors_path, index_path, tmp_path / "out")
    assert len(result["groups"]) == 16
    assert len(result["head_relationships"]) == 8
    assert all(r["effective_slice_fraction_mean"] == pytest.approx(1) for r in result["groups"])
    assert all(r["normalized_entropy_mean"] == pytest.approx(1) for r in result["groups"])
    index["entries"] = list(reversed(entries))
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="order"):
        audit.run_localization_audit(anchors_path, index_path, tmp_path / "out")


def test_cli_executes_header_and_attention_audits_with_explicit_inputs(monkeypatch, tmp_path):
    calls = []

    def build(sources, index, prepared, rsna):
        calls.append((sources.name, index.name, prepared.name, rsna.name))
        return {"records": []}

    def assess(anchors, index, out):
        assert json.loads(anchors.read_text()) == {"records": []}
        calls.append((anchors.name, index.name, out.name))

    monkeypatch.setattr(audit, "build_anchor_records", build)
    monkeypatch.setattr(audit, "run_localization_audit", assess)
    out = tmp_path / "private-audit"
    audit.main(
        [
            "--sources",
            "sources.json",
            "--index",
            "cache.json",
            "--prepared",
            "prepared",
            "--rsna-root",
            "rsna",
            "--attention-index",
            "fixed-attention.json",
            "--out",
            str(out),
        ]
    )
    assert calls == [
        ("sources.json", "cache.json", "prepared", "rsna"),
        ("anchors.json", "fixed-attention.json", "private-audit"),
    ]
    assert json.loads((out / "geometry-summary.json").read_text())["n_studies"] == 0
    assert json.loads((out / "anatomical-order-summary.json").read_text()) == {}


def test_geometry_summary_does_not_pair_anatomy_across_distinct_series():
    upper = slice_descriptor(20, series="1")
    lower = slice_descriptor(0, series="2")
    records = [
        {
            "dataset": "synthetic",
            "study_id": "patient",
            "cache_positions_exact": True,
            "missing_coordinate_files_or_series": 0,
            "selected_slices": [upper, lower],
            "anchors": [anchor(upper, level="L4-L5"), anchor(lower)],
        }
    ]
    summary, order, exceptions = audit.summarize_geometry(records)
    assert summary["n_studies"] == 1 and summary["exact_cache_positions_all"]
    assert order["synthetic"]["same_series_level_pairs"] == 0
    assert order["synthetic"]["median_L4minusL5_z_mm"] is None
    assert exceptions == []
