import numpy as np
import pytest

from ai_spine_med.axial_diagnosis_gallery import (
    _anchor_position,
    _safe,
    build_gallery,
    select_examples,
)


def test_anchor_interpolation_stays_in_the_same_frame_and_series():
    slices = [
        {"series_id": "s", "frame_sha256": "f", "raw_index": 4, "selected_index": 0},
        {"series_id": "s", "frame_sha256": "f", "raw_index": 8, "selected_index": 2},
        {"series_id": "other", "frame_sha256": "f", "raw_index": 6, "selected_index": 8},
    ]
    anchor = {"series_id": "s", "frame_sha256": "f", "raw_index": 6}
    assert _anchor_position(anchor, slices) == 1
    assert _anchor_position({**anchor, "frame_sha256": "different"}, slices) is None
    assert _anchor_position({**anchor, "raw_index": 10}, slices) is None


def test_selection_has_declared_extremes_all_rare_cases_and_no_duplicates():
    entries = [f"{phase}_seed_{seed}" for phase in ("before", "after") for seed in range(3)]
    records, arrays = [], {}
    for dataset, name, labels, top in [
        ("aisslab", "a0", [0, 2], 0),
        ("aisslab", "a1", [0, 2], 1),
        ("aisslab", "a2", [0, 2], 2),
        ("aisslab", "a3", [0, 2], 3),
        ("aisslab", "a4", [0, 2], 4),
        ("rsna", "r0", [0, 2], 0),
        ("rsna", "r1", [0, 2], 0),
        ("rsna", "r2", [0, 2], 0),
        ("rsna", "failure", [2, 0], 0),
    ]:
        slices = [
            {"series_id": "s", "frame_sha256": "f", "raw_index": i, "selected_index": i}
            for i in range(5)
        ]
        records.append(
            {
                "dataset": dataset,
                "study_id": name,
                "labels": labels,
                "selected_slices": slices,
                "anchors": [
                    {"level": "L5-S1", "series_id": "s", "frame_sha256": "f", "raw_index": 0}
                ],
            }
        )
        attention = np.zeros((6, 2, 5))
        attention[:, :, top] = 1
        probabilities = np.zeros((6, 2, 3))
        probabilities[:, :, 0] = 1
        if name == "failure":
            probabilities[:3, 0] = [0, 0, 1]
        arrays[(dataset, name)] = {"attention": attention, "probabilities": probabilities}
    chosen, reasons = select_examples(records, arrays, entries, [{"study_id": "a0"}])
    assert ("aisslab", "a2") not in chosen
    assert set(chosen) == {
        ("aisslab", "a0"),
        ("aisslab", "a1"),
        ("aisslab", "a3"),
        ("aisslab", "a4"),
        ("rsna", "r0"),
        ("rsna", "r1"),
        ("rsna", "r2"),
        ("rsna", "failure"),
    }
    assert len(chosen) == len(set(chosen))
    assert len(reasons[("aisslab", "a0")]) == 2
    assert (
        chosen == select_examples(list(reversed(records)), arrays, entries, [{"study_id": "a0"}])[0]
    )


def test_gallery_refuses_to_write_pixels_into_public_documents(tmp_path):
    attention = tmp_path / "docs/example/attention"
    attention.mkdir(parents=True)
    (attention / "index.json").write_text("{}")
    with pytest.raises(ValueError, match="must stay private"):
        build_gallery(tmp_path, "docs/example")
    assert not (tmp_path / "docs/example/gallery").exists()


def test_raw_source_paths_cannot_escape_the_source_root(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (tmp_path / "elsewhere.dcm").write_bytes(b"not a source file")
    with pytest.raises(ValueError, match="escapes"):
        _safe(source, "../elsewhere.dcm")
