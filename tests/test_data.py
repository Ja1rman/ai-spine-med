"""Data-integrity tests use generated CSVs, never distributed RSNA cases."""

import csv
import hashlib

import pytest

from ai_spine_med.data import CLASS_NAMES, LEVELS, prepare_rsna


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_metadata(root):
    grades, series, coordinates = [], [], []
    for index in range(30):
        study = str(100 + index)
        name = ("Normal/Mild", "Moderate", "Severe")[index // 10]
        grades.append(
            {
                "study_id": study,
                list(LEVELS.values())[0]: name,
                list(LEVELS.values())[1]: "" if index == 0 else name,
            }
        )
        for series_id in ("10", "20"):
            series.append(
                {
                    "study_id": study,
                    "series_id": series_id,
                    "series_description": "Sagittal T2/STIR",
                }
            )
            for level in ("L4/L5", "L5/S1"):
                coordinates.append(
                    {
                        "study_id": study,
                        "series_id": series_id,
                        "instance_number": "4",
                        "condition": "Spinal Canal Stenosis",
                        "level": level,
                        "x": "20.5",
                        "y": "30",
                    }
                )
    grades.extend(
        [
            {"study_id": "998", **{column: "" for column in LEVELS.values()}},
            {"study_id": "999", **{column: "Severe" for column in LEVELS.values()}},
        ]
    )
    series.extend(
        [
            {"study_id": "998", "series_id": "10", "series_description": "Sagittal T2/STIR"},
            {"study_id": "999", "series_id": "10", "series_description": "Sagittal T1"},
        ]
    )
    write_csv(root / "train.csv", ["study_id", *LEVELS.values()], grades)
    write_csv(
        root / "train_series_descriptions.csv",
        ["study_id", "series_id", "series_description"],
        series,
    )
    write_csv(
        root / "train_label_coordinates.csv",
        ["study_id", "series_id", "instance_number", "condition", "level", "x", "y"],
        coordinates,
    )


def read_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def test_grouping_missing_values_and_integrity(tmp_path):
    root, out = tmp_path / "raw", tmp_path / "prepared"
    make_metadata(root)
    summary = prepare_rsna(root, out)
    studies = read_rows(out / "studies.csv")
    groups = {row["study_id"]: row["split"] for row in studies}
    assert len(groups) == 30
    assert summary["n_excluded_no_sagittal_t2_stir"] == 1
    assert summary["n_excluded_all_target_labels_missing_with_series"] == 1
    assert summary["n_series"] == 60
    assert summary["n_local_dicoms"] == 0
    assert summary["grouping"]["key"] == "study_id"
    for name in ("series.csv", "labels.csv", "coordinates.csv"):
        assert all(row["split"] == groups[row["study_id"]] for row in read_rows(out / name))
    labels = read_rows(out / "labels.csv")
    assert len(labels) == 59
    assert not any(row["study_id"] == "100" and row["level"] == "L5-S1" for row in labels)
    assert {row["label"] for row in labels} == {"0", "1", "2"}
    for split in ("train", "validation", "test"):
        assert all(summary["splits"][split]["class_counts"][name] > 0 for name in CLASS_NAMES)
    for name, digest in summary["output_sha256"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest


def test_repeat_is_byte_reproducible_and_independent_of_row_order(tmp_path):
    root = tmp_path / "raw"
    make_metadata(root)
    first, second = tmp_path / "a", tmp_path / "b"
    prepare_rsna(root, first)
    initial = {path.name: path.read_bytes() for path in first.iterdir()}
    prepare_rsna(root, first)
    assert initial == {path.name: path.read_bytes() for path in first.iterdir()}
    for path in root.glob("*.csv"):
        rows = read_rows(path)
        write_csv(path, list(rows[0]), list(reversed(rows)))
    prepare_rsna(root, second)
    for name in ("studies.csv", "series.csv", "labels.csv", "coordinates.csv"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_split_is_not_silently_replaced_and_availability_can_refresh(tmp_path):
    root, out = tmp_path / "raw", tmp_path / "out"
    make_metadata(root)
    prepare_rsna(root, out)
    original = (out / "studies.csv").read_bytes()
    with pytest.raises(ValueError, match="existing study split differs"):
        prepare_rsna(root, out, seed=123)
    assert (out / "studies.csv").read_bytes() == original
    image = root / "train_images" / "100" / "10" / "4.dcm"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"synthetic-test-file")
    refreshed = prepare_rsna(root, out)
    assert refreshed["n_local_dicoms"] == 1
    assert (out / "studies.csv").read_bytes() == original
    series = next(
        row
        for row in read_rows(out / "series.csv")
        if row["study_id"] == "100" and row["series_id"] == "10"
    )
    assert series["geometry_qc_status"] == "pending"
    assert series["sequence_subtype"] == "unknown"
    assert series["download_status"] == "present_unverified"


def test_small_strata_do_not_leak_or_drop_studies(tmp_path):
    root = tmp_path / "raw"
    make_metadata(root)
    rows = read_rows(root / "train.csv")
    rows = [row for row in rows if row["study_id"] in ("100", "110", "120", "121")]
    write_csv(root / "train.csv", list(rows[0]), rows)
    result = prepare_rsna(root, tmp_path / "out")
    assert sum(item["studies"] for item in result["splits"].values()) == 4
    assert result["warnings"]
    studies = read_rows(tmp_path / "out" / "studies.csv")
    assert len({row["study_id"] for row in studies}) == 4


@pytest.mark.parametrize("bad_value", ["Critical", "0", "normal"])
def test_unknown_labels_fail_instead_of_becoming_normal(tmp_path, bad_value):
    root = tmp_path / "raw"
    make_metadata(root)
    rows = read_rows(root / "train.csv")
    rows[0]["spinal_canal_stenosis_l4_l5"] = bad_value
    write_csv(root / "train.csv", list(rows[0]), rows)
    with pytest.raises(ValueError, match="Unexpected severity"):
        prepare_rsna(root, tmp_path / "out")


def test_duplicate_study_and_unsafe_series_id_are_rejected(tmp_path):
    root = tmp_path / "raw"
    make_metadata(root)
    rows = read_rows(root / "train.csv")
    write_csv(root / "train.csv", list(rows[0]), rows + [rows[0]])
    with pytest.raises(ValueError, match="duplicate study"):
        prepare_rsna(root, tmp_path / "out")
    make_metadata(root)
    series = read_rows(root / "train_series_descriptions.csv")
    series[0]["series_id"] = "../outside"
    write_csv(root / "train_series_descriptions.csv", list(series[0]), series)
    with pytest.raises(ValueError, match="numeric identifier"):
        prepare_rsna(root, tmp_path / "out")


def test_bad_coordinates_are_reported_without_discarding_classification_labels(tmp_path):
    root = tmp_path / "raw"
    make_metadata(root)
    path = root / "train_label_coordinates.csv"
    rows = read_rows(path)
    rows[0]["x"] = "nan"
    rows.append(rows[1].copy())
    write_csv(path, list(rows[0]), rows)
    result = prepare_rsna(root, tmp_path / "out")
    assert result["n_invalid_target_coordinates"] == 1
    assert result["n_duplicate_target_coordinates"] == 1
    assert sum(value["labels"] for value in result["splits"].values()) == 59


@pytest.mark.parametrize("ratios", [(0.8, 0.2, 0.2), (-0.1, 0.5, 0.6), (float("nan"), 0.5, 0.5)])
def test_invalid_ratios_fail_early(tmp_path, ratios):
    with pytest.raises(ValueError, match="ratios"):
        prepare_rsna(
            tmp_path,
            tmp_path / "out",
            train_ratio=ratios[0],
            val_ratio=ratios[1],
            test_ratio=ratios[2],
        )
