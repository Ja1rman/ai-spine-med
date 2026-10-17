import csv
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ai_spine_med import features as module
from ai_spine_med.assets import sha256_file


def write_csv(path, fields, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_cohort(tmp_path):
    data, manifests, output = (tmp_path / name for name in ("data", "manifests", "features"))
    data.mkdir()
    manifests.mkdir()
    for name in module.METADATA_FILES:
        (data / name).write_text("unchanged source metadata\n")
    studies = [
        {"study_id": str(i), "split": split}
        for i, split in ((10, "train"), (20, "validation"), (30, "test"))
    ]
    write_csv(manifests / "studies.csv", ["study_id", "split"], studies)
    series = [
        {
            **row,
            "series_id": str(int(row["study_id"]) + 1),
            "series_path": f"train_images/{row['study_id']}/{int(row['study_id']) + 1}",
            "series_description": "Sagittal T2/STIR",
        }
        for row in studies
    ]
    write_csv(
        manifests / "series.csv",
        ["study_id", "split", "series_id", "series_path", "series_description"],
        series,
    )
    labels = [
        {**row, "level": level, "label": str(index)}
        for row in studies
        for index, level in enumerate(module.LEVELS)
    ]
    write_csv(manifests / "labels.csv", ["study_id", "split", "level", "label"], labels)
    hashes = {
        name: sha256_file(manifests / name) for name in ("studies.csv", "series.csv", "labels.csv")
    }
    (manifests / "summary.json").write_text(
        json.dumps(
            {
                "output_sha256": hashes,
                "input_sha256": {name: sha256_file(data / name) for name in module.METADATA_FILES},
            }
        )
    )
    receipt_hash = hashlib.sha256(
        (
            json.dumps(
                {name: hashes[name] for name in ("studies.csv", "series.csv")},
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode()
    ).hexdigest()
    return data, manifests, output, series, receipt_hash


def make_available(data, row, receipt_hash):
    directory = data / row["series_path"]
    directory.mkdir(parents=True)
    for name in ("1.dcm", "2.dcm"):
        (directory / name).write_bytes(
            b"dummy bytes; pixels are provided by the focused test loader"
        )
    (directory / module.COMPLETE_MARKER).write_text(
        json.dumps(
            {
                "complete": True,
                "expected_files": 2,
                "files": ["1.dcm", "2.dcm"],
                "manifest_sha256": receipt_hash,
                "metadata_sha256": "1" * 64,
                "source_identity_sha256": "2" * 64,
            }
        )
    )


class Grading(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_ccs = torch.nn.Linear(512, 4)

    def forward(self, images):
        values = images.mean(dim=(1, 2, 3, 4))[:, None].expand(-1, 512)
        return None, None, self.fc_ccs(values)


class Spine:
    def __init__(self):
        self.grading_model = Grading().eval().requires_grad_(False)
        self.calls = 0
        self.missing_s1 = False

    def detect_vb(self, volume, spacing):
        self.calls += 1
        names = ["L4", "L5"] if self.missing_s1 else ["L4", "L5", "S1"]
        return [
            {
                "predicted_label": name,
                "slice_nos": [1, 2],
                "polys": np.zeros((2, 4, 2)),
                "average_polygon": np.zeros((4, 2)),
            }
            for name in names
        ]

    def get_ivds_from_vert_dicts(self, vertices, volume):
        roi = np.arange(9 * 112 * 224, dtype=np.float64).reshape(9, 112, 224) / 100_000
        levels = ["L4-L5"] if self.missing_s1 else ["L4-L5", "L5-S1"]
        return [{"level_name": level, "volume": roi + index} for index, level in enumerate(levels)]


@pytest.fixture
def fake_pipeline(monkeypatch):
    model = Spine()
    monkeypatch.setattr(
        module,
        "_source_signature",
        lambda cache: {"upstream_commit": "fixed", "weights_sha256": {"grading": "0" * 64}},
    )
    monkeypatch.setattr(module, "_load_model", lambda cache, device: model)

    def load(_path):
        return SimpleNamespace(
            volume=np.ones((16, 16, 9), np.float32),
            pixel_spacing=(0.5, 0.5),
            metadata={"shape_hws": [16, 16, 9]},
        )

    monkeypatch.setattr(module, "load_sagittal_series", load)
    return model


def test_probability_merge_precedes_argmax_and_is_stable():
    probabilities = np.array([[0.30, 0.29, 0.31, 0.10]])
    logits = np.log(probabilities)
    merged = module.merge_original_logits(logits + 10_000)
    assert logits.argmax() == 2
    assert merged.argmax() == 0
    np.testing.assert_allclose(merged, [[0.59, 0.31, 0.10]], atol=1e-7)
    with pytest.raises(ValueError, match="finite"):
        module.merge_original_logits(np.array([[0, 0, np.nan, 0]]))


def test_exact_feature_capture_keeps_frozen_model_unchanged():
    model = Grading().eval().requires_grad_(False)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    rois = np.arange(2 * 9 * 112 * 224, dtype=np.float64).reshape(2, 9, 112, 224) / 100_000
    features, logits = module._capture_grading(model, rois, "cpu")
    assert features.shape == (2, 512)
    np.testing.assert_allclose(
        logits, model.fc_ccs(torch.from_numpy(features)).detach().numpy(), atol=1e-6
    )
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_resume_new_downloads_and_never_touch_test(tmp_path, fake_pipeline):
    data, manifests, output, rows, receipt = make_cohort(tmp_path)
    make_available(data, rows[0], receipt)
    make_available(data, rows[2], receipt)  # Test bytes exist but must not be inspected/inferred.
    first = module.extract_features(data, manifests, tmp_path / "cache", output, device="cpu")
    assert first["n_features"] == 2
    assert first["extraction_complete"] is False
    assert fake_pipeline.calls == 1
    with np.load(output / "features.npz", allow_pickle=False) as arrays:
        assert set(arrays["split"]) == {"train"}
        assert arrays["features"].shape == (2, 512)
        assert arrays["original_logits"].shape == (2, 4)
    with np.load(output / "records/10_11.npz", allow_pickle=False) as arrays:
        assert arrays["rois"].dtype == np.float64
    make_available(data, rows[1], receipt)
    second = module.extract_features(data, manifests, tmp_path / "cache", output, device="cpu")
    assert second["extraction_complete"] is True
    assert second["n_features"] == 4
    assert second["runtime"]["reused"] == 1
    assert fake_pipeline.calls == 2
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["extraction_complete"] is True
    for name, digest in metadata["outputs_sha256"].items():
        assert sha256_file(output / name) == digest
    assert not (output / "records/30_31.json").exists()
    with pytest.raises(ValueError, match="test is sealed"):
        module.extract_features(
            data, manifests, tmp_path / "cache", output, device="cpu", splits=("test",)
        )


def test_missing_target_and_changed_source_are_explicit(tmp_path, fake_pipeline):
    data, manifests, output, rows, receipt = make_cohort(tmp_path)
    make_available(data, rows[0], receipt)
    fake_pipeline.missing_s1 = True
    result = module.extract_features(data, manifests, tmp_path / "cache", output, device="cpu")
    assert result["n_features"] == 1
    record = json.loads((output / "records/10_11.json").read_text())
    assert record["targets"][1]["status"] == "not_assessed"
    assert record["targets"][1]["reason"] == "missing_vertebra:S1"
    (data / rows[0]["series_path"] / "1.dcm").write_bytes(b"changed source bytes")
    with pytest.raises(ValueError, match="Source DICOM bytes changed"):
        module.extract_features(data, manifests, tmp_path / "cache", output, device="cpu")


def test_incomplete_marker_never_becomes_a_processing_failure(tmp_path, fake_pipeline):
    data, manifests, output, rows, receipt = make_cohort(tmp_path)
    make_available(data, rows[0], receipt)
    (data / rows[0]["series_path"] / "2.dcm").unlink()
    result = module.extract_features(data, manifests, tmp_path / "cache", output, device="cpu")
    assert result["n_features"] == 0
    assert result["runtime"]["attempted"] == 0
    assert fake_pipeline.calls == 0
    assert not (output / "records/10_11.json").exists()


def test_corrupt_cached_arrays_are_refused(tmp_path, fake_pipeline):
    data, manifests, output, rows, receipt = make_cohort(tmp_path)
    make_available(data, rows[0], receipt)
    module.extract_features(data, manifests, tmp_path / "cache", output, device="cpu")
    (output / "records/10_11.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="missing or corrupted"):
        module.extract_features(data, manifests, tmp_path / "cache", output, device="cpu")
