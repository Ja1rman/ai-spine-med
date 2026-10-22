"""Prefix caches preserve real ResNet algebra and synthetic frozen ROI receipts."""

import copy
import json

import numpy as np
import pytest
import torch
from torchvision.models import resnet18

from ai_spine_med import axial_features as af
from ai_spine_med import roi_layer4_data as data
from ai_spine_med.assets import sha256_file, write_json


@pytest.fixture(autouse=True)
def small_thread_count():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def row(level="L4-L5", index=0, **updates):
    result = {
        "dataset": "rsna", "study_id": "synthetic-a", "split": "train",
        "level": level, "label": 2, "status": "ok", "reason": "", "feature_index": index,
    }
    result.update(updates)
    return result


def test_prefix_and_tail_reproduce_original_resnet18_without_bn_updates():
    torch.manual_seed(9)
    encoder = resnet18(weights=None).eval()
    encoder.fc = torch.nn.Identity()
    before = {k: v.clone() for k, v in encoder.state_dict().items() if "running_" in k}
    prefix, tail = data.prefix_and_tail(encoder)
    images = torch.randn(3, 3, 224, 224)
    with torch.inference_mode():
        maps = prefix(images)
        assert maps.shape == (3, 256, 14, 14)
        torch.testing.assert_close(tail(maps), encoder(images), atol=0, rtol=0)
    assert all(not p.requires_grad for p in prefix.parameters())
    assert all(torch.equal(encoder.state_dict()[k], value) for k, value in before.items())


def test_test_split_rejected_before_image_descriptor_access(monkeypatch):
    monkeypatch.setattr(data.v8, "image_descriptor", lambda _: pytest.fail("read TEST images"))
    with pytest.raises(ValueError, match="TEST is forbidden"):
        data.fitting_records({"records": [{"dataset": "rsna", "split": "test"}]})


def test_fitting_patient_overlap_is_rejected(monkeypatch):
    monkeypatch.setattr(data.v8, "image_descriptor", lambda x: dict(x))
    records = [{"dataset": "rsna", "study_id": "same", "split": split} for split in data.SPLITS]
    with pytest.raises(ValueError, match="disjoint"):
        data.fitting_records({"records": records})


def test_reference_join_reindexes_exact_frozen_features_and_preserves_unknown_failure():
    image = [row(), row("L5-S1", None, status="failed", reason="source_missing")]
    for r in image:
        r.pop("label")
    original = [row(index=1), row("L5-S1", None, label=None, status="failed", reason="source_missing")]
    original.insert(0, row(index=0, dataset="aisslab", study_id="other"))
    source_features = np.stack([np.full(512, -1), np.full(512, 7)]).astype(np.float32)
    targets, frozen = data.join_targets(image, original, source_features)
    assert targets[0]["feature_index"] == 0
    assert targets[0]["label"] == 2
    assert targets[1]["label"] is None
    assert targets[1]["status"] == "failed"
    np.testing.assert_array_equal(frozen, source_features[1:])


@pytest.mark.parametrize("field,value", [("split", "validation"), ("status", "failed"), ("reason", "changed")])
def test_reference_join_cannot_hide_changed_coverage_or_split(field, value):
    source = row()
    with pytest.raises(ValueError, match="status or split"):
        data.join_targets([row(**{field: value})], [source], np.zeros((1, 512), np.float32))


def test_offsets_preserve_variable_size_bags_and_failures():
    rows = [row(), row("L5-S1", 1), row(index=None, study_id="failed", status="failed", reason="missing")]
    maps = np.zeros((4, *data.SHAPE), np.float32)
    offsets = np.array([[0, 1], [1, 4]], np.int64)
    data.validate_arrays(rows, maps, offsets, np.zeros((2, 512), np.float32))
    for invalid in (np.array([[0, 2], [1, 4]]), np.array([[0, 0], [0, 4]])):
        with pytest.raises(ValueError, match="aligned"):
            data.validate_arrays(rows, maps, invalid, np.zeros((2, 512), np.float32))


class SyntheticPrefix(torch.nn.Module):
    def forward(self, x):
        return x[:, :1, :1, :1].expand(-1, 256, 14, 14).clone()


class SyntheticTail(torch.nn.Module):
    def forward(self, x):
        return x[:, :1, 0, 0].expand(-1, 512)


def synthetic_source(tmp_path, monkeypatch):
    source = {"dataset": "rsna", "study_id": "synthetic", "split": "train"}
    directory = tmp_path / "source"
    (directory / "records").mkdir(parents=True)
    token = data._token(source)
    np.savez(directory / "records" / f"{token}.npz", features=np.array([[2] * 512, [5] * 512], np.float32))
    features_path = directory / "records" / f"{token}.npz"
    parameters = {"roi_mm": 50, "max_centers": 3, "window_mm": 10}
    crop_metadata = {level: {"centers": [{"synthetic": level}] * n} for level, n in zip(data.severity.LEVELS, (1, 3))}
    original = {
        **source,
        "image_input_sha256": af._hash_json(source),
        "features_path": f"records/{token}.npz",
        "features_sha256": sha256_file(features_path),
        "targets": [{"level": level, "status": "ok", "reason": "", "feature_index": i} for i, level in enumerate(data.severity.LEVELS)],
        "provenance": {
            "prediction": {"image_only": True},
            "parameters": parameters,
            "crops": {level: {"status": "ok", "metadata": crop_metadata[level]} for level in data.severity.LEVELS},
        },
    }
    write_json(directory / "records" / f"{token}.json", original)
    crops = {
        "parameters": parameters,
        "levels": {
            level: {
                "status": "ok", "metadata": crop_metadata[level],
                "images": torch.full((n, 3, 224, 224), float(value)),
            }
            for level, n, value in zip(data.severity.LEVELS, (1, 3), (2, 5))
        },
    }
    monkeypatch.setattr(data, "crop_level_inputs", lambda row, prediction, **kwargs: crops)
    return source, directory, crops


def test_exact_roi_cache_reconstruction_resumes_without_reading_images(tmp_path, monkeypatch):
    source, original, _ = synthetic_source(tmp_path, monkeypatch)
    config = {"roi_mm": 50, "roi_max_centers": 3, "roi_window_mm": 10}
    out = tmp_path / "cache"
    result = data.extract_partition([source], original, out, SyntheticPrefix(), SyntheticTail(), config, "cpu", 1e-4)
    assert result["max_abs_feature_error"] == 0
    assert result["n_crops"] == 4
    assert result["n_assessed"] == 2
    assert all("label" not in r for r in json.loads((out / "image-targets.json").read_text()))
    np.testing.assert_array_equal(np.load(out / "offsets.npy"), [[0, 1], [1, 4]])
    monkeypatch.setattr(data, "crop_level_inputs", lambda *_a, **_k: pytest.fail("reopened images"))
    assert data.extract_partition([source], original, out, SyntheticPrefix(), SyntheticTail(), config, "cpu", 1e-4) == result
    maps = np.load(out / "prefix.npy", mmap_mode="r+")
    maps[0, 0, 0, 0] += 1
    maps.flush()
    with pytest.raises(ValueError, match="per-study prefix cache changed"):
        data.extract_partition([source], original, out, SyntheticPrefix(), SyntheticTail(), config, "cpu", 1e-4)


@pytest.mark.parametrize("corrupt", ["geometry", "features"])
def test_reconstruction_rejects_changed_roi_or_feature_value(tmp_path, monkeypatch, corrupt):
    source, original, crops = synthetic_source(tmp_path, monkeypatch)
    crops = copy.deepcopy(crops)
    if corrupt == "geometry":
        crops["levels"]["L5-S1"]["metadata"]["centers"][0]["synthetic"] = "wrong-level"
    else:
        crops["levels"]["L5-S1"]["images"] += 1
    monkeypatch.setattr(data, "crop_level_inputs", lambda *_a, **_k: crops)
    with pytest.raises(ValueError, match="geometry changed|exceeds tolerance"):
        data.extract_partition([source], original, tmp_path / "cache", SyntheticPrefix(), SyntheticTail(), {"roi_mm": 50, "roi_max_centers": 3, "roi_window_mm": 10}, "cpu", 1e-4)


def test_load_cache_rejects_test_before_opening_any_files():
    with pytest.raises(ValueError, match="TRAIN/validation"):
        data.load_cache("nonexistent", split="test")


def test_verify_protocol_rejects_test_before_hashing_missing_files(tmp_path):
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps({"phase": "test", "test_evaluated": True, "arms": list(data.ARMS), "frozen_files": {"missing": "wrong"}}))
    with pytest.raises(ValueError, match="development-only"):
        data.verify_protocol(path)
