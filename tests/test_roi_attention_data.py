import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from ai_spine_med import roi_attention_data as data
from ai_spine_med.assets import sha256_file, write_json


@pytest.fixture(autouse=True)
def small_thread_count():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def row(level="L4-L5", index=0, **updates):
    result = {
        "dataset": "rsna",
        "study_id": "synthetic-a",
        "split": "train",
        "level": level,
        "label": 2,
        "status": "ok",
        "reason": "",
        "feature_index": index,
    }
    result.update(updates)
    return result


def synthetic_cache():
    features = np.repeat(np.arange(1, 5, dtype=np.float32)[:, None], 512, 1)
    offsets = np.array([[0, 1], [1, 4]], np.int64)
    rows = [
        row(),
        row("L5-S1", 1),
        row(index=None, study_id="failed", status="failed", reason="missing", label=None),
    ]
    return rows, features, offsets, data.mean_features(features, offsets)


class IdentityTail(nn.Module):
    def forward(self, x):
        return x[:, 0, 0, 0, None].expand(-1, 512)


def test_extraction_preserves_each_center_instead_of_mean():
    maps = np.repeat(np.arange(1, 5, dtype=np.float32)[:, None, None, None], 256, 1)
    maps = np.broadcast_to(maps, (4, 256, 14, 14)).copy()
    actual = data.extract_centers(maps, IdentityTail(), "cpu", batch_size=2)
    assert actual.shape == (4, 512)
    np.testing.assert_array_equal(actual[:, 0], [1, 2, 3, 4])
    np.testing.assert_array_equal(
        data.mean_features(actual, np.array([[0, 1], [1, 4]]))[:, 0], [1, 3]
    )


def test_original_frozen_tail_remains_eval_and_preserves_bn_statistics():
    tail = nn.Sequential(
        nn.Conv2d(256, 512, 1), nn.BatchNorm2d(512), nn.AdaptiveAvgPool2d(1), nn.Flatten(1)
    ).train()
    original = copy.deepcopy(tail.state_dict())
    data.extract_centers(np.ones((3, 256, 14, 14), np.float32), tail, "cpu", batch_size=2)
    assert not tail.training and not tail[1].training
    assert not any(p.requires_grad for p in tail.parameters())
    assert all(torch.equal(value, tail.state_dict()[key]) for key, value in original.items())


def test_offsets_keep_variable_center_counts_and_original_failures():
    rows, centers, offsets, frozen = synthetic_cache()
    data.validate_arrays(rows, centers, offsets, frozen)
    assert len(centers) == 4 and len(frozen) == 2
    assert rows[-1]["feature_index"] is None and rows[-1]["label"] is None
    with pytest.raises(ValueError, match="aligned"):
        data.validate_arrays(rows, centers[:2], offsets, frozen)


@pytest.mark.parametrize("change", ["test", "domain", "duplicate", "offsets", "nan", "grade"])
def test_invalid_reference_or_feature_alignment_rejected(change):
    rows, centers, offsets, frozen = synthetic_cache()
    if change == "test":
        rows[0]["split"] = "test"
    elif change == "domain":
        rows[0]["dataset"] = "aisslab"
    elif change == "duplicate":
        rows[1]["level"] = rows[0]["level"]
    elif change == "offsets":
        offsets[1, 0] = 0
    elif change == "nan":
        centers[0, 0] = np.nan
    else:
        rows[0]["label"] = 7
    with pytest.raises(ValueError):
        data.validate_arrays(rows, centers, offsets, frozen)


def test_partition_preserves_rows_and_exact_old_mean_then_resumes(tmp_path, monkeypatch):
    rows, centers, offsets, frozen = synthetic_cache()
    maps = np.broadcast_to(centers[:, :1, None, None], (4, 256, 14, 14)).copy()
    source_receipt = {"sha256": {"prefix.npy": "synthetic-prefix-hash"}}
    monkeypatch.setattr(
        data.previous, "load_cache", lambda *_a: (rows, maps, offsets, frozen, source_receipt)
    )
    output = tmp_path / "part"
    receipt = data.extract_partition("unused", "mixed", "train", output, IdentityTail(), "cpu")
    assert receipt["max_abs_mean_feature_error"] == 0
    assert receipt["n_targets"] == 3 and receipt["n_assessed"] == 2
    np.testing.assert_array_equal(np.load(output / "frozen_features.npy"), frozen)
    assert json.loads((output / "rows.json").read_text()) == rows
    monkeypatch.setattr(data, "extract_centers", lambda *_a, **_k: pytest.fail("reran frozen tail"))
    assert (
        data.extract_partition("unused", "mixed", "train", output, IdentityTail(), "cpu") == receipt
    )
    changed = np.load(output / "center_features.npy")
    changed[0, 0] += 1
    np.save(output / "center_features.npy", changed)
    with pytest.raises(ValueError, match="partition changed"):
        data.extract_partition("unused", "mixed", "train", output, IdentityTail(), "cpu")


def test_partition_rejects_changed_mean_features(tmp_path, monkeypatch):
    rows, centers, offsets, frozen = synthetic_cache()
    maps = np.broadcast_to(centers[:, :1, None, None], (4, 256, 14, 14)).copy()
    monkeypatch.setattr(
        data.previous, "load_cache", lambda *_a: (rows, maps, offsets, frozen + 1, {"sha256": {}})
    )
    with pytest.raises(ValueError, match="historical feature tolerance"):
        data.extract_partition(
            "unused", "mixed", "train", tmp_path / "cache", IdentityTail(), "cpu"
        )


def test_test_split_rejected_before_opening_source_cache(monkeypatch):
    monkeypatch.setattr(data.previous, "load_cache", lambda *_a: pytest.fail("read TEST cache"))
    with pytest.raises(ValueError, match="TRAIN/validation"):
        data.extract_partition("missing", "mixed", "test", "unused", IdentityTail(), "cpu")
    with pytest.raises(ValueError, match="TRAIN/validation"):
        data.load_cache("missing", split="test")


def test_verify_protocol_rejects_test_before_hashing_missing_files(tmp_path):
    path = tmp_path / "protocol.json"
    write_json(
        path,
        {
            "phase": "test",
            "test_evaluated": True,
            "arms": list(data.ARMS),
            "frozen_files": {"missing": "wrong"},
        },
    )
    with pytest.raises(ValueError, match="development-only"):
        data.verify_protocol(path)


def write_cache(tmp_path):
    rows, centers, offsets, frozen = synthetic_cache()
    directory = tmp_path / "mixed" / "train"
    directory.mkdir(parents=True)
    write_json(directory / "rows.json", rows)
    for name, values in (
        ("center_features.npy", centers),
        ("offsets.npy", offsets),
        ("frozen_features.npy", frozen),
    ):
        np.save(directory / name, values)
    receipt = {"sha256": {name: sha256_file(directory / name) for name in data.FILES}}
    summary = {
        "phase": "train_validation",
        "test_evaluated": False,
        "encoder": "original_frozen_imagenet_resnet18",
        "per_center_features_preserved": True,
        "labels_used_for_feature_extraction": False,
        "provenance": {"implementation_sha256": sha256_file(Path(data.__file__))},
        "partitions": {"mixed/train": receipt},
    }
    write_json(tmp_path / "summary.json", summary)
    return summary


def test_load_cache_checks_hashes_and_returns_nonpooled_centers(tmp_path):
    write_cache(tmp_path)
    rows, centers, offsets, frozen, _ = data.load_cache(tmp_path)
    assert centers.shape == (4, 512) and frozen.shape == (2, 512)
    assert offsets.tolist() == [[0, 1], [1, 4]]
    assert rows[-1]["status"] == "failed"
    (tmp_path / "mixed" / "train" / "rows.json").write_text("[]")
    with pytest.raises(ValueError, match="cache changed"):
        data.load_cache(tmp_path)


def test_load_cache_rejects_altered_data_provenance(tmp_path):
    summary = write_cache(tmp_path)
    summary["provenance"]["implementation_sha256"] = "altered"
    write_json(tmp_path / "summary.json", summary)
    with pytest.raises(ValueError, match="original frozen"):
        data.load_cache(tmp_path)
