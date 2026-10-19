import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ai_spine_med import multimodal as module
from ai_spine_med.assets import sha256_file, write_json


def make_caches(tmp_path, *, failures=False, long_bag=False):
    sag, axial = tmp_path / "sag", tmp_path / "axial"
    sag.mkdir()
    (axial / "records").mkdir(parents=True)
    rng = np.random.default_rng(29)
    targets, groups, entries, outputs = [], [], [], {}
    axial_config = {
        "schema_version": 1,
        "allowed_splits": ["train", "validation"],
        "encoder": module.EXPECTED_ENCODER,
        "manifest_input_sha256": {"studies.csv": "2" * 64},
        "preprocessing": {"encoder_updated": False},
    }
    axial_id = module._axial_config_id(axial_config)
    for i in range(15):
        study, split = str(100 + i), "train" if i < 12 else "validation"
        for level_index, level in enumerate(module.LEVELS):
            missing = failures and i == 0 and level_index == 1
            label = (i + level_index) % 3
            row = {
                "study_id": study,
                "series_id": str(1000 + i),
                "split": split,
                "level": level,
                "label": label,
                "status": "not_assessed" if missing else "ok",
                "reason": "missing_vertebra:S1" if missing else "",
            }
            targets.append(row)
            if not missing:
                groups.append({**row, "features": rng.normal(size=512).astype(np.float32) + label})
        failed = failures and i == 1
        n = 113 if long_bag and i == 0 else 4 + i % 4
        relative = f"records/{study}.npz"
        array_sha = ""
        if not failed:
            np.savez_compressed(
                axial / relative,
                features=rng.normal(size=(n, 512)).astype(np.float32),
                positions=np.column_stack([np.linspace(0, 1, n), np.linspace(0, 1, n)]).astype(
                    np.float32
                ),
                instance=np.arange(1, n + 1, dtype=np.int64),
                series_id=np.full(n, str(5000 + i)),
            )
            array_sha = sha256_file(axial / relative)
            outputs[relative] = array_sha
        record = {
            "schema_version": 1,
            "cache_id": axial_id,
            "study_id": study,
            "split": split,
            "status": "failed" if failed else "ok",
            "reason": "invalid_geometry" if failed else "",
            "arrays_path": "" if failed else relative,
            "arrays_sha256": array_sha,
            "n_slices": 0 if failed else n,
            "n_series": 1,
            "source_ready": True,
        }
        record_path = f"records/{study}.json"
        write_json(axial / record_path, record)
        outputs[record_path] = sha256_file(axial / record_path)
        entries.append(
            {**record, "record_path": record_path, "record_sha256": outputs[record_path]}
        )
    np.savez_compressed(
        sag / "features.npz",
        features=np.stack([g["features"] for g in groups]),
        original_logits=np.zeros((len(groups), 4), np.float32),
        labels=np.array([g["label"] for g in groups], np.int64),
        **{
            name: np.array([g[name] for g in groups])
            for name in ("study_id", "series_id", "split", "level")
        },
    )
    with (sag / "target_manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(targets[0]))
        writer.writeheader()
        writer.writerows(targets)
    write_json(
        sag / "metadata.json",
        {
            "cache_id": "synthetic-fixed-sag",
            "config": {"manifests": {"studies.csv": "2" * 64}},
            "extraction_complete": True,
            "outputs_sha256": {
                name: sha256_file(sag / name) for name in ("features.npz", "target_manifest.csv")
            },
        },
    )
    write_json(
        axial / "metadata.json",
        {
            "schema_version": 1,
            "scope": "train_validation",
            "cache_id": axial_id,
            "extraction_complete": True,
            "config": axial_config,
            "study_records": entries,
            "outputs_sha256": outputs,
        },
    )
    return sag, axial


def identity_normalizers():
    return {
        field: torch.ones(512) if field.endswith("std") else torch.zeros(512)
        for field in ("sag_mean", "sag_std", "axial_mean", "axial_std")
    }


def test_loader_preserves_failures_and_identical_common_target_mask(tmp_path):
    sag, axial = make_caches(tmp_path, failures=True, long_bag=True)
    data = module.load_paired_data(sag, axial)
    assert data.sag.shape == (15, 2, 512) and data.axial.shape == (15, 96, 512)
    assert data.slice_mask[0].sum() == 96
    np.testing.assert_array_equal(data.selected_indices[0], np.linspace(0, 112, 96, dtype=np.int64))
    assert data.paired[0].tolist() == [True, False]
    assert data.paired[1].tolist() == [False, False]
    assert not data.slice_mask[1].any()
    audit = module.availability(data)["train"]
    assert (
        audit["n_requested"] == 24
        and audit["sagittal_available"] == 23
        and audit["axial_available"] == 22
    )
    assert audit["paired_assessed"] == 21 and audit["not_assessed"] == 3
    probabilities = np.full((15, 2, 3), 1 / 3, np.float32)
    rows = module._rows(data, np.arange(12), probabilities)
    assert len(rows) == 24
    missing = [row for row in rows if row["status"] != "ok"]
    assert len(missing) == 3 and all(
        row["prediction"] is None and row["p_normal_mild"] is None for row in missing
    )
    assert {row["reason"] for row in missing} == {
        "sagittal:missing_vertebra:S1",
        "axial:processing_failed",
    }


def test_normalizers_use_only_fitting_train_and_valid_slices(tmp_path):
    sag, axial = make_caches(tmp_path, failures=True)
    data = module.load_paired_data(sag, axial)
    fitting = np.arange(2, 8)
    baseline = module.fit_normalizers(data, fitting)
    expected_sag = data.sag[fitting][data.paired[fitting]].numpy()
    expected_axial = np.concatenate([data.axial[i][data.slice_mask[i]].numpy() for i in fitting])
    np.testing.assert_allclose(baseline["sag_mean"], expected_sag.mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(baseline["axial_std"], expected_axial.std(axis=0), atol=1e-6)
    held = [0, 1, 8, 9, 10, 11, 12, 13, 14]
    data.sag[held] = 1e8
    data.axial[held] = -1e8
    # Padding is not an image and must not affect fit statistics.
    for i in fitting:
        data.axial[i][~data.slice_mask[i]] = 1e9
    after = module.fit_normalizers(data, fitting)
    for key in ("sag_mean", "sag_std", "axial_mean", "axial_std"):
        torch.testing.assert_close(after[key], baseline[key], rtol=0, atol=0)
    assert baseline["n_sagittal_targets"] == 12
    with pytest.raises(ValueError, match="TRAIN only"):
        module.fit_normalizers(data, np.array([12]))


def test_variant_input_independence_and_masked_attention():
    torch.manual_seed(9)
    sag = torch.randn(2, 2, 512)
    axial = torch.randn(2, 7, 512)
    positions = torch.rand(2, 7, 2)
    mask = torch.tensor(
        [
            [True, True, False, False, False, False, False],
            [True, True, True, True, False, False, False],
        ]
    )
    sagittal_model = module.MultimodalHead("sagittal", identity_normalizers()).eval()
    expected = sagittal_model(sag, axial, positions, mask)
    torch.testing.assert_close(
        sagittal_model(sag, axial * 10000, positions * 100, ~mask), expected, rtol=0, atol=0
    )
    axial_model = module.MultimodalHead("axial", identity_normalizers()).eval()
    expected, weights = axial_model(sag, axial, positions, mask, return_attention=True)
    torch.testing.assert_close(
        axial_model(torch.full_like(sag, float("nan")), axial, positions, mask),
        expected,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 2), atol=1e-6, rtol=0)
    assert torch.count_nonzero(weights.masked_select(~mask[:, None].expand_as(weights))) == 0
    padded = axial.clone()
    padded[~mask] = 1e12
    torch.testing.assert_close(axial_model(sag, padded, positions, mask), expected, rtol=0, atol=0)
    permutation = torch.tensor([6, 1, 4, 0, 3, 5, 2])
    torch.testing.assert_close(
        axial_model(sag, axial[:, permutation], positions[:, permutation], mask[:, permutation]),
        expected,
        rtol=0,
        atol=1e-6,
    )
    with pytest.raises(ValueError, match="nonempty"):
        axial_model(sag, axial, positions, torch.zeros_like(mask))


def test_cache_corruption_wrong_encoder_or_test_scope_are_rejected(tmp_path):
    sag, axial = make_caches(tmp_path)
    path = axial / "metadata.json"
    metadata = json.loads(path.read_text())
    bad = json.loads(path.read_text())
    bad["scope"] = "test"
    write_json(path, bad)
    with pytest.raises(ValueError, match="TEST is forbidden"):
        module.load_paired_data(sag, axial)
    bad = json.loads(json.dumps(metadata))
    bad["config"]["encoder"]["sha256"] = "0" * 64
    write_json(path, bad)
    with pytest.raises(ValueError, match="ImageNet weights"):
        module.load_paired_data(sag, axial)
    write_json(path, metadata)
    first = metadata["study_records"][0]
    record_path = axial / first["record_path"]
    original_record = json.loads(record_path.read_text())
    # Even when all file hashes are updated, records from another cache/config
    # cannot be silently mixed into this frozen feature inventory.
    mixed = dict(original_record)
    mixed["cache_id"] = "9" * 64
    write_json(record_path, mixed)
    first["record_sha256"] = sha256_file(record_path)
    metadata["outputs_sha256"][first["record_path"]] = first["record_sha256"]
    write_json(path, metadata)
    with pytest.raises(ValueError, match="per-study record disagree"):
        module.load_paired_data(sag, axial)
    write_json(record_path, original_record)
    first["record_sha256"] = sha256_file(record_path)
    metadata["outputs_sha256"][first["record_path"]] = first["record_sha256"]
    write_json(path, metadata)
    first = metadata["study_records"][0]
    (axial / first["arrays_path"]).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="bag checksum"):
        module.load_paired_data(sag, axial)


def test_cv_earliest_tie_final_training_and_safe_checkpoint_roundtrip(tmp_path):
    sag, axial = make_caches(tmp_path)
    data = module.load_paired_data(sag, axial)
    fit = np.arange(6)
    held = np.arange(6, 12)
    normalizers = module.fit_normalizers(data, fit)
    config = {"learning_rate": 0.0, "weight_decay": 0.0, "batch_size": 4, "patience": 1}
    model, history, epoch, weights = module._fit(
        "fusion", 42, data, fit, normalizers, config, epochs=4, held=held
    )
    assert epoch == 1 and len(history) == 2
    original_sag = data.sag.clone()
    original_axial = data.axial.clone()
    provenance = {"protocol_sha256": "a" * 64, "implementation_sha256": "b" * 64}
    result, rows, attention = module._export_run(
        tmp_path / "checkpoint",
        model,
        history,
        epoch,
        42,
        data,
        held,
        normalizers,
        weights,
        provenance,
        scope="synthetic TRAIN held fold",
        fold=0,
    )
    assert result["checkpoint_roundtrip_max_probability_error"] == 0.0
    assert len(rows) == 12 and attention.shape == (15, 2, 96)
    reopened, checkpoint = module.load_multimodal_checkpoint(tmp_path / "checkpoint/model.pt")
    assert checkpoint["encoder_updated"] is False
    assert all(
        not buffer.requires_grad
        for buffer in (reopened.sag_mean, reopened.sag_std, reopened.axial_mean, reopened.axial_std)
    )
    torch.testing.assert_close(data.sag, original_sag, rtol=0, atol=0)
    torch.testing.assert_close(data.axial, original_axial, rtol=0, atol=0)
    final, final_history, final_epoch, _ = module._fit(
        "axial",
        42,
        data,
        np.arange(12),
        module.fit_normalizers(data, np.arange(12)),
        config,
        epochs=2,
    )
    assert final_epoch == 2 and len(final_history) == 2
    assert all("heldout_macro_f1" not in row for row in final_history)
    assert final.variant == "axial"
    with pytest.raises(ValueError, match="disjoint"):
        module._fit("fusion", 42, data, fit, normalizers, config, epochs=1, held=np.arange(3))


def test_attention_mapping_uses_original_selected_slice_indices(tmp_path):
    sag, axial = make_caches(tmp_path, long_bag=True)
    data = module.load_paired_data(sag, axial)
    model = module.MultimodalHead("axial", identity_normalizers()).eval()
    indices = np.arange(12, 15)
    _, attention = module._probabilities(model, data, indices, attention=True)
    path = tmp_path / "attention.json"
    module._attention_report(path, data, indices, attention)
    report = json.loads(path.read_text())
    assert len(report["records"]) == 3 and "PRIVATE" in report["scope"]
    for record in report["records"]:
        i = data.study_ids.index(record["study_id"])
        for level in module.LEVELS:
            assert len(record["levels"][level]["top3"]) == 3
            for item in record["levels"][level]["top3"]:
                index = item["bag_index"]
                assert item["source_index"] == int(data.selected_indices[i][index])
                assert item["instance"] == int(data.selected_instances[i][index])
                assert item["series_id"] == str(data.selected_series[i][index])


def test_end_to_end_fixed_folds_complete_outputs_and_validation_epoch_lock(tmp_path, monkeypatch):
    sag, axial = make_caches(tmp_path, failures=True)
    # Exercise the whole 27-CV + nine-final artifact graph using one epoch per
    # tiny synthetic fit. Real protocol validation remains active and unchanged.
    root = Path(__file__).parents[1]
    protocol = json.loads((root / "configs/multimodal-v3.json").read_text())
    protocol.update(
        feature_cache_id="synthetic-fixed-sag", sagittal_feature_cache_id="synthetic-fixed-sag"
    )
    path = tmp_path / "protocol.json"
    write_json(path, protocol)
    output = tmp_path / "result"
    original_fit, original_predict = module._fit, module._probabilities

    def one_epoch(*args, **kwargs):
        kwargs["epochs"] = 1
        return original_fit(*args, **kwargs)

    def check_epoch_lock(model, data, indices, **kwargs):
        if np.any(data.splits[indices] == "validation"):
            assert (output / "final-epochs.json").is_file()
        return original_predict(model, data, indices, **kwargs)

    monkeypatch.setattr(module, "_fit", one_epoch)
    monkeypatch.setattr(module, "_probabilities", check_epoch_lock)
    summary = module.run_multimodal(sag, axial, path, output, threads=2)
    assert (
        summary["test_evaluated"] is False and summary["new_version_independently_tested"] is False
    )
    assert summary["availability"]["train"]["paired_assessed"] == 21
    folds = json.loads((output / "folds.json").read_text())
    assert len({(row["study_id"], row["fold"]) for row in folds["assignments"]}) == 12
    for variant in module.VARIANTS:
        cv = summary["variants"][variant]["cv"]
        assert len(cv["folds"]) == 3 and len(cv["oof"]["runs"]) == 3
        assert cv["oof"]["ensemble"]["metrics"]["n_requested"] == 24
        final = summary["variants"][variant]["validation"]
        assert len(final["runs"]) == 3 and final["ensemble"]["metrics"]["n_requested"] == 6
        assert all(run["checkpoint_roundtrip_max_probability_error"] == 0 for run in final["runs"])
        if variant != "sagittal":
            attention_path = output / "validation" / variant / "attention.json"
            assert final["attention_sha256"] == sha256_file(attention_path)
    assert set(summary["bootstrap"]["contrasts"]) == {
        "fusion_minus_sagittal",
        "axial_minus_sagittal",
    }
    assert set(summary["bootstrap"]["by_level"]) == set(module.LEVELS)
    with pytest.raises(ValueError, match="Output already exists"):
        module.run_multimodal(sag, axial, path, output)


def test_unregistered_frame_groups_may_reset_z_but_each_series_keeps_its_order(tmp_path):
    sag, axial = make_caches(tmp_path)
    metadata_path = axial / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    entry = metadata["study_records"][0]
    array_path, record_path = axial / entry["arrays_path"], axial / entry["record_path"]
    with np.load(array_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["series_id"] = np.array(["series_a", "series_a", "series_b", "series_b"])
    arrays["instance"] = np.array([1, 2, 1, 2], np.int64)
    arrays["positions"] = np.array([[0, 0], [1, 1], [0, 0], [1, 1]], np.float32)

    def publish():
        np.savez_compressed(array_path, **arrays)
        record = json.loads(record_path.read_text())
        entry["arrays_sha256"] = record["arrays_sha256"] = sha256_file(array_path)
        entry["n_series"] = record["n_series"] = 2
        write_json(record_path, record)
        entry["record_sha256"] = sha256_file(record_path)
        metadata["outputs_sha256"][entry["arrays_path"]] = entry["arrays_sha256"]
        metadata["outputs_sha256"][entry["record_path"]] = entry["record_sha256"]
        write_json(metadata_path, metadata)

    publish()
    loaded = module.load_paired_data(sag, axial)
    assert loaded.slice_mask[0].sum() == 4
    assert loaded.selected_series[0].tolist() == arrays["series_id"].tolist()
    arrays["positions"][:2, 1] = [1, 0]
    publish()
    with pytest.raises(ValueError, match="unordered axial"):
        module.load_paired_data(sag, axial)
