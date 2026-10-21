"""Synthetic last-block transfer tests; no downloaded patients or TEST data."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torchvision.models import resnet18

import ai_spine_med.axial_transfer as transfer
from ai_spine_med.multimodal import MultimodalHead, architecture
from ai_spine_med.training import evaluate_predictions


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def head(seed=42):
    torch.manual_seed(seed)
    normalizers = {
        name: torch.ones(512) if name.endswith("std") else torch.zeros(512)
        for name in transfer._NORMALIZERS
    }
    return MultimodalHead("axial", normalizers).eval()


def model(seed=42):
    torch.manual_seed(seed)
    return transfer.AxialTransferModel(resnet18(weights=None).layer4[1], head(seed))


def arrays(n=2):
    generator = torch.Generator().manual_seed(18)
    return torch.rand((n, 512, 7, 7), generator=generator), torch.linspace(0, 1, n)[:, None].expand(
        -1, 2
    ).clone()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol():
    return {
        "seeds": list(transfer.SEEDS),
        "classes": list(transfer.CLASSES),
        "levels": list(transfer.LEVELS),
        "max_epochs": 8,
        "patience": 3,
        "learning_rate_block": 1e-5,
        "learning_rate_head": 1e-4,
        "weight_decay": 1e-4,
        "batch_size": 1,
        "replay_ratio": 1.0,
        "replay_seed": 2026,
    }


def write_record(root, dataset, study, split, labels, *, n=1, status="ok"):
    row = {
        "dataset": dataset,
        "study_id": study,
        "split": split,
        "labels": labels,
        "status": status,
        "reason": "" if status == "ok" else "synthetic_missing",
        "arrays_path": None,
        "arrays_sha256": None,
    }
    if status == "ok":
        a, p = arrays(n)
        path = root / f"{dataset}-{study}.npz"
        np.savez(path, activations=a.numpy(), positions=p.numpy())
        row.update(arrays_path=path.name, arrays_sha256=sha(path))
    return row


def write_index(path, rows):
    path.write_text(json.dumps({"schema_version": 1, "records": rows}))
    return path


def export_original_head(path, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    h = head(seed)
    torch.save(
        {
            "schema_version": 1,
            "variant": "axial",
            "architecture": architecture("axial"),
            "classes": list(transfer.CLASSES),
            "levels": list(transfer.LEVELS),
            "encoder_updated": False,
            "seed": seed,
            "fold": None,
            "state_dict": h.state_dict(),
        },
        path,
    )


def test_warmstart_equals_original_and_training_preserves_prefix_and_buffers(tmp_path):
    torch.manual_seed(18)
    original = resnet18(weights=None).eval().requires_grad_(False)
    original_before = transfer._state_hash(original.state_dict())
    path = tmp_path / "head.pt"
    export_original_head(path, 42)
    m = transfer._new_model(original, path, 42)
    a, p = arrays()
    untouched = a.clone()
    old_head = head(42)
    with torch.inference_mode():
        features = original.avgpool(original.layer4[1](a)).flatten(1)
        expected = (
            old_head(
                torch.zeros((1, 2, 512)),
                features[None],
                p[None],
                torch.ones((1, len(a)), dtype=torch.bool),
            )[0]
            .softmax(-1)
            .numpy()
        )
    np.testing.assert_allclose(transfer.predict_record(m, a, p), expected, atol=1e-7, rtol=0)
    before = transfer._state(m)
    buffers = transfer._buffers(m)
    m.train()
    assert m.head.training
    assert all(
        not module.training
        for module in m.block.modules()
        if isinstance(module, torch.nn.BatchNorm2d)
    )
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-4)
    loss = torch.nn.functional.cross_entropy(m(a, p), torch.tensor([1, 2]))
    loss.backward()
    optimizer.step()
    transfer._assert_buffers(m, buffers)
    assert transfer._parameter_delta(m, before) > 0
    assert not torch.equal(m.block.conv1.weight, before["block.conv1.weight"])
    assert not torch.equal(m.head.head[3].weight, before["head.head.3.weight"])
    assert torch.equal(a, untouched)
    assert transfer._state_hash(original.state_dict()) == original_before


def test_masked_slice_does_not_change_predictions():
    m = model()
    a, p = arrays()
    expected = transfer.predict_record(m, a, p)
    padded = torch.cat([a, torch.full_like(a[:1], 100)])
    positions = torch.cat([p, torch.tensor([[0.5, 0.5]])])
    actual = transfer.predict_record(m, padded, positions, torch.tensor([True, True, False]))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
    with pytest.raises(ValueError, match="nonempty"):
        transfer.predict_record(m, a, p, torch.zeros(2, dtype=torch.bool))


def test_checkpoint_reload_is_exact_and_rejects_wrong_family_or_nonfinite(tmp_path):
    m = model()
    path = tmp_path / "model.pt"
    provenance = {
        "original_encoder_weights_sha256": transfer.WEIGHTS_SHA256,
        "protocol_sha256": "a" * 64,
        "training_index_sha256": "b" * 64,
        "initial_head_sha256": "c" * 64,
    }
    transfer._save_checkpoint(path, m, seed=42, epoch=0, phase="after", provenance=provenance)
    reopened, header = transfer.load_transfer_checkpoint(path)
    assert header["best_epoch"] == 0
    assert all(
        torch.equal(value, reopened.state_dict()[name]) for name, value in m.state_dict().items()
    )
    np.testing.assert_array_equal(
        transfer.predict_record(m, *arrays()), transfer.predict_record(reopened, *arrays())
    )
    state = torch.load(path, weights_only=True)
    state["model_family"] = "old_or_unrelated"
    torch.save(state, path)
    with pytest.raises(ValueError, match="architecture"):
        transfer.load_transfer_checkpoint(path)
    state["model_family"] = transfer.FAMILY
    state["state_dict"]["head.axial_std"][0] = float("nan")
    torch.save(state, path)
    with pytest.raises(ValueError):
        transfer.load_transfer_checkpoint(path)


def test_index_only_reads_metadata_and_rejects_traversal_or_split_duplicates(tmp_path, monkeypatch):
    rows = [write_record(tmp_path, "aisslab", "study", "test", [0, 2])]
    path = write_index(tmp_path / "index.json", rows)
    monkeypatch.setattr(
        np, "load", lambda *a, **k: pytest.fail("Index parser must not read any array")
    )
    records, digest = transfer.read_activation_index(path)
    assert records[0].split == "test" and digest == sha(path)
    unsafe = copy.deepcopy(rows)
    unsafe[0]["arrays_path"] = "../escape.npz"
    write_index(path, unsafe)
    with pytest.raises(ValueError, match="escapes"):
        transfer.read_activation_index(path)
    write_index(path, rows + [{**rows[0], "split": "train"}])
    with pytest.raises(ValueError, match="crosses splits"):
        transfer.read_activation_index(path)


@pytest.mark.parametrize("bad", ["shape", "dtype", "position", "checksum", "too_many"])
def test_invalid_activation_arrays_rejected(tmp_path, bad):
    row = write_record(tmp_path, "aisslab", "study", "train", [0, None])
    path = tmp_path / row["arrays_path"]
    a, p = arrays()
    if bad == "shape":
        a = a[:, :, :6]
    elif bad == "dtype":
        a = a.double()
    elif bad == "position":
        p[0, 0] = float("nan")
    elif bad == "too_many":
        a, p = arrays(97)
    np.savez(path, activations=a.numpy(), positions=p.numpy())
    if bad != "checksum":
        row["arrays_sha256"] = sha(path)
    index = write_index(tmp_path / "index.json", [row])
    records, _ = transfer.read_activation_index(index)
    with pytest.raises(ValueError):
        transfer.load_record(records[0])


@pytest.fixture
def completed_synthetic_run(tmp_path, monkeypatch):
    rows = [
        write_record(tmp_path, "aisslab", "train", "train", [0, 1]),
        write_record(tmp_path, "aisslab", "validation", "validation", [1, 2]),
        write_record(tmp_path, "rsna", "replay", "train", [2, None]),
    ]
    for dataset, study, split in (
        ("aisslab", "forbidden_test", "test"),
        ("rsna", "unused_old_val", "validation"),
    ):
        forbidden = tmp_path / f"{study}.npz"
        forbidden.write_bytes(b"THIS IS NOT AN ARRAY AND MUST NEVER BE OPENED")
        rows.append(
            {
                "dataset": dataset,
                "study_id": study,
                "split": split,
                "labels": [2, 0],
                "status": "ok",
                "reason": "",
                "arrays_path": forbidden.name,
                "arrays_sha256": sha(forbidden),
            }
        )
    index = write_index(tmp_path / "index.json", rows)
    configuration = tmp_path / "protocol.json"
    configuration.write_text(json.dumps(protocol()))
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"SYNTHETIC-WEIGHTS-ONLY")
    monkeypatch.setattr(transfer, "WEIGHTS_SHA256", sha(weights))
    torch.manual_seed(1)
    original = resnet18(weights=None).eval().requires_grad_(False)
    monkeypatch.setattr(transfer, "_original_encoder", lambda _: original)
    heads = tmp_path / "heads"
    for seed in transfer.SEEDS:
        export_original_head(heads / f"seed-{seed}" / "model.pt", seed)
    seen = []
    load = transfer.load_record

    def guarded(record, **kwargs):
        assert record.split != "test" and record.study_id != "unused_old_val"
        seen.append(record.key)
        return load(record, **kwargs)

    monkeypatch.setattr(transfer, "load_record", guarded)
    reference_rows = [
        {"label": grade, "prediction": grade, "status": "ok", "level": level}
        for grade, level in zip((1, 2), transfer.LEVELS)
    ]
    # Flat held-out score intentionally selects epoch zero despite real updates.
    monkeypatch.setattr(transfer, "_assess", lambda *_: (evaluate_predictions(reference_rows), 1.0))
    output = tmp_path / "training"
    result = transfer.train_axial_transfer(
        index, configuration, output, head_dir=heads, weights=weights, device="cpu", threads=1
    )
    return output, result, seen, load


def test_training_never_opens_test_or_old_validation_and_can_keep_epoch_zero(
    completed_synthetic_run,
):
    output, result, seen, _ = completed_synthetic_run
    assert result["status"] == "complete" and result["test_evaluated"] is False
    assert [run["seed"] for run in result["runs"]] == list(transfer.SEEDS)
    assert all(run["best_epoch"] == 0 and run["epochs_run"] == 3 for run in result["runs"])
    assert all(
        run["training_parameter_delta_max_abs"] > 0 and run["selected_parameter_delta_max_abs"] == 0
        for run in result["runs"]
    )
    assert all(run["frozen_buffers_unchanged"] for run in result["runs"])
    assert all(run["block_parameter_delta_max_abs"] > 0 for run in result["runs"])
    assert all(run["selected_block_parameter_delta_max_abs"] == 0 for run in result["runs"])
    assert set(seen) == {("aisslab", "train"), ("rsna", "replay")}
    selection = json.loads((output / "selection.json").read_text())
    assert selection["frozen"] is True and selection["training_summary_sha256"] == sha(
        output / "summary.json"
    )


def test_evaluation_separate_test_index_and_overlap_rejection(
    completed_synthetic_run, tmp_path, monkeypatch
):
    output, _, _, load = completed_synthetic_run
    monkeypatch.setattr(transfer, "load_record", load)
    test_rows = [
        write_record(tmp_path, "aisslab", "heldout", "test", [0, 2]),
        write_record(tmp_path, "rsna", "unassessed", "test", [2, None], status="not_assessed"),
    ]
    path = write_index(tmp_path / "test-index.json", test_rows)
    result = transfer.evaluate_axial_transfer(path, output, tmp_path / "evaluation", device="cpu")
    assert len(result["entries"]) == 8
    assert result["evaluation_index_sha256"] != result["training_index_sha256"]
    assert result["entries"]["after_ensemble"]["metrics"]["n_not_assessed"] == 1
    assert result["entries"]["after_ensemble"]["metrics"]["n_missing_reference_label"] == 1
    for seed in transfer.SEEDS:
        assert (
            result["entries"][f"before_seed_{seed}"]["metrics"]
            == result["entries"][f"after_seed_{seed}"]["metrics"]
        )
    write_index(path, [{**test_rows[0], "study_id": "train"}])
    with pytest.raises(ValueError, match="overlaps"):
        transfer.evaluate_axial_transfer(path, output, tmp_path / "overlap", device="cpu")


def test_protocol_rejects_changed_settings():
    transfer._validate_protocol(protocol())
    for field, value in (
        ("max_epochs", 9),
        ("replay_ratio", 1),
        ("seeds", [42]),
        ("batch_size", True),
    ):
        changed = {**protocol(), field: value}
        with pytest.raises(ValueError, match="protocol mismatch"):
            transfer._validate_protocol(changed)


@pytest.mark.parametrize("n_targets", [1, 2])
def test_class_weight_scales_gradient_even_for_one_homogeneous_study(n_targets):
    logits = torch.tensor([[0.3, -0.2, 0.8]] * n_targets, requires_grad=True)
    labels = torch.full((n_targets,), 2, dtype=torch.long)
    ordinary = transfer._weighted_target_loss(logits, labels, torch.ones(3))
    ordinary_gradient = torch.autograd.grad(ordinary, logits)[0]
    heavy = transfer._weighted_target_loss(logits, labels, torch.tensor([1.0, 1.0, 3.5]))
    heavy_gradient = torch.autograd.grad(heavy, logits)[0]
    torch.testing.assert_close(heavy, ordinary * 3.5)
    torch.testing.assert_close(heavy_gradient, ordinary_gradient * 3.5)
    # Target averaging also keeps a duplicated same-class study's loss unchanged.
    one = transfer._weighted_target_loss(logits[:1], labels[:1], torch.ones(3))
    torch.testing.assert_close(ordinary, one)


def test_unlabeled_processed_level_keeps_prediction_but_is_excluded_from_metrics():
    record = transfer.ActivationRecord(
        "aisslab", "synthetic", "test", (None, 2), "ok", "", None, None
    )
    probabilities = np.asarray([[0.1, 0.8, 0.1], [0.05, 0.1, 0.85]])
    rows = transfer._prediction_rows(record, probabilities)
    assert rows[0]["status"] == "ok" and rows[0]["prediction"] == 1
    assert rows[0]["label"] is None and rows[0]["reference_status"] == "missing"
    assert rows[0]["p_moderate"] == 0.8
    assert rows[1]["reference_status"] == "available"
    metrics = evaluate_predictions(rows)
    assert metrics["n_requested"] == 2 and metrics["n_labeled"] == 1
    assert metrics["n_missing_reference_label"] == 1 and metrics["n_assessed"] == 1
    assert metrics["conditional"]["confusion_matrix"] == [[0, 0, 0], [0, 0, 0], [0, 0, 1]]


def test_validation_forward_includes_all_unlabeled_images_but_loss_does_not(monkeypatch):
    records = [
        transfer.ActivationRecord(
            "aisslab", "known", "validation", (1, None), "ok", "", None, None
        ),
        transfer.ActivationRecord(
            "aisslab", "unknown", "validation", (None, None), "ok", "", None, None
        ),
    ]
    seen = []

    def load(record, **_):
        seen.append(record.study_id)
        return None, None

    probabilities = np.asarray([[0.1, 0.7, 0.2], [0.9, 0.05, 0.05]])
    monkeypatch.setattr(transfer, "load_record", load)
    monkeypatch.setattr(transfer, "predict_record", lambda *_: probabilities)
    metrics, loss = transfer._assess(None, records)
    assert seen == ["known", "unknown"]
    assert metrics["n_labeled"] == 1 and metrics["n_missing_reference_label"] == 3
    assert metrics["n_assessed"] == 1
    assert loss == pytest.approx(-np.log(0.7))
