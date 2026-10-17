import json

import pytest

from ai_spine_med.reporting import draw_experiment


def test_report_refuses_to_compare_different_target_cohorts(tmp_path):
    baseline, training = tmp_path / "baseline", tmp_path / "training"
    for directory, targets_hash in ((baseline, "original"), (training, "different")):
        directory.mkdir()
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "feature_cache_sha256": {
                        "features.npz": "features",
                        "target_manifest.csv": targets_hash,
                    },
                    "feature_cache_metadata": {"cache_id": "same"},
                }
            )
        )
    with pytest.raises(ValueError, match="different feature caches"):
        draw_experiment(tmp_path / "unused", baseline, training, tmp_path / "images")
    assert not (tmp_path / "images").exists()
