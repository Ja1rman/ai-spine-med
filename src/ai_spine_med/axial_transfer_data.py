"""Assemble immutable fitting and test inventories for the axial transfer study."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .assets import sha256_file, write_json
from .training import LEVELS


def _load(path):
    return json.loads(Path(path).read_text())


def _publish(path: Path, value: dict):
    if path.exists() and _load(path) != value:
        raise ValueError("An immutable source inventory already exists with different contents")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        write_json(path, value)


def prepare_transfer_sources(
    prepared: str | Path,
    out: str | Path,
    *,
    protocol: str | Path = "configs/axial-transfer-v4.json",
    data_root: str | Path = "data/rsna",
    targets: str | Path = "artifacts/features-v1/target_manifest.csv",
    axial_cache: str | Path = "artifacts/axial-features-v3",
) -> dict:
    prepared, out, protocol = Path(prepared), Path(out), Path(protocol)
    data_root, targets, axial_cache = Path(data_root), Path(targets), Path(axial_cache)
    config = _load(protocol)
    new_rows = _load(prepared / "records.json")
    if isinstance(new_rows, dict):
        new_rows = new_rows["records"]
    with targets.open(newline="") as stream:
        old_targets = list(csv.DictReader(stream))
    feature_metadata = _load(targets.parent / "metadata.json")
    if sha256_file(targets) != feature_metadata["outputs_sha256"]["target_manifest.csv"]:
        raise ValueError("Original training target manifest checksum differs")
    labels = {}
    for target in old_targets:
        if target["split"] == "train" and target["status"] == "ok":
            group = labels.setdefault(target["study_id"], [None, None])
            group[LEVELS.index(target["level"])] = int(target["label"])
    metadata = _load(axial_cache / "metadata.json")
    old_records = []
    old_checksums, old_uids = set(), set()
    for item in metadata["study_records"]:
        path = axial_cache / item["record_path"]
        if not path.is_file():
            raise ValueError("Original axial source receipt is missing")
        if (
            sha256_file(path) != item["record_sha256"]
            or metadata["outputs_sha256"].get(item["record_path"]) != item["record_sha256"]
        ):
            raise ValueError("Original axial source receipt checksum differs")
        record = _load(path)
        for series in record.get("source", {}).get("series", []):
            old_checksums.update(x["sha256"] for x in series["files"])
        for series in record.get("qc", {}).get("series", []):
            old_uids.add(series["study_uid_sha256"])
        if record["split"] == "train" and record["status"] == "ok" and record["study_id"] in labels:
            old_records.append(
                {
                    "dataset": "rsna",
                    "study_id": record["study_id"],
                    "split": "train",
                    "labels": labels[record["study_id"]],
                    "source_root": str(data_root.resolve()),
                    "source": record["source"],
                }
            )
    new_sources = []
    exact_cross_dataset_duplicates = 0
    candidate_uid_overlaps = 0
    new_file_owners = {}
    for original in new_rows:
        row = {key: original[key] for key in ("dataset", "study_id", "split", "labels", "source")}
        row["source_root"] = str((prepared / "extracted").resolve())
        if row["dataset"] != "aisslab":
            raise ValueError("Unexpected external dataset")
        if row["source"].get("study_uid_sha256") in old_uids:
            candidate_uid_overlaps += 1
        for series in row["source"]["series"]:
            for item in series["files"]:
                checksum = item["sha256"]
                if checksum in old_checksums:
                    exact_cross_dataset_duplicates += 1
                owner = new_file_owners.setdefault(checksum, row["study_id"])
                if owner != row["study_id"]:
                    raise ValueError("Identical DICOM files occur in different external patients")
        new_sources.append(row)
    if exact_cross_dataset_duplicates:
        raise ValueError("Exact DICOM duplicates between new patients and original RSNA")
    if candidate_uid_overlaps:
        raise ValueError("Study identity overlap needs investigation before training")
    new_train = [
        x for x in new_sources if x["split"] == "train" and any(v is not None for v in x["labels"])
    ]
    ordered_old = sorted(old_records, key=lambda x: x["study_id"])
    replay_count = min(len(ordered_old), round(len(new_train) * config["replay_ratio"]))
    selected = np.random.default_rng(config["replay_seed"]).choice(
        len(ordered_old), size=replay_count, replace=False
    )
    replay = [ordered_old[int(i)] for i in sorted(selected)]
    provenance = {
        "protocol": {"path": str(protocol.resolve()), "sha256": sha256_file(protocol)},
        "aisslab_manifest": {
            "path": str((prepared / "manifest.json").resolve()),
            "sha256": sha256_file(prepared / "manifest.json"),
        },
        "aisslab_records": {
            "path": str((prepared / "records.json").resolve()),
            "sha256": sha256_file(prepared / "records.json"),
        },
        "aisslab_splits": {
            "path": str((prepared / "patient_splits.json").resolve()),
            "sha256": sha256_file(prepared / "patient_splits.json"),
        },
        "rsna_training_targets_sha256": sha256_file(targets),
        "rsna_axial_metadata_sha256": sha256_file(axial_cache / "metadata.json"),
        "duplicate_audit": {
            "exact_cross_dataset_dicom_duplicates": exact_cross_dataset_duplicates,
            "candidate_study_uid_overlaps": candidate_uid_overlaps,
            "scope": "Exact DICOM bytes and available de-identified study identities; does not rule out re-encoded or separately anonymized copies.",
        },
        "replay_studies": len(replay),
    }
    fitting = [x for x in new_sources if x["split"] in {"train", "validation"}] + replay
    held = [x for x in new_sources if x["split"] == "test"]
    _publish(
        out / "fitting-sources.json",
        {
            "schema_version": 1,
            "scope": "external_axial_v4_fitting",
            "provenance": provenance,
            "records": fitting,
        },
    )
    _publish(
        out / "new-test-sources.json",
        {
            "schema_version": 1,
            "scope": "external_axial_v4_new_test",
            "provenance": provenance,
            "records": held,
        },
    )
    summary = {
        "new_patients_by_split": dict(Counter(x["split"] for x in new_sources)),
        "replay_studies": len(replay),
        "fitting_source_studies": len(fitting),
        "new_test_studies": len(held),
        "new_classes_by_split": {
            split: dict(
                Counter(
                    str(v)
                    for x in new_sources
                    if x["split"] == split
                    for v in x["labels"]
                    if v is not None
                )
            )
            for split in ("train", "validation", "test")
        },
        "duplicate_audit": provenance["duplicate_audit"],
        "fitting_sources_sha256": sha256_file(out / "fitting-sources.json"),
        "new_test_sources_sha256": sha256_file(out / "new-test-sources.json"),
    }
    _publish(out / "summary.json", summary)
    return summary
