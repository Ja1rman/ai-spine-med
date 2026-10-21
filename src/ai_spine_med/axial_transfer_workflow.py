"""Command-line workflow for the external axial transfer experiment."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from .assets import sha256_file, write_json

PREPARED = Path("artifacts/aisslab-axial-v4")
SOURCES = Path("artifacts/axial-transfer-sources-v4")
FITTING = Path("artifacts/axial-transfer-fit-v4")
TEST = Path("artifacts/axial-transfer-test-v4")
RUN = Path("reports/runs/axial-transfer-v4")
PROTOCOL = Path("configs/axial-transfer-v4.json")
RSNA_TEST = Path("artifacts/axial-test-v4")


def prepare():
    from .aisslab import prepare_aisslab
    from .axial_test import prepare_rsna_axial_test_manifest
    from .axial_transfer_data import prepare_transfer_sources

    prepare_aisslab("data/external/aisslab-axial-v1/source.zip", PREPARED)
    prepare_rsna_axial_test_manifest("data/rsna", "artifacts/rsna-v1", RSNA_TEST)
    return prepare_transfer_sources(PREPARED, SOURCES, protocol=PROTOCOL)


def cache_fitting(device="auto"):
    from .axial_transfer_cache import build_transfer_cache

    return build_transfer_cache(SOURCES / "fitting-sources.json", FITTING, device=device)


def train(device="auto"):
    from .axial_transfer import train_axial_transfer

    return train_axial_transfer(FITTING / "index.json", PROTOCOL, RUN, device=device)


def prepare_evaluation_sources(*, workers=4):
    """Open/download TEST sources only after the full training selection is sealed."""
    from .axial_test import (
        build_source_inventory,
        download_rsna_axial_test,
        validate_rsna_axial_test_manifest,
        validate_transfer_selection,
    )

    test_rows, test_summary = validate_rsna_axial_test_manifest("data/rsna", RSNA_TEST)
    selection = RUN / "selection.json"
    if not selection.is_file():
        raise ValueError("Complete and seal all three training runs before TEST preparation")
    test_seal = {
        "schema_version": 1,
        "scope": "external_axial_v4_rsna_test",
        "frozen": True,
        "test_manifest_sha256": test_summary["manifest_sha256"],
        "protocol": {"path": str(PROTOCOL.resolve()), "sha256": sha256_file(PROTOCOL)},
        "training_selection": {"path": str(selection.resolve()), "sha256": sha256_file(selection)},
    }
    seal_path = RUN / "rsna-test-download-seal.json"
    if seal_path.exists() and json.loads(seal_path.read_text()) != test_seal:
        raise ValueError("The previously sealed TEST authorization changed")
    if not seal_path.exists():
        write_json(seal_path, test_seal)
    validate_transfer_selection("data/rsna", RSNA_TEST, seal_path)
    fixed = Path(test_summary["input_paths"]["studies.csv"]).parent
    fixed_summary = json.loads((fixed / "summary.json").read_text())
    labels_path = fixed / "labels.csv"
    labels_sha = sha256_file(labels_path)
    if labels_sha != fixed_summary["output_sha256"]["labels.csv"]:
        raise ValueError("The original frozen RSNA reference labels changed")
    labels, label_keys = {}, set()
    with labels_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "test":
                key = row["study_id"], row["level"]
                if key in label_keys or row["level"] not in ("L4-L5", "L5-S1"):
                    raise ValueError("Duplicate or unexpected RSNA TEST reference level")
                label_keys.add(key)
                value = None if row["label"] == "" else int(row["label"])
                if value is not None and value not in (0, 1, 2):
                    raise ValueError("Invalid RSNA TEST reference grade")
                labels.setdefault(row["study_id"], [None, None])[
                    ["L4-L5", "L5-S1"].index(row["level"])
                ] = value
    expected_keys = {(row["study_id"], level) for row in test_rows for level in ("L4-L5", "L5-S1")}
    if label_keys != expected_keys or sha256_file(labels_path) != labels_sha:
        raise ValueError("RSNA TEST labels do not match the complete fixed patient cohort")
    download_rsna_axial_test("data/rsna", RSNA_TEST, seal_path, workers=workers)
    source_path = SOURCES / "rsna-test-inventory.json"
    build_source_inventory("data/rsna", RSNA_TEST, seal_path, source_path)
    raw = json.loads(source_path.read_text())
    groups = defaultdict(list)
    for series in raw["source_series"]:
        groups[series["study_id"]].append(series)
    new_test_path = SOURCES / "new-test-sources.json"
    new_inventory = json.loads(new_test_path.read_text())
    if (
        sha256_file(new_test_path)
        != json.loads((SOURCES / "summary.json").read_text())["new_test_sources_sha256"]
    ):
        raise ValueError("The external held-out patient inventory changed")
    records = list(new_inventory["records"])
    for study, series in sorted(groups.items()):
        records.append(
            {
                "dataset": "rsna",
                "study_id": study,
                "split": "test",
                "labels": labels[study],
                "source_root": str(Path("data/rsna").resolve()),
                "source": {"series": series},
            }
        )
    output = SOURCES / "evaluation-sources.json"
    inventory = {
        "schema_version": 1,
        "scope": "external_axial_v4_rsna_test",
        "provenance": {
            **new_inventory["provenance"],
            "training_selection_sha256": sha256_file(selection),
            "new_test_sources_sha256": sha256_file(new_test_path),
            "rsna_test_sources_sha256": sha256_file(source_path),
            "rsna_test_manifest_sha256": test_summary["manifest_sha256"],
            "rsna_test_labels_sha256": labels_sha,
        },
        "records": records,
    }
    if output.exists() and json.loads(output.read_text()) != inventory:
        raise ValueError("The sealed evaluation source inventory changed")
    if not output.exists():
        write_json(output, inventory)
    return output


def evaluate(device="auto", *, workers=4):
    from .axial_transfer import evaluate_axial_transfer
    from .axial_transfer_cache import build_transfer_cache

    sources = prepare_evaluation_sources(workers=workers)
    build_transfer_cache(sources, TEST, device=device, selection=RUN / "selection.json")
    return evaluate_axial_transfer(TEST / "index.json", RUN, RUN / "evaluation", device=device)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Дообучение аксиальной модели на AISSLab")
    parser.add_argument("stage", choices=["prepare", "cache", "train", "evaluate", "report"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.stage == "prepare":
        result = prepare()
    elif args.stage == "cache":
        result = cache_fitting(args.device)
    elif args.stage == "train":
        result = train(args.device)
    elif args.stage == "evaluate":
        result = evaluate(args.device, workers=args.workers)
    else:
        from .axial_transfer_report import report_axial_transfer

        result = report_axial_transfer(RUN, PREPARED, SOURCES)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
