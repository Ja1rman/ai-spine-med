"""Run the separately sealed stages of the explicit axial localization experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROTOCOL = Path("configs/axial-localizer-v6.json")
RUN = Path("reports/runs/explicit-localization-v6")
FITTING = Path("artifacts/axial-localizer-v6")
FIT_FEATURES = Path("artifacts/axial-localized-features-v6")
TEST = Path("artifacts/axial-localizer-test-v6")
TEST_FEATURES = Path("artifacts/axial-localized-test-features-v6")


def run_stage(stage, *, device="cpu"):
    """Each component checks its own immutable data/model prerequisites."""
    if stage == "prepare":
        from .axial_localizer_data import prepare_localizer_data

        return prepare_localizer_data(out=FITTING, device=device)
    if stage == "localizer":
        from .axial_localizer import train_localizer

        return train_localizer(FITTING / "index.json", PROTOCOL, RUN / "localizer", device)
    if stage in {"features", "test-features"}:
        from .axial_localizer_extract import extract_localized_features

        test = stage == "test-features"
        return extract_localized_features(
            (TEST if test else FITTING) / "index.json",
            RUN / "localizer/selection.json",
            TEST_FEATURES if test else FIT_FEATURES,
            PROTOCOL,
            device=device,
            datasets=("rsna", "aisslab") if test else ("rsna",),
        )
    if stage == "severity":
        from .localized_stenosis import train_localized_stenosis

        return train_localized_stenosis(
            FIT_FEATURES / "features.npz",
            FIT_FEATURES / "targets.csv",
            RUN / "severity",
            json.loads(PROTOCOL.read_text()),
            device=device,
        )
    if stage == "prepare-test":
        from .axial_localizer_eval_data import prepare_evaluation_data

        return prepare_evaluation_data(
            RUN / "localizer/selection.json",
            RUN / "severity/selection.json",
            PROTOCOL,
            FIT_FEATURES / "targets.csv",
            fitting_index=FITTING / "index.json",
            out=TEST,
            device=device,
        )
    if stage == "evaluate":
        from .localized_stenosis import evaluate_localized_stenosis

        return evaluate_localized_stenosis(
            TEST_FEATURES / "features.npz",
            TEST_FEATURES / "targets.csv",
            RUN / "severity",
            device=device,
        )
    if stage == "score":
        from .axial_localizer_scoring import score_localizer_predictions

        return score_localizer_predictions(
            TEST / "index.json",
            TEST_FEATURES,
            RUN / "localizer/selection.json",
            RUN / "localization-test",
            references=TEST / "references.json",
        )
    if stage == "report":
        from .localized_stenosis_report import report_localized_stenosis

        return report_localized_stenosis()
    raise ValueError("Unknown localization workflow stage")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "prepare",
            "localizer",
            "features",
            "severity",
            "prepare-test",
            "test-features",
            "evaluate",
            "score",
            "report",
        ),
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args(argv)
    result = run_stage(args.stage, device=args.device)
    # Full model/target inventories stay private; print an aggregate completion.
    print(
        json.dumps({"stage": args.stage, "completed": True, "result_type": type(result).__name__})
    )


if __name__ == "__main__":
    main()
