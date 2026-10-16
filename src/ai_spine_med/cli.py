from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Исследовательский модуль SpineNet / RSNA")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Проверить окружение, не показывая токены")
    fetch = sub.add_parser("fetch-assets", help="Скачать исходники, веса и примеры авторов")
    fetch.add_argument("--cache", type=Path, default=Path(".cache"))
    fetch.add_argument("--examples-root", type=Path, default=Path("data/examples"))
    fetch.add_argument("--no-examples", action="store_true")
    smoke = sub.add_parser("smoke", help="Запустить исходный SpineNet и сохранить оценки")
    smoke.add_argument("--cache", type=Path, default=Path(".cache"))
    inputs = smoke.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--examples-root", type=Path)
    inputs.add_argument("--dicom-dir", type=Path)
    smoke.add_argument("--out", type=Path, required=True)
    smoke.add_argument("--device", default="cpu")
    smoke.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            from .baseline import environment

            print(json.dumps(environment(), indent=2))
        elif args.command == "fetch-assets":
            from .assets import fetch_assets

            summary = fetch_assets(
                args.cache, args.examples_root, with_examples=not args.no_examples
            )
            print(
                f"Ready: {len(summary['weights'])} weight files, {len(summary['examples'])} examples"
            )
        elif args.command == "smoke":
            from .baseline import run_smoke

            summary = run_smoke(
                cache=args.cache,
                out=args.out,
                examples_root=args.examples_root,
                dicom_dir=args.dicom_dir,
                device=args.device,
                threads=args.threads,
            )
            print(
                f"Completed: {summary['successes']}/{len(summary['cases'])} series; "
                f"{summary['target_levels_assessed']} target levels assessed"
            )
            return 0 if summary["successes"] == len(summary["cases"]) else 1
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0
