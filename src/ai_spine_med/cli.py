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
    download = sub.add_parser("download-cohort", help="Скачать только train/validation T2/STIR")
    download.add_argument("--data-root", type=Path, default=Path("data/rsna"))
    download.add_argument("--manifest-dir", type=Path, default=Path("artifacts/rsna-v1"))
    download.add_argument("--workers", type=int, default=4)
    features = sub.add_parser("extract-features", help="Извлечь области и замороженные признаки")
    features.add_argument("--data-root", type=Path, default=Path("data/rsna"))
    features.add_argument("--manifest-dir", type=Path, default=Path("artifacts/rsna-v1"))
    features.add_argument("--cache", type=Path, default=Path(".cache"))
    features.add_argument("--out", type=Path, required=True)
    features.add_argument("--device", default="auto")
    features.add_argument("--threads", type=int, default=4)
    features.add_argument("--limit", type=int)
    features.add_argument("--retry-failed", action="store_true")
    features.add_argument("--worker-index", type=int, default=0)
    features.add_argument("--num-workers", type=int, default=1)
    for command, help_text in (
        ("evaluate-baseline", "Оценить исходный SpineNet на валидации"),
        ("train-head", "Обучить новую голову на неизменных 512 признаках"),
    ):
        task = sub.add_parser(command, help=help_text)
        task.add_argument("--features", type=Path, required=True)
        task.add_argument("--out", type=Path, required=True)
        if command == "train-head":
            task.add_argument("--seeds", type=int, nargs="+", default=[42, 2024, 3407])
            task.add_argument("--max-epochs", type=int, default=20)
            task.add_argument("--patience", type=int, default=5)
            task.add_argument("--threads", type=int, default=4)
    figures = sub.add_parser("report-figures", help="Построить графики без изображений пациентов")
    figures.add_argument("--manifest-dir", type=Path, default=Path("artifacts/rsna-v1"))
    figures.add_argument("--baseline", type=Path, required=True)
    figures.add_argument("--training", type=Path, required=True)
    figures.add_argument("--out", type=Path, default=Path("docs/images"))
    figures.add_argument("--update-readme", type=Path)
    qc = sub.add_parser("roi-qc", help="Сохранить локальные изображения областей из train")
    qc.add_argument("--features", type=Path, required=True)
    qc.add_argument("--data-root", type=Path, default=Path("data/rsna"))
    qc.add_argument("--out", type=Path, required=True)
    qc.add_argument("--count", type=int, default=24)
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
        elif args.command == "download-cohort":
            from .rsna_bulk import download_train_validation

            summary = download_train_validation(args.data_root, args.manifest_dir, args.workers)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        elif args.command == "extract-features":
            from .features import extract_features

            summary = extract_features(
                args.data_root,
                args.manifest_dir,
                args.cache,
                args.out,
                device=args.device,
                threads=args.threads,
                limit=args.limit,
                retry_failed=args.retry_failed,
                worker_index=args.worker_index,
                num_workers=args.num_workers,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        elif args.command == "evaluate-baseline":
            from .training import evaluate_baseline

            result = evaluate_baseline(args.features, args.out)
            print(
                f"Validation macro-F1={result['conditional']['macro_f1']:.4f}; "
                f"assessed={result['n_assessed']}/{result['n_labeled']}"
            )
        elif args.command == "train-head":
            from .training import train_head

            result = train_head(
                args.features,
                args.out,
                seeds=tuple(args.seeds),
                max_epochs=args.max_epochs,
                patience=args.patience,
                threads=args.threads,
            )
            print(
                f"Validation macro-F1={result['validation_macro_f1_mean']:.4f} "
                f"± {result['validation_macro_f1_std']:.4f} across {len(result['runs'])} seeds"
            )
        elif args.command == "report-figures":
            from .reporting import draw_experiment, update_readme_results

            result = draw_experiment(args.manifest_dir, args.baseline, args.training, args.out)
            if args.update_readme:
                update_readme_results(args.update_readme, result)
            print(f"Figures saved to {args.out}")
        elif args.command == "roi-qc":
            from .qc import generate_qc

            generate_qc(args.features, args.data_root, args.out, count=args.count)
            print(f"Local training ROI images saved to {args.out}; do not redistribute RSNA images")
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0
