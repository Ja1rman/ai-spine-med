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
    activation = sub.add_parser(
        "cache-last-block", help="Сохранить вход последнего блока для train/validation"
    )
    activation.add_argument("--features", type=Path, required=True)
    activation.add_argument("--cache", type=Path, default=Path(".cache"))
    activation.add_argument("--out", type=Path, required=True)
    activation.add_argument("--device", default="auto")
    activation.add_argument("--batch-size", type=int, default=4)
    activation.add_argument("--limit", type=int)
    fine = sub.add_parser("train-last-block", help="Дообучить layer4.2 и голову с тремя seed")
    fine.add_argument("--features", type=Path, required=True)
    fine.add_argument("--activations", type=Path, required=True)
    fine.add_argument("--heads", type=Path, required=True)
    fine.add_argument("--cache", type=Path, default=Path(".cache"))
    fine.add_argument("--out", type=Path, required=True)
    fine.add_argument("--device", default="cpu")
    fine.add_argument("--threads", type=int, default=4)
    fine.add_argument("--seeds", type=int, nargs="+", default=[42, 2024, 3407])
    fine.add_argument("--max-epochs", type=int, default=10)
    fine.add_argument("--patience", type=int, default=3)
    fine.add_argument("--batch-size", type=int, default=16)
    fine.add_argument("--protocol", type=Path, default=Path("configs/finetune-v1.json"))
    seal = sub.add_parser(
        "freeze-selection", help="Зафиксировать протокол и шесть checkpoint до открытия теста"
    )
    seal.add_argument("--protocol", type=Path, default=Path("configs/finetune-v1.json"))
    seal.add_argument("--manifest-dir", type=Path, default=Path("artifacts/rsna-v1"))
    seal.add_argument("--features", type=Path, required=True)
    seal.add_argument("--heads", type=Path, required=True)
    seal.add_argument("--fine-tuned", type=Path, required=True)
    seal.add_argument("--out", type=Path, required=True)
    test_download = sub.add_parser(
        "download-test", help="Скачать отложенный тест после фиксации моделей"
    )
    test_download.add_argument("--selection", type=Path, required=True)
    test_download.add_argument("--data-root", type=Path, default=Path("data/rsna"))
    test_download.add_argument("--manifest-dir", type=Path, default=Path("artifacts/rsna-v1"))
    test_download.add_argument("--workers", type=int, default=4)
    test_extract = sub.add_parser(
        "extract-test", help="Обработать только test неизменённым SpineNet"
    )
    test_extract.add_argument("--selection", type=Path, required=True)
    test_extract.add_argument("--data-root", type=Path, default=Path("data/rsna"))
    test_extract.add_argument("--manifest-dir", type=Path, default=Path("artifacts/rsna-v1"))
    test_extract.add_argument("--cache", type=Path, default=Path(".cache"))
    test_extract.add_argument("--out", type=Path, required=True)
    test_extract.add_argument("--device", default="auto")
    test_extract.add_argument("--threads", type=int, default=4)
    test_extract.add_argument("--limit", type=int)
    test_extract.add_argument("--retry-failed", action="store_true")
    test_evaluate = sub.add_parser(
        "evaluate-test", help="Сравнить все семь фиксированных моделей на test"
    )
    test_evaluate.add_argument("--features", type=Path, required=True)
    test_evaluate.add_argument("--selection", type=Path, required=True)
    test_evaluate.add_argument("--heads", type=Path, required=True)
    test_evaluate.add_argument("--fine-tuned", type=Path, required=True)
    test_evaluate.add_argument("--activations", type=Path, required=True)
    test_evaluate.add_argument("--cache", type=Path, default=Path(".cache"))
    test_evaluate.add_argument("--out", type=Path, required=True)
    test_evaluate.add_argument("--device", default="auto")
    test_evaluate.add_argument("--batch-size", type=int, default=16)
    test_evaluate.add_argument("--warmup", type=int, default=8)
    test_evaluate.add_argument("--draws", type=int, default=32)
    test_evaluate.add_argument("--skip-benchmark", action="store_true")
    final_report = sub.add_parser(
        "report-final", help="Построить графики итогового теста и дообучения"
    )
    final_report.add_argument("--heads", type=Path, required=True)
    final_report.add_argument("--fine-tuned", type=Path, required=True)
    final_report.add_argument("--test-results", type=Path, required=True)
    final_report.add_argument("--out", type=Path, default=Path("docs/images"))
    final_report.add_argument("--update-readme", type=Path)
    final_report.add_argument("--protocol-doc", type=Path)
    final_report.add_argument("--manifest-dir", type=Path, default=Path("artifacts/rsna-v1"))
    inference = sub.add_parser("predict", help="Оценить два уровня одной DICOM-серии")
    inference.add_argument("--dicom-dir", type=Path, required=True)
    inference.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    inference.add_argument("--cache", type=Path, default=Path(".cache"))
    inference.add_argument("--out", type=Path, required=True)
    inference.add_argument("--device", default="auto")
    inference.add_argument("--threads", type=int, default=4)
    experiments = sub.add_parser(
        "head-experiments", help="Сравнить головы на трёх CV-разбиениях train"
    )
    experiments.add_argument("--features", type=Path, default=Path("artifacts/features-v1"))
    experiments.add_argument("--out", type=Path, required=True)
    experiments.add_argument("--protocol", type=Path, default=Path("configs/head-v2.json"))
    experiments.add_argument("--threads", type=int, default=4)
    review = sub.add_parser(
        "error-review", help="Подготовить локальную галерею для проверки с врачом"
    )
    review.add_argument("--features", type=Path, default=Path("artifacts/features-v1"))
    review.add_argument("--heads", type=Path, default=Path("reports/runs/head-v1"))
    review.add_argument("--data-root", type=Path, default=Path("data/rsna"))
    review.add_argument("--out", type=Path, required=True)
    review.add_argument("--count", type=int, default=48)
    review.add_argument("--seed", type=int, default=2026)
    review.add_argument(
        "--experiments", type=Path, help="Прикрепить сравнение с готовыми v2-головами"
    )
    bootstrap = sub.add_parser(
        "bootstrap-v1", help="Оценить неопределённость фиксированного теста v1"
    )
    bootstrap.add_argument("--test-results", type=Path, default=Path("reports/runs/test-v1"))
    bootstrap.add_argument("--out", type=Path, required=True)
    bootstrap.add_argument("--resamples", type=int, default=2000)
    bootstrap.add_argument("--seed", type=int, default=2026)
    legacy = sub.add_parser(
        "ensemble-validation", help="Проверить ансамбль прежних голов на validation"
    )
    legacy.add_argument("--heads", type=Path, default=Path("reports/runs/head-v1"))
    legacy.add_argument("--out", type=Path, required=True)
    dev_bootstrap = sub.add_parser(
        "bootstrap-development", help="Bootstrap новых development-оценок"
    )
    dev_bootstrap.add_argument("--experiments", type=Path, default=Path("reports/runs/head-v2"))
    dev_bootstrap.add_argument(
        "--legacy-ensemble", type=Path, default=Path("reports/runs/ensemble-v1-validation")
    )
    dev_bootstrap.add_argument("--out", type=Path, required=True)
    dev_bootstrap.add_argument("--resamples", type=int, default=2000)
    dev_bootstrap.add_argument("--seed", type=int, default=2026)
    improvements = sub.add_parser(
        "report-improvements", help="Построить агрегированный отчёт новых экспериментов"
    )
    improvements.add_argument("--experiments", type=Path, default=Path("reports/runs/head-v2"))
    improvements.add_argument(
        "--old-bootstrap", type=Path, default=Path("reports/runs/bootstrap-v1-verified")
    )
    improvements.add_argument(
        "--dev-bootstrap", type=Path, default=Path("reports/runs/bootstrap-v2-validation")
    )
    improvements.add_argument(
        "--legacy-ensemble", type=Path, default=Path("reports/runs/ensemble-v1-validation")
    )
    improvements.add_argument("--review", type=Path, default=Path("reports/runs/error-review-v2"))
    improvements.add_argument("--out", type=Path, default=Path("docs/images"))
    improvements.add_argument("--report-doc", type=Path, default=Path("docs/improvements.md"))
    axial_prepare = sub.add_parser(
        "prepare-axial", help="Подготовить Axial T2 для прежних train/validation пациентов"
    )
    axial_prepare.add_argument("--data-root", type=Path, default=Path("data/rsna"))
    axial_prepare.add_argument("--manifest-dir", type=Path, default=Path("artifacts/rsna-v1"))
    axial_prepare.add_argument("--out", type=Path, default=Path("artifacts/axial-v3"))
    axial_download = sub.add_parser(
        "download-axial", help="Скачать выбранные train/validation Axial T2"
    )
    axial_download.add_argument("--data-root", type=Path, default=Path("data/rsna"))
    axial_download.add_argument("--manifest", type=Path, default=Path("artifacts/axial-v3"))
    axial_download.add_argument("--workers", type=int, default=8)
    axial_extract = sub.add_parser(
        "extract-axial", help="Извлечь признаки всех аксиальных срезов без координат разметки"
    )
    axial_extract.add_argument("--data-root", type=Path, default=Path("data/rsna"))
    axial_extract.add_argument("--manifest", type=Path, default=Path("artifacts/axial-v3"))
    axial_extract.add_argument("--out", type=Path, default=Path("artifacts/axial-features-v3"))
    axial_extract.add_argument(
        "--weights", type=Path, default=Path(".cache/axial-v3/resnet18-f37072fd.pth")
    )
    axial_extract.add_argument("--device", default="auto")
    axial_extract.add_argument("--threads", type=int, default=4)
    axial_extract.add_argument("--limit", type=int)
    multimodal = sub.add_parser(
        "multimodal-experiment",
        help="Сравнить sagittal, axial и fusion на прежнем train/validation",
    )
    multimodal.add_argument("--sagittal", type=Path, default=Path("artifacts/features-v1"))
    multimodal.add_argument("--axial", type=Path, default=Path("artifacts/axial-features-v3"))
    multimodal.add_argument("--protocol", type=Path, default=Path("configs/multimodal-v3.json"))
    multimodal.add_argument("--out", type=Path, default=Path("reports/runs/multimodal-v3"))
    multimodal.add_argument("--threads", type=int, default=4)
    multimodal_report = sub.add_parser(
        "report-multimodal", help="Сохранить графики и агрегированный отчёт сравнения МРТ"
    )
    multimodal_report.add_argument(
        "--experiment", type=Path, default=Path("reports/runs/multimodal-v3")
    )
    multimodal_report.add_argument("--document", type=Path, default=Path("docs/multimodal.md"))
    multimodal_report.add_argument("--figures", type=Path, default=Path("docs/images"))
    multi_review = sub.add_parser(
        "multimodal-review", help="Собрать приватную галерею ошибок и аксиального внимания"
    )
    multi_review.add_argument("--experiment", type=Path, default=Path("reports/runs/multimodal-v3"))
    multi_review.add_argument("--sagittal", type=Path, default=Path("artifacts/features-v1"))
    multi_review.add_argument("--axial", type=Path, default=Path("artifacts/axial-features-v3"))
    multi_review.add_argument("--data-root", type=Path, default=Path("data/rsna"))
    multi_review.add_argument("--out", type=Path, default=Path("reports/runs/multimodal-review-v3"))
    multi_review.add_argument("--count", type=int, default=24)
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
        elif args.command == "cache-last-block":
            from .finetuning import cache_last_block_activations

            result = cache_last_block_activations(
                args.features,
                args.cache,
                args.out,
                device=args.device,
                batch_size=args.batch_size,
                limit=args.limit,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "train-last-block":
            from .finetuning import train_last_block

            result = train_last_block(
                args.features,
                args.activations,
                args.heads,
                args.cache,
                args.out,
                device=args.device,
                threads=args.threads,
                seeds=tuple(args.seeds),
                max_epochs=args.max_epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                protocol_path=args.protocol,
            )
            print(
                f"Validation macro-F1={result['validation_macro_f1_mean']:.4f} ± {result['validation_macro_f1_std']:.4f}"
            )
        elif args.command == "freeze-selection":
            from .evaluation import freeze_selection

            freeze_selection(
                args.protocol,
                args.manifest_dir,
                args.features,
                args.heads,
                args.fine_tuned,
                args.out,
            )
            print(f"Selected model hashes fixed in {args.out}")
        elif args.command == "download-test":
            from .rsna_bulk import download_test

            result = download_test(args.data_root, args.manifest_dir, args.selection, args.workers)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "extract-test":
            from .heldout_features import extract_heldout_features

            result = extract_heldout_features(
                args.data_root,
                args.manifest_dir,
                args.cache,
                args.out,
                args.selection,
                device=args.device,
                threads=args.threads,
                limit=args.limit,
                retry_failed=args.retry_failed,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "evaluate-test":
            from .evaluation import evaluate_test

            result = evaluate_test(
                args.features,
                args.selection,
                args.heads,
                args.fine_tuned,
                args.activations,
                args.out,
                cache_dir=args.cache,
                device=args.device,
                batch_size=args.batch_size,
                benchmark=not args.skip_benchmark,
                warmup=args.warmup,
                draws=args.draws,
            )
            print(json.dumps(result["variants"], ensure_ascii=False, indent=2))
        elif args.command == "report-final":
            from .comparison_report import draw_final_comparison

            draw_final_comparison(
                args.heads,
                args.fine_tuned,
                args.test_results,
                args.out,
                readme=args.update_readme,
                protocol_doc=args.protocol_doc,
                manifest_dir=args.manifest_dir,
            )
            print(f"Final figures saved to {args.out}")
        elif args.command == "predict":
            from .inference import run_inference

            result = run_inference(
                args.dicom_dir,
                args.out,
                checkpoint=args.checkpoint,
                cache_dir=args.cache,
                device=args.device,
                threads=args.threads,
            )
            print(f"Saved research predictions to {args.out}; status={result['status']}")
            return 1 if result["status"] == "failed" else 0
        elif args.command == "head-experiments":
            from .head_experiments import run_head_experiments

            result = run_head_experiments(
                args.features, args.out, protocol_path=args.protocol, threads=args.threads
            )
            print(
                f"CV-selected candidate: {result['selected_candidate']}; results saved to {args.out}"
            )
        elif args.command == "error-review":
            from .error_review import attach_development_comparison, generate_error_review

            generate_error_review(
                args.features,
                args.heads,
                args.data_root,
                args.out,
                count=args.count,
                seed=args.seed,
            )
            if args.experiments:
                attach_development_comparison(args.out, args.experiments)
            print(f"Local physician review saved to {args.out}; clinical review is pending")
        elif args.command == "bootstrap-v1":
            from .uncertainty import bootstrap_v1_test

            bootstrap_v1_test(args.test_results, args.out, resamples=args.resamples, seed=args.seed)
            print(f"Fixed-model study-bootstrap intervals saved to {args.out}")
        elif args.command == "ensemble-validation":
            from .uncertainty import legacy_validation_ensemble

            legacy_validation_ensemble(args.heads, args.out)
            print(f"Legacy validation ensemble saved to {args.out}")
        elif args.command == "bootstrap-development":
            from .uncertainty import bootstrap_v2_validation

            bootstrap_v2_validation(
                args.experiments,
                args.legacy_ensemble,
                args.out,
                resamples=args.resamples,
                seed=args.seed,
            )
            print(f"Development bootstrap intervals saved to {args.out}")
        elif args.command == "report-improvements":
            from .improvement_report import draw_improvements

            draw_improvements(
                args.experiments,
                args.old_bootstrap,
                args.dev_bootstrap,
                args.legacy_ensemble,
                args.review,
                args.out,
                report_doc=args.report_doc,
            )
            print(f"Aggregate development report saved to {args.report_doc}")
        elif args.command == "prepare-axial":
            from .axial_data import prepare_axial_manifest

            result = prepare_axial_manifest(args.data_root, args.manifest_dir, args.out)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "download-axial":
            from .axial_data import download_axial

            result = download_axial(args.data_root, args.manifest, workers=args.workers)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "extract-axial":
            from .axial_features import extract_axial_features

            result = extract_axial_features(
                args.data_root,
                args.manifest,
                args.out,
                args.weights,
                device=args.device,
                limit=args.limit,
                threads=args.threads,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "multimodal-experiment":
            from .multimodal import run_multimodal

            result = run_multimodal(
                args.sagittal, args.axial, args.protocol, args.out, threads=args.threads
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "report-multimodal":
            from .multimodal_report import draw_multimodal_report

            draw_multimodal_report(args.experiment, document=args.document, figures=args.figures)
            print(f"Aggregate MRI comparison report saved to {args.document}")
        elif args.command == "multimodal-review":
            from .multimodal_review import generate_multimodal_review

            generate_multimodal_review(
                args.experiment,
                args.sagittal,
                args.axial,
                args.data_root,
                args.out,
                count=args.count,
            )
            print(f"Private MRI review saved to {args.out}; physician review pending")
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0
