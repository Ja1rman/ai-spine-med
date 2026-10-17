"""Aggregate-only figures and notes for the fixed v2 development experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .assets import write_json
from .comparison_report import _replace_block
from .reporting import COLORS, _save, _style, draw_confusions, plt

NAMES = {"linear": "Линейная", "mlp": "MLP 512→64→3", "coral": "Ordinal / CORAL"}


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _public_guard(value) -> None:
    forbidden = {
        "study_id",
        "series_id",
        "patient_id",
        "case_id",
        "roi_path",
        "dicom_path",
        "token",
    }
    if isinstance(value, dict):
        if forbidden.intersection(value):
            raise ValueError("Individual case fields cannot enter the public aggregate report")
        for item in value.values():
            _public_guard(item)
    elif isinstance(value, list):
        for item in value:
            _public_guard(item)


def _plot_scores(summary: dict, out: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for index, (name, candidate) in enumerate(summary["candidates"].items()):
        values = [
            f["ensemble"]["metrics"]["conditional"]["macro_f1"] for f in candidate["cv"]["folds"]
        ]
        mean = np.mean(values)
        axes[0].bar(index, mean, color=COLORS[index], alpha=0.8, width=0.55)
        axes[0].scatter(
            np.linspace(index - 0.08, index + 0.08, 3), values, color="#243447", zorder=3
        )
        axes[0].text(index, max(values) + 0.025, f"{mean:.3f}", ha="center")
        runs = candidate["validation"]["runs"]
        values = [run["metrics"]["conditional"]["macro_f1"] for run in runs]
        score = candidate["validation"]["ensemble"]["metrics"]["conditional"]["macro_f1"]
        axes[1].bar(index, score, color=COLORS[index], alpha=0.8, width=0.55)
        axes[1].scatter(
            np.linspace(index - 0.08, index + 0.08, 3), values, color="#243447", zorder=3
        )
        axes[1].text(index, max(values + [score]) + 0.025, f"{score:.3f}", ha="center")
    for ax in axes:
        ax.set(xticks=range(3), xticklabels=list(NAMES.values()), ylim=(0, 0.9), ylabel="Macro-F1")
        ax.grid(axis="y", alpha=0.15)
    axes[0].set_title("CV внутри train: среднее ансамбля\nТочки — три held-fold")
    axes[1].set_title("Development validation: ансамбль\nТочки — три отдельных seed")
    fig.text(
        0.5,
        -0.02,
        "Новые головы: encoder заморожен. Это development-оценки, не результаты нового теста.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(w_pad=3)
    _save(fig, out / "head_v2_comparison.png")


def _plot_intervals(bootstrap: dict, out: Path, filename: str, title: str, labels: dict) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for ax, metric, factor, xlabel in zip(
        axes,
        ("macro_f1", "severe_rate"),
        (1, 100),
        ("Разница macro-F1", "Разница доли найденных тяжёлых, п.п."),
    ):
        for i, (key, label) in enumerate(labels.items()):
            interval = bootstrap["contrasts"][key]["metrics"][metric]
            point, low, high = [factor * interval[k] for k in ("point", "lower_95", "upper_95")]
            ax.hlines(i, low, high, color=COLORS[i % 3], linewidth=3)
            ax.scatter(point, i, color="#243447", zorder=3)
            ax.text(
                high, i + 0.13, f" {point:+.3f}" if factor == 1 else f" {point:+.1f}", fontsize=9
            )
        ax.axvline(0, color="#607a8c", linestyle="--", linewidth=1)
        ax.set(
            yticks=range(len(labels)),
            yticklabels=list(labels.values()),
            xlabel=xlabel,
            ylim=(-0.5, len(labels) - 0.4),
        )
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.15)
    fig.suptitle(title, fontsize=13)
    fig.text(
        0.5,
        -0.025,
        f"{bootstrap['resamples']} парных bootstrap-выборок по исследованиям; 95% percentile-интервалы для фиксированных моделей.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(w_pad=3)
    _save(fig, out / filename)


def _results_text(summary: dict, legacy: dict, old_boot: dict, dev_boot: dict, review: dict) -> str:
    lines = [
        "Все 36 обучений завершены. Нелинейная голова MLP выбрана по CV **до** проверки прежней validation. Нового независимого теста в этом эксперименте нет.",
        "",
        "| Голова | CV: среднее трёх ансамблей | Validation: ансамбль | Найдено тяжёлых / всего | F1 умеренной категории |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, candidate in summary["candidates"].items():
        m = candidate["validation"]["ensemble"]["metrics"]
        lines.append(
            f"| {NAMES[name]} | {candidate['cv']['selection_macro_f1']:.3f} | {m['conditional']['macro_f1']:.3f} | {m['severe']['correctly_predicted_severe']} / {m['severe']['n_total']} | {m['conditional']['f1']['moderate']:.3f} |"
        )
    m = legacy["metrics"]
    lines += [
        "",
        f"Ансамбль прежних трёх голов на этой validation дал **{m['conditional']['macro_f1']:.3f}**, отдельные прежние головы — 0.676–0.682. Явного выигрыша от одного усреднения здесь нет. Старые головы выбирали эпоху по этой validation, новые — по CV, поэтому это историческое сравнение с разными правилами выбора эпохи.",
        "",
        "У нового MLP-ансамбля 29 из 42 тяжёлых уровней определены верно против 26 у нового линейного ансамбля. Тяжёлый → норма/лёгкий: 5 против 8; завышений нормы/лёгкого у обоих по 27 из 504. Один тяжёлый уровень не обработан и остаётся в общем знаменателе. На L5–S1 оба ансамбля не нашли ни одного из трёх тяжёлых случаев validation.",
        "",
        "Проверенный ordinal-вариант не предсказывает умеренную категорию. У обученных голов расстояние между границами 1.04–1.06: при выбранном argmax восстановленных вероятностей это делает умеренную категорию недостижимой. Это отрицательный результат данной параметризации и правила решения; его нельзя переносить на ordinal regression вообще. Правило после просмотра результатов не менял.",
    ]
    interval = dev_boot["contrasts"]["mlp_ensemble_minus_linear_ensemble"]["metrics"]["macro_f1"]
    old = old_boot["contrasts"]["frozen_mean_minus_original"]["metrics"]["macro_f1"]
    lines += [
        "",
        f"Разница validation macro-F1 MLP и нового линейного ансамбля — **{interval['point']:+.3f}**, 95% bootstrap-интервал **[{interval['lower_95']:+.3f}; {interval['upper_95']:+.3f}]**. Интервал включает ноль: MLP остаётся перспективным кандидатом, а не доказанно лучшим решением.",
        "",
        f"Дополнительный анализ старого теста v1 дал разницу macro-F1 новой головы и исходного SpineNet **{old['point']:+.3f}**, интервал **[{old['lower_95']:+.3f}; {old['upper_95']:+.3f}]**. Это среднее метрик трёх отдельных голов, не ансамбль вероятностей. Разброс по исследованиям заметно больше разброса по seed. Интервалы условны на этих весах и данных; неопределённость разметки и перенос на другую клинику они не измеряют.",
        "",
        f"Для врача подготовил локальную галерею из **{review['selected_cases']} случаев**: {review['by_split']['validation']} validation и {review['by_split']['train']} train, включая восемь правильных контролей и два отказа поиска уровня. Все отображённые исходные серии прошли проверку контрольных сумм и геометрии; 46 доступных ROI конечны и непусты. Я просмотрел все шесть контактных листов технически. Нумерацию, пригодность ROI и разметку врач ещё не подтверждал. Целевой отбор ошибок не позволяет оценивать по галерее частоту ошибок в популяции.",
    ]
    return "\n".join(lines)


def draw_improvements(
    experiment_dir: str | Path,
    old_bootstrap_dir: str | Path,
    dev_bootstrap_dir: str | Path,
    legacy_ensemble_dir: str | Path,
    review_dir: str | Path,
    out: str | Path = "docs/images",
    *,
    report_doc: str | Path = "docs/improvements.md",
) -> dict:
    root, old_root, dev_root, legacy_root, review_root, out = map(
        Path,
        (
            experiment_dir,
            old_bootstrap_dir,
            dev_bootstrap_dir,
            legacy_ensemble_dir,
            review_dir,
            out,
        ),
    )
    summary, old_boot, dev_boot, legacy, review = [
        _read(p / "summary.json") for p in (root, old_root, dev_root, legacy_root, review_root)
    ]
    if (
        summary["test_evaluated"] is not False
        or summary["validation_used_for_epoch_or_candidate_selection"] is not False
        or dev_boot["source_summary_sha256"] != _sha(root / "summary.json")
        or summary["selection_sha256"] != _sha(root / "selection.json")
        or review["test_accessed"] is not False
    ):
        raise ValueError("Inconsistent development/test boundary or selection provenance")
    candidates = {}
    for name, candidate in summary["candidates"].items():
        candidates[name] = {
            "cv_mean_fold_ensemble_macro_f1": candidate["cv"]["selection_macro_f1"],
            "cv_fold_ensemble_metrics": [
                f["ensemble"]["metrics"] for f in candidate["cv"]["folds"]
            ],
            "validation": {
                "runs": [
                    {
                        k: run[k]
                        for k in (
                            "seed",
                            "best_epoch",
                            "epochs_run",
                            "checkpoint_sha256",
                            "metrics",
                        )
                    }
                    for run in candidate["validation"]["runs"]
                ],
                "ensemble": candidate["validation"]["ensemble"],
            },
        }
    payload = {
        "scope": "v2 development only; historical v1 test uncertainty shown separately",
        "frozen_reference_tag": "practice-v1",
        "selected_candidate_by_train_cv": summary["selected_candidate"],
        "protocol_sha256": summary["configuration"]["protocol_sha256"],
        "selection_sha256": summary["selection_sha256"],
        "implementation_sha256": summary["configuration"]["implementation_sha256"],
        "candidates": candidates,
        "legacy_validation_ensemble": legacy,
        "v1_test_bootstrap": old_boot,
        "v2_development_bootstrap": dev_boot,
        "local_review_aggregate": {
            k: review[k]
            for k in (
                "selected_cases",
                "by_split",
                "by_sampling_reason",
                "by_level_and_reference",
                "technical",
                "missing_roi_reasons",
                "clinical_review",
                "representative_accuracy_sample",
            )
        },
        "new_independent_test_performed": False,
        "expert_review_performed": False,
    }
    _public_guard(payload)
    _plot_scores(summary, out)
    _plot_intervals(
        old_boot,
        out,
        "v1_bootstrap.png",
        "Исторический тест v1: 295 исследований",
        {
            "frozen_mean_minus_original": "Голова − исходный\nсреднее трёх seed",
            "fine_mean_minus_original": "Блок + голова − исходный\nсреднее трёх seed",
            "fine_mean_minus_frozen_mean": "Блок + голова − голова\nсредние трёх seed",
        },
    )
    _plot_intervals(
        dev_boot,
        out,
        "head_v2_bootstrap.png",
        "Новые варианты: development validation, 297 исследований",
        {
            "mlp_ensemble_minus_linear_ensemble": "MLP − новая линейная\nансамбли",
            "mlp_ensemble_minus_legacy_ensemble": "MLP − прежние головы\nансамбли, разные правила эпох",
        },
    )
    matrices = [
        (
            NAMES[name],
            candidate["validation"]["ensemble"]["metrics"]["conditional"]["confusion_matrix"],
        )
        for name, candidate in summary["candidates"].items()
    ]
    draw_confusions(matrices, out, filename="head_v2_confusion.png")
    write_json(out.parent / "improvement-results.json", payload)
    _replace_block(
        Path(report_doc),
        "IMPROVEMENTS_RESULTS",
        _results_text(summary, legacy, old_boot, dev_boot, review),
    )
    return payload
