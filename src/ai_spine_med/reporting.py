"""Figures for the public report; never copy patient pixels or identifiers."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

COLORS = ("#4b89a6", "#e4a44c", "#ca6473")
CLASS_LABELS = ("Норма / лёгкий", "Умеренный", "Тяжёлый")


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "#fafbfc",
            "axes.facecolor": "#fafbfc",
            "savefig.facecolor": "#fafbfc",
            "text.color": "#243447",
            "axes.labelcolor": "#243447",
        }
    )


def _save(fig: plt.Figure, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=170, bbox_inches="tight")
    plt.close(fig)


def draw_pipeline(out: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.set(xlim=(0, 13), ylim=(0, 4.5))
    ax.axis("off")
    boxes = [
        (0.15, 2.0, 2.0, "Сагиттальная\nT2/STIR-серия", "#e4edf4"),
        (2.65, 2.0, 2.3, "SpineNet\nПозвонки и уровни", "#e4edf4"),
        (5.45, 2.0, 2.0, "Области дисков\nL4–L5 и L5–S1", "#e4edf4"),
        (7.95, 2.0, 2.1, "Замороженная сеть\n512 признаков", "#e4edf4"),
        (10.55, 2.0, 2.3, "Новая голова\nLinear(512, 3)", "#e5f1e9"),
    ]
    for x, y, width, label, color in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                1.0,
                boxstyle="round,pad=0.04",
                facecolor=color,
                edgecolor="#a6bac8",
                linewidth=1.2,
            )
        )
        ax.text(x + width / 2, y + 0.5, label, ha="center", va="center", fontsize=10)
    for left, right in zip(boxes, boxes[1:]):
        ax.add_patch(
            FancyArrowPatch(
                (left[0] + left[2] + 0.08, 2.5),
                (right[0] - 0.08, 2.5),
                arrowstyle="-|>",
                mutation_scale=16,
                color="#607a8c",
            )
        )
    ax.text(6.45, 1.62, "9 × 112 × 224", ha="center", color="#607a8c", fontsize=10)
    ax.text(9.0, 1.62, "веса не меняются", ha="center", color="#607a8c", fontsize=10)
    ax.text(11.7, 1.62, "обучаю на RSNA", ha="center", color="#43805b", fontsize=10)
    ax.text(0.15, 4.0, "Что меняется в эксперименте", fontsize=18, weight="bold")
    ax.text(
        0.15,
        3.55,
        "Локализация и основная сеть исходные. Обучается только последний слой.",
        fontsize=11,
        color="#607a8c",
    )
    ax.text(
        0.15,
        0.55,
        "Нет нужной области → not_assessed     •     Выход → норма/лёгкий, умеренный, тяжёлый",
        fontsize=12,
    )
    _save(fig, out / "pipeline.png")


def draw_cohort(manifest_dir: Path, out: Path) -> None:
    _style()
    summary = json.loads((manifest_dir / "summary.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))
    class_names = ("normal_mild", "moderate", "severe")
    splits = ("train", "validation", "test")
    labels = tuple(
        f"{name}\n{summary['splits'][split]['studies']:,} исследований".replace(",", " ")
        for name, split in zip(("Train", "Validation", "Test"), splits)
    )
    totals = np.array([summary["splits"][s]["labels"] for s in splits])
    bottom = np.zeros(3)
    for name, label, color in zip(class_names, CLASS_LABELS, COLORS):
        values = np.array([summary["splits"][s]["class_counts"][name] for s in splits])
        heights = values / totals * 100
        axes[0].bar(labels, heights, bottom=bottom, label=label, color=color, width=0.6)
        for i, (value, height, base) in enumerate(zip(values, heights, bottom)):
            axes[0].text(i, base + height / 2, str(value), ha="center", va="center", fontsize=9)
        bottom += heights
    axes[0].set(
        title="Классы в каждой выборке",
        ylabel="Доля меток, % (внутри столбцов — число)",
        ylim=(0, 104),
    )
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.07), ncol=3, fontsize=8)
    levels = ("L4-L5", "L5-S1")
    x = np.arange(2)
    for index, (name, label, color) in enumerate(zip(class_names, CLASS_LABELS, COLORS)):
        values = [summary["splits"]["train"]["by_level"][level][name] for level in levels]
        axes[1].bar(x + (index - 1) * 0.25, values, width=0.24, color=color, label=label)
        for pos, value in zip(x + (index - 1) * 0.25, values):
            axes[1].text(pos, value + 17, str(value), ha="center", fontsize=9)
    axes[1].set(
        xticks=x, xticklabels=("L4–L5", "L5–S1"), ylim=(0, 1500), title="Обучающие метки по уровням"
    )
    fig.text(
        0.5,
        -0.015,
        "Test оставлен для следующего этапа: его изображения и результаты здесь не используются.",
        ha="center",
        fontsize=10,
        color="#607a8c",
    )
    fig.tight_layout(w_pad=3)
    _save(fig, out / "cohort.png")


def draw_confusions(matrices: list[tuple[str, list[list[int]]]], out: Path) -> None:
    _style()
    columns = min(len(matrices), 2)
    rows = (len(matrices) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4.5 * rows), squeeze=False)
    maximum = max(max(max(row) for row in matrix) for _, matrix in matrices)
    short = ("Норма /\nлёгкий", "Умеренный", "Тяжёлый")
    for ax, (title, matrix) in zip(axes.flat, matrices):
        values = np.asarray(matrix)
        ax.imshow(values, cmap="Blues", vmin=0, vmax=maximum)
        for i in range(3):
            for j in range(3):
                ax.text(
                    j,
                    i,
                    str(values[i, j]),
                    ha="center",
                    va="center",
                    fontsize=14,
                    color="white" if values[i, j] > maximum * 0.55 else "#243447",
                )
        ax.set(
            xticks=range(3),
            xticklabels=short,
            yticks=range(3),
            yticklabels=short,
            xlabel="Предсказание",
            ylabel="Метка RSNA",
            title=title,
        )
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in list(axes.flat)[len(matrices) :]:
        ax.axis("off")
    fig.tight_layout(w_pad=2)
    _save(fig, out / "confusion.png")


def draw_experiment(
    manifest_dir: str | Path,
    baseline_dir: str | Path,
    training_dir: str | Path,
    out: str | Path = "docs/images",
) -> dict:
    """Export aggregate-only figures and JSON; do not read prediction identifiers."""
    manifest_dir, baseline_dir, training_dir, out = map(
        Path, (manifest_dir, baseline_dir, training_dir, out)
    )
    baseline_manifest = json.loads((baseline_dir / "manifest.json").read_text())
    training_manifest = json.loads((training_dir / "manifest.json").read_text())
    for name in ("features.npz", "target_manifest.csv"):
        expected = baseline_manifest.get("feature_cache_sha256", {}).get(name)
        actual = training_manifest.get("feature_cache_sha256", {}).get(name)
        if not expected or expected != actual:
            raise ValueError("Cannot compare models evaluated on different feature caches")
    if (
        baseline_manifest["feature_cache_metadata"]["cache_id"]
        != training_manifest["feature_cache_metadata"]["cache_id"]
    ):
        raise ValueError("Baseline and trained head use different extraction provenance")
    baseline = json.loads((baseline_dir / "metrics.json").read_text())
    training = json.loads((training_dir / "summary.json").read_text())
    runs = training["runs"]
    if not runs:
        raise ValueError("No completed training runs")
    draw_pipeline(out)
    draw_cohort(manifest_dir, out)
    _style()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for run in runs:
        history = json.loads(
            (training_dir / "runs" / f"seed-{run['seed']}" / "history.json").read_text()
        )
        epochs = [row["epoch"] for row in history]
        axes[0].plot(
            epochs,
            [row["training_loss"] for row in history],
            marker=".",
            label=f"seed {run['seed']}",
        )
        axes[1].plot(
            epochs,
            [row["validation_macro_f1"] for row in history],
            marker=".",
            label=f"seed {run['seed']}",
        )
    axes[0].set(title="Обучение новой головы", xlabel="Эпоха", ylabel="Weighted cross-entropy")
    axes[1].axhline(
        baseline["conditional"]["macro_f1"], color="#607a8c", ls="--", label="Исходный SpineNet"
    )
    axes[1].set(title="Macro-F1 на валидации", xlabel="Эпоха", ylabel="Macro-F1", ylim=(0, 1))
    for ax in axes:
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.15)
    fig.tight_layout(w_pad=3)
    _save(fig, out / "training.png")

    # All seeds are shown; the plot does not promote the best seed to primary.
    draw_confusions(
        [("Исходный SpineNet", baseline["conditional"]["confusion_matrix"])]
        + [
            (
                f"Новая голова · seed {run['seed']}",
                run["metrics"]["conditional"]["confusion_matrix"],
            )
            for run in runs
        ],
        out,
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    labels = ("L4–L5", "L5–S1")
    level_metrics = [baseline["by_level"][level] for level in ("L4-L5", "L5-S1")]
    assessed = np.array([m["n_assessed"] for m in level_metrics])
    not_assessed = np.array([m["n_not_assessed"] for m in level_metrics])
    axes[0].bar(labels, assessed, color=COLORS[0], label="Оценено")
    axes[0].bar(labels, not_assessed, bottom=assessed, color="#d9dfe4", label="Не оценено")
    for i, metric in enumerate(level_metrics):
        axes[0].text(
            i,
            assessed[i] / 2,
            f"{assessed[i]} / {metric['n_labeled']}\n{metric['coverage']:.1%}",
            ha="center",
            va="center",
        )
        if not_assessed[i]:
            axes[0].text(
                i,
                assessed[i] + not_assessed[i] / 2,
                str(not_assessed[i]),
                ha="center",
                va="center",
                fontsize=9,
            )
    axes[0].set(title="Покрытие уровней на валидации", ylabel="Число целевых уровней")
    axes[0].legend(fontsize=9)
    entries = [("SpineNet", baseline)] + [(f"seed {run['seed']}", run["metrics"]) for run in runs]
    x = np.arange(len(entries))
    width = 0.35
    axes[1].bar(
        x - width / 2,
        [m["conditional"]["macro_f1"] for _, m in entries],
        width,
        label="Macro-F1",
        color=COLORS[0],
    )
    axes[1].bar(
        x + width / 2,
        [m["severe"]["detection_rate_all_requested"] for _, m in entries],
        width,
        label="Найдено тяжёлых / все тяжёлые",
        color=COLORS[2],
    )
    axes[1].set(
        xticks=x,
        xticklabels=[name for name, _ in entries],
        ylim=(0, 1.12),
        title="Один набор областей для всех моделей",
    )
    axes[1].legend(fontsize=8, loc="upper center")
    for index, (_, metric) in enumerate(entries):
        for offset, value in (
            (-width / 2, metric["conditional"]["macro_f1"]),
            (width / 2, metric["severe"]["detection_rate_all_requested"]),
        ):
            axes[1].text(index + offset, value + 0.02, f"{value:.3f}", ha="center", fontsize=8)
    fig.tight_layout(w_pad=3)
    _save(fig, out / "comparison.png")

    public = {
        "scope": "Validation only; test untouched. Aggregate results without patient identifiers or pixels.",
        "feature_cache_id": baseline_manifest["feature_cache_metadata"]["cache_id"],
        "feature_cache_sha256": baseline_manifest["feature_cache_sha256"],
        "training_configuration": training["configuration"],
        "baseline": baseline,
        "runs": [
            {
                "seed": run["seed"],
                "best_epoch": run["best_epoch"],
                "epochs_run": run["epochs_run"],
                "metrics": run["metrics"],
            }
            for run in runs
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / "results.json").write_text(
        json.dumps(public, ensure_ascii=False, indent=2) + "\n"
    )
    return public


def update_readme_results(readme: str | Path, results: dict) -> None:
    """Replace only the marked result block, using completed aggregate metrics."""
    readme = Path(readme)
    text = readme.read_text()
    start, end = "<!-- RESULTS_START -->", "<!-- RESULTS_END -->"
    if text.count(start) != 1 or text.count(end) != 1 or text.index(start) > text.index(end):
        raise ValueError("README needs exactly one ordered RESULTS_START / RESULTS_END block")
    baseline = results["baseline"]
    entries = [("Исходный SpineNet", baseline)] + [
        (f"Новая голова, seed {run['seed']}", run["metrics"]) for run in results["runs"]
    ]
    lines = [
        f"На валидации обработано **{baseline['n_assessed']} из {baseline['n_labeled']} "
        f"целевых уровней ({baseline['coverage']:.1%})**.",
        "",
        "| Модель | Macro-F1 на обработанных областях | Найдено тяжёлых / все тяжёлые |",
        "| --- | ---: | ---: |",
    ]
    for name, metrics in entries:
        severe = metrics["severe"]
        lines.append(
            f"| {name} | {metrics['conditional']['macro_f1']:.3f} | "
            f"{severe['correctly_predicted_severe']} / {severe['n_total']} "
            f"({severe['detection_rate_all_requested']:.1%}) |"
        )
    scores = [run["metrics"]["conditional"]["macro_f1"] for run in results["runs"]]
    std = np.std(scores, ddof=1) if len(scores) > 1 else 0
    lines += [
        "",
        f"Средний macro-F1 новой головы — **{np.mean(scores):.3f} ± {std:.3f}** по "
        f"{len(scores)} запускам (среднее и стандартное отклонение). "
        "Валидация использовалась для выбора эпохи; это ещё не результат на отложенном тесте.",
    ]
    prefix, remainder = text.split(start)
    _, suffix = remainder.split(end)
    readme.write_text(prefix + start + "\n" + "\n".join(lines) + "\n" + end + suffix)
