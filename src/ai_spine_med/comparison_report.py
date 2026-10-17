"""Public aggregate-only report for the frozen, sealed test comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .reporting import COLORS, _save, _style, draw_cohort, draw_confusions, plt

VARIANTS = ("original", "frozen_head", "fine_tuned")
VARIANT_LABELS = ("Исходный\nSpineNet", "Новая\nголова", "Последний блок\nи голова")


def _number(value, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _mean_std(values: list) -> tuple[float | None, float | None]:
    if any(value is None for value in values):
        return None, None
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _read_inputs(head_dir: Path, finetune_dir: Path, test_dir: Path):
    head = json.loads((head_dir / "summary.json").read_text())
    fine = json.loads((finetune_dir / "summary.json").read_text())
    test = json.loads((test_dir / "summary.json").read_text())
    if test.get("scope") != "test" or not test.get("selection_sha256"):
        raise ValueError("The final report requires a sealed test comparison")
    expected = {
        (variant, run["seed"]): run["checkpoint_sha256"]
        for variant, result in (("frozen_head", head), ("fine_tuned", fine))
        for run in result["runs"]
    }
    entries = test["entries"]
    keys = [(entry["variant"], entry["seed"]) for entry in entries]
    if len(set(keys)) != len(keys) or set(keys) != set(expected) | {("original", None)}:
        raise ValueError("Report must include the original model and every fixed seed exactly once")
    for entry in entries:
        key = entry["variant"], entry["seed"]
        if key in expected and entry["checkpoint_sha256"] != expected[key]:
            raise ValueError("Test results and selected training checkpoints differ")
    counts = {(entry["metrics"]["n_labeled"], entry["metrics"]["n_assessed"]) for entry in entries}
    expected_counts = (test["coverage"]["n_labeled"], test["coverage"]["n_assessed"])
    if counts != {expected_counts}:
        raise ValueError("Models were evaluated on different test cohorts")
    return head, fine, test


def _curves(head: dict, fine: dict, finetune_dir: Path, out: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for run in fine["runs"]:
        history = json.loads(
            (finetune_dir / "runs" / f"seed-{run['seed']}" / "history.json").read_text()
        )
        epochs = [row["epoch"] for row in history]
        losses = [
            np.nan if row.get("training_loss") is None else row["training_loss"] for row in history
        ]
        line = axes[0].plot(epochs, losses, marker=".", label=f"seed {run['seed']}")[0]
        scores = [row["validation_macro_f1"] for row in history]
        axes[1].plot(
            epochs, scores, marker=".", color=line.get_color(), label=f"seed {run['seed']}"
        )
        chosen = next(row for row in history if row["epoch"] == run["best_epoch"])
        axes[1].scatter(
            [chosen["epoch"]],
            [chosen["validation_macro_f1"]],
            color=line.get_color(),
            marker="*",
            s=110,
            zorder=4,
        )
    frozen_mean = np.mean([run["metrics"]["conditional"]["macro_f1"] for run in head["runs"]])
    axes[1].axhline(frozen_mean, ls="--", color="#607a8c", label="Замороженная сеть: среднее")
    axes[0].set(
        title="Дообучение layer4.2 и головы", xlabel="Эпоха", ylabel="Weighted cross-entropy"
    )
    axes[1].set(
        title="Выбор эпохи по валидации",
        xlabel="Эпоха (0 — исходное состояние)",
        ylabel="Macro-F1",
        ylim=(0, 1),
    )
    for ax in axes:
        ax.grid(axis="y", alpha=0.15)
        ax.legend(fontsize=8)
    fig.tight_layout(w_pad=3)
    _save(fig, out / "finetuning.png")


def _test_comparison(test: dict, out: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    measures = (
        (lambda m: m["conditional"]["macro_f1"], "Macro-F1 на обработанных областях"),
        (lambda m: m["severe"]["detection_rate_all_requested"], "Найдено тяжёлых / все тяжёлые"),
    )
    for ax, (measure, title) in zip(axes, measures):
        for i, variant in enumerate(VARIANTS):
            values = [measure(e["metrics"]) for e in test["entries"] if e["variant"] == variant]
            mean, std = _mean_std(values)
            if mean is None:
                ax.text(i, 0.04, "Нет оценки", ha="center", fontsize=9)
                continue
            ax.bar(i, mean, width=0.55, color=COLORS[i], alpha=0.8)
            ax.errorbar(i, mean, yerr=std, fmt="none", ecolor="#243447", capsize=5)
            ax.scatter(
                np.linspace(i - 0.1, i + 0.1, len(values)), values, color="#243447", s=20, zorder=4
            )
            label = f"{mean:.3f}" if len(values) == 1 else f"{mean:.3f} ± {std:.3f}"
            ax.text(i, min(mean + std + 0.035, 1.07), label, ha="center", fontsize=9)
        ax.set(xticks=range(3), xticklabels=VARIANT_LABELS, ylim=(0, 1.14), title=title)
        ax.grid(axis="y", alpha=0.15)
    fig.text(
        0.5,
        -0.01,
        "Отложенный тест. Точки — все seed; отрезки — стандартное отклонение между запусками, не доверительный интервал.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(w_pad=3)
    _save(fig, out / "test_comparison.png")


def _by_level(test: dict, out: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for ax, level in zip(axes, ("L4-L5", "L5-S1")):
        for i, variant in enumerate(VARIANTS):
            metrics = [
                e["metrics"]["by_level"][level] for e in test["entries"] if e["variant"] == variant
            ]
            values = [m["conditional"]["macro_f1"] for m in metrics]
            mean, std = _mean_std(values)
            if mean is not None:
                ax.bar(i, mean, color=COLORS[i], width=0.55, alpha=0.8)
                ax.errorbar(i, mean, yerr=std, fmt="none", ecolor="#243447", capsize=5)
                ax.scatter(
                    np.linspace(i - 0.1, i + 0.1, len(values)),
                    values,
                    color="#243447",
                    s=20,
                    zorder=4,
                )
                ax.text(i, min(mean + std + 0.03, 1.03), f"{mean:.3f}", ha="center", fontsize=9)
            severe_counts = [m["severe"]["correctly_predicted_severe"] for m in metrics]
            found = (
                str(severe_counts[0])
                if len(set(severe_counts)) == 1
                else f"{min(severe_counts)}–{max(severe_counts)}"
            )
            ax.text(
                i,
                -0.24,
                f"Тяжёлые: {found}/{metrics[0]['severe']['n_total']}",
                ha="center",
                transform=ax.get_xaxis_transform(),
                fontsize=9,
            )
        first = next(e for e in test["entries"] if e["variant"] == "original")["metrics"][
            "by_level"
        ][level]
        ax.set(
            xticks=range(3),
            xticklabels=VARIANT_LABELS,
            ylabel="Macro-F1",
            ylim=(0, 1.12),
            title=f"{level.replace('-', '–')} · обработано {first['n_assessed']}/{first['n_labeled']}",
        )
        ax.grid(axis="y", alpha=0.15)
    fig.tight_layout(w_pad=3)
    _save(fig, out / "test_by_level.png")


def _public_payload(test: dict, fine: dict) -> dict:
    payload = {
        "scope": "test",
        "selection_sha256": test["selection_sha256"],
        "test_feature_cache_sha256": test.get("test_feature_cache_sha256", {}),
        "coverage": test["coverage"],
        "entries": [
            {k: e[k] for k in ("variant", "seed", "metrics", "checkpoint_sha256")}
            for e in test["entries"]
        ],
        "variants": test["variants"],
        "timings": test.get("timings", {}),
        "finetuning_validation": {
            "runs": [
                {
                    k: run[k]
                    for k in (
                        "seed",
                        "best_epoch",
                        "epochs_run",
                        "validation_macro_f1",
                        "checkpoint_sha256",
                    )
                }
                for run in fine["runs"]
            ],
            "macro_f1_mean": fine["validation_macro_f1_mean"],
            "macro_f1_std": fine["validation_macro_f1_std"],
        },
        "interpretation": "Test evaluated once after checkpoint selection. Seed standard deviation is not a confidence interval. ROI localization and clinical labels have not been expert-validated.",
    }
    forbidden = {
        "study_id",
        "series_id",
        "patient_id",
        "roi_path",
        "dicom_path",
        "api_key",
        "token",
    }

    def inspect(value):
        if isinstance(value, dict):
            if forbidden.intersection(k.lower() for k in value):
                raise ValueError("Patient-level or secret fields cannot enter the public report")
            for item in value.values():
                inspect(item)
        elif isinstance(value, list):
            for item in value:
                inspect(item)

    inspect(payload)
    return payload


def results_markdown(test: dict) -> str:
    original = next(e for e in test["entries"] if e["variant"] == "original")["metrics"]
    lines = [
        f"На отложенном тесте оценил **{original['n_assessed']} из {original['n_labeled']} уровней ({original['coverage']:.1%})**.",
        "",
        "| Модель | Seed | Macro-F1 на обработанных областях | Тяжёлые: определены / всего | Тяжёлый → норма/лёгкий | Тяжёлый → умеренный |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    names = {
        "original": "Исходный SpineNet",
        "frozen_head": "Новая голова",
        "fine_tuned": "Последний блок + голова",
    }
    for e in test["entries"]:
        m, severe = e["metrics"], e["metrics"]["severe"]
        lines.append(
            f"| {names[e['variant']]} | {e['seed'] if e['seed'] is not None else '—'} | {_number(m['conditional']['macro_f1'])} | {severe['correctly_predicted_severe']} / {severe['n_total']} | {severe['predicted_normal_mild']} | {severe['predicted_moderate']} |"
        )
    lines += ["", "Среднее и стандартное отклонение macro-F1 по трём seed:", ""]
    for variant in VARIANTS[1:]:
        values = [
            e["metrics"]["conditional"]["macro_f1"]
            for e in test["entries"]
            if e["variant"] == variant
        ]
        mean, std = _mean_std(values)
        lines.append(f"- {names[variant]}: **{_number(mean)} ± {_number(std)}**.")
    if original["severe"]["not_assessed"]:
        severe_coverage = (
            f"Среди всех {original['severe']['n_total']} тяжёлых уровней "
            f"**{original['severe']['not_assessed']} не обработано**. "
            "Они остаются в общем знаменателе обнаружения тяжёлого стеноза."
        )
    else:
        severe_coverage = f"Все **{original['severe']['n_total']} тяжёлых уровней обработаны**."
    lines += [
        "",
        severe_coverage + " Стандартное отклонение описывает разброс между запусками, "
        "а не неопределённость на новых пациентах.",
    ]
    timings = test.get("timings", {})
    benchmark = timings.get("classifier_batch1", [])
    if benchmark:
        device = timings.get("hardware", {}).get("device", "не указано")
        lines += [
            "",
            f"Полную сеть классификации замерил на **{device}**, по одной готовой области за раз. "
            f"Для каждой модели — {benchmark[0]['draws']} измерения после {benchmark[0]['warmup']} прогревочных запусков.",
            "",
            "| Модель | Seed | Медиана, мс/область | 90-й процентиль, мс |",
            "| --- | ---: | ---: | ---: |",
        ]
        for row in benchmark:
            lines.append(
                f"| {names[row['variant']]} | {row['seed'] if row['seed'] is not None else '—'} | "
                f"{1000 * row['median_seconds']:.1f} | {1000 * row['p90_seconds']:.1f} |"
            )
        common = timings.get("common_preprocessing", {})
        full = common.get("full_original_seconds", {})
        residual = common.get("common_DICOM_localization_ROI_plus_tensorIO_seconds", {})
        if full.get("median_seconds") is not None:
            lines += [
                "",
                f"Отдельно во время извлечения тестовых признаков полный исходный конвейер "
                f"— от чтения DICOM до оценок — занял "
                f"в медиане **{full['median_seconds']:.2f} с на серию** ({full['count']} серий).",
            ]
        if residual.get("median_seconds") is not None:
            lines.append(
                f"Общая часть без forward классификатора заняла в медиане "
                f"{residual['median_seconds']:.2f} с на серию; сюда входят чтение, поиск уровней, "
                "подготовка областей и перенос результатов. Это не отдельный замер чистого детектора."
            )
        lines += [
            "",
            "В замер классификации не входят чтение с диска, поиск уровней и загрузка весов. "
            "Во всех вариантах почти одинаковое число операций; небольшие различия времени "
            "не считаю доказательством ускорения после дообучения.",
        ]
    return "\n".join(lines)


def _replace_block(path: Path, name: str, body: str) -> None:
    content = path.read_text()
    start, end = f"<!-- {name}_START -->", f"<!-- {name}_END -->"
    if (
        content.count(start) != 1
        or content.count(end) != 1
        or content.index(start) > content.index(end)
    ):
        raise ValueError(f"Expected one ordered {name} block in {path.name}")
    prefix, remainder = content.split(start)
    _, suffix = remainder.split(end)
    path.write_text(prefix + start + "\n" + body + "\n" + end + suffix)


def _validation_markdown(fine: dict) -> str:
    lines = [
        "Выбранные модели после дообучения последнего блока (валидация):",
        "",
        "| Seed | Обучено эпох | Выбранная эпоха | Macro-F1 | Тяжёлые: определены / всего |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in fine["runs"]:
        severe = run["metrics"]["severe"]
        lines.append(
            f"| {run['seed']} | {run['epochs_run']} | {run['best_epoch']} | "
            f"{run['validation_macro_f1']:.3f} | "
            f"{severe['correctly_predicted_severe']} / {severe['n_total']} |"
        )
    lines += [
        "",
        f"Средний validation macro-F1 — **{fine['validation_macro_f1_mean']:.3f} ± "
        f"{fine['validation_macro_f1_std']:.3f}** (стандартное отклонение между seed). "
        "Эпоха 0 означает, что по валидации сохранилось исходное состояние этого запуска. "
        "Эта таблица использовалась для выбора эпохи, итоговую оценку даёт отдельный тест.",
    ]
    return "\n".join(lines)


def draw_final_comparison(
    head_dir: str | Path,
    finetune_dir: str | Path,
    test_dir: str | Path,
    out: str | Path = "docs/images",
    *,
    readme: str | Path | None = None,
    protocol_doc: str | Path | None = None,
    manifest_dir: str | Path | None = None,
) -> dict:
    """Render completed aggregate metrics; never open a patient prediction CSV."""
    head_dir, finetune_dir, test_dir, out = map(Path, (head_dir, finetune_dir, test_dir, out))
    head, fine, test = _read_inputs(head_dir, finetune_dir, test_dir)
    payload = _public_payload(test, fine)
    if manifest_dir:
        draw_cohort(Path(manifest_dir), out, test_evaluated=True)
    _curves(head, fine, finetune_dir, out)
    _test_comparison(test, out)
    _by_level(test, out)
    original = next(e for e in test["entries"] if e["variant"] == "original")
    for variant, filename in (
        ("frozen_head", "test_confusion_frozen.png"),
        ("fine_tuned", "test_confusion_finetuned.png"),
    ):
        entries = [original] + [e for e in test["entries"] if e["variant"] == variant]
        matrices = [
            (
                "Исходный SpineNet"
                if e["variant"] == "original"
                else f"{'Голова' if variant == 'frozen_head' else 'Блок + голова'} · seed {e['seed']}",
                e["metrics"]["conditional"]["confusion_matrix"],
            )
            for e in entries
        ]
        draw_confusions(matrices, out, filename=filename)
    (out.parent / "test-results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    body = results_markdown(test)
    if readme:
        _replace_block(Path(readme), "FINAL_RESULTS", body)
        if "<!-- FINETUNE_VALIDATION_START -->" in Path(readme).read_text():
            _replace_block(Path(readme), "FINETUNE_VALIDATION", _validation_markdown(fine))
    if protocol_doc:
        _replace_block(Path(protocol_doc), "FINETUNE_RESULTS", body)
        if "<!-- FINETUNE_VALIDATION_START -->" in Path(protocol_doc).read_text():
            _replace_block(Path(protocol_doc), "FINETUNE_VALIDATION", _validation_markdown(fine))
    return payload
