"""Private-input/public-aggregate report for the XML level-localizer comparison.

Only fixed arm names and typed aggregate values enter publication. Neither
patient identities nor arbitrary source metadata are copied into the report.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .assets import sha256_file, write_json

ARMS = ("xml_only", "mixed")
SEEDS = (42, 2024, 3407)
DATASETS = ("aisslab", "rsna")
LEVELS = ("L4-L5", "L5-S1")
GROUPS = tuple(f"{dataset}/{level}" for dataset in DATASETS for level in LEVELS)
NAMES = {"xml_only": "Только XML AISSLab", "mixed": "AISSLab + RSNA"}
COLORS = {"xml_only": "#257d89", "mixed": "#bd8358"}


def _number(value, *, count=False, rate=False, nullable=False):
    if nullable and value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("Invalid nonnegative report aggregate")
    if count and type(value) is not int:
        raise ValueError("A report count must be an integer")
    if rate and value > 1:
        raise ValueError("A report rate must be in [0, 1]")
    return value


def level_snapshot(value):
    """Validate denominators as well as rates; do not copy unknown source fields."""
    result = {"groups": {}}
    for group in GROUPS:
        source = value["groups"][group]
        n = _number(source["n"], count=True)
        if n == 0:
            raise ValueError("A comparison group must have referenced validation levels")
        row = {
            "n": n,
            "hit5": _number(source["hit5"], rate=True),
            "hit10": _number(source["hit10"], rate=True),
            "unverifiable_top_series": _number(source["unverifiable_top_series"], count=True),
            "median_distance_mm_when_verifiable": _number(
                source["median_distance_mm_when_verifiable"], nullable=True
            ),
            "p90_distance_mm_when_verifiable": _number(
                source["p90_distance_mm_when_verifiable"], nullable=True
            ),
        }
        for field in ("requested", "predicted", "missing_reference"):
            row[field] = _number(source[field], count=True)
        if row["requested"] != n + row["missing_reference"] or row["predicted"] > row["requested"]:
            raise ValueError("Requested, predicted and referenced validation counts disagree")
        for metric in ("hit5", "hit10"):
            hits = round(row[metric] * n)
            if not math.isclose(row[metric], hits / n, abs_tol=1e-8, rel_tol=0):
                raise ValueError("A hit rate is inconsistent with its reference denominator")
            if hits > n - row["unverifiable_top_series"]:
                raise ValueError("An unverifiable prediction cannot count as a localization hit")
            row[f"{metric}_count"] = hits
            if hits != _number(source[f"{metric}_count"], count=True):
                raise ValueError("Reported hit count differs from its rate and denominator")
        if row["hit5_count"] > row["hit10_count"]:
            raise ValueError("Hits within 5mm cannot exceed hits within 10mm")
        if row["unverifiable_top_series"] > n:
            raise ValueError("Unverifiable count exceeds reference denominator")
        result["groups"][group] = row
    expected = float(np.mean([r["hit10"] for r in result["groups"].values()]))
    if "macro_hit10" in value and not math.isclose(
        _number(value["macro_hit10"], rate=True), expected, abs_tol=1e-8, rel_tol=0
    ):
        raise ValueError("Macro hit10 differs from the four declared groups")
    result["macro_hit10"] = expected
    return result


def _validate_cases(cases):
    """Validate the private cohort before computing anything publishable."""
    from .xml_localizer_ablation import aggregate_cases

    seen = set()
    for row in cases:
        key = row["dataset"], row["study_id"], row["level"]
        if (
            row["split"] != "validation"
            or row["dataset"] not in DATASETS
            or row["level"] not in LEVELS
            or not isinstance(row["study_id"], str)
            or not row["study_id"]
            or key in seen
            or type(row["reference_present"]) is not bool
            or type(row["predicted"]) is not bool
        ):
            raise ValueError("Invalid or duplicate private validation identity")
        seen.add(key)
        distance = _number(row["distance_mm"], nullable=True)
        if not row["reference_present"] and distance is not None:
            raise ValueError("Missing anatomical reference cannot provide a distance")
        for metric, limit in (("hit5", 5), ("hit10", 10)):
            if type(row[metric]) is not bool or row[metric] != (
                row["reference_present"]
                and row["predicted"]
                and distance is not None
                and distance <= limit
            ):
                raise ValueError("Localization hit disagrees with physical reference distance")
    return seen, level_snapshot(aggregate_cases(cases))


def public_snapshot(summary, private):
    """Require the same real patient/level keys, not merely equal-sized cohorts."""
    if summary.get("phase") != "validation" or summary.get("center_reused_exactly") is not True:
        raise ValueError("Need a VAL-only comparison with the frozen v6 center")
    if set(summary["arms"]) != set(ARMS):
        raise ValueError("Expected both fixed comparison arms")
    output = {
        "schema_version": 1,
        "experiment": "xml_only_vs_mixed_axial_level_localization",
        "phase": "validation",
        "seeds": list(SEEDS),
        "selection_dataset": "aisslab",
        "selection_metric": "mean hit10 over AISSLab levels; tie: cross-entropy",
        "center_reused_exactly": True,
        "test_evaluated": False,
        "severity_evaluated": False,
        "original_validation_failures": _number(
            summary["original_validation_failures"], count=True
        ),
        "arms": {},
    }
    cohort = None
    masks = None
    denominators = None
    for arm in ARMS:
        source = summary["arms"][arm]
        expected_datasets = ["aisslab"] if arm == "xml_only" else ["aisslab", "rsna"]
        if source["training_datasets"] != expected_datasets:
            raise ValueError("Training source labels do not match the fixed arm")
        seeds = source["seeds"]
        if sorted(row["seed"] for row in seeds) != list(SEEDS):
            raise ValueError("Require every predetermined seed once")
        arm_output = {"training_datasets": expected_datasets, "seeds": []}
        fit_counts = None
        for run in sorted(seeds, key=lambda row: row["seed"]):
            snapshot = level_snapshot(run["validation_levels"])
            prior = level_snapshot(run["positional_baseline"])
            counts = {
                dataset: _number(run["fit_counts"][dataset], count=True)
                for dataset in expected_datasets
            }
            if set(run["fit_counts"]) != set(expected_datasets) or not all(counts.values()):
                raise ValueError("Unexpected or empty training domain")
            if fit_counts is not None and counts != fit_counts:
                raise ValueError("Seed training cohorts differ in size")
            fit_counts = counts
            cases = private["seeds"][f"{arm}/seed_{run['seed']}"]
            keys, recomputed = _validate_cases(cases)
            reference_masks = {
                (r["dataset"], r["study_id"], r["level"]): r["reference_present"] for r in cases
            }
            if snapshot != recomputed:
                raise ValueError("Published aggregates differ from private predictions")
            if cohort is not None and (keys != cohort or reference_masks != masks):
                raise ValueError("Compared patient/level cohorts or reference masks differ")
            cohort, masks = keys, reference_masks
            sizes = {group: snapshot["groups"][group]["n"] for group in GROUPS}
            if denominators is not None and sizes != denominators:
                raise ValueError("Compared reference denominators differ")
            denominators = sizes
            if any(
                prior["groups"][g][field] != snapshot["groups"][g][field]
                for g in GROUPS
                for field in ("n", "requested", "missing_reference")
            ):
                raise ValueError("Positional prior uses a different reference cohort")
            arm_output["seeds"].append(
                {
                    "seed": run["seed"],
                    "selected_epoch": _number(run["selected_epoch"], count=True),
                    "fit_counts": counts,
                    "validation_levels": snapshot,
                    "positional_baseline": prior,
                }
            )
        ensemble = level_snapshot(source["ensemble"])
        keys, recomputed = _validate_cases(private["ensembles"][arm])
        ensemble_masks = {
            (r["dataset"], r["study_id"], r["level"]): r["reference_present"]
            for r in private["ensembles"][arm]
        }
        if keys != cohort or ensemble_masks != masks or ensemble != recomputed:
            raise ValueError("Ensemble cohort or aggregates differ from the fixed comparison")
        arm_output["ensemble"] = ensemble
        arm_output["aggregates"] = {
            group: {
                metric: _mean_std(
                    [s["validation_levels"]["groups"][group][metric] for s in arm_output["seeds"]]
                )
                for metric in ("hit5", "hit10")
            }
            for group in GROUPS
        }
        output["arms"][arm] = arm_output
    paired = summary["paired_ensemble_comparison"]
    comparison = {
        "unit": "patient within each dataset/level",
        "resamples": _number(paired["resamples"], count=True),
        "groups": {},
    }
    if comparison["resamples"] < 1:
        raise ValueError("Bootstrap must contain resamples")
    for group in GROUPS:
        row = paired["groups"][group]
        n = _number(row["n_patients"], count=True)
        delta, intervals = row["delta_hit10"], row["ci95"]
        if (
            n != denominators[group]
            or type(delta) not in (int, float)
            or not math.isfinite(delta)
            or not -1 <= delta <= 1
            or len(intervals) != 2
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or not -1 <= v <= 1
                for v in intervals
            )
            or intervals[0] > intervals[1]
        ):
            raise ValueError("Invalid paired patient bootstrap aggregate")
        expected = (
            output["arms"]["xml_only"]["ensemble"]["groups"][group]["hit10"]
            - output["arms"]["mixed"]["ensemble"]["groups"][group]["hit10"]
        )
        if not math.isclose(delta, expected, abs_tol=1e-8, rel_tol=0):
            raise ValueError("Bootstrap difference disagrees with the fixed ensembles")
        comparison["groups"][group] = {
            "n_patients": n,
            "delta_hit10": delta,
            "ci95": list(intervals),
        }
    output["paired_ensemble_comparison"] = comparison
    return output


def _mean_std(values):
    return {"mean": float(np.mean(values)), "std_across_seeds": float(np.std(values, ddof=0))}


def xml_training_counts(index_path, summary, public):
    """Count source XML level targets without publishing their identities."""
    index_path = Path(index_path)
    if sha256_file(index_path) != summary["provenance"]["training_index_sha256"]:
        raise ValueError("Training-index fingerprint differs from the fitted comparison")
    index = json.loads(index_path.read_text())
    patients, targets = set(), 0
    for row in index["records"]:
        if row["dataset"] != "aisslab" or row["split"] != "train" or row["status"] != "ok":
            continue
        if row["study_id"] in patients:
            raise ValueError("Duplicate XML TRAIN identity")
        patients.add(row["study_id"])
        path = (index_path.parent / row["arrays_path"]).resolve()
        if (
            not path.is_relative_to(index_path.parent.resolve())
            or sha256_file(path) != row["arrays_sha256"]
        ):
            raise ValueError("XML target arrays escaped their root or changed")
        with np.load(path, allow_pickle=False) as archive:
            mask = archive["level_mask"]
        if mask.shape != (2,) or mask.dtype != np.bool_:
            raise ValueError("XML training level mask must contain two booleans")
        targets += int(mask.sum())
    expected = public["arms"]["xml_only"]["seeds"][0]["fit_counts"]["aisslab"]
    if len(patients) != expected:
        raise ValueError("XML training patient count differs from the fitted arm")
    return {"patients": len(patients), "referenced_levels": targets}


def draw_comparison(public, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), sharey=True)
    fig.patch.set_facecolor("#faf9f6")
    for ax, dataset in zip(axes, DATASETS):
        ax.set_facecolor("#faf9f6")
        for offset, arm in zip((-0.18, 0.18), ARMS):
            values = [public["arms"][arm]["aggregates"][f"{dataset}/{level}"] for level in LEVELS]
            means = [v["hit10"]["mean"] for v in values]
            deviations = [v["hit10"]["std_across_seeds"] for v in values]
            positions = np.arange(2) + offset
            ax.bar(
                positions,
                means,
                0.32,
                color=COLORS[arm],
                label=NAMES[arm],
                yerr=deviations,
                capsize=4,
            )
            for position, mean in zip(positions, means):
                ax.text(position, min(mean + 0.035, 1.045), f"{mean:.1%}", ha="center", fontsize=10)
        ax.set_xticks(range(2), LEVELS)
        ax.set_ylim(0, 1.09)
        ax.set_yticks(np.arange(0, 1.01, 0.2), [f"{v:.0%}" for v in np.arange(0, 1.01, 0.2)])
        ax.set_title("AISSLab VAL · выбор эпохи" if dataset == "aisslab" else "RSNA VAL · перенос")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Срез не дальше 10 мм от ориентира")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.03)
    )
    fig.suptitle("Источники разметки для выбора уровня", fontsize=15)
    fig.text(
        0.5,
        0.005,
        "Среднее трёх запусков; отрезки — стандартное отклонение, не доверительный интервал.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    fig.savefig(output, dpi=160, metadata={"Creator": "ai-spine-med aggregate report"})
    plt.close(fig)


def render_markdown(public, *, image="images/xml_localizer_comparison.png"):
    def percent(value):
        return f"{value:.1%}".replace(".", ",")

    lines = [
        "# Можно ли находить уровни только по аксиальным МРТ и XML",
        "",
        "Я отдельно обучил модель выбора уровня только по XML AISSLab. "
        "Сагиттальные снимки для него не нужны. В предыдущем v6 центр канала уже обучался "
        "только по этим XML; здесь я оставил его веса без изменений и сравнил именно выбор среза.",
        "",
        "На AISSLab оба варианта уверенно находят размеченные уровни. При переносе на RSNA "
        "вариант только с XML заметно уступает смешанному; разница сохраняется в парных "
        "интервалах ниже. Поэтому для изображений другого источника я пока оставляю "
        "смешанное обучение, а XML использую как полезную анатомическую разметку.",
        "",
        "Второй вариант использует и AISSLab, и приблизительные ориентиры уровней RSNA. "
        "Для каждого варианта сделал три запуска с seed 42, 2024 и 3407. "
        "Архитектура, лимит эпох и критерий выбора эпохи одинаковые: среднее попадание "
        "в пределах 10 мм по двум уровням AISSLab VAL, при равенстве — меньшая cross-entropy. "
        "Вариант без дообучения, использующий только позиционный prior, тоже мог быть выбран.",
        "",
        "Все шесть моделей зафиксированы до оценки RSNA VAL. Центр, классификатор степени "
        "стеноза и TEST на этом шаге не обучал и не переоценивал.",
        "",
        "## Результаты",
        "",
        "В таблице — ансамбли трёх заранее заданных запусков: усреднение вероятностей срезов "
        "перед выбором максимума. Числа в числителе — попадания, в знаменателе — уровни "
        "с доступным анатомическим ориентиром. Отсутствующая разметка не считается нормой "
        "или правильным предсказанием.",
        "",
        "| Данные и уровень | Только XML AISSLab, ≤10 мм | AISSLab + RSNA, ≤10 мм | "
        "Разница XML − mixed, 95% ДИ |",
        "|---|---:|---:|---:|",
    ]
    if "xml_training_annotation_counts" in public:
        counts = public["xml_training_annotation_counts"]
        lines[2:2] = [
            f"В XML-ветке {counts['patients']} пациентов TRAIN и "
            f"{counts['referenced_levels']} анатомических ориентиров двух нужных уровней.",
            "",
        ]
    for group in GROUPS:
        xml = public["arms"]["xml_only"]["ensemble"]["groups"][group]
        mixed = public["arms"]["mixed"]["ensemble"]["groups"][group]
        comparison = public["paired_ensemble_comparison"]["groups"][group]
        delta = 100 * comparison["delta_hit10"]
        low, high = (100 * v for v in comparison["ci95"])
        lines.append(
            f"| {group.replace('/', ' · ')} | {xml['hit10_count']}/{xml['n']} "
            f"({percent(xml['hit10'])}) | {mixed['hit10_count']}/{mixed['n']} "
            f"({percent(mixed['hit10'])}) | {delta:+.1f} п.п. [{low:+.1f}; {high:+.1f}] |"
        )
    lines.extend(
        [
            "",
            "Проверил и пользу обучения сверх позиции среза. На AISSLab VAL "
            "исходный позиционный prior XML-ветки даёт:",
            "",
        ]
    )
    for level in LEVELS:
        group = f"aisslab/{level}"
        prior = public["arms"]["xml_only"]["seeds"][0]["positional_baseline"]["groups"][group]
        trained = public["arms"]["xml_only"]["ensemble"]["groups"][group]
        lines.append(
            f"- {level}: {prior['hit10_count']}/{prior['n']} → "
            f"{trained['hit10_count']}/{trained['n']} после обучения и ансамблирования."
        )
    lines.extend(
        [
            "",
            f"![Сравнение трёх запусков по каждому источнику разметки]({image})",
            "",
            "На графике показаны отдельные запуски в агрегированном виде: среднее и стандартное "
            "отклонение между тремя seed. Это не доверительные интервалы. Интервалы в таблице "
            "получены парным bootstrap по пациентам отдельно для каждого датасета и уровня; "
            "если интервал включает ноль, убедительного различия здесь нет.",
            "",
            "| Вариант | Пациенты TRAIN | Выбранные эпохи для seed 42 / 2024 / 3407 |",
            "|---|---|---|",
        ]
    )
    for arm in ARMS:
        runs = public["arms"][arm]["seeds"]
        counts = ", ".join(f"{dataset}: {n}" for dataset, n in runs[0]["fit_counts"].items())
        epochs = " / ".join(str(run["selected_epoch"]) for run in runs)
        lines.append(f"| {NAMES[arm]} | {counts} | {epochs} |")
    lines.extend(
        [
            "",
            "## Что сравнивается и где ограничения",
            "",
            "Это сравнение двух схем обучения по источнику разметки. Одновременно меняются "
            "количество пациентов, домен изображений, нормализаторы и позиционный prior, "
            "поэтому разницу нельзя приписать только качеству XML. Prior рассчитывается "
            "отдельно по TRAIN каждого варианта; в JSON сохранены результаты его проверки. "
            "Смешанная ветка также получает больше обновлений оптимизатора за эпоху, "
            "поскольку её TRAIN больше. Число обновлений между ветками не выравнивал.",
            "",
            "AISSLab VAL используется для выбора эпохи, поэтому его оценки оптимистичны. "
            "RSNA VAL в этом эксперименте используется только после фиксации весов, но эта "
            "выборка уже применялась в прошлой работе: это не новый независимый тест. "
            "Прежний смешанный v6 выбирал эпоху по другому критерию — по обоим датасетам; "
            "его прежние числа нельзя трактовать как третью равноправную ветку этого сравнения.",
            "",
            "У AISSLab ориентир — размеченный XML срез и центр bounding box канала. "
            "Это слабый анатомический ориентир, а не маска сегментации канала. "
            "Клиническую правильность предсказанного центра ещё нужно проверить с врачом. "
            "У RSNA координаты субартикулярных областей дают приблизительный уровень, "
            "а не точную разметку центра канала. Попадание в 10 мм не означает правильную "
            "оценку стеноза. На этом шаге я не проверял улучшение классификации степени, "
            "контура или площади канала.",
            "",
            "## Покрытие",
            "",
            "| Данные и уровень | Запрошено в доступных сериях | Предсказано | "
            "Нет анатомического ориентира | Непроверяемый срез XML / mixed |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for group in GROUPS:
        xml = public["arms"]["xml_only"]["ensemble"]["groups"][group]
        mixed = public["arms"]["mixed"]["ensemble"]["groups"][group]
        lines.append(
            f"| {group.replace('/', ' · ')} | {xml['requested']} | {xml['predicted']} / "
            f"{mixed['predicted']} | {xml['missing_reference']} | "
            f"{xml['unverifiable_top_series']} / {mixed['unverifiable_top_series']} |"
        )
    lines.extend(
        [
            "",
            f"Ещё {public['original_validation_failures']} исходных VAL исследований не удалось "
            "обработать в исходной подготовке. Они не входят в доступные серии этой таблицы "
            "и не превращаются в успешные предсказания.",
            "",
            "Агрегированные метрики, включая попадания в 5 мм, расстояния и каждый seed: "
            "[xml-localizer-results.json](xml-localizer-results.json). "
            "Протокол: [xml-localizer-v7.json](../configs/xml-localizer-v7.json). "
            "Предыдущий этап: [локализация и оценка стеноза v6](localized-stenosis.md).",
            "",
            "## Повторить",
            "",
            "```bash",
            "python -m ai_spine_med.xml_localizer_ablation --device mps",
            "python -m ai_spine_med.xml_localizer_report",
            "```",
            "",
            "Для повторного обучения нужна новая пустая папка вывода. Исходные данные, "
            "снимки, веса и предсказания по пациентам остаются в локальных игнорируемых папках.",
            "",
        ]
    )
    return "\n".join(lines)


def report_xml_localizer(
    summary_path=Path("reports/runs/xml-localizer-v7/summary.json"),
    *,
    private_path=None,
    training_index=None,
    out_json=Path("docs/xml-localizer-results.json"),
    out_markdown=Path("docs/xml-localizer.md"),
    out_image=Path("docs/images/xml_localizer_comparison.png"),
):
    summary_path = Path(summary_path)
    private_path = (
        Path(private_path) if private_path else summary_path.parent / "cases-private.json"
    )
    summary = json.loads(summary_path.read_text())
    private = json.loads(private_path.read_text())
    public = public_snapshot(summary, private)
    if training_index is not None:
        public["xml_training_annotation_counts"] = xml_training_counts(
            training_index, summary, public
        )
    out_json, out_markdown, out_image = map(Path, (out_json, out_markdown, out_image))
    write_json(out_json, public)
    out_markdown.parent.mkdir(parents=True, exist_ok=True)
    image_link = "images/" + out_image.name
    out_markdown.write_text(render_markdown(public, image=image_link))
    draw_comparison(public, out_image)
    return public


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", type=Path, default=Path("reports/runs/xml-localizer-v7/summary.json")
    )
    parser.add_argument("--private", type=Path)
    parser.add_argument(
        "--training-index", type=Path, default=Path("artifacts/axial-localizer-v6/index.json")
    )
    parser.add_argument("--out-json", type=Path, default=Path("docs/xml-localizer-results.json"))
    parser.add_argument("--out-markdown", type=Path, default=Path("docs/xml-localizer.md"))
    parser.add_argument(
        "--out-image", type=Path, default=Path("docs/images/xml_localizer_comparison.png")
    )
    args = parser.parse_args(argv)
    report_xml_localizer(
        args.summary,
        private_path=args.private,
        training_index=args.training_index,
        out_json=args.out_json,
        out_markdown=args.out_markdown,
        out_image=args.out_image,
    )


if __name__ == "__main__":
    main()
