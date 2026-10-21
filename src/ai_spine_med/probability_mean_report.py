"""Public aggregates for the post-hoc equal-probability MRI comparison."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .training import evaluate_predictions

MODELS = ("sagittal", "axial", "fusion", "probability_mean")
CLASSES = ("normal_mild", "moderate", "severe")
LEVELS = ("L4-L5", "L5-S1")
NAMES = {
    "sagittal": "Сагиттальная",
    "axial": "Аксиальная",
    "fusion": "Fusion признаков",
    "probability_mean": "Среднее вероятностей",
}
COLORS = ("#4979a5", "#d79138", "#498568", "#80549b")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_COUNT_FIELDS = (
    "n_requested",
    "n_labeled",
    "n_missing_reference_label",
    "n_assessed",
    "n_not_assessed",
)
_SEVERE_FIELDS = (
    "n_total",
    "n_assessed",
    "correctly_predicted_severe",
    "predicted_normal_mild",
    "predicted_moderate",
    "not_assessed",
    "recall_conditional",
    "detection_rate_all_requested",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value, *, integer=False, unit=False):
    if value is None and not integer:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Invalid aggregate number")
    if integer and (type(value) is not int or value < 0):
        raise ValueError("Invalid aggregate count")
    if unit and not 0 <= value <= 1:
        raise ValueError("Invalid aggregate rate")
    return value


def _metric_snapshot(metric: dict, *, levels=True) -> dict:
    """Whitelist typed aggregates; never copy arbitrary metadata or local IDs."""
    result = {key: _number(metric[key], integer=True) for key in _COUNT_FIELDS}
    result["coverage"] = _number(metric["coverage"], unit=True)
    for field in ("class_counts", "assessed_class_counts", "not_assessed_class_counts"):
        result[field] = {name: _number(metric[field][name], integer=True) for name in CLASSES}
    conditional = metric["conditional"]
    result["conditional"] = {
        field: _number(conditional[field], unit=True) for field in ("macro_f1", "accuracy")
    }
    for field in ("precision", "recall", "f1"):
        result["conditional"][field] = {
            name: _number(conditional[field][name], unit=True) for name in CLASSES
        }
    matrix = conditional["confusion_matrix"]
    if not isinstance(matrix, list) or len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError("Expected a three-class confusion matrix")
    result["conditional"]["confusion_matrix"] = [
        [_number(value, integer=True) for value in row] for row in matrix
    ]
    result["conditional"]["confusion_axes"] = (
        "rows=true, columns=predicted; normal_mild,moderate,severe"
    )
    result["severe"] = {
        field: _number(
            metric["severe"][field],
            integer=field not in {"recall_conditional", "detection_rate_all_requested"},
            unit=field in {"recall_conditional", "detection_rate_all_requested"},
        )
        for field in _SEVERE_FIELDS
    }
    if levels:
        result["by_level"] = {
            level: _metric_snapshot(metric["by_level"][level], levels=False) for level in LEVELS
        }
    if (
        sum(sum(row) for row in matrix) != result["n_assessed"]
        or result["n_assessed"] + result["n_not_assessed"] != result["n_labeled"]
        or sum(result["class_counts"].values()) != result["n_labeled"]
        or sum(result["assessed_class_counts"].values()) != result["n_assessed"]
        or sum(result["not_assessed_class_counts"].values()) != result["n_not_assessed"]
    ):
        raise ValueError("Aggregate coverage or confusion counts disagree")
    severe = result["severe"]
    if (
        sum(
            severe[k]
            for k in (
                "correctly_predicted_severe",
                "predicted_normal_mild",
                "predicted_moderate",
                "not_assessed",
            )
        )
        != severe["n_total"]
    ):
        raise ValueError("Severe errors and abstentions do not sum to all requested severe targets")
    severe["missed_all_requested"] = severe["n_total"] - severe["correctly_predicted_severe"]
    if (
        "missed_all_requested" in metric["severe"]
        and metric["severe"]["missed_all_requested"] != severe["missed_all_requested"]
    ):
        raise ValueError("Reported severe misses disagree with the complete denominator")
    return result


def _same_tree(left, right):
    if isinstance(right, dict):
        if not isinstance(left, dict) or set(left) != set(right):
            raise ValueError("Stored aggregate structure differs from recomputed predictions")
        for key in right:
            _same_tree(left[key], right[key])
    elif isinstance(right, list):
        if not isinstance(left, list) or len(left) != len(right):
            raise ValueError("Stored aggregate array differs from recomputed predictions")
        for a, b in zip(left, right):
            _same_tree(a, b)
    elif isinstance(right, (int, float)) and not isinstance(right, bool):
        if (
            not isinstance(left, (int, float))
            or isinstance(left, bool)
            or not math.isclose(left, right, rel_tol=0, abs_tol=1e-12)
        ):
            raise ValueError("Stored aggregate value differs from recomputed predictions")
    elif left != right:
        raise ValueError("Stored aggregate value differs from recomputed predictions")


def _read_predictions(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    rows, keys = [], set()
    for item in raw:
        key = (item["study_id"], item["series_id"], item["level"])
        if key in keys or item["level"] not in LEVELS or item["split"] != "validation":
            raise ValueError("Prediction identity, level or development split is invalid")
        keys.add(key)
        label = int(item["label"])
        if label not in (0, 1, 2):
            raise ValueError("Missing or invalid reference label")
        prediction = None if item["prediction"] == "" else int(item["prediction"])
        fields = [item[f"p_{name}"] for name in CLASSES]
        if item["status"] == "ok":
            p = np.asarray([float(value) for value in fields])
            if (
                not np.isfinite(p).all()
                or np.any(p < 0)
                or np.any(p > 1)
                or not np.isclose(p.sum(), 1, rtol=0, atol=1e-6)
                or prediction != int(p.argmax())
            ):
                raise ValueError("Invalid probability vector or category")
        elif (
            item["status"] != "not_assessed"
            or prediction is not None
            or any(value != "" for value in fields)
        ):
            raise ValueError("An abstention must not carry a category or imputed probabilities")
        rows.append({**item, "label": label, "prediction": prediction})
    return rows


def _intervals(source):
    bootstrap = source["bootstrap"]
    if (
        bootstrap.get("n_studies") != source["n_studies"]
        or bootstrap.get("n_targets") != source["n_targets"]
        or bootstrap.get("n_assessed") != source["n_assessed"]
        or bootstrap.get("resamples") != 2000
        or bootstrap.get("seed") != 2026
    ):
        raise ValueError("Bootstrap population or fixed settings differ")
    public = {
        "method": "Paired study-level percentile bootstrap; both levels and abstentions stay together",
        "resamples": 2000,
        "seed": 2026,
        "n_studies": source["n_studies"],
        "n_targets": source["n_targets"],
        "n_assessed": source["n_assessed"],
        "models": {},
        "contrasts": {},
    }
    for name in MODELS:
        public["models"][name] = {}
        for metric in ("macro_f1", "moderate_f1", "severe_rate"):
            item = bootstrap["models"][name][metric]
            public["models"][name][metric] = _interval(item)
    for name in MODELS[:3]:
        key = f"mean_minus_{name}"
        contrast = bootstrap["contrasts"][key]
        if contrast["left"] != ["probability_mean"] or contrast["right"] != [name]:
            raise ValueError("Bootstrap contrast is not probability mean minus its baseline")
        public["contrasts"][key] = {
            "left": ["probability_mean"],
            "right": [name],
            "metrics": {
                metric: _interval(contrast["metrics"][metric])
                for metric in ("macro_f1", "moderate_f1", "severe_rate")
            },
        }
    return public


def _interval(item):
    result = {field: _number(item[field]) for field in ("point", "lower_95", "upper_95")}
    result["valid_resamples"] = _number(item["valid_resamples"], integer=True)
    if (
        any(result[field] is None for field in ("point", "lower_95", "upper_95"))
        or result["lower_95"] > result["upper_95"]
    ):
        raise ValueError("Invalid bootstrap interval")
    return result


def _transitions(source, assessed):
    fields = ("corrected", "worsened", "both_correct", "both_wrong")
    public = {}
    for name in MODELS[:3]:
        item = source["transitions"][f"vs_{name}"]
        result = {field: _number(item[field], integer=True) for field in fields}
        result["by_level"] = {
            level: {
                field: _number(item["by_level"][level][field], integer=True) for field in fields
            }
            for level in LEVELS
        }
        if sum(result[field] for field in fields) != assessed or any(
            sum(result["by_level"][level][field] for level in LEVELS) != result[field]
            for field in fields
        ):
            raise ValueError("Error-transition population differs")
        public[f"vs_{name}"] = result
    return public


def _plots(public: dict, figures: Path):
    metrics = {name: public["models"][name]["metrics"] for name in MODELS}
    labels = [NAMES[name].replace(" ", "\n", 1) for name in MODELS]
    files = [
        "probability_mean_comparison.png",
        "probability_mean_confusion.png",
        "probability_mean_bootstrap.png",
    ]
    with plt.rc_context({"font.size": 10, "figure.dpi": 140, "savefig.dpi": 160}):
        fig, ax = plt.subplots(figsize=(9.3, 4.9))
        bars = ax.bar(
            range(4),
            [metrics[name]["conditional"]["macro_f1"] for name in MODELS],
            color=COLORS,
            width=0.65,
        )
        ax.bar_label(bars, fmt="%.3f", padding=4)
        ax.set(
            xticks=range(4),
            xticklabels=labels,
            ylim=(0, 1),
            ylabel="Macro-F1",
            title="Четыре варианта на одинаковых обработанных уровнях",
        )
        ax.spines[["top", "right"]].set_visible(False)
        fig.text(
            0.5,
            0.015,
            "Ранее использованная validation · post hoc · независимое подтверждение отсутствует",
            ha="center",
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(figures / files[0])
        plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(10.4, 9.0))
        for axis, name in zip(axes.flat, MODELS):
            matrix = np.asarray(metrics[name]["conditional"]["confusion_matrix"])
            normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
            axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
            for i in range(3):
                for j in range(3):
                    axis.text(
                        j,
                        i,
                        f"{matrix[i, j]}\n{normalized[i, j]:.0%}",
                        ha="center",
                        va="center",
                        color="white" if normalized[i, j] > 0.55 else "#17212b",
                    )
            axis.set(
                xticks=range(3),
                yticks=range(3),
                xticklabels=["Норма /\nлёгкий", "Умеренный", "Тяжёлый"],
                yticklabels=["Норма / лёгкий", "Умеренный", "Тяжёлый"],
                xlabel="Предсказание",
                ylabel="Разметка RSNA",
                title=NAMES[name],
            )
        fig.suptitle("Ошибки на общей обработанной части validation; отказы учтены отдельно")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(figures / files[1])
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(9.4, 4.3))
        ax.axvline(0, color="#87919a", linestyle="--", linewidth=1)
        for i, name in enumerate(MODELS[:3]):
            interval = public["bootstrap"]["contrasts"][f"mean_minus_{name}"]["metrics"]["macro_f1"]
            y = 2 - i
            ax.hlines(y, interval["lower_95"], interval["upper_95"], color=COLORS[-1], linewidth=2)
            ax.vlines(
                [interval["lower_95"], interval["upper_95"]], y - 0.06, y + 0.06, color=COLORS[-1]
            )
            ax.plot(interval["point"], y, "o", color=COLORS[-1])
        ax.set(
            yticks=[2, 1, 0],
            yticklabels=[f"Среднее − {NAMES[name].lower()}" for name in MODELS[:3]],
            ylim=(-0.5, 2.5),
            xlabel="Разница macro-F1 · точка и парный 95% bootstrap-интервал",
            title="Различия на тех же пациентах; анализ post hoc",
        )
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(figures / files[2])
        plt.close(fig)
    return files


def draw_probability_mean_report(
    experiment: str | Path,
    *,
    document: str | Path = "docs/probability-mean.md",
    figures: str | Path = "docs/images",
) -> dict:
    """Verify the private result, then publish aggregate-only notes and figures."""
    root, document, figures = Path(experiment), Path(document), Path(figures)
    source = json.loads((root / "summary.json").read_text())
    if (
        not isinstance(source, dict)
        or source.get("schema_version") != 1
        or source.get("experiment") != "probability_mean_v3"
        or any(
            source.get(key) is not False
            for key in ("training_performed", "test_evaluated", "new_version_independently_tested")
        )
    ):
        raise ValueError(
            "Require the fixed post-hoc development comparison without training or test access"
        )
    protocol = source["protocol"]
    if (
        not isinstance(protocol, dict)
        or protocol.get("weights") != {"sagittal": 0.5, "axial": 0.5}
        or protocol.get("seeds") != [42, 2024, 3407]
        or protocol.get("classes") != list(CLASSES)
        or protocol.get("levels") != list(LEVELS)
        or protocol.get("weight_or_threshold_tuning") is not False
    ):
        raise ValueError(
            "The report requires fixed 0.5/0.5 probabilities and all three original seeds"
        )
    configuration = source["configuration"]
    if not isinstance(configuration, dict):
        raise ValueError("Missing typed report provenance")
    for key in ("protocol_sha256", "implementation_sha256"):
        if not isinstance(configuration.get(key), str) or not _SHA.fullmatch(configuration[key]):
            raise ValueError("Missing protocol or implementation checksum")
    inputs = configuration.get("inputs_sha256")
    if (
        not isinstance(inputs, dict)
        or not inputs
        or any(not isinstance(value, str) or not _SHA.fullmatch(value) for value in inputs.values())
    ):
        raise ValueError("Missing input prediction checksums")
    prediction_path = root / "predictions.csv"
    digest = _sha(prediction_path)
    if digest != source["models"]["probability_mean"].get("predictions_sha256"):
        raise ValueError("New probability-mean prediction checksum mismatch")
    rows = _read_predictions(prediction_path)
    if (source.get("n_studies"), source.get("n_targets"), source.get("n_assessed")) != (
        297,
        594,
        582,
    ):
        raise ValueError("Expected the fixed 297-study, 594-target development cohort")
    studies = {row["study_id"] for row in rows}
    if (
        len(rows) != 594
        or len(studies) != 297
        or any(
            {row["level"] for row in rows if row["study_id"] == study} != set(LEVELS)
            for study in studies
        )
    ):
        raise ValueError("New predictions do not retain both levels of every requested study")
    metrics = {name: _metric_snapshot(source["models"][name]["metrics"]) for name in MODELS}
    computed = _metric_snapshot(evaluate_predictions(rows))
    _same_tree(metrics["probability_mean"], computed)
    for metric in metrics.values():
        if (
            metric["n_requested"],
            metric["n_labeled"],
            metric["n_assessed"],
            metric["n_not_assessed"],
        ) != (594, 594, 582, 12) or metric["class_counts"] != computed["class_counts"]:
            raise ValueError("Variants must use one common reference population and assessed count")
        for level in LEVELS:
            a, b = metric["by_level"][level], computed["by_level"][level]
            if any(
                a[key] != b[key]
                for key in (
                    "n_requested",
                    "n_labeled",
                    "n_assessed",
                    "n_not_assessed",
                    "class_counts",
                )
            ):
                raise ValueError("Per-level coverage/reference populations differ")
    public = {
        "schema_version": 1,
        "experiment": "probability_mean_v3",
        "scope": "Post-hoc comparison on previously used validation; no independent confirmation",
        "protocol": {
            "weights": {"sagittal": 0.5, "axial": 0.5},
            "seeds": [42, 2024, 3407],
            "classes": list(CLASSES),
            "levels": list(LEVELS),
            "operation": "Mean probabilities across three seeds per modality, then fixed 0.5/0.5 mean, then argmax",
            "post_hoc": True,
        },
        "provenance": {
            "private_summary_sha256": _sha(root / "summary.json"),
            "protocol_sha256": configuration["protocol_sha256"],
            "implementation_sha256": configuration["implementation_sha256"],
            "predictions_sha256": digest,
        },
        "models": {name: {"metrics": metrics[name]} for name in MODELS},
        "bootstrap": _intervals(source),
        "transitions": _transitions(source, 582),
        "n_studies": 297,
        "n_targets": 594,
        "n_assessed": 582,
        "n_not_assessed": 12,
        "training_performed": False,
        "test_evaluated": False,
        "new_version_independently_tested": False,
    }
    for name in MODELS:
        intervals = public["bootstrap"]["models"][name]
        for field, expected in (
            ("macro_f1", metrics[name]["conditional"]["macro_f1"]),
            ("moderate_f1", metrics[name]["conditional"]["f1"]["moderate"]),
            ("severe_rate", metrics[name]["severe"]["detection_rate_all_requested"]),
        ):
            if not math.isclose(intervals[field]["point"], expected, rel_tol=0, abs_tol=1e-12):
                raise ValueError("Bootstrap point differs from reported classification metrics")
    figures.mkdir(parents=True, exist_ok=True)
    document.parent.mkdir(parents=True, exist_ok=True)
    files = _plots(public, figures)
    links = [Path(os.path.relpath(figures / name, document.parent)).as_posix() for name in files]
    table = []
    severe_rows = []
    for name in MODELS:
        metric, severe = metrics[name], metrics[name]["severe"]
        table.append(
            f"| {NAMES[name]} | {metric['conditional']['macro_f1']:.3f} | {metric['conditional']['f1']['moderate']:.3f} | {severe['correctly_predicted_severe']} / {severe['n_total']} | {metric['n_assessed']} / {metric['n_requested']} |"
        )
        for level in LEVELS:
            s = metric["by_level"][level]["severe"]
            severe_rows.append(
                f"| {NAMES[name]} | {level} | {s['predicted_normal_mild']} | {s['predicted_moderate']} | {s['not_assessed']} | {s['correctly_predicted_severe']} / {s['n_total']} |"
            )
    contrast = public["bootstrap"]["contrasts"]["mean_minus_sagittal"]["metrics"]["macro_f1"]
    transition_text = []
    for name in MODELS[:3]:
        item = public["transitions"][f"vs_{name}"]
        transition_text.append(f"| {NAMES[name]} | {item['corrected']} | {item['worsened']} |")
    text = """# Среднее вероятностей двух моделей

Я проверил простой способ совместить сагиттальную и аксиальную модели: усреднить их готовые вероятности с равными весами. Новую сеть не обучал. Сначала усреднил ответы трёх seed внутри каждой модальности, затем посчитал `p_mean(c) = 0.5 × p_sag(c) + 0.5 × p_axial(c)` и выбрал класс с наибольшей вероятностью. Усредняю вероятности, а не номера классов или logits. Веса и пороги по результатам не подбирал.

```mermaid
flowchart LR
    S["Сагиттальные модели · 3 seed"] --> PS["Среднее вероятностей p_sag"]
    A["Аксиальные модели · 3 seed"] --> PA["Среднее вероятностей p_axial"]
    PS --> M["0.5 × p_sag + 0.5 × p_axial"]
    PA --> M
    M --> C["Класс с наибольшей вероятностью"]
```

Это отличается от прежнего fusion: тот объединяет признаки и обучает собственный классификатор и внимание к срезам. Здесь использую уже сохранённые ответы двух отдельных моделей.

Эта проверка появилась после разбора отдельной ошибки fusion. Поэтому считаю её анализом post hoc на ранее использованной validation, а не независимым подтверждением улучшения. Новых тестовых пациентов не открывал.

## Результаты на общей выборке

Сравниваю четыре варианта на тех же 297 пациентах и 594 уровнях L4–L5 и L5–S1. Обработаны 582 уровня; 12 отказов сохраняю без класса и без вероятностей. Ни один отказ не заменяю нормой. Macro-F1 считаю по общей обработанной части для фиксированных трёх классов. При подсчёте найденных тяжёлых случаев оставляю отказы в полном знаменателе.

| Вариант | Macro-F1 | F1 умеренного | Тяжёлые: найдены / всего | Обработано / запрошено |
| --- | ---: | ---: | ---: | ---: |
""" + "\n".join(table)
    text += f"\n\n![Macro-F1 четырёх вариантов]({links[0]})\n\n![Матрицы ошибок]({links[1]})\n"
    text += f"\nРазница macro-F1 среднего вероятностей и сагиттальной модели: **{contrast['point']:+.3f}**, парный 95% bootstrap-интервал **[{contrast['lower_95']:+.3f}; {contrast['upper_95']:+.3f}]**. "
    if contrast["lower_95"] <= 0 <= contrast["upper_95"]:
        text += "Интервал включает ноль: устойчивый прирост на этой выборке не установлен.\n"
    elif contrast["lower_95"] > 0:
        text += "Интервал выше нуля на этой development-выборке; перенос на новых пациентов ещё не проверен.\n"
    else:
        text += "На этой development-выборке среднее вероятностей уступает сагиттальной модели.\n"
    text += f"\n![Парные bootstrap-интервалы разницы macro-F1]({links[2]})\n"
    text += """
Bootstrap выполняю по пациентам, 2000 повторов с seed 2026. Оба уровня и отказы одного пациента попадают в повтор вместе. Интервалы относятся к фиксированным моделям и этой выборке. Они не учитывают неопределённость разметки, повторное обучение, перенос в другую клинику или выбор этого способа после просмотра результатов.

## Пропуски тяжёлого стеноза по уровням

В таблице ниже показано, во что превратились тяжёлые случаи по разметке RSNA. Последний столбец включает все запрошенные тяжёлые уровни, в том числе необработанные.

| Вариант | Уровень | → норма / лёгкий | → умеренный | Нет оценки | Найдено / всего тяжёлых |
| --- | --- | ---: | ---: | ---: | ---: |
""" + "\n".join(severe_rows)
    l5 = metrics["probability_mean"]["by_level"]["L5-S1"]
    l5_matrix = l5["conditional"]["confusion_matrix"]
    text += (
        f"\n\nНа L5–S1 среднее вероятностей правильно определило **{l5_matrix[1][1]} / "
        f"{l5['class_counts']['moderate']} умеренных** и **{l5['severe']['correctly_predicted_severe']} / "
        f"{l5['severe']['n_total']} тяжёлых** случаев. Преобладание нормы / лёгкой степени может скрывать "
        "эти ошибки при оценке общей accuracy. Редких классов на этом уровне мало, поэтому "
        "надёжную оценку чувствительности пока не делаю. Разметка RSNA служит исследовательским "
        "ориентиром, а не новым заключением врача.\n"
    )
    text += """

На обработанных уровнях среднее вероятностей меняет ошибки так:

| По сравнению с | Исправлено ошибок | Добавлено ошибок |
| --- | ---: | ---: |
""" + "\n".join(transition_text)
    text += """

Вероятности классов не откалиброваны; не трактую их как вероятность правильного клинического решения. Этот анализ помогает выбрать следующий эксперимент, но не подтверждает клиническую пригодность и не заменяет независимую проверку на новых пациентах.

Перед построением отчёта проверяю фиксированные веса, контрольную сумму нового CSV, корректность вероятностей и повторно считаю его метрики. Публичные [агрегированные результаты](probability-mean-results.json) не содержат снимков или идентификаторов пациентов. Старые результаты трёх моделей сохранены отдельно в [сравнении сагиттальных и аксиальных T2](multimodal.md).

## Как повторить

Использую сохранённые validation CSV исходных трёх ансамблей и [фиксированный протокол](../configs/probability-mean-v3.json). Повторное обучение и загрузка МРТ не нужны. Для нового расчёта выбираю ещё не существующий каталог внутри игнорируемого `reports/runs/`, чтобы сохранить предыдущие результаты:

```bash
ai-spine probability-mean --out reports/runs/probability-mean-v3-repeat
ai-spine report-probability-mean --experiment reports/runs/probability-mean-v3-repeat
```

Повторный арифметический расчёт проверяет воспроизводимость чисел. Он не создаёт новую независимую клиническую проверку.
"""
    public_path = document.with_name("probability-mean-results.json")
    public_path.write_text(
        json.dumps(public, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    document.write_text(text, encoding="utf-8")
    return public
