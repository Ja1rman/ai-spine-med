"""Public aggregate figures and first-person notes for the paired MRI experiment."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VARIANTS = ("sagittal", "axial", "fusion")
NAMES = {"sagittal": "Сагиттальные", "axial": "Аксиальные", "fusion": "Вместе"}
COLORS = ("#4979a5", "#d79138", "#498568")


def draw_multimodal_report(
    experiment: str | Path,
    *,
    document: str | Path = "docs/multimodal.md",
    figures: str | Path = "docs/images",
) -> dict:
    root, document, figures = Path(experiment), Path(document), Path(figures)
    source = json.loads((root / "summary.json").read_text())
    if source.get("test_evaluated") is not False:
        raise ValueError("This report accepts development experiments without test access only")
    figures.mkdir(parents=True, exist_ok=True)
    document.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "figure.dpi": 140, "savefig.dpi": 160})
    public = {
        "schema_version": 1,
        "experiment": "sagittal_axial_fusion_v3",
        "scope": "Development: original TRAIN grouped CV and previously used validation; no new independent test",
        "test_evaluated": False,
        "new_version_independently_tested": False,
        "provenance": {
            key: source["configuration"][key]
            for key in (
                "protocol_sha256",
                "implementation_sha256",
                "axial_metadata_sha256",
                "axial_output_inventory_sha256",
                "folds_sha256",
            )
        },
        "variants": {},
        "availability": source["availability"],
        "bootstrap": source["bootstrap"],
    }
    metrics, cv_scores, val_scores = {}, [], []
    rows = []
    seed_rows = []
    for name in VARIANTS:
        result = source["variants"][name]
        validation = result["validation"]
        metric = validation["ensemble"]["metrics"]
        metrics[name] = metric
        cv_score = result.get("mean_fold_ensemble_macro_f1")
        if cv_score is None:
            cv_score = result["cv"]["mean_fold_ensemble_macro_f1"]
        cv_scores.append(float(cv_score))
        val_scores.append(metric["conditional"]["macro_f1"])
        seeds = [
            {key: item[key] for key in ("seed", "best_epoch", "macro_f1")}
            for item in validation["runs"]
        ]
        public["variants"][name] = {
            "cv_mean_fold_ensemble_macro_f1": cv_score,
            "validation": {"runs": seeds, "ensemble_metrics": metric},
        }
        severe = metric["severe"]
        rows.append(
            f"| {NAMES[name]} | {cv_score:.3f} | {metric['conditional']['macro_f1']:.3f} "
            f"| {metric['conditional']['f1']['moderate']:.3f} "
            f"| {severe['correctly_predicted_severe']} / {severe['n_total']} "
            f"| {metric['n_assessed']} / {metric['n_labeled']} |"
        )
        seed_rows.extend(
            f"| {NAMES[name]} | {item['seed']} | {item['best_epoch']} | {item['macro_f1']:.3f} |"
            for item in seeds
        )
    curve_audit = {}
    for name in VARIANTS:
        histories = [
            json.loads(path.read_text())
            for path in sorted((root / "cv" / name).glob("fold-*/seed-*/history.json"))
        ]
        if len(histories) != 9 or any(not history for history in histories):
            raise ValueError("Require all nine CV training histories for each variant")
        curve_audit[name] = {
            "n_cv_runs": 9,
            "median_weighted_train_loss_first": float(
                np.median([h[0]["training_loss"] for h in histories])
            ),
            "median_weighted_train_loss_last": float(
                np.median([h[-1]["training_loss"] for h in histories])
            ),
            "median_unweighted_held_loss_first": float(
                np.median([h[0]["heldout_loss_unweighted"] for h in histories])
            ),
            "median_unweighted_held_loss_last": float(
                np.median([h[-1]["heldout_loss_unweighted"] for h in histories])
            ),
            "n_held_loss_last_above_first": sum(
                h[-1]["heldout_loss_unweighted"] > h[0]["heldout_loss_unweighted"]
                for h in histories
            ),
            "scope": "Descriptive post-hoc curve audit. Last epoch may follow selected best checkpoint. Train loss is weighted, held loss unweighted; absolute loss levels are not directly comparable.",
        }
    public["learning_curve_audit"] = curve_audit
    prediction_tables = {}
    for name in ("sagittal", "fusion"):
        prediction_path = root / "validation" / name / "ensemble" / "predictions.csv"
        expected = source["variants"][name]["validation"]["ensemble"]["predictions_sha256"]
        if hashlib.sha256(prediction_path.read_bytes()).hexdigest() != expected:
            raise ValueError("Validation prediction checksum differs from the experiment")
        with prediction_path.open(newline="") as handle:
            prediction_tables[name] = {
                (row["study_id"], row["level"]): row for row in csv.DictReader(handle)
            }
    if prediction_tables["sagittal"].keys() != prediction_tables["fusion"].keys():
        raise ValueError("Validation error audit requires matching study/level keys")
    transitions = {"corrected": 0, "worsened": 0, "both_correct": 0, "both_wrong": 0}
    for key, sag in prediction_tables["sagittal"].items():
        fusion = prediction_tables["fusion"][key]
        if (sag["label"], sag["status"]) != (fusion["label"], fusion["status"]):
            raise ValueError("Validation error audit labels/status differ")
        if sag["status"] != "ok":
            continue
        correct_sag, correct_fusion = (
            sag["prediction"] == sag["label"],
            fusion["prediction"] == fusion["label"],
        )
        group = (
            "both_correct"
            if correct_sag and correct_fusion
            else "corrected"
            if correct_fusion
            else "worsened"
            if correct_sag
            else "both_wrong"
        )
        transitions[group] += 1
    public["validation_error_transitions_fusion_vs_sagittal"] = transitions
    fig, ax = plt.subplots(figsize=(9.4, 5.1))
    positions = np.arange(3)
    bars1 = ax.bar(
        positions - 0.18,
        cv_scores,
        0.34,
        color=COLORS,
        alpha=0.5,
        label="TRAIN: среднее 3 CV-фолдов",
    )
    bars2 = ax.bar(
        positions + 0.18, val_scores, 0.34, color=COLORS, label="Validation: ансамбль 3 seed"
    )
    ax.bar_label(bars1, fmt="%.3f", padding=3)
    ax.bar_label(bars2, fmt="%.3f", padding=3)
    ax.set(
        xticks=positions,
        xticklabels=[NAMES[v] for v in VARIANTS],
        ylim=(0, 1),
        ylabel="Macro-F1",
        title="Три варианта на одинаковых пациентах и уровнях",
    )
    ax.legend(loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5,
        0.01,
        "CV использовалась для выбора эпох; validation уже участвовала в развитии проекта. Это development-оценки.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(figures / "multimodal_comparison.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13.3, 4.3), sharey=True)
    for axis, name in zip(axes, VARIANTS):
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
            title=NAMES[name],
            xlabel="Предсказание",
        )
    axes[0].set_ylabel("Разметка RSNA")
    fig.suptitle("Validation: ошибки ансамблей на общих обработанных уровнях")
    fig.tight_layout()
    fig.savefig(figures / "multimodal_confusion.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11.1, 4.8))
    for axis, level in zip(axes, ("L4-L5", "L5-S1")):
        scores = [
            metrics[v]["by_level"][level]["severe"]["detection_rate_all_requested"] or 0
            for v in VARIANTS
        ]
        bars = axis.bar(range(3), scores, color=COLORS)
        for bar, name in zip(bars, VARIANTS):
            severe = metrics[name]["by_level"][level]["severe"]
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.025,
                f"{severe['correctly_predicted_severe']} / {severe['n_total']}",
                ha="center",
            )
        axis.set(
            xticks=range(3),
            xticklabels=[NAMES[v] for v in VARIANTS],
            ylim=(0, 1.12),
            title=level,
            ylabel="Доля тяжёлых случаев, определённых как тяжёлые",
        )
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Validation: тяжёлый стеноз, включая отказы в знаменателе")
    fig.tight_layout()
    fig.savefig(figures / "multimodal_severe.png")
    plt.close(fig)
    contrasts = source["bootstrap"]["contrasts"]
    labels = ("Аксиальные − сагиттальные", "Вместе − сагиттальные")
    keys = ("axial_minus_sagittal", "fusion_minus_sagittal")
    intervals = [contrasts[key]["metrics"]["macro_f1"] for key in keys]
    points = np.asarray([item["point"] for item in intervals])
    lower = np.asarray([item["lower_95"] for item in intervals])
    upper = np.asarray([item["upper_95"] for item in intervals])
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    ax.axvline(0, color="#87919a", linestyle="--", linewidth=1)
    ax.errorbar(
        points, [1, 0], xerr=[points - lower, upper - points], fmt="o", color="#376d5e", capsize=5
    )
    ax.set(
        yticks=[1, 0],
        yticklabels=labels,
        ylim=(-0.6, 1.6),
        xlabel="Разница macro-F1: точка и парный 95% bootstrap-интервал",
        title="Validation: насколько устойчиво отличается результат",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures / "multimodal_bootstrap.png")
    plt.close(fig)
    public_path = document.with_name("multimodal-results.json")
    public_path.write_text(json.dumps(public, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    text = """# Сагиттальные и аксиальные T2

Я проверил, помогает ли аксиальная T2-серия определять степень центрального стеноза на L4–L5 и L5–S1. Для этого обучил три варианта на одинаковых пациентах и доступных уровнях: по сагиттальным снимкам, по аксиальным и по их сочетанию.

Старое сравнение на тесте сохранил без изменений. Здесь использую только прежние train и validation. Validation уже использовалась в предыдущих экспериментах, поэтому эти числа считаю результатами разработки. Для независимого подтверждения нужны новые пациенты.

## Как устроены варианты

```mermaid
flowchart TB
    S["Сагиттальная T2/STIR"] --> SN["Замороженный SpineNet<br/>область каждого уровня, 512 признаков"]
    A["Все аксиальные T2-серии пациента"] --> RN["Замороженный ResNet18 ImageNet<br/>512 признаков на срез"]
    RN --> AT["Обучаемое внимание<br/>отдельно для L4–L5 и L5–S1"]
    SN --> SH["Сагиттальный классификатор"]
    AT --> AH["Аксиальный классификатор"]
    SN --> F["Объединение признаков"]
    AT --> F
    F --> FH["Общий классификатор"]
    SH --> C["Норма / лёгкий, умеренный, тяжёлый"]
    AH --> C
    FH --> C
```

Аксиальному варианту сагиттальные снимки не нужны. Он получает все аксиальные срезы пациента и их относительное положение по оси «выше–ниже» внутри совместимой системы координат. Обучаемое внимание собирает признаки для каждого из двух уровней. Координаты из разметки RSNA для выбора срезов не использую, включая обучение. Внимание пока не проверено врачом и само по себе не доказывает правильную анатомическую локализацию. Наличие обработанной аксиальной серии также не подтверждает, что в её поле зрения действительно присутствуют оба нужных уровня: это отдельно надо проверить.

Для аксиальных снимков использую соседний, текущий и следующий срез как три канала; соседей беру только из той же серии, при близкой ориентации и расстоянии. Интенсивность ограничиваю 1-м и 99-м процентилями внутри серии. Сохраняю всё поле зрения, привожу размер к 224×224 и применяю нормализацию ImageNet. При разных системах координат не считаю серии совмещёнными: нормирую положение внутри каждой группы и объединяю признаки как набор. Группы упорядочиваю по идентификатору серии, срезы внутри группы — по физическому положению. Если срезов больше 96, беру 96 равномерно по этому фиксированному порядку. Не подбираю их по тяжести или координатам врача.

Это первое лёгкое сравнение: обе сети извлечения признаков заморожены, обучаются внимание и небольшие классификаторы. У SpineNet медицинское предобучение, а у ResNet18 — ImageNet; их архитектуры тоже различаются. Результат этого эксперимента не показывает предельные возможности аксиальных МРТ.

## Как сравниваю

Сохраняю исходное разбиение пациентов. Обе области и все серии одного пациента остаются вместе. Для трёх вариантов использую одинаковые обработанные пары «пациент–уровень», а недоступные области отдельно учитываю как отказы. Основной macro-F1 вычисляю по общей обработанной части; наличие обеих модальностей и полное покрытие проверяю отдельно.

На исходном train провожу три CV-разбиения и три запуска каждого варианта: seed 42, 2024 и 3407. Нормализацию признаков и веса классов считаю только по обучающей части каждого фолда. По CV выбираю эпоху, затем обучаю финальную модель на всём train фиксированное число эпох и один раз оцениваю validation. Ансамбль — простое среднее вероятностей трёх seed. Веса ансамбля и пороги по результатам не подбираю.

CV также используется для выбора эпох, поэтому её оценки тоже относятся к разработке. Не называю их независимым тестом.

## Полученные результаты

| Вариант | CV: средний macro-F1 ансамбля | Validation: macro-F1 ансамбля | F1 умеренного | Тяжёлые: найдены / всего | Обработано / запрошено |
| --- | ---: | ---: | ---: | ---: | ---: |
"""
    text += "\n".join(rows)
    train_availability = source["availability"]["train"]
    val_availability = source["availability"]["validation"]
    text += (
        f"\n\nОбщие обработанные уровни: train **{train_availability['paired_assessed']} / "
        f"{train_availability['n_requested']}**, validation **{val_availability['paired_assessed']} / "
        f"{val_availability['n_requested']}**. В таблице три варианта сравниваются на одном и том же наборе. "
        "Отказы не считаю нормой; для тяжёлых случаев оставляю их в полном знаменателе.\n"
    )
    interval = intervals[1]
    text += (
        f"\n\nНа validation объединение изменило macro-F1 относительно сагиттального варианта "
        f"на **{interval['point']:+.3f}**. Парный 95% bootstrap-интервал разницы: "
        f"**[{interval['lower_95']:+.3f}; {interval['upper_95']:+.3f}]**. "
    )
    if interval["lower_95"] <= 0 <= interval["upper_95"]:
        text += "Интервал включает ноль, поэтому устойчивый прирост по этим данным пока не установлен.\n"
    elif interval["lower_95"] > 0:
        text += "На этой development-выборке интервал выше нуля; перенос результата на новых пациентов ещё не проверен.\n"
    else:
        text += "На этой development-выборке объединённый вариант оказался хуже; это не доказывает бесполезность аксиальных снимков для других моделей.\n"
    text += """

![Сравнение трёх вариантов](images/multimodal_comparison.png)

![Матрицы ошибок](images/multimodal_confusion.png)

![Тяжёлый стеноз по уровням](images/multimodal_severe.png)

![Неопределённость разницы между вариантами](images/multimodal_bootstrap.png)

Каждый отдельный запуск:

| Вариант | Seed | Эпохи финального обучения | Validation macro-F1 |
| --- | ---: | ---: | ---: |
"""
    text += "\n".join(seed_rows)
    text += (
        "\n\nОбъединение исправило **"
        + str(transitions["corrected"])
        + "** ошибок сагиттального варианта, "
        "но добавило **" + str(transitions["worsened"]) + "** новых. "
        "Это описание тех же обработанных уровней, не дополнительная оценка точности.\n"
    )
    l5_severe = [metrics[name]["by_level"]["L5-S1"]["severe"] for name in VARIANTS]
    if all(item["n_total"] == 3 and item["correctly_predicted_severe"] == 0 for item in l5_severe):
        text += "\nНа L5–S1 все три ансамбля пропустили три тяжёлых случая из трёх; примеров слишком мало для надёжной оценки чувствительности, но проблему оставляю видимой.\n"
    text += """

Неопределённость разницы оцениваю парным bootstrap по пациентам, 2000 повторов. Оба уровня и отказы пациента попадают в повтор вместе. Интервалы относятся к этим фиксированным моделям и этим пациентам; они не учитывают неопределённость разметки и перенос на другую клинику. Числа и интервалы сохранены в [агрегированных результатах](multimodal-results.json).

Вероятности классов не откалиброваны; не трактую их как вероятность правильного клинического решения.

По кривым обучения вижу признаки переобучения аксиального и общего вариантов: потери на обучающей части снижаются, а на отложенных CV-пациентах к остановке растут. Сохранены более ранние checkpoints с лучшим macro-F1, а не последние эпохи. Веса классов применяются только к training loss, поэтому абсолютные значения training и held loss напрямую не сравниваю.

Моя рабочая гипотеза: общие ImageNet-признаки по всему полю зрения дают слишком слабый сигнал о нужной области и уровне. Это ещё не установленная причина. Сначала хочу проверить с врачом выбранные срезы, затем решать, нужны ли локализация канала, более подходящее предобучение или дообучение аксиальной сети. Эти варианты в текущий эксперимент не подмешиваю.

## Как повторить

Официальный набор RSNA и доступ Kaggle нужны локально. Новые файлы не содержат тестовых пациентов.

```bash
mkdir -p .cache/axial-v3
curl --fail --location https://download.pytorch.org/models/resnet18-f37072fd.pth \
  --output .cache/axial-v3/resnet18-f37072fd.pth
ai-spine prepare-axial
ai-spine download-axial
ai-spine extract-axial --device auto
ai-spine multimodal-experiment
ai-spine report-multimodal
ai-spine multimodal-review
```

Для извлечения аксиальных признаков использую официальные веса `resnet18-f37072fd.pth` из [PyTorch](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html). Файл должен лежать в `.cache/axial-v3/`; его контрольная сумма проверяется перед загрузкой. Исходный сагиттальный cache готовлю командами из основного README. Все checkpoint, индивидуальные предсказания и медицинские изображения остаются локально.

Полный заранее заданный протокол — [multimodal-v3.json](../configs/multimodal-v3.json). Команда `multimodal-review` создаёт приватную галерею из 24 примеров: шесть исправленных объединением ошибок, шесть новых ошибок, шесть общих ошибок и шесть правильных контрольных примеров. Это целевой разбор с видимыми метками и ответами, а не независимая врачебная оценка точности. Для проверки нумерации и степени стеноза врачу также нужна полная DICOM-серия: трёх выбранных срезов и сагиттальной вырезки может быть недостаточно.

Галерея находится локально в `reports/runs/multimodal-review-v3/index.html`, пустые поля для врача — в соседнем `doctor_review.csv`. При повторном запуске выбираю новый `--out`, чтобы сохранить уже введённые ответы. Изменение моделей не меняет ограничения лицензии исходного SpineNet.
"""
    document.write_text(text, encoding="utf-8")
    return public
