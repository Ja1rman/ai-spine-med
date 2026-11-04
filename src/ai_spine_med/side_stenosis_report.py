"""Aggregate-only reports plus a private DICOM handedness audit for v15."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydicom

from . import axial_features as af
from . import side_stenosis_data as data
from .assets import sha256_file, write_json
from .axial_localizer_crops import _physical_crop, canonical_to_source_xy


def audit_handedness(out):
    """Use actual selected ROI source headers, no severity/coordinate tables."""
    checks, examples = [], []
    paths = (
        ("train", Path("artifacts/localization-finetuning-v9/mixed/records")),
        ("validation", Path("reports/runs/localization-severity-v8/arms/mixed/records")),
    )
    seen = set()
    for split, directory in paths:
        example_count = 0
        for receipt_path in sorted(directory.glob("*.json")):
            receipt = json.loads(receipt_path.read_text())
            if receipt["dataset"] != "rsna":
                continue
            if "crops" not in receipt["provenance"]:
                if any(row["status"] == "ok" for row in receipt["targets"]):
                    raise ValueError("Assessed image receipt lacks crop provenance")
                continue
            for level in data.LEVELS:
                crop = receipt["provenance"]["crops"][level]
                if crop["status"] != "ok":
                    continue
                meta = crop["metadata"]["centers"][0]["channels"][1]
                source = (
                    Path("data/rsna/train_images")
                    / receipt["study_id"]
                    / meta["series_id"]
                    / meta["source_file"]
                )
                key = str(source)
                if key in seen:
                    continue
                seen.add(key)
                ds = pydicom.dcmread(source, stop_before_pixels=True)
                if sha256_file(source) != meta["source_sha256"]:
                    raise ValueError("ROI source image changed")
                orientation = np.asarray(ds.ImageOrientationPatient, dtype=np.float64)
                shape = (int(ds.Rows), int(ds.Columns))
                dummy, action = af._canonical_image(
                    np.zeros(shape), orientation[:3], orientation[3:]
                )
                center = (np.asarray(dummy.shape)[::-1] - 1) / 2
                a = canonical_to_source_xy(center, shape, action)
                b = canonical_to_source_xy(center + [1, 0], shape, action)
                direction = (b[0] - a[0]) * orientation[:3] + (b[1] - a[1]) * orientation[3:]
                direction /= np.linalg.norm(direction)
                if direction[0] < 0.8:
                    raise ValueError("Canonical ROI x does not point toward anatomical LEFT")
                checks.append(
                    {
                        "split": split,
                        "level": level,
                        "action": action,
                        "positive_x_lps_left_component": float(direction[0]),
                        "source_sha256": meta["source_sha256"],
                    }
                )
                if example_count < 2:
                    full = pydicom.dcmread(source).pixel_array.astype(np.float32)
                    full, _ = af._canonical_image(full, orientation[:3], orientation[3:])
                    spacing = np.asarray(ds.PixelSpacing, float)
                    if action.startswith("transpose=1"):
                        spacing = spacing[::-1]
                    patch, _ = _physical_crop(full, meta["canonical_xy"], spacing, 50)
                    examples.append((split, level, patch.numpy()))
                    example_count += 1
    result = {
        "actual_source_images_checked": len(checks),
        "by_split": dict(Counter(c["split"] for c in checks)),
        "orientation_actions": dict(Counter(c["action"] for c in checks)),
        "minimum_positive_canonical_x_lps_left": min(
            c["positive_x_lps_left_component"] for c in checks
        ),
        "definition": "DICOM LPS positive X is patient LEFT. Canonical image/ROI right half is patient LEFT; image left half is patient RIGHT.",
        "checks": checks,
    }
    write_json(out / "handedness-audit.json", result)
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
    for ax, (split, level, patch) in zip(axes.flat, examples):
        lo, hi = np.percentile(patch, [1, 99])
        ax.imshow(patch, cmap="gray", vmin=lo, vmax=hi)
        ax.axvline(111.5, color="#f3cd65", linewidth=1)
        ax.text(10, 18, "R", color="#f3cd65", fontsize=15, weight="bold")
        ax.text(200, 18, "L", color="#f3cd65", fontsize=15, weight="bold")
        ax.set_title(f"{split} | {level}")
        ax.axis("off")
    fig.suptitle("Actual predicted 50 mm ROIs: anatomical R/L from DICOM geometry")
    fig.savefig(out / "private-handedness-examples.png", dpi=150)
    plt.close(fig)
    return result


def report(out):
    out = Path(out)
    result = json.loads((out / "summary.json").read_text())
    doc = [
        "# Фораминальный стеноз и латеральные рецессусы по сторонам",
        "",
        "В этом эксперименте добавил две отдельные задачи RSNA на L4–L5 и L5–S1: степень фораминального сужения и степень стеноза латерального рецессуса, слева и справа. В каждой задаче три класса: норма/лёгкий, умеренный, тяжёлый. Воздействие на конкретный корешок здесь не оцениваю.",
        "",
        "Это первый эксперимент на доступных снимках. Для фораминальной задачи использую сагиттальные T2/STIR и исходный encoder SpineNet, а не T1. Для рецессусов — аксиальные T2 с прежней автоматической смешанной локализацией. Новый результат не заменяет основной центральный классификатор и пока не подключён к сайту.",
        "",
        "## Данные и изображения",
        "",
        "Сохранил прежнее разделение: 1382 исследования TRAIN и 297 validation. Тест не открывал. Использую все восемь целевых оценок исследования: две патологии × два уровня × две стороны. Нетипичные/необработанные области сохраняю как отказ, отсутствующую метку не заменяю нормой; прогноз без метки сохраняю, но точность по нему не считаю.",
        "",
        "Фораминальные головы получают 512 исходных признаков SpineNet из прежней автоматически найденной области из девяти сагиттальных слоёв. Одинаковые признаки идут в отдельные выходы слева и справа. Область была разработана для оценки центрального стеноза, поэтому полного охвата фораминальных зон она не гарантирует; T1 и отдельная локализация отверстий остаются следующим опытом.",
        "",
        "Аксиальный вход — прежняя область 50×50 мм и до трёх соседних центров с 2.5D-контекстом. Сеть ResNet18 заморожена. Из её карты layer4 сохраняю глобальное среднее и отдельные средние анатомических левой/правой половин. Это пространственный признак, не выделение отдельных рецессусов или корешков. Референсные координаты RSNA на validation для выбора области не использую.",
        "",
        "## Обучение",
        "",
        "Заранее зафиксировал две архитектуры: линейный классификатор и MLP с общим скрытым слоем 64, ReLU и dropout 0.2. У каждой четыре независимых трёхклассовых выхода для сочетаний уровня и стороны. Каждую архитектуру в обеих задачах обучил с seed 42, 2024, 3407: всего 12 запусков.",
        "",
        "Нормализацию и веса классов считаю только по TRAIN; веса — обратный корень частот внутри каждого выхода. AdamW, learning rate 0.001, weight decay 0.0001, batch 128, до 25 эпох, patience 5. Эпоху выбираю по общему validation macro-F1; при равенстве оставляю более раннюю. Для каждой архитектуры усредняю вероятности трёх запусков. Базовый контроль — наиболее частый класс TRAIN отдельно для каждого уровня и стороны, с тем же покрытием обработки.",
        "",
        "Это повторно используемая разработочная валидация, по ней же выбираются эпохи. Разницу моделей здесь нельзя выдавать за независимое подтверждение улучшения. Контрольные суммы данных, кэшей и кода сохранил до первого обучения в [протоколе](../configs/side-stenosis-v15.json).",
        "",
        "## Результаты на validation",
        "",
        "| Задача | Вариант | Macro-F1 | Тяжёлые: найдены / все | Тяжёлый → норма/лёгкий | Оценено / запрошено |",
        "|---|---|---:|---:|---:|---:|",
    ]
    names = {"foraminal": "Фораминальный", "subarticular": "Латеральный рецессус"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    matrices = []
    for ax, condition in zip(axes, data.CONDITIONS):
        r = result["conditions"][condition]
        models = {
            "Контроль TRAIN": r["majority_train"],
            **{a: r["architectures"][a]["ensemble"] for a in ("linear", "mlp")},
        }
        for name, model in models.items():
            m = model["overall"]
            doc.append(
                f"| {names[condition]} | {name} | {m['macro_f1']:.3f} | {m['severe_correct']} / {m['severe_total']} | {m['severe_to_normal']} | {m['assessed']} / {m['requested']} |"
            )
        xs = np.arange(3)
        scores = [m["overall"]["macro_f1"] for m in models.values()]
        bars = ax.bar(xs, scores, color=["#aeb7be", "#5b86a4", "#2c716c"])
        ax.set_xticks(xs, ["TRAIN majority", "Linear 3-seed", "MLP 3-seed"])
        ax.set_ylim(0, 1)
        ax.set_title(
            {
                "foraminal": "Foraminal | sagittal T2-like",
                "subarticular": "Lateral recess | axial T2",
            }[condition]
        )
        ax.set_ylabel("Validation macro-F1")
        ax.bar_label(bars, fmt="%.3f")
        for a in ("linear", "mlp"):
            matrices.append(
                (f"{condition} | {a}", r["architectures"][a]["ensemble"]["overall"]["confusion"])
            )
    Path("docs/images").mkdir(parents=True, exist_ok=True)
    fig.savefig("docs/images/side_stenosis_comparison.png", dpi=160)
    plt.close(fig)
    doc += [
        "",
        "![Сравнение двух новых задач](images/side_stenosis_comparison.png)",
        "",
        "На этом этапе линейный вариант для фораминального сужения дал macro-F1 0.502, MLP — 0.501. Для рецессусов MLP выше по общему F1: 0.541 против 0.512. При этом на L5–S1 линейная модель нашла 10 из 44 тяжёлых оценок рецессусов, MLP — 8 из 44. Поэтому улучшение общего F1 не считаю решением задачи тяжёлого стеноза на этом уровне.",
        "",
        "![Матрицы ошибок двух архитектур](images/side_stenosis_confusion.png)",
        "",
        "## Уровни и стороны",
        "",
    ]
    doc += [
        "| Задача | Уровень | Сторона | Вариант | Macro-F1 | Тяжёлые: найдены / все | Тяжёлый → норма/лёгкий | Метки 0 / 1 / 2 |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for condition in data.CONDITIONS:
        for architecture in ("linear", "mlp"):
            model = result["conditions"][condition]["architectures"][architecture]
            for key, m in model["ensemble"]["by_level_side"].items():
                level, side = key.split(":")
                counts = " / ".join(map(str, m["class_counts_known"]))
                doc.append(
                    f"| {names[condition]} | {level} | {'слева' if side == 'left' else 'справа'} | {architecture} | {m['macro_f1']:.3f} | {m['severe_correct']} / {m['severe_total']} | {m['severe_to_normal']} | {counts} |"
                )
    doc += [
        "",
        "## Все запуски",
        "",
        "| Задача | Архитектура | Seed | Эпоха | Macro-F1 |",
        "|---|---|---:|---:|---:|",
    ]
    for condition in data.CONDITIONS:
        for architecture in ("linear", "mlp"):
            model = result["conditions"][condition]["architectures"][architecture]
            for run in model["seeds"]:
                doc.append(
                    f"| {names[condition]} | {architecture} | {run['seed']} | {run['best_epoch']} | {run['metrics']['overall']['macro_f1']:.3f} |"
                )
            doc.append(
                f"| {names[condition]} | {architecture}, среднее ± SD | — | — | {model['mean_seed_macro_f1']:.3f} ± {model['std_seed_macro_f1']:.3f} |"
            )
    audit = audit_handedness(out)
    doc += [
        "",
        "## Проверка сторон",
        "",
        f"Проверил геометрию {audit['actual_source_images_checked']} фактически использованных исходных DICOM (TRAIN {audit['by_split'].get('train', 0)}, validation {audit['by_split'].get('validation', 0)}). После штатной канонизации и физического crop положительное направление оси X массива соответствует анатомической левой стороне: минимальная LPS-X компонента {audit['minimum_positive_canonical_x_lps_left']:.3f}. Поэтому правая половина изображения относится к левой стороне пациента. Unit-тесты отдельно проверяют транспонирование, отражения и выбор пространственных половин. Это техническая проверка направлений, не врачебная проверка локализации рецессусов.",
        "",
        "Четыре реальных примера с отметками R/L и подробный отчёт проверки оставил локально в `reports/runs/side-stenosis-v15/`; медицинские изображения и индивидуальные прогнозы в Git не включаю.",
        "",
        "## Ограничения и следующие проверки",
        "",
        "Три класса RSNA не дают полноценную Lee 0–3: норму и лёгкую степень здесь не разделить. Градации и нейросетевые признаки не являются измерением компрессии конкретного корешка. Для оценки фораминальной задачи по стандартному входу следующим отдельным опытом нужны сагиттальные T1 и локализация левого/правого отверстия. Для рецессусов нужна врачебная проверка охвата анатомии текущими 50-мм областями, особенно на L5–S1.",
        "",
        "Исходные бинарные фораминальные выходы SpineNet не сравниваю по трёхклассовому macro-F1: у них другой набор классов и неподтверждённое соответствие порогов RSNA. Их нельзя автоматически превратить в оценки умеренной и тяжёлой степени.",
        "",
        "## Повторение",
        "",
        "```bash",
        "python -m ai_spine_med.side_stenosis prepare",
        "python -m ai_spine_med.side_stenosis train",
        "python -m ai_spine_med.side_stenosis report",
        "```",
        "",
        "Подготовка требует прежних локальных image-only кэшей: `artifacts/features-v1` и `artifacts/roi-layer4-v11/mixed`. Их нельзя заменить координатами или вручную выбранными validation-срезами. Веса новых голов, TRAIN-нормализаторы, вероятности всех оценок и контрольные суммы находятся в локальном `reports/runs/side-stenosis-v15/`.",
    ]
    Path("docs/side-stenosis.md").write_text("\n".join(doc) + "\n")
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    for ax, (name, matrix) in zip(axes.flat, matrices):
        values = np.asarray(matrix)
        ax.imshow(values, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(
                    j,
                    i,
                    str(values[i, j]),
                    ha="center",
                    va="center",
                    color="white" if values[i, j] > values.max() / 2 else "black",
                )
        ax.set_title(name)
        ax.set_xticks(range(3), ["Normal/mild", "Moderate", "Severe"], rotation=15)
        ax.set_yticks(range(3), ["Normal/mild", "Moderate", "Severe"])
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Reference RSNA")
    fig.savefig("docs/images/side_stenosis_confusion.png", dpi=160)
    plt.close(fig)
    write_json(Path("docs/side-stenosis-results.json"), result)
    write_json(
        out / "report-receipt.json",
        {
            "summary_sha256": sha256_file(out / "summary.json"),
            "docs_sha256": sha256_file(Path("docs/side-stenosis.md")),
            "handedness_sources_checked": audit["actual_source_images_checked"],
        },
    )
    return result
