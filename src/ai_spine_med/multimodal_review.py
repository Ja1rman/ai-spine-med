"""Private, unblinded validation error review with frozen sagittal/axial images.

Attention weights select model inputs; they do not establish anatomical level,
causality or clinical validity. No model is trained or evaluated on TEST here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .assets import sha256_file, write_json
from .data import CLASS_NAMES, LEVELS
from .error_review import _private_output
from .multimodal import SEEDS, VARIANTS, load_paired_data

GRADES = ("норма / лёгкая степень", "умеренный стеноз", "тяжёлый стеноз")
NAMES = {"sagittal": "Сагиттальная", "axial": "Аксиальная", "fusion": "Объединённая"}
REASONS = {
    "fusion_corrected": "Объединённая модель исправила сагиттальную",
    "fusion_worsened": "Объединённая модель испортила верный ответ",
    "shared_error": "Ошибка сагиттальной и объединённой моделей",
    "correct_control": "Все три модели ответили верно",
    "additional_case": "Дополнительный случай после исчерпания квот",
}
CLINICIAN_FIELDS = (
    "level_correct",
    "axial_level_present",
    "canal_visible",
    "reference_agreement",
    "suggested_grade",
    "comment",
)


def _key(row):
    return str(row["study_id"]), str(row["series_id"]), row["level"]


def _path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Review cache paths must be relative")
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("Review cache path escapes its directory")
    return result


def _probabilities(row):
    values = np.asarray([float(row[f"p_{name}"]) for name in CLASS_NAMES])
    if (
        not np.isfinite(values).all()
        or np.any(values < 0)
        or not np.isclose(values.sum(), 1, rtol=0, atol=1e-6)
    ):
        raise ValueError("Invalid validation probability distribution")
    return values


def _read_predictions(path: Path, digest: str, expected: dict) -> dict:
    if sha256_file(path) != digest:
        raise ValueError("Validation prediction checksum differs from completed summary")
    result = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            row = dict(raw)
            key = _key(row)
            if row["split"] != "validation" or key not in expected or key in result:
                raise ValueError(
                    "Review accepts each fixed validation target once; TEST is forbidden"
                )
            row["label"] = None if row["label"] == "" else int(row["label"])
            reference = expected[key]
            if any(
                row[field] != reference[field] for field in ("split", "label", "status", "reason")
            ):
                raise ValueError("Validation target identity, reference or coverage differs")
            if row["status"] == "ok":
                values = _probabilities(row)
                row["prediction"] = int(row["prediction"])
                if row["prediction"] != int(values.argmax()):
                    raise ValueError("Validation category differs from probability argmax")
                row["probabilities"] = values.tolist()
            elif row["prediction"] or any(row[f"p_{name}"] for name in CLASS_NAMES):
                raise ValueError("Unassessed validation targets must not carry predictions")
            result[key] = row
    if result.keys() != expected.keys():
        raise ValueError("Validation prediction inventory is incomplete")
    return result


def _validate_attention(document: dict, data) -> dict:
    if document.get("seeds") != list(SEEDS):
        raise ValueError("Attention must contain the fixed three-seed average")
    expected = {
        study: i
        for i, study in enumerate(data.study_ids)
        if data.splits[i] == "validation" and bool(data.paired[i].any())
    }
    result = {}
    for record in document["records"]:
        study = record["study_id"]
        if study not in expected or study in result or record.get("split") != "validation":
            raise ValueError("Attention study identity or validation scope differs")
        i = expected[study]
        if record.get("selected_source_indices") != data.selected_indices[i].tolist():
            raise ValueError("Attention slice selection differs from its frozen cache")
        if set(record["levels"]) != set(LEVELS):
            raise ValueError("Attention target levels differ")
        for level, name in enumerate(LEVELS):
            item = record["levels"][name]
            if not bool(data.paired[i, level]):
                if item != {"status": "not_assessed", "weights": [], "top3": []}:
                    raise ValueError("Unassessed attention target unexpectedly carries weights")
                continue
            values = np.asarray(item["weights"], dtype=float)
            n = len(data.selected_indices[i])
            if (
                item.get("status") != "ok"
                or values.shape != (n,)
                or not np.isfinite(values).all()
                or np.any(values < 0)
                or not np.isclose(values.sum(), 1, rtol=0, atol=1e-6)
            ):
                raise ValueError("Invalid attention weights")
            top = np.argsort(-values, kind="stable")[:3]
            if len(item["top3"]) != len(top):
                raise ValueError("Attention top-three inventory differs")
            for saved, index in zip(item["top3"], top):
                if (
                    saved.get("bag_index") != int(index)
                    or saved.get("source_index") != int(data.selected_indices[i][index])
                    or saved.get("series_id") != str(data.selected_series[i][index])
                    or saved.get("instance") != int(data.selected_instances[i][index])
                    or not np.isclose(saved.get("weight", np.nan), values[index], rtol=0, atol=1e-7)
                ):
                    raise ValueError("Attention points to a different source slice")
        result[study] = record
    if result.keys() != expected.keys():
        raise ValueError("Attention validation inventory is incomplete")
    return result


def _load_inputs(experiment: Path, sag_features: Path, axial_features: Path):
    data = load_paired_data(sag_features, axial_features)
    summary_path = experiment / "summary.json"
    summary = json.loads(summary_path.read_text())
    provenance = summary["configuration"]
    if (
        summary.get("test_evaluated") is not False
        or summary.get("all_seeds_reported") is not True
        or summary.get("all_variants_share_paired_cohort") is not True
        or provenance.get("sagittal_feature_cache_sha256") != data.sagittal.file_sha256
        or provenance.get("axial_metadata_sha256") != data.axial_metadata_sha256
        or provenance.get("axial_output_inventory_sha256") != data.axial_output_sha256
    ):
        raise ValueError("Completed development experiment and frozen cache provenance differ")
    expected = {_key(row): row for row in data.targets if row["split"] == "validation"}
    predictions, hashes = {}, {"summary.json": sha256_file(summary_path)}
    for variant in VARIANTS:
        relative = f"validation/{variant}/ensemble/predictions.csv"
        digest = summary["variants"][variant]["validation"]["ensemble"]["predictions_sha256"]
        predictions[variant] = _read_predictions(experiment / relative, digest, expected)
        hashes[relative] = digest
    attention = {}
    for variant in ("axial", "fusion"):
        relative = f"validation/{variant}/attention.json"
        digest = summary["variants"][variant]["validation"]["attention_sha256"]
        if sha256_file(experiment / relative) != digest:
            raise ValueError("Attention checksum differs from completed experiment summary")
        attention[variant] = _validate_attention(
            json.loads((experiment / relative).read_text()), data
        )
        hashes[relative] = digest
    candidates = []
    for key, target in expected.items():
        if target["status"] == "ok":
            if not all(
                re.fullmatch(r"[0-9]+", target[field]) for field in ("study_id", "series_id")
            ):
                raise ValueError("Review source identifiers must be local numeric IDs")
            candidates.append({**target, "predictions": {v: predictions[v][key] for v in VARIANTS}})
    return data, candidates, attention, hashes


def select_multimodal_cases(candidates: list[dict], count: int = 24) -> list[dict]:
    """Fixed purposive quarters; select distinct validation study/level cases."""
    if type(count) is not int or count < 1:
        raise ValueError("Review count must be a positive integer")
    seen, prepared = set(), []
    for row in candidates:
        key = row["study_id"], row["level"]
        if row["split"] != "validation":
            raise ValueError("Multimodal review accepts validation only; TEST is forbidden")
        if (
            key in seen
            or row["status"] != "ok"
            or row["label"] not in (0, 1, 2)
            or row["level"] not in LEVELS
        ):
            raise ValueError("Require distinct assessed validation study/level cases")
        seen.add(key)
        prediction = {variant: row["predictions"][variant]["prediction"] for variant in VARIANTS}
        if any(value not in (0, 1, 2) for value in prediction.values()):
            raise ValueError("Review categories must be 0, 1 or 2")
        sag_ok, fusion_ok = (
            prediction["sagittal"] == row["label"],
            prediction["fusion"] == row["label"],
        )
        category = (
            "fusion_corrected"
            if not sag_ok and fusion_ok
            else "fusion_worsened"
            if sag_ok and not fusion_ok
            else "shared_error"
            if not sag_ok and not fusion_ok
            else "correct_control"
            if all(value == row["label"] for value in prediction.values())
            else "additional_case"
        )
        prepared.append({**row, "sampling_category": category})
    prepared.sort(
        key=lambda row: hashlib.sha256(f"multimodal-v3:2026:{_key(row)}".encode()).hexdigest()
    )
    selected, used = [], set()
    categories = tuple(REASONS)[:4]
    for i, category in enumerate(categories):
        quota = count // 4 + (i < count % 4)
        pool = [row for row in prepared if row["sampling_category"] == category][:quota]
        selected.extend({**row, "sampling_reason": category} for row in pool)
        used.update((row["study_id"], row["level"]) for row in pool)
    for row in prepared:
        if len(selected) >= count:
            break
        key = row["study_id"], row["level"]
        if key not in used:
            selected.append({**row, "sampling_reason": "additional_case"})
            used.add(key)
    selected.sort(key=lambda row: hashlib.sha256(f"gallery:{_key(row)}".encode()).hexdigest())
    for row in selected:
        row["case_id"] = (
            "case-mm-" + hashlib.sha256(f"private:{_key(row)}".encode()).hexdigest()[:14]
        )
    return selected


def _roi_image(item: dict, feature_dir: Path, data) -> tuple[np.ndarray, dict]:
    path = feature_dir / "records" / f"{item['study_id']}_{item['series_id']}.json"
    record = json.loads(path.read_text())
    stored = [row for row in record["targets"] if row["level"] == item["level"]]
    if (
        record.get("cache_id") != data.sagittal.metadata["cache_id"]
        or len(stored) != 1
        or any(
            stored[0][field] != item[field]
            for field in ("study_id", "series_id", "split", "label", "status")
        )
        or int(stored[0]["roi_index"]) != int(item["roi_index"])
        or record["arrays_path"] != item["roi_path"]
    ):
        raise ValueError("sagittal_roi_identity_mismatch")
    array_path = _path(feature_dir, item["roi_path"])
    if sha256_file(array_path) != record["arrays_sha256"]:
        raise ValueError("sagittal_roi_hash_mismatch")
    with np.load(array_path, allow_pickle=False) as arrays:
        index, feature_index = int(item["roi_index"]), item["feature_index"]
        roi = arrays["rois"][index]
        if (
            str(arrays["levels"][index]) != item["level"]
            or not np.array_equal(
                arrays["features"][index], data.sagittal.arrays["features"][feature_index]
            )
            or not np.array_equal(
                arrays["original_logits"][index],
                data.sagittal.arrays["original_logits"][feature_index],
            )
        ):
            raise ValueError("sagittal_roi_feature_alignment_mismatch")
    if roi.shape != (9, 112, 224) or not np.isfinite(roi).all() or np.ptp(roi[4]) <= 1e-12:
        raise ValueError("sagittal_roi_nonfinite_or_blank")
    return roi[4], {"record_sha256": sha256_file(path), "arrays_sha256": record["arrays_sha256"]}


def _axial_images(item: dict, root: Path, data_root: Path, data, attention: dict):
    from .axial_features import _implementation_signature, _study_inputs, _study_source

    if data.axial_metadata["config"]["implementation_sha256"] != _implementation_signature():
        raise ValueError("axial_preprocessing_implementation_changed")
    entry = next(
        row for row in data.axial_metadata["study_records"] if row["study_id"] == item["study_id"]
    )
    record_path = _path(root, entry["record_path"])
    if sha256_file(record_path) != entry["record_sha256"]:
        raise ValueError("axial_record_hash_mismatch")
    record = json.loads(record_path.read_text())
    source = record["source"]
    rows = []
    for series in source["series"]:
        if (
            not re.fullmatch(r"[0-9]+", series["series_id"])
            or series["series_path"] != f"train_images/{item['study_id']}/{series['series_id']}"
        ):
            raise ValueError("axial_source_path_mismatch")
        rows.append({"series_id": series["series_id"], "series_path": series["series_path"]})
    verified = _study_source(data_root, rows, data.axial_metadata["config"]["manifest_sha256"])
    if verified != source:
        raise ValueError("axial_source_receipt_or_hash_changed")
    images, positions, _, _ = _study_inputs(data_root, source)
    array_path = _path(root, entry["arrays_path"])
    if sha256_file(array_path) != entry["arrays_sha256"]:
        raise ValueError("axial_arrays_hash_mismatch")
    with np.load(array_path, allow_pickle=False) as arrays:
        if (
            len(images) != entry["n_slices"]
            or [image.instance for image in images] != arrays["instance"].tolist()
            or [image.series_id for image in images] != arrays["series_id"].tolist()
            or not np.array_equal(positions, arrays["positions"])
        ):
            raise ValueError("axial_canonical_source_alignment_mismatch")
    selected = []
    for top in attention[item["study_id"]]["levels"][item["level"]]["top3"]:
        image = images[top["source_index"]]
        if image.instance != top["instance"] or image.series_id != top["series_id"]:
            raise ValueError("axial_attention_source_alignment_mismatch")
        if not np.isfinite(image.image).all() or np.ptp(image.image) <= 1e-12:
            raise ValueError("axial_display_slice_nonfinite_or_blank")
        selected.append((image.image, top))
    return selected, {
        "record_sha256": entry["record_sha256"],
        "arrays_sha256": entry["arrays_sha256"],
        "source_sha256": source["sha256"],
    }


def _render_case(item, sag_features, axial_features, data_root, data, attention, out, plt):
    figure, axes = plt.subplots(1, 4, figsize=(16, 5))
    technical = {}
    try:
        try:
            image, receipt = _roi_image(item, sag_features, data)
            axes[0].imshow(image, cmap="gray")
            axes[0].set_title("Сагиттальная ROI: срез 5 из 9", fontsize=10)
            technical["sagittal"] = {"status": "ok", **receipt}
        except Exception as error:
            axes[0].text(
                0.5,
                0.5,
                "ROI не показана:\nпроверка не пройдена",
                ha="center",
                va="center",
                transform=axes[0].transAxes,
            )
            technical["sagittal"] = {"status": "failed", "reason": type(error).__name__}
        try:
            selected, receipt = _axial_images(item, axial_features, data_root, data, attention)
            for axis, (image, top) in zip(axes[1:], selected):
                axis.imshow(image, cmap="gray", vmin=0, vmax=1)
                axis.set_title(
                    f"Аксиальный: instance {top['instance']}\nвес внимания {top['weight']:.4f}",
                    fontsize=10,
                )
            for axis in axes[1 + len(selected) :]:
                axis.text(
                    0.5,
                    0.5,
                    "Дополнительного среза нет",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
            technical["axial"] = {"status": "ok", "n_displayed": len(selected), **receipt}
        except Exception as error:
            for axis in axes[1:]:
                axis.text(
                    0.5,
                    0.5,
                    "Аксиальный срез не показан:\nпроверка не пройдена",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
            technical["axial"] = {"status": "failed", "reason": type(error).__name__}
        for axis in axes:
            axis.axis("off")
        prediction = " | ".join(
            f"{NAMES[v]}: {GRADES[item['predictions'][v]['prediction']]}" for v in VARIANTS
        )
        figure.suptitle(
            f"{item['case_id']} | {item['level']} | Разметка RSNA: {GRADES[item['label']]}\n{prediction}\nВнимание модели не подтверждает анатомический уровень или причину симптомов.",
            fontsize=10,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.83))
        figure.savefig(
            out / "images" / f"{item['case_id']}.png",
            dpi=130,
            metadata={"Software": "ai-spine-med private review"},
        )
    finally:
        plt.close(figure)
    return technical


def _html(cases: list[dict]) -> str:
    parts = [
        """<!doctype html><html lang="ru"><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>Локальный разбор МРТ для врача</title><style>body{font:16px sans-serif;max-width:1250px;margin:30px auto;padding:0 18px}img{width:100%}article{margin:36px 0;border-top:1px solid #aaa}table{border-collapse:collapse}td,th{padding:6px 12px;border:1px solid #ccc}aside{background:#fff2cc;padding:18px}</style>
<h1>Разбор ошибок: сагиттальная и аксиальная МРТ</h1>
<aside>Приватная локальная галерея. Это целевой отбор ошибок и контрольных примеров, а не репрезентативная выборка. Разметка и ответы видны: проверка не слепая. По этой галерее нельзя оценивать точность или клиническую пригодность. Врач ещё не проверял случаи.<br><br>
Три аксиальных среза выбраны по среднему весу внимания трёх моделей. Вес внимания не подтверждает правильный анатомический уровень, локализацию стеноза, причинную связь или объяснение диагноза. Сагиттальная ROI и ответы получены из замороженного исследовательского пайплайна.</aside>
<p>Заполняйте <a href="doctor_review.csv">doctor_review.csv</a> локально. Отправки формы и сетевых запросов нет. Не публикуйте эту папку.</p>"""
    ]
    for item in cases:
        parts.append(
            f'<article><h2>{html.escape(item["case_id"])} · {html.escape(item["level"])}</h2><p>{html.escape(REASONS[item["sampling_reason"]])}. Разметка RSNA: {GRADES[item["label"]]}.</p><img src="images/{item["case_id"]}.png" alt="ROI и срезы выбранного случая"><table><tr><th>Модель</th><th>Ответ</th><th>P(норма/лёгкий)</th><th>P(умеренный)</th><th>P(тяжёлый)</th></tr>'
        )
        for variant in VARIANTS:
            row = item["predictions"][variant]
            parts.append(
                f"<tr><td>{NAMES[variant]}</td><td>{GRADES[row['prediction']]}</td>"
                + "".join(f"<td>{value:.3f}</td>" for value in row["probabilities"])
                + "</tr>"
            )
        parts.append("</table></article>")
    return "\n".join(parts) + "</html>\n"


def generate_multimodal_review(
    experiment, sag_features, axial_features, data_root, out, count=24
) -> dict:
    """Create only a separate ignored, private local validation worksheet."""
    out = _private_output(Path(out))  # Fail before loading cache or MRI outside this boundary.
    experiment, sag_features, axial_features, data_root = map(
        Path, (experiment, sag_features, axial_features, data_root)
    )
    data, candidates, attention, hashes = _load_inputs(experiment, sag_features, axial_features)
    cases = select_multimodal_cases(candidates, count)
    if not cases:
        raise ValueError("No commonly assessed validation cases available for review")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir()
    doctor, audits = [], []
    for item in cases:
        technical = _render_case(
            item, sag_features, axial_features, data_root, data, attention["fusion"], out, plt
        )
        doctor.append(
            {
                "case_id": item["case_id"],
                "study_id": item["study_id"],
                "sagittal_series_id": item["series_id"],
                "level": item["level"],
                "sampling_reason": item["sampling_reason"],
                "reference": CLASS_NAMES[item["label"]],
                **{
                    f"prediction_{variant}": CLASS_NAMES[item["predictions"][variant]["prediction"]]
                    for variant in VARIANTS
                },
                **{field: "" for field in CLINICIAN_FIELDS},
            }
        )
        audits.append(
            {
                "case_id": item["case_id"],
                "study_id": item["study_id"],
                "level": item["level"],
                "sampling_reason": item["sampling_reason"],
                "technical": technical,
                "clinical_review": "not_performed",
            }
        )
    with (out / "doctor_review.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(doctor[0]))
        writer.writeheader()
        writer.writerows(doctor)
    (out / "index.html").write_text(_html(cases), encoding="utf-8")
    write_json(out / "cases-private.json", audits)
    write_json(
        out / "inputs-private.json",
        {
            "experiment_files_sha256": hashes,
            "sagittal_cache_sha256": data.sagittal.file_sha256,
            "axial_metadata_sha256": data.axial_metadata_sha256,
            "axial_output_inventory_sha256": data.axial_output_sha256,
        },
    )
    summary = {
        "schema_version": 1,
        "scope": "private_unblinded_purposive_validation_review",
        "n_requested": count,
        "n_selected": len(cases),
        "sampling_counts": dict(Counter(item["sampling_reason"] for item in cases)),
        "display_counts": {
            kind: dict(Counter(item["technical"][kind]["status"] for item in audits))
            for kind in ("sagittal", "axial")
        },
        "clinical_review_performed": False,
        "representative_accuracy_estimate": False,
        "attention_is_confirmed_localization": False,
        "test_accessed": False,
    }
    write_json(out / "summary.json", summary)
    (out / "README.txt").write_text(
        "ПРИВАТНО: здесь МРТ и локальные идентификаторы случаев.\nОткройте index.html без сервера. Врач заполняет пустые поля doctor_review.csv.\nОтбор целевой, разметка и ответы видны; это разбор ошибок, не оценка клинической точности.\nВеса внимания не подтверждают анатомический уровень или объяснение диагноза.\nНе публикуйте папку. Новый запуск создавайте в новой подпапке reports/runs, чтобы сохранить ответы врача.\n",
        encoding="utf-8",
    )
    write_json(
        out / "files_sha256.json",
        {
            str(path.relative_to(out)): sha256_file(path)
            for path in sorted(out.rglob("*"))
            if path.is_file()
        },
    )
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("experiment", "sag-features", "axial-features", "data-root", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--count", default=24, type=int)
    args = parser.parse_args(argv)
    try:
        result = generate_multimodal_review(
            args.experiment,
            args.sag_features,
            args.axial_features,
            args.data_root,
            args.out,
            args.count,
        )
    except Exception as error:
        parser.exit(
            2, f"Private review failed ({type(error).__name__}); no DICOM details are logged.\n"
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
