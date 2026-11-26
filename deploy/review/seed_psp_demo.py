"""Create the synthetic PSP example and optionally write its public projection."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

from ai_spine_med.analysis_modules import LEVELS, patient_summary
from ai_spine_med.psp import BODY_VERSION, UPLOAD, PSPStore

DEMO_KEY = "synthetic-psp-demo-v1"
PATIENT_CODE = "ДЕМО · вымышленный эпизод"
CLINICIAN = "Вымышленный осмотр · демонстрация PSP"
PAIN = "Боль в пояснице с отдачей в левую ягодицу и по наружной стороне левой голени."
SENSATION = "Периодическое онемение на тыле левой стопы."
WALKING = "После примерно 300 м ходьбы боль усиливается. В положении сидя становится легче."
INTENSITY = {"current": 5, "worst": 7, "typical": 4}
MAP = [
    {
        "view": "back",
        "type": "pain",
        "x": 0.46,
        "y": 0.40,
        "side": "left",
        "intensity": 5,
        "region": "Поясница",
    },
    {
        "view": "back",
        "type": "pain",
        "x": 0.43,
        "y": 0.47,
        "side": "left",
        "intensity": 5,
        "region": "Ягодица",
    },
    {
        "view": "back",
        "type": "pain",
        "x": 0.40,
        "y": 0.74,
        "side": "left",
        "intensity": 6,
        "region": "Наружная сторона голени",
    },
    {
        "view": "front",
        "type": "numbness",
        "x": 0.59,
        "y": 0.93,
        "side": "left",
        "intensity": 3,
        "region": "Тыл стопы",
    },
]


def _answers():
    answers = {
        "Q01": {"value": ["both"], "details": f"{PAIN} {SENSATION}"},
        "Q02": {"date": "Около трёх месяцев назад", "onset": "gradual", "course": "first"},
        "Q03": {"value": "no"},
        "Q05": {"value": PAIN, "side": "left"},
        "Q06": INTENSITY,
        "Q07": {"value": ["aching", "shooting"]},
        "Q08": {"pattern": "episodic", "course": "stable"},
        "Q09": {"worsens": "Длительная ходьба и стояние", "relieves": "Отдых сидя"},
        "Q10": {"value": "no"},
        "Q12": {"value": ["none"]},
        "Q14": {"distance_m": 300, "relieved": "yes", "details": WALKING},
        "Q25": {"operations": "Операций на позвоночнике не было."},
        "Q26": {"tests": "Для демонстрации привязана ранее загруженная серия МРТ."},
        "Q27": {"walking": 6, "sleep": 0, "sitting": 1, "work": 3, "self_care": 0},
        "Q28": {"value": "Хочу снова гулять без частых остановок."},
        "Q29": {"value": "yes"},
    }
    for question in ("Q15", "Q16", "Q17", "Q18", "Q19", "Q24"):
        answers[question] = {"value": "no"}
    for question in ("Q20", "Q21", "Q22", "Q23"):
        answers[question] = {"value": ["none"]}
    return answers


def _exam(catalog):
    observations = {}
    for test in catalog["tests"]:
        code = test["id"]
        for side in test["sides"]:
            left = side == "left"
            if code.startswith("M"):
                result = {"mrc": 4 if code == "M05" and left else 5, "pain_limited": False}
            elif code in {"R01", "R02"}:
                result = {"value": 2}
            elif code.startswith("S_"):
                result = {
                    "value": "decreased" if code == "S_L5" and left else "normal",
                    "modality": "Лёгкое прикосновение",
                }
            elif code == "T01_SLR":
                result = {"angle": 55 if left else 70, "typical_symptoms": "yes" if left else "no"}
                if left:
                    result["symptom_area"] = "Левая ягодица и наружная сторона голени"
            else:
                raise ValueError(f"Неизвестная проба в каталоге: {code}")
            observations[f"{code}_{side}"] = {"status": "observed", **result}
    return {
        "clinician": CLINICIAN,
        "notes": {
            "N00": "Учебный сценарий. Жалобы, согласие и осмотр смоделированы.",
            "N01": "В сценарии пациент выполняет пробы по инструкции.",
            "N02": "Ходьба без опоры, темп слегка замедлен.",
        },
        "observations": observations,
    }


def read_upload(root: Path, upload_id: str):
    if not UPLOAD.fullmatch(upload_id):
        raise ValueError("Некорректный идентификатор исследования.")
    record = json.loads((root / upload_id / "record.json").read_text(encoding="utf-8"))
    if record.get("upload_id") != upload_id or record.get("status") != "completed":
        raise ValueError("Для примера выберите полностью обработанное исследование.")
    return record


def seed_demo(store: PSPStore, upload_id: str):
    # Replay the same requests across repeated runs, including an interrupted first
    # run. Their original revisions also prevent overwriting later manual edits.
    created = store.create(
        {
            "idempotency_key": f"{DEMO_KEY}-create",
            "patient_code": PATIENT_CODE,
            "identity_confirmed": True,
            "adult_confirmed": True,
            "consent": True,
            "upload_id": upload_id,
        }
    )
    episode_id = created["episode_id"]
    intake = store.save(
        episode_id,
        {
            "idempotency_key": f"{DEMO_KEY}-intake",
            "revision": created["revision"],
            "action": "submit_intake",
            "answers": _answers(),
            "pain_map": [
                {**point, "id": f"psp-demo-{index}", "body_version": BODY_VERSION}
                for index, point in enumerate(MAP, start=1)
            ],
            "exam": _exam(store.catalog()),
        },
    )
    store.save(
        episode_id,
        {
            "idempotency_key": f"{DEMO_KEY}-exam",
            "revision": intake["revision"],
            "action": "review_exam",
        },
    )
    return store.get(episode_id)


def _public_mri(record):
    # Keep only known level/category codes and bounded numeric measurements.
    # Never copy a study's free text, identifiers, links, contours or DICOM files.
    targets = [
        {"level": entry["level"], "status": "ok", "prediction": entry["prediction"]}
        for entry in record.get("targets", [])
        if entry.get("level") in LEVELS
        and entry.get("status") == "ok"
        and type(entry.get("prediction")) is int
        and entry["prediction"] in (0, 1, 2)
    ]
    morphology = record.get("modules", {}).get("disc_morphology", {})
    levels = []
    if morphology.get("status") == "completed":
        for entry in morphology.get("levels", []):
            level = entry.get("level")
            if level not in LEVELS:
                continue
            voxels = entry.get("candidate_voxels")
            candidate = entry.get("geometric_candidate")
            frames = [
                frame
                for frame in entry.get("frames", [])
                if frame.get("slice_key") == entry.get("representative_slice_key")
            ]
            measurements = []
            if len(frames) == 1:
                frame = frames[0]
                clean = {
                    "slice_key": "representative",
                    "geometry_measurable": frame.get("geometry_measurable") is True,
                }
                for key in ("B_mm", "D_mm", "H_mm", "D_over_B", "uncertainty_mm"):
                    value = frame.get(key)
                    if type(value) in (float, int) and 0 <= value <= 1000 and math.isfinite(value):
                        clean[key] = value
                measurements.append(clean)
            levels.append(
                {
                    "level": level,
                    "candidate_voxels": type(voxels) is int and voxels > 0,
                    "geometric_candidate": candidate
                    if candidate in ("extrusion", "protrusion")
                    else "indeterminate",
                    "representative_slice_key": "representative",
                    "frames": measurements,
                }
            )
    report = patient_summary(
        targets,
        {
            "disc_morphology": {
                "status": "completed" if levels else "unavailable",
                "levels": levels,
            },
        },
    )
    count = record.get("n_slices")
    title = (
        f"Загруженная серия МРТ · {count} срезов"
        if type(count) is int and 0 < count < 1000
        else "Загруженная серия МРТ"
    )
    return {"title": title, "findings": report["findings"], "unassessed": report["unassessed"]}


def public_projection(upload):
    # Clinical material comes only from this authored fixture. The editable
    # private episode is deliberately never read by this projection.
    return {
        "title": "Пример сводки обследования",
        "context": "Жалобы, карта симптомов и осмотр вымышлены. Результаты анализа взяты из загруженной серии МРТ.",
        "complaints": [
            {"label": "Основная жалоба", "value": PAIN},
            {"label": "Чувствительность", "value": SENSATION},
            {"label": "Ходьба", "value": WALKING},
            {
                "label": "Интенсивность боли",
                "value": f"{INTENSITY['current']}/10 сейчас, до {INTENSITY['worst']}/10 за неделю.",
            },
        ],
        "pain_map": MAP,
        "exam": [
            {"label": "Разгибание большого пальца", "left": "4/5 по MRC", "right": "5/5 по MRC"},
            {"label": "Коленный рефлекс", "left": "2+", "right": "2+"},
            {"label": "Ахиллов рефлекс", "left": "2+", "right": "2+"},
            {
                "label": "Лёгкое прикосновение на тыле стопы",
                "left": "Снижено",
                "right": "Без изменений",
            },
            {
                "label": "Подъём выпрямленной ноги (SLR)",
                "left": "Привычная боль при 55°",
                "right": "Без привычной боли при 70°",
            },
        ],
        "mri": _public_mri(upload),
        "summary": [
            "Со слов пациента, боль распространяется от поясницы в левую ногу. Длительная ходьба усиливает симптомы.",
            "В вымышленном осмотре слева снижены сила разгибания большого пальца и чувствительность тыла стопы.",
            "Результаты анализа МРТ относятся к загруженной серии. Степени стеноза и геометрия масок требуют проверки по снимкам.",
        ],
    }


def _write_json(destination, payload):
    if destination == "-":
        print(json.dumps(payload, ensure_ascii=False, allow_nan=False))
        return
    path = Path(destination)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.close()
            temporary.chmod(0o644)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upload-root", type=Path, default=os.environ.get("REVIEW_UPLOAD_ROOT", "/uploads")
    )
    parser.add_argument("--psp-root", type=Path, default=os.environ.get("REVIEW_PSP_ROOT"))
    parser.add_argument("--upload-id", required=True)
    parser.add_argument("--public-json", help="Output path for public JSON, or - for stdout.")
    args = parser.parse_args()
    try:
        upload = read_upload(args.upload_root, args.upload_id)
        root = args.psp_root or args.upload_root / ".psp"
        episode = seed_demo(PSPStore(root), args.upload_id)
        if args.public_json:
            _write_json(args.public_json, public_projection(upload))
    except (OSError, ValueError) as error:
        parser.exit(1, f"Не удалось подготовить пример PSP: {error}\n")
    details = {
        key: episode[key] for key in ("episode_id", "revision", "intake_status", "exam_status")
    }
    print(
        json.dumps(details, ensure_ascii=False),
        file=sys.stderr if args.public_json == "-" else sys.stdout,
    )


if __name__ == "__main__":
    main()
