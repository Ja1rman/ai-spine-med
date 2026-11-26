"""Private, versioned PSP intake and manual examination records.

This collection prototype has no authenticated clinician signatures, treatment
logic, device adapters or autonomous diagnosis. Clinical routing rules are draft.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

VERSION = "psp-lumbar-draft-1.0"
BODY_VERSION = "psp-body-2d-v1"
MAX_BYTES = 512 * 1024
EPISODE = re.compile(r"^episode-[a-f0-9]{32}$")
UPLOAD = re.compile(r"^upload-[a-f0-9]{32}$")
SIDES = ["left", "right", "bilateral", "midline", "unknown"]
OBS_STATUSES = ["observed", "not_tested", "refused", "unsafe_to_perform", "invalid", "unknown"]
UNKNOWN_VALUES = {"unknown", "refused"}


class PSPError(ValueError):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def _error(message="Проверьте заполненные поля.", code="invalid_data", status=422):
    raise PSPError(status, code, message)


def _options(*pairs):
    return [{"value": value, "label": label} for value, label in pairs]


YES = _options(("yes", "Да"), ("no", "Нет"), ("unknown", "Не знаю"), ("refused", "Обсужу с врачом"))
SIDE_OPTIONS = _options(("left", "Слева"), ("right", "Справа"), ("bilateral", "С обеих сторон"), ("midline", "По центру"), ("unknown", "Не знаю"))


def _field(key, label, kind="text", **kwargs):
    return {"key": key, "label": label, "type": kind, **kwargs}


def _choice(key="value", label="Ответ", options=None):
    return _field(key, label, "choice", options=options or YES)


def _number(key, label, maximum=10):
    return _field(key, label, "number", min=0, max=maximum)


def _question(identifier, title, section, fields, **kwargs):
    return {"id": identifier, "title": title, "section": section, "type": "group", "fields": fields, **kwargs}


QUESTIONS = [
    _question("Q01", "Что беспокоит больше всего?", "Жалобы", [_field("value", "Основная жалоба", "multi", options=_options(("low_back", "Поясница"), ("leg", "Нога"), ("both", "Поясница и нога"), ("other", "Другое"))), _field("details", "Уточнение")]),
    _question("Q02", "Когда и как это началось?", "Жалобы", [_field("date", "Дата или примерное время начала"), _choice("onset", "Начало", _options(("sudden", "Внезапно"), ("gradual", "Постепенно"), ("unknown", "Не знаю"))), _choice("course", "Эпизод", _options(("first", "Впервые"), ("recurrent", "Повторный"), ("unknown", "Не знаю")))]),
    _question("Q03", "Была травма, падение или необычная нагрузка?", "Жалобы", [_choice(), _field("details", "Обстоятельства")]),
    _question("Q04", "Где болит? Отметьте зоны на карте тела.", "Карта симптомов", [], type="map"),
    _question("Q05", "Где проходит боль и с какой стороны?", "Жалобы", [_field("value", "Поясница, ягодица, бедро, голень, стопа; направление боли", "textarea"), _choice("side", "Сторона", SIDE_OPTIONS)]),
    _question("Q06", "Насколько сильная боль по шкале 0–10?", "Жалобы", [_number("current", "Сейчас"), _number("worst", "Максимум за 7 дней"), _number("typical", "Обычно")]),
    _question("Q07", "Какой характер боли?", "Жалобы", [_field("value", "Характер", "multi", options=_options(("aching", "Ноющая"), ("burning", "Жгучая"), ("electric", "Как электрический разряд"), ("shooting", "Стреляющая"), ("other", "Другая"), ("unknown", "Не знаю")))]),
    _question("Q08", "Как меняется боль?", "Жалобы", [_choice("pattern", "Частота", _options(("continuous", "Постоянная"), ("episodic", "Приступами"), ("unknown", "Не знаю"))), _choice("course", "Изменение", _options(("worsening", "Усиливается"), ("improving", "Уменьшается"), ("stable", "Без изменений"), ("unknown", "Не знаю")))]),
    _question("Q09", "Что усиливает и уменьшает боль?", "Жалобы", [_field("worsens", "Усиливает: ходьба, стояние, сидение, наклон, кашель и другое", "textarea"), _field("relieves", "Уменьшает", "textarea")]),
    _question("Q10", "Боль будит ночью или беспокоит в покое?", "Жалобы", [_choice(), _field("details", "Уточнение")]),
    _question("Q11", "Есть онемение или покалывание? Отметьте отдельным слоем карты.", "Карта симптомов", [], type="map"),
    _question("Q12", "Замечаете слабость в ногах?", "Слабость и ходьба", [_field("value", "Проявления", "multi", options=_options(("none", "Не замечаю"), ("weakness", "Слабость"), ("knee_buckle", "Подкашивается колено"), ("foot_drop", "Цепляюсь носком"), ("toe_heel", "Трудно встать на носки или пятки"), ("unknown", "Не знаю"), ("refused", "Обсужу с врачом"))), _choice("side", "Сторона", SIDE_OPTIONS)]),
    _question("Q13", "Как меняется слабость?", "Слабость и ходьба", [_choice("onset", "Изменение", _options(("new", "Появилась внезапно / недавно"), ("progressive", "Нарастает"), ("stable", "Стабильная"), ("unknown", "Не знаю"))), _choice("side", "Сторона", SIDE_OPTIONS), _choice("severity", "Выраженность", _options(("mild", "Небольшая"), ("severe", "Выраженная"), ("unknown", "Не знаю"))), _field("time", "Когда началось")], show_if={"question": "Q12", "field": "value", "any": ["weakness", "knee_buckle", "foot_drop", "toe_heel"]}),
    _question("Q14", "Как долго можете идти до остановки из-за симптомов?", "Слабость и ходьба", [_number("distance_m", "Расстояние, м", 100000), _choice("relieved", "Становится легче сидя или при наклоне вперёд?"), _field("details", "Уточнение")]),
]
for _identifier, _title in [
    ("Q15", "Появились или усилились трудности начала мочеиспускания, слабая струя или изменение ощущения наполнения?"),
    ("Q16", "Появилась задержка мочи, подтекание или потеря контроля?"),
    ("Q17", "Появилось или изменилось онемение промежности, гениталий, вокруг ануса или ощущение при подтирании?"),
    ("Q18", "Появилась потеря контроля стула или ощущения наполнения прямой кишки?"),
    ("Q19", "Появилось нарушение генитальной чувствительности или сексуальной функции?"),
]:
    QUESTIONS.append(_question(_identifier, _title, "Важные симптомы", [_choice(), _field("onset_details", "Когда началось или усилилось")], safety=True))
for _identifier, _title, _pairs in [
    ("Q20", "Лихорадка, инфекция и недавние вмешательства", [("fever", "Лихорадка / озноб"), ("recent_infection", "Недавняя инфекция"), ("spinal_procedure", "Инъекция или операция на позвоночнике")]),
    ("Q21", "Онкологические заболевания и снижение веса", [("cancer", "Онкологическое заболевание"), ("weight_loss", "Необъяснимое похудение")]),
    ("Q22", "Факторы риска инфекции", [("immunosuppression", "Иммуносупрессия"), ("iv_drugs", "Внутривенные наркотики"), ("diabetes", "Диабет")]),
    ("Q23", "Факторы риска перелома", [("osteoporosis", "Остеопороз"), ("steroids", "Длительные системные гормоны"), ("fracture", "Переломы")]),
]:
    QUESTIONS.append(_question(_identifier, _title, "Анамнез", [_field("value", "Отметьте подходящее", "multi", options=_options(*_pairs, ("none", "Ничего из перечисленного"), ("unknown", "Не знаю"), ("refused", "Обсужу с врачом"))), _field("details", "Подробности и время")], safety=True))
QUESTIONS.extend([
    _question("Q24", "Есть резкая необычная боль в спине вместе с болью в животе, обмороком или выраженной общей слабостью?", "Важные симптомы", [_choice(), _field("onset_details", "Когда началось")], safety=True),
    _question("Q25", "Предшествующие болезни, операции и лекарства", "Анамнез", [_field("operations", "Операции и предыдущие эпизоды", "textarea"), _field("illnesses", "Заболевания", "textarea"), _field("medications", "Лекарства, в том числе антикоагулянты", "textarea")]),
    _question("Q26", "Что уже обследовали и лечили?", "Анамнез", [_field("tests", "Обследования", "textarea"), _field("treatment", "Лечение", "textarea"), _field("effects", "Эффект и нежелательные реакции", "textarea")]),
    _question("Q27", "Насколько симптомы ограничивают жизнь? 0 — не мешают, 10 — полностью ограничивают.", "Функция", [_number("walking", "Ходьба"), _number("sleep", "Сон"), _number("sitting", "Сидение"), _number("work", "Работа"), _number("self_care", "Самообслуживание")]),
    _question("Q28", "Главная цель и дополнительная информация", "Функция", [_field("value", "Что важно обсудить с врачом?", "textarea")]),
    _question("Q29", "Проверьте ответы перед отправкой", "Проверка", [_choice("value", "Подтверждаю актуальность введённых сведений", _options(("yes", "Подтверждаю"), ("no", "Нужно исправить")))]),
])
QUESTIONS.sort(key=lambda q: q["id"])
QUESTION_INDEX = {q["id"]: q for q in QUESTIONS}

TESTS = []
for _index, _title in enumerate(["Сгибание бедра", "Приведение бедра", "Разгибание колена", "Тыльное сгибание стопы", "Разгибание большого пальца", "Эверсия стопы", "Подошвенное сгибание стопы", "Сгибание колена"], 1):
    TESTS.append({"id": f"M{_index:02}", "title": _title, "category": "Сила мышц", "sides": ["left", "right"], "required": True, "fields": [_number("mrc", "MRC, 0–5", 5), _number("force_n", "Сила, Н (отдельное измерение)", 5000), _choice("pain_limited", "Ограничено болью?", _options(("yes", "Да"), ("no", "Нет")))]})
for _identifier, _title in [("R01", "Коленный рефлекс"), ("R02", "Ахиллов рефлекс")]:
    TESTS.append({"id": _identifier, "title": _title, "category": "Рефлексы", "sides": ["left", "right"], "required": True, "fields": [_number("value", "Оценка 0–4+", 4)]})
for _level, _title in [("L2", "Передняя поверхность бедра"), ("L3", "Медиальная поверхность колена"), ("L4", "Медиальная голень / лодыжка"), ("L5", "Тыл стопы / I–II промежуток"), ("S1", "Латеральный край стопы")]:
    TESTS.append({"id": "S_" + _level, "title": _title + " (" + _level + ")", "category": "Чувствительность", "sides": ["left", "right"], "required": True, "fields": [_choice("value", "Результат", _options(("normal", "Без выявленных изменений"), ("decreased", "Снижена"), ("absent", "Отсутствует"), ("increased", "Повышена"), ("unreliable", "Недостоверно"))), _field("modality", "Метод (например, лёгкое прикосновение)")]})
TESTS.append({"id": "T01_SLR", "title": "Подъём выпрямленной ноги (SLR)", "category": "Нейродинамика", "sides": ["left", "right"], "required": True, "fields": [_number("angle", "Угол, градусы", 180), _choice("typical_symptoms", "Воспроизведена привычная иррадиирующая боль?"), _field("symptom_area", "Где возникли симптомы")]})
TEST_INDEX = {test["id"]: test for test in TESTS}
STEPS = [{"id": f"N{index:02}", "title": title} for index, title in enumerate(["Идентичность, согласие и новые симптомы", "Безопасность движений и ограничения", "Поза и походка", "Функциональные пробы", "Сила мышц", "Рефлексы", "Чувствительность", "Нейродинамика", "Другие клинические ориентиры", "Дополнительные тесты и приборы", "Полнота и ограничения", "Проверка врачом"])]
SAFETY_IDS = ["Q15", "Q16", "Q17", "Q18", "Q19", "Q24"]
SCREEN_IDS = ["Q12", "Q20", "Q21", "Q22", "Q23", *SAFETY_IDS]
CATALOG = {"version": VERSION, "body_version": BODY_VERSION, "protocol_status": "draft", "questions": QUESTIONS, "tests": TESTS, "exam_steps": STEPS, "safety_question_ids": SAFETY_IDS, "observation_statuses": OBS_STATUSES, "sides": SIDES}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json(value):
    try:
        result = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (ValueError, TypeError):
        _error()
    if len(result.encode()) > MAX_BYTES:
        _error("Слишком большой объём данных.", "payload_limit", 413)
    return result


def _text(value, limit=2000, allow_empty=True):
    if not isinstance(value, str) or len(value) > limit or any(ord(c) < 32 and c not in "\n\t\r" for c in value):
        _error()
    value = value.strip()
    if not allow_empty and not value:
        _error()
    return value


def _fields(value, fields):
    if not isinstance(value, dict) or set(value) - {f["key"] for f in fields}:
        _error("Неизвестное поле ответа.")
    result = {}
    for field in fields:
        key, kind = field["key"], field["type"]
        raw = value.get(key)
        if raw is None or raw == "" or raw == []:
            continue
        if kind == "number":
            if type(raw) not in {float, int} or not math.isfinite(raw) or not field.get("min", 0) <= raw <= field.get("max", 100):
                _error("Число выходит за допустимый диапазон.")
            if key in {"current", "worst", "typical", "mrc", "value", "walking", "sleep", "sitting", "work", "self_care"} and int(raw) != raw:
                _error("Для этой шкалы требуется целое число.")
            result[key] = raw
        elif kind in {"choice", "multi"}:
            allowed = {o["value"] for o in field["options"]}
            if kind == "choice":
                if not isinstance(raw, str) or raw not in allowed:
                    _error("Неизвестный вариант ответа.")
            else:
                if not isinstance(raw, list) or len(raw) > len(allowed) or any(not isinstance(v, str) or v not in allowed for v in raw) or len(set(raw)) != len(raw):
                    _error("Некорректный список ответов.")
                if len(raw) > 1 and any(v in {"none", "unknown", "refused"} for v in raw):
                    _error("Ответ «нет» или «не знаю» нельзя сочетать с другими вариантами.")
            result[key] = raw
        else:
            result[key] = _text(raw, 4000 if kind == "textarea" else 500)
    return result


def _answers(raw, safety_only=False):
    if not isinstance(raw, dict) or len(raw) > 29:
        _error()
    result = {}
    for identifier, answer in raw.items():
        if identifier not in QUESTION_INDEX or (safety_only and identifier not in SCREEN_IDS + ["Q13"]):
            _error("Неизвестный вопрос.")
        result[identifier] = _fields(answer, QUESTION_INDEX[identifier]["fields"])
    return result


def _map(raw):
    if not isinstance(raw, list) or len(raw) > 150:
        _error("На карте слишком много отметок.")
    result, ids = [], set()
    for area in raw:
        if not isinstance(area, dict) or set(area) - {"id", "view", "type", "x", "y", "side", "intensity", "region", "body_version", "body_model_version", "recorded_at", "source"}:
            _error("Некорректная отметка на карте.")
        identifier = _text(area.get("id", ""), 80, False)
        if identifier in ids or area.get("view") not in {"front", "back"} or area.get("type") not in {"pain", "numbness", "weakness"} or area.get("side") not in SIDES:
            _error("Некорректная отметка на карте.")
        ids.add(identifier)
        for key in ("x", "y"):
            v = area.get(key)
            if type(v) not in {int, float} or not math.isfinite(v) or not 0 <= v <= 1:
                _error("Координаты должны быть в пределах карты.")
        intensity = area.get("intensity")
        if type(intensity) is not int or not 0 <= intensity <= 10:
            _error()
        if area.get("body_version", area.get("body_model_version", BODY_VERSION)) != BODY_VERSION:
            _error("Неизвестная версия карты тела.")
        recorded = area.get("recorded_at") or _now()
        try:
            if datetime.fromisoformat(recorded.replace("Z", "+00:00")).tzinfo is None:
                _error()
        except (ValueError, TypeError, AttributeError):
            _error("Некорректное время отметки.")
        result.append({"id": identifier, "view": area["view"], "type": area["type"], "x": area["x"], "y": area["y"], "side": area["side"], "intensity": intensity, "region": _text(area.get("region", ""), 200), "body_version": BODY_VERSION, "recorded_at": recorded, "source": "patient_self_report"})
    return result


def _exam(raw):
    if not isinstance(raw, dict) or set(raw) - {"clinician", "notes", "observations", "safety_answers", "reviewed_by", "reviewed_at", "review_revision"}:
        _error("Некорректный протокол осмотра.")
    notes = raw.get("notes", {})
    if not isinstance(notes, dict) or set(notes) - {s["id"] for s in STEPS}:
        _error()
    observations = raw.get("observations", {})
    if not isinstance(observations, dict) or len(observations) > 40:
        _error()
    normalized = {}
    for key, obs in observations.items():
        try:
            test_id, side = key.rsplit("_", 1)
            test = TEST_INDEX[test_id]
        except (KeyError, ValueError):
            _error("Неизвестная проба.")
        if side not in test["sides"] or not isinstance(obs, dict):
            _error()
        allowed = {f["key"] for f in test["fields"]} | {"status", "quality", "method", "reason", "note", "recorded_at", "record_id", "source", "operator", "test_id", "side"}
        if set(obs) - allowed or obs.get("status", "not_tested") not in OBS_STATUSES:
            _error()
        values = {f["key"]: obs[f["key"]] for f in test["fields"] if f["key"] in obs}
        # HTML choices use yes/no; stored pain limitation remains a boolean.
        if type(values.get("pain_limited")) is bool:
            values["pain_limited"] = "yes" if values["pain_limited"] else "no"
        values = _fields(values, test["fields"])
        if "pain_limited" in values:
            values["pain_limited"] = values["pain_limited"] == "yes"
        status = obs.get("status", "not_tested")
        has_value = any(k in values for k in ({"mrc", "force_n"} if test_id.startswith("M") else {"value", "angle", "typical_symptoms"}))
        if status == "observed" and not has_value:
            _error("Для выполненной пробы укажите результат.")
        if status != "observed" and has_value:
            _error("Результат нельзя записать как невыполненную пробу.")
        quality = obs.get("quality", "valid" if status == "observed" else "limited")
        if quality not in {"valid", "limited", "invalid"}:
            _error()
        if quality == "invalid" and status == "observed":
            _error("Невалидное измерение отметьте статусом «недостоверно».")
        normalized[key] = {**values, "status": status, "quality": quality, "method": _text(obs.get("method", "manual"), 200), "reason": _text(obs.get("reason", "")), "note": _text(obs.get("note", "")), "test_id": test_id, "side": side, "source": "clinician_exam", "operator": _text(raw.get("clinician", ""), 120)}
    return {"clinician": _text(raw.get("clinician", ""), 120), "notes": {k: _text(v, 4000) for k, v in notes.items()}, "observations": normalized, "safety_answers": _answers(raw.get("safety_answers", {}), True)}


def _screen(answers, exam, previous, symptom_map=None):
    """Conservative explicit-input draft rules; historical signals are retained."""
    current = []
    sources = [("patient_self_report", answers), ("clinician_exam", exam.get("safety_answers", {}))]
    for source, values in sources:
        for question_id in SAFETY_IDS:
            if values.get(question_id, {}).get("value") == "yes":
                current.append((f"{source}:{question_id}", "E", question_id, QUESTION_INDEX[question_id]["title"], source))
        weakness = values.get("Q13", {})
        if weakness.get("side") == "bilateral" and weakness.get("onset") in {"new", "progressive"} and (weakness.get("severity") == "severe" or weakness.get("onset") == "progressive"):
            current.append((f"{source}:bilateral_weakness", "E", "Q13", "Зарегистрирована новая выраженная или нарастающая слабость обеих ног.", source))
        for question_id in ("Q20", "Q21", "Q22", "Q23"):
            value = values.get(question_id, {}).get("value", [])
            if any(v not in {"none", "unknown", "refused"} for v in value):
                current.append((f"{source}:{question_id}", "U", question_id, QUESTION_INDEX[question_id]["title"], source))
        if any(v not in {"none", "unknown", "refused"} for v in values.get("Q12", {}).get("value", [])):
            current.append((f"{source}:weakness", "U", "Q12", "Сообщается о слабости ноги; нужна оценка клиницистом.", source))
        if values.get("Q03", {}).get("value") == "yes":
            current.append((f"{source}:injury", "U", "Q03", "Сообщается о травме или необычной нагрузке; требуется уточнение врачом.", source))
        if values.get("Q08", {}).get("course") == "worsening":
            current.append((f"{source}:worsening", "U", "Q08", "Сообщается об ухудшении симптомов.", source))
    map_weakness = any(point["type"] == "weakness" for point in (symptom_map or []))
    if map_weakness:
        current.append(("patient_self_report:map_weakness", "U", "Q12", "На карте отмечена субъективная слабость; нужно уточнение врачом.", "patient_self_report"))
    missing = []
    for qid in SCREEN_IDS:
        value = answers.get(qid, {}).get("value")
        if value is None or value == [] or (isinstance(value, str) and value in UNKNOWN_VALUES) or (isinstance(value, list) and any(v in UNKNOWN_VALUES for v in value)):
            missing.append(qid)
    if map_weakness or any(v not in {"none", "unknown", "refused"} for v in answers.get("Q12", {}).get("value", [])):
        q13 = answers.get("Q13", {})
        if any(q13.get(key) in {None, "unknown"} for key in ("onset", "side", "severity")):
            missing.append("Q13")
    alerts = copy.deepcopy(previous.get("alerts", []))
    known = {a["rule_id"] for a in alerts}
    for rule_id, level, qid, message, source in current:
        if rule_id not in known:
            alerts.append({"id": "alert-" + uuid.uuid4().hex, "rule_id": rule_id, "level": level, "message": message, "question_ids": [qid], "source": source, "evidence": copy.deepcopy(dict(sources).get(source, {}).get(qid, {})), "raised_at": _now(), "acknowledged": False, "rules_version": VERSION})
            known.add(rule_id)
    level = "E" if any(a["level"] == "E" for a in alerts) else "U" if alerts else "X" if missing else "R"
    messages = {"E": "Нужна срочная очная медицинская оценка. Не ждите завершения анкеты. Обратитесь за экстренной помощью по местному маршруту.", "U": "Нужна врачебная оценка отмеченных ответов. Срок и маршрут определяет медицинский специалист.", "X": "Важные вопросы заполнены не полностью. Риск не определён; перед продолжением нужна проверка сотрудником.", "R": "Скрининг заполнен. По введённым ответам триггеры правил не выявлены; это не исключает заболевание."}
    return {"level": level, "message": messages[level], "alerts": alerts, "missing_questions": missing, "rules_version": VERSION, "protocol_status": "draft", "delivery_status": "not_configured"}


def _summary(record):
    findings, missing, unperformed = [], [], []
    answers = record["answers"]
    for question in QUESTIONS:
        qid = question["id"]
        value = answers.get(qid, {})
        parts = []
        for field in question["fields"]:
            raw = value.get(field["key"])
            if raw is None or raw == "" or raw == []:
                continue
            options = {o["value"]: o["label"] for o in field.get("options", [])}
            if isinstance(raw, list):
                shown = ", ".join(options.get(v, v) for v in raw)
            else:
                shown = options.get(str(raw), str(raw))
            parts.append(f"{field['label']}: {shown}")
        if parts:
            findings.append(f"{qid} · {question['title']} " + "; ".join(parts))
        elif question["type"] != "map" and qid != "Q13":
            missing.append(qid)
    if record["pain_map"]:
        findings.append(f"Карта субъективных симптомов: {len(record['pain_map'])} отметок.")
    for key, obs in record["exam"]["observations"].items():
        test = TEST_INDEX[obs["test_id"]]
        if obs["status"] == "observed":
            parts = []
            for field in test["fields"]:
                if field["key"] in obs:
                    raw = obs[field["key"]]
                    choices = {o["value"]: o["label"] for o in field.get("options", [])}
                    shown = "Да" if raw is True else "Нет" if raw is False else choices.get(str(raw), str(raw))
                    parts.append(f"{field['label']}: {shown}")
            side = {"left": "слева", "right": "справа"}.get(obs["side"], obs["side"])
            findings.append(f"Осмотр · {test['title']} ({side}): " + "; ".join(parts))
        elif obs.get("reason"):
            side = {"left": "слева", "right": "справа"}.get(obs["side"], obs["side"])
            status = {"not_tested": "не выполнено", "refused": "отказ", "unsafe_to_perform": "небезопасно выполнять", "invalid": "недостоверный результат", "unknown": "неизвестно"}.get(obs["status"], obs["status"])
            unperformed.append(f"{test['title']} ({side}): {status}; {obs['reason']}")
    limitations = ["Анкета — сведения со слов пациента; осмотр — ручной ввод указанного специалиста.", "Правила скрининга имеют статус проекта и требуют медицинского утверждения.", "Подтверждение осмотра здесь не является удостоверенной электронной подписью.", "Автоматическая доставка предупреждений медицинскому персоналу не подключена."]
    if not record.get("upload_id"):
        limitations.append("МРТ к эпизоду не привязана.")
    return {"findings": findings, "missing": missing, "unperformed": unperformed, "limitations": limitations}


class PSPStore:
    """SQLite snapshots with optimistic concurrency and an append-only audit chain."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        self.database = self.root / "psp.sqlite3"
        self.lock = threading.RLock()
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY, revision INTEGER NOT NULL, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS audit (
                    seq INTEGER PRIMARY KEY, episode_id TEXT NOT NULL, revision INTEGER NOT NULL,
                    occurred_at TEXT NOT NULL, action TEXT NOT NULL, previous_hash TEXT NOT NULL,
                    hash TEXT NOT NULL, snapshot TEXT NOT NULL, UNIQUE(episode_id, revision));
                CREATE TABLE IF NOT EXISTS requests (
                    scope TEXT NOT NULL, key TEXT NOT NULL, digest TEXT NOT NULL, response TEXT NOT NULL,
                    PRIMARY KEY(scope, key));
            """)
        self.database.chmod(0o600)

    @contextmanager
    def _connect(self):
        db = None
        try:
            db = sqlite3.connect(self.database, timeout=5)
            db.execute("PRAGMA foreign_keys=ON")
            with db:
                yield db
        except sqlite3.Error as error:
            raise PSPError(503, "storage_unavailable", "Хранилище временно недоступно.") from error
        finally:
            if db is not None:
                db.close()

    def catalog(self):
        return copy.deepcopy(CATALOG)

    def _get(self, db, identifier):
        if not isinstance(identifier, str) or not EPISODE.fullmatch(identifier):
            _error("Эпизод не найден.", "not_found", 404)
        row = db.execute("SELECT data FROM episodes WHERE id=?", (identifier,)).fetchone()
        if row is None:
            _error("Эпизод не найден.", "not_found", 404)
        return json.loads(row[0])

    def get(self, identifier):
        with self._connect() as db:
            return self._get(db, identifier)

    def list_episodes(self):
        with self._connect() as db:
            records = [json.loads(r[0]) for r in db.execute("SELECT data FROM episodes ORDER BY rowid DESC")]
        return {"episodes": [{k: r[k] for k in ("episode_id", "patient_code", "revision", "created_at", "updated_at", "upload_id", "intake_status", "exam_status")} | {"safety_level": r["safety"]["level"]} for r in records]}

    def _request(self, payload, scope, db):
        if not isinstance(payload, dict):
            _error()
        encoded = _json(payload)
        key = payload.get("idempotency_key")
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_-]{8,100}", key):
            _error("Для сохранения нужен уникальный ключ запроса.", "idempotency_required")
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        previous = db.execute("SELECT digest,response FROM requests WHERE scope=? AND key=?", (scope, key)).fetchone()
        if previous:
            if previous[0] != digest:
                _error("Ключ запроса уже использован с другими данными.", "idempotency_conflict", 409)
            return key, digest, json.loads(previous[1])
        if db.execute("SELECT count(*) FROM requests").fetchone()[0] >= 20000:
            _error("Хранилище прототипа заполнено.", "storage_full", 507)
        return key, digest, None

    def _write(self, db, record, action, scope, key, digest):
        encoded = _json(record)
        # Bound each snapshot and the complete database, including audit and retries.
        if len(encoded.encode()) > 128 * 1024:
            _error("Эпизод превышает допустимый размер.", "episode_limit", 413)
        if self.database.stat().st_size + len(encoded.encode()) * 3 + 8192 > 256 * 1024 * 1024:
            _error("Хранилище прототипа заполнено. Сохраните экспорт.", "storage_full", 507)
        previous = db.execute("SELECT hash FROM audit WHERE episode_id=? ORDER BY revision DESC LIMIT 1", (record["episode_id"],)).fetchone()
        previous_hash = previous[0] if previous else ""
        audit_hash = hashlib.sha256((previous_hash + action + encoded).encode()).hexdigest()
        db.execute("INSERT OR REPLACE INTO episodes(id,revision,data) VALUES(?,?,?)", (record["episode_id"], record["revision"], encoded))
        db.execute("INSERT INTO audit(episode_id,revision,occurred_at,action,previous_hash,hash,snapshot) VALUES(?,?,?,?,?,?,?)", (record["episode_id"], record["revision"], record["updated_at"], action, previous_hash, audit_hash, encoded))
        db.execute("INSERT INTO requests(scope,key,digest,response) VALUES(?,?,?,?)", (scope, key, digest, encoded))
        return copy.deepcopy(record)

    def create(self, payload):
        with self.lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            key, digest, replay = self._request(payload, "create", db)
            if replay is not None:
                return replay
            if set(payload) - {"idempotency_key", "patient_code", "identity_confirmed", "adult_confirmed", "consent", "upload_id"}:
                _error("Неизвестное поле нового эпизода.")
            if db.execute("SELECT count(*) FROM episodes").fetchone()[0] >= 100:
                _error("Достигнут лимит эпизодов прототипа.", "episode_limit", 507)
            now = _now()
            record = {"schema_version": "1.0", "protocol_version": VERSION, "episode_id": "episode-" + uuid.uuid4().hex, "patient_id": "patient-" + uuid.uuid4().hex, "patient_code": _text(payload.get("patient_code", ""), 80, False), "revision": 1, "created_at": now, "updated_at": now, "identity_confirmed": False, "adult_confirmed": False, "consent": False, "upload_id": None, "answers": {}, "answer_records": {}, "pain_map": [], "safety_events": [], "exam": {"clinician": "", "notes": {}, "observations": {}, "safety_answers": {}}, "intake_status": "draft", "exam_status": "draft"}
            self._identity(record, payload)
            record["safety"] = _screen({}, record["exam"], {})
            record["summary"] = _summary(record)
            return self._write(db, record, "create", "create", key, digest)

    def _identity(self, record, payload):
        if "patient_code" in payload and _text(payload["patient_code"], 80, False) != record["patient_code"]:
            _error("Для другого пациента создайте новый эпизод.", "identity_conflict", 409)
        for field in ("identity_confirmed", "adult_confirmed", "consent"):
            if field in payload:
                if type(payload[field]) is not bool:
                    _error()
                record[field] = payload[field]
        if "upload_id" in payload:
            identifier = payload["upload_id"]
            if identifier is not None and (not isinstance(identifier, str) or not UPLOAD.fullmatch(identifier)):
                _error("Некорректная ссылка на МРТ.")
            if record["upload_id"] is not None and record["upload_id"] != identifier:
                _error("МРТ уже связана с эпизодом. Для другой привязки создайте новый эпизод.", "identity_conflict", 409)
            if identifier and not record["identity_confirmed"]:
                _error("Подтвердите принадлежность исследования пациенту.", "identity_required")
            record["upload_id"] = identifier

    def save(self, identifier, payload):
        with self.lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            key, digest, replay = self._request(payload, identifier, db)
            if replay is not None:
                return replay
            before = self._get(db, identifier)
            if type(payload.get("revision")) is not int or payload["revision"] != before["revision"]:
                _error("Эпизод изменён в другой вкладке. Обновите данные перед сохранением.", "revision_conflict", 409)
            if before["revision"] >= 2000:
                _error("Достигнут лимит изменений эпизода. Сохраните экспорт.", "revision_limit", 507)
            allowed = {"revision", "idempotency_key", "action", "patient_code", "identity_confirmed", "adult_confirmed", "consent", "upload_id", "answers", "pain_map", "exam", "alert_id", "acknowledgement", "acknowledged_by", "safety_events"}
            if set(payload) - allowed:
                _error("Неизвестное поле эпизода.")
            action = payload.get("action", "save")
            if action not in {"save", "submit_intake", "review_exam", "acknowledge_alert", "record_safety"}:
                _error("Неизвестное действие.")
            if action == "record_safety" and set(payload) - {"revision", "idempotency_key", "action", "safety_events"}:
                _error("Запись сигнала должна содержать только события скрининга.")
            record = copy.deepcopy(before)
            self._identity(record, payload)
            if any(k in payload for k in ("answers", "pain_map", "exam", "safety_events")) and not all(record[k] for k in ("identity_confirmed", "adult_confirmed", "consent")):
                _error("Перед вводом данных подтвердите пациента, возраст 18+ и согласие.", "consent_required")
            if "answers" in payload:
                record["answers"] = _answers(payload["answers"])
                for qid in set(before["answers"]) | set(record["answers"]):
                    if before["answers"].get(qid) != record["answers"].get(qid):
                        record["answer_records"][qid] = {"record_id": "fact-" + uuid.uuid4().hex, "source": "patient_self_report", "recorded_at": _now(), "question_version": VERSION, "status": "observed" if record["answers"].get(qid) else "unknown"}
            if "pain_map" in payload:
                record["pain_map"] = _map(payload["pain_map"])
            if "exam" in payload:
                record["exam"] = _exam(payload["exam"])
                prior_normalized = _exam(before["exam"])["observations"]
                for obs_key, obs in record["exam"]["observations"].items():
                    prior = before["exam"]["observations"].get(obs_key, {})
                    changed = obs != prior_normalized.get(obs_key)
                    obs["record_id"] = "fact-" + uuid.uuid4().hex if changed else prior.get("record_id", "fact-" + uuid.uuid4().hex)
                    obs["recorded_at"] = _now() if changed else prior.get("recorded_at", _now())
            if record["answers"] != before["answers"] or record["pain_map"] != before["pain_map"]:
                record["intake_status"] = "draft"
            # Any substantive amendment removes current review while preserving its snapshot.
            if _exam(record["exam"]) != _exam(before["exam"]) or record["answers"] != before["answers"] or record["pain_map"] != before["pain_map"]:
                record["exam_status"] = "draft"
                for review_field in ("reviewed_by", "reviewed_at", "review_revision"):
                    record["exam"].pop(review_field, None)
            elif before["exam_status"] == "reviewed":
                for f in ("reviewed_by", "reviewed_at", "review_revision"):
                    if f in before["exam"]:
                        record["exam"][f] = before["exam"][f]
            previous_safety = self._safety_events(record, payload.get("safety_events", []))
            record["safety"] = _screen(record["answers"], record["exam"], previous_safety, record["pain_map"])
            if action == "acknowledge_alert":
                alert_id = payload.get("alert_id")
                alert = next((a for a in record["safety"]["alerts"] if a["id"] == alert_id), None)
                if alert is None:
                    _error("Предупреждение не найдено.", "not_found", 404)
                alert.update(acknowledged=True, acknowledged_at=_now(), acknowledgement=_text(payload.get("acknowledgement", ""), 2000, False), acknowledged_by=_text(payload.get("acknowledged_by", ""), 120, False))
            if action == "submit_intake":
                if not all(record[k] for k in ("identity_confirmed", "adult_confirmed", "consent")) or record["answers"].get("Q29", {}).get("value") != "yes":
                    _error("Подтвердите пациента, согласие и актуальность ответов.", "confirmation_required")
                record["intake_status"] = "emergency_alert" if record["safety"]["level"] == "E" else "needs_review" if record["safety"]["level"] in {"U", "X"} else "submitted"
            if action == "review_exam":
                self._review(record)
            if record["safety"]["level"] == "E":
                record["exam_status"] = "emergency_stop"
            record["revision"] += 1
            record["updated_at"] = _now()
            record["summary"] = _summary(record)
            return self._write(db, record, action, identifier, key, digest)

    def _safety_events(self, record, events):
        """Replay captured raw inputs, including unsent positives later corrected.

        This deliberately never replaces current questionnaire/exam data. Client
        severity assertions are not accepted; each event is evaluated here.
        """
        if not isinstance(events, list) or len(events) > 100:
            _error("За один раз можно передать не более 100 событий скрининга.")
        history = record.setdefault("safety_events", [])
        existing = {event["id"]: event for event in history}
        safety = record["safety"]
        for event in events:
            if not isinstance(event, dict) or set(event) - {"id", "occurred_at", "answers", "exam_answers", "pain_map"}:
                _error("Некорректное событие скрининга.")
            identifier = event.get("id")
            if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]{8,100}", identifier):
                _error("У события скрининга должен быть уникальный идентификатор.")
            occurred_at = event.get("occurred_at")
            try:
                if not isinstance(occurred_at, str) or datetime.fromisoformat(occurred_at.replace("Z", "+00:00")).tzinfo is None:
                    _error("Укажите время события с часовым поясом.")
            except (ValueError, TypeError):
                _error("Некорректное время события скрининга.")
            normalized = {"id": identifier, "occurred_at": occurred_at,
                          "answers": _answers(event.get("answers", {})),
                          "exam_answers": _answers(event.get("exam_answers", {}), True),
                          "pain_map": _map(event.get("pain_map", []))}
            digest = hashlib.sha256(_json(event).encode()).hexdigest()
            if identifier in existing:
                if existing[identifier]["input_hash"] != digest:
                    _error("Событие с этим идентификатором уже записано с другими данными.", "event_conflict", 409)
                continue
            if len(history) >= 200:
                _error("Достигнут лимит событий эпизода. Сохраните экспорт.", "event_limit", 507)
            previous_ids = {alert["id"] for alert in safety["alerts"]}
            safety = _screen(normalized["answers"], {"safety_answers": normalized["exam_answers"]}, safety, normalized["pain_map"])
            received_at = _now()
            for alert in safety["alerts"]:
                if alert["id"] not in previous_ids:
                    alert.update(source_event_id=identifier, client_occurred_at=occurred_at, received_at=received_at)
            stored = {**normalized, "received_at": received_at, "input_hash": digest}
            history.append(stored)
            existing[identifier] = stored
        return safety

    def _review(self, record):
        if not all(record[k] for k in ("identity_confirmed", "adult_confirmed", "consent")):
            _error("Подтвердите пациента, возраст и согласие.", "confirmation_required")
        if record["safety"]["level"] == "E":
            # Record an attempted review, retain all data, never label it reviewed.
            return
        exam = record["exam"]
        if not exam["clinician"] or not all(exam["notes"].get(k) for k in ("N00", "N01")):
            _error("Укажите специалиста, проверку пациента и ограничения обследования.", "review_incomplete")
        for test in TESTS:
            for side in test["sides"]:
                obs = exam["observations"].get(test["id"] + "_" + side)
                if obs is None or (obs["status"] != "observed" and not obs["reason"]):
                    _error("Для невыполненных проб укажите явный статус и причину.", "review_incomplete")
        exam.update(reviewed_by=exam["clinician"], reviewed_at=_now(), review_revision=record["revision"] + 1)
        record["exam_status"] = "reviewed"

    def export(self, identifier):
        with self._connect() as db:
            record = self._get(db, identifier)
            record["history"] = [{"revision": row[0], "occurred_at": row[1], "action": row[2], "previous_hash": row[3], "hash": row[4], "snapshot": json.loads(row[5])} for row in db.execute("SELECT revision,occurred_at,action,previous_hash,hash,snapshot FROM audit WHERE episode_id=? ORDER BY revision", (identifier,))]
        facts = []
        base = {"schema_version": "1.0", "patient_id": record["patient_id"], "episode_id": identifier, "protocol_version": VERSION, "regulatory_status": "research_draft", "authentication": "shared_private_link"}
        for qid, value in record["answers"].items():
            meta = record["answer_records"].get(qid, {})
            facts.append({**base, **meta, "kind": "patient_report", "code": qid, "value": value, "anatomy": {"region": "lumbar", "side": value.get("side", "unknown")}, "source_record_ids": [meta.get("record_id")] if meta.get("record_id") else []})
        for point in record["pain_map"]:
            facts.append({**base, "record_id": point["id"], "kind": "subjective_pain_area", "code": point["type"], "value": point, "status": "observed", "source": "patient_self_report", "recorded_at": point["recorded_at"], "anatomy": {"region": point["region"], "side": point["side"]}, "source_record_ids": [point["id"]]})
        for key, obs in record["exam"]["observations"].items():
            facts.append({**base, "record_id": obs["record_id"], "kind": "neuro_observation", "code": obs["test_id"], "value": obs, "status": obs["status"], "source": "clinician_exam", "recorded_at": obs["recorded_at"], "anatomy": {"region": "lumbar", "side": obs["side"]}, "source_record_ids": [obs["record_id"]]})
        record["feature_records"] = facts
        return record
