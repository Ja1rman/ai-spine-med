"use strict";

(() => {
  const root = new URL("./", window.location.href);
  const $ = (id) => document.getElementById(id);
  const grades = ["Норма / лёгкая степень", "Умеренный стеноз", "Тяжёлый стеноз"];
  const levels = ["L4-L5", "L5-S1"];
  const probabilityNames = ["Норма / лёгкий", "Умеренный", "Тяжёлый"];
  const statusNames = {queued: "В очереди", processing: "Анализируется", completed: "Готово", partial: "Оценено частично", failed: "Не удалось оценить"};
  const pending = (item) => ["queued", "processing"].includes(item?.status) || Object.values(item?.modules || {}).some((module) => ["queued", "processing"].includes(module?.status));
  const state = {mode: "cohort", index: null, cases: new Map(), patientCases: new Map(), patients: new Map(), patientNumbers: new Map(), patientCache: new Map(), uploads: [], activeUpload: null, api: null, apiAvailable: false, poll: null, sending: false, detailLoading: false, detailRetry: false, navigation: 0, listRequest: 0, active: null, patient: null, series: null, slice: 0, filter: "all", search: "", zoom: 1, panX: 0, panY: 0, pointer: null, request: 0};
  const plural = (n, words) => words[(n % 100 > 10 && n % 100 < 20) ? 2 : n % 10 === 1 ? 0 : (n % 10 > 1 && n % 10 < 5) ? 1 : 2];
  const caseNumber = (item) => String(item.review_order ?? (state.index.cases.indexOf(item) + 1)).padStart(3, "0");
  const patientName = (item) => `Пациент ${String(state.patientNumbers.get(item.patient_id) || "—").padStart(3, "0")}`;
  const grade = (value) => Number.isInteger(value) && value >= 0 && value < 3 ? grades[value] : "Не оценено";
  let morphologyLevel = null;
  let showMorphologyContour = true;
  let activeAnalysisModule = null;
  let moduleSubmitting = false;
  let deletingUpload = null;
  const deletedUploads = new Set();

  function safeURL(relative) {
    const value = new URL(relative, root);
    if (value.origin !== root.origin || !value.pathname.startsWith(root.pathname) || !["http:", "https:"].includes(value.protocol)) throw new Error("Unsupported local resource");
    return value;
  }

  async function fetchJSON(relative) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 30000);
    try {
      const response = await fetch(safeURL(relative), {cache: "no-store", credentials: "same-origin", referrerPolicy: "no-referrer", signal: controller.signal});
      if (!response.ok) throw new Error("Resource unavailable");
      return await response.json();
    } finally { clearTimeout(timeout); }
  }

  function node(tag, className, content) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (content !== undefined) element.textContent = content;
    return element;
  }

  function validPrediction(prediction) {
    return prediction.status === "ok" && Number.isInteger(prediction.prediction) && prediction.prediction >= 0 && prediction.prediction < 3 && Array.isArray(prediction.probabilities) && prediction.probabilities.length === 3 && prediction.probabilities.every((p) => Number.isFinite(p) && p >= 0 && p <= 1) && Math.abs(prediction.probabilities.reduce((sum, p) => sum + p, 0) - 1) < 0.0001;
  }

  function flags(item) {
    const prediction = item.predictions?.sagittal || {};
    const assessed = validPrediction(prediction) && Number.isInteger(item.label) && item.label >= 0 && item.label < 3;
    return {
      errors: assessed && prediction.prediction !== item.label,
      correct: assessed && prediction.prediction === item.label,
      unassessed: !assessed
    };
  }

  function addBadges(container, item) {
    container.replaceChildren();
    const f = flags(item);
    if (f.errors) container.append(node("span", "badge error", "Расхождение с RSNA"));
    if (f.unassessed) container.append(node("span", "badge unassessed", "Не оценено"));
    if (f.correct) container.append(node("span", "badge control", "Совпало с RSNA"));
  }

  function filteredCases() {
    const needle = state.search.trim().toLocaleLowerCase("ru");
    return state.index.cases.filter((item) => {
      if (state.filter !== "all" && !flags(item)[state.filter]) return false;
      const text = `${caseNumber(item)} ${item.case_id} ${patientName(item)} ${item.patient_id} ${item.level}`.toLocaleLowerCase("ru");
      return !needle || text.includes(needle);
    });
  }

  function renderList() {
    if (state.mode === "uploads") { renderUploadsList(); return; }
    if (!state.index) return;
    const list = $("case-list"), matched = filteredCases();
    list.replaceChildren();
    $("list-count").textContent = `${matched.length} из ${state.index.cases.length} ${plural(state.index.cases.length, ["случая", "случаев", "случаев"])}`;
    for (const item of matched) {
      const button = node("button", "case-option" + (state.active?.case_id === item.case_id ? " active" : ""));
      button.type = "button";
      button.dataset.case = item.case_id;
      button.setAttribute("aria-current", state.active?.case_id === item.case_id ? "true" : "false");
      const title = node("span", "case-option-title");
      title.append(node("span", "case-number", `Случай ${caseNumber(item)}`), node("span", "case-level", item.level));
      const badges = node("span", "badges");
      addBadges(badges, item);
      button.append(title, node("div", "case-option-patient", patientName(item)), badges);
      button.addEventListener("click", () => activateCase(item.case_id, {history: "push"}));
      list.append(button);
    }
    if (!matched.length) list.append(node("p", "empty", "Нет случаев с таким фильтром. Попробуйте другой фильтр или поиск."));
    updateCaseButtons();
  }

  function updateCaseButtons() {
    const list = state.mode === "uploads" ? filteredUploads() : state.index ? filteredCases() : [];
    const index = list.findIndex((item) => state.mode === "uploads" ? item.upload_id === state.activeUpload?.upload_id : item.case_id === state.active?.case_id);
    $("previous-case").disabled = index <= 0;
    $("next-case").disabled = index < 0 || index >= list.length - 1;
  }

  function moveCase(direction) {
    const list = state.mode === "uploads" ? filteredUploads() : filteredCases();
    const index = list.findIndex((item) => state.mode === "uploads" ? item.upload_id === state.activeUpload?.upload_id : item.case_id === state.active?.case_id);
    if (index >= 0 && list[index + direction]) {
      if (state.mode === "uploads") activateUpload(list[index + direction].upload_id, {history: "push"});
      else activateCase(list[index + direction].case_id, {history: "push"});
    }
  }

  function processingExplanation(reason) {
    const missing = /^(?:sagittal:)?missing_vertebra:((?:L[1-5]|S1)(?:,(?:L[1-5]|S1))*)$/.exec(reason);
    let explanation = "входные серии не прошли необходимую техническую обработку";
    if (missing) {
      const landmarks = missing[1].split(",");
      const names = landmarks.length > 1 ? `${landmarks.slice(0, -1).join(", ")} и ${landmarks.at(-1)}` : landmarks[0];
      explanation = `сагиттальный этап не определил ${landmarks.length > 1 ? "ориентиры" : "ориентир"} ${names}`;
    }
    return `Не удалось получить оценку: ${explanation}. Такой результат не означает отсутствие стеноза.`;
  }

  function predictionCard(prediction, label, reference = null, {cohort = false, issue = ""} = {}) {
      const ok = validPrediction(prediction);
      const hasReference = Number.isInteger(reference) && reference >= 0 && reference < 3;
      const match = ok && prediction.prediction === reference;
      const card = node("div", "prediction-item");
      const top = node("div", "prediction-top");
      top.append(node("span", "prediction-model", label));
      if (hasReference || !ok) top.append(node("span", `comparison-label ${ok ? match ? "match" : "mismatch" : ""}`, ok ? match ? "Совпало с RSNA" : "Не совпало с RSNA" : "Не оценено"));
      const probabilities = node("div", "prediction-probs");
      for (let i = 0; i < 3; i += 1) {
        const value = prediction.probabilities?.[i];
        const field = node("div", "prob" + (ok && prediction.prediction === i ? " selected" : ""));
        const score = ok && Number.isFinite(value) ? Math.min(100, Math.max(0, value * 100)) : 0;
        field.style.setProperty("--score", `${score}%`);
        field.append(node("span", "", probabilityNames[i]), node("strong", "", ok && Number.isFinite(value) ? `${(value * 100).toFixed(1).replace(".", ",")}%` : "—"));
        probabilities.append(field);
      }
      card.append(top, node("div", "prediction-grade", ok ? grade(prediction.prediction) : "Не оценено"));
      if (cohort) card.append(node("p", "prediction-reference", `RSNA: ${hasReference ? grade(reference) : "метка недоступна"}`));
      card.append(probabilities);
      if (issue) card.append(node("p", "small-muted", issue));
      else if (!ok && prediction.reason) card.append(node("p", "small-muted", processingExplanation(prediction.reason)));
      return card;
  }

  function patientLevels(item) {
    const samePatient = state.patientCases.get(item.patient_id) || [];
    return levels.map((level) => {
      const matches = samePatient.filter((candidate) => candidate.level === level);
      return {
        level,
        item: matches.length === 1 ? matches[0] : null,
        issue: matches.length === 1 ? "" : matches.length ? "В подборке несколько записей этого уровня; оценка не выбрана." : "В подборке нет записи для этого уровня."
      };
    });
  }

  function renderPatientBadges(container, entries) {
    container.replaceChildren();
    const results = entries.map((entry) => flags(entry.item || {}));
    if (results.some((result) => result.errors)) container.append(node("span", "badge error", "Есть расхождение с RSNA"));
    if (results.some((result) => result.unassessed)) container.append(node("span", "badge unassessed", "Есть неоценённый уровень"));
    if (results.every((result) => result.correct)) container.append(node("span", "badge control", "Оба уровня совпали с RSNA"));
  }

  function renderLevelFocus(item, entries) {
    $("level-focus-panel").hidden = false;
    const container = $("level-focus");
    container.replaceChildren();
    for (const entry of entries) {
      const button = node("button", "level-focus-button" + (entry.item?.case_id === item.case_id ? " active" : ""), entry.level);
      button.type = "button";
      button.dataset.level = entry.level;
      button.disabled = !entry.item;
      button.setAttribute("aria-pressed", entry.item?.case_id === item.case_id ? "true" : "false");
      if (entry.item) button.addEventListener("click", () => activateCase(entry.item.case_id, {history: "push"}));
      container.append(button);
    }
    $("level-focus-label").textContent = `Ссылки и обзор: ${item.level} · случай ${caseNumber(item)}`;
  }

  function renderPredictions(item) {
    const entries = patientLevels(item);
    $("predictions").replaceChildren();
    for (const entry of entries) {
      const prediction = entry.item ? {...entry.item.predictions?.sagittal, reason: entry.item.processing_reason || entry.item.predictions?.sagittal?.reason || ""} : {};
      $("predictions").append(predictionCard(prediction, entry.level, entry.item?.label, {cohort: true, issue: entry.issue}));
    }
    renderPatientBadges($("case-badges"), entries);
    renderLevelFocus(item, entries);
    $("processing-reason").hidden = true;
    $("processing-reason").textContent = "";
    renderPatientSummary(entries.map((entry) => ({level: entry.level, ...(entry.item?.predictions?.sagittal || {})})), null, true);
  }

  function renderPatientSummary(targets, summary = null, cohort = false) {
    const findings = $("patient-summary-findings");
    findings.replaceChildren();
    const knownFindings = Array.isArray(summary?.findings) ? summary.findings.filter((value) => typeof value === "string") : targets.filter(validPrediction).map((target) => `${target.level} · центральный стеноз по модели — ${grade(target.prediction).toLocaleLowerCase("ru")}.`);
    for (const text of knownFindings) findings.append(node("li", "", text));
    const unassessed = Array.isArray(summary?.unassessed) ? summary.unassessed.filter((value) => typeof value === "string") : targets.filter((target) => !validPrediction(target)).map((target) => `${target.level} — центральный стеноз не оценён.`);
    $("patient-summary-text").textContent = typeof summary?.text === "string" ? summary.text : cohort ? "Автоматические оценки двух уровней исследования RSNA. Исходные метки набора приведены в карточках ниже." : "Оценки моделей по доступным сериям исследования.";
    $("patient-summary-unassessed").textContent = unassessed.join(" ");
    $("patient-summary-unassessed").hidden = !unassessed.length;
    $("patient-summary").hidden = !knownFindings.length && !unassessed.length && !summary?.text;
  }

  function clearAnalysisModules() {
    morphologyLevel = null;
    showMorphologyContour = true;
    activeAnalysisModule = null;
    $("analysis-modules").hidden = true;
    $("module-tabs").replaceChildren();
    $("module-content").replaceChildren();
    $("module-feedback").hidden = true;
    $("module-feedback").textContent = "";
    $("run-analysis-modules").hidden = true;
    $("morphology-overlay").hidden = true;
    $("patient-summary").hidden = true;
    $("patient-summary-findings").replaceChildren();
    $("patient-summary-text").textContent = "";
    $("patient-summary-unassessed").textContent = "";
  }

  const morphologyNames = {extrusion: "Экструзия", protrusion: "Протрузия", indeterminate: "Не определена"};

  function selectedMorphology() {
    if (state.mode !== "uploads") return null;
    const module = state.activeUpload?.modules?.disc_morphology;
    if (module?.status !== "completed" || !Array.isArray(module.levels)) return null;
    return module.levels.find((level) => level.level === morphologyLevel) || null;
  }

  function morphologyFrame(level) {
    const key = state.series?.slices[state.slice]?.slice_key;
    return level?.frames?.find((frame) => frame.slice_key === key && frame.plane === "sagittal") || null;
  }

  function openMorphologySlice(level) {
    if (!state.patient) return;
    const series = state.patient.series.find((value) => value.slices?.some((slice) => slice.slice_key === level.representative_slice_key));
    if (series) setSeries(series, level.representative_slice_key, "push");
  }

  function drawMorphologyOverlay(frame) {
    const canvas = $("morphology-overlay");
    canvas.hidden = true;
    if (!showMorphologyContour || !frame || !Number.isInteger(frame.width) || !Number.isInteger(frame.height) || frame.width < 1 || frame.height < 1 || frame.width > 2048 || frame.height > 2048) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    canvas.width = frame.width;
    canvas.height = frame.height;
    context.clearRect(0, 0, canvas.width, canvas.height);
    const validPoint = (point) => Array.isArray(point) && point.length === 2 && point.every(Number.isFinite) && point[0] >= 0 && point[0] <= frame.width && point[1] >= 0 && point[1] <= frame.height;
    context.lineWidth = Math.max(1.2, frame.width / 380);
    context.strokeStyle = "#7ce3ce";
    for (const polygon of frame.mask_polygons_xy || []) {
      if (!Array.isArray(polygon) || polygon.length < 3 || !polygon.every(validPoint)) continue;
      context.beginPath();
      polygon.forEach((point, i) => i ? context.lineTo(...point) : context.moveTo(...point));
      context.closePath();
      context.stroke();
    }
    if (Array.isArray(frame.reference_line_xy) && frame.reference_line_xy.length === 2 && frame.reference_line_xy.every(validPoint)) {
      context.strokeStyle = "#e9b887";
      context.setLineDash([5, 4]);
      context.beginPath();
      context.moveTo(...frame.reference_line_xy[0]);
      context.lineTo(...frame.reference_line_xy[1]);
      context.stroke();
      context.setLineDash([]);
    }
    context.font = `${Math.max(8, frame.width / 44)}px sans-serif`;
    for (const [name, symbol, color] of [["base", "B", "#e9b887"], ["displaced", "D", "#e6a4b4"], ["depth", "H", "#a6b9ed"]]) {
      const segment = frame.segments_xy?.[name];
      if (!Array.isArray(segment) || segment.length !== 2 || !segment.every(validPoint)) continue;
      context.strokeStyle = color;
      context.fillStyle = color;
      context.beginPath();
      context.moveTo(...segment[0]);
      context.lineTo(...segment[1]);
      context.stroke();
      context.fillText(symbol, Math.min(frame.width - 10, segment[1][0] + 3), Math.max(10, segment[1][1] - 3));
    }
    canvas.hidden = false;
    canvas.style.transform = $("slice-image").style.transform;
  }

  function renderDiscMorphology(module) {
    const content = $("module-content");
    content.replaceChildren();
    $("morphology-overlay").hidden = true;
    if (module.status !== "completed") {
      const status = {processing: "Морфология дисков анализируется…", queued: "Морфология дисков в очереди", disabled: "Модуль морфологии не подключён", failed: "Не удалось оценить морфологию"};
      content.append(node("p", "module-status", status[module.status] || "Морфология ещё не оценена"));
      if (module.message) content.append(node("p", "small-muted", module.message));
      return;
    }
    const choices = Array.isArray(module.levels) ? module.levels.filter((level) => /^L[1-5]-(?:L[2-5]|S1)$/.test(level.level)) : [];
    if (!choices.length) { content.append(node("p", "small-muted", "Для этого исследования нет результатов по уровням.")); return; }
    if (!choices.some((level) => level.level === morphologyLevel)) morphologyLevel = (choices.find((level) => levels.includes(level.level) && level.candidate_voxels > 0) || choices.find((level) => level.candidate_voxels > 0) || choices[0]).level;
    const picker = node("div", "morphology-levels");
    for (const level of choices) {
      const button = node("button", "level-focus-button" + (level.level === morphologyLevel ? " active" : ""), level.level);
      button.type = "button";
      button.dataset.level = level.level;
      button.setAttribute("aria-pressed", String(level.level === morphologyLevel));
      button.addEventListener("click", () => { morphologyLevel = level.level; renderDiscMorphology(module); });
      picker.append(button);
    }
    content.append(picker);
    const level = selectedMorphology(), frame = morphologyFrame(level);
    const hasCandidate = level?.candidate_voxels > 0;
    content.append(node("p", "module-status", hasCandidate ? "Есть кандидатная маска диска" : "Кандидатная маска не получена"));
    if (hasCandidate) {
      content.append(node("p", "morphology-hypothesis", `Геометрическая гипотеза · ${morphologyNames[level.geometric_candidate] || "Не определена"}`));
      content.append(node("p", "small-muted", "Морфологический тип требует проверки. Контур и опорная линия предложены автоматически."));
    } else content.append(node("p", "small-muted", "Для этого уровня нет кандидатной маски. Наличие грыжи остаётся неоценённым."));
    const caption = frame ? `Измерения текущего среза · ${state.slice + 1} / ${state.series.slices.length}` : "Измерения текущего среза";
    content.append(node("h3", "morphology-slice-caption", caption));
    const metrics = node("dl", "morphology-metrics");
    for (const [symbol, field, description, unit] of [["B", "B_mm", "Основание", " мм"], ["D", "D_mm", "Наружная ширина", " мм"], ["H", "H_mm", "Глубина", " мм"], ["D/B", "D_over_B", "Отношение ширин", ""]]) {
      const value = frame?.[field];
      const entry = node("div", "morphology-metric");
      entry.append(node("dt", "", symbol), node("dd", "", Number.isFinite(value) && value >= 0 ? value.toFixed(2).replace(".", ",") + unit : "—"), node("span", "", description));
      metrics.append(entry);
    }
    content.append(metrics);
    if (frame?.geometry_measurable === false) content.append(node("p", "small-muted morphology-quality", "Геометрия этого среза не прошла проверку пригодности. Доступные числа относятся к кандидатной маске."));
    const qualityLabels = {empty_mask: "На этом срезе маска пуста.", too_few_mask_pixels: "В маске слишком мало пикселей.", disconnected_mask: "Маска содержит раздельные компоненты.", touches_frame_edge: "Маска достигает края снимка.", multiple_base_intervals: "Опорная линия пересекает несколько участков маски.", base_not_intersecting_reference: "Основание не пересекает опорную линию.", missing_base: "Основание не удалось определить.", no_resolved_displaced_tissue: "Наружная часть не разрешается на этом срезе.", reference_unavailable: "Опорная линия недоступна."};
    const qualityNotes = (Array.isArray(frame?.quality_flags) ? frame.quality_flags : []).map((flag) => qualityLabels[flag]).filter(Boolean);
    if (qualityNotes.length) content.append(node("p", "small-muted morphology-quality", qualityNotes.join(" ")));
    if (Number.isFinite(frame?.uncertainty_mm) && frame.uncertainty_mm >= 0) content.append(node("p", "small-muted", `Нижняя оценка неопределённости · ±${frame.uncertainty_mm.toFixed(2).replace(".", ",")} мм`));
    if (!frame) content.append(node("p", "small-muted", "Для открытого среза измерений нет. Выберите срез с маской или перелистайте серию."));
    const actions = node("div", "morphology-actions");
    const jump = node("button", "button button-quiet", "К срезу с маской ↗");
    jump.type = "button";
    jump.disabled = !level?.representative_slice_key || !state.patient?.series?.some((series) => series.slices?.some((slice) => slice.slice_key === level.representative_slice_key));
    jump.addEventListener("click", () => openMorphologySlice(level));
    actions.append(jump);
    const toggle = node("button", "button button-quiet", showMorphologyContour ? "Скрыть контур" : "Показать контур");
    toggle.type = "button";
    toggle.setAttribute("aria-pressed", String(showMorphologyContour));
    toggle.addEventListener("click", () => { showMorphologyContour = !showMorphologyContour; renderDiscMorphology(module); });
    actions.append(toggle);
    content.append(actions, node("p", "small-muted morphology-legend", "Зелёный контур · кандидатная маска. Пунктир · опорная линия. B, D и H рассчитаны относительно этой линии."));
    if (["L1-L2", "L2-L3", "L3-L4"].includes(level.level)) content.append(node("p", "small-muted", "Обучающее покрытие весов для этого уровня не подтверждено."));
    drawMorphologyOverlay(frame);
  }

  const moduleRenderers = {disc_morphology: {title: "Морфология дисков", render: renderDiscMorphology}};

  function renderAnalysisModules(item) {
    const tabs = $("module-tabs");
    tabs.replaceChildren();
    const available = Object.entries(moduleRenderers).filter(([id]) => item.modules?.[id]);
    $("analysis-modules").hidden = !available.length && !item.modules_available;
    $("run-analysis-modules").hidden = !item.modules_available;
    $("run-analysis-modules").disabled = moduleSubmitting || !!deletingUpload || pending(item) || !item.patient?.series?.some((series) => series.slices?.length);
    $("run-analysis-modules").textContent = item.modules?.disc_morphology?.status === "completed" ? "Повторить анализ морфологии" : "Оценить морфологию";
    if (!available.length) {
      $("module-content").replaceChildren(...(item.modules_available ? [node("p", "small-muted", "Морфология этого исследования ещё не оценена. Можно запустить подключённый модуль отдельно.")] : []));
      $("morphology-overlay").hidden = true;
      return;
    }
    if (!available.some(([id]) => id === activeAnalysisModule)) activeAnalysisModule = available[0][0];
    for (const [id, renderer] of available) {
      const selected = id === activeAnalysisModule;
      const tab = node("button", "module-tab" + (selected ? " active" : ""), renderer.title);
      tab.type = "button";
      tab.setAttribute("aria-pressed", String(selected));
      tab.addEventListener("click", () => { activeAnalysisModule = id; renderAnalysisModules(item); });
      tabs.append(tab);
    }
    const [id, renderer] = available.find(([id]) => id === activeAnalysisModule);
    renderer.render(item.modules[id]);
  }

  async function runAnalysisModules() {
    const item = state.activeUpload;
    if (state.mode !== "uploads" || !item?.modules_available || moduleSubmitting || deletingUpload || pending(item)) return;
    const id = item.upload_id;
    const navigation = state.navigation;
    moduleSubmitting = true;
    updateDeleteButton();
    $("run-analysis-modules").disabled = true;
    $("module-feedback").hidden = false;
    $("module-feedback").textContent = "Добавляю анализ в очередь…";
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 30000);
    try {
      const response = await fetch(safeURL(`api/uploads/${encodeURIComponent(id)}/modules`), {method: "POST", headers: {"X-Upload-Intent": "dicom-series"}, body: "", credentials: "same-origin", referrerPolicy: "no-referrer", signal: controller.signal});
      if (!response.ok) throw new Error("Module request rejected");
      const queued = await response.json();
      if (queued.upload_id !== id) throw new Error("Invalid module job");
      const index = state.uploads.findIndex((value) => value.upload_id === id);
      if (index >= 0) state.uploads[index] = queued;
      if (navigation === state.navigation && state.mode === "uploads" && state.activeUpload?.upload_id === id) {
        state.activeUpload = queued;
        renderAnalysisModules(queued);
        $("module-feedback").textContent = "Анализ морфологии добавлен в очередь.";
        await refreshUploads({quiet: true, refreshActive: true});
      }
      schedulePoll();
    } catch (_) {
      if (navigation === state.navigation && state.mode === "uploads" && state.activeUpload?.upload_id === id) {
        $("module-feedback").hidden = false;
        $("module-feedback").textContent = "Не удалось подтвердить запуск. Обновите список перед повторной попыткой.";
      }
    } finally {
      clearTimeout(timer);
      moduleSubmitting = false;
      updateDeleteButton();
      if (state.mode === "uploads" && state.activeUpload) renderAnalysisModules(state.activeUpload);
    }
  }

  function seriesName(series) {
    const value = series.series_description || "МР-серия";
    if (/Sagittal T2/i.test(value)) return "Сагиттальная T2 / STIR";
    if (/Sagittal T1/i.test(value)) return "Сагиттальная T1";
    if (/Sagittal MRI/i.test(value)) return "Сагиттальная МР-серия";
    if (/Axial T2/i.test(value)) return "Аксиальная T2";
    return value;
  }

  function defaultView(item) {
    const input = item.model_input;
    if (typeof input?.series_key === "string" && Array.isArray(input.layers)) {
      const real = input.layers.filter((layer) => layer && !layer.padding && Number.isInteger(layer.roi_layer) && typeof layer.slice_key === "string" && layer.slice_key);
      real.sort((a, b) => Math.abs(a.roi_layer - 5) - Math.abs(b.roi_layer - 5) || a.roi_layer - b.roi_layer);
      if (real.length) return {series: input.series_key, slice: real[0].slice_key};
    }
    for (const link of item.view_links || []) {
      try {
        const query = safeURL(link.url).searchParams;
        if (query.get("series")) return {series: query.get("series"), slice: query.get("slice")};
      } catch (_) { /* Ignore a malformed exported shortcut, never load an external URL. */ }
    }
    return {};
  }

  function renderShortcuts(item) {
    const links = $("view-links");
    links.replaceChildren();
    for (const link of item.view_links || []) {
      try {
        const url = safeURL(link.url);
        const anchor = node("a", "view-link", link.label || "Открыть срез");
        anchor.href = url.href;
        anchor.addEventListener("click", (event) => {
          event.preventDefault();
          activateCase(url.searchParams.get("case") || item.case_id, {series: url.searchParams.get("series"), slice: url.searchParams.get("slice"), history: "push"});
        });
        links.append(anchor);
      } catch (_) { /* An invalid link does not prevent reviewing other slices. */ }
    }
    $("quick-links").hidden = !links.childElementCount;
  }

  async function activateCase(id, options = {}) {
    if (state.mode !== "cohort") return;
    const item = state.cases.get(id);
    if (!item) return;
    ++state.navigation;
    const request = ++state.request;
    state.active = item;
    state.activeUpload = null;
    state.series = null;
    state.patient = null;
    state.detailLoading = false;
    state.detailRetry = false;
    clearAnalysisModules();
    $("series-picker").disabled = true;
    disableDownloads();
    updateHistory(options);
    $("workspace").setAttribute("aria-busy", "true");
    $("fatal-error").hidden = true;
    $("case-content").hidden = false;
    $("upload-empty").hidden = true;
    $("upload-status").hidden = true;
    $("reference-card").hidden = true;
    $("viewer-card").hidden = false;
    $("case-grid").hidden = false;
    $("case-grid").classList.remove("without-images");
    $("prediction-title").textContent = "Степень центрального стеноза";
    $("model-name").textContent = state.index.model?.title || "SpineNet · frozen-head v1 · ансамбль 3 запусков";
    $("case-loading").hidden = false;
    $("case-context").textContent = `${patientName(item)} · исследование RSNA`;
    $("case-title").textContent = "Оценки L4–L5 и L5–S1";
    $("case-id").textContent = `Идентификатор случая: ${item.case_id}`;
    $("reference-title").textContent = `RSNA · ${item.level} · центральный стеноз`;
    $("reference-grade").textContent = grade(item.label);
    renderPredictions(item);
    renderShortcuts(item);
    $("overview-image").removeAttribute("src");
    $("overview-link").removeAttribute("href");
    $("overview-card").hidden = !item.overview_url;
    if (item.overview_url) try {
        const url = safeURL(item.overview_url);
        $("overview-image").src = url.href;
        $("overview-link").href = url.href;
      } catch (_) { $("overview-card").hidden = true; }
    renderList();
    showImageMessage("Загрузка серий…");
    try {
      let patient = state.patientCache.get(item.patient_id);
      if (!patient) {
        patient = await fetchJSON(`data/patients/${encodeURIComponent(item.patient_id)}.json`);
        state.patientCache.set(item.patient_id, patient);
      }
      if (request !== state.request) return;
      if (patient.patient_id !== item.patient_id || !Array.isArray(patient.series)) throw new Error("Invalid patient index");
      openPatient(patient, item, {...options, history: options.history === "none" ? "none" : "replace"});
    } catch (_) {
      if (request !== state.request) return;
      showImageMessage("Не удалось загрузить серии этого пациента. Попробуйте другой случай или обновите страницу.");
      disableDownloads();
    } finally {
      if (request === state.request) {
        $("case-loading").hidden = true;
        $("workspace").setAttribute("aria-busy", "false");
      }
    }
  }

  function openPatient(patient, item, options = {}) {
    state.patient = patient;
    const series = patient.series.filter((value) => Array.isArray(value.slices) && value.slices.length);
    const picker = $("series-picker");
    picker.replaceChildren();
    picker.disabled = false;
    for (const value of series) {
      const n = value.slices.length;
      const option = node("option", "", `${seriesName(value)} · ${n} ${plural(n, ["срез", "среза", "срезов"])}`);
      option.value = value.series_key;
      picker.append(option);
    }
    $("series-count").textContent = `${series.length} ${plural(series.length, ["серия", "серии", "серий"])}`;
    if (!series.length) throw new Error("No available series");
    const suggested = defaultView(item);
    const chosen = series.find((value) => value.series_key === options.series) || series.find((value) => value.series_key === suggested.series) || series.find((value) => String(value.series_id) === String(item.sagittal_series_id)) || series[0];
    const slice = options.series === chosen.series_key ? options.slice : suggested.series === chosen.series_key ? suggested.slice : null;
    setSeries(chosen, slice, options.history || "replace");
  }

  function setSeries(series, sliceKey = null, historyMode = "replace") {
    state.series = series;
    $("series-picker").value = series.series_key;
    const found = series.slices.findIndex((slice) => slice.slice_key === sliceKey);
    state.slice = found >= 0 ? found : Math.floor(series.slices.length / 2);
    $("slice-slider").max = String(series.slices.length - 1);
    resetZoom();
    updateSlice(historyMode);
  }

  function showImageMessage(text) {
    $("image-message").hidden = false;
    $("image-message").textContent = text;
  }

  function disableDownloads() {
    for (const id of ["download-dicom", "download-zip"]) {
      $(id).removeAttribute("href");
      $(id).classList.add("disabled");
      $(id).setAttribute("aria-disabled", "true");
    }
  }

  function downloadLink(id, relative) {
    const anchor = $(id);
    if (!relative) {
      anchor.removeAttribute("href"); anchor.classList.add("disabled"); anchor.setAttribute("aria-disabled", "true");
      return;
    }
    anchor.href = safeURL(relative).href;
    anchor.classList.remove("disabled");
    anchor.removeAttribute("aria-disabled");
  }

  function updateSlice(historyMode = "replace") {
    if (!state.series) return;
    const series = state.series, slice = series.slices[state.slice];
    const image = $("slice-image"), url = safeURL(slice.image_url).href;
    showImageMessage("Загрузка среза…");
    image.onload = () => { if (image.src === url) $("image-message").hidden = true; };
    image.onerror = () => { if (image.src === url) showImageMessage("Предпросмотр этого среза недоступен. Можно загрузить исходный DICOM."); };
    image.src = url;
    image.alt = `${seriesName(series)}, срез ${state.slice + 1} из ${series.slices.length}`;
    $("slice-caption").textContent = `Срез ${state.slice + 1} / ${series.slices.length}`;
    $("slice-slider").value = String(state.slice);
    $("slice-slider").setAttribute("aria-valuetext", `Срез ${state.slice + 1} из ${series.slices.length}`);
    $("previous-slice").disabled = state.slice === 0;
    $("next-slice").disabled = state.slice === series.slices.length - 1;
    const size = slice.columns && slice.rows ? `${slice.columns} × ${slice.rows} пикс.` : "";
    $("slice-details").textContent = [`${seriesName(series)}`, `InstanceNumber: ${slice.instance_number ?? "—"}`, size].filter(Boolean).join(" · ");
    downloadLink("download-dicom", slice.dicom_url);
    downloadLink("download-zip", series.zip_url);
    if (state.mode === "uploads") renderAnalysisModules(state.activeUpload);
    if (historyMode !== "none") {
      const urlState = new URL(root);
      if (state.mode === "uploads") { urlState.searchParams.set("view", "uploads"); urlState.searchParams.set("upload", state.activeUpload.upload_id); }
      else urlState.searchParams.set("case", state.active.case_id);
      urlState.searchParams.set("series", series.series_key);
      urlState.searchParams.set("slice", slice.slice_key);
      if (historyMode === "push") window.history.pushState({}, "", urlState);
      else window.history.replaceState({}, "", urlState);
    }
  }

  function moveSlice(direction) {
    if (!state.series) return;
    const index = Math.min(state.series.slices.length - 1, Math.max(0, state.slice + direction));
    if (index !== state.slice) { state.slice = index; updateSlice(); }
  }

  function applyZoom() {
    $("slice-image").style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.zoom})`;
    $("morphology-overlay").style.transform = $("slice-image").style.transform;
    $("zoom-value").textContent = `${Math.round(state.zoom * 100)}%`;
  }

  function changeZoom(factor) {
    state.zoom = Math.min(6, Math.max(1, state.zoom * factor));
    if (state.zoom === 1) { state.panX = 0; state.panY = 0; }
    applyZoom();
  }

  function resetZoom() { state.zoom = 1; state.panX = 0; state.panY = 0; applyZoom(); }

  function uploadName(item) {
    const index = state.uploads.findIndex((value) => value.upload_id === item.upload_id);
    return `Загрузка ${String(index < 0 ? "—" : state.uploads.length - index).padStart(3, "0")}`;
  }

  function uploadDate(item) {
    const date = new Date(item.created_at_utc);
    return Number.isNaN(date.getTime()) ? "" : date.toLocaleString("ru", {day: "2-digit", month: "2-digit", year: "numeric", hour: "2-digit", minute: "2-digit"});
  }

  function availableUploadID(preferred) {
    if (preferred && !deletedUploads.has(preferred) && (!state.apiAvailable || state.uploads.some((item) => item.upload_id === preferred))) return preferred;
    return state.uploads[0]?.upload_id;
  }

  function filteredUploads() {
    const needle = state.search.trim().toLocaleLowerCase("ru");
    return state.uploads.filter((item) => !needle || `${uploadName(item)} ${item.upload_id} ${uploadDate(item)} ${statusNames[item.status] || ""} ${item.message || ""}`.toLocaleLowerCase("ru").includes(needle));
  }

  function renderUploadsList() {
    const list = $("case-list"), matched = filteredUploads();
    list.replaceChildren();
    $("list-count").textContent = `${matched.length} из ${state.uploads.length} ${plural(state.uploads.length, ["исследования", "исследований", "исследований"])}`;
    $("collection-count").textContent = `${state.uploads.length} ${plural(state.uploads.length, ["загрузка", "загрузки", "загрузок"])}`;
    for (const item of matched) {
      const button = node("button", "case-option" + (state.activeUpload?.upload_id === item.upload_id ? " active" : ""));
      button.type = "button";
      button.setAttribute("aria-current", String(state.activeUpload?.upload_id === item.upload_id));
      const title = node("span", "case-option-title");
      title.append(node("span", "case-number", uploadName(item)), node("span", "case-level", item.n_slices ? `${item.n_slices} ср.` : ""));
      button.append(title, node("div", "case-option-patient", uploadDate(item)), node("span", `badge ${item.status === "completed" ? "control" : item.status === "failed" ? "error" : "unassessed"}`, statusNames[item.status] || "Статус неизвестен"));
      if (item.status === "failed" && item.message) button.append(node("div", "upload-failure-reason", item.message));
      button.addEventListener("click", () => activateUpload(item.upload_id, {history: "push"}));
      list.append(button);
    }
    if (!matched.length) list.append(node("p", "empty", state.search ? "Поиск не дал результатов." : "Исследования появятся здесь после загрузки."));
    updateCaseButtons();
  }

  function updateCohortCount() {
    if (!state.index) return;
    $("collection-count").textContent = `${state.index.cases.length} ${plural(state.index.cases.length, ["случай", "случая", "случаев"])} · ${state.index.patients.length} ${plural(state.index.patients.length, ["пациент", "пациента", "пациентов"])}`;
  }

  function changeMode(mode) {
    state.mode = mode;
    ++state.request;
    ++state.navigation;
    state.detailLoading = false;
    state.detailRetry = false;
    state.series = null;
    state.patient = null;
    clearAnalysisModules();
    $("series-picker").disabled = true;
    disableDownloads();
    state.search = "";
    $("search").value = "";
    $("search").placeholder = mode === "uploads" ? "Поиск по номеру или дате" : "Поиск по номеру или уровню";
    $("case-content").hidden = true;
    $("delete-upload").hidden = true;
    $("delete-upload-feedback").hidden = true;
    $("fatal-error").hidden = true;
    $("upload-panel").hidden = mode !== "uploads";
    $("filters").hidden = mode === "uploads";
    $("refresh-uploads").hidden = mode !== "uploads";
    $("sidebar-label").textContent = mode === "uploads" ? "Загруженные исследования" : "Случаи · валидация";
    $("sidebar-footnote").textContent = mode === "uploads" ? "Архивы и результаты сохраняются на сервере. Раздел доступен по той же закрытой ссылке." : "Полная валидационная выборка: правильные ответы, расхождения и отказы обработки. Метки RSNA ещё требуют врачебной проверки.";
    for (const [id, active] of [["tab-cohort", mode === "cohort"], ["tab-uploads", mode === "uploads"]]) {
      $(id).classList.toggle("active", active);
      $(id).setAttribute("aria-pressed", String(active));
    }
    $("upload-empty").hidden = mode !== "uploads" || state.uploads.length > 0;
    $("workspace").setAttribute("aria-busy", "false");
    if (mode === "cohort") updateCohortCount();
    renderList();
  }

  function showUploadFeedback(text, error = false) {
    $("upload-feedback").hidden = !text;
    $("upload-feedback").textContent = text;
    $("upload-feedback").classList.toggle("is-error", error);
  }

  function checkUploadFile() {
    const file = $("upload-file").files[0];
    const limit = state.api?.limits?.max_upload_bytes || 256 * 1024 * 1024;
    const valid = file && /\.zip$/i.test(file.name) && file.size > 0 && file.size <= limit;
    $("upload-file-name").textContent = file ? `${file.name} · ${(file.size / 1048576).toLocaleString("ru", {maximumFractionDigits: 1})} МБ` : "Файл не выбран";
    $("upload-submit").disabled = state.sending || !!deletingUpload || !valid || !state.apiAvailable || !state.api?.model?.configured || !$("upload-consent").checked;
    updateDeleteButton();
    if (file && !valid) showUploadFeedback(`Нужен непустой ZIP размером не более ${Math.floor(limit / 1048576)} МБ.`, true);
  }

  async function refreshUploads({quiet = false, refreshActive = false} = {}) {
    const request = ++state.listRequest;
    try {
      const data = await fetchJSON("api/uploads");
      if (request !== state.listRequest) return false;
      if (!Array.isArray(data.uploads)) throw new Error("Invalid upload library");
      state.api = data;
      state.apiAvailable = true;
      state.uploads = data.uploads.filter((item) => !deletedUploads.has(item.upload_id));
      $("uploads-total").textContent = String(state.uploads.length);
      $("upload-limits").textContent = data.model?.configured ? `До ${Math.floor((data.limits?.max_upload_bytes || 268435456) / 1048576)} МБ · одна сагиттальная T2-серия` : "Модель временно недоступна. Загрузка отключена.";
      checkUploadFile();
      if (state.mode === "uploads") {
        renderUploadsList();
        $("upload-empty").hidden = state.uploads.length > 0;
        if (state.activeUpload && !state.uploads.some((item) => item.upload_id === state.activeUpload.upload_id)) {
          ++state.request;
          ++state.navigation;
          clearUploadSelection();
          updateHistory({history: "replace"});
          if (refreshActive && state.uploads.length) await activateUpload(state.uploads[0].upload_id, {history: "replace"});
        } else if (refreshActive && state.activeUpload && !state.detailLoading) await activateUpload(state.activeUpload.upload_id, {history: "none", refresh: true});
      }
      schedulePoll();
      return true;
    } catch (_) {
      if (request !== state.listRequest) return false;
      state.apiAvailable = false;
      $("upload-limits").textContent = "Сервис загрузки недоступен. Попробуйте обновить список позже.";
      if (!quiet) showUploadFeedback("Не удалось получить список загрузок. Уже загруженные файлы не удалены; попробуйте ещё раз.", true);
      $("upload-submit").disabled = true;
      schedulePoll(true);
      return false;
    }
  }

  function schedulePoll(retry = false) {
    clearTimeout(state.poll);
    retry = retry || (state.mode === "uploads" && state.detailRetry);
    if (retry || state.uploads.some(pending)) state.poll = setTimeout(() => refreshUploads({quiet: true, refreshActive: true}), retry ? 10000 : 4000);
  }

  function updateHistory(options = {}) {
    if (options.history === "none") return;
    const url = new URL(root);
    if (state.mode === "uploads") url.searchParams.set("view", "uploads");
    if (state.mode === "uploads" && state.activeUpload) url.searchParams.set("upload", state.activeUpload.upload_id);
    if (state.mode === "cohort" && state.active) url.searchParams.set("case", state.active.case_id);
    if (options.history === "push") window.history.pushState({}, "", url);
    else window.history.replaceState({}, "", url);
  }

  async function activateUpload(id, options = {}) {
    if (state.mode !== "uploads") return;
    if (deletedUploads.has(id)) {
      const next = state.uploads.find((item) => !deletedUploads.has(item.upload_id));
      if (next) return activateUpload(next.upload_id, {...options, history: "replace"});
      clearUploadSelection();
      updateHistory({history: "replace"});
      return;
    }
    if (!options.refresh) ++state.navigation;
    const request = ++state.request;
    state.detailLoading = true;
    const sameStudy = state.activeUpload?.upload_id === id;
    const retainedSeries = sameStudy ? state.series?.series_key : null;
    const retainedSlice = sameStudy ? state.series?.slices[state.slice]?.slice_key : null;
    const summary = state.uploads.find((item) => item.upload_id === id);
    state.activeUpload = summary || {upload_id: id};
    state.active = null;
    if (!options.refresh) {
      state.series = null;
      state.patient = null;
      clearAnalysisModules();
      $("series-picker").disabled = true;
      disableDownloads();
      $("case-content").hidden = true;
      $("delete-upload").hidden = true;
      $("delete-upload-feedback").hidden = true;
      $("workspace").setAttribute("aria-busy", "true");
      updateHistory(options);
    }
    $("upload-empty").hidden = true;
    $("fatal-error").hidden = true;
    renderUploadsList();
    try {
      const item = await fetchJSON(`api/uploads/${encodeURIComponent(id)}`);
      if (request !== state.request || state.mode !== "uploads" || deletedUploads.has(id)) return;
      if (item.upload_id !== id) throw new Error("Invalid upload index");
      state.detailRetry = false;
      state.activeUpload = item;
      $("case-content").hidden = false;
      $("case-loading").hidden = true;
      $("case-context").textContent = uploadDate(item);
      $("case-title").textContent = uploadName(item);
      $("case-id").textContent = `Идентификатор исследования: ${id}`;
      $("open-psp").href = safeURL(`psp.html?upload=${encodeURIComponent(id)}`).href;
      $("case-badges").replaceChildren(node("span", `badge ${item.status === "completed" ? "control" : item.status === "failed" ? "error" : "unassessed"}`, statusNames[item.status] || "Статус неизвестен"));
      $("reference-card").hidden = true;
      $("level-focus-panel").hidden = true;
      $("overview-card").hidden = true;
      $("quick-links").hidden = true;
      $("upload-status").hidden = false;
      updateDeleteButton();
      $("upload-status-label").textContent = statusNames[item.status] || "Статус неизвестен";
      $("upload-status").classList.toggle("is-pending", pending(item));
      $("upload-status-message").textContent = item.message || (pending(item) ? "Исследование обрабатывается. Статус обновляется автоматически; страницу можно закрыть и вернуться позже." : "");
      $("download-report").hidden = !item.report_url;
      if (item.report_url) $("download-report").href = safeURL(item.report_url).href;
      else $("download-report").removeAttribute("href");
      $("prediction-title").textContent = "Степень центрального стеноза";
      $("model-name").textContent = state.api?.model?.name || "SpineNet · frozen-head v1 · ансамбль 3 запусков";
      $("predictions").replaceChildren();
      for (const level of levels) {
        const prediction = (item.targets || []).find((target) => target.level === level) || {};
        $("predictions").append(predictionCard(prediction, level));
      }
      renderPatientSummary(item.targets || [], item.patient_summary);
      $("processing-reason").hidden = true;
      const hasImages = item.patient?.series?.some((series) => series.slices?.length);
      $("case-grid").hidden = pending(item) && !hasImages;
      $("case-grid").classList.toggle("without-images", !hasImages);
      $("viewer-card").hidden = !hasImages;
      if (hasImages) {
        // Polling a finished analysis should not move the slice the reader selected.
        if (!options.refresh || !retainedSeries || !state.patient || state.patient.patient_id !== item.patient.patient_id) {
          openPatient(item.patient, {}, {series: options.series || retainedSeries, slice: options.slice || retainedSlice, history: options.history === "none" ? "none" : "replace"});
        }
      } else {
        state.series = null;
        state.patient = null;
        disableDownloads();
      }
      renderAnalysisModules(item);
      renderUploadsList();
    } catch (_) {
      if (request !== state.request) return;
      state.detailRetry = true;
      schedulePoll(true);
      $("fatal-error").hidden = false;
      $("fatal-error").textContent = "Не удалось открыть это исследование. Обновите список загрузок и попробуйте снова.";
    } finally {
      if (request === state.request) {
        state.detailLoading = false;
        $("workspace").setAttribute("aria-busy", "false");
      }
    }
  }

  function updateDeleteButton() {
    const button = $("delete-upload");
    button.hidden = state.mode !== "uploads" || !state.activeUpload || $("case-content").hidden;
    button.disabled = !!deletingUpload || state.sending || moduleSubmitting;
    button.textContent = deletingUpload === state.activeUpload?.upload_id ? "Удаляю…" : "Удалить исследование";
  }

  function clearUploadSelection() {
    state.activeUpload = null;
    state.series = null;
    state.patient = null;
    state.detailLoading = false;
    state.detailRetry = false;
    state.pointer = null;
    clearAnalysisModules();
    disableDownloads();
    $("series-picker").replaceChildren();
    $("series-picker").disabled = true;
    $("slice-image").onload = null;
    $("slice-image").onerror = null;
    $("slice-image").removeAttribute("src");
    $("download-report").hidden = true;
    $("download-report").removeAttribute("href");
    $("predictions").replaceChildren();
    $("case-title").textContent = "";
    $("case-id").textContent = "";
    $("case-content").hidden = true;
    $("case-grid").hidden = true;
    $("viewer-card").hidden = true;
    $("upload-status").hidden = true;
    $("delete-upload").hidden = true;
    $("delete-upload-feedback").hidden = true;
    $("case-loading").hidden = true;
    $("fatal-error").hidden = true;
    $("upload-empty").hidden = state.uploads.length > 0;
    $("workspace").setAttribute("aria-busy", "false");
  }

  async function deleteActiveUpload() {
    const item = state.activeUpload;
    if (state.mode !== "uploads" || !item || deletingUpload || state.sending || moduleSubmitting || $("delete-upload").disabled) return;
    if (!window.confirm("Удалить исследование вместе со снимками и результатами?")) return;
    const id = item.upload_id;
    const index = filteredUploads().findIndex((value) => value.upload_id === id);
    deletingUpload = id;
    updateDeleteButton();
    checkUploadFile();
    renderAnalysisModules(item);
    $("delete-upload-feedback").hidden = false;
    $("delete-upload-feedback").textContent = "Удаляю исследование…";
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 30000);
    let rejection = "";
    try {
      const response = await fetch(safeURL(`api/uploads/${encodeURIComponent(id)}`), {method: "DELETE", headers: {"X-Upload-Intent": "dicom-series"}, body: "", credentials: "same-origin", referrerPolicy: "no-referrer", signal: controller.signal});
      const result = await response.json();
      if (!response.ok) {
        rejection = typeof result.message === "string" ? result.message : "Сервер не подтвердил удаление. Обновите список и попробуйте ещё раз.";
        throw new Error("Deletion rejected");
      }
      if (result.deleted !== id) throw new Error("Invalid deletion response");
      deletedUploads.add(id);
      ++state.listRequest;
      clearTimeout(state.poll);
      state.uploads = state.uploads.filter((value) => value.upload_id !== id);
      $("uploads-total").textContent = String(state.uploads.length);
      const selected = state.activeUpload?.upload_id === id;
      if (selected) {
        ++state.request;
        ++state.navigation;
        if (state.mode === "uploads") {
          clearUploadSelection();
          updateHistory({history: "replace"});
        } else state.activeUpload = null;
      }
      const navigation = state.navigation;
      renderList();
      if (state.mode === "uploads") showUploadFeedback("Исследование удалено вместе со снимками и результатами.");
      await refreshUploads({quiet: true});
      if (selected && navigation === state.navigation && state.mode === "uploads" && !state.activeUpload) {
        const next = filteredUploads()[Math.min(Math.max(index, 0), filteredUploads().length - 1)];
        if (next) await activateUpload(next.upload_id, {history: "replace"});
      }
      schedulePoll();
    } catch (_) {
      const message = rejection || "Не удалось подтвердить удаление. Обновите список перед повторной попыткой: запрос мог успеть выполниться.";
      if (state.mode === "uploads" && state.activeUpload?.upload_id === id) {
        $("delete-upload-feedback").hidden = false;
        $("delete-upload-feedback").textContent = message;
      } else if (state.mode === "uploads") showUploadFeedback(message, true);
    } finally {
      clearTimeout(timer);
      deletingUpload = null;
      checkUploadFile();
      if (state.mode === "uploads" && state.activeUpload) renderAnalysisModules(state.activeUpload);
    }
  }

  function sendArchive(file) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", safeURL("api/uploads").href);
      xhr.setRequestHeader("Content-Type", "application/zip");
      xhr.setRequestHeader("X-Upload-Intent", "dicom-series");
      xhr.responseType = "json";
      xhr.timeout = 180000;
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          $("upload-progress").value = Math.round(100 * event.loaded / event.total);
          showUploadFeedback(event.loaded === event.total ? "Архив передан. Сервер проверяет файлы…" : `Передача архива · ${Math.round(100 * event.loaded / event.total)}%`);
        }
      };
      xhr.onload = () => {
        if (xhr.status === 202 && typeof xhr.response?.upload_id === "string") resolve(xhr.response);
        else reject(new Error(xhr.status === 413 ? "Архив превышает допустимый размер." : xhr.response?.message || "Сервер не принял архив. Нужна одна полная сагиттальная T2-серия в DICOM."));
      };
      xhr.onerror = xhr.ontimeout = () => reject(new Error("Связь с сервером прервалась. Сначала обновите список: архив мог успеть загрузиться."));
      xhr.send(file);
    });
  }

  async function submitUpload(event) {
    event.preventDefault();
    const file = $("upload-file").files[0];
    if (!file || state.sending || !$("upload-form").reportValidity() || $("upload-submit").disabled) return;
    state.sending = true;
    checkUploadFile();
    $("upload-progress").hidden = false;
    $("upload-progress").value = 0;
    showUploadFeedback("Передаю архив…");
    try {
      const item = await sendArchive(file);
      state.uploads.unshift(item);
      $("upload-file").value = "";
      $("upload-consent").checked = false;
      showUploadFeedback("Исследование добавлено. Анализ выполняется на сервере, статус обновляется автоматически.");
      await refreshUploads({quiet: true});
      if (state.mode === "uploads") await activateUpload(item.upload_id, {history: "push"});
      schedulePoll();
    } catch (error) {
      showUploadFeedback(error.message, true);
    } finally {
      state.sending = false;
      $("upload-progress").hidden = true;
      checkUploadFile();
    }
  }

  function bindControls() {
    $("tab-cohort").addEventListener("click", () => {
      changeMode("cohort");
      if (state.index?.cases.length) activateCase(state.active?.case_id || state.index.cases[0].case_id, {history: "push"});
      else {
        $("fatal-error").hidden = false;
        $("fatal-error").textContent = "Подборка ещё не загружена или недоступна. Обновите страницу; раздел загрузок работает отдельно.";
      }
    });
    $("tab-uploads").addEventListener("click", async () => {
      changeMode("uploads");
      const navigation = state.navigation;
      updateHistory({history: "push"});
      await refreshUploads();
      if (navigation === state.navigation && state.mode === "uploads" && state.uploads.length) activateUpload(availableUploadID(state.activeUpload?.upload_id), {history: "replace"});
    });
    $("refresh-uploads").addEventListener("click", () => refreshUploads({refreshActive: true}));
    $("run-analysis-modules").addEventListener("click", runAnalysisModules);
    $("delete-upload").addEventListener("click", deleteActiveUpload);
    $("upload-file").addEventListener("change", checkUploadFile);
    $("upload-consent").addEventListener("change", checkUploadFile);
    $("upload-form").addEventListener("submit", submitUpload);
    $("search").addEventListener("input", (event) => { state.search = event.target.value; renderList(); });
    for (const button of $("filters").querySelectorAll("button")) button.addEventListener("click", () => {
      state.filter = button.dataset.filter;
      for (const sibling of $("filters").querySelectorAll("button")) { const active = sibling === button; sibling.classList.toggle("active", active); sibling.setAttribute("aria-pressed", String(active)); }
      renderList();
    });
    $("previous-case").addEventListener("click", () => moveCase(-1));
    $("next-case").addEventListener("click", () => moveCase(1));
    $("series-picker").addEventListener("change", (event) => {
      const series = state.patient?.series.find((value) => value.series_key === event.target.value);
      if (series) setSeries(series, null, "push");
    });
    $("slice-slider").addEventListener("input", (event) => { state.slice = Number(event.target.value); updateSlice(); });
    $("previous-slice").addEventListener("click", () => moveSlice(-1));
    $("next-slice").addEventListener("click", () => moveSlice(1));
    $("zoom-in").addEventListener("click", () => changeZoom(1.25));
    $("zoom-out").addEventListener("click", () => changeZoom(1 / 1.25));
    $("zoom-reset").addEventListener("click", resetZoom);
    const viewport = $("viewport");
    viewport.addEventListener("wheel", (event) => { if (state.series) { event.preventDefault(); changeZoom(event.deltaY < 0 ? 1.12 : 1 / 1.12); } }, {passive: false});
    viewport.addEventListener("pointerdown", (event) => {
      viewport.focus({preventScroll: true});
      if (state.zoom <= 1) return;
      state.pointer = {id: event.pointerId, x: event.clientX, y: event.clientY, panX: state.panX, panY: state.panY};
      viewport.setPointerCapture(event.pointerId); viewport.classList.add("dragging");
    });
    viewport.addEventListener("pointermove", (event) => {
      if (!state.pointer || state.pointer.id !== event.pointerId) return;
      state.panX = state.pointer.panX + event.clientX - state.pointer.x;
      state.panY = state.pointer.panY + event.clientY - state.pointer.y;
      applyZoom();
    });
    const stopDrag = () => { state.pointer = null; viewport.classList.remove("dragging"); };
    viewport.addEventListener("pointerup", stopDrag); viewport.addEventListener("pointercancel", stopDrag);
    document.addEventListener("keydown", (event) => {
      if (event.altKey || event.ctrlKey || event.metaKey || event.target.closest("input,select,textarea,[contenteditable=true]")) return;
      if (event.key === "ArrowLeft") { event.preventDefault(); moveSlice(-1); }
      else if (event.key === "ArrowRight") { event.preventDefault(); moveSlice(1); }
      else if (event.key === "+" || event.key === "=") { event.preventDefault(); changeZoom(1.25); }
      else if (event.key === "-") { event.preventDefault(); changeZoom(1 / 1.25); }
      else if (event.key === "0" || event.key === "Escape") resetZoom();
      else if (event.key === "Home" && state.series) { event.preventDefault(); state.slice = 0; updateSlice(); }
      else if (event.key === "End" && state.series) { event.preventDefault(); state.slice = state.series.slices.length - 1; updateSlice(); }
    });
    window.addEventListener("popstate", async () => {
      const query = new URL(window.location.href).searchParams;
      const uploads = query.get("view") === "uploads" || query.has("upload");
      changeMode(uploads ? "uploads" : "cohort");
      const navigation = state.navigation;
      if (uploads) {
        await refreshUploads({quiet: true});
        if (navigation === state.navigation && state.mode === "uploads") {
          const id = availableUploadID(query.get("upload"));
          if (id) activateUpload(id, {series: query.get("series"), slice: query.get("slice"), history: id === query.get("upload") ? "none" : "replace"});
          else updateHistory({history: "replace"});
        }
      } else if (state.index?.cases.length) {
        activateCase(state.cases.has(query.get("case")) ? query.get("case") : state.index.cases[0].case_id, {series: query.get("series"), slice: query.get("slice"), history: "none"});
      }
    });
  }

  async function start() {
    bindControls();
    disableDownloads();
    const query = new URL(window.location.href).searchParams;
    const uploads = query.get("view") === "uploads" || query.has("upload");
    changeMode(uploads ? "uploads" : "cohort");
    const navigation = state.navigation;
    const library = refreshUploads({quiet: true});
    try {
      const index = await fetchJSON("data/index.json");
      if (!Array.isArray(index.cases) || !index.cases.length || !Array.isArray(index.patients)) throw new Error("Invalid collection");
      state.index = index;
      for (const item of index.cases) {
        state.cases.set(item.case_id, item);
        if (typeof item.patient_id === "string") {
          if (!state.patientCases.has(item.patient_id)) state.patientCases.set(item.patient_id, []);
          state.patientCases.get(item.patient_id).push(item);
        }
      }
      index.patients.forEach((patient, i) => { state.patients.set(patient.patient_id, patient); state.patientNumbers.set(patient.patient_id, i + 1); });
      $("cohort-total").textContent = String(index.cases.length);
      if (state.mode === "cohort") updateCohortCount();
      renderList();
      if (navigation !== state.navigation) return;
      if (uploads) {
        await library;
        if (navigation !== state.navigation || state.mode !== "uploads") return;
        const initial = availableUploadID(query.get("upload"));
        if (initial) await activateUpload(initial, {series: query.get("series"), slice: query.get("slice"), history: "replace"});
        else updateHistory({history: "replace"});
      } else {
        const initial = state.cases.has(query.get("case")) ? query.get("case") : index.cases[0].case_id;
        await activateCase(initial, {series: query.get("series"), slice: query.get("slice"), history: "replace"});
      }
    } catch (_) {
      if (navigation !== state.navigation) return;
      $("workspace").setAttribute("aria-busy", "false");
      if (!uploads) {
        $("fatal-error").hidden = false;
        $("fatal-error").textContent = "Не удалось открыть подборку. Обновите страницу или проверьте, что ссылка указана полностью. Раздел загрузок доступен отдельно.";
        $("case-list").replaceChildren(node("p", "empty", "Подборка недоступна"));
        $("collection-count").textContent = "Подборка недоступна";
      } else {
        await library;
        if (navigation !== state.navigation || state.mode !== "uploads") return;
        const initial = availableUploadID(query.get("upload"));
        if (initial) await activateUpload(initial, {history: "replace"});
        else updateHistory({history: "replace"});
      }
    }
  }

  start();
})();
