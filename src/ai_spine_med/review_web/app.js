"use strict";

(() => {
  const root = new URL("./", window.location.href);
  const $ = (id) => document.getElementById(id);
  const grades = ["Норма / лёгкая степень", "Умеренный стеноз", "Тяжёлый стеноз"];
  const probabilityNames = ["Норма / лёгкий", "Умеренный", "Тяжёлый"];
  const statusNames = {queued: "В очереди", processing: "Анализируется", completed: "Готово", partial: "Оценено частично", failed: "Не удалось оценить"};
  const pending = (item) => ["queued", "processing"].includes(item?.status);
  const state = {mode: "cohort", index: null, cases: new Map(), patients: new Map(), patientNumbers: new Map(), patientCache: new Map(), uploads: [], activeUpload: null, api: null, apiAvailable: false, poll: null, sending: false, detailLoading: false, detailRetry: false, navigation: 0, listRequest: 0, active: null, patient: null, series: null, slice: 0, filter: "all", search: "", zoom: 1, panX: 0, panY: 0, pointer: null, request: 0};
  const plural = (n, words) => words[(n % 100 > 10 && n % 100 < 20) ? 2 : n % 10 === 1 ? 0 : (n % 10 > 1 && n % 10 < 5) ? 1 : 2];
  const caseNumber = (item) => String(item.review_order ?? (state.index.cases.indexOf(item) + 1)).padStart(3, "0");
  const patientName = (item) => `Пациент ${String(state.patientNumbers.get(item.patient_id) || "—").padStart(3, "0")}`;
  const grade = (value) => Number.isInteger(value) && value >= 0 && value < 3 ? grades[value] : "Не оценено";

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

  function predictionCard(prediction, label, reference = null) {
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
      card.append(top, node("div", "prediction-grade", ok ? grade(prediction.prediction) : "Не оценено"), probabilities);
      if (!ok && prediction.reason) card.append(node("p", "small-muted", processingExplanation(prediction.reason)));
      return card;
  }

  function renderPredictions(item) {
    $("predictions").replaceChildren(predictionCard(item.predictions?.sagittal || {}, item.level || "Уровень не указан", item.label));
    const reason = item.processing_reason || "";
    $("processing-reason").hidden = !reason;
    $("processing-reason").textContent = reason ? processingExplanation(reason) : "";
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
    $("series-picker").disabled = true;
    disableDownloads();
    updateHistory(options);
    $("workspace").setAttribute("aria-busy", "true");
    $("fatal-error").hidden = true;
    $("case-content").hidden = false;
    $("upload-empty").hidden = true;
    $("upload-status").hidden = true;
    $("reference-card").hidden = false;
    $("viewer-card").hidden = false;
    $("case-grid").hidden = false;
    $("case-grid").classList.remove("without-images");
    $("prediction-title").textContent = "Ответ модели";
    $("model-name").textContent = state.index.model?.title || "SpineNet · frozen-head v1 · ансамбль 3 запусков";
    $("case-loading").hidden = false;
    $("case-context").textContent = `${patientName(item)} · случай для проверки`;
    $("case-title").textContent = `Случай ${caseNumber(item)} · ${item.level}`;
    $("case-id").textContent = `Идентификатор случая: ${item.case_id}`;
    $("reference-title").textContent = `RSNA · ${item.level} · центральный стеноз`;
    $("reference-grade").textContent = grade(item.label);
    addBadges($("case-badges"), item);
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

  function filteredUploads() {
    const needle = state.search.trim().toLocaleLowerCase("ru");
    return state.uploads.filter((item) => !needle || `${uploadName(item)} ${item.upload_id} ${uploadDate(item)} ${statusNames[item.status] || ""}`.toLocaleLowerCase("ru").includes(needle));
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
    $("series-picker").disabled = true;
    disableDownloads();
    state.search = "";
    $("search").value = "";
    $("search").placeholder = mode === "uploads" ? "Поиск по номеру или дате" : "Поиск по номеру или уровню";
    $("case-content").hidden = true;
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
    $("upload-submit").disabled = state.sending || !valid || !state.apiAvailable || !state.api?.model?.configured || !$("upload-consent").checked;
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
      state.uploads = data.uploads;
      $("uploads-total").textContent = String(data.uploads.length);
      $("upload-limits").textContent = data.model?.configured ? `До ${Math.floor((data.limits?.max_upload_bytes || 268435456) / 1048576)} МБ · одна сагиттальная T2-серия` : "Модель временно недоступна. Загрузка отключена.";
      checkUploadFile();
      if (state.mode === "uploads") {
        renderUploadsList();
        $("upload-empty").hidden = state.uploads.length > 0;
        if (refreshActive && state.activeUpload && !state.detailLoading) await activateUpload(state.activeUpload.upload_id, {history: "none", refresh: true});
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
      $("series-picker").disabled = true;
      disableDownloads();
      $("case-content").hidden = true;
      $("workspace").setAttribute("aria-busy", "true");
      updateHistory(options);
    }
    $("upload-empty").hidden = true;
    $("fatal-error").hidden = true;
    renderUploadsList();
    try {
      const item = await fetchJSON(`api/uploads/${encodeURIComponent(id)}`);
      if (request !== state.request || state.mode !== "uploads") return;
      if (item.upload_id !== id) throw new Error("Invalid upload index");
      state.detailRetry = false;
      state.activeUpload = item;
      $("case-content").hidden = false;
      $("case-loading").hidden = true;
      $("case-context").textContent = uploadDate(item);
      $("case-title").textContent = uploadName(item);
      $("case-id").textContent = `Идентификатор исследования: ${id}`;
      $("case-badges").replaceChildren(node("span", `badge ${item.status === "completed" ? "control" : item.status === "failed" ? "error" : "unassessed"}`, statusNames[item.status] || "Статус неизвестен"));
      $("reference-card").hidden = true;
      $("overview-card").hidden = true;
      $("quick-links").hidden = true;
      $("upload-status").hidden = false;
      $("upload-status-label").textContent = statusNames[item.status] || "Статус неизвестен";
      $("upload-status").classList.toggle("is-pending", pending(item));
      $("upload-status-message").textContent = item.message || (pending(item) ? "Исследование обрабатывается. Статус обновляется автоматически; страницу можно закрыть и вернуться позже." : "");
      $("download-report").hidden = !item.report_url;
      if (item.report_url) $("download-report").href = safeURL(item.report_url).href;
      else $("download-report").removeAttribute("href");
      $("prediction-title").textContent = "Степень центрального стеноза";
      $("model-name").textContent = state.api?.model?.name || "SpineNet · frozen-head v1 · ансамбль 3 запусков";
      $("predictions").replaceChildren();
      for (const level of ["L4-L5", "L5-S1"]) {
        const prediction = (item.targets || []).find((target) => target.level === level) || {};
        $("predictions").append(predictionCard(prediction, level));
      }
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
      if (navigation === state.navigation && state.mode === "uploads" && state.uploads.length) activateUpload(state.activeUpload?.upload_id || state.uploads[0].upload_id, {history: "replace"});
    });
    $("refresh-uploads").addEventListener("click", () => refreshUploads({refreshActive: true}));
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
        if (navigation === state.navigation && state.mode === "uploads" && query.get("upload")) activateUpload(query.get("upload"), {series: query.get("series"), slice: query.get("slice"), history: "none"});
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
      for (const item of index.cases) state.cases.set(item.case_id, item);
      index.patients.forEach((patient, i) => { state.patients.set(patient.patient_id, patient); state.patientNumbers.set(patient.patient_id, i + 1); });
      $("cohort-total").textContent = String(index.cases.length);
      if (state.mode === "cohort") updateCohortCount();
      renderList();
      if (navigation !== state.navigation) return;
      if (uploads) {
        await library;
        if (navigation !== state.navigation || state.mode !== "uploads") return;
        const initial = query.get("upload") || state.uploads[0]?.upload_id;
        if (initial) await activateUpload(initial, {series: query.get("series"), slice: query.get("slice"), history: "replace"});
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
        const initial = query.get("upload") || state.uploads[0]?.upload_id;
        if (initial) await activateUpload(initial, {history: "replace"});
      }
    }
  }

  start();
})();
