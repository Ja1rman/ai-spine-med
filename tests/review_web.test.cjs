"use strict";

// Run the actual browser entry point with controlled HTTP responses. These
// regressions exercise navigation under delayed/failed requests; no models or
// browser libraries are substituted into production code.
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const web = path.resolve(__dirname, "../src/ai_spine_med/review_web");
const html = fs.readFileSync(path.join(web, "index.html"), "utf8");
const source = fs.readFileSync(path.join(web, "app.js"), "utf8");
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
};
const settle = async () => { for (let i = 0; i < 8; i++) await new Promise(setImmediate); };

class Element {
  constructor(tag = "div") {
    this.tagName = tag; this.children = []; this.dataset = {}; this.attributes = {};
    this.listeners = {}; this.value = ""; this.files = []; this.checked = false;
    this.hidden = false; this.disabled = false; this._text = "";
    this.style = {setProperty() {}};
    const classes = new Set();
    this.classList = {
      add: (...names) => names.forEach((name) => classes.add(name)),
      remove: (...names) => names.forEach((name) => classes.delete(name)),
      contains: (name) => classes.has(name),
      toggle: (name, force) => {
        const add = force ?? !classes.has(name);
        if (add) classes.add(name); else classes.delete(name);
        return add;
      }
    };
  }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map((child) => child.textContent ?? child).join(""); }
  get childElementCount() { return this.children.length; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this._text = ""; this.children = children; }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  removeAttribute(name) { delete this.attributes[name]; if (name === "href") delete this.href; }
  getContext() {
    this.canvasCalls ||= [];
    return Object.fromEntries(["clearRect", "beginPath", "moveTo", "lineTo", "closePath", "stroke", "setLineDash", "fillText"].map((name) => [name, (...args) => this.canvasCalls.push([name, ...args])]));
  }
  addEventListener(name, callback) { (this.listeners[name] ||= []).push(callback); }
  querySelectorAll(selector) { return selector === "button" ? this.children.filter((child) => child.tagName === "button") : []; }
  async fire(type) { await Promise.all((this.listeners[type] || []).map((fn) => fn({target: this, preventDefault() {}}))); }
}

function collection(count = 2) {
  return {
    patients: [{patient_id: "patient-one"}, {patient_id: "patient-two"}],
    model: {title: "Fixed sagittal ensemble"},
    cases: Array.from({length: count}, (_, i) => ({
      case_id: `case-${i}`, review_order: i + 1, level: "L4-L5", label: 0,
      patient_id: i % 2 ? "patient-two" : "patient-one",
      predictions: {sagittal: {status: "ok", prediction: 0, probabilities: [0.8, 0.1, 0.1]}}
    }))
  };
}

const patient = (id) => ({patient_id: id, series: [{series_key: "sag", series_description: "Sagittal T2", zip_url: "originals/series.zip", slices: [{slice_key: "s0", image_url: "images/s0.jpg", dicom_url: "dicom/s0.dcm", rows: 100, columns: 100, instance_number: 1}]}]});
const library = {uploads: [], limits: {max_upload_bytes: 268435456}, model: {configured: true}};

test("failed upload reason is visible in both library and detail without implying a grade", async () => {
  const record = {upload_id: "upload-axial", created_at_utc: "2026-10-26T20:07:00+00:00", status: "failed", error_code: "unsupported_plane_axial", targets: [], message: "В архиве аксиальные снимки. Для анализа нужна полная сагиттальная T2-серия поясницы в ZIP."};
  const app = harness({"data/index.json": collection(), "api/uploads": {...library, uploads: [record]}, "api/uploads/upload-axial": record}, "https://review.example/r/private-token/?view=uploads&upload=upload-axial");
  try {
    await settle();
    assert.match(app.elements.get("case-list").textContent, /аксиальные снимки/);
    assert.match(app.elements.get("upload-status-message").textContent, /сагиттальная T2-серия/);
    assert.equal(app.elements.get("viewer-card").hidden, true);
    assert.match(app.elements.get("predictions").textContent, /L4-L5Не оценено/);
    assert.match(app.elements.get("predictions").textContent, /L5-S1Не оценено/);
    assert.doesNotMatch(app.elements.get("predictions").textContent, /[0-9]%/);
    app.elements.get("search").value = "аксиальные";
    await app.elements.get("search").fire("input");
    assert.equal(app.elements.get("case-list").children.length, 1);
  } finally { app.close(); }
});

function harness(routes, href = "https://review.example/r/private-token/") {
  const elements = new Map([...html.matchAll(/<([a-z][\w-]*)\b[^>]*\bid="([^"]+)"[^>]*>/gi)].map((match) => {
    const item = new Element(match[1]); item.hidden = /\bhidden\b/.test(match[0]);
    return [match[2], item];
  }));
  for (const match of html.matchAll(/<button[^>]*data-filter="([^"]+)"[^>]*>/g)) {
    const button = new Element("button"); button.dataset.filter = match[1];
    elements.get("filters").append(button);
  }
  const timers = new Set();
  const requests = [];
  const window = {location: {href}, addEventListener() {}, history: {pushState(_state, _title, url) { window.location.href = String(url); }, replaceState(_state, _title, url) { window.location.href = String(url); }}};
  const context = {
    URL, AbortController, console, window,
    document: {getElementById: (id) => elements.get(id), createElement: (tag) => new Element(tag), addEventListener() {}},
    setTimeout: (fn, ms) => { const timer = setTimeout(fn, ms); timer.unref(); timers.add(timer); return timer; },
    clearTimeout: (timer) => { clearTimeout(timer); timers.delete(timer); },
    fetch: async (url, options = {}) => {
      const resource = new URL(url).pathname.replace(/^\/r\/private-token\//, "");
      requests.push({resource, ...options});
      if (!(resource in routes)) throw new Error(`Unexpected resource ${resource}`);
      const value = await (typeof routes[resource] === "function" ? routes[resource](options) : routes[resource]);
      return {ok: true, json: async () => value};
    }
  };
  vm.runInNewContext(source, context, {filename: "app.js"});
  return {elements, window, requests, close: () => timers.forEach(clearTimeout)};
}

test("delayed full cohort cannot reopen itself after switching to uploads", async () => {
  const delayed = deferred();
  const app = harness({"data/index.json": delayed.promise, "api/uploads": library});
  try {
    await app.elements.get("tab-uploads").fire("click");
    delayed.resolve(collection(594));
    await settle();
    assert.equal(app.elements.get("tab-uploads").attributes["aria-pressed"], "true");
    assert.equal(app.elements.get("upload-panel").hidden, false);
    assert.equal(app.elements.get("case-content").hidden, true);
    assert.match(app.elements.get("collection-count").textContent, /^0 загрузок/);
    await app.elements.get("tab-cohort").fire("click");
    assert.equal(app.elements.get("case-list").children.length, 594);
  } finally { app.close(); }
});

test("switching patients immediately invalidates old viewer controls and URL", async () => {
  const nextPatient = deferred();
  const app = harness({"data/index.json": collection(), "api/uploads": library, "data/patients/patient-one.json": patient("patient-one"), "data/patients/patient-two.json": nextPatient.promise});
  try {
    await settle();
    assert.equal(app.elements.get("series-picker").disabled, false);
    const selection = app.elements.get("case-list").children.find((child) => child.dataset.case === "case-1");
    // Do not await the delayed fetch; inspect the immediate state after click.
    const navigation = selection.fire("click");
    await settle();
    assert.equal(app.elements.get("series-picker").disabled, true);
    assert.equal(app.elements.get("download-dicom").href, undefined);
    assert.equal(new URL(app.window.location.href).searchParams.get("case"), "case-1");
    assert.equal(app.elements.get("image-message").hidden, false);
    nextPatient.resolve(patient("patient-two"));
    await navigation;
    assert.equal(app.elements.get("series-picker").disabled, false);
  } finally { app.close(); }
});

test("malformed model grades remain unassessed and cannot count as correct", async () => {
  const data = collection();
  data.cases[0].predictions.sagittal = {status: "ok", prediction: 0, probabilities: [1.2, -0.2, 0]};
  const app = harness({"data/index.json": data, "api/uploads": library, "data/patients/patient-one.json": patient("patient-one")});
  try {
    await settle();
    assert.match(app.elements.get("case-badges").textContent, /неоцен[ёе]нный уровень/i);
    assert.doesNotMatch(app.elements.get("predictions").textContent, /120|Совпало с RSNA/);
    const correctFilter = app.elements.get("filters").children.find((child) => child.dataset.filter === "correct");
    await correctFilter.fire("click");
    assert.equal(app.elements.get("case-list").children.length, 1);
    assert.equal(app.elements.get("case-list").children[0].dataset.case, "case-1");
  } finally { app.close(); }
});

function roiCase() {
  const data = collection(1), scan = patient("patient-one");
  scan.series[0].slices = Array.from({length: 9}, (_, i) => ({slice_key: `s${i + 1}`, image_url: `images/s${i + 1}.jpg`, dicom_url: `dicom/s${i + 1}.dcm`, instance_number: i + 1}));
  data.cases[0].model_input = {series_key: "sag", layers: Array.from({length: 9}, (_, i) => ({roi_layer: i + 1, slice_key: `s${i + 1}`, padding: false}))};
  data.cases[0].view_links = [{url: "?case=case-0&series=sag&slice=s1", label: "ROI 1/9"}];
  return {data, scan};
}

test("default MRI view selects the central model input layer instead of the first shortcut", async () => {
  const {data, scan} = roiCase();
  const app = harness({"data/index.json": data, "api/uploads": library, "data/patients/patient-one.json": scan});
  try {
    await settle();
    assert.equal(new URL(app.window.location.href).searchParams.get("slice"), "s5");
    assert.equal(app.elements.get("slice-caption").textContent, "Срез 5 / 9");
  } finally { app.close(); }
});

test("padded central model layer falls back to the nearest real layer", async () => {
  const {data, scan} = roiCase();
  data.cases[0].model_input.layers[4] = {roi_layer: 5, slice_key: null, padding: true};
  const app = harness({"data/index.json": data, "api/uploads": library, "data/patients/patient-one.json": scan});
  try {
    await settle();
    assert.equal(new URL(app.window.location.href).searchParams.get("slice"), "s4");
    assert.equal(app.elements.get("slice-caption").textContent, "Срез 4 / 9");
  } finally { app.close(); }
});

test("an explicit source slice URL overrides the central model input default", async () => {
  const {data, scan} = roiCase();
  const app = harness({"data/index.json": data, "api/uploads": library, "data/patients/patient-one.json": scan}, "https://review.example/r/private-token/?case=case-0&series=sag&slice=s8");
  try {
    await settle();
    assert.equal(new URL(app.window.location.href).searchParams.get("slice"), "s8");
    assert.equal(app.elements.get("slice-caption").textContent, "Срез 8 / 9");
  } finally { app.close(); }
});

function pairedCollection() {
  const data = collection(4);
  data.cases[0].label = 1;
  data.cases[0].predictions.sagittal = {status: "ok", prediction: 2, probabilities: [0.05, 0.15, 0.8]};
  data.cases[1].label = 0;
  data.cases[2].level = "L5-S1";
  data.cases[2].label = 0;
  data.cases[3].level = "L5-S1";
  data.cases[3].label = 1;
  data.cases[3].predictions.sagittal = {status: "ok", prediction: 1, probabilities: [0.1, 0.8, 0.1]};
  return data;
}

function descendant(element, className) {
  if (String(element.className || "").split(" ").includes(className)) return element;
  for (const child of element.children || []) {
    const match = descendant(child, className);
    if (match) return match;
  }
  return null;
}

function levelCards(app) {
  return new Map(app.elements.get("predictions").children.map((card) => [descendant(card, "prediction-model").textContent, card]));
}

test("RSNA shows each patient's two independent predictions and their own references", async () => {
  const app = harness({"data/index.json": pairedCollection(), "api/uploads": library, "data/patients/patient-one.json": patient("patient-one")});
  try {
    await settle();
    const cards = levelCards(app);
    assert.deepEqual([...cards.keys()], ["L4-L5", "L5-S1"]);
    assert.equal(descendant(cards.get("L4-L5"), "prediction-grade").textContent, "Тяжёлый стеноз");
    assert.match(cards.get("L4-L5").textContent, /RSNA.*Умеренный стеноз/);
    assert.match(cards.get("L4-L5").textContent, /Не совпало с RSNA/);
    assert.equal(descendant(cards.get("L5-S1"), "prediction-grade").textContent, "Норма / лёгкая степень");
    assert.match(cards.get("L5-S1").textContent, /RSNA.*Норма \/ лёгкая степень/);
    assert.match(cards.get("L5-S1").textContent, /Совпало с RSNA/);
  } finally { app.close(); }
});

test("cohort summary uses both independent patient-level model estimates and identifies RSNA", async () => {
  const app = harness({"data/index.json": pairedCollection(), "api/uploads": library, "data/patients/patient-one.json": patient("patient-one")});
  try {
    await settle();
    assert.equal(app.elements.get("patient-summary").hidden, false);
    assert.match(app.elements.get("patient-summary-text").textContent, /исследования RSNA/);
    const findings = app.elements.get("patient-summary-findings").children;
    assert.equal(findings.length, 2);
    assert.match(findings[0].textContent, /L4-L5.*тяжёлый стеноз/);
    assert.match(findings[1].textContent, /L5-S1.*норма \/ лёгкая степень/);
    assert.doesNotMatch(app.elements.get("patient-summary-findings").textContent, /морфолог|маск|протрузи|экструзи/i);
    assert.equal(app.elements.get("analysis-modules").hidden, true);
  } finally { app.close(); }
});

test("changing patient replaces both level predictions without taking adjacent patient's data", async () => {
  const app = harness({"data/index.json": pairedCollection(), "api/uploads": library, "data/patients/patient-one.json": patient("patient-one"), "data/patients/patient-two.json": patient("patient-two")});
  try {
    await settle();
    await app.elements.get("case-list").children.find((child) => child.dataset.case === "case-1").fire("click");
    const cards = levelCards(app);
    assert.equal(descendant(cards.get("L4-L5"), "prediction-grade").textContent, "Норма / лёгкая степень");
    assert.equal(descendant(cards.get("L5-S1"), "prediction-grade").textContent, "Умеренный стеноз");
    assert.doesNotMatch(app.elements.get("predictions").textContent, /Тяжёлый стеноз/);
  } finally { app.close(); }
});

test("a failed second level remains unassessed while the first retains its own prediction", async () => {
  const data = pairedCollection();
  data.cases[2].predictions.sagittal = {status: "not_assessed", prediction: null, probabilities: null};
  data.cases[2].processing_reason = "missing_vertebra:S1";
  const app = harness({"data/index.json": data, "api/uploads": library, "data/patients/patient-one.json": patient("patient-one")});
  try {
    await settle();
    const cards = levelCards(app);
    assert.equal(descendant(cards.get("L4-L5"), "prediction-grade").textContent, "Тяжёлый стеноз");
    assert.equal(descendant(cards.get("L5-S1"), "prediction-grade").textContent, "Не оценено");
    assert.match(cards.get("L5-S1").textContent, /S1/);
    assert.doesNotMatch(cards.get("L5-S1").textContent, /Совпало с RSNA/);
  } finally { app.close(); }
});

test("missing or ambiguous level records cannot silently become a normal second prediction", async () => {
  for (const ambiguous of [false, true]) {
    const data = pairedCollection();
    if (ambiguous) data.cases.push({...data.cases[2], case_id: "duplicate-level"});
    else data.cases = data.cases.filter((item) => item.case_id !== "case-2");
    const app = harness({"data/index.json": data, "api/uploads": library, "data/patients/patient-one.json": patient("patient-one")});
    try {
      await settle();
      assert.equal(descendant(levelCards(app).get("L5-S1"), "prediction-grade").textContent, "Не оценено");
    } finally { app.close(); }
  }
});

test("changing the displayed slice preserves both study-level conclusions", async () => {
  const data = pairedCollection(), scan = patient("patient-one");
  scan.series[0].slices.push({...scan.series[0].slices[0], slice_key: "s1", image_url: "images/s1.jpg", dicom_url: "dicom/s1.dcm", instance_number: 2});
  const app = harness({"data/index.json": data, "api/uploads": library, "data/patients/patient-one.json": scan});
  try {
    await settle();
    const before = app.elements.get("predictions").textContent;
    app.elements.get("slice-slider").value = "0";
    await app.elements.get("slice-slider").fire("input");
    assert.equal(app.elements.get("slice-caption").textContent, "Срез 1 / 2");
    assert.equal(app.elements.get("predictions").textContent, before);
    assert.equal(levelCards(app).size, 2);
  } finally { app.close(); }
});

test("level focus changes the correct ROI and case URL while retaining both conclusions", async () => {
  const data = pairedCollection(), scan = patient("patient-one");
  scan.series[0].slices.push({...scan.series[0].slices[0], slice_key: "s1", image_url: "images/s1.jpg", dicom_url: "dicom/s1.dcm", instance_number: 2});
  for (const [index, slice] of [[0, "s0"], [2, "s1"]]) {
    data.cases[index].model_input = {series_key: "sag", layers: [{roi_layer: 5, slice_key: slice, padding: false}]};
    data.cases[index].overview_url = `overviews/${slice}.jpg`;
  }
  const app = harness({"data/index.json": data, "api/uploads": library, "data/patients/patient-one.json": scan});
  try {
    await settle();
    const before = app.elements.get("predictions").textContent;
    await app.elements.get("level-focus").children.find((child) => child.dataset.level === "L5-S1").fire("click");
    const query = new URL(app.window.location.href).searchParams;
    assert.equal(query.get("case"), "case-2");
    assert.equal(query.get("slice"), "s1");
    assert.match(app.elements.get("overview-link").href, /overviews\/s1\.jpg$/);
    assert.equal(app.elements.get("level-focus").children.find((child) => child.dataset.level === "L5-S1").attributes["aria-pressed"], "true");
    assert.equal(app.elements.get("predictions").textContent, before);
  } finally { app.close(); }
});

test("uploaded studies keep their own two targets without inheriting RSNA reference labels", async () => {
  const uploaded = {upload_id: "upload-synthetic", status: "completed", created_at_utc: "2026-10-26T13:57:00Z", n_slices: 1};
  const detail = {...uploaded, patient: patient("uploaded-patient"), targets: [
    {level: "L4-L5", status: "ok", prediction: 1, probabilities: [0.1, 0.8, 0.1]},
    {level: "L5-S1", status: "not_assessed", prediction: null, probabilities: null, reason: "missing_vertebra:S1"}
  ]};
  const app = harness({"data/index.json": pairedCollection(), "api/uploads": {...library, uploads: [uploaded]}, "api/uploads/upload-synthetic": detail}, "https://review.example/r/private-token/?view=uploads&upload=upload-synthetic");
  try {
    await settle();
    const cards = levelCards(app);
    assert.deepEqual([...cards.keys()], ["L4-L5", "L5-S1"]);
    assert.equal(descendant(cards.get("L4-L5"), "prediction-grade").textContent, "Умеренный стеноз");
    assert.equal(descendant(cards.get("L5-S1"), "prediction-grade").textContent, "Не оценено");
    assert.doesNotMatch(app.elements.get("predictions").textContent, /RSNA/);
    assert.equal(app.elements.get("reference-card").hidden, true);
    assert.equal(app.elements.get("level-focus-panel").hidden, true);
  } finally { app.close(); }
});

function morphologyUpload() {
  const scan = patient("uploaded-patient");
  scan.series[0].slices.push({...scan.series[0].slices[0], slice_key: "s1", image_url: "images/s1.jpg", dicom_url: "dicom/s1.dcm", instance_number: 2});
  const frame = (slice, B, D, H) => ({slice_key: slice, plane: "sagittal", width: 100, height: 100, B_mm: B, D_mm: D, H_mm: H, D_over_B: D / B, uncertainty_mm: 0.75, geometry_measurable: true, geometric_candidate: "extrusion", quality_flags: ["reference_unreviewed"], mask_polygons_xy: [[[30, 35], [40, 35], [40, 45]]], reference_line_xy: [[20, 35], [50, 35]]});
  return {
    upload_id: "upload-module", created_at_utc: "2026-11-22T18:00:00Z", status: "completed", n_slices: 2,
    patient: scan,
    targets: [{level: "L4-L5", status: "ok", prediction: 1, probabilities: [0.1, 0.8, 0.1]}, {level: "L5-S1", status: "ok", prediction: 0, probabilities: [0.8, 0.1, 0.1]}],
    patient_summary: {status: "research", text: "Автоматическая сводка по исследованию.", findings: ["L4-L5 · центральный стеноз умеренной степени по модели.", "L4-L5 · кандидатная маска, геометрическая гипотеза экструзии."], unassessed: ["Морфологический тип требует врачебной проверки."]},
    modules: {disc_morphology: {module_id: "disc_morphology", status: "completed", clinical_status: "indeterminate", levels: [
      {level: "L4-L5", candidate_voxels: 30, status: "candidate_needs_review", geometric_candidate: "extrusion", clinical_status: "indeterminate", representative_slice_key: "s0", frames: [frame("s0", 5, 9, 3), frame("s1", 3, 6, 2)]},
      {level: "L5-S1", candidate_voxels: 0, status: "no_candidate_mask_not_an_absence_diagnosis", geometric_candidate: "indeterminate", clinical_status: "indeterminate", representative_slice_key: null, frames: []}
    ]}}
  };
}

function moduleHarness(record, extra = {}) {
  return harness({"data/index.json": pairedCollection(), "api/uploads": {...library, uploads: [record]}, [`api/uploads/${record.upload_id}`]: record, ...extra}, `https://review.example/r/private-token/?view=uploads&upload=${record.upload_id}`);
}

test("morphology shows current-slice dimensions and contour, while summary remains at study scope", async () => {
  const app = moduleHarness(morphologyUpload());
  try {
    await settle();
    assert.equal(app.elements.get("analysis-modules").hidden, false);
    assert.match(app.elements.get("patient-summary").hidden ? "" : app.elements.get("patient-summary-findings").textContent, /гипотеза экструзии/);
    assert.match(app.elements.get("module-content").textContent, /3,00 мм/);
    assert.match(app.elements.get("module-content").textContent, /Геометрическая гипотеза · Экструзия/);
    assert.match(app.elements.get("module-content").textContent, /Морфологический тип требует проверки/);
    const summary = app.elements.get("patient-summary-findings").textContent;
    const overlay = app.elements.get("morphology-overlay");
    assert.equal(overlay.hidden, false);
    assert.equal(overlay.width, 100);
    assert.equal(overlay.height, 100);
    assert.ok(overlay.canvasCalls.some((call) => call[0] === "moveTo" && call[1] === 30 && call[2] === 35));
    app.elements.get("slice-slider").value = "0";
    await app.elements.get("slice-slider").fire("input");
    assert.match(app.elements.get("module-content").textContent, /5,00 мм/);
    assert.match(app.elements.get("module-content").textContent, /9,00 мм/);
    assert.equal(app.elements.get("patient-summary-findings").textContent, summary);
    await app.elements.get("zoom-in").fire("click");
    assert.equal(overlay.style.transform, app.elements.get("slice-image").style.transform);
  } finally { app.close(); }
});

test("no candidate stays unassessed, and representative shortcut opens its exact source slice", async () => {
  const app = moduleHarness(morphologyUpload());
  try {
    await settle();
    const content = app.elements.get("module-content");
    await descendant(content, "morphology-actions").children[0].fire("click");
    assert.equal(app.elements.get("slice-caption").textContent, "Срез 1 / 2");
    const levels = descendant(content, "morphology-levels");
    await levels.children.find((button) => button.dataset.level === "L5-S1").fire("click");
    assert.match(content.textContent, /Наличие грыжи остаётся неоценённым/);
    assert.doesNotMatch(content.textContent, /Экструзия|0,00 мм|Норма/);
    assert.equal(descendant(content, "morphology-actions").children[0].disabled, true);
    assert.equal(app.elements.get("morphology-overlay").hidden, true);
    assert.match(app.elements.get("predictions").textContent, /Умеренный стеноз/);
  } finally { app.close(); }
});

test("module failure preserves central predictions and does not fabricate dimensions", async () => {
  const record = morphologyUpload();
  record.modules.disc_morphology = {status: "failed", message: "Модуль не завершил обработку."};
  const app = moduleHarness(record);
  try {
    await settle();
    assert.match(app.elements.get("module-content").textContent, /Не удалось оценить морфологию/);
    assert.doesNotMatch(app.elements.get("module-content").textContent, /мм|Экструзия/);
    assert.match(app.elements.get("predictions").textContent, /Умеренный стеноз/);
    assert.equal(app.elements.get("morphology-overlay").hidden, true);
  } finally { app.close(); }
});

test("navigation clears old patient's module and overlays before the next request resolves", async () => {
  const record = morphologyUpload(), next = deferred();
  const other = {upload_id: "upload-old", status: "completed"};
  const app = moduleHarness(record, {"api/uploads": {...library, uploads: [record, other]}, "api/uploads/upload-old": next.promise});
  try {
    await settle();
    assert.equal(app.elements.get("morphology-overlay").hidden, false);
    const navigation = app.elements.get("case-list").children[1].fire("click");
    await settle();
    assert.equal(app.elements.get("analysis-modules").hidden, true);
    assert.equal(app.elements.get("morphology-overlay").hidden, true);
    assert.equal(app.elements.get("patient-summary").hidden, true);
    next.resolve({...other, patient: patient("old-upload-patient"), targets: []});
    await navigation;
    assert.equal(app.elements.get("analysis-modules").hidden, true);
    assert.doesNotMatch(app.elements.get("module-content").textContent, /Экструзия|мм/);
    await app.elements.get("tab-cohort").fire("click");
    assert.equal(app.elements.get("analysis-modules").hidden, true);
    assert.equal(app.elements.get("morphology-overlay").hidden, true);
    assert.doesNotMatch(app.elements.get("patient-summary-findings").textContent, /кандидатная маска/);
  } finally { app.close(); }
});

test("unusable slice measurements keep their quality flags and unknowns", async () => {
  const record = morphologyUpload();
  Object.assign(record.modules.disc_morphology.levels[0].frames[1], {B_mm: null, D_over_B: null, geometry_measurable: false, quality_flags: ["disconnected_mask"]});
  const app = moduleHarness(record);
  try {
    await settle();
    const content = app.elements.get("module-content");
    assert.match(content.textContent, /не прошла проверку пригодности/);
    assert.match(content.textContent, /раздельные компоненты/);
    assert.doesNotMatch(content.textContent, /0,00/);
    assert.match(content.textContent, /6,00 мм/);
  } finally { app.close(); }
});

test("a completed central job with a pending module remains visibly processing", async () => {
  const record = morphologyUpload();
  record.modules.disc_morphology = {status: "processing"};
  const app = moduleHarness(record);
  try {
    await settle();
    assert.match(app.elements.get("module-content").textContent, /анализируется/);
    assert.equal(app.elements.get("upload-status").classList.contains("is-pending"), true);
    assert.match(app.elements.get("predictions").textContent, /Умеренный стеноз/);
  } finally { app.close(); }
});

test("existing uploads can queue the optional module through an intentional empty same-origin POST", async () => {
  let record = {...morphologyUpload(), modules_available: true, modules: {}};
  let requested = false;
  const app = moduleHarness(record, {
    "api/uploads": () => ({...library, uploads: [record]}),
    "api/uploads/upload-module": () => record,
    "api/uploads/upload-module/modules": (options) => { assert.equal(options.method, "POST"); requested = true; record = {...record, status: "queued", modules: {disc_morphology: {status: "queued"}}}; return record; }
  });
  try {
    await settle();
    assert.equal(app.elements.get("analysis-modules").hidden, false);
    assert.equal(app.elements.get("run-analysis-modules").hidden, false);
    assert.equal(app.elements.get("run-analysis-modules").disabled, false);
    assert.match(app.elements.get("module-content").textContent, /ещё не оценена/);
    await app.elements.get("run-analysis-modules").fire("click");
    assert.equal(requested, true);
    const request = app.requests.find((request) => request.method === "POST");
    assert.equal(request.headers["X-Upload-Intent"], "dicom-series");
    assert.equal(request.body, "");
    assert.equal(request.credentials, "same-origin");
    assert.equal(app.elements.get("run-analysis-modules").disabled, true);
    assert.match(app.elements.get("module-content").textContent, /в очереди/);
    assert.match(app.elements.get("predictions").textContent, /Умеренный стеноз/);
  } finally { app.close(); }
});

test("failed module requests surface uncertainty and unsupported old uploads have no run action", async () => {
  const record = {...morphologyUpload(), modules_available: true};
  const app = moduleHarness(record, {"api/uploads/upload-module/modules": () => { throw new Error("lost response"); }});
  try {
    await settle();
    await app.elements.get("run-analysis-modules").fire("click");
    assert.match(app.elements.get("module-feedback").textContent, /Обновите список перед повторной попыткой/);
    assert.equal(app.elements.get("run-analysis-modules").disabled, false);
  } finally { app.close(); }
  delete record.modules_available;
  const old = moduleHarness(record);
  try {
    await settle();
    assert.equal(old.elements.get("run-analysis-modules").hidden, true);
  } finally { old.close(); }
});
