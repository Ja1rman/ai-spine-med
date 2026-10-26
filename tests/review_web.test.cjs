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
  const window = {location: {href}, addEventListener() {}, history: {pushState(_state, _title, url) { window.location.href = String(url); }, replaceState(_state, _title, url) { window.location.href = String(url); }}};
  const context = {
    URL, AbortController, console, window,
    document: {getElementById: (id) => elements.get(id), createElement: (tag) => new Element(tag), addEventListener() {}},
    setTimeout: (fn, ms) => { const timer = setTimeout(fn, ms); timer.unref(); timers.add(timer); return timer; },
    clearTimeout: (timer) => { clearTimeout(timer); timers.delete(timer); },
    fetch: async (url) => {
      const resource = new URL(url).pathname.replace(/^\/r\/private-token\//, "");
      if (!(resource in routes)) throw new Error(`Unexpected resource ${resource}`);
      const value = await (typeof routes[resource] === "function" ? routes[resource]() : routes[resource]);
      return {ok: true, json: async () => value};
    }
  };
  vm.runInNewContext(source, context, {filename: "app.js"});
  return {elements, window, close: () => timers.forEach(clearTimeout)};
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
