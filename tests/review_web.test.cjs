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
    assert.match(app.elements.get("case-badges").textContent, /Не оценено/);
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
