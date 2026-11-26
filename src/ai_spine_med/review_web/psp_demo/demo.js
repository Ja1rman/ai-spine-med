"use strict";

const symptomLabels = {
  pain: "Боль",
  numbness: "Онемение",
  weakness: "Субъективная слабость",
};
const sides = { left: "слева", right: "справа", bilateral: "с обеих сторон", midline: "по центру", unknown: "сторона не уточнена" };
const views = {
  front: { label: "Спереди", sides: "Правая сторона · Левая сторона" },
  back: { label: "Сзади", sides: "Левая сторона · Правая сторона" },
};
const silhouette = "M134 77 L134 93 L111 99 Q96 101 91 121 L70 212 L57 273 Q55 286 65 289 Q75 291 80 277 L102 215 L112 176 L110 264 Q108 283 116 302 L110 408 L105 526 L96 558 Q93 569 108 571 L128 571 Q136 570 136 559 L141 417 L150 320 L159 417 L164 559 Q164 570 172 571 L192 571 Q207 569 204 558 L195 526 L190 408 L184 302 Q192 283 190 264 L188 176 L198 215 L220 277 Q225 291 235 289 Q245 286 243 273 L230 212 L209 121 Q204 101 189 99 L166 93 L166 77 Z";
const bodyAxis = "M150 102 L150 275 M115 271 Q150 285 185 271 M119 307 Q150 294 181 307 M116 413 L138 413 M162 413 L184 413";

function element(tag, text) {
  const node = document.createElement(tag);
  if (typeof text === "string" || typeof text === "number") node.textContent = text;
  return node;
}

function svgElement(tag, attributes) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
  return node;
}

function setText(id, value) {
  if (typeof value !== "string" || !value) return;
  const node = document.getElementById(id);
  node.textContent = value;
  node.hidden = false;
}

function appendFindings(id, values, fallback = "") {
  const list = document.getElementById(id);
  for (const value of values || []) {
    if (typeof value === "string") list.append(element("li", value));
  }
  if (!list.children.length && fallback) list.append(element("li", fallback));
  list.hidden = !list.children.length;
}

function validMark(mark) {
  return mark && Object.hasOwn(views, mark.view) && Object.hasOwn(symptomLabels, mark.type)
    && [mark.x, mark.y].every(value => Number.isFinite(value) && value >= 0 && value <= 1);
}

function markLabel(mark) {
  const parts = [symptomLabels[mark.type], views[mark.view].label.toLowerCase()];
  if (typeof mark.region === "string" && mark.region) parts.push(mark.region);
  if (Object.hasOwn(sides, mark.side)) parts.push(sides[mark.side]);
  if (Number.isFinite(mark.intensity)) parts.push(`${mark.intensity}/10`);
  return parts.join(" · ");
}

function drawView(view, marks) {
  const figure = element("figure");
  const { label, sides } = views[view];
  const canvas = svgElement("svg", { viewBox: "0 0 300 600", role: "img", "aria-label": `Карта тела. Вид ${label.toLowerCase()}` });
  // The viewer and the PSP form share this body coordinate system.
  canvas.append(
    svgElement("circle", { cx: 150, cy: 48, r: 28, class: "body-outline" }),
    svgElement("path", { d: silhouette, class: "body-outline" }),
    svgElement("path", { d: bodyAxis, class: "body-axis" }),
  );
  for (const mark of marks.filter(mark => mark.view === view)) {
    const circle = svgElement("circle", { cx: mark.x * 300, cy: mark.y * 600, r: 13, class: `mark ${mark.type}` });
    const title = svgElement("title", {});
    title.textContent = markLabel(mark);
    circle.append(title);
    canvas.append(circle);
  }
  const caption = element("figcaption", label);
  const sideLabels = element("span", sides);
  sideLabels.className = "body-sides";
  caption.append(sideLabels);
  figure.append(canvas, caption);
  return figure;
}

function render(data) {
  setText("title", data.title);
  setText("context", data.context);
  appendFindings("summary-list", data.summary, "Общая сводка недоступна.");
  setText("mri-title", data.mri?.title);
  appendFindings("mri-findings", data.mri?.findings, "Данные анализа МРТ недоступны.");
  appendFindings("mri-unassessed", data.mri?.unassessed);

  const complaints = document.getElementById("complaints-list");
  for (const item of data.complaints || []) {
    if (typeof item?.label === "string" && typeof item.value === "string") {
      complaints.append(element("dt", item.label), element("dd", item.value));
    }
  }
  const marks = (data.pain_map || []).filter(validMark);
  document.getElementById("body-views").append(drawView("front", marks), drawView("back", marks));
  appendFindings("map-list", marks.map(markLabel), "Отметок нет.");

  const rows = document.getElementById("exam-rows");
  for (const item of data.exam || []) {
    if (typeof item?.label !== "string") continue;
    const row = element("tr");
    const test = element("th", item.label);
    test.scope = "row";
    row.append(test, element("td", item.left ?? "—"), element("td", item.right ?? "—"));
    rows.append(row);
  }
}

async function load() {
  const status = document.getElementById("loading");
  try {
    const response = await fetch("./demo.json", { credentials: "omit", cache: "no-store" });
    if (!response.ok) throw new Error("Demo data unavailable");
    render(await response.json());
    status.hidden = true;
    document.getElementById("demo").hidden = false;
  } catch {
    status.textContent = "Не удалось открыть пример. Обновите страницу, чтобы повторить загрузку.";
  }
}

load();
