"""Private post-hoc illustrations of fixed axial models and anatomical anchors.

Every raster, source identity and local DICOM link stays below reports/runs.
This gallery does not select models, retrain, or estimate representative accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from matplotlib.patches import Rectangle

LEVELS = ("L4-L5", "L5-S1")
GRADES = ("норма/лёгкий", "умеренный", "тяжёлый")


def _load(path):
    return json.loads(Path(path).read_text())


def _key(row):
    return row["dataset"], row["study_id"]


def _format_probability(probability):
    return " / ".join(f"{x:.3f}" for x in probability)


def _safe(root, relative):
    root = Path(root).resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise ValueError("Source path is missing or escapes its documented source root")
    return candidate


def _anchor_position(anchor, slices):
    """Place an annotation on bag-index axis within its own frame and series only."""
    candidates = [
        x
        for x in slices
        if x["series_id"] == anchor["series_id"] and x["frame_sha256"] == anchor["frame_sha256"]
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda x: x["raw_index"])
    raw = np.asarray([x["raw_index"] for x in candidates])
    if not raw[0] <= anchor["raw_index"] <= raw[-1]:
        return None
    return float(np.interp(anchor["raw_index"], raw, [x["selected_index"] for x in candidates]))


def _ensemble(arrays, entries, phase, name):
    selected = [i for i, entry in enumerate(entries) if entry.startswith(phase + "_seed_")]
    if len(selected) != 3:
        raise ValueError("Require all three predetermined seeds per phase")
    return arrays[name][selected].astype(np.float64).mean(axis=0)


def select_examples(anchors, arrays, entries, ap_zero):
    """Deterministic illustrative selection; never a representative case sample."""
    selected, reasons = [], {}

    def add(row, reason):
        key = _key(row)
        if key not in reasons:
            selected.append(key)
            reasons[key] = []
        reasons[key].append(reason)

    candidates = []
    for row in anchors:
        if row["dataset"] != "aisslab" or row["labels"][1] != 2:
            continue
        positions = [
            _anchor_position(a, row["selected_slices"])
            for a in row["anchors"]
            if a["level"] == "L5-S1"
        ]
        positions = [p for p in positions if p is not None]
        if not positions:
            continue
        weights = _ensemble(arrays[_key(row)], entries, "after", "attention")[1]
        distances = np.min(abs(np.arange(len(weights))[:, None] - positions), axis=1)
        expected_distance = float(weights @ distances / max(1, len(weights) - 1))
        candidates.append((expected_distance, _key(row), row))
    candidates.sort(key=lambda x: (x[0], x[1]))
    for _, _, row in candidates[:2]:
        add(row, "AISSLab severe L5–S1: smallest mean attention distance to XML anchor")
    for _, _, row in candidates[-2:]:
        add(row, "AISSLab severe L5–S1: largest mean attention distance to XML anchor")
    by_key = {_key(row): row for row in anchors}
    for row in sorted(anchors, key=_key):
        if row["dataset"] == "rsna" and row["labels"][1] == 2:
            add(row, "All RSNA TEST severe L5–S1 cases")
    for item in sorted(ap_zero, key=lambda x: x["study_id"]):
        key = "aisslab", item["study_id"]
        if key in by_key:
            add(by_key[key], "AISSLab target annotation has AP diameter 0 mm; inspect source")
    failures = []
    for row in anchors:
        if row["dataset"] != "rsna" or row["labels"][0] != 2:
            continue
        before = _ensemble(arrays[_key(row)], entries, "before", "probabilities")[0]
        after = _ensemble(arrays[_key(row)], entries, "after", "probabilities")[0]
        if before.argmax() == 2 and after.argmax() != 2:
            failures.append((float(after[2] - before[2]), _key(row), row))
    for _, _, row in sorted(failures, key=lambda x: (x[0], x[1]))[:2]:
        add(row, "RSNA severe L4–L5: before correct, after wrong; largest severe-probability drop")
    return selected, reasons


def _pixels(path):
    ds = pydicom.dcmread(path)
    if int(getattr(ds, "NumberOfFrames", 1)) != 1:
        raise ValueError("The gallery expects one frame per DICOM")
    pixels = np.asarray(ds.pixel_array, dtype=np.float32)
    if pixels.ndim != 2 or not np.isfinite(pixels).all():
        raise ValueError("Invalid raw source pixels")
    pixels = pixels * float(getattr(ds, "RescaleSlope", 1)) + float(
        getattr(ds, "RescaleIntercept", 0)
    )
    return pixels, str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2"))


def _reference_boxes(source_root, record):
    result = {}
    if record is None:
        return result
    for reference in record["label_references"]:
        tree = ET.parse(_safe(source_root, reference["xml_path"]))
        obj = tree.findall("object")[reference["object_index"]]
        box = obj.find("bndbox")
        if box is None or not obj.findtext("name", "").startswith("CCS_"):
            raise ValueError("The XML reference is not an explicit CCS box")
        coords = [float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
        result[(reference["dicom_path"], reference["level"])] = {
            "bbox_xyxy": coords,
            "xml_path": str(_safe(source_root, reference["xml_path"])),
            "source_class": obj.findtext("name"),
            "ap_diameter": obj.findtext("ap_diameter"),
        }
    return result


def _reference_tile(row, level):
    anchors = sorted(
        [a for a in row["anchors"] if a["level"] == level],
        key=lambda a: (a["series_id"], a["instance"], a["path"]),
    )
    if not anchors:
        return None
    return anchors[len(anchors) // 2]


def _render_case(case_code, row, arrays, entries, source, new_record, output):
    slices = row["selected_slices"]
    before_a = _ensemble(arrays, entries, "before", "attention")
    after_a = _ensemble(arrays, entries, "after", "attention")
    before_p = _ensemble(arrays, entries, "before", "probabilities")
    after_p = _ensemble(arrays, entries, "after", "probabilities")
    source_root = Path(source["source_root"])
    boxes = _reference_boxes(source_root, new_record)
    pixels, series_windows = {}, {}
    # A single fixed 1st/99th-percentile display window per series, never per tile.
    for series in source["source"]["series"]:
        series_id = series["series_id"]
        series_images = []
        for item in series["files"]:
            relative = str(Path(series["series_path"]) / item["name"])
            raw_path = _safe(source_root, relative)
            if hashlib.sha256(raw_path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("Raw source DICOM checksum changed")
            image, photo = _pixels(raw_path)
            pixels[relative] = image, photo
            series_images.append(image.ravel())
        lo, hi = np.percentile(np.concatenate(series_images), [1, 99])
        series_windows[series_id] = float(lo), float(max(hi, lo + 1e-6))
    fig = plt.figure(figsize=(15, 13), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=[0.85, 1, 1])
    profile = fig.add_subplot(grid[0, :])
    colors = ("#2563eb", "#c2410c")
    x = np.arange(len(slices))
    for li, (level, color) in enumerate(zip(LEVELS, colors)):
        profile.plot(x, before_a[li], color=color, ls="--", lw=1.7, label=f"{level}: до")
        profile.plot(x, after_a[li], color=color, lw=2.0, label=f"{level}: после")
        marks = [_anchor_position(a, slices) for a in row["anchors"] if a["level"] == level]
        marks = [p for p in marks if p is not None]
        for mark in marks:
            profile.axvline(mark, color=color, alpha=0.45, lw=1)
        if marks:
            profile.axvspan(min(marks) - 0.25, max(marks) + 0.25, color=color, alpha=0.10)
    for i in range(1, len(slices)):
        if slices[i]["frame_group"] != slices[i - 1]["frame_group"]:
            profile.axvline(i - 0.5, color="#111827", ls=":", lw=1.4)
            profile.text(
                i - 0.5,
                0.99,
                " новый frame",
                transform=profile.get_xaxis_transform(),
                va="top",
                fontsize=8,
            )
    profile.set_title(f"{case_code} · {row['dataset'].upper()} · внимание к срезам", loc="left")
    profile.set_xlabel("Индекс среза в сумке модели; внутри frame: сверху → вниз")
    profile.set_ylabel("Вес attention")
    profile.set_xlim(-0.5, len(slices) - 0.5)
    profile.set_ylim(bottom=0)
    profile.grid(alpha=0.18)
    profile.legend(ncol=4, fontsize=9, loc="upper right")
    tile_manifest = []
    for li, level in enumerate(LEVELS):
        reference = _reference_tile(row, level)
        selections = [
            ("reference", reference),
            ("before_top", slices[int(before_a[li].argmax())]),
            ("after_top", slices[int(after_a[li].argmax())]),
        ]
        for column, (kind, image_meta) in enumerate(selections):
            ax = fig.add_subplot(grid[li + 1, column])
            ax.axis("off")
            if image_meta is None:
                ax.text(0.5, 0.5, "Нет координат уровня", ha="center")
                continue
            relative = image_meta["path"]
            image, photo = pixels[relative]
            lo, hi = series_windows[image_meta["series_id"]]
            display = np.clip((image - lo) / (hi - lo), 0, 1)
            if photo == "MONOCHROME1":
                display = 1 - display
            ax.imshow(display, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            reference_name = (
                "reference XML CCS" if row["dataset"] == "aisslab" else ("RSNA subarticular approx")
            )
            if kind == "reference":
                title = f"{level} · {reference_name}"
            else:
                phase, probability, attention = (
                    ("до", before_p, before_a)
                    if kind == "before_top"
                    else ("после", after_p, after_a)
                )
                title = f"{level} · top attention {phase}"
                title += f"\nОтвет: {GRADES[int(probability[li].argmax())]}"
            box = boxes.get((relative, level))
            if box is not None:
                xmin, ymin, xmax, ymax = box["bbox_xyxy"]
                ax.add_patch(
                    Rectangle(
                        (xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color="#22d3ee", lw=1.4
                    )
                )
                title += f"\n{box['source_class']} · AP {box['ap_diameter']}"
            bag_index = next((s["selected_index"] for s in slices if s["path"] == relative), None)
            title += f"\nСрез {image_meta['instance']} · индекс {bag_index if bag_index is not None else 'вне bag'}"
            ax.set_title(title, fontsize=10)
            dicom_path = _safe(source_root, relative)
            tile_manifest.append(
                {
                    "kind": kind,
                    "level": level,
                    "path": str(dicom_path),
                    "bag_index": bag_index,
                    "instance": image_meta["instance"],
                    "series_id": image_meta["series_id"],
                    "display_window": [lo, hi],
                    "reference_box": box,
                }
            )
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return {
        "labels": row["labels"],
        "before_probabilities": before_p.tolist(),
        "after_probabilities": after_p.tolist(),
        "tiles": tile_manifest,
        "selected_source_dicom_paths": [str(_safe(source_root, x["path"])) for x in slices],
        "all_source_dicom_paths": [
            str(_safe(source_root, str(Path(series["series_path"]) / item["name"])))
            for series in source["source"]["series"]
            for item in series["files"]
        ],
    }


def build_gallery(root=".", diagnosis="reports/runs/axial-diagnosis-v5"):
    root = Path(root).resolve()
    diagnosis = _safe(root, str(Path(diagnosis) / "attention/index.json")).parent.parent
    if not diagnosis.is_relative_to(root / "reports/runs"):
        raise ValueError("Gallery contains source pixels and must stay private below reports/runs")
    output = diagnosis / "gallery"
    output.mkdir(parents=True, exist_ok=True)
    attention = _load(diagnosis / "attention/index.json")
    anchors_doc = _load(diagnosis / "localization/anchors.json")
    anchors = anchors_doc["records"]
    sources = _load(root / "artifacts/axial-transfer-sources-v4/evaluation-sources.json")
    sources = {_key(x): x for x in sources["records"]}
    new_records = {_key(x): x for x in _load(root / "artifacts/aisslab-axial-v4/records.json")}
    arrays = {}
    for item in attention["records"]:
        path = _safe(diagnosis / "attention", item["arrays_path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["arrays_sha256"]:
            raise ValueError("Attention archive checksum changed")
        with np.load(path, allow_pickle=False) as archive:
            arrays[_key(item)] = {k: archive[k] for k in ("attention", "probabilities")}
    selected, reasons = select_examples(
        anchors,
        arrays,
        attention["entries"],
        _load(diagnosis / "label-audit.json")["ap_zero_cases_private"],
    )
    by_key = {_key(x): x for x in anchors}
    manifest = {
        "scope": "PRIVATE post-hoc illustrative cases; not random and not an accuracy estimate",
        "selection_rationale": "Four AISSLab severe L5–S1 extremes in mean attention distance, all three RSNA severe L5–S1, two AISSLab zero-AP targets, two RSNA severe L4–L5 regressions; deduplicated.",
        "raw_pixels": True,
        "training_performed": False,
        "axis": "Selected bag index; cranial-to-caudal within frame groups. No physical comparison across frames.",
        "display": "Original DICOM pixel orientation, rescale slope/intercept, fixed 1st/99th percentile window per source series; no header metadata burned into images.",
        "reference_limit": "AISSLab exact XML CCS source slice. RSNA axial subarticular-coordinate level proxy; not a central-canal or Schizas reference.",
        "cases": [],
    }
    sections, nav = [], []
    for number, key in enumerate(selected, 1):
        code = f"case-{number:02d}"
        image_path = output / f"{code}.png"
        details = _render_case(
            code,
            by_key[key],
            arrays[key],
            attention["entries"],
            sources[key],
            new_records.get(key),
            image_path,
        )
        manifest["cases"].append(
            {
                "case_code": code,
                "dataset": key[0],
                "study_id": key[1],
                "selection_reasons": reasons[key],
                "image": image_path.name,
                **details,
            }
        )
        nav.append(f'<a href="#{code}">{code}</a>')
        rows = []
        for li, level in enumerate(LEVELS):
            label = details["labels"][li]
            truth = "нет метки" if label is None else GRADES[label]
            rows.append(
                f"<tr><td>{level}</td><td>{truth}</td><td>{_format_probability(details['before_probabilities'][li])}</td><td>{_format_probability(details['after_probabilities'][li])}</td></tr>"
            )
        links = []
        for tile in details["tiles"]:
            relative = os.path.relpath(tile["path"], output)
            links.append(
                f'<a href="{quote(relative)}">{html.escape(tile["level"] + " · " + tile["kind"])} DICOM</a>'
            )
        sections.append(
            f'<section id="{code}"><h2>{code} · {key[0].upper()}</h2>'
            f"<p>{html.escape('; '.join(reasons[key]))}</p>"
            f'<img src="{image_path.name}" alt="{code}: attention и исходные срезы">'
            "<table><thead><tr><th>Уровень</th><th>Метка источника</th><th>До: норма / умеренный / тяжёлый</th><th>После</th></tr></thead>"
            f'<tbody>{"".join(rows)}</tbody></table><div class="links">{"".join(links)}</div></section>'
        )
    document = (
        '<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Аксиальная модель — разбор срезов</title><style>body{font:16px/1.5 system-ui;margin:0;background:#f1f4f7;color:#172536}main{max-width:1400px;margin:auto;padding:28px}nav{display:flex;flex-wrap:wrap;gap:12px;position:sticky;top:0;background:#f1f4f7ee;padding:12px;backdrop-filter:blur(8px)}section{background:white;padding:22px;margin:24px 0;border-radius:12px;scroll-margin-top:70px}img{width:100%;height:auto}h1{font-size:27px}h2{font-size:21px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;text-align:left;border-bottom:1px solid #dae2e8}.links{display:flex;gap:15px;flex-wrap:wrap;margin-top:16px;font-size:13px}a{color:#245b99}p{max-width:1100px}</style><main><h1>Аксиальная модель: куда направлено внимание</h1><p>Подобрал показательные случаи после завершения эксперимента. Это иллюстрации возможных проблем, а не случайная выборка для оценки точности. Сплошные линии — после дообучения, пунктир — до; цветные вертикальные отметки — координаты уровней. RSNA даёт здесь только приближение по subarticular-разметке. Рамка на AISSLab — исходная XML-разметка CCS. Изображения в исходной ориентации DICOM, с одинаковым фиксированным окном для каждой серии.</p><nav>'
        + "".join(nav)
        + "</nav>"
        + "".join(sections)
        + "</main></html>"
    )
    (output / "index.html").write_text(document)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return {
        "cases": len(selected),
        "index": str(output / "index.html"),
        "manifest": str(output / "manifest.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--diagnosis", default="reports/runs/axial-diagnosis-v5")
    args = parser.parse_args()
    print(json.dumps(build_gallery(args.root, args.diagnosis), ensure_ascii=False))


if __name__ == "__main__":
    main()
