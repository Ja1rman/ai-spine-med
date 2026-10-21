"""Private review images from already-produced predicted physical ROIs.

GT selects review cases and supplies a separate reference overlay only. Neither
GT nor grades are passed to the crop reconstruction path. All images remain in
ignored reports/runs directories and are unsuitable for public Git artifacts.
"""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .assets import sha256_file, write_json
from .axial_features import IMAGENET_MEAN, IMAGENET_STD
from .axial_localization_audit import LEVELS
from .axial_localizer_crops import (
    _lps_to_source_xy,
    crop_level_inputs,
    prepare_crop_source,
    source_to_canonical_xy,
)
from .axial_localizer_scoring import score_case


def _reference_display(prepared, anchors, level, predicted_key):
    images = {(i.series_id, i.instance): i for i in prepared["images"]}
    candidates = [
        a
        for a in anchors
        if a["level"] == level and (str(a["series_id"]), int(a["instance"])) in images
    ]
    if not candidates:
        return None
    predicted = images[predicted_key]
    candidates.sort(
        key=lambda a: (
            a["series_id"] != predicted_key[0],
            abs(float((np.asarray(a["point_lps"]) - predicted.center_lps) @ predicted.normal)),
            int(a["instance"]),
        )
    )
    reference = candidates[0]
    key = str(reference["series_id"]), int(reference["instance"])
    image, header = images[key], prepared["headers"][key]
    original_xy = _lps_to_source_xy(reference["point_lps"], header)
    canonical = source_to_canonical_xy(original_xy, header["shape"], image.orientation_action)
    return image.image, canonical, reference


def render_case(path, prepared, prediction, crops, anchors, level, score, title):
    """Render pixels from the actual image-only crop path, references separately."""
    landmark = prediction["levels"][level]
    key = str(landmark["series_id"]), int(landmark["instance"])
    images = {(i.series_id, i.instance): i for i in prepared["images"]}
    full = images[key].image
    header = prepared["headers"][key]
    center = np.asarray(landmark["center_xy_norm"]) * (np.asarray(full.shape[::-1]) - 1)
    crop = crops["levels"][level]
    if crop["status"] != "ok":
        raise ValueError("Review image needs an actual successful predicted ROI")
    tensors = crop["images"].cpu().numpy()
    actual_channels = (
        tensors[0] * np.asarray(IMAGENET_STD)[:, None, None]
        + np.asarray(IMAGENET_MEAN)[:, None, None]
    )
    reference = _reference_display(prepared, anchors, level, key)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=130)
    fig.set_facecolor("#111821")
    for ax in axes.flat:
        ax.set_facecolor("#111821")
        ax.tick_params(colors="#b5c0d0", labelsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0, 0].imshow(full, cmap="gray", vmin=0, vmax=1)
    spacing = np.asarray(header["canonical_spacing"])
    width, height = crops["parameters"]["roi_mm"] / spacing[::-1]
    axes[0, 0].add_patch(
        Rectangle(
            center - [width / 2, height / 2],
            width,
            height,
            fill=False,
            edgecolor="#ffd166",
            linewidth=1.4,
        )
    )
    axes[0, 0].plot(*center, "+", color="#ffd166", markersize=12, markeredgewidth=1.5)
    axes[0, 0].set_title(
        f"Predicted source slice · instance {key[1]}\nYellow: predicted canal center + {crops['parameters']['roi_mm']:g} mm ROI",
        color="white",
        fontsize=10,
    )
    if reference is not None:
        image, xy, ref = reference
        axes[0, 1].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[0, 1].plot(*xy, "+", color="#b39dff", markersize=12, markeredgewidth=1.5)
        kind = (
            "CCS box center"
            if ref["kind"] == "aisslab_xml_ccs_exact_slice"
            else "approximate level point, NOT canal center"
        )
        axes[0, 1].set_title(
            f"Reference slice · instance {ref['instance']}\nPurple: {kind}",
            color="white",
            fontsize=10,
        )
    else:
        axes[0, 1].text(
            0.5,
            0.5,
            "No displayable axial reference",
            color="white",
            ha="center",
            transform=axes[0, 1].transAxes,
        )
    axes[0, 2].axis("off")
    distance = score["top1_plane_distance_mm"]
    metrics = [
        title,
        "",
        f"Plane status: {score['plane_status']}",
        f"Plane distance: {distance:.2f} mm"
        if distance is not None
        else "Plane distance: unavailable",
        f"Hit within 5/10 mm: {score['hit5']} / {score['hit10']}",
        f"Center reference: {score['center']['status']}",
    ]
    if score["center"]["status"] == "verifiable_exact_ccs":
        metrics += [
            f"Center in-plane: {score['center']['inplane_error_mm']:.2f} mm",
            f"Center plane: {score['center']['plane_error_mm']:.2f} mm",
            f"Center 3D: {score['center']['distance3d_mm']:.2f} mm",
        ]
    metrics += [
        "",
        "Actual 2.5D ROI input below:",
        "previous / current / next",
        "GT is used for the separate",
        "reference overlay only.",
    ]
    axes[0, 2].text(
        0,
        1,
        "\n".join(metrics),
        va="top",
        color="#dbe4f1",
        fontsize=10,
        family="monospace",
        linespacing=1.5,
        transform=axes[0, 2].transAxes,
    )
    for column, name in enumerate(("Previous context", "Current anchor", "Next context")):
        axes[1, column].imshow(np.clip(actual_channels[column], 0, 1), cmap="gray", vmin=0, vmax=1)
        axes[1, column].set_title(
            name + " · actual 224×224 model input", color="white", fontsize=10
        )
    fig.tight_layout(pad=2)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return {
        "actual_crop_channels": list(actual_channels.shape),
        "reference_displayed": reference is not None,
    }


def create_localization_gallery(
    input_index, features_dir, out, *, cases=None, references=None, maximum=12
):
    """Render explicit cases or the first available validation records.

    cases is a list of (dataset, study_id, level). TEST galleries require an
    input index whose preparation already verified both frozen model selections.
    """
    input_index, features_dir, out = map(Path, (input_index, features_dir, out))
    index = json.loads(input_index.read_text())
    source_rows = {(r["dataset"], r["study_id"]): r for r in index["records"]}
    phase = index.get("configuration", {}).get("phase")
    if phase == "evaluation_image_inputs" and (
        references is None
        or index["configuration"].get("seals", {}).get("both_selections_frozen_before_test")
        is not True
    ):
        raise ValueError(
            "TEST gallery requires separate references and both frozen model selections"
        )
    reference_rows = None
    if references is not None:
        ref = json.loads(Path(references).read_text())
        if ref["input_index_sha256"] != sha256_file(input_index):
            raise ValueError("Gallery references refer to different frozen image inputs")
        reference_rows = {(r["dataset"], r["study_id"]): r for r in ref["records"]}
    if cases is None:
        cases = []
        for row in index["records"]:
            if row["split"] != "validation" or row["status"] != "ok":
                continue
            token = hashlib.sha256(f"{row['dataset']}:{row['study_id']}".encode()).hexdigest()[:24]
            if (features_dir / "records" / f"{token}.json").is_file():
                cases.append((row["dataset"], row["study_id"], "L5-S1"))
            if len(cases) == maximum:
                break
    if not cases or len(cases) > maximum:
        raise ValueError("Select between one and the configured maximum review cases")
    if (
        any(source_rows[(dataset, study)]["split"] == "test" for dataset, study, _ in cases)
        and phase != "evaluation_image_inputs"
    ):
        raise ValueError("TEST cannot be rendered through a fitting gallery")
    out.mkdir(parents=True, exist_ok=True)
    rendered, prepared_sources = [], {}
    for dataset, study_id, level in cases:
        if level not in LEVELS:
            raise ValueError("Unknown review level")
        source = source_rows[(dataset, study_id)]
        token = hashlib.sha256(f"{dataset}:{study_id}".encode()).hexdigest()[:24]
        record = json.loads((features_dir / "records" / f"{token}.json").read_text())
        if any(record[key] != source[key] for key in ("dataset", "study_id", "split")):
            raise ValueError("Gallery prediction identity differs from image inputs")
        prediction = record["provenance"]["prediction"]
        # Only these two image-source fields reach the crop decoder.
        image_source = {key: source[key] for key in ("source", "source_root")}
        if (dataset, study_id) not in prepared_sources:
            prepared_sources[(dataset, study_id)] = prepare_crop_source(image_source)
        prepared = prepared_sources[(dataset, study_id)]
        params = record["provenance"]["parameters"]
        crops = crop_level_inputs(
            image_source,
            prediction,
            roi_mm=params["roi_mm"],
            max_centers=params["max_centers"],
            window_mm=params["window_mm"],
            _prepared=prepared,
        )
        original_crop = record["provenance"]["crops"][level]
        if (
            original_crop["status"] != crops["levels"][level]["status"]
            or original_crop["metadata"] != crops["levels"][level]["metadata"]
        ):
            raise ValueError("Reconstructed review ROI differs from actual recorded prediction")
        anchors = (
            source.get("anchors", [])
            if reference_rows is None
            else reference_rows[(dataset, study_id)]["anchors"]
        )
        score = [r for r in score_case(source, record, anchors) if r["level"] == level][0]
        name = f"{dataset}-{token[:12]}-{level.lower()}.png"
        title = f"{dataset.upper()} · {level} · {source['split']} · {token[:12]}"
        image_info = render_case(
            out / name, prepared, prediction, crops, anchors, level, score, title
        )
        item = {
            "dataset": dataset,
            "study_id": study_id,
            "level": level,
            "split": source["split"],
            "image": name,
            "score": score,
            **image_info,
        }
        rendered.append(item)
        write_json(
            out / name.replace(".png", ".json"),
            item
            | {
                "image_source": image_source,
                "prediction": prediction,
                "crop_metadata": original_crop["metadata"],
            },
        )
    cards = []
    for row in rendered:
        caption = f"{row['dataset'].upper()} · {row['level']} · {row['split']}"
        cards.append(
            f'<article><h2>{html.escape(caption)}</h2><a href="{html.escape(row["image"])}"><img src="{html.escape(row["image"])}" alt="Actual predicted axial ROI and reference slice"></a><a href="{html.escape(row["image"].replace(".png", ".json"))}">Исходные параметры и метрики</a></article>'
        )
    page = (
        '<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Проверка локализации</title><style>body{background:#0b1119;color:#dbe4f1;font:16px system-ui;max-width:1450px;margin:35px auto;padding:0 20px}article{margin:35px 0;padding:18px;background:#111b27;border-radius:12px}h1,h2{font-weight:500}h2{font-size:18px}img{display:block;width:100%;height:auto;margin:16px 0}a{color:#b5cbec}p{line-height:1.6;max-width:950px}</style><h1>Проверка выбора уровня и области канала</h1><p>Жёлтым показаны предсказанный центр и физическая область 50 мм. Справа отдельно находится размеченный опорный срез. Нижний ряд — реальные три канала входа классификатора. Для RSNA точка разметки указывает приблизительный уровень и не служит эталоном центра канала. Разметка не участвовала в выборе среза или формировании области.</p>'
        + "".join(cards)
        + "</html>"
    )
    (out / "index.html").write_text(page)
    write_json(
        out / "gallery-private.json",
        {
            "input_index_sha256": sha256_file(input_index),
            "records": rendered,
            "images_are_private": True,
        },
    )
    return {"cases": len(rendered), "index": str((out / "index.html").resolve())}
