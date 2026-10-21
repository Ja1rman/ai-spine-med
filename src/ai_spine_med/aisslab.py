"""Prepare the pinned AISSLab axial release without decoding image pixels.

Patient directories define groups. Only explicit CCS annotations yield labels;
missing annotations remain unknown. The user-authorized grade mapping is a
research harmonization, not a claim that the two clinical grading scales agree.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import shutil
import stat
import uuid
import warnings
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import pydicom

EXPECTED_SHA256 = "b7d2119046f8a8e16d759edf54015b267857ff8d7fb0d3d153efef30d0dcd924"
EXPECTED_SIZE = 2_867_638_203
DATASET_URL = "https://data.mendeley.com/datasets/wjxkrkxhb8/1"
LEVELS = ("L4-L5", "L5-S1")
ALL_LEVELS = ("L1-L2", "L2-L3", "L3-L4", *LEVELS)
SPLITS = ("train", "validation", "test")
GRADE_MAP = {"normal": 0, "stenosis": 1, "severe_stenosis": 2}
MAX_ENTRIES = 50_000
MAX_FILE_BYTES = 64_000_000
MAX_TOTAL_BYTES = 8_000_000_000
MAX_COMPRESSION_RATIO = 500
MAX_XML_BYTES = 1_000_000
COMPONENTS = {"DICOM", "Localization_and_Severity_Classification", "Segmentation"}


class AisslabError(ValueError):
    """Stable diagnostics deliberately omit source identifiers and DICOM text."""


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _normalize_member(name: str) -> str:
    name = name.replace("\\", "/")
    if (
        not name
        or "\x00" in name
        or name.startswith("/")
        or re.match(r"^[A-Za-z]:", name)
        or any(part in {"", ".", ".."} for part in name.rstrip("/").split("/"))
        or any(":" in part for part in name.split("/"))
    ):
        raise AisslabError("unsafe_archive_path")
    return name.rstrip("/")


def _archive_inventory(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    if len(infos) > MAX_ENTRIES:
        raise AisslabError("archive_entry_limit")
    result, kinds, total = {}, {}, 0
    for info in infos:
        name = _normalize_member(info.filename)
        folded = name.casefold()
        if folded in kinds:
            raise AisslabError("archive_path_collision")
        mode = info.external_attr >> 16
        kind = stat.S_IFMT(mode)
        is_dir = info.filename.replace("\\", "/").endswith("/")
        if kind not in {0, stat.S_IFREG, stat.S_IFDIR} or (kind == stat.S_IFDIR and not is_dir):
            raise AisslabError("archive_special_file")
        if info.flag_bits & 1:
            raise AisslabError("encrypted_archive_entry")
        if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
            raise AisslabError("unsupported_zip_compression")
        if not 0 <= info.file_size <= MAX_FILE_BYTES:
            raise AisslabError("archive_file_limit")
        if info.file_size > max(1, info.compress_size) * MAX_COMPRESSION_RATIO:
            raise AisslabError("archive_compression_limit")
        total += info.file_size
        kinds[folded] = is_dir
        if not is_dir:
            result[name] = info
    if total > MAX_TOTAL_BYTES:
        raise AisslabError("archive_total_limit")
    for name in kinds:
        if any(kinds.get(str(parent)) is False for parent in PurePosixPath(name).parents):
            raise AisslabError("archive_file_directory_collision")
    return result


def _payload_index(inventory: dict) -> tuple[dict, dict, dict, dict]:
    groups = {component: defaultdict(dict) for component in COMPONENTS}
    roots = set()
    for name, info in inventory.items():
        parts = PurePosixPath(name).parts
        if len(parts) != 4 or parts[1] not in COMPONENTS or not re.fullmatch(r"[0-9]+", parts[2]):
            raise AisslabError("unexpected_release_layout")
        roots.add(parts[0])
        component, patient, filename = parts[1:]
        suffix = PurePosixPath(filename).suffix.lower()
        allowed = {".dcm"} if component == "DICOM" else {".png", ".xml"}
        if suffix not in allowed or (component == "Segmentation" and suffix != ".png"):
            raise AisslabError("unexpected_release_file_type")
        groups[component][patient][filename] = (name, info)
    if len(roots) != 1 or not groups["DICOM"]:
        raise AisslabError("unexpected_release_layout")
    dicoms = groups["DICOM"]
    if set(groups["Localization_and_Severity_Classification"]) - set(dicoms):
        raise AisslabError("annotations_without_patient_dicom")
    xmls, pngs = defaultdict(dict), defaultdict(dict)
    for patient, entries in groups["Localization_and_Severity_Classification"].items():
        for name, value in entries.items():
            target = xmls if name.lower().endswith(".xml") else pngs
            target[patient][PurePosixPath(name).stem] = value
    counts = {
        component: {
            "patient_directories": len(patients),
            "files_by_extension": dict(
                Counter(
                    PurePosixPath(name).suffix.lower()
                    for files in patients.values()
                    for name in files
                )
            ),
        }
        for component, patients in groups.items()
    }
    return dicoms, xmls, pngs, counts


def _extract_one(archive, info, root: Path, relative: str) -> dict:
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()) or path.exists():
        raise AisslabError("extraction_path_collision")
    path.parent.mkdir(parents=True, exist_ok=True)
    h, size = hashlib.sha256(), 0
    with archive.open(info) as source, path.open("xb") as destination:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            size += len(chunk)
            if size > info.file_size or size > MAX_FILE_BYTES:
                raise AisslabError("extraction_size_limit")
            h.update(chunk)
            destination.write(chunk)
    if size != info.file_size:
        raise AisslabError("truncated_archive_entry")
    return {"path": relative, "bytes": size, "sha256": h.hexdigest()}


def _numbers(ds, name, length):
    try:
        values = [float(x) for x in getattr(ds, name)]
    except (AttributeError, TypeError, ValueError):
        return None
    return values if len(values) == length and all(map(math.isfinite, values)) else None


def _dicom_header(path: Path) -> dict:
    try:
        with warnings.catch_warnings(record=True):
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        identity = {
            name: str(getattr(ds, name, ""))
            for name in (
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "SOPInstanceUID",
                "FrameOfReferenceUID",
            )
        }
        if not all(
            identity[name] for name in ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID")
        ):
            raise AisslabError("missing_dicom_identity")
        issues = []
        orientation = _numbers(ds, "ImageOrientationPatient", 6)
        position = _numbers(ds, "ImagePositionPatient", 3)
        spacing = _numbers(ds, "PixelSpacing", 2)
        if orientation is None or position is None or spacing is None:
            issues.append("missing_or_invalid_geometry")
        else:
            x, y = orientation[:3], orientation[3:]
            if (
                abs(math.hypot(*x) - 1) > 1e-3
                or abs(math.hypot(*y) - 1) > 1e-3
                or abs(sum(a * b for a, b in zip(x, y))) > 1e-3
            ):
                issues.append("nonorthonormal_image_orientation")
            normal_z = x[0] * y[1] - x[1] * y[0]
            if abs(normal_z) < math.cos(math.radians(45)):
                issues.append("not_axial_within_supported_obliquity")
            xnorm, ynorm = math.hypot(*x[:2]), math.hypot(*y[:2])
            if min(xnorm, ynorm) < 1e-6:
                issues.append("unsupported_inplane_rotation")
            else:
                col = [v / xnorm for v in x[:2]]
                row = [v / ynorm for v in y[:2]]
                if abs(col[1]) > abs(col[0]):
                    col, row = row, col
                if min(abs(col[0]), abs(row[1])) < math.cos(math.radians(20)):
                    issues.append("unsupported_inplane_rotation")
            if min(spacing) <= 0:
                issues.append("invalid_pixel_spacing")
        if str(getattr(ds, "Modality", "MR")) != "MR" or int(getattr(ds, "NumberOfFrames", 1)) != 1:
            issues.append("unsupported_dicom_modality_or_multiframe")
        if int(getattr(ds, "SamplesPerPixel", 1)) != 1:
            issues.append("unsupported_color_image")
        photometric = str(getattr(ds, "PhotometricInterpretation", ""))
        if photometric not in {"MONOCHROME1", "MONOCHROME2"}:
            issues.append("unsupported_photometric_interpretation")
        shape = [int(getattr(ds, key, 0)) for key in ("Rows", "Columns")]
        if min(shape) < 2:
            issues.append("invalid_image_shape")
        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        if not math.isfinite(slope) or not math.isfinite(intercept) or slope == 0:
            issues.append("invalid_rescale_parameters")
        instance = int(ds.InstanceNumber) if "InstanceNumber" in ds else None
        if instance is None or instance < 0:
            issues.append("missing_or_invalid_instance_number")
        return {
            "identity": identity,
            "issues": issues,
            "instance": instance,
            "shape": shape,
            "orientation": orientation,
            "position": position,
            "spacing": spacing,
            "photometric": photometric,
        }
    except AisslabError:
        raise
    except Exception:
        raise AisslabError("dicom_header_failed") from None


def _parse_labels(content: bytes) -> list[dict]:
    if (
        len(content) > MAX_XML_BYTES
        or b"\x00" in content
        or re.search(rb"<!\s*(DOCTYPE|ENTITY)", content, re.I)
    ):
        raise AisslabError("unsafe_annotation_xml")
    try:
        root = ET.fromstring(content)
        if root.tag != "annotation":
            raise AisslabError("invalid_annotation_root")
        labels = []
        for index, obj in enumerate(root.findall("./object")):
            name = (obj.findtext("name") or "").strip()
            if not name.startswith("CCS_"):
                continue
            suffix = name.removeprefix("CCS_")
            if suffix not in GRADE_MAP:
                raise AisslabError("unknown_ccs_grade")
            level = (obj.findtext("level") or "").strip()
            if level not in ALL_LEVELS:
                raise AisslabError("missing_or_unknown_ccs_level")
            if level in LEVELS:
                labels.append(
                    {
                        "level": level,
                        "grade": GRADE_MAP[suffix],
                        "source_grade": suffix,
                        "object_index": index,
                    }
                )
        return labels
    except AisslabError:
        raise
    except (ET.ParseError, ValueError, TypeError):
        raise AisslabError("invalid_annotation_xml") from None


def _allocate(n: int, ratios: tuple) -> list[int]:
    values = [n * x for x in ratios]
    result = [int(x) for x in values]
    for i in sorted(range(3), key=lambda i: (-(values[i] - result[i]), i))[: n - sum(result)]:
        result[i] += 1
    return result


def _patient_splits(records: list[dict], seed: int, ratios: tuple) -> dict[str, str]:
    """Joint-label strata >=10 use proportional quotas; sparse strata are pooled."""
    targets = _allocate(len(records), ratios)
    remaining = targets.copy()
    groups = defaultdict(list)
    for row in sorted(records, key=lambda r: r["study_id"]):
        groups[tuple(row["labels"])].append(row["study_id"])
    rng, result, pooled = random.Random(seed), {}, []
    for key, studies in sorted(groups.items(), key=lambda item: repr(item[0])):
        rng.shuffle(studies)
        if len(studies) < 10:
            pooled.extend(studies)
            continue
        quotas = [
            min(count, remaining[i]) for i, count in enumerate(_allocate(len(studies), ratios))
        ]
        offset = 0
        for i, count in enumerate(quotas):
            result.update({study: SPLITS[i] for study in studies[offset : offset + count]})
            remaining[i] -= count
            offset += count
        pooled.extend(studies[offset:])
    rng.shuffle(pooled)
    offset = 0
    for i, count in enumerate(remaining):
        result.update({study: SPLITS[i] for study in pooled[offset : offset + count]})
        offset += count
    if len(result) != len(records) or len({r["study_id"] for r in records}) != len(records):
        raise AisslabError("duplicate_or_unassigned_patient")
    return result


def _prepare_records(archive, extracted, dicoms, xmls, pngs):
    records, extracted_files = [], []
    payload_owners = {}
    uid_owners = {key: defaultdict(set) for key in ("StudyInstanceUID", "SOPInstanceUID")}
    matched, fallback = 0, 0
    for patient in sorted(dicoms, key=lambda x: (int(x), x)):
        token = _hash("aisslab-patient:" + patient)
        prefix = f"patients/{token[:2]}/{token}"
        series, stem_links, headers = {}, {}, defaultdict(list)
        for filename, (member, info) in sorted(dicoms[patient].items()):
            # Header inspection never accesses pixel_array or patient-name fields.
            temporary = _extract_one(
                archive, info, extracted, f"{prefix}/incoming/{_hash(member)}.dcm"
            )
            old_path = extracted / temporary["path"]
            header = _dicom_header(old_path)
            identity = header.pop("identity")
            owner = payload_owners.setdefault(temporary["sha256"], patient)
            if owner != patient:
                raise AisslabError("exact_dicom_payload_crosses_patients")
            for kind in uid_owners:
                uid_owners[kind][_hash(identity[kind])].add(f"aisslab-{patient}")
            uid = identity["SeriesInstanceUID"]
            series_id = str(int(_hash("aisslab-series:" + patient + ":" + uid)[:16], 16))
            series_path = f"{prefix}/series/{series_id}"
            name = f"{_hash(member)}.dcm"
            destination = extracted / series_path / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            old_path.replace(destination)
            receipt = {"name": name, "bytes": temporary["bytes"], "sha256": temporary["sha256"]}
            extracted_files.append(
                {"path": f"{series_path}/{name}", **{k: receipt[k] for k in ("bytes", "sha256")}}
            )
            current = series.setdefault(
                uid,
                {
                    "series_id": series_id,
                    "series_path": series_path,
                    "files": [],
                    "series_uid_sha256": _hash(uid),
                    "geometry_issues": [],
                },
            )
            current["files"].append(receipt)
            header["study_uid_sha256"] = _hash(identity["StudyInstanceUID"])
            header["frame_uid_sha256"] = _hash(identity["FrameOfReferenceUID"])
            header["sop_uid_sha256"] = _hash(identity["SOPInstanceUID"])
            headers[uid].append(header)
            stem = PurePosixPath(filename).stem
            if stem in stem_links:
                raise AisslabError("ambiguous_dicom_stem")
            stem_links[stem] = {"series_id": series_id, "dicom_path": f"{series_path}/{name}"}
        for uid, current in series.items():
            items = headers[uid]
            issues = set(issue for h in items for issue in h["issues"])
            for key in ("study_uid_sha256", "frame_uid_sha256", "photometric"):
                if len({h[key] for h in items}) != 1:
                    issues.add(
                        "mixed_dicom_identity_or_frame"
                        if key != "photometric"
                        else "inconsistent_photometric_interpretation"
                    )
            for key in ("instance", "sop_uid_sha256"):
                if len({h[key] for h in items}) != len(items):
                    issues.add(
                        "duplicate_instance_number"
                        if key == "instance"
                        else "duplicate_sop_instance_uid"
                    )
            current["geometry_issues"] = sorted(issues)
            current["metadata_inventory"] = {
                "slices": len(items),
                "image_shapes": sorted({tuple(h["shape"]) for h in items}),
                "distinct_orientations": len({tuple(h["orientation"] or []) for h in items}),
                "distinct_positions": len({tuple(h["position"] or []) for h in items}),
                "distinct_pixel_spacings": len({tuple(h["spacing"] or []) for h in items}),
            }
        refs = []
        for stem, (member, info) in sorted(xmls.get(patient, {}).items()):
            if info.file_size > MAX_XML_BYTES:
                raise AisslabError("annotation_size_limit")
            annotations = _parse_labels(archive.read(info))
            xml_receipt = _extract_one(
                archive, info, extracted, f"{prefix}/annotations/{_hash(member)}.xml"
            )
            extracted_files.append(xml_receipt)
            if stem not in pngs.get(patient, {}):
                raise AisslabError("annotation_png_missing")
            link = stem_links.get(stem)
            if link is not None:
                matched += 1
                image_ref = {"image_type": "dicom", **link}
            else:
                fallback += 1
                png_member, png_info = pngs[patient][stem]
                png_receipt = _extract_one(
                    archive, png_info, extracted, f"{prefix}/annotation_png/{_hash(png_member)}.png"
                )
                extracted_files.append(png_receipt)
                image_ref = {"image_type": "png_fallback", "png_path": png_receipt["path"]}
            for annotation in annotations:
                refs.append({**annotation, "xml_path": xml_receipt["path"], **image_ref})
        labels = [
            max((ref["grade"] for ref in refs if ref["level"] == level), default=None)
            for level in LEVELS
        ]
        source_series = sorted(series.values(), key=lambda s: int(s["series_id"]))
        study_hashes = sorted({h["study_uid_sha256"] for group in headers.values() for h in group})
        source = {
            "series": source_series,
            "study_uid_sha256": study_hashes[0] if len(study_hashes) == 1 else None,
            "study_uid_sha256s": study_hashes,
        }
        source["sha256"] = hashlib.sha256(_json_bytes(source_series)).hexdigest()
        records.append(
            {
                "dataset": "aisslab",
                "study_id": f"aisslab-{patient}",
                "labels": labels,
                "label_references": refs,
                "source": source,
                "metadata_geometry_pass": all(not s["geometry_issues"] for s in source_series),
            }
        )
        (extracted / prefix / "incoming").rmdir()
    reused = {
        kind: {uid: owners for uid, owners in values.items() if len(owners) > 1}
        for kind, values in uid_owners.items()
    }
    for record in records:
        record["candidate_uid_overlap"] = {
            kind: sorted(uid for uid, owners in values.items() if record["study_id"] in owners)
            for kind, values in reused.items()
        }
    audit = {
        "confirmed_cross_patient_exact_payload_duplicates": 0,
        "candidate_reused_uid_hashes": {
            kind: {uid: len(owners) for uid, owners in values.items()}
            for kind, values in reused.items()
        },
        "patients_with_candidate_uid_overlap": sum(
            any(r["candidate_uid_overlap"].values()) for r in records
        ),
        "interpretation": "A reused UID can reflect anonymization resets and is a candidate overlap, not proof of a shared patient. No automatic merges; compare exact bytes and review before training.",
    }
    return (
        records,
        extracted_files,
        {"xml_dicom_stem_matches": matched, "xml_png_fallbacks": fallback},
        audit,
    )


def _validate_existing(out: Path, config: dict) -> dict:
    try:
        manifest = json.loads((out / "manifest.json").read_text())
        if manifest["configuration"] != config:
            raise AisslabError("existing_preparation_configuration_differs")
        for name, checksum in manifest["output_sha256"].items():
            if (
                name not in {"records.json", "patient_splits.json", "extracted_files.json"}
                or _sha(out / name) != checksum
            ):
                raise AisslabError("existing_manifest_changed")
        for item in json.loads((out / "extracted_files.json").read_text()):
            relative = _normalize_member(item["path"])
            path = out / "extracted" / relative
            if path.is_symlink() or not path.resolve().is_relative_to(
                (out / "extracted").resolve()
            ):
                raise AisslabError("unsafe_existing_extraction")
            if path.stat().st_size != item["bytes"] or _sha(path) != item["sha256"]:
                raise AisslabError("existing_extraction_changed")
        return manifest
    except AisslabError:
        raise
    except (OSError, ValueError, KeyError, TypeError):
        raise AisslabError("incomplete_existing_preparation") from None


def prepare_aisslab(
    archive: str | Path,
    out: str | Path,
    *,
    seed: int = 2026,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    expected_sha256: str = EXPECTED_SHA256,
    expected_size: int = EXPECTED_SIZE,
) -> dict:
    """Atomically prepare local records and patient splits from a verified ZIP.

    Explicit checksum/size overrides support synthetic fixtures or a separately
    verified new release. All image reads here use stop_before_pixels=True.
    The original ZIP is retained. Existing outputs are verified, never resplit.
    """
    archive, out = Path(archive), Path(out)
    ratios = tuple(ratios)
    if (
        type(seed) is not int
        or len(ratios) != 3
        or not all(math.isfinite(x) and 0 < x < 1 for x in ratios)
        or not math.isclose(sum(ratios), 1, abs_tol=1e-12)
    ):
        raise AisslabError("invalid_split_configuration")
    if (
        not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
        or type(expected_size) is not int
        or expected_size <= 0
    ):
        raise AisslabError("invalid_archive_pin")
    if archive.is_symlink() or not archive.is_file() or archive.stat().st_size != expected_size:
        raise AisslabError("archive_size_or_path_mismatch")
    if _sha(archive) != expected_sha256:
        raise AisslabError("archive_checksum_mismatch")
    archive_stat = archive.stat()
    config = {
        "seed": seed,
        "ratios": list(ratios),
        "archive_sha256": expected_sha256,
        "archive_bytes": expected_size,
        "levels": list(LEVELS),
        "grade_mapping": GRADE_MAP,
    }
    if out.is_symlink():
        raise AisslabError("unsafe_output_directory")
    if out.exists() and any(out.iterdir()):
        return _validate_existing(out, config)
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = out.parent / f".{out.name}.prepare-{uuid.uuid4().hex}"
    stage.mkdir()
    try:
        extracted = stage / "extracted"
        extracted.mkdir()
        with zipfile.ZipFile(archive) as source:
            inventory = _archive_inventory(source)
            dicoms, xmls, pngs, counts = _payload_index(inventory)
            records, extracted_files, linking, identity_audit = _prepare_records(
                source, extracted, dicoms, xmls, pngs
            )
        if (archive.stat().st_size, archive.stat().st_mtime_ns) != (
            archive_stat.st_size,
            archive_stat.st_mtime_ns,
        ):
            raise AisslabError("source_archive_changed")
        splits = _patient_splits(records, seed, ratios)
        for row in records:
            row["split"] = splits[row["study_id"]]
        outputs = {
            "records.json": records,
            "patient_splits.json": splits,
            "extracted_files.json": extracted_files,
        }
        for name, value in outputs.items():
            (stage / name).write_bytes(_json_bytes(value))
        manifest = {
            "schema_version": 1,
            "dataset": "aisslab",
            "dataset_url": DATASET_URL,
            "configuration": config,
            "archive_verified": True,
            "extracted_root": "extracted",
            "records_path": "records.json",
            "patients": len(records),
            "source_inventory": counts,
            "annotation_linking": linking,
            "identity_audit": identity_audit,
            "patients_by_split": dict(Counter(splits.values())),
            "patients_with_metadata_geometry_issues": sum(
                not r["metadata_geometry_pass"] for r in records
            ),
            "patients_with_multiple_series": sum(len(r["source"]["series"]) > 1 for r in records),
            "known_labels_by_level": {
                level: sum(r["labels"][i] is not None for r in records)
                for i, level in enumerate(LEVELS)
            },
            "grade_counts_by_level": {
                level: dict(Counter(str(r["labels"][i]) for r in records))
                for i, level in enumerate(LEVELS)
            },
            "policies": {
                "mapping": "User-authorized research mapping normal=0, stenosis=1, severe_stenosis=2; clinical equivalence is not established.",
                "missing_label": "null; absence of an explicit CCS annotation never means normal",
                "aggregation": "Maximum explicit CCS grade for each patient and level; retain every target-level annotation reference, including discordant grades.",
                "split": "Group by original numeric patient directory. Joint two-level label strata with >=10 patients receive proportional quotas; smaller strata and rounding overflow use deterministic shuffled global-quota fallback.",
                "leakage": "Reject identical DICOM payloads across patient directories. Reused StudyInstanceUIDs or SOPInstanceUIDs without identical bytes are candidate overlaps requiring review, not automatic patient merges. All series stay in their patient's split.",
                "extraction": "Only DICOM and XML; same-stem PNG only when an XML has no same-stem DICOM. Segmentation images excluded. Output paths hash and shard patient identifiers.",
                "geometry": "All DICOM headers inspected without pixels; series grouped by actual SeriesInstanceUID. Geometry problems are explicit per series and per patient, not silently repaired.",
            },
            "image_pixels_decoded": False,
            "output_sha256": {name: _sha(stage / name) for name in outputs},
        }
        (stage / "manifest.json").write_bytes(_json_bytes(manifest))
        if out.exists():
            out.rmdir()
        stage.replace(out)
        return manifest
    except AisslabError:
        raise
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile):
        raise AisslabError("release_preparation_failed") from None
    finally:
        if stage.exists():
            shutil.rmtree(stage)
