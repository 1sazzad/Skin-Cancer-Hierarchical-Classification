#!/usr/bin/env python3
"""VM-only audit of official EMB metadata and acquired images."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.emb_stage03 import map_stage_ajcc

ALIASES = {
    "image_id": ("image_id", "img_id", "image", "isic_id", "name", "filename"),
    "modality": ("modality", "image_type", "type"),
    "source": ("source", "dataset", "origin"),
    "thickness": ("thickness", "breslow", "breslow_thickness", "tumor_thickness"),
    "patient": ("patient_id", "patient"),
    "lesion": ("lesion_id", "lesion"),
}


def column(frame: pd.DataFrame, name: str, required: bool = False) -> str | None:
    lower = {item.lower(): item for item in frame.columns}
    for candidate in ALIASES[name]:
        if candidate in lower:
            return lower[candidate]
    if required:
        raise ValueError(f"Cannot identify {name} column; columns={list(frame.columns)}")
    return None


def resolve_image(images: Path, image_id: str) -> Path | None:
    direct = images / image_id
    if direct.is_file():
        return direct
    for suffix in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        matches = list(images.rglob(f"{image_id}{suffix}"))
        if matches:
            return matches[0]
    return None


def normalize_modality(value: object) -> str:
    raw = str(value).strip().lower().replace("_", " ").replace("-", " ")
    if raw in {"dermoscopy", "dermoscopic", "dermatoscopic", "dermatoscopy"}:
        return "dermoscopic"
    if raw in {"clinical", "clinical photograph", "clinical photography"}:
        return "clinical"
    return raw or "unknown"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-non-vm-fixture", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.project_root.resolve()
    if not args.allow_non_vm_fixture and (os.name != "posix" or not Path("/proc/driver/nvidia").exists()):
        raise SystemExit("NO-GO: real EMB audit is Azure GPU VM-only.")
    metadata = args.metadata or root / "data/raw/emb/early_melanoma_benchmark_dataset_labels.csv"
    images = args.images or root / "data/raw/emb/images"
    output = args.output or root / "reports/dataset_audits/emb_stage03_audit.json"
    frame = pd.read_csv(metadata)
    if "stage_ajcc" not in frame:
        raise ValueError("Official metadata is missing required stage_ajcc.")
    id_col = column(frame, "image_id", required=True)
    modality_col, source_col = column(frame, "modality"), column(frame, "source")
    thickness_col = column(frame, "thickness")
    ids = frame[id_col].astype(str).str.strip()
    stage_labels = frame["stage_ajcc"].map(map_stage_ajcc)
    hashes: dict[str, list[int]] = defaultdict(list)
    missing: list[str] = []
    unreadable: list[str] = []
    available: list[dict[str, str]] = []
    for index, image_id in ids.items():
        path = resolve_image(images, image_id)
        if path is None:
            missing.append(image_id)
            continue
        try:
            with Image.open(path) as opened:
                opened.verify()
        except (UnidentifiedImageError, OSError):
            unreadable.append(image_id)
            continue
        digest = sha256(path)
        hashes[digest].append(index)
        normalized = {str(key): value for key, value in frame.loc[index].to_dict().items()}
        normalized.update(
            {
                "image_id": image_id,
                "image_path": str(path.relative_to(root) if path.is_relative_to(root) else path),
                "file_sha256": digest,
                "stage_ajcc": frame.loc[index, "stage_ajcc"],
                "t_category": stage_labels.loc[index],
                "modality": (
                    normalize_modality(frame.loc[index, modality_col])
                    if modality_col else "unknown"
                ),
                "source": (
                    str(frame.loc[index, source_col]).strip().lower()
                    if source_col else "unknown"
                ),
            }
        )
        available.append(normalized)
    duplicates = {
        digest: [ids.loc[index] for index in indices]
        for digest, indices in hashes.items() if len(indices) > 1
    }
    conflicts = {
        digest: sorted({stage_labels.loc[index] for index in indices})
        for digest, indices in hashes.items()
        if len({stage_labels.loc[index] for index in indices}) > 1
    }
    isic_manifest = root / "data/manifests/isic2019_dataset_manifest.csv"
    old_ids: set[str] = set()
    old_hashes: set[str] = set()
    if isic_manifest.is_file():
        old = pd.read_csv(isic_manifest, dtype=str, keep_default_na=False)
        old_ids = set(old.get("image_id", pd.Series(dtype=str)).str.strip())
        old_hashes = set(old.get("file_sha256", pd.Series(dtype=str)).str.lower().str.strip()) - {""}
    thickness_issues: list[dict[str, object]] = []
    if thickness_col:
        for index, raw in frame[thickness_col].items():
            if pd.isna(raw) or str(raw).strip() == "":
                continue
            try:
                value = float(raw)
            except ValueError:
                thickness_issues.append({"image_id": ids.loc[index], "issue": "non_numeric"})
                continue
            label = stage_labels.loc[index]
            expected = (
                "T1" if 0 < value <= 1 else "T2" if value <= 2 else
                "T3" if value <= 4 else "T4" if value > 4 else None
            )
            if label != "Tis" and expected and label != expected:
                thickness_issues.append(
                    {"image_id": ids.loc[index], "stage": label, "thickness": value}
                )
    modalities = Counter(
        frame[modality_col].map(normalize_modality)
        if modality_col else ["unknown"] * len(frame)
    )
    sources = Counter(
        frame[source_col].astype(str).str.strip().str.lower()
        if source_col else ["unknown"] * len(frame)
    )
    available_hashes = set(hashes)
    fatal = bool(missing or unreadable or conflicts)
    payload = {
        "metadata_rows": len(frame),
        "unique_image_identifiers": ids.nunique(),
        "stage_counts": {name: int((stage_labels == name).sum()) for name in ("Tis","T1","T2","T3","T4")},
        "modality_counts": dict(modalities),
        "source_counts": dict(sources),
        "available_image_count": len(available),
        "missing_image_count": len(missing),
        "missing_image_ids": missing,
        "unreadable_image_count": len(unreadable),
        "unreadable_image_ids": unreadable,
        "exact_duplicate_sha256_groups": duplicates,
        "conflicting_duplicate_labels": conflicts,
        "isic_id_overlap": sorted(set(ids) & old_ids),
        "isic_sha256_overlap": sorted(available_hashes & old_hashes),
        "thickness_stage_consistency_issues": thickness_issues,
        "grouping_columns": {"patient": column(frame, "patient"), "lesion": column(frame, "lesion")},
        "verdict": "NO-GO" if fatal else "GO",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    available_path = output.with_name("emb_stage03_available_images.csv")
    pd.DataFrame(available).to_csv(available_path, index=False)
    print(payload["verdict"], output)
    return 1 if fatal else 0


if __name__ == "__main__":
    raise SystemExit(main())
