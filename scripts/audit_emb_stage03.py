#!/usr/bin/env python3
"""VM-only audit of the official ISIC-derived Stage-3 inventory and images."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.emb_stage03 import map_stage_ajcc


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-non-vm-fixture", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.project_root.resolve()
    if not args.allow_non_vm_fixture and (
        os.name != "posix"
        or not (Path("/proc/driver/nvidia").exists() or shutil.which("nvidia-smi"))
    ):
        raise SystemExit("NO-GO: real ISIC Stage-3 audit is GPU VM-only.")
    metadata = args.metadata or root / "data/raw/emb/isic_stage03_official_inventory.csv"
    images = args.images or root / "data/raw/emb/images/isic"
    output = args.output or root / "reports/dataset_audits/isic_stage03_image_audit.json"
    frame = pd.read_csv(metadata, dtype=str, keep_default_na=False)
    required = {
        "image_id", "eligible", "derived_stage_ajcc", "t_category", "modality",
        "copyright_license", "attribution", "patient_id", "lesion_id",
        "original_vs_official_agreement", "image_path", "file_sha256",
    }
    missing_columns = sorted(required - set(frame))
    if missing_columns:
        raise ValueError(f"Official inventory is missing columns: {missing_columns}")
    selected = frame.loc[frame["eligible"].map(truthy)].copy()
    if selected.empty:
        raise ValueError("Official inventory contains no eligible rows.")
    mapped = selected["derived_stage_ajcc"].map(map_stage_ajcc)
    if not mapped.equals(selected["t_category"].str.strip()):
        raise ValueError("derived_stage_ajcc disagrees with t_category.")
    missing_images: list[str] = []
    unreadable: list[str] = []
    hashes: dict[str, list[str]] = defaultdict(list)
    observed_hash_by_id: dict[str, str] = {}
    inventory_hash_mismatches: list[str] = []
    for _, row in selected.iterrows():
        raw_path = str(row["image_path"]).strip()
        path = (root / raw_path).resolve() if raw_path else images / f"{row['image_id']}.jpg"
        if not path.is_file():
            missing_images.append(row["image_id"])
            continue
        try:
            with Image.open(path) as opened:
                opened.verify()
        except (UnidentifiedImageError, OSError):
            unreadable.append(row["image_id"])
            continue
        digest = sha256(path)
        observed_hash_by_id[row["image_id"]] = digest
        hashes[digest].append(row["image_id"])
        recorded = str(row["file_sha256"]).strip().lower()
        if recorded and recorded != digest:
            inventory_hash_mismatches.append(row["image_id"])
    duplicate_groups = {
        digest: ids for digest, ids in hashes.items() if len(ids) > 1
    }
    label_by_id = dict(zip(selected["image_id"], selected["t_category"], strict=True))
    conflicting_duplicates = {
        digest: sorted({label_by_id[image_id] for image_id in ids})
        for digest, ids in duplicate_groups.items()
        if len({label_by_id[image_id] for image_id in ids}) > 1
    }
    old_ids: set[str] = set()
    old_hashes: set[str] = set()
    old_manifest = root / "data/manifests/isic2019_dataset_manifest.csv"
    if old_manifest.is_file():
        old = pd.read_csv(old_manifest, dtype=str, keep_default_na=False)
        old_ids = set(old.get("image_id", pd.Series(dtype=str)).str.strip())
        old_hashes = set(
            old.get("file_sha256", pd.Series(dtype=str)).str.strip().str.lower()
        ) - {""}
    overlap_ids = sorted(set(selected["image_id"]) & old_ids)
    overlap_set = set(overlap_ids)
    fatal = bool(
        missing_images
        or unreadable
        or conflicting_duplicates
        or inventory_hash_mismatches
    )
    payload = {
        "dataset_name": "ISIC-derived melanoma T-category subset",
        "metadata_provenance": "official ISIC Archive API v2 image responses",
        "authoritative_label_fields": ["derived_stage_ajcc", "t_category"],
        "original_emb_stage_role": "audit_comparison_only",
        "eligible_row_count": len(selected),
        "official_t_category_counts": dict(Counter(selected["t_category"])),
        "licence_counts": dict(Counter(selected["copyright_license"])),
        "attribution_recorded_count": int(selected["attribution"].str.strip().ne("").sum()),
        "attribution_missing_count": int(selected["attribution"].str.strip().eq("").sum()),
        "patient_id_available_count": int(selected["patient_id"].str.strip().ne("").sum()),
        "lesion_id_available_count": int(selected["lesion_id"].str.strip().ne("").sum()),
        "unique_patient_count": selected.loc[selected["patient_id"].str.strip().ne(""), "patient_id"].nunique(),
        "unique_lesion_count": selected.loc[selected["lesion_id"].str.strip().ne(""), "lesion_id"].nunique(),
        "original_official_agreement_count": int(
            selected["original_vs_official_agreement"].eq("true").sum()
        ),
        "original_official_disagreement_ids": selected.loc[
            selected["original_vs_official_agreement"].eq("false"), "image_id"
        ].tolist(),
        "missing_image_count": len(missing_images),
        "missing_image_ids": missing_images,
        "unreadable_image_count": len(unreadable),
        "unreadable_image_ids": unreadable,
        "exact_duplicate_sha256_groups": duplicate_groups,
        "conflicting_duplicate_labels": conflicting_duplicates,
        "inventory_sha256_mismatch_ids": inventory_hash_mismatches,
        "isic2019_id_overlap_count": len(overlap_ids),
        "isic2019_overlap_ids": overlap_ids,
        "isic2019_id_overlap_by_t_category": dict(Counter(
            row["t_category"] for _, row in selected.iterrows()
            if row["image_id"] in overlap_set
        )),
        "isic2019_sha256_overlap": sorted(set(observed_hash_by_id.values()) & old_hashes),
        "verdict": "NO-GO" if fatal else "GO",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(payload["verdict"], output)
    return 1 if fatal else 0


if __name__ == "__main__":
    raise SystemExit(main())
