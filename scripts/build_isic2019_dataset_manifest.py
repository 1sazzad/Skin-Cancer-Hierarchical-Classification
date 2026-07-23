#!/usr/bin/env python3
"""Build the ISIC 2019 manifest and lightweight audit reports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from PIL import Image, UnidentifiedImageError
except ImportError as exc:
    raise SystemExit(
        "Pillow is required. Install it with: python -m pip install Pillow"
    ) from exc

EXPECTED_COUNT = 25_331
DIAGNOSIS_COLUMNS = ("MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK")
CANONICAL = {
    "MEL": "melanoma",
    "NV": "melanocytic_nevus",
    "BCC": "basal_cell_carcinoma",
    "AK": "actinic_keratosis",
    "BKL": "benign_keratosis_like_lesion",
    "DF": "dermatofibroma",
    "VASC": "vascular_lesion",
    "SCC": "squamous_cell_carcinoma",
    "UNK": "unknown",
}
STAGE_2 = {"MEL": "melanoma", "BCC": "bcc", "SCC": "scc"}

MANIFEST_COLUMNS = (
    "dataset", "image_id", "image_path", "source_split",
    "diagnosis_original", "diagnosis_canonical", "stage_1_label",
    "stage_2_label", "stage_3_label", "patient_id", "lesion_id",
    "age_approx", "sex", "anatom_site_general", "include_stage_1",
    "include_stage_2", "include_stage_3", "exclusion_reason",
    "file_extension", "file_size_bytes", "file_sha256",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument(
        "--ak-policy",
        choices=("exclude", "non_malignant"),
        default="exclude",
        help="Primary default excludes premalignant AK from Stage 1.",
    )
    parser.add_argument("--skip-image-hash", action="store_true")
    parser.add_argument("--skip-image-decode-check", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def project_root_from(args: argparse.Namespace) -> Path:
    if args.project_root:
        return args.project_root.expanduser().resolve()
    return Path(__file__).resolve().parent.parent


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV header missing: {path}")
        return [dict(row) for row in reader], list(reader.fieldnames)


def require_columns(headers: list[str], required: tuple[str, ...], name: str) -> None:
    missing = [column for column in required if column not in headers]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def require_unique(values: list[str], name: str) -> None:
    duplicates = [key for key, count in Counter(values).items() if count > 1]
    if duplicates:
        raise ValueError(f"{name} duplicate IDs, examples: {duplicates[:10]}")


def one_hot_diagnosis(row: dict[str, str]) -> str:
    active: list[str] = []
    image_id = (row.get("image") or "").strip()
    for column in DIAGNOSIS_COLUMNS:
        raw = (row.get(column) or "").strip()
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(
                f"Non-numeric label: image={image_id}, column={column}, value={raw!r}"
            ) from exc
        if value not in (0.0, 1.0):
            raise ValueError(
                f"Non-binary label: image={image_id}, column={column}, value={value}"
            )
        if value == 1.0:
            active.append(column)
    if len(active) != 1:
        raise ValueError(f"Invalid one-hot row: image={image_id}, active={active}")
    return active[0]


def label_mapping(diagnosis: str, ak_policy: str) -> tuple[str, str, int, int, str]:
    if diagnosis in {"MEL", "BCC", "SCC"}:
        return "malignant", STAGE_2[diagnosis], 1, 1, ""
    if diagnosis in {"NV", "BKL", "DF", "VASC"}:
        return "non_malignant", "", 1, 0, ""
    if diagnosis == "AK":
        if ak_policy == "non_malignant":
            return "non_malignant", "", 1, 0, ""
        return "", "", 0, 0, "actinic_keratosis_excluded_from_primary_binary_task"
    if diagnosis == "UNK":
        return "", "", 0, 0, "unknown_diagnosis"
    raise ValueError(f"Unsupported diagnosis: {diagnosis}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(path: Path, hash_image: bool, verify_decode: bool) -> dict[str, Any]:
    width = height = 0
    mode = ""
    if verify_decode:
        try:
            with Image.open(path) as image:
                width, height = image.size
                mode = image.mode
                image.verify()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValueError(f"Image verification failed: {path}") from exc
    stat = path.stat()
    return {
        "extension": path.suffix.lower(),
        "size_bytes": stat.st_size,
        "sha256": sha256_file(path) if hash_image else "",
        "width": width,
        "height": height,
        "mode": mode,
    }


def atomic_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", delete=False,
        dir=path.parent, prefix=f"{path.name}.", suffix=".tmp"
    ) as handle:
        temp_path = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", delete=False,
        dir=path.parent, prefix=f"{path.name}.", suffix=".tmp"
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temp_path, path)


def main() -> int:
    args = parse_args()
    root = project_root_from(args)

    image_root = root / "data" / "raw" / "isic2019" / "images"
    metadata_root = root / "data" / "raw" / "isic2019" / "metadata"
    ground_truth_path = metadata_root / "ISIC_2019_Training_GroundTruth.csv"
    metadata_path = metadata_root / "ISIC_2019_Training_Metadata.csv"

    manifest_path = root / "data" / "manifests" / "isic2019_dataset_manifest.csv"
    audit_path = root / "reports" / "dataset_audits" / "isic2019_manifest_audit.json"
    distribution_path = root / "reports" / "dataset_audits" / "isic2019_class_distribution.csv"

    if not image_root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_root}")

    gt_rows, gt_headers = read_csv(ground_truth_path)
    metadata_rows, metadata_headers = read_csv(metadata_path)

    require_columns(gt_headers, ("image", *DIAGNOSIS_COLUMNS), "Ground truth")
    require_columns(
        metadata_headers,
        ("image", "age_approx", "anatom_site_general", "lesion_id", "sex"),
        "Metadata",
    )

    if len(gt_rows) != EXPECTED_COUNT or len(metadata_rows) != EXPECTED_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_COUNT} rows; got ground_truth={len(gt_rows)}, "
            f"metadata={len(metadata_rows)}"
        )

    gt_ids = [(row.get("image") or "").strip() for row in gt_rows]
    metadata_ids = [(row.get("image") or "").strip() for row in metadata_rows]
    require_unique(gt_ids, "Ground truth")
    require_unique(metadata_ids, "Metadata")

    metadata_by_id = {(row.get("image") or "").strip(): row for row in metadata_rows}
    image_by_id: dict[str, Path] = {}
    for path in image_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}:
            if path.stem in image_by_id:
                raise ValueError(f"Duplicate image basename: {path.stem}")
            image_by_id[path.stem] = path

    gt_set = set(gt_ids)
    metadata_set = set(metadata_ids)
    image_set = set(image_by_id)
    if gt_set != metadata_set or gt_set != image_set:
        raise ValueError(
            "ID mismatch: "
            f"missing_metadata={len(gt_set - metadata_set)}, "
            f"unexpected_metadata={len(metadata_set - gt_set)}, "
            f"missing_images={len(gt_set - image_set)}, "
            f"unexpected_images={len(image_set - gt_set)}"
        )

    diagnosis_counts: Counter[str] = Counter()
    stage1_counts: Counter[str] = Counter()
    stage2_counts: Counter[str] = Counter()
    exclusion_counts: Counter[str] = Counter()
    missing = Counter({"age_approx": 0, "sex": 0, "anatom_site_general": 0, "lesion_id": 0})
    lesion_counts: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    manifest_rows: list[dict[str, Any]] = []

    sorted_rows = sorted(gt_rows, key=lambda row: (row.get("image") or "").strip())
    print(f"[START] Processing {len(sorted_rows)} images")
    print(f"[POLICY] AK: {args.ak_policy}")

    for index, gt_row in enumerate(sorted_rows, start=1):
        image_id = (gt_row.get("image") or "").strip()
        metadata = metadata_by_id[image_id]
        diagnosis = one_hot_diagnosis(gt_row)
        diagnosis_counts[diagnosis] += 1

        stage1, stage2, include1, include2, exclusion = label_mapping(diagnosis, args.ak_policy)
        if include1:
            stage1_counts[stage1] += 1
        if include2:
            stage2_counts[stage2] += 1
        if exclusion:
            exclusion_counts[exclusion] += 1

        age = (metadata.get("age_approx") or "").strip()
        sex = (metadata.get("sex") or "").strip()
        site = (metadata.get("anatom_site_general") or "").strip()
        lesion_id = (metadata.get("lesion_id") or "").strip()
        for field, value in (("age_approx", age), ("sex", sex), ("anatom_site_general", site), ("lesion_id", lesion_id)):
            if not value:
                missing[field] += 1
        if lesion_id:
            lesion_counts[lesion_id] += 1

        image_path = image_by_id[image_id]
        image_info = inspect_image(
            image_path,
            hash_image=not args.skip_image_hash,
            verify_decode=not args.skip_image_decode_check,
        )
        if image_info["width"] and image_info["height"]:
            dimensions[f"{image_info['width']}x{image_info['height']}"] += 1
        if image_info["mode"]:
            modes[image_info["mode"]] += 1

        manifest_rows.append({
            "dataset": "isic2019",
            "image_id": image_id,
            "image_path": image_path.relative_to(root).as_posix(),
            "source_split": "official_training_pool",
            "diagnosis_original": diagnosis,
            "diagnosis_canonical": CANONICAL[diagnosis],
            "stage_1_label": stage1,
            "stage_2_label": stage2,
            "stage_3_label": "",
            "patient_id": "",
            "lesion_id": lesion_id,
            "age_approx": age,
            "sex": sex,
            "anatom_site_general": site,
            "include_stage_1": include1,
            "include_stage_2": include2,
            "include_stage_3": 0,
            "exclusion_reason": exclusion,
            "file_extension": image_info["extension"],
            "file_size_bytes": image_info["size_bytes"],
            "file_sha256": image_info["sha256"],
        })

        if args.progress_every > 0 and index % args.progress_every == 0:
            print(f"[PROGRESS] {index}/{len(sorted_rows)}")

    distribution_rows: list[dict[str, Any]] = []
    for diagnosis in DIAGNOSIS_COLUMNS:
        stage1, stage2, include1, include2, exclusion = label_mapping(diagnosis, args.ak_policy)
        distribution_rows.append({
            "diagnosis_original": diagnosis,
            "diagnosis_canonical": CANONICAL[diagnosis],
            "count": diagnosis_counts[diagnosis],
            "stage_1_label": stage1,
            "stage_2_label": stage2,
            "include_stage_1": include1,
            "include_stage_2": include2,
            "exclusion_reason": exclusion,
        })

    repeated_groups = {key: count for key, count in lesion_counts.items() if count > 1}
    audit = {
        "dataset": "isic2019",
        "status": "complete",
        "policy": {
            "ak_policy": args.ak_policy,
            "unknown_policy": "exclude",
            "stage_1_malignant_classes": ["MEL", "BCC", "SCC"],
            "stage_1_non_malignant_classes": ["NV", "BKL", "DF", "VASC"],
            "stage_2_classes": ["MEL", "BCC", "SCC"],
        },
        "counts": {
            "ground_truth_rows": len(gt_rows),
            "metadata_rows": len(metadata_rows),
            "image_files": len(image_by_id),
            "manifest_rows": len(manifest_rows),
            "stage_1_eligible": sum(stage1_counts.values()),
            "stage_1_excluded": sum(exclusion_counts.values()),
            "stage_2_eligible": sum(stage2_counts.values()),
        },
        "class_distribution_original": dict(sorted(diagnosis_counts.items())),
        "class_distribution_stage_1": dict(sorted(stage1_counts.items())),
        "class_distribution_stage_2": dict(sorted(stage2_counts.items())),
        "exclusions": dict(sorted(exclusion_counts.items())),
        "metadata_missingness": dict(missing),
        "identifier_audit": {
            "patient_id_column_available": "patient_id" in metadata_headers,
            "lesion_id_column_available": "lesion_id" in metadata_headers,
            "non_empty_lesion_id_rows": sum(lesion_counts.values()),
            "unique_non_empty_lesion_ids": len(lesion_counts),
            "repeated_lesion_group_count": len(repeated_groups),
            "images_in_repeated_lesion_groups": sum(repeated_groups.values()),
            "maximum_images_per_lesion": max(lesion_counts.values(), default=0),
            "recommended_split_group": "lesion_id",
            "fallback_required_for_missing_lesion_id": missing["lesion_id"] > 0,
        },
        "image_audit": {
            "decode_check_performed": not args.skip_image_decode_check,
            "per_image_sha256_computed": not args.skip_image_hash,
            "invalid_image_count": 0,
            "image_mode_distribution": dict(modes.most_common()),
            "top_dimension_distribution": dict(dimensions.most_common(20)),
            "unique_dimension_count": len(dimensions),
        },
        "identifier_consistency": {
            "ground_truth_ids_unique": True,
            "metadata_ids_unique": True,
            "image_ids_unique": True,
            "ground_truth_metadata_ids_match": True,
            "ground_truth_image_ids_match": True,
        },
        "outputs": {
            "manifest": manifest_path.relative_to(root).as_posix(),
            "class_distribution": distribution_path.relative_to(root).as_posix(),
            "audit": audit_path.relative_to(root).as_posix(),
        },
    }

    atomic_csv(manifest_path, MANIFEST_COLUMNS, manifest_rows)
    atomic_csv(
        distribution_path,
        (
            "diagnosis_original", "diagnosis_canonical", "count",
            "stage_1_label", "stage_2_label", "include_stage_1",
            "include_stage_2", "exclusion_reason",
        ),
        distribution_rows,
    )
    atomic_json(audit_path, audit)

    print("\n================================================")
    print("ISIC 2019 dataset manifest successfully created")
    print("================================================")
    print(f"Manifest rows:      {len(manifest_rows)}")
    print(f"Stage 1 eligible:   {sum(stage1_counts.values())}")
    print(f"Stage 1 excluded:   {sum(exclusion_counts.values())}")
    print(f"Stage 2 eligible:   {sum(stage2_counts.values())}")
    print(f"Manifest:           {manifest_path}")
    print(f"Class distribution: {distribution_path}")
    print(f"Manifest audit:     {audit_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
