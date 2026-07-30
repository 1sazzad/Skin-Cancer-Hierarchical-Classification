#!/usr/bin/env python3
"""VM-only official ISIC metadata acquisition and eligible image download."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.emb_stage03 import derive_t_category_from_isic_metadata, map_stage_ajcc

API_TEMPLATE = "https://api.isic-archive.com/api/v2/images/{image_id}/"
USER_AGENT = "Skin-Cancer-Hierarchical-Classification/Phase09 research metadata audit"
SUPPORTED_LICENCES = {"CC-0", "CC0", "CC-BY", "CC-BY-NC"}
ISIC_PATTERN = re.compile(r"^ISIC_[0-9]+$", re.IGNORECASE)
INVENTORY_FIELDS = [
    "image_id", "api_url", "full_image_url", "expected_file_size", "public",
    "copyright_license", "attribution", "modality", "diagnosis_2", "diagnosis_3",
    "diagnosis_confirm_type", "mel_thick_mm", "mel_ulcer", "patient_id",
    "lesion_id", "derived_stage_ajcc", "t_category", "original_emb_stage_ajcc",
    "original_emb_t_category", "original_vs_official_agreement", "eligible",
    "exclusion_reason", "image_path", "file_sha256", "download_status",
]


def require_vm() -> None:
    if os.name != "posix":
        raise SystemExit("NO-GO: official ISIC acquisition is Linux GPU VM-only.")
    gpu_present = Path("/proc/driver/nvidia").exists() or shutil.which("nvidia-smi")
    if not gpu_present:
        raise SystemExit("NO-GO: NVIDIA VM environment was not detected.")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", delete=False, dir=path.parent
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, records: dict[str, dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(records[key], sort_keys=True, ensure_ascii=False) + "\n"
        for key in sorted(records)
    )
    atomic_write_text(path, text)


def atomic_write_inventory(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", delete=False, dir=path.parent
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=INVENTORY_FIELDS)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in INVENTORY_FIELDS} for row in rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def nested(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def normalize_modality(value: object) -> str:
    raw = " ".join(str(value or "").strip().lower().replace("_", " ").split())
    return "dermoscopic" if raw in {"dermoscopic", "dermoscopy"} else raw


def is_histopathology(value: object) -> bool:
    raw = " ".join(str(value or "").strip().lower().replace("_", " ").split())
    return "histopath" in raw


def candidate_rows(source_csv: Path) -> list[dict[str, Any]]:
    try:
        frame = pd.read_csv(source_csv, dtype=str, keep_default_na=False)
    except Exception as exc:
        raise ValueError(f"Malformed candidate source CSV: {exc}") from exc
    required = {"source", "type", "image"}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"Candidate CSV is missing columns: {missing}")
    selected = frame.loc[
        frame["source"].str.strip().str.casefold().eq("isic")
        & frame["type"].str.strip().str.casefold().eq("dermoscopic")
        & frame["image"].str.strip().map(lambda value: bool(ISIC_PATTERN.fullmatch(value)))
    ].copy()
    selected["image"] = selected["image"].str.strip().str.upper()
    if selected["image"].duplicated().any():
        selected = selected.drop_duplicates("image", keep="first")
    return selected.to_dict("records")


def fetch_json(
    image_id: str, timeout: float, retries: int,
    opener: Any = urllib.request.urlopen,
) -> dict[str, Any]:
    url = API_TEMPLATE.format(image_id=image_id)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries + 1):
        try:
            with opener(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if not retryable or attempt == retries:
                raise
        except (TimeoutError, urllib.error.URLError, ConnectionError):
            if attempt == retries:
                raise
        time.sleep(min(2 ** attempt, 10))
    raise RuntimeError("unreachable")


def inventory_row(candidate: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    image_id = str(candidate["image"]).strip().upper()
    clinical = nested(payload, "metadata", "clinical") or {}
    modality = normalize_modality(nested(payload, "metadata", "acquisition", "image_type"))
    diagnosis_3 = clinical.get("diagnosis_3")
    thickness = clinical.get("mel_thick_mm")
    returned_id = str(payload.get("isic_id") or "").strip().upper()
    licence = str(payload.get("copyright_license") or "").strip()
    attribution = str(payload.get("attribution") or "").strip()
    full_url = str(nested(payload, "files", "full", "url") or "").strip()
    reasons: list[str] = []
    is_public = payload.get("public") is True or truthy(payload.get("public"))
    if not is_public:
        reasons.append("not_public")
    if returned_id != image_id:
        reasons.append("isic_id_mismatch")
    if modality != "dermoscopic":
        reasons.append("not_dermoscopic")
    if not is_histopathology(clinical.get("diagnosis_confirm_type")):
        reasons.append("not_histopathology")
    if not full_url:
        reasons.append("missing_full_image_url")
    if licence not in SUPPORTED_LICENCES:
        reasons.append("unsupported_or_missing_licence")
    if not attribution:
        reasons.append("missing_attribution")
    derived_stage: int | str = ""
    t_category = ""
    try:
        derived_stage, t_category = derive_t_category_from_isic_metadata(
            diagnosis_3, thickness
        )
    except ValueError:
        reasons.append("no_official_t_category")
    original_stage = str(candidate.get("stage_ajcc", "")).strip()
    try:
        original_t = map_stage_ajcc(original_stage) if original_stage else ""
    except ValueError:
        original_t = ""
    agreement = (
        str(original_t == t_category).lower() if original_t and t_category else ""
    )
    return {
        "image_id": image_id,
        "api_url": API_TEMPLATE.format(image_id=image_id),
        "full_image_url": full_url,
        "expected_file_size": nested(payload, "files", "full", "size") or "",
        "public": str(is_public).lower(),
        "copyright_license": licence,
        "attribution": attribution,
        "modality": modality,
        "diagnosis_2": clinical.get("diagnosis_2") or "",
        "diagnosis_3": diagnosis_3 or "",
        "diagnosis_confirm_type": clinical.get("diagnosis_confirm_type") or "",
        "mel_thick_mm": thickness if thickness is not None else "",
        "mel_ulcer": clinical.get("mel_ulcer") or "",
        "patient_id": clinical.get("patient_id") or "",
        "lesion_id": clinical.get("lesion_id") or "",
        "derived_stage_ajcc": derived_stage,
        "t_category": t_category,
        "original_emb_stage_ajcc": original_stage,
        "original_emb_t_category": original_t,
        "original_vs_official_agreement": agreement,
        "eligible": str(not reasons).lower(),
        "exclusion_reason": "|".join(reasons),
        "image_path": "",
        "file_sha256": "",
        "download_status": "not_requested",
    }


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            record = json.loads(line)
            records[str(record["requested_isic_id"])] = record
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError(f"Corrupted JSONL record at line {number}.") from exc
    return records


def metadata_audit(
    candidates: list[dict[str, Any]], records: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]], isic_manifest: Path,
) -> dict[str, Any]:
    eligible = [row for row in rows if truthy(row["eligible"])]
    old_ids: set[str] = set()
    if isic_manifest.is_file():
        old = pd.read_csv(isic_manifest, dtype=str, keep_default_na=False)
        old_ids = set(old.get("image_id", pd.Series(dtype=str)).str.strip())
    overlap = sorted({row["image_id"] for row in eligible} & old_ids)
    overlap_set = set(overlap)
    official_counts = Counter(row["t_category"] for row in eligible)
    return {
        "dataset_name": "ISIC-derived melanoma T-category subset",
        "api_provenance": "https://api.isic-archive.com/api/v2/images/{ISIC_ID}/",
        "candidate_count": len(candidates),
        "successful_api_response_count": sum("payload" in item for item in records.values()),
        "failed_api_response_count": sum("error" in item for item in records.values()),
        "eligible_count": len(eligible),
        "excluded_count": len(rows) - len(eligible),
        "exclusion_reasons": dict(Counter(
            reason for row in rows for reason in str(row["exclusion_reason"]).split("|") if reason
        )),
        "licence_counts": dict(Counter(row["copyright_license"] or "<missing>" for row in rows)),
        "attribution_counts": dict(Counter(row["attribution"] or "<missing>" for row in rows)),
        "official_modality_counts": dict(Counter(row["modality"] for row in rows)),
        "official_diagnosis_counts": dict(Counter(row["diagnosis_3"] for row in rows)),
        "official_t_category_counts": dict(official_counts),
        "original_emb_t_category_counts": dict(Counter(row["original_emb_t_category"] for row in rows)),
        "original_vs_official_agreement_count": sum(
            row["original_vs_official_agreement"] == "true" for row in rows
        ),
        "disagreement_image_ids": [
            row["image_id"] for row in rows if row["original_vs_official_agreement"] == "false"
        ],
        "patient_id_available_count": sum(bool(str(row["patient_id"]).strip()) for row in rows),
        "lesion_id_available_count": sum(bool(str(row["lesion_id"]).strip()) for row in rows),
        "unique_patient_count": len({row["patient_id"] for row in rows if row["patient_id"]}),
        "unique_lesion_count": len({row["lesion_id"] for row in rows if row["lesion_id"]}),
        "isic2019_id_overlap_count": len(overlap),
        "isic2019_overlap_ids": overlap,
        "overlap_counts_by_official_t_category": dict(Counter(
            row["t_category"] for row in eligible if row["image_id"] in overlap_set
        )),
        "non_overlap_counts_by_official_t_category": dict(Counter(
            row["t_category"] for row in eligible if row["image_id"] not in overlap_set
        )),
    }


def run_metadata(args: argparse.Namespace) -> int:
    root = args.project_root.resolve()
    raw = root / "data/raw/emb"
    source = raw / "early_melanoma_benchmark_dataset_labels.csv"
    jsonl = raw / "isic_stage03_official_metadata.jsonl"
    inventory = raw / "isic_stage03_official_inventory.csv"
    report = root / "reports/dataset_audits/isic_stage03_metadata_audit.json"
    candidates = candidate_rows(source)
    records = load_jsonl(jsonl) if args.resume else {}
    pending = [
        row for row in candidates
        if row["image"] not in records or "payload" not in records[row["image"]]
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_json, row["image"], args.timeout, args.retries): row
            for row in pending
        }
        for future in as_completed(futures):
            candidate = futures[future]
            image_id = candidate["image"]
            try:
                record = {"requested_isic_id": image_id, "payload": future.result()}
            except Exception as exc:
                record = {"requested_isic_id": image_id, "error": f"{type(exc).__name__}: {exc}"}
            records[image_id] = record
            atomic_write_jsonl(jsonl, records)
    rows = [
        inventory_row(candidate, records[candidate["image"]]["payload"])
        for candidate in candidates
        if candidate["image"] in records and "payload" in records[candidate["image"]]
    ]
    if args.resume and inventory.is_file():
        previous = {
            row["image_id"]: row
            for row in pd.read_csv(
                inventory, dtype=str, keep_default_na=False
            ).to_dict("records")
        }
        for row in rows:
            old = previous.get(row["image_id"], {})
            for field in ("image_path", "file_sha256", "download_status"):
                if old.get(field):
                    row[field] = old[field]
    atomic_write_inventory(inventory, rows)
    audit = metadata_audit(
        candidates, records, rows, root / "data/manifests/isic2019_dataset_manifest.csv"
    )
    atomic_write_text(report, json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if audit["successful_api_response_count"] == 0 or audit["eligible_count"] == 0:
        return 2
    print(f"Metadata inventory: {inventory}")
    print(f"Metadata audit: {report}")
    return 0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_image(path: Path) -> None:
    with Image.open(path) as image:
        image.verify()


def sanitized_exception_message(exc: Exception, limit: int = 200) -> str:
    message = " ".join(str(exc).split())
    message = re.sub(r"(https?://[^\s?]+)\?[^\s]+", r"\1?[redacted]", message)
    message = re.sub(
        r"(?i)\b(token|api[_-]?key|authorization|password|secret)=([^&\s]+)",
        r"\1=[redacted]",
        message,
    )
    return (message or "no details")[:limit]


def download_one(row: dict[str, Any], destination: Path, timeout: float, retries: int) -> tuple[str, str, str]:
    image_id = str(row["image_id"])
    suffix = Path(urlparse(str(row["full_image_url"])).path).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        suffix = ".jpg"
    final = destination / f"{image_id}{suffix}"
    expected = int(float(row["expected_file_size"])) if str(row["expected_file_size"]).strip() else None
    if final.is_file():
        try:
            verify_image(final)
            actual = final.stat().st_size
            status = (
                "existing_valid_size_mismatch"
                if expected is not None and actual != expected
                else "existing_valid"
            )
            return str(final), file_sha256(final), status
        except (UnidentifiedImageError, OSError):
            pass
    part = final.with_suffix(final.suffix + ".part")
    if part.exists():
        part.unlink()
    request = urllib.request.Request(str(row["full_image_url"]), headers={"User-Agent": USER_AGENT})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response, part.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            actual = part.stat().st_size
            verify_image(part)
            os.replace(part, final)
            status = (
                "downloaded_size_mismatch"
                if expected is not None and actual != expected
                else "downloaded"
            )
            return str(final), file_sha256(final), status
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
            if part.exists():
                part.unlink()
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 10))
    raise RuntimeError("unreachable")


def run_download(args: argparse.Namespace) -> int:
    root = args.project_root.resolve()
    inventory = root / "data/raw/emb/isic_stage03_official_inventory.csv"
    if not inventory.is_file():
        raise FileNotFoundError("Run --metadata-only first.")
    rows = pd.read_csv(inventory, dtype=str, keep_default_na=False).to_dict("records")
    eligible = [row for row in rows if truthy(row["eligible"])]
    destination = root / "data/raw/emb/images/isic"
    destination.mkdir(parents=True, exist_ok=True)
    pending: list[dict[str, Any]] = []
    for row in eligible:
        if args.resume and str(row["download_status"]).startswith(
            ("downloaded", "existing_valid")
        ):
            path = root / row["image_path"] if row["image_path"] else Path()
            if path.is_file():
                try:
                    verify_image(path)
                    continue
                except (UnidentifiedImageError, OSError):
                    pass
        pending.append(row)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                download_one, row, destination, args.timeout, args.retries
            ): row
            for row in pending
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                path, digest, status = future.result()
                absolute = Path(path)
                row["image_path"] = (
                    absolute.relative_to(root).as_posix()
                    if absolute.is_relative_to(root) else str(absolute)
                )
                row["file_sha256"] = digest
                row["download_status"] = status
            except Exception as exc:
                row["download_status"] = (
                    f"failed:{type(exc).__name__}: {sanitized_exception_message(exc)}"
                )
            atomic_write_inventory(inventory, rows)
    failures = [row["image_id"] for row in eligible if str(row["download_status"]).startswith("failed")]
    mismatches = [
        row["image_id"]
        for row in eligible
        if str(row["download_status"]).endswith("size_mismatch")
    ]
    print(
        f"Eligible downloads: {len(eligible)}; failures: {len(failures)}; "
        f"size mismatches: {len(mismatches)}"
    )
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--metadata-only", action="store_true")
    modes.add_argument("--download-images", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("--workers must be between 1 and 4")
    if args.timeout <= 0 or args.retries < 0:
        parser.error("--timeout must be positive and --retries non-negative")
    return args


def main() -> int:
    args = parse_args()
    require_vm()
    return run_metadata(args) if args.metadata_only else run_download(args)


if __name__ == "__main__":
    raise SystemExit(main())
