"""Inventory preserved HIBA API metadata without networking or label mapping."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_IMAGE_COUNT = 1616
DEFAULT_ROOT = Path("data/external/hiba")
STATUS = "metadata_inventory_pending_human_review"
FIELD_PATHS: dict[str, tuple[str, ...]] = {
    "image_id": ("isic_id", "isicId", "name", "_id", "id"),
    "patient_id": (
        "metadata.clinical.patient_id",
        "metadata.clinical.patientId",
        "patient_id",
        "patientId",
    ),
    "lesion_id": (
        "metadata.clinical.lesion_id",
        "metadata.clinical.lesionId",
        "lesion_id",
        "lesionId",
    ),
    "diagnosis": (
        "metadata.clinical.diagnosis",
        "metadata.diagnosis",
        "diagnosis",
    ),
    "diagnosis_confirmation": (
        "metadata.clinical.diagnosis_confirm_type",
        "metadata.clinical.diagnosisConfirmation",
        "diagnosis_confirmation",
    ),
    "modality": (
        "metadata.acquisition.image_type",
        "metadata.acquisition.modality",
        "modality",
    ),
    "license": (
        "metadata.acquisition.license",
        "metadata.license",
        "license",
    ),
    "attribution": (
        "metadata.acquisition.attribution",
        "metadata.attribution",
        "attribution",
    ),
    "file_type": (
        "metadata.acquisition.mime_type",
        "metadata.acquisition.file_extension",
        "mimeType",
        "extension",
    ),
    "public_status": (
        "public",
        "isPublic",
        "metadata.public",
        "metadata.access.public",
    ),
}
CSV_COLUMNS = tuple(
    item
    for field in FIELD_PATHS
    for item in (field, f"{field}_source_path")
)


class InventoryError(RuntimeError):
    """Raised when preserved metadata cannot be inventoried safely."""


def _nested(payload: Mapping[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def extract_exact(
    payload: Mapping[str, Any], paths: Sequence[str]
) -> tuple[Any, str]:
    for path in paths:
        value = _nested(payload, path)
        if value is not None:
            return value, path
    return None, ""


def read_raw_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise InventoryError(f"Blank JSONL line at {line_number}.")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InventoryError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise InventoryError(
                    f"JSONL line {line_number} is not an object."
                )
            records.append(record)
    return records


def _display(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def build_inventory(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if len(records) != EXPECTED_IMAGE_COUNT:
        raise InventoryError(
            f"Raw record count must be exactly {EXPECTED_IMAGE_COUNT}; "
            f"got {len(records)}."
        )
    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    source_paths: dict[str, Counter[str]] = {
        field: Counter() for field in FIELD_PATHS
    }
    vocabularies: dict[str, Counter[str]] = {
        field: Counter()
        for field in (
            "diagnosis",
            "diagnosis_confirmation",
            "modality",
            "license",
            "file_type",
            "public_status",
        )
    }
    presence = {
        field: {"present": 0, "missing": 0}
        for field in ("patient_id", "lesion_id", "attribution")
    }

    for index, record in enumerate(records, start=1):
        row: dict[str, str] = {}
        for field, candidates in FIELD_PATHS.items():
            value, source_path = extract_exact(record, candidates)
            row[field] = _display(value)
            row[f"{field}_source_path"] = source_path
            source_paths[field][source_path or "<missing>"] += 1
            if field in vocabularies:
                vocabularies[field][_display(value) or "<missing>"] += 1
            if field in presence:
                key = "missing" if value is None or _display(value) == "" else "present"
                presence[field][key] += 1
        identifier = row["image_id"]
        if not identifier:
            raise InventoryError(f"Record {index} has no image ID.")
        if identifier in seen_ids:
            raise InventoryError(f"Duplicate image ID in raw JSONL: {identifier}")
        seen_ids.add(identifier)
        rows.append(row)

    inventory = {
        "dataset": "hiba",
        "collection_id": 251,
        "doi": "10.34970/587329",
        "status": STATUS,
        "record_count": len(records),
        "unique_image_id_count": len(seen_ids),
        "duplicate_image_id_count": 0,
        "patient_id": presence["patient_id"],
        "lesion_id": presence["lesion_id"],
        "attribution": presence["attribution"],
        "exact_vocabularies": {
            field: dict(sorted(counts.items()))
            for field, counts in vocabularies.items()
        },
        "source_paths": {
            field: dict(sorted(counts.items()))
            for field, counts in source_paths.items()
        },
        "unknown_and_missing_values_preserved": True,
        "diagnosis_substring_inference_performed": False,
        "label_mapping_performed": False,
        "dermoscopic_cohort_selected": False,
        "evaluation_approval_assigned": False,
        "split_created": False,
        "network_access_performed": False,
    }
    return rows, inventory


def _ensure_under_root(path: Path, root: Path) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise InventoryError(f"Path escapes HIBA root: {path}")
    return resolved


def _write_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="",
    )


def publish_inventory(
    root: Path,
    rows: Sequence[Mapping[str, str]],
    inventory: Mapping[str, Any],
) -> tuple[Path, Path]:
    root = root.resolve()
    json_path = root / "metadata" / "collection_251_metadata_inventory.json"
    csv_path = root / "metadata" / "collection_251_metadata_inventory.csv"
    for path in (json_path, csv_path):
        _ensure_under_root(path, root)
    existing = [str(path) for path in (json_path, csv_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite finalized inventory file(s): "
            + ", ".join(existing)
        )

    temporary: list[Path] = []
    published: list[Path] = []
    try:
        for final_path in (json_path, csv_path):
            final_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(
                prefix=f".{final_path.name}.", suffix=".tmp", dir=final_path.parent
            )
            os.close(descriptor)
            temporary.append(Path(name))
        _write_json(temporary[0], inventory)
        _write_csv(temporary[1], rows)
        if json_path.exists() or csv_path.exists():
            raise FileExistsError("Inventory output appeared during publication.")
        for temporary_path, final_path in zip(temporary, (json_path, csv_path)):
            os.rename(temporary_path, final_path)
            published.append(final_path)
    except BaseException:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
    return json_path, csv_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--raw-jsonl",
        type=Path,
        default=DEFAULT_ROOT / "metadata" / "collection_251_images.raw.jsonl",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    if root != DEFAULT_ROOT.resolve():
        raise InventoryError(
            "Inventory output root is fixed at data/external/hiba."
        )
    raw_path = _ensure_under_root(args.raw_jsonl, root)
    records = read_raw_jsonl(raw_path)
    rows, inventory = build_inventory(records)
    if args.dry_run:
        print(json.dumps(inventory, indent=2, sort_keys=True))
        return 0
    publish_inventory(root, rows, inventory)
    print(f"Inventoried {len(rows)} raw records; human review remains required.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (InventoryError, FileExistsError, OSError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
