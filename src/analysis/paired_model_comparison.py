"""Deterministic integrity and identity audit for paired prediction CSVs."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence


COMMON_CLASSES = ("non_malignant", "melanoma", "bcc", "scc")


class PairingAuditError(ValueError):
    """Raised when locked predictions cannot be paired safely."""


@dataclass(frozen=True)
class CsvInspection:
    """Structural inventory for one prediction CSV."""

    path: str
    size_bytes: int
    sha256: str
    row_count: int
    columns: tuple[str, ...]
    duplicate_columns: tuple[str, ...]
    empty_rows: int
    malformed_rows: int
    missing_value_counts: dict[str, int]


@dataclass(frozen=True)
class PredictionSchema:
    """Columns and index mapping needed to audit one model."""

    identifier: str
    target_label: str
    target_index: str
    predicted_label: str
    predicted_index: str
    class_to_index: Mapping[str, int]


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_label(value: str) -> str:
    """Normalize harmless spelling differences without changing semantics."""
    return "_".join(value.strip().lower().replace("-", " ").split())


def inspect_csv(path: Path, *, display_path: str | None = None) -> CsvInspection:
    """Inspect CSV structure without altering the source."""
    if not path.is_file():
        raise PairingAuditError(f"Prediction CSV does not exist: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            columns = tuple(next(reader))
        except StopIteration as exc:
            raise PairingAuditError(f"Prediction CSV is empty: {path}") from exc
        duplicate_columns = tuple(
            sorted(name for name, count in Counter(columns).items() if count > 1)
        )
        empty_rows = 0
        malformed_rows = 0
        row_count = 0
        missing = Counter({column: 0 for column in columns})
        for row in reader:
            if not row or all(not value.strip() for value in row):
                empty_rows += 1
                continue
            row_count += 1
            if len(row) != len(columns):
                malformed_rows += 1
                continue
            for column, value in zip(columns, row):
                if not value.strip():
                    missing[column] += 1
    return CsvInspection(
        path=display_path or path.as_posix(),
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
        row_count=row_count,
        columns=columns,
        duplicate_columns=duplicate_columns,
        empty_rows=empty_rows,
        malformed_rows=malformed_rows,
        missing_value_counts=dict(missing),
    )


def _read_rows(
    path: Path,
    schema: PredictionSchema,
    *,
    expected_count: int,
    model_name: str,
) -> tuple[dict[str, dict[str, str]], CsvInspection]:
    inspection = inspect_csv(path)
    if inspection.duplicate_columns:
        raise PairingAuditError(
            f"{model_name}: duplicate CSV columns: {inspection.duplicate_columns}"
        )
    if inspection.empty_rows or inspection.malformed_rows:
        raise PairingAuditError(
            f"{model_name}: found {inspection.empty_rows} empty and "
            f"{inspection.malformed_rows} malformed rows."
        )
    if inspection.row_count != expected_count:
        raise PairingAuditError(
            f"{model_name}: expected {expected_count} prediction rows, "
            f"observed {inspection.row_count}."
        )
    required = {
        schema.identifier,
        schema.target_label,
        schema.target_index,
        schema.predicted_label,
        schema.predicted_index,
    }
    absent = sorted(required.difference(inspection.columns))
    if absent:
        raise PairingAuditError(f"{model_name}: missing required columns: {absent}")

    rows: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            identifier = row[schema.identifier].strip()
            if not identifier:
                raise PairingAuditError(
                    f"{model_name}: missing sample identifier at CSV line "
                    f"{line_number}."
                )
            if identifier in rows:
                raise PairingAuditError(
                    f"{model_name}: duplicate sample identifier {identifier!r}."
                )
            for role, label_column, index_column in (
                ("target", schema.target_label, schema.target_index),
                ("prediction", schema.predicted_label, schema.predicted_index),
            ):
                label = normalize_label(row[label_column])
                if label not in schema.class_to_index:
                    raise PairingAuditError(
                        f"{model_name}: unsupported {role} label "
                        f"{row[label_column]!r} for sample {identifier!r}."
                    )
                try:
                    index = int(row[index_column])
                except ValueError as exc:
                    raise PairingAuditError(
                        f"{model_name}: invalid {role} index "
                        f"{row[index_column]!r} for sample {identifier!r}."
                    ) from exc
                expected_index = schema.class_to_index[label]
                if index != expected_index:
                    raise PairingAuditError(
                        f"{model_name}: {role} label/index disagreement for "
                        f"{identifier!r}: {label!r} maps to {expected_index}, "
                        f"not {index}."
                    )
                row[label_column] = label
            rows[identifier] = row
    return rows, inspection


def audit_prediction_pairing(
    hierarchical_path: Path,
    flat_path: Path,
    *,
    hierarchical_schema: PredictionSchema,
    flat_schema: PredictionSchema,
    expected_count: int = 3668,
) -> tuple[dict[str, object], list[dict[str, str]]]:
    """Validate and pair two prediction CSVs by sorted stable identifier."""
    expected_mapping = {name: index for index, name in enumerate(COMMON_CLASSES)}
    for model_name, schema in (
        ("hierarchical", hierarchical_schema),
        ("flat", flat_schema),
    ):
        if dict(schema.class_to_index) != expected_mapping:
            raise PairingAuditError(
                f"{model_name}: class mapping must be {expected_mapping}, "
                f"observed {dict(schema.class_to_index)}."
            )

    hierarchical, hierarchical_info = _read_rows(
        hierarchical_path,
        hierarchical_schema,
        expected_count=expected_count,
        model_name="hierarchical",
    )
    flat, flat_info = _read_rows(
        flat_path,
        flat_schema,
        expected_count=expected_count,
        model_name="flat",
    )
    hierarchical_ids = set(hierarchical)
    flat_ids = set(flat)
    only_hierarchical = sorted(hierarchical_ids - flat_ids)
    only_flat = sorted(flat_ids - hierarchical_ids)
    if only_hierarchical or only_flat:
        raise PairingAuditError(
            "Identifier sets differ: "
            f"{len(only_hierarchical)} only in hierarchical "
            f"(examples={only_hierarchical[:5]}), {len(only_flat)} only in "
            f"flat (examples={only_flat[:5]})."
        )

    paired: list[dict[str, str]] = []
    disagreements: list[str] = []
    for identifier in sorted(hierarchical_ids):
        hierarchical_row = hierarchical[identifier]
        flat_row = flat[identifier]
        target = hierarchical_row[hierarchical_schema.target_label]
        flat_target = flat_row[flat_schema.target_label]
        if target != flat_target:
            disagreements.append(
                f"{identifier}: hierarchical={target}, flat={flat_target}"
            )
            continue
        paired.append(
            {
                "sample_id": identifier,
                "target_label": target,
                "target_index": str(expected_mapping[target]),
                "hierarchical_predicted_label": hierarchical_row[
                    hierarchical_schema.predicted_label
                ],
                "hierarchical_predicted_index": hierarchical_row[
                    hierarchical_schema.predicted_index
                ],
                "flat_predicted_label": flat_row[flat_schema.predicted_label],
                "flat_predicted_index": flat_row[flat_schema.predicted_index],
            }
        )
    if disagreements:
        raise PairingAuditError(
            f"Ground-truth labels disagree for {len(disagreements)} samples "
            f"(examples={disagreements[:5]})."
        )

    support = Counter(row["target_label"] for row in paired)
    audit = {
        "status": "passed",
        "expected_sample_count": expected_count,
        "paired_sample_count": len(paired),
        "pairing_key": hierarchical_schema.identifier,
        "pairing_method": "exact identifier match followed by ascending lexical sort",
        "row_order_independent": True,
        "identifier_sets_identical": True,
        "ground_truth_labels_identical_after_normalization": True,
        "label_normalization": "strip, lowercase, hyphen/whitespace to underscore",
        "class_order": list(COMMON_CLASSES),
        "class_to_index": expected_mapping,
        "ground_truth_support": {
            name: support[name] for name in COMMON_CLASSES
        },
        "hierarchical_csv": asdict(hierarchical_info),
        "flat_csv": asdict(flat_info),
    }
    return audit, paired


def write_json(path: Path, payload: object) -> None:
    """Write stable, human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_rows(
    path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]
) -> None:
    """Write deterministic CSV rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
