"""Build and audit a fail-closed HIBA external-evaluation manifest.

This module performs metadata, integrity, modality, label, and ISIC-overlap
checks only. It never loads models or modifies raw dataset files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


RELEASE_ID = "ISIC collection 251; DOI 10.34970/587329"
FINAL_LABELS = {"non_malignant", "melanoma", "bcc", "scc"}
MANIFEST_COLUMNS = (
    "dataset", "release_id", "image_id", "image_path", "patient_id",
    "lesion_id", "original_diagnosis", "canonical_diagnosis", "modality",
    "mapped_final_label", "mapped_stage_1_label", "mapped_stage_2_label",
    "include_primary_evaluation", "exclusion_reason", "attribution",
    "license", "source_reference", "file_size_bytes", "file_sha256",
)
REQUIRED_METADATA_COLUMNS = {
    "image_id", "image_path", "diagnosis", "modality", "attribution",
    "license", "source_reference",
}
DERMOSCOPIC_MODALITIES = {"dermoscopic", "dermoscopy"}
CLINICAL_MODALITIES = {"clinical", "clinical smartphone", "smartphone clinical"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MAPPING_KEYS = {
    "canonical_diagnosis",
    "mapped_final_label",
    "mapped_stage_1_label",
    "mapped_stage_2_label",
    "include_primary_evaluation",
    "exclusion_reason",
}


class AuditError(ValueError):
    """Raised when an audit input is structurally unsafe."""


def normalize_term(value: str) -> str:
    """Normalize an explicit vocabulary term without semantic inference."""
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    return " ".join(normalized.split())


def stream_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a file SHA-256 using bounded-memory reads."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_mapping_entry(name: str, entry: Mapping[str, Any]) -> None:
    missing = sorted(MAPPING_KEYS - set(entry))
    extra = sorted(set(entry) - MAPPING_KEYS)
    if missing or extra:
        details = []
        if missing:
            details.append("missing keys: " + ", ".join(missing))
        if extra:
            details.append("unexpected keys: " + ", ".join(extra))
        raise AuditError(f"Mapping entry {name!r} has " + "; ".join(details))
    include = entry["include_primary_evaluation"]
    if type(include) is not bool:
        raise AuditError(
            f"Mapping entry {name!r} include_primary_evaluation must be boolean."
        )
    values = {
        key: entry[key]
        for key in MAPPING_KEYS - {"include_primary_evaluation"}
    }
    if any(not isinstance(value, str) for value in values.values()):
        raise AuditError(f"Mapping entry {name!r} label fields must be strings.")

    final = entry["mapped_final_label"].strip()
    stage_1 = entry["mapped_stage_1_label"].strip()
    stage_2 = entry["mapped_stage_2_label"].strip()
    reason = entry["exclusion_reason"].strip()
    if not entry["canonical_diagnosis"].strip():
        raise AuditError(
            f"Mapping entry {name!r} requires a canonical diagnosis."
        )
    if include:
        if final not in FINAL_LABELS:
            raise AuditError(f"Mapping entry {name!r} has invalid final label.")
        if reason:
            raise AuditError(
                f"Included mapping entry {name!r} must have blank exclusion reason."
            )
        if final == "non_malignant":
            if stage_1 != "non_malignant" or stage_2:
                raise AuditError(
                    f"Included non-malignant mapping entry {name!r} has "
                    "inconsistent stage labels."
                )
        elif stage_1 != "malignant" or stage_2 != final:
            raise AuditError(
                f"Included malignant mapping entry {name!r} has inconsistent "
                "stage labels."
            )
    else:
        if not reason:
            raise AuditError(
                f"Excluded mapping entry {name!r} requires an exclusion reason."
            )
        if final or stage_1 or stage_2:
            raise AuditError(
                f"Excluded mapping entry {name!r} must have empty mapped labels."
            )


def load_mapping(path: Path) -> tuple[dict[str, dict[str, Any]], set[str]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or not isinstance(payload.get("mappings"), dict):
        raise AuditError(f"Mapping YAML has no mappings object: {path}")
    licence_policy = payload.get("licence_policy")
    if not isinstance(licence_policy, dict):
        raise AuditError(f"Mapping YAML has no licence_policy object: {path}")
    accepted_values = licence_policy.get("accepted_values")
    if not isinstance(accepted_values, list) or not accepted_values:
        raise AuditError("licence_policy.accepted_values must be a non-empty list.")
    if any(not isinstance(value, str) or not value.strip() for value in accepted_values):
        raise AuditError("Every accepted licence value must be a non-empty string.")
    accepted_licences = {normalize_term(value) for value in accepted_values}
    if len(accepted_licences) != len(accepted_values):
        raise AuditError("Accepted licence values collide after normalization.")

    mappings: dict[str, dict[str, Any]] = {}
    for raw_key, raw_value in payload["mappings"].items():
        key = normalize_term(str(raw_key))
        if not key:
            raise AuditError("Mapping keys must be non-empty strings.")
        if key in mappings:
            raise AuditError(f"Duplicate normalized mapping key: {raw_key!r}")
        if not isinstance(raw_value, dict):
            raise AuditError(f"Mapping entry {raw_key!r} must be an object.")
        _validate_mapping_entry(str(raw_key), raw_value)
        mappings[key] = dict(raw_value)
    return mappings, accepted_licences


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise AuditError(f"CSV has no header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def _serialize_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _serialize_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="")


def publish_outputs_transactionally(
    manifest_path: Path,
    audit_path: Path,
    checksum_path: Path,
    manifest: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
    checksum_lines: Sequence[str],
) -> None:
    """Publish all evidence outputs together or leave none of them."""
    final_paths = (manifest_path, audit_path, checksum_path)
    existing = [str(path) for path in final_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing HIBA audit output(s): "
            + ", ".join(existing)
        )

    temporary_paths: list[Path] = []
    published_paths: list[Path] = []
    try:
        for final_path in final_paths:
            final_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{final_path.name}.",
                suffix=".tmp",
                dir=final_path.parent,
            )
            os.close(descriptor)
            temporary_paths.append(Path(temporary_name))

        _serialize_csv(temporary_paths[0], MANIFEST_COLUMNS, manifest)
        _serialize_text(
            temporary_paths[1],
            json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n",
        )
        _serialize_text(
            temporary_paths[2],
            "\n".join(checksum_lines) + ("\n" if checksum_lines else ""),
        )

        race_existing = [str(path) for path in final_paths if path.exists()]
        if race_existing:
            raise FileExistsError(
                "HIBA audit output appeared during publication; refusing "
                "overwrite: " + ", ".join(race_existing)
            )
        for temporary_path, final_path in zip(temporary_paths, final_paths):
            os.rename(temporary_path, final_path)
            published_paths.append(final_path)
    except BaseException:
        for published_path in published_paths:
            published_path.unlink(missing_ok=True)
        raise
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)


def _safe_image_path(raw_root: Path, value: str) -> Path:
    candidate = Path(value)
    resolved_root = raw_root.resolve()
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (resolved_root / candidate).resolve()
    )
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise AuditError(f"Image path escapes the HIBA raw root: {value!r}")
    return resolved


def _relative_display(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _isic_index(path: Path) -> tuple[set[str], set[str]]:
    if not path.is_file():
        raise AuditError(f"Existing ISIC 2019 manifest is missing: {path}")
    headers, rows = _read_csv(path)
    if "image_id" not in headers or "file_sha256" not in headers:
        raise AuditError(
            "ISIC 2019 manifest must contain image_id and file_sha256."
        )
    ids = {row["image_id"].strip() for row in rows if row["image_id"].strip()}
    hashes = {
        row["file_sha256"].strip().lower()
        for row in rows
        if SHA256_RE.fullmatch(row["file_sha256"].strip().lower())
    }
    return ids, hashes


def audit_dataset(
    metadata_path: Path,
    raw_root: Path,
    mapping_path: Path,
    isic_manifest_path: Path,
    *,
    project_root: Path,
    release_id: str = RELEASE_ID,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    """Audit HIBA inputs and return manifest rows, report, and checksum lines."""
    headers, metadata_rows = _read_csv(metadata_path)
    missing_columns = sorted(REQUIRED_METADATA_COLUMNS - set(headers))
    if missing_columns:
        raise AuditError(
            "HIBA metadata is missing required columns: "
            + ", ".join(missing_columns)
        )
    mappings, accepted_licences = load_mapping(mapping_path)
    isic_ids, isic_hashes = _isic_index(isic_manifest_path)

    manifest: list[dict[str, Any]] = []
    blockers: list[str] = []
    seen_ids: set[str] = set()
    hash_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    lesion_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    overlap_ids: list[str] = []
    overlap_hashes: list[str] = []
    unresolved: set[str] = set()

    for number, source in enumerate(metadata_rows, start=2):
        image_id = source["image_id"].strip()
        if not image_id:
            raise AuditError(f"Empty image_id at metadata row {number}.")
        if image_id in seen_ids:
            raise AuditError(f"Duplicate image_id {image_id!r} at row {number}.")
        seen_ids.add(image_id)

        original = source["diagnosis"]
        mapping = mappings.get(normalize_term(original))
        if mapping is None:
            unresolved.add(original)
            mapping = {
                "canonical_diagnosis": "",
                "mapped_final_label": "",
                "mapped_stage_1_label": "",
                "mapped_stage_2_label": "",
                "include_primary_evaluation": False,
                "exclusion_reason": "unresolved_diagnosis_mapping",
            }

        modality = source["modality"].strip()
        modality_key = normalize_term(modality)
        include = bool(mapping.get("include_primary_evaluation", False))
        reason = str(mapping.get("exclusion_reason", "")).strip()
        if modality_key not in DERMOSCOPIC_MODALITIES:
            include = False
            if modality_key in CLINICAL_MODALITIES:
                reason = "excluded_non_dermoscopic_clinical"
            elif not modality_key:
                reason = "excluded_missing_modality"
            else:
                reason = "excluded_unsupported_or_ambiguous_modality"

        attribution = source["attribution"].strip()
        license_value = source["license"].strip()
        normalized_licence = normalize_term(license_value)
        source_reference = source["source_reference"].strip()
        if not license_value:
            include = False
            reason = "excluded_missing_licence"
        elif normalized_licence not in accepted_licences:
            include = False
            reason = "excluded_unsupported_or_unknown_licence"
        if not attribution or not source_reference:
            include = False
            reason = "excluded_missing_attribution_or_source"

        image_path = _safe_image_path(raw_root, source["image_path"].strip())
        if not image_path.is_file():
            raise AuditError(f"Image file is missing for {image_id}: {image_path}")
        size = image_path.stat().st_size
        if size <= 0:
            raise AuditError(f"Image file is empty for {image_id}: {image_path}")
        sha256 = stream_sha256(image_path)

        final_label = str(mapping.get("mapped_final_label", "")).strip()
        if include and final_label not in FINAL_LABELS:
            include = False
            reason = "excluded_invalid_primary_label"

        row: dict[str, Any] = {
            "dataset": "hiba",
            "release_id": release_id,
            "image_id": image_id,
            "image_path": _relative_display(image_path, project_root),
            "patient_id": source.get("patient_id", "").strip(),
            "lesion_id": source.get("lesion_id", "").strip(),
            "original_diagnosis": original,
            "canonical_diagnosis": mapping.get("canonical_diagnosis", ""),
            "modality": modality,
            "mapped_final_label": final_label,
            "mapped_stage_1_label": mapping.get("mapped_stage_1_label", ""),
            "mapped_stage_2_label": mapping.get("mapped_stage_2_label", ""),
            "include_primary_evaluation": str(include).lower(),
            "exclusion_reason": "" if include else reason,
            "attribution": attribution,
            "license": license_value,
            "source_reference": source_reference,
            "file_size_bytes": size,
            "file_sha256": sha256,
        }
        if row["include_primary_evaluation"] == "false" and not str(
            row["exclusion_reason"]
        ).strip():
            raise AuditError(
                f"Excluded row {image_id!r} has no exclusion reason."
            )
        manifest.append(row)
        hash_rows[sha256].append(row)
        if row["lesion_id"]:
            lesion_rows[str(row["lesion_id"])].append(row)
        if image_id in isic_ids:
            overlap_ids.append(image_id)
        if sha256 in isic_hashes:
            overlap_hashes.append(sha256)

    duplicate_hash_groups = {
        digest: [str(row["image_id"]) for row in rows]
        for digest, rows in hash_rows.items() if len(rows) > 1
    }

    def conflicts(
        groups: Mapping[str, list[dict[str, Any]]],
        column: str,
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for group_id, rows in groups.items():
            labels = {str(row[column]) for row in rows}
            if len(labels) > 1:
                output.append({
                    "group": group_id,
                    "image_ids": [row["image_id"] for row in rows],
                    "values": sorted(labels),
                })
        return output

    hash_original_conflicts = conflicts(hash_rows, "original_diagnosis")
    hash_canonical_conflicts = conflicts(hash_rows, "canonical_diagnosis")
    hash_final_conflicts = conflicts(hash_rows, "mapped_final_label")
    lesion_original_conflicts = conflicts(lesion_rows, "original_diagnosis")
    lesion_canonical_conflicts = conflicts(lesion_rows, "canonical_diagnosis")
    lesion_final_conflicts = conflicts(lesion_rows, "mapped_final_label")
    hash_licence_conflicts = conflicts(hash_rows, "license")
    lesion_licence_conflicts = conflicts(lesion_rows, "license")
    if unresolved:
        blockers.append("unresolved_diagnosis_mappings")
    if duplicate_hash_groups:
        blockers.append("exact_duplicate_hashes_require_resolution")
    if any((
        hash_original_conflicts,
        hash_canonical_conflicts,
        hash_final_conflicts,
    )):
        blockers.append("conflicting_labels_within_identical_hash")
    if any((
        lesion_original_conflicts,
        lesion_canonical_conflicts,
        lesion_final_conflicts,
    )):
        blockers.append("conflicting_labels_within_same_lesion")
    if hash_licence_conflicts or lesion_licence_conflicts:
        blockers.append("conflicting_licence_values")
    if overlap_ids or overlap_hashes:
        blockers.append("isic2019_overlap_requires_documented_resolution")
    licence_exclusions = {
        "excluded_missing_licence",
        "excluded_unsupported_or_unknown_licence",
        "excluded_missing_attribution_or_source",
    }
    if any(row["exclusion_reason"] in licence_exclusions for row in manifest):
        blockers.append("licence_attribution_or_source_gate_failed")

    approved_support = {
        label: sum(
            row["include_primary_evaluation"] == "true"
            and row["mapped_final_label"] == label
            for row in manifest
        )
        for label in sorted(FINAL_LABELS)
    }
    approved_count = sum(approved_support.values())
    absent_classes = [
        label for label, support in approved_support.items() if support == 0
    ]
    if approved_count == 0:
        blockers.append("zero_approved_primary_rows")
    if absent_classes:
        blockers.append("absent_mapped_classes_require_human_feasibility_review")

    def count(column: str) -> dict[str, int]:
        return dict(sorted(Counter(str(row[column]) for row in manifest).items()))

    audit: dict[str, Any] = {
        "dataset": "hiba",
        "release_id": release_id,
        "status": "candidate_pending_official_acquisition_audit",
        "approval": {
            "automated_gates_passed": not blockers,
            "approved_for_primary_evaluation": False,
            "human_official_metadata_and_licence_review_required": True,
            "inference_authorized": False,
            "blockers": sorted(set(blockers)),
        },
        "counts": {
            "metadata_rows": len(metadata_rows),
            "manifest_rows": len(manifest),
            "approved_primary_rows": sum(
                row["include_primary_evaluation"] == "true" for row in manifest
            ),
            "by_original_diagnosis": count("original_diagnosis"),
            "by_canonical_diagnosis": count("canonical_diagnosis"),
            "by_mapped_class": count("mapped_final_label"),
            "by_modality": count("modality"),
            "by_license": count("license"),
            "by_inclusion": count("include_primary_evaluation"),
            "by_exclusion_reason": count("exclusion_reason"),
        },
        "unresolved_original_diagnoses": sorted(unresolved),
        "class_support_gate": {
            "approved_primary_support": approved_support,
            "absent_classes": absent_classes,
            "zero_approved_rows_blocks_approval": True,
            "absent_class_requires_human_feasibility_review": True,
            "numeric_minimum_support_threshold": None,
        },
        "licence_gate": {
            "accepted_normalized_values": sorted(accepted_licences),
            "original_values_preserved": True,
            "unknown_unsupported_conflicting_or_missing_blocks_approval": True,
            "identical_hash_licence_conflicts": hash_licence_conflicts,
            "same_lesion_licence_conflicts": lesion_licence_conflicts,
        },
        "integrity": {
            "duplicate_image_ids": 0,
            "exact_duplicate_hash_group_count": len(duplicate_hash_groups),
            "exact_duplicate_hash_groups": duplicate_hash_groups,
            "identical_hash_original_diagnosis_conflicts": (
                hash_original_conflicts
            ),
            "identical_hash_canonical_diagnosis_conflicts": (
                hash_canonical_conflicts
            ),
            "identical_hash_final_label_conflicts": hash_final_conflicts,
            "same_lesion_original_diagnosis_conflicts": (
                lesion_original_conflicts
            ),
            "same_lesion_canonical_diagnosis_conflicts": (
                lesion_canonical_conflicts
            ),
            "same_lesion_final_label_conflicts": lesion_final_conflicts,
        },
        "isic2019_overlap": {
            "image_id_overlap_count": len(set(overlap_ids)),
            "image_ids": sorted(set(overlap_ids)),
            "sha256_overlap_count": len(set(overlap_hashes)),
            "sha256_values": sorted(set(overlap_hashes)),
            "approval_blocked_when_nonzero": True,
        },
        "cohort": {
            "split_created": False,
            "policy": "single_external_evaluation_cohort",
        },
    }
    checksum_lines = [
        f"{row['file_sha256']}  {row['image_path']}" for row in manifest
    ]
    return manifest, audit, checksum_lines


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=Path("data/external/hiba"))
    parser.add_argument(
        "--mapping", type=Path,
        default=Path("configs/datasets/hiba_external_label_mapping.yaml"),
    )
    parser.add_argument(
        "--isic-manifest", type=Path,
        default=Path("data/manifests/isic2019_dataset_manifest.csv"),
    )
    parser.add_argument(
        "--manifest-output", type=Path,
        default=Path("data/manifests/hiba_dataset_manifest.csv"),
    )
    parser.add_argument(
        "--audit-output", type=Path,
        default=Path("data/manifests/hiba_dataset_manifest.audit.json"),
    )
    parser.add_argument(
        "--checksum-output", type=Path,
        default=Path("data/checksums/hiba_sha256.txt"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--release-id", default=RELEASE_ID)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_paths = (
        args.manifest_output,
        args.audit_output,
        args.checksum_output,
    )
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing HIBA audit output(s): "
            + ", ".join(existing)
        )
    manifest, audit, checksum_lines = audit_dataset(
        args.metadata, args.raw_root, args.mapping, args.isic_manifest,
        project_root=args.project_root.resolve(), release_id=args.release_id,
    )
    publish_outputs_transactionally(
        args.manifest_output,
        args.audit_output,
        args.checksum_output,
        manifest,
        audit,
        checksum_lines,
    )
    print(f"Wrote {len(manifest)} HIBA audit rows; inference remains unauthorized.")
    return 0 if audit["approval"]["automated_gates_passed"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AuditError, FileNotFoundError, OSError, yaml.YAMLError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
