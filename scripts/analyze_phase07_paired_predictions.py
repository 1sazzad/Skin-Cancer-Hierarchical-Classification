"""Run the Phase 07 stored-prediction integrity and pairing audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import tempfile
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.analysis.paired_model_comparison import (  # noqa: E402
    COMMON_CLASSES,
    PairingAuditError,
    PredictionSchema,
    audit_prediction_pairing,
    inspect_csv,
    sha256_file,
    write_json,
    write_rows,
)


def _manifest_entries(text: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in text.splitlines():
        digest, path = line.strip().split(maxsplit=1)
        entries[path] = digest
    return entries


def _verify_manifest_file(manifest: Path, root: Path) -> dict[str, str]:
    entries = _manifest_entries(manifest.read_text(encoding="utf-8"))
    for relative_path, expected in entries.items():
        path = root / relative_path
        if not path.is_file() or sha256_file(path) != expected:
            raise PairingAuditError(
                f"Locked manifest verification failed for {relative_path}."
            )
    return entries


def _verify_archive(
    archive: Path, expected_hash: str, extract_root: Path, manifest_member: str
) -> tuple[Path, dict[str, str]]:
    observed = sha256_file(archive)
    if observed != expected_hash:
        raise PairingAuditError(
            f"Phase 06C backup hash mismatch: expected {expected_hash}, "
            f"observed {observed}."
        )
    with tarfile.open(archive, "r:gz") as bundle:
        members = {member.name: member for member in bundle.getmembers()}
        if manifest_member not in members:
            raise PairingAuditError(
                f"Archive lacks artifact manifest {manifest_member}."
            )
        manifest_handle = bundle.extractfile(members[manifest_member])
        if manifest_handle is None:
            raise PairingAuditError("Could not read embedded artifact manifest.")
        entries = _manifest_entries(
            manifest_handle.read().decode("utf-8")
        )
        for member_name, expected in entries.items():
            member = members.get(member_name)
            if member is None:
                raise PairingAuditError(
                    f"Archive manifest member is missing: {member_name}."
                )
            handle = bundle.extractfile(member)
            if handle is None:
                raise PairingAuditError(
                    f"Archive member is not a readable file: {member_name}."
                )
            observed_member = hashlib.sha256(handle.read()).hexdigest()
            if observed_member != expected:
                raise PairingAuditError(
                    f"Archive member hash mismatch: {member_name}."
                )
        prediction_member = next(
            name for name in entries if name.endswith("/internal_test_predictions.csv")
        )
        member = members[prediction_member]
        destination = extract_root / "internal_test_predictions.csv"
        handle = bundle.extractfile(member)
        if handle is None:
            raise PairingAuditError("Could not extract Phase 06C predictions.")
        destination.write_bytes(handle.read())
    return destination, entries


def _schema(section: dict[str, object]) -> PredictionSchema:
    mapping = {name: index for index, name in enumerate(COMMON_CLASSES)}
    return PredictionSchema(
        identifier=str(section["identifier_column"]),
        target_label=str(section["target_label_column"]),
        target_index=str(section["target_index_column"]),
        predicted_label=str(section["predicted_label_column"]),
        predicted_index=str(section["predicted_index_column"]),
        class_to_index=mapping,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/analysis/phase07_paired_model_comparison.yaml"),
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = Path(config["output_directory"])
    hierarchical_config = config["hierarchical"]
    flat_config = config["flat"]
    hierarchical_path = Path(hierarchical_config["predictions_path"])

    phase05_entries = _verify_manifest_file(
        Path(hierarchical_config["sha256_manifest"]), Path(".")
    )
    canonical_flat_path = str(flat_config["canonical_predictions_path"])
    with tempfile.TemporaryDirectory(prefix="phase07_pairing_") as temporary:
        flat_path, phase06_entries = _verify_archive(
            Path(flat_config["backup_archive"]),
            str(flat_config["backup_archive_sha256"]),
            Path(temporary),
            str(flat_config["artifact_manifest_member"]),
        )
        audit, paired = audit_prediction_pairing(
            hierarchical_path,
            flat_path,
            hierarchical_schema=_schema(hierarchical_config),
            flat_schema=_schema(flat_config),
            expected_count=int(config["expected_sample_count"]),
        )
        audit["hierarchical_csv"]["path"] = hierarchical_path.as_posix()
        audit["flat_csv"]["path"] = canonical_flat_path

        inventory = []
        candidate_paths = sorted(Path("runs").glob("**/*predictions.csv"))
        for path in candidate_paths:
            item = inspect_csv(path)
            record = {
                **item.__dict__,
                "columns": json.dumps(item.columns),
                "duplicate_columns": json.dumps(item.duplicate_columns),
                "missing_value_counts": json.dumps(
                    item.missing_value_counts, sort_keys=True
                ),
                "provenance": (
                    "selected_phase05_locked_manifest"
                    if path == hierarchical_path
                    else "alternative_historical_prediction_artifact"
                ),
            }
            inventory.append(record)
        flat_item = inspect_csv(flat_path, display_path=canonical_flat_path)
        inventory.append(
            {
                **flat_item.__dict__,
                "columns": json.dumps(flat_item.columns),
                "duplicate_columns": json.dumps(flat_item.duplicate_columns),
                "missing_value_counts": json.dumps(
                    flat_item.missing_value_counts, sort_keys=True
                ),
                "provenance": "selected_phase06c_verified_backup_archive",
            }
        )

    selected = {
        "hierarchical": {
            "canonical_path": hierarchical_path.as_posix(),
            "sha256": audit["hierarchical_csv"]["sha256"],
            "manifest_path": str(hierarchical_config["sha256_manifest"]),
            "manifest_sha256": sha256_file(
                Path(hierarchical_config["sha256_manifest"])
            ),
            "manifest_entry_count": len(phase05_entries),
        },
        "flat": {
            "canonical_path": canonical_flat_path,
            "sha256": audit["flat_csv"]["sha256"],
            "backup_archive": str(flat_config["backup_archive"]),
            "backup_archive_sha256": str(flat_config["backup_archive_sha256"]),
            "embedded_manifest_entry_count": len(phase06_entries),
        },
    }
    write_json(output / "selected_prediction_artifacts.json", selected)
    write_json(output / "prediction_pairing_audit.json", audit)
    write_rows(
        output / "prediction_file_inventory.csv",
        inventory,
        [
            "path", "size_bytes", "sha256", "row_count", "columns",
            "duplicate_columns", "empty_rows", "malformed_rows",
            "missing_value_counts", "provenance",
        ],
    )
    write_rows(
        output / "paired_prediction_manifest.csv",
        paired,
        [
            "sample_id", "target_label", "target_index",
            "hierarchical_predicted_label", "hierarchical_predicted_index",
            "flat_predicted_label", "flat_predicted_index",
        ],
    )
    print(f"Phase 07 pairing audit passed for {len(paired)} samples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
