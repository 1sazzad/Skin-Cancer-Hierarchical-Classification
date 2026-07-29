from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.audit_phase08_evidence_completeness import (
    EvidenceAuditError,
    LOCKED_REQUIRED,
    OBJECTIVE_COLUMNS,
    build_inventory,
    generate_audit,
    validate_locked_evidence,
)


def _fixture_repository(root: Path) -> list[str]:
    tracked = list(LOCKED_REQUIRED) + [
        "configs/project.yaml",
        "scripts/example.py",
        "tests/test_example.py",
    ]
    for relative in tracked:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")
    return tracked


def test_missing_locked_evidence_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(EvidenceAuditError, match="Missing required locked evidence"):
        validate_locked_evidence(tmp_path)


def test_inventory_distinguishes_tracked_and_external_references(tmp_path: Path) -> None:
    tracked = _fixture_repository(tmp_path)
    rows = build_inventory(tmp_path, tracked)
    config = next(row for row in rows if row["artifact_path"] == "configs/project.yaml")
    checkpoint = next(row for row in rows if row["artifact_type"] == "checkpoint")
    assert config["tracked"] == "yes"
    assert config["exists_locally"] == "yes"
    assert checkpoint["tracked"] == "no"
    assert checkpoint["exists_locally"] == "not_audited"
    assert checkpoint["missing"] == "not_audited"
    assert checkpoint["reusable"] == "external_artifact_reference"


def test_generation_is_deterministic_and_has_required_matrix_columns(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    tracked = _fixture_repository(repository)
    first = generate_audit(repository, tmp_path / "one", tracked)
    second = generate_audit(repository, tmp_path / "two", tracked)
    assert [path.name for path in first] == [path.name for path in second]
    for left, right in zip(first, second):
        assert left.read_bytes() == right.read_bytes()
    with first[0].open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == OBJECTIVE_COLUMNS
        rows = list(reader)
    assert any(row["objective_id"] == "O03" and row["claim_allowed_now"] == "no" for row in rows)


def test_inventory_normalizes_windows_and_posix_tracked_paths(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    tracked = _fixture_repository(repository)
    mixed_paths = [
        path.replace("/", "\\") if index % 2 else path
        for index, path in enumerate(tracked)
    ]
    rows = build_inventory(repository, mixed_paths)
    artifact_paths = {row["artifact_path"] for row in rows}
    assert "configs/project.yaml" in artifact_paths
    assert "scripts/example.py" in artifact_paths
    assert all("\\" not in path for path in artifact_paths)


def test_local_artifact_presence_does_not_change_generated_bytes(
    tmp_path: Path,
) -> None:
    without_local = tmp_path / "without_local"
    with_local = tmp_path / "with_local"
    tracked_without = _fixture_repository(without_local)
    tracked_with = _fixture_repository(with_local)
    local_path = (
        with_local
        / "runs/phase05_hierarchical_internal_test/locked_primary_evaluation"
        / "per_image_hierarchical_predictions.csv"
    )
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text("ignored local fixture\n", encoding="utf-8")

    first = generate_audit(without_local, tmp_path / "without_output", tracked_without)
    second = generate_audit(with_local, tmp_path / "with_output", tracked_with)

    assert [path.name for path in first] == [path.name for path in second]
    for left, right in zip(first, second):
        assert left.read_bytes() == right.read_bytes()


def test_audit_implementation_is_excluded_from_inventory(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    tracked = _fixture_repository(repository)
    tracked.extend(
        [
            "scripts/audit_phase08_evidence_completeness.py",
            "tests/test_phase08_evidence_completeness.py",
        ]
    )
    rows = build_inventory(repository, tracked)
    artifact_paths = {row["artifact_path"] for row in rows}
    assert "scripts/audit_phase08_evidence_completeness.py" not in artifact_paths
    assert "tests/test_phase08_evidence_completeness.py" not in artifact_paths


def test_generated_outputs_use_lf_line_endings_only(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    tracked = _fixture_repository(repository)
    outputs = generate_audit(repository, tmp_path / "output", tracked)
    for output in outputs:
        content = output.read_bytes()
        assert b"\r" not in content
        assert content.endswith(b"\n")

    payload = json.loads(outputs[2].read_text(encoding="utf-8"))
    assert payload["audit_version"] == 2
    assert (
        payload["local_artifact_presence_policy"]
        == "not_audited_in_committed_inventory"
    )
