from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.analysis.paired_model_comparison import (
    COMMON_CLASSES,
    PairingAuditError,
    PredictionSchema,
    audit_prediction_pairing,
)


FIELDS = ["id", "target_index", "target", "pred_index", "prediction"]
MAPPING = {name: index for index, name in enumerate(COMMON_CLASSES)}
SCHEMA = PredictionSchema("id", "target", "target_index", "prediction", "pred_index", MAPPING)


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _rows() -> list[dict[str, str]]:
    return [
        {"id": "b", "target_index": "1", "target": "melanoma", "pred_index": "2", "prediction": "bcc"},
        {"id": "a", "target_index": "0", "target": "non_malignant", "pred_index": "0", "prediction": "non_malignant"},
        {"id": "c", "target_index": "3", "target": "scc", "pred_index": "3", "prediction": "scc"},
    ]


def _audit(tmp_path: Path, left: list[dict[str, str]], right: list[dict[str, str]]):
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    _write(first, left)
    _write(second, right)
    return audit_prediction_pairing(
        first, second, hierarchical_schema=SCHEMA, flat_schema=SCHEMA, expected_count=3
    )


def test_valid_one_to_one_pairing_is_row_order_independent(tmp_path: Path) -> None:
    audit, paired = _audit(tmp_path, _rows(), list(reversed(_rows())))
    assert audit["status"] == "passed"
    assert [row["sample_id"] for row in paired] == ["a", "b", "c"]


def test_duplicate_identifiers_fail(tmp_path: Path) -> None:
    rows = _rows()
    rows[2]["id"] = "a"
    with pytest.raises(PairingAuditError, match="duplicate sample identifier"):
        _audit(tmp_path, rows, _rows())


def test_missing_identifiers_fail(tmp_path: Path) -> None:
    rows = _rows()
    rows[0]["id"] = ""
    with pytest.raises(PairingAuditError, match="missing sample identifier"):
        _audit(tmp_path, rows, _rows())


def test_mismatched_identifier_sets_fail(tmp_path: Path) -> None:
    rows = _rows()
    rows[0]["id"] = "different"
    with pytest.raises(PairingAuditError, match="Identifier sets differ"):
        _audit(tmp_path, rows, _rows())


def test_ground_truth_disagreement_fails(tmp_path: Path) -> None:
    rows = _rows()
    rows[0]["target"], rows[0]["target_index"] = "bcc", "2"
    with pytest.raises(PairingAuditError, match="Ground-truth labels disagree"):
        _audit(tmp_path, rows, _rows())


def test_unsupported_class_label_fails(tmp_path: Path) -> None:
    rows = _rows()
    rows[0]["prediction"] = "nevus"
    with pytest.raises(PairingAuditError, match="unsupported prediction label"):
        _audit(tmp_path, rows, _rows())


def test_wrong_expected_sample_count_fails(tmp_path: Path) -> None:
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    _write(first, _rows())
    _write(second, _rows())
    with pytest.raises(PairingAuditError, match="expected 4 prediction rows"):
        audit_prediction_pairing(
            first, second, hierarchical_schema=SCHEMA, flat_schema=SCHEMA, expected_count=4
        )


def test_output_is_deterministic(tmp_path: Path) -> None:
    first_audit, first_rows = _audit(tmp_path, _rows(), list(reversed(_rows())))
    second_audit, second_rows = _audit(tmp_path, _rows(), list(reversed(_rows())))
    assert json.dumps(first_audit, sort_keys=True) == json.dumps(second_audit, sort_keys=True)
    assert first_rows == second_rows
