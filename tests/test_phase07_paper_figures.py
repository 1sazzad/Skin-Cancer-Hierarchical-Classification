from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.paper_figures import (
    CLASSES,
    FigureEvidenceError,
    generate_figures,
    normalize_confusion,
)
from src.analysis.stored_prediction_statistics import sha256_file


SOURCE = Path("reports/phase07/generated")


def test_fixed_class_order_and_normalization() -> None:
    assert CLASSES == ("non-malignant", "melanoma", "BCC", "SCC")
    matrix = np.array([[2, 1, 1, 0], [0, 2, 0, 0], [1, 0, 3, 0], [0, 0, 1, 1]])
    normalized = normalize_confusion(matrix)
    np.testing.assert_allclose(normalized.sum(axis=1), 1)
    assert normalized[0, 0] == 0.5


def test_committed_f1_uses_model_specific_intervals() -> None:
    table = pd.read_csv(SOURCE / "paper_table_per_class_f1.csv")
    assert table["class"].tolist() == ["non_malignant", "melanoma", "bcc", "scc"]
    assert table.loc[3, "support"] == 94
    assert table.loc[1, "flat_ci_lower"] < table.loc[1, "flat_f1"] < table.loc[1, "flat_ci_upper"]


def test_missing_source_fails(tmp_path: Path) -> None:
    with pytest.raises(FigureEvidenceError, match="Missing figure source"):
        generate_figures(tmp_path, tmp_path / "figures", tmp_path / "audit", "test")


def test_deterministic_outputs_and_audit_content(tmp_path: Path) -> None:
    first = generate_figures(SOURCE, tmp_path / "one/figures", tmp_path / "one/generated", "fixed")
    second = generate_figures(SOURCE, tmp_path / "two/figures", tmp_path / "two/generated", "fixed")
    assert len(first) == len(second) == 11
    for left, right in zip(first, second):
        assert left.suffix == right.suffix
        assert left.read_bytes() == right.read_bytes()
    audit = json.loads((tmp_path / "one/generated/figure_data_audit.json").read_text())
    assert audit["confusion"]["row_sum_check"] is True
    assert audit["per_class_f1"]["paired_difference_intervals_used"] is False
    assert audit["per_class_f1"]["exploratory"] is True
    assert audit["architecture"]["stage2_is_conditional"] is True
    assert "significance" not in json.dumps(audit).lower()


def test_generation_does_not_modify_committed_sources(tmp_path: Path) -> None:
    before = {path.name: sha256_file(path) for path in SOURCE.iterdir() if path.is_file()}
    generate_figures(SOURCE, tmp_path / "figures", tmp_path / "generated", "fixed")
    after = {path.name: sha256_file(path) for path in SOURCE.iterdir() if path.is_file()}
    assert before == after
