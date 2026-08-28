from pathlib import Path

import pandas as pd
import pytest

from src.analysis.phase04_final_internal_test import (
    EXPECTED_COMMIT,
    analyze,
    audit_inputs,
)
from src.analysis.stored_prediction_statistics import StatisticalAnalysisError

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/phase04_controlled_comparative/final_internal_test"
CONFIG = ROOT / "configs/evaluation/phase04_controlled_comparative_internal_test.yaml"


def _audit():
    return audit_inputs(OUTPUT / "final_internal_test_summary.json", OUTPUT / "paired_internal_test_predictions.csv", CONFIG)


def test_frozen_artifacts_pass_integrity_and_recompute() -> None:
    summary, frame, recomputed, integrity = _audit()
    assert integrity["status"] == "PASS"
    assert integrity["frozen_git_commit"] == EXPECTED_COMMIT
    assert len(frame) == 3668
    assert recomputed["flat_four_class"]["macro_f1"] == pytest.approx(0.6192224685168973)
    assert recomputed["shared_predicted_gate"]["macro_f1"] == pytest.approx(0.5685909456725847)


def test_analysis_is_deterministic_and_has_expected_paired_counts() -> None:
    summary, frame, recomputed, integrity = _audit()
    first = analyze(summary, frame, recomputed, integrity, iterations=200, seed=42)
    second = analyze(summary, frame, recomputed, integrity, iterations=200, seed=42)
    assert first["paired_differences"] == second["paired_differences"]
    assert first["mcnemar_exact"] == second["mcnemar_exact"]
    assert sum(first["mcnemar_exact"][key] for key in ("both_correct", "hierarchy_only_correct", "flat_only_correct", "both_wrong")) == 3668


def test_duplicate_identifier_fails_loudly(tmp_path: Path) -> None:
    summary = OUTPUT / "final_internal_test_summary.json"
    frame = pd.read_csv(OUTPUT / "paired_internal_test_predictions.csv")
    frame.loc[1, "sample_id"] = frame.loc[0, "sample_id"]
    bad = tmp_path / "paired.csv"
    frame.to_csv(bad, index=False)
    with pytest.raises(StatisticalAnalysisError, match="duplicate"):
        audit_inputs(summary, bad, CONFIG)
