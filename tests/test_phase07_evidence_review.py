from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.analysis.phase07_evidence_review import (
    PROHIBITED,
    QUALIFIED,
    SUPPORTED,
    generate_gate4,
    review_gate3,
)
from src.analysis.stored_prediction_statistics import sha256_file


SOURCE = Path("reports/phase07/generated")


def test_independent_review_arithmetic_and_zero_inclusion() -> None:
    review = review_gate3(SOURCE)
    assert review["status"] == "passed"
    assert review["accuracy_arithmetic"]["flat_correct"] == 2722
    assert review["accuracy_arithmetic"]["hierarchical_correct"] == 2715
    assert review["primary_ci_includes_zero"] is True
    assert review["independent_mcnemar"]["status"] == "passed"


def test_claim_categories_and_required_qualifications() -> None:
    supported = " ".join(SUPPORTED).lower()
    qualified = " ".join(QUALIFIED).lower()
    assert "statistically distinguishable" in supported
    assert "exploratory" in qualified
    assert "support was 94" in supported
    assert len(PROHIBITED) == len(set(PROHIBITED))
    for phrase in PROHIBITED:
        assert phrase.lower() not in supported


def test_routing_denominators_and_overlap_warning() -> None:
    review = review_gate3(SOURCE)
    routing = review["routing_audit"]
    assert routing["malignant_partition"] == "255 + 169 + 846 = 1270"
    assert routing["non_malignant_partition"] == "529 + 1869 = 2398"
    assert "data-availability state" in routing["overlap_note"]
    assert "must not be presented as an error" in routing["overlap_note"]
    generated = Path("reports/phase07/generated/routing_metric_data_dictionary.csv")
    if generated.exists():
        dictionary = pd.read_csv(generated).set_index("metric_name")
        assert dictionary.loc["true_malignant_routed_non_malignant", "denominator"] == 1270
        assert dictionary.loc["correct_malignant_route_wrong_subtype", "denominator"] == 1015
        assert dictionary.loc["correct_route_correct_subtype", "denominator"] == 1015
        assert dictionary.loc["true_non_malignant_routed_stage2", "denominator"] == 2398
        assert dictionary.loc["structural_stage2_missing_not_invoked", "denominator"] == 3668


def test_generation_order_and_determinism(tmp_path: Path) -> None:
    first_generated = tmp_path / "first" / "generated"
    first_reports = tmp_path / "first" / "reports"
    second_generated = tmp_path / "second" / "generated"
    second_reports = tmp_path / "second" / "reports"
    first = generate_gate4(SOURCE, first_generated, first_reports)
    second = generate_gate4(SOURCE, second_generated, second_reports)
    assert len(first) == len(second) == 12
    for left, right in zip(first, second):
        assert left.name == right.name
        assert left.read_bytes() == right.read_bytes()
    classes = pd.read_csv(first_generated / "paper_table_per_class_f1.csv")["class"]
    assert classes.tolist() == ["non_malignant", "melanoma", "bcc", "scc"]
    claims = json.loads((first_generated / "claims_lock.json").read_text())
    assert claims["status"] == "locked"


def test_generation_does_not_modify_gate3_artifacts(tmp_path: Path) -> None:
    before = {
        path.name: sha256_file(path)
        for path in SOURCE.iterdir()
        if path.is_file()
    }
    generate_gate4(SOURCE, tmp_path / "generated", tmp_path / "reports")
    after = {
        path.name: sha256_file(path)
        for path in SOURCE.iterdir()
        if path.is_file()
    }
    assert before == after
