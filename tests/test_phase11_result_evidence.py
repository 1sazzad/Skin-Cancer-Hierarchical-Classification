from __future__ import annotations

import csv
import json
from pathlib import Path


METRICS_PATH = Path(
    "experiments/evaluations/"
    "phase11_densenet121_internal_test_seed42__best_epoch04/"
    "internal_test_metrics.json"
)

COMPARISON_PATH = Path(
    "reports/phase11/generated/final_model_comparison.csv"
)

REGISTRY_PATH = Path("experiments/experiment_registry.csv")

REPORT_PATH = Path(
    "reports/phase11/phase11_final_densenet_baseline_result.md"
)


def test_phase11_locked_metrics() -> None:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    assert metrics["sample_count"] == 3668
    assert metrics["accuracy"] == 0.7914394765539804
    assert metrics["balanced_accuracy"] == 0.6168282266253176
    assert metrics["macro_f1"] == 0.6351074531606824
    assert metrics["weighted_f1"] == 0.7861623640231898


def test_phase11_three_model_comparison() -> None:
    with COMPARISON_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert [row["system"] for row in rows] == [
        "Flat DenseNet-121",
        "Flat EfficientNet-B0",
        "Predicted-gate hierarchy",
    ]

    assert rows[0]["accuracy"] == "0.7914394766"
    assert rows[0]["balanced_accuracy"] == "0.6168282266"
    assert rows[0]["macro_f1"] == "0.6351074532"
    assert rows[0]["weighted_f1"] == "0.7861623640"
    assert "Descriptive" in rows[0]["inferential_scope"]


def test_phase11_registry_entry() -> None:
    with REGISTRY_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    matching = [
        row
        for row in rows
        if row["run_id"] ==
        "phase11_flat_four_class_densenet121_seed42"
    ]

    assert len(matching) == 1

    row = matching[0]

    assert row["status"] == "completed_locked"
    assert row["primary_metric"] == "internal_test_macro_f1"
    assert row["primary_metric_value"] == "0.6351074531606824"
    assert row["git_commit"] == (
        "133442a185060d78f23019d6997b12526e6cef3c"
    )
    assert "rerun allowed=false" in row["notes"]


def test_phase11_claim_boundary() -> None:
    report = REPORT_PATH.read_text(encoding="utf-8")
    normalized_report = " ".join(report.split())

    assert "No new paired confidence interval" in normalized_report
    assert (
        "must not be reported as statistically significant superiority"
        in normalized_report
    )
    assert "No additional model training" in normalized_report
def test_phase11_final_audit_amendment() -> None:
    audit_path = Path(
        "reports/complete_scientific_audit_iccit2026.md"
    )

    audit = audit_path.read_text(encoding="utf-8")
    normalized_audit = " ".join(audit.split())

    assert (
        "Post-Audit Amendment: Final Phase 11 DenseNet-121 Comparator"
        in audit
    )
    assert "Flat DenseNet-121 | 0.791439" in audit
    assert "SCC recall remained `0.297872`" in normalized_audit
    assert "No new paired confidence interval" in normalized_audit
    assert (
        "must not be presented as statistically significant superiority"
        in normalized_audit
    )
    assert (
        "Oracle routing increased hierarchical macro-F1"
        in normalized_audit
    )


def test_iccit_manuscript_is_overleaf_owned() -> None:
    gitignore = Path(".gitignore").read_text(encoding="utf-8")

    assert "/paper/iccit2026/main.tex" in gitignore
    assert "/paper/iccit2026/references.bib" in gitignore

    assert not Path("paper/iccit2026/main.tex").exists()
    assert not Path("paper/iccit2026/references.bib").exists()
