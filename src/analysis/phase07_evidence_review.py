"""Independent Gate 4 review and deterministic paper-table generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.analysis.stored_prediction_statistics import (
    CLASSES,
    calculate_metrics,
    sha256_file,
    write_csv,
    write_json,
)


class EvidenceReviewError(ValueError):
    """Raised when committed Gate 3 evidence is internally inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceReviewError(message)


def _ci_text(lower: float, upper: float) -> str:
    return f"[{lower:.6f}, {upper:.6f}]"


def _load_confusion(path: Path) -> np.ndarray:
    frame = pd.read_csv(path)
    _require(frame["true_label"].tolist() == list(CLASSES), "Confusion class order changed.")
    return frame.drop(columns="true_label").to_numpy(dtype=np.int64)


def review_gate3(generated: Path) -> dict[str, Any]:
    """Independently recompute and validate committed Gate 3 evidence."""
    results = json.loads((generated / "statistical_analysis_results.json").read_text())
    intervals = pd.read_csv(generated / "bootstrap_confidence_intervals.csv")
    points = pd.read_csv(generated / "model_metric_point_estimates.csv")
    correctness = pd.read_csv(generated / "paired_correctness_table.csv").iloc[0]
    agreement = pd.read_csv(generated / "prediction_agreement_summary.csv")
    scc = pd.read_csv(generated / "scc_error_analysis.csv")
    routing = pd.read_csv(generated / "hierarchical_routing_decomposition.csv")

    metrics = {
        "flat": calculate_metrics(
            np.repeat(np.arange(4), _load_confusion(generated / "confusion_matrix_flat.csv").sum(axis=1)),
            np.concatenate(
                [
                    np.repeat(np.arange(4), row)
                    for row in _load_confusion(generated / "confusion_matrix_flat.csv")
                ]
            ),
        ),
        "hierarchical": calculate_metrics(
            np.repeat(
                np.arange(4),
                _load_confusion(generated / "confusion_matrix_hierarchical.csv").sum(axis=1),
            ),
            np.concatenate(
                [
                    np.repeat(np.arange(4), row)
                    for row in _load_confusion(generated / "confusion_matrix_hierarchical.csv")
                ]
            ),
        ),
    }
    tolerance = 1e-12
    for model, calculated in metrics.items():
        for metric in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"):
            committed = float(
                points.query(
                    "model == @model and metric == @metric and `class` == 'overall'"
                )["value"].iloc[0]
            )
            _require(
                abs(getattr(calculated, metric) - committed) <= tolerance,
                f"{model} {metric} does not reconcile.",
            )
        for index, class_name in enumerate(CLASSES):
            rows = points.query("model == @model and `class` == @class_name")
            for metric in ("precision", "recall", "f1"):
                committed = float(rows.query("metric == @metric")["value"].iloc[0])
                _require(
                    abs(getattr(calculated, metric)[index] - committed) <= tolerance,
                    f"{model} {class_name} {metric} does not reconcile.",
                )

    ci = intervals.set_index("estimand")
    _require((intervals["lower"] <= intervals["upper"]).all(), "Invalid CI ordering.")
    for name in (
        "flat_accuracy",
        "hierarchical_accuracy",
        "flat_balanced_accuracy",
        "hierarchical_balanced_accuracy",
        "flat_macro_f1",
        "hierarchical_macro_f1",
    ):
        _require(
            ci.loc[name, "lower"]
            <= ci.loc[name, "point_estimate"]
            <= ci.loc[name, "upper"],
            f"Model point estimate outside interval: {name}.",
        )
    for metric in ("accuracy", "balanced_accuracy", "macro_f1", *[f"f1_{x}" for x in CLASSES]):
        difference = ci.loc[f"difference_{metric}", "point_estimate"]
        expected = (
            ci.loc[f"flat_{metric}", "point_estimate"]
            - ci.loc[f"hierarchical_{metric}", "point_estimate"]
        )
        _require(abs(difference - expected) <= tolerance, f"Difference direction failed: {metric}.")

    cells = [
        int(correctness["both_correct"]),
        int(correctness["flat_correct_hierarchy_wrong"]),
        int(correctness["flat_wrong_hierarchy_correct"]),
        int(correctness["both_wrong"]),
    ]
    _require(sum(cells) == 3668, "Correctness cells do not sum to 3668.")
    flat_correct = cells[0] + cells[1]
    hierarchy_correct = cells[0] + cells[2]
    accuracy_difference = (cells[1] - cells[2]) / 3668
    _require(abs(flat_correct / 3668 - metrics["flat"].accuracy) <= tolerance, "Flat accuracy arithmetic failed.")
    _require(
        abs(hierarchy_correct / 3668 - metrics["hierarchical"].accuracy) <= tolerance,
        "Hierarchical accuracy arithmetic failed.",
    )
    _require(
        abs(accuracy_difference - float(correctness["paired_accuracy_difference"])) <= tolerance,
        "Paired accuracy difference arithmetic failed.",
    )
    _require(
        abs(accuracy_difference - float(correctness["net_paired_correctness_advantage"]))
        <= tolerance,
        "Net advantage arithmetic failed.",
    )
    odds = cells[1] / cells[2]
    committed_odds = results["mcnemar"]["raw_discordant_pair_odds_ratio"]["numeric_value"]
    _require(abs(odds - committed_odds) <= tolerance, "Discordant odds ratio failed.")
    recomputed_p = float(binomtest(cells[1], cells[1] + cells[2], 0.5).pvalue)
    committed_p = float(correctness["exact_two_sided_p_value"])
    p_difference = abs(recomputed_p - committed_p)
    _require(p_difference <= tolerance, "Independent McNemar verification failed.")

    agreement_counts = dict(zip(agreement["category"], agreement["count"]))
    _require(
        int(agreement_counts["exact_prediction_agreement"])
        + int(agreement_counts["prediction_disagreement"])
        == 3668,
        "Agreement arithmetic failed.",
    )
    np.testing.assert_array_equal(metrics["flat"].support, [2398, 678, 498, 94])
    np.testing.assert_array_equal(metrics["hierarchical"].support, [2398, 678, 498, 94])
    for model in ("flat", "hierarchical"):
        row = scc.query("model == @model").iloc[0]
        _require(
            sum(int(row[f"predicted_{name}"]) for name in CLASSES) == 94,
            f"{model} SCC counts do not sum to 94.",
        )
    paired_scc = scc.query("model == 'paired_categories'").iloc[0]
    _require(
        sum(int(paired_scc[name]) for name in ("correct_only_flat", "correct_only_hierarchy", "both_wrong", "both_correct"))
        == 94,
        "SCC correctness categories do not sum to 94.",
    )

    routing_counts = dict(zip(routing["category"], routing["count"]))
    _require(
        routing_counts
        == {
            "true_malignant_routed_non_malignant": 255,
            "true_non_malignant_routed_stage2": 529,
            "correct_malignant_route_wrong_subtype": 169,
            "correct_route_correct_subtype": 846,
            "structural_stage2_missing_not_invoked": 1869,
            "anomalous_stage2_missing": 0,
        },
        "Routing counts changed.",
    )
    _require(255 + 169 + 846 == 1270, "Malignant routing partition failed.")
    _require(529 + 1869 == 2398, "Non-malignant routing partition failed.")

    primary = ci.loc["difference_macro_f1"]
    _require(primary["lower"] <= 0 <= primary["upper"], "Primary zero-inclusion failed.")
    _require(results["primary"]["includes_zero"] is True, "Committed zero decision failed.")
    return {
        "status": "passed",
        "sample_count": 3668,
        "class_order": list(CLASSES),
        "class_support": dict(zip(CLASSES, [2398, 678, 498, 94])),
        "primary_ci_includes_zero": True,
        "independent_mcnemar": {
            "b": 354,
            "c": 347,
            "n": 701,
            "recomputed_p_value": recomputed_p,
            "committed_p_value": committed_p,
            "absolute_difference": p_difference,
            "tolerance": tolerance,
            "status": "passed",
        },
        "accuracy_arithmetic": {
            "flat_correct": flat_correct,
            "hierarchical_correct": hierarchy_correct,
            "difference": accuracy_difference,
        },
        "routing_audit": {
            "status": "passed_with_denominator_clarification",
            "malignant_partition": "255 + 169 + 846 = 1270",
            "non_malignant_partition": "529 + 1869 = 2398",
            "entire_set_partition": "255 + 169 + 846 + 529 + 1869 = 3668",
            "structural_missingness_scope": (
                "All Stage-2-not-invoked samples; under the stored union execution policy "
                "these are exactly 1,869 true non-malignant correctly not routed cases."
            ),
            "overlap_note": (
                "The six emitted rows are mutually exclusive, but structural missingness is "
                "a data-availability state that coincides with the implicit correct "
                "non-malignant routing category; it must not be presented as an error."
            ),
        },
    }


SUPPORTED = [
    "Both architectures were evaluated on the same locked leakage-aware internal-test split.",
    "Flat observed macro-F1 was 0.619 and hierarchical observed macro-F1 was 0.605.",
    "The observed flat-minus-hierarchical macro-F1 difference was approximately 0.014.",
    "The paired 95% confidence interval included zero.",
    "The analysis did not establish a statistically distinguishable macro-F1 difference.",
    "Overall paired correctness was similar, and exact McNemar testing did not detect a difference.",
    "Flat observed balanced accuracy was higher, but its paired interval included zero.",
    "Per-class findings are exploratory and SCC estimates are uncertain because support was 94.",
    "Results apply only to the locked ISIC 2019 internal-test split.",
    "The flat system uses one model decision path per image; the hierarchical design uses conditional routing.",
]
QUALIFIED = [
    "On this single split, the flat model showed a possible melanoma-specific advantage; this is exploratory, no class-wise p-values were generated, and multiplicity-adjusted class-wise inference was not performed.",
    "The hierarchical model had slightly higher observed non-malignant and BCC F1; these descriptive differences do not establish superiority.",
    "Conditional routing changed the observed error distribution on this split; external operational consequences were not evaluated.",
    "The designs may offer different operational trade-offs, but efficiency and external deployment evidence are not established here.",
]
PROHIBITED = [
    "statistically equivalent",
    "non-inferior",
    "clinically superior",
    "clinically validated",
    "improves diagnosis",
    "reduces mortality",
    "ready for deployment",
    "robust across datasets",
    "generalizes across populations",
    "fair across skin tones",
    "externally validated",
    "causal benefit",
    "statistically significant melanoma advantage",
    "statistically significant SCC advantage",
    "definitive rare-class superiority",
    "claims based on selected individual examples",
]


def generate_gate4(source: Path, destination: Path, reports: Path) -> list[Path]:
    """Generate deterministic Gate 4 tables, locks and review documents."""
    review = review_gate3(source)
    ci = pd.read_csv(source / "bootstrap_confidence_intervals.csv").set_index("estimand")
    points = pd.read_csv(source / "model_metric_point_estimates.csv")
    correctness = pd.read_csv(source / "paired_correctness_table.csv").iloc[0]
    agreement = pd.read_csv(source / "prediction_agreement_summary.csv")
    routing = pd.read_csv(source / "hierarchical_routing_decomposition.csv")

    def point(model: str, metric: str, class_name: str = "overall") -> float:
        return float(
            points.query(
                "model == @model and metric == @metric and `class` == @class_name"
            )["value"].iloc[0]
        )

    main = pd.DataFrame(
        [
            {
                "model": model,
                "accuracy": point(model, "accuracy"),
                "accuracy_ci_lower": ci.loc[f"{model}_accuracy", "lower"],
                "accuracy_ci_upper": ci.loc[f"{model}_accuracy", "upper"],
                "balanced_accuracy": point(model, "balanced_accuracy"),
                "balanced_accuracy_ci_lower": ci.loc[f"{model}_balanced_accuracy", "lower"],
                "balanced_accuracy_ci_upper": ci.loc[f"{model}_balanced_accuracy", "upper"],
                "macro_f1": point(model, "macro_f1"),
                "macro_f1_ci_lower": ci.loc[f"{model}_macro_f1", "lower"],
                "macro_f1_ci_upper": ci.loc[f"{model}_macro_f1", "upper"],
                "weighted_f1": point(model, "weighted_f1"),
            }
            for model in ("flat", "hierarchical")
        ]
    )
    paired = pd.DataFrame(
        [
            {
                "metric": metric,
                "flat_minus_hierarchical": ci.loc[f"difference_{key}", "point_estimate"],
                "ci_lower": ci.loc[f"difference_{key}", "lower"],
                "ci_upper": ci.loc[f"difference_{key}", "upper"],
                "interval_includes_zero": bool(
                    ci.loc[f"difference_{key}", "lower"]
                    <= 0
                    <= ci.loc[f"difference_{key}", "upper"]
                ),
                "analysis_status": "secondary" if key != "macro_f1" else "primary",
            }
            for metric, key in (
                ("Accuracy", "accuracy"),
                ("Balanced accuracy", "balanced_accuracy"),
                ("Macro-F1", "macro_f1"),
            )
        ]
    )
    per_class = pd.DataFrame(
        [
            {
                "class": name,
                "support": support,
                "flat_f1": point("flat", "f1", name),
                "flat_ci_lower": ci.loc[f"flat_f1_{name}", "lower"],
                "flat_ci_upper": ci.loc[f"flat_f1_{name}", "upper"],
                "hierarchical_f1": point("hierarchical", "f1", name),
                "hierarchical_ci_lower": ci.loc[f"hierarchical_f1_{name}", "lower"],
                "hierarchical_ci_upper": ci.loc[f"hierarchical_f1_{name}", "upper"],
                "flat_minus_hierarchical": ci.loc[f"difference_f1_{name}", "point_estimate"],
                "difference_ci_lower": ci.loc[f"difference_f1_{name}", "lower"],
                "difference_ci_upper": ci.loc[f"difference_f1_{name}", "upper"],
            }
            for name, support in zip(CLASSES, [2398, 678, 498, 94])
        ]
    )
    agreement_counts = dict(zip(agreement["category"], agreement["count"]))
    correctness_table = pd.DataFrame(
        [
            {"measure": "both_correct", "value": correctness["both_correct"], "denominator": 3668},
            {"measure": "flat_only_correct", "value": correctness["flat_correct_hierarchy_wrong"], "denominator": 3668},
            {"measure": "hierarchy_only_correct", "value": correctness["flat_wrong_hierarchy_correct"], "denominator": 3668},
            {"measure": "both_wrong", "value": correctness["both_wrong"], "denominator": 3668},
            {"measure": "exact_prediction_agreement", "value": agreement_counts["exact_prediction_agreement"], "denominator": 3668},
            {"measure": "prediction_disagreement", "value": agreement_counts["prediction_disagreement"], "denominator": 3668},
            {"measure": "exact_mcnemar_p_value", "value": correctness["exact_two_sided_p_value"], "denominator": 701},
            {"measure": "raw_discordant_odds_ratio", "value": 354 / 347, "denominator": 701},
        ]
    )
    definitions = [
        ("true_malignant_routed_non_malignant", "True malignant with Stage 1 predicted non-malignant", 1270, "true malignant", "true non-malignant", "true_malignant_partition", "none"),
        ("correct_malignant_route_wrong_subtype", "True malignant, Stage 1 malignant, final subtype incorrect", 1015, "true malignant and correctly routed", "blocked malignant", "correctly_routed_malignant_subtype_partition", "none"),
        ("correct_route_correct_subtype", "True malignant, Stage 1 malignant, final subtype correct", 1015, "true malignant and correctly routed", "blocked malignant", "correctly_routed_malignant_subtype_partition", "none"),
        ("true_non_malignant_routed_stage2", "True non-malignant with Stage 1 predicted malignant", 2398, "true non-malignant", "true malignant", "true_non_malignant_partition", "none"),
        ("structural_stage2_missing_not_invoked", "Stage 2 not invoked among all 3,668 samples; under union execution policy exactly 1,869 of 2,398 true non-malignant cases correctly not routed", 3668, "Stage 2 not invoked", "all true malignant and routed non-malignant", "stage2_invocation_state", "implicit_correct_non_malignant_routing"),
        ("anomalous_stage2_missing", "Stage 2 invoked but stored subtype prediction missing", 1799, "Stage 2 invoked", "Stage 2 not invoked", "stage2_data_quality", "none"),
    ]
    routing_counts = dict(zip(routing["category"], routing["count"]))
    dictionary = pd.DataFrame(
        [
            {
                "metric_name": name,
                "definition": definition,
                "denominator": denominator,
                "inclusion_rule": inclusion,
                "exclusion_rule": exclusion,
                "mutually_exclusive_group": group,
                "overlaps_with": overlaps,
                "value": int(routing_counts[name]),
                "evidence_source": "Phase 05 stored prediction columns and routing reports",
            }
            for name, definition, denominator, inclusion, exclusion, group, overlaps in definitions
        ]
    )
    routing_table = dictionary.iloc[:5][["metric_name", "value", "denominator", "definition"]].copy()
    routing_table.insert(3, "percentage", routing_table["value"] / routing_table["denominator"] * 100)

    destination.mkdir(parents=True, exist_ok=True)
    outputs = [
        destination / "paper_table_main_model_comparison.csv",
        destination / "paper_table_paired_comparison.csv",
        destination / "paper_table_per_class_f1.csv",
        destination / "paper_table_correctness_agreement.csv",
        destination / "routing_metric_data_dictionary.csv",
        destination / "paper_table_routing_decomposition.csv",
    ]
    for path, frame in zip(
        outputs, (main, paired, per_class, correctness_table, dictionary, routing_table)
    ):
        write_csv(path, frame)
    claims = {
        "status": "locked",
        "supported_claims": SUPPORTED,
        "carefully_qualified_claims": QUALIFIED,
        "prohibited_claims": PROHIBITED,
        "scope": "one locked ISIC 2019 internal-test split",
    }
    write_json(destination / "claims_lock.json", claims)
    write_json(destination / "gate04_evidence_review.json", review)
    outputs.extend([destination / "claims_lock.json", destination / "gate04_evidence_review.json"])

    primary_text = (
        "On the locked ISIC 2019 internal-test split, the flat model achieved a "
        "higher observed macro-F1 than the hierarchical model, but the paired 95% "
        "bootstrap confidence interval for the difference included zero. The analysis "
        "therefore did not establish a statistically distinguishable macro-F1 difference."
    )
    results_paragraph = (
        "Macro-F1 was 0.619 for the flat model and 0.605 for the hierarchical model; "
        "the paired flat-minus-hierarchical difference was 0.014 (95% CI −0.014 to "
        "0.042), and the interval included zero. Accuracy was 0.742 and 0.740, "
        "respectively, while the exact McNemar test did not detect a difference in "
        "paired correctness (p=0.821). SCC estimates remain highly uncertain because "
        "support was only 94. These findings apply to one locked ISIC 2019 internal-test "
        "split and do not establish performance beyond this dataset and split."
    )
    discussion = (
        "The small observed macro-F1 advantage for the flat system is not conclusive "
        "because its paired interval included zero. Conditional routing may still offer "
        "interpretability or operational value, but those benefits were not established "
        "by this statistical comparison. The observed melanoma difference is exploratory: "
        "no class-wise p-values or multiplicity-adjusted class-wise inference were "
        "performed. SCC estimates are unstable at support 94, and external evaluation is "
        "needed before broader claims. A non-significant difference does not demonstrate "
        "equivalence or non-inferiority."
    )
    review_md = f"""# Phase 07 Gate 4 — Independent Evidence Review

## Decision

PASS. All committed Gate 3 numerical evidence independently reconciled.

## Numerical review

- Confusion-matrix metrics, class supports, macro/weighted F1, balanced accuracy, paired differences, interval ordering and original point estimates reconciled.
- Flat correct: 2,722; hierarchical correct: 2,715; all four correctness cells sum to 3,668.
- Accuracy difference: `(354 - 347) / 3668 = {review['accuracy_arithmetic']['difference']:.17g}`.
- Prediction agreement and disagreement sum to 3,668.
- The unrounded primary interval includes zero.

## Independent McNemar check

- Recomputed exact two-sided p-value: {review['independent_mcnemar']['recomputed_p_value']:.17g}
- Committed p-value: {review['independent_mcnemar']['committed_p_value']:.17g}
- Absolute difference: {review['independent_mcnemar']['absolute_difference']:.17g}
- Tolerance: {review['independent_mcnemar']['tolerance']:.1e}; status: PASS.

## Routing audit

The malignant subset partitions as `255 + 169 + 846 = 1,270`; the non-malignant subset partitions as `529 + 1,869 = 2,398`. Together the five substantive routing rows partition all 3,668 samples. The emitted rows are mutually exclusive, but structural Stage 2 missingness is a data-availability state that coincides with the implicit correct-non-malignant-routing category; it is not an error.

Because Phase 05 stored Stage 2 outputs for the union of true and predicted malignant samples, every true malignant sample invoked Stage 2. Thus structural missingness 1,869 covers all Stage-2-not-invoked samples and, in this execution policy, exactly the true non-malignant samples correctly not routed. It excludes the 255 malignant Stage 1 routing failures, whose Stage 2 outputs were nevertheless stored.

Exact definitions and denominators are locked in `generated/routing_metric_data_dictionary.csv`.

## Primary interpretation

{primary_text}
"""
    claims_md = "# Phase 07 Claims Lock\n\n## Supported\n\n" + "\n".join(
        f"- {claim}" for claim in SUPPORTED
    ) + "\n\n## Carefully qualified\n\n" + "\n".join(
        f"- {claim}" for claim in QUALIFIED
    ) + "\n\n## Prohibited formulations\n\n" + "\n".join(
        f"- `{claim}`" for claim in PROHIBITED
    ) + "\n"
    tables_md = """# Phase 07 Paper-Table Recommendations

## Six-page ICCIT policy

Use the main architecture comparison and compact per-class F1 table in the main text. Integrate the paired macro-F1 difference, accuracy difference, discordant counts (354 versus 347), and exact McNemar p=0.8207 into the main comparison footnote or adjacent text.

Keep the detailed correctness/agreement and routing-decomposition tables as supporting evidence unless conditional routing is central to the narrative. Do not reduce table fonts below a readable conference-paper standard.

## Table notes

- All systems use the same locked 3,668-sample split.
- Bootstrap intervals are paired and ground-truth-class stratified, with 10,000 replicates and seed 42.
- Per-class comparisons are exploratory; no class-wise p-values or multiplicity-adjusted class-wise inference were produced.
- SCC uncertainty is high because support is 94.
- Routing rows include explicit denominators and must not be described collectively as error categories.
"""
    paper_text = f"""# Phase 07 Paper-Ready Results and Discussion

## Results

{results_paragraph}

## Discussion

{discussion}
"""
    reports.mkdir(parents=True, exist_ok=True)
    human = [
        reports / "phase07_gate04_independent_evidence_review.md",
        reports / "phase07_claims_lock.md",
        reports / "phase07_paper_table_recommendations.md",
        reports / "phase07_paper_ready_results_text.md",
    ]
    for path, text in zip(human, (review_md, claims_md, tables_md, paper_text)):
        path.write_text(text, encoding="utf-8", newline="\n")
    return outputs + human
