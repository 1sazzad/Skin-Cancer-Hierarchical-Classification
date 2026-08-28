"""Generate Phase 11 paired statistical evidence from locked predictions."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.densenet121_paired_statistics import (  # noqa: E402
    METRICS,
    align_predictions,
    paired_comparison,
)
from src.analysis.stored_prediction_statistics import (  # noqa: E402
    CLASSES,
    environment_versions,
    sha256_file,
    write_csv,
    write_json,
)

DENSE_PATH = Path("experiments/evaluations/phase11_densenet121_internal_test_seed42__best_epoch04/internal_test_predictions.csv")
MANIFEST_PATH = Path("reports/phase07/generated/paired_prediction_manifest.csv")
HIERARCHY_PATH = Path("runs/phase05_hierarchical_internal_test/locked_primary_evaluation/per_image_hierarchical_predictions.csv")
OUTPUT = Path("reports/phase11/generated")
REPORT = Path("reports/phase11/phase11_densenet121_paired_statistical_analysis.md")


def main() -> int:
    source_hashes = {path.as_posix(): sha256_file(path) for path in (DENSE_PATH, MANIFEST_PATH, HIERARCHY_PATH)}
    aligned = align_predictions(
        pd.read_csv(DENSE_PATH), pd.read_csv(MANIFEST_PATH), pd.read_csv(HIERARCHY_PATH)
    )
    target = aligned["target_index"].to_numpy()
    dense = aligned["densenet_predicted_index"].to_numpy()
    results = {
        "flat_efficientnet_b0": paired_comparison(
            target, dense, aligned["flat_predicted_index"].to_numpy(),
            comparator_name="flat_efficientnet_b0",
        ),
        "predicted_gate_hierarchy": paired_comparison(
            target, dense, aligned["hierarchy_predicted_index"].to_numpy(),
            comparator_name="predicted_gate_hierarchy",
        ),
    }
    differences = pd.DataFrame(
        [row for result in results.values() for row in result["differences"]]
    )
    mcnemar = pd.DataFrame([result["mcnemar"] for result in results.values()])
    dense_metrics = results["flat_efficientnet_b0"]["densenet_metrics"]
    per_class = pd.DataFrame(
        [
            {
                "class_name": name,
                "precision": dense_metrics.precision[index],
                "recall": dense_metrics.recall[index],
                "f1": dense_metrics.f1[index],
                "support": int(dense_metrics.support[index]),
            }
            for index, name in enumerate(CLASSES)
        ]
    )
    payload = {
        "provenance": {
            "prediction_files": source_hashes,
            "environment": environment_versions(),
            "sample_count": len(aligned),
            "class_order": list(CLASSES),
            "ground_truth_support": dict(zip(CLASSES, map(int, dense_metrics.support))),
        },
        "method": {
            "bootstrap": "paired ground-truth-class-stratified percentile bootstrap",
            "replicate_count": 10000,
            "seed": 42,
            "confidence_level": 0.95,
            "quantile_method": "numpy linear",
            "mcnemar": "exact two-sided binomial test on discordant correctness pairs",
            "difference_direction": "DenseNet-121 minus comparator",
        },
        "comparisons": {
            name: {
                "metrics": {row["metric"]: row for row in result["differences"]},
                "mcnemar": result["mcnemar"],
            }
            for name, result in results.items()
        },
        "densenet121_per_class": per_class.to_dict(orient="records"),
    }
    write_json(OUTPUT / "densenet121_paired_statistics.json", payload)
    write_csv(OUTPUT / "densenet121_paired_metric_differences.csv", differences)
    write_csv(OUTPUT / "densenet121_mcnemar_results.csv", mcnemar)
    write_csv(OUTPUT / "densenet121_per_class_metrics.csv", per_class)

    lines = [
        "# Phase 11 DenseNet-121 Paired Statistical Analysis", "",
        "## Evidence and alignment", "",
        "No training or inference was performed. The analysis used these locked stored predictions:",
        "",
        *[f"- `{path}` — SHA-256 `{digest}`" for path, digest in source_hashes.items()],
        "",
        f"All three sources contained {len(aligned):,} unique matching sample IDs. Missing IDs, "
        "duplicates, target mismatches, unsupported labels, and non-finite DenseNet probabilities "
        "were absent. Pairing was by sample ID after stable sorting, never CSV row order.",
        "",
        "Ground-truth support was non-malignant 2,398; melanoma 678; BCC 498; SCC 94.",
        "",
        "## Methods", "",
        "Metrics use the fixed endpoint order `non_malignant, melanoma, bcc, scc` and "
        "zero-division value 0. Confidence intervals are percentile 95% intervals from "
        "10,000 paired bootstrap replicates (seed 42), resampling with replacement within "
        "each true endpoint class and preserving class support. Quantiles use unrounded "
        "float64 values and NumPy's linear method. McNemar p-values are exact, two-sided "
        "binomial tests over discordant paired correctness outcomes.",
        "",
        "## Results", "",
    ]
    for name, label in (
        ("flat_efficientnet_b0", "Flat EfficientNet-B0"),
        ("predicted_gate_hierarchy", "Predicted-gate hierarchy"),
    ):
        result = results[name]
        lines += [f"### DenseNet-121 versus {label}", "",
                  "| Metric | DenseNet-121 | Comparator | Difference | Paired 95% CI |",
                  "|---|---:|---:|---:|---:|"]
        for row in result["differences"]:
            lines.append(
                f"| {row['metric'].replace('_', ' ').title()} | {row['densenet121']:.6f} | "
                f"{row['comparator']:.6f} | {row['difference_densenet121_minus_comparator']:+.6f} | "
                f"[{row['ci_lower']:+.6f}, {row['ci_upper']:+.6f}] |"
            )
        mc = result["mcnemar"]
        lines += ["", f"Both correct: {mc['both_correct']}; DenseNet-only correct: "
                  f"{mc['densenet121_only_correct']}; comparator-only correct: "
                  f"{mc['comparator_only_correct']}; both incorrect: {mc['both_incorrect']}. "
                  f"Exact two-sided McNemar p = {mc['exact_two_sided_p_value']:.17g}.", ""]
    lines += [
        "## DenseNet-121 per-class verification", "",
        "| Class | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|",
        *[
            f"| {row.class_name} | {row.precision:.6f} | {row.recall:.6f} | "
            f"{row.f1:.6f} | {int(row.support)} |"
            for row in per_class.itertuples()
        ],
        "", "SCC recall reproduces the locked value: 28/94 = 0.297872.", "",
        "## Interpretation", "",
        "A confidence interval excluding zero supports a difference for that metric under "
        "this prespecified resampling analysis; McNemar tests only paired accuracy/correctness. "
        "The results do not establish clinical superiority, equivalence, non-inferiority, or "
        "external generalization. SCC conclusions remain imprecise because support is only 94.",
        "", "## Manuscript-ready factual values", "",
        "DenseNet-121 achieved accuracy 0.791439, balanced accuracy 0.616828, macro-F1 "
        "0.635107, and weighted-F1 0.786162 on the locked 3,668-image internal test set. "
        "The tables above provide DenseNet-minus-comparator paired differences, percentile "
        "95% confidence intervals, and exact two-sided McNemar results for both comparisons.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("Phase 11 DenseNet-121 paired statistical analysis completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
