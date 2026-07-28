"""Run frozen Phase 07 statistics using only committed stored predictions."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.analysis.stored_prediction_statistics import (  # noqa: E402
    CLASSES,
    StatisticalAnalysisError,
    build_analysis,
    confusion_frame,
    environment_versions,
    metric_rows,
    sha256_file,
    verify_archive_member,
    verify_hash,
    write_csv,
    write_json,
)


PROTOCOL_HASH = "efaace517733ae7c91d2284bb4a7ca55fa7f8052790f75ebb146256ba7d8a73f"
HIERARCHICAL_HASH = "391557deb9a1aeb9b9f97edc9d3d38759e597d56b54bfdbab9ea7482451a221a"
ARCHIVE_HASH = "b76762b53a35a8d9b0aa96621d78ea0e4421aa6e8052d068ffc10648a4e63e91"
MEMBER_HASH = "08b3462549210ed7f2330a687c37a6de4e013e00185fadc3167aa980995e497d"
MANIFEST_HASH = "d53e8581a95661de0446961b81458bc17295efe9c6a513c0225e442a281bf941"
CANONICAL_MEMBER = (
    "runs/phase06c/selected_flat_internal_test/locked_primary_evaluation/"
    "internal_test_predictions.csv"
)
GATE3_GENERATED_FILES = (
    "statistical_analysis_results.json",
    "model_metric_point_estimates.csv",
    "bootstrap_confidence_intervals.csv",
    "paired_difference_summary.csv",
    "per_class_metric_summary.csv",
    "confusion_matrix_flat.csv",
    "confusion_matrix_hierarchical.csv",
    "paired_correctness_table.csv",
    "prediction_agreement_summary.csv",
    "prediction_transition_matrix.csv",
    "ground_truth_error_transitions.csv",
    "paired_sample_error_categories.csv",
    "scc_error_analysis.csv",
    "hierarchical_routing_decomposition.csv",
    "bootstrap_replicates.csv",
    "statistical_execution_manifest.json",
)
CANONICAL_COMMAND = (
    ".\\.venv\\Scripts\\python.exe scripts/run_phase07_statistical_analysis.py "
    "--output-directory reports/phase07/generated "
    "--control-directory reports/phase07/control "
    "--report-path reports/phase07/phase07_statistical_analysis_results.md"
)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _report(result: dict[str, object], provenance: dict[str, object], command: str) -> str:
    ci = result["confidence_intervals"].set_index("estimand")
    flat = result["flat_metrics"]
    hierarchical = result["hierarchical_metrics"]
    mcnemar = result["mcnemar"]
    primary = ci.loc["difference_macro_f1"]
    interpretation = (
        "The observed macro-F1 difference was statistically distinguishable "
        "under the prespecified paired bootstrap protocol."
        if primary["lower"] > 0 or primary["upper"] < 0
        else "The analysis did not establish a statistically distinguishable "
        "macro-F1 difference."
    )
    lines = [
        "# Phase 07 Stored-Prediction Statistical Analysis",
        "",
        "## Provenance and scope",
        "",
        f"- Samples: 3,668; support: non_malignant 2,398, melanoma 678, bcc 498, scc 94.",
        f"- Amended protocol SHA-256: `{provenance['protocol_sha256']}`.",
        f"- Phase 05 predictions SHA-256: `{provenance['hierarchical_sha256']}`.",
        f"- Phase 06C archive SHA-256: `{provenance['archive_sha256']}`.",
        f"- Phase 06C member SHA-256: `{provenance['member_sha256']}`.",
        f"- Paired manifest SHA-256: `{provenance['paired_manifest_sha256']}`.",
        f"- Environment: `{json.dumps(provenance['environment'], sort_keys=True)}`.",
        "",
        "The Phase 06C member was read only after safe archive validation: exactly one "
        "regular canonical member, with links, absolute paths, traversal and duplicates rejected.",
        "",
        "## Point estimates and paired intervals",
        "",
        f"- Flat macro-F1: {flat.macro_f1:.6f}, 95% CI "
        f"[{ci.loc['flat_macro_f1','lower']:.6f}, {ci.loc['flat_macro_f1','upper']:.6f}].",
        f"- Hierarchical macro-F1: {hierarchical.macro_f1:.6f}, 95% CI "
        f"[{ci.loc['hierarchical_macro_f1','lower']:.6f}, {ci.loc['hierarchical_macro_f1','upper']:.6f}].",
        f"- Flat minus hierarchical macro-F1: {primary['point_estimate']:.6f}, "
        f"paired 95% CI [{primary['lower']:.6f}, {primary['upper']:.6f}].",
        f"- {interpretation}",
        f"- Flat accuracy: {flat.accuracy:.6f}; hierarchical accuracy: {hierarchical.accuracy:.6f}; "
        f"paired difference CI [{ci.loc['difference_accuracy','lower']:.6f}, "
        f"{ci.loc['difference_accuracy','upper']:.6f}].",
        f"- Flat balanced accuracy: {flat.balanced_accuracy:.6f}; hierarchical balanced accuracy: "
        f"{hierarchical.balanced_accuracy:.6f}; paired difference CI "
        f"[{ci.loc['difference_balanced_accuracy','lower']:.6f}, "
        f"{ci.loc['difference_balanced_accuracy','upper']:.6f}].",
        "",
        "Point estimates use the complete original sample. Intervals use all 10,000 paired, "
        "ground-truth-stratified replicates, seed 42, and explicit NumPy `method=\"linear\"` "
        "on unrounded float64 values.",
        "",
        "## Paired correctness and McNemar",
        "",
        f"- Both correct: {mcnemar['both_correct']}; flat only: "
        f"{mcnemar['flat_correct_hierarchy_wrong']}; hierarchy only: "
        f"{mcnemar['flat_wrong_hierarchy_correct']}; both wrong: {mcnemar['both_wrong']}.",
        f"- Exact two-sided McNemar p-value: {mcnemar['exact_two_sided_p_value']:.17g}.",
        f"- Net paired correctness advantage: {mcnemar['net_paired_correctness_advantage']:.17g}.",
        f"- Raw discordant-pair odds ratio: "
        f"`{json.dumps(mcnemar['raw_discordant_pair_odds_ratio'], sort_keys=True)}`.",
        "",
        "## Per-class, SCC, agreement and routing",
        "",
        "Per-class precision, recall, F1, support and descriptive paired intervals are in "
        "`per_class_metric_summary.csv` and `bootstrap_confidence_intervals.csv`. No class-wise "
        "inferential p-values were calculated.",
        "",
        "SCC results are descriptive and uncertainty is high because support is only 94. "
        "Complete SCC counts and metrics are in `scc_error_analysis.csv`.",
        "",
        "Complete agreement, transition and ground-truth-stratified error counts are provided "
        "in the generated CSV outputs; no samples were cherry-picked.",
        "",
        "Routing decomposition uses stored columns `final_target_index`, "
        "`stage_1_predicted_index`, `stage_2_executed`, `stage_2_predicted_index`, and "
        "`predicted_gate_correct`. Structural Stage 2 missingness is kept separate from "
        "anomalous missingness.",
        "",
        "## Interpretation boundaries and reproducibility",
        "",
        "Accuracy, balanced accuracy and McNemar are secondary; per-class effects are "
        "exploratory descriptive comparisons. Results concern one locked ISIC 2019 "
        "internal-test split and do not establish cross-dataset or population generalization.",
        "",
        "Clinical superiority, clinical validation, improved diagnosis, mortality reduction, "
        "deployment readiness, equivalence, non-inferiority and causal claims are prohibited.",
        "",
        f"Exact command: `{command}`",
        "",
        "No training, inference, evaluation rerun, checkpoint loading, model initialization, "
        "or GPU work occurred.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=Path("reports/phase07/generated"))
    parser.add_argument("--control-directory", type=Path, default=Path("reports/phase07/control"))
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/phase07/phase07_statistical_analysis_results.md"),
    )
    args = parser.parse_args()
    command = CANONICAL_COMMAND
    output = args.output_directory
    control = args.control_directory
    report_path = args.report_path
    repository_root = REPOSITORY_ROOT.resolve()
    locked_root = (repository_root / "runs").resolve()
    for destination in (output.resolve(), control.resolve(), report_path.resolve()):
        if destination == locked_root or locked_root in destination.parents:
            raise StatisticalAnalysisError(
                f"Output destination cannot be inside locked runs: {destination}"
            )
    output.mkdir(parents=True, exist_ok=True)
    control.mkdir(parents=True, exist_ok=True)
    (control / "statistical_analysis_started_at_utc.txt").write_text(
        datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
    )

    protocol = Path("reports/phase07/generated/statistical_protocol_lock.json")
    hierarchical_path = Path(
        "runs/phase05_hierarchical_internal_test/locked_primary_evaluation/"
        "per_image_hierarchical_predictions.csv"
    )
    archive_path = Path(
        "runs/backups/phase06c/phase06c_selected_flat_internal_test_550e7cdb1144.tar.gz"
    )
    paired_path = Path("reports/phase07/generated/paired_prediction_manifest.csv")
    verify_hash(protocol, PROTOCOL_HASH, "Amended protocol lock")
    verify_hash(hierarchical_path, HIERARCHICAL_HASH, "Phase 05 predictions")
    verify_hash(paired_path, MANIFEST_HASH, "Paired manifest")
    verify_archive_member(archive_path, ARCHIVE_HASH, CANONICAL_MEMBER, MEMBER_HASH)

    paired = pd.read_csv(paired_path)
    routing = pd.read_csv(hierarchical_path)
    result = build_analysis(paired, routing, replicate_count=10000, seed=42)

    flat = result["flat_metrics"]
    hierarchical = result["hierarchical_metrics"]
    tolerance = 1e-10
    expected = {
        "flat_accuracy": (flat.accuracy, 0.7420937840785169),
        "flat_balanced_accuracy": (flat.balanced_accuracy, 0.6503125394090663),
        "flat_macro_f1": (flat.macro_f1, 0.6192224685168973),
        "flat_weighted_f1": (flat.weighted_f1, 0.7525567213826209),
        "hierarchical_accuracy": (hierarchical.accuracy, 0.740185387131952),
        "hierarchical_balanced_accuracy": (hierarchical.balanced_accuracy, 0.6311989238877984),
        "hierarchical_macro_f1": (hierarchical.macro_f1, 0.6053674005561019),
        "hierarchical_weighted_f1": (hierarchical.weighted_f1, 0.7503315694435969),
    }
    for name, (observed, locked) in expected.items():
        if abs(observed - locked) > tolerance:
            raise StatisticalAnalysisError(
                f"Locked point-estimate mismatch for {name}: {observed} versus {locked}."
            )

    point_rows = metric_rows("flat", flat) + metric_rows("hierarchical", hierarchical)
    write_csv(output / "model_metric_point_estimates.csv", pd.DataFrame(point_rows))
    write_csv(output / "bootstrap_replicates.csv", result["replicates"])
    write_csv(output / "bootstrap_confidence_intervals.csv", result["confidence_intervals"])
    write_csv(
        output / "paired_difference_summary.csv",
        result["confidence_intervals"][
            result["confidence_intervals"]["estimand"].str.startswith("difference_")
        ],
    )
    write_csv(
        output / "per_class_metric_summary.csv",
        pd.DataFrame(point_rows).query("`class` != 'overall'"),
    )
    write_csv(output / "confusion_matrix_flat.csv", confusion_frame(flat))
    write_csv(output / "confusion_matrix_hierarchical.csv", confusion_frame(hierarchical))
    write_csv(output / "paired_correctness_table.csv", pd.DataFrame([result["mcnemar"]]).drop(columns=["raw_discordant_pair_odds_ratio"]))
    write_csv(output / "prediction_agreement_summary.csv", result["agreement"])
    write_csv(output / "prediction_transition_matrix.csv", result["transitions"])
    write_csv(output / "ground_truth_error_transitions.csv", result["error_transitions"])
    write_csv(output / "paired_sample_error_categories.csv", result["sample_categories"])
    write_csv(output / "scc_error_analysis.csv", result["scc"])
    write_csv(output / "hierarchical_routing_decomposition.csv", result["routing"])

    implementation_commit = _git("rev-parse", "HEAD")
    provenance = {
        "protocol_path": protocol.as_posix(),
        "protocol_sha256": PROTOCOL_HASH,
        "hierarchical_path": hierarchical_path.as_posix(),
        "hierarchical_sha256": HIERARCHICAL_HASH,
        "archive_path": archive_path.as_posix(),
        "archive_sha256": ARCHIVE_HASH,
        "member_path": CANONICAL_MEMBER,
        "member_sha256": MEMBER_HASH,
        "paired_manifest_path": paired_path.as_posix(),
        "paired_manifest_sha256": MANIFEST_HASH,
        "implementation_commit": implementation_commit,
        "environment": environment_versions(),
        "seed": 42,
        "replicate_count": 10000,
        "quantile_method": "linear",
    }
    primary = result["confidence_intervals"].set_index("estimand").loc["difference_macro_f1"]
    payload = {
        "provenance": provenance,
        "point_estimates": result["point_lookup"],
        "primary": {
            "estimand": "flat_minus_hierarchical_macro_f1",
            "point_estimate": primary["point_estimate"],
            "lower": primary["lower"],
            "upper": primary["upper"],
            "includes_zero": bool(primary["lower"] <= 0 <= primary["upper"]),
        },
        "mcnemar": result["mcnemar"],
        "scc_support_limitation": "Uncertainty is high because SCC support is only 94.",
        "multiplicity": "One primary estimand; model-level secondary; per-class exploratory descriptive only.",
    }
    write_json(output / "statistical_analysis_results.json", payload)
    write_json(output / "statistical_execution_manifest.json", provenance)

    deterministic = [output / name for name in GATE3_GENERATED_FILES]
    if not all(path.is_file() for path in deterministic):
        raise StatisticalAnalysisError("A required deterministic output is missing.")
    manifest_text = "".join(
        f"{sha256_file(path)}  {path.name}\n" for path in deterministic
    )
    (output / "artifact_sha256_manifest.txt").write_text(
        manifest_text, encoding="utf-8", newline="\n"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report(result, provenance, command), encoding="utf-8", newline="\n")

    write_json(control / "execution_environment.json", environment_versions())
    (control / "exact_command.txt").write_text(command + "\n", encoding="utf-8")
    (control / "implementation_git_commit.txt").write_text(
        implementation_commit + "\n", encoding="utf-8"
    )
    (control / "execution_git_status.txt").write_text(
        _git("status", "--short") + "\n", encoding="utf-8"
    )
    (control / "statistical_analysis_completed_at_utc.txt").write_text(
        datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
    )
    print("Phase 07 stored-prediction statistical analysis completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
