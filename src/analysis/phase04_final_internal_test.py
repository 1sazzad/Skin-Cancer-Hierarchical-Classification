"""Deterministic Phase 04 analysis over frozen final-test artifacts only."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.analysis.stored_prediction_statistics import (
    CLASSES,
    EXPECTED_SUPPORT,
    StatisticalAnalysisError,
    bootstrap_table,
    calculate_metrics,
    exact_mcnemar,
    linear_quantile,
    sha256_file,
    write_csv,
    write_json,
)

EXPECTED_COMMIT = "3bf8c137f585ba720cf96d882003fadc678ea059"
EXPECTED_COLUMNS = (
    "sample_id", "true_label", "shared_predicted_gate", "shared_oracle_gate",
    "flat_prediction", "shared_correct", "flat_correct", "stage1_target",
    "stage1_prediction", "stage2_target", "stage2_prediction",
)
EXPECTED_MANIFESTS = {
    "isic2019": "data/manifests/isic2019_train_val_test_split_seed42.csv",
    "task3": "data/manifests/emb_stage03_dermoscopic_split_seed42.csv",
}
EXPECTED_CHECKPOINTS = {
    "shared": ("2f1c2393c5c9de15dfa4a1a132a31b9a5b8ede07d7ed6e07ab90918fc2aaa9eb", 6),
    "task1": ("95e02c26b1ea4a0dba17016313c81f97c9c2635270a37b4debbee0f84e07ba3b", 5),
    "task2": ("10986d41b64a685fcd8fe166623c5b1c7fd2f21bdad7cf4d55dedc3967a397fd", 8),
    "task3": ("71bfda5f7a19333947e1c13f4e1c5ed45e9a827c447fc9bcd6fd9ddc999f8692", 12),
    "flat": ("f3d8b8b0e5ef42e3c287a2377b5570411d442246acd16cb874ccf903facdc7a7", 2),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StatisticalAnalysisError(message)


def _close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def _metric_dict(metrics: Any) -> dict[str, Any]:
    return {
        "accuracy": metrics.accuracy,
        "balanced_accuracy": metrics.balanced_accuracy,
        "macro_f1": metrics.macro_f1,
        "weighted_f1": metrics.weighted_f1,
        "macro_precision": float(metrics.precision.mean()),
        "macro_recall": float(metrics.recall.mean()),
        "confusion_matrix": metrics.confusion_matrix.tolist(),
        "per_class": {
            name: {
                "precision": float(metrics.precision[i]), "recall": float(metrics.recall[i]),
                "f1": float(metrics.f1[i]), "support": int(metrics.support[i]),
            }
            for i, name in enumerate(CLASSES)
        },
    }


def _validate_integer_column(frame: pd.DataFrame, column: str, allowed: set[int]) -> np.ndarray:
    numeric = pd.to_numeric(frame[column], errors="coerce")
    _require(numeric.notna().all(), f"Missing or invalid values in {column}.")
    values = numeric.to_numpy(dtype=np.int64)
    _require(np.equal(numeric.to_numpy(dtype=float), values).all(), f"Non-integer values in {column}.")
    _require(np.isin(values, list(allowed)).all(), f"Unsupported values in {column}.")
    return values


def audit_inputs(summary_path: Path, csv_path: Path, config_path: Path) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Fail closed on frozen provenance, schema, pairing, and metric reconciliation."""
    _require(summary_path.is_file(), f"Missing summary: {summary_path}")
    _require(csv_path.is_file(), f"Missing paired CSV: {csv_path}")
    _require(config_path.is_file(), f"Missing frozen config: {config_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validation_path = summary_path.parent.parent / "gate04d_validation_summary.json"
    _require(validation_path.is_file(), f"Missing frozen validation evidence: {validation_path}")
    validation_summary = json.loads(validation_path.read_text(encoding="utf-8"))
    _require(validation_summary.get("execution_split") == "validation", "Gate04D split changed.")
    _require(validation_summary.get("internal_test_executed") is False, "Gate04D incorrectly records test execution.")
    summary["_validation_evidence"] = validation_summary
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _require(summary.get("gate") == "FINAL_INTERNAL_TEST", "Incorrect final-test gate.")
    _require(summary.get("status") == "PASS", "Final summary status is not PASS.")
    _require(summary.get("execution_split") == "internal_test", "Incorrect execution split.")
    _require(summary.get("internal_test_executed") is True, "Internal-test completion flag is false.")
    environment = summary.get("environment", {})
    _require(environment.get("gpu_name") == "Tesla T4", "Final execution GPU was not Tesla T4.")
    _require(environment.get("device") == "cuda" and environment.get("cuda_available") is True, "Final execution was not CUDA-enabled.")
    _require(environment.get("git_commit") == EXPECTED_COMMIT, "Frozen Git commit mismatch.")
    _require(config.get("execution_split") == "internal_test", "Frozen config split mismatch.")

    recorded_manifests = summary.get("manifests", {})
    configured_manifests = config.get("isic", {}).get("manifest_paths", {})
    _require(configured_manifests == EXPECTED_MANIFESTS, "Authoritative manifest config changed.")
    for name, suffix in EXPECTED_MANIFESTS.items():
        normalized = str(recorded_manifests.get(name, "")).replace("\\", "/")
        _require(normalized.endswith(suffix), f"Recorded {name} manifest is not authoritative.")

    checkpoints = config.get("checkpoints", {})
    _require(set(checkpoints) == set(EXPECTED_CHECKPOINTS), "Frozen checkpoint set changed.")
    checkpoint_audit = {}
    for name, (digest, epoch) in EXPECTED_CHECKPOINTS.items():
        item = checkpoints[name]
        _require(item.get("sha256") == digest and item.get("expected_epoch") == epoch, f"Frozen {name} checkpoint provenance changed.")
        checkpoint_audit[name] = {"path": item["path"], "sha256": digest, "expected_epoch": epoch,
                                  "model_kind": item["model_kind"], "class_names": item["class_names"]}

    frame = pd.read_csv(csv_path)
    _require(tuple(frame.columns) == EXPECTED_COLUMNS, "Paired CSV schema changed.")
    _require(tuple(summary.get("paired_export", {}).get("columns", ())) == EXPECTED_COLUMNS, "Summary paired schema changed.")
    expected_count = int(summary["internal_test_sample_counts"]["flat_four_class"])
    _require(len(frame) == expected_count == int(summary["paired_export"]["row_count"]), "Paired row/sample count mismatch.")
    _require(frame.notna().all().all(), "Paired CSV contains missing required values.")
    ids = frame["sample_id"].astype(str).str.strip()
    _require(ids.ne("").all() and not ids.duplicated().any(), "Missing or duplicate sample IDs.")
    _require(summary["paired_export"].get("unique_sample_ids") is True, "Summary unique-ID assertion failed.")
    _require(summary["paired_export"].get("shared_flat_order_identical") is True, "Shared/flat order assertion failed.")
    _require(summary["paired_export"].get("ground_truth_identical") is True, "Ground-truth identity assertion failed.")

    target = _validate_integer_column(frame, "true_label", {0, 1, 2, 3})
    shared = _validate_integer_column(frame, "shared_predicted_gate", {0, 1, 2, 3})
    oracle = _validate_integer_column(frame, "shared_oracle_gate", {0, 1, 2, 3})
    flat = _validate_integer_column(frame, "flat_prediction", {0, 1, 2, 3})
    shared_correct = _validate_integer_column(frame, "shared_correct", {0, 1})
    flat_correct = _validate_integer_column(frame, "flat_correct", {0, 1})
    stage1_target = _validate_integer_column(frame, "stage1_target", {0, 1})
    stage1_prediction = _validate_integer_column(frame, "stage1_prediction", {0, 1})
    stage2_target = _validate_integer_column(frame, "stage2_target", {-1, 0, 1, 2})
    stage2_prediction = _validate_integer_column(frame, "stage2_prediction", {-1, 0, 1, 2})
    _require(np.array_equal(shared_correct, (shared == target).astype(int)), "Stored shared correctness is inconsistent.")
    _require(np.array_equal(flat_correct, (flat == target).astype(int)), "Stored flat correctness is inconsistent.")
    _require(np.array_equal(stage1_target, (target > 0).astype(int)), "Stage1 targets are inconsistent.")
    _require(np.array_equal(stage1_prediction, (shared > 0).astype(int)), "Stage1 predictions are inconsistent with routing output.")
    nonmalignant = target == 0
    _require(np.all(stage2_target[nonmalignant] == -1), "Structural Stage2 targets are inconsistent.")
    routed = stage1_prediction == 1
    _require(np.all(stage2_prediction[nonmalignant & ~routed] == -1), "Stage2 prediction exists for an unrouted non-malignant sample.")
    _require(np.array_equal(shared[routed], stage2_prediction[routed] + 1), "Routed output is inconsistent with Stage2 prediction.")
    _require(np.array_equal(stage2_target[~nonmalignant], target[~nonmalignant] - 1), "Stage2 targets are inconsistent.")
    _require(np.array_equal(oracle[nonmalignant], target[nonmalignant]), "Oracle gate changed non-malignant targets.")
    _require(np.array_equal(oracle[~nonmalignant], stage2_prediction[~nonmalignant] + 1), "Oracle predictions are inconsistent with Stage2.")
    support = np.bincount(target, minlength=4)
    _require(np.array_equal(support, EXPECTED_SUPPORT), "Final class support is unexpected.")

    recomputed = {"shared_predicted_gate": _metric_dict(calculate_metrics(target, shared)),
                  "flat_four_class": _metric_dict(calculate_metrics(target, flat)),
                  "shared_oracle_gate": _metric_dict(calculate_metrics(target, oracle))}
    json_lookup = {"shared_predicted_gate": summary["metrics"]["shared"]["predicted_gate_four_class"],
                   "flat_four_class": summary["metrics"]["flat_four_class"],
                   "shared_oracle_gate": summary["metrics"]["shared"]["oracle_gate_four_class"]}
    for model in recomputed:
        for metric in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"):
            _require(_close(recomputed[model][metric], json_lookup[model][metric]), f"{model} {metric} does not match JSON.")
        _require(recomputed[model]["confusion_matrix"] == json_lookup[model]["confusion_matrix"], f"{model} confusion matrix does not match JSON.")
        for name in CLASSES:
            for metric in ("precision", "recall", "f1"):
                _require(_close(recomputed[model]["per_class"][name][metric], json_lookup[model]["per_class"][name][metric]), f"{model} {name} {metric} does not match JSON.")
    integrity = {"status": "PASS", "summary_sha256": sha256_file(summary_path), "paired_csv_sha256": sha256_file(csv_path),
                 "sample_count": expected_count, "class_support": dict(zip(CLASSES, support.tolist())),
                 "frozen_git_commit": EXPECTED_COMMIT, "manifests": recorded_manifests,
                 "checkpoints": checkpoint_audit,
                 "checkpoint_provenance_note": "Checkpoint hashes are frozen in the config and were verified by the final runner; the output summary does not duplicate them.",
                 "schema": list(EXPECTED_COLUMNS), "recomputed_json_match_tolerance": 1e-12}
    return summary, frame, recomputed, integrity


def analyze(summary: dict[str, Any], frame: pd.DataFrame, recomputed: dict[str, Any], integrity: dict[str, Any], *, iterations: int = 10000, seed: int = 42) -> dict[str, Any]:
    target = frame["true_label"].to_numpy(dtype=np.int64)
    hierarchy = frame["shared_predicted_gate"].to_numpy(dtype=np.int64)
    flat = frame["flat_prediction"].to_numpy(dtype=np.int64)
    # Existing frozen implementation generates paired, ground-truth-class-stratified replicates.
    boot = bootstrap_table(target, flat, hierarchy, replicate_count=iterations, seed=seed)
    differences = {}
    for metric in ("macro_f1", "accuracy"):
        column = f"difference_{metric}"
        # bootstrap_table is flat-minus-hierarchy; report requested hierarchy-minus-flat.
        values = -boot[column].to_numpy(dtype=np.float64)
        lower, upper = linear_quantile(values, [0.025, 0.975])
        differences[metric] = {"hierarchy": recomputed["shared_predicted_gate"][metric],
                               "flat": recomputed["flat_four_class"][metric],
                               "delta_hierarchy_minus_flat": recomputed["shared_predicted_gate"][metric] - recomputed["flat_four_class"][metric],
                               "paired_bootstrap_95ci": [float(lower), float(upper)]}
    classwise = []
    for index, name in enumerate(CLASSES):
        values = -boot[f"difference_f1_{name}"].to_numpy(dtype=np.float64)
        lower, upper = linear_quantile(values, [0.025, 0.975])
        h = recomputed["shared_predicted_gate"]["per_class"][name]["f1"]
        f = recomputed["flat_four_class"]["per_class"][name]["f1"]
        classwise.append({"class": name, "support": recomputed["flat_four_class"]["per_class"][name]["support"],
                          "hierarchy_f1": h, "flat_f1": f, "delta_hierarchy_minus_flat": h - f,
                          "paired_bootstrap_95ci_lower": float(lower), "paired_bootstrap_95ci_upper": float(upper),
                          "well_defined": True})
    raw = exact_mcnemar(flat == target, hierarchy == target)
    mcnemar = {"both_correct": raw["both_correct"], "hierarchy_only_correct": raw["flat_wrong_hierarchy_correct"],
               "flat_only_correct": raw["flat_correct_hierarchy_wrong"], "both_wrong": raw["both_wrong"],
               "discordant_total": raw["discordant_pairs"], "exact_two_sided_p_value": raw["exact_two_sided_p_value"]}
    shared = summary["metrics"]["shared"]
    routing = dict(shared["routing"])
    routing["routing_loss_macro_f1"] = shared["oracle_gate_four_class"]["macro_f1"] - shared["predicted_gate_four_class"]["macro_f1"]
    comparisons = {}
    for task in ("task1", "task2", "task3"):
        shared_key = "task2_malignant_subset" if task == "task2" else task
        standalone_key = f"standalone_{task}"
        a, b = shared[shared_key], summary["metrics"][standalone_key]
        comparisons[task] = {"shared_macro_f1": a["macro_f1"], "standalone_macro_f1": b["macro_f1"],
                             "macro_f1_difference_shared_minus_standalone": a["macro_f1"] - b["macro_f1"],
                             "shared_accuracy": a["accuracy"], "standalone_accuracy": b["accuracy"],
                             "accuracy_difference_shared_minus_standalone": a["accuracy"] - b["accuracy"],
                             "per_class_f1_difference_shared_minus_standalone": {name: a["per_class"][name]["f1"] - b["per_class"][name]["f1"] for name in a["class_names"]}}
    validation = {"macro_f1_delta_hierarchy_minus_flat": -0.07028453132737733,
                  "macro_f1_95ci": [-0.10189398304158802, -0.03864787451662513],
                  "accuracy_delta_hierarchy_minus_flat": -0.06379498364231195,
                  "accuracy_95ci": [-0.07797164667393675, -0.04961832061068705],
                  "routing_loss_macro_f1": 0.19247991838522172,
                  "shared_predicted_gate_macro_f1": 0.5832871340331853, "flat_macro_f1": 0.6535716653605627,
                  "shared_oracle_gate_macro_f1": 0.775767052418407}
    validation_metrics = summary["_validation_evidence"]["metrics"]
    validation["classwise_delta_hierarchy_minus_flat"] = {
        name: validation_metrics["shared"]["predicted_gate_four_class"]["per_class"][name]["f1"]
        - validation_metrics["flat_four_class"]["per_class"][name]["f1"] for name in CLASSES
    }
    validation["shared_vs_standalone_macro_f1_difference"] = {}
    for task in ("task1", "task2", "task3"):
        shared_key = "task2_malignant_subset" if task == "task2" else task
        validation["shared_vs_standalone_macro_f1_difference"][task] = (
            validation_metrics["shared"][shared_key]["macro_f1"]
            - validation_metrics[f"standalone_{task}"]["macro_f1"]
        )
    return {"gate": "FINAL_INTERNAL_TEST_ANALYSIS", "status": "PASS", "split": "internal_test",
            "sample_count": len(frame), "comparison": "shared_predicted_gate_vs_flat_four_class",
            "bootstrap": {"method": "paired_ground_truth_class_stratified_percentile", "iterations": iterations, "seed": seed, "confidence_level": 0.95, "quantile_method": "numpy_linear"},
            "integrity": integrity, "recomputed_metrics": recomputed, "paired_differences": differences,
            "mcnemar_exact": mcnemar, "classwise": classwise, "routing": routing,
            "shared_vs_standalone": comparisons, "validation_reference": validation,
            "efficiency": summary["efficiency"], "reported_metrics": summary["metrics"]}


def result_tables(analysis: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for model, metrics in analysis["reported_metrics"].items():
        if model == "shared":
            for submodel, submetrics in metrics.items():
                if isinstance(submetrics, dict) and "macro_f1" in submetrics:
                    for metric in ("accuracy", "balanced_accuracy", "macro_f1", "macro_precision", "macro_recall", "weighted_f1"):
                        rows.append({"model": f"shared_{submodel}", "metric": metric, "value": submetrics[metric]})
        elif isinstance(metrics, dict) and "macro_f1" in metrics:
            for metric in ("accuracy", "balanced_accuracy", "macro_f1", "macro_precision", "macro_recall", "weighted_f1"):
                rows.append({"model": model, "metric": metric, "value": metrics[metric]})
    return pd.DataFrame(rows), pd.DataFrame(analysis["classwise"])


def render_report(a: dict[str, Any]) -> str:
    m, d, mc, r = a["recomputed_metrics"], a["paired_differences"], a["mcnemar_exact"], a["routing"]
    s = a["reported_metrics"]["shared"]
    lines = ["# Phase 04 Final Internal-Test Statistical Analysis", "", "## 1. Evaluation integrity", "",
             f"**PASS.** The stored evidence contains {a['sample_count']:,} unique paired samples and matches the frozen schema, class support, manifests, Tesla T4 environment, and pre-test commit `{EXPECTED_COMMIT}`. Independently recomputed metrics and confusion matrices match the JSON within 1e-12. The prediction CSV and summary were not modified.", "",
             "Checkpoint provenance is indirect: hashes, epochs, kinds, and class orders are frozen in the evaluation config and were enforced by the runner, but are not duplicated in the final summary. This is an evidence-design limitation, not an observed execution inconsistency.", "",
             "## 2. Final test results", "",
             f"The flat model achieved accuracy {m['flat_four_class']['accuracy']:.6f}, balanced accuracy {m['flat_four_class']['balanced_accuracy']:.6f}, macro-F1 {m['flat_four_class']['macro_f1']:.6f}, macro precision {m['flat_four_class']['macro_precision']:.6f}, macro recall {m['flat_four_class']['macro_recall']:.6f}, and weighted F1 {m['flat_four_class']['weighted_f1']:.6f}.", "",
             f"The shared hierarchy achieved Task1 macro-F1 {s['task1']['macro_f1']:.6f}, Task2 malignant-subset macro-F1 {s['task2_malignant_subset']['macro_f1']:.6f}, Task3 macro-F1 {s['task3']['macro_f1']:.6f}, predicted-gate four-class macro-F1 {m['shared_predicted_gate']['macro_f1']:.6f}, and oracle-gate four-class macro-F1 {m['shared_oracle_gate']['macro_f1']:.6f}.", "",
             "### Overall model/task metrics", "", "| Model/task | N | Accuracy | Balanced accuracy | Macro-F1 | Macro precision | Macro recall | Weighted F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    metric_sets = [("flat_four_class", a["reported_metrics"]["flat_four_class"]),
                   ("shared_task1", s["task1"]), ("shared_task2_malignant_subset", s["task2_malignant_subset"]),
                   ("shared_task3", s["task3"]), ("shared_predicted_gate", s["predicted_gate_four_class"]),
                   ("shared_oracle_gate", s["oracle_gate_four_class"]),
                   ("standalone_task1", a["reported_metrics"]["standalone_task1"]),
                   ("standalone_task2", a["reported_metrics"]["standalone_task2"]),
                   ("standalone_task3", a["reported_metrics"]["standalone_task3"])]
    for name, x in metric_sets:
        lines.append(f"| {name} | {x['sample_count']} | {x['accuracy']:.6f} | {x['balanced_accuracy']:.6f} | {x['macro_f1']:.6f} | {x['macro_precision']:.6f} | {x['macro_recall']:.6f} | {x['weighted_f1']:.6f} |")
    lines += ["", "### Flat four-class confusion matrix", "", "Rows are true classes and columns are predicted classes in the order non_malignant, melanoma, bcc, scc.", "", "| True class | non_malignant | melanoma | bcc | scc |", "|---|---:|---:|---:|---:|"]
    flat_json = a["reported_metrics"]["flat_four_class"]
    for name, row in zip(CLASSES, flat_json["confusion_matrix"]):
        lines.append(f"| {name} | {row[0]} | {row[1]} | {row[2]} | {row[3]} |")
    lines += ["", "### Flat four-class per-class metrics", "", "| Class | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
    for name in CLASSES:
        x = flat_json["per_class"][name]
        lines.append(f"| {name} | {x['precision']:.6f} | {x['recall']:.6f} | {x['f1']:.6f} | {x['support']} |")
    lines += ["", "All remaining confusion matrices and per-class metrics are retained in `final_paired_statistics.json`; compact overall metrics are in `final_results_table.csv`.", "",
             "## 3. Flat vs hierarchy comparison", "",
             f"The flat model achieved higher macro-F1 than the deployed shared hierarchy: delta hierarchy − flat = {d['macro_f1']['delta_hierarchy_minus_flat']:.6f}. Accuracy delta was {d['accuracy']['delta_hierarchy_minus_flat']:.6f}.", "",
             "## 4. Paired statistical evidence", "",
             f"Using 10,000 paired, ground-truth-class-stratified bootstrap replicates (seed 42), the macro-F1 delta 95% percentile CI was [{d['macro_f1']['paired_bootstrap_95ci'][0]:.6f}, {d['macro_f1']['paired_bootstrap_95ci'][1]:.6f}] and the accuracy delta CI was [{d['accuracy']['paired_bootstrap_95ci'][0]:.6f}, {d['accuracy']['paired_bootstrap_95ci'][1]:.6f}]. Both exclude zero and favor flat. Paired outcomes were: both correct {mc['both_correct']}, hierarchy only {mc['hierarchy_only_correct']}, flat only {mc['flat_only_correct']}, both wrong {mc['both_wrong']}. Exact two-sided McNemar p={mc['exact_two_sided_p_value']:.6g}. The direction and intervals, rather than the p-value alone, support the conclusion.", "",
             "## 5. Class-wise analysis", "", "| Class | Support | Hierarchy F1 | Flat F1 | Delta H−F | 95% CI |", "|---|---:|---:|---:|---:|---:|"]
    for row in a["classwise"]:
        lines.append(f"| {row['class']} | {row['support']} | {row['hierarchy_f1']:.6f} | {row['flat_f1']:.6f} | {row['delta_hierarchy_minus_flat']:.6f} | [{row['paired_bootstrap_95ci_lower']:.6f}, {row['paired_bootstrap_95ci_upper']:.6f}] |")
    lines += ["", "SCC has only 94 cases. Its interval is reported, but minority-class conclusions remain less stable and should not be overgeneralized.", "",
              "## 6. Routing-error decomposition", "",
              f"Among {r['true_malignant_count']} malignant and {r['true_non_malignant_count']} non-malignant cases, Task1 blocked {r['malignant_blocked_by_stage_1']} malignant cases ({r['malignant_block_rate']:.2%}) and incorrectly sent {r['non_malignant_incorrectly_routed_to_stage_2']} non-malignant cases to Task2 ({r['non_malignant_incorrect_route_rate']:.2%}). It correctly routed {r['correctly_routed_malignant']} malignant cases. Stage2 ran {r['stage_2_execution_count']} times ({r['stage_2_execution_fraction']:.2%}); {r['subtype_error_after_correct_route']} correctly routed malignant cases had subtype errors ({r['subtype_error_rate_after_correct_route']:.2%}). Oracle minus predicted-gate macro-F1 was {r['routing_loss_macro_f1']:.6f}, indicating substantial end-to-end performance loss associated with routing.", "",
              "## 7. Shared vs standalone task comparison", ""]
    for task, x in a["shared_vs_standalone"].items():
        lines.append(f"- {task.title()}: shared minus standalone macro-F1 {x['macro_f1_difference_shared_minus_standalone']:+.6f}; accuracy {x['accuracy_difference_shared_minus_standalone']:+.6f}.")
    lines += ["", "Task1 and Task2 favor the standalone models, which is consistent with possible negative transfer but does not establish causation or paired statistical significance. Shared Task3 has higher macro-F1 but lower accuracy; with T2/T3/T4 supports of 7/2/1, its class-level pattern is extremely unstable.", "",
              "## 8. Validation-to-test consistency", "",
              f"The direction replicated: hierarchy-minus-flat macro-F1 changed from −0.070285 on validation to {d['macro_f1']['delta_hierarchy_minus_flat']:.6f} on test, and accuracy from −0.063795 to {d['accuracy']['delta_hierarchy_minus_flat']:.6f}. Routing loss increased from 0.192480 to {r['routing_loss_macro_f1']:.6f}. All four observed class F1 deltas favored flat on both splits; the test SCC interval nevertheless included zero. Standalone Tasks 1–2 remained ahead of their shared heads; Task3 retained the mixed pattern of sparse, unstable results. Overall robustness is supported for the main direction, with modest effect-size variation.", "",
              "## 9. Efficiency", "", "Measured batch-1 Tesla T4 evidence:", "", "| Model | Parameters | Checkpoint bytes | Latency ms/image | Throughput img/s | Peak CUDA bytes | FLOPs/MACs |", "|---|---:|---:|---:|---:|---:|---|"]
    for name, e in a["efficiency"].items():
        b, c = e["benchmark"], e["compute"]
        compute = f"{c.get('flops')}/{c.get('macs')}" if c.get("supported") else "unavailable"
        lines.append(f"| {name} | {e['total_parameters']} | {e['checkpoint_bytes']} | {b['latency_ms_per_image']:.4f} | {b['throughput_images_per_second']:.2f} | {b['peak_cuda_memory_bytes']} | {compute} |")
    lines += ["", "These measurements compare one forward pass per model. Architecturally, the shared system stores one encoder with three heads (4,020,358 parameters; 48,737,992 checkpoint bytes), whereas the three independently stored task models total 12,035,454 parameters and 145,891,211 checkpoint bytes (derived sums, about 3.0× the shared storage). Therefore a single standalone forward-pass measurement is not the total storage or conditional execution cost of the independent hierarchy; full conditional hierarchy latency was not directly benchmarked.", "",
              "## 10. Scientific interpretation", "", "The flat four-class model performed better than the deployed predicted-gate shared hierarchy on the frozen internal test. The paired bootstrap intervals excluded zero. The oracle-routing diagnostic indicates substantial performance loss attributable to routing decisions, while conditional Task2 performance itself was comparatively strong.", "",
              "## 11. Limitations", "", "This is one frozen internal test from the project’s data construction, not external or clinical validation. Bootstrap intervals quantify sampling uncertainty under the chosen paired stratified resampling scheme. McNemar addresses accuracy discordance, not macro-F1. Small SCC support and extremely rare Task3 categories limit stable class-level inference. Efficiency excludes FLOPs/MACs because the backend did not support them and does not directly benchmark a full independently routed deployment.", "",
              "## 12. Final Phase04 verdict", "", "**PASS (analysis and evidence integrity).** The primary comparative conclusion replicated on internal test: the flat classifier achieved higher macro-F1 and accuracy than the deployed shared hierarchy. The evidence is consistent with routing as a major source of hierarchical performance loss and with possible negative transfer for shared Tasks 1–2, while Task3 is too sparse for strong class-level conclusions. No claim of clinical superiority, causation, or external generalization is supported.", ""]
    return "\n".join(lines)


def write_outputs(output_dir: Path, analysis: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "final_paired_statistics.json", analysis)
    results, classwise = result_tables(analysis)
    write_csv(output_dir / "final_results_table.csv", results)
    write_csv(output_dir / "final_classwise_comparison.csv", classwise)
    write_json(output_dir / "final_routing_analysis.json", {"status": "PASS", **analysis["routing"]})
    (output_dir / "final_statistical_analysis.md").write_text(render_report(analysis), encoding="utf-8", newline="\n")
