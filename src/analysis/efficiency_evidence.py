"""Static and stored-only Phase 07 efficiency evidence audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch

from src.analysis.stored_prediction_statistics import write_csv, write_json


GRADES = {"A", "B", "C", "U"}
PROHIBITED = [
    "real-time",
    "faster",
    "more memory-efficient",
    "energy-efficient",
    "mobile-ready",
    "lower FLOPs",
    "lower latency",
    "scalable",
    "production-ready",
]


class EfficiencyEvidenceError(ValueError):
    """Raised when efficiency evidence is unsafe or internally inconsistent."""


def routing_compute(total: int, invoked: int) -> dict[str, float | int]:
    """Derive conditional routing workload counts."""
    if total <= 0 or invoked < 0 or invoked > total:
        raise EfficiencyEvidenceError("Invalid routing counts.")
    rate = invoked / total
    return {
        "sample_count": total,
        "stage2_invoked_count": invoked,
        "stage2_not_invoked_count": total - invoked,
        "stage2_invocation_rate": rate,
        "stage2_bypass_rate": 1.0 - rate,
        "average_forward_passes_per_input": 1.0 + rate,
        "minimum_forward_passes": 1,
        "maximum_forward_passes": 2,
    }


def bytes_to_mib(size: int) -> float:
    """Convert exact bytes to binary mebibytes."""
    if size < 0:
        raise EfficiencyEvidenceError("Byte size cannot be negative.")
    return size / 1_048_576


def mean_milliseconds(elapsed_seconds: float, sample_count: int) -> float:
    """Derive mean wall-clock milliseconds from one stored measurement."""
    if elapsed_seconds <= 0 or sample_count <= 0:
        raise EfficiencyEvidenceError("Timing inputs must be positive.")
    return elapsed_seconds / sample_count * 1000.0


def count_state_dictionary(state: Mapping[str, Any]) -> dict[str, int]:
    """Count model tensors while explicitly excluding standard BN buffers."""
    tensors = {key: value for key, value in state.items() if isinstance(value, torch.Tensor)}
    suffixes = ("running_mean", "running_var", "num_batches_tracked")
    buffers = {key: value for key, value in tensors.items() if key.endswith(suffixes)}
    parameters = {key: value for key, value in tensors.items() if key not in buffers}
    return {
        "state_tensor_count": len(tensors),
        "state_tensor_elements": sum(value.numel() for value in tensors.values()),
        "parameter_tensor_count": len(parameters),
        "parameter_elements": sum(value.numel() for value in parameters.values()),
        "buffer_tensor_count": len(buffers),
        "buffer_elements": sum(value.numel() for value in buffers.values()),
    }


def inspect_checkpoint(path: Path, expected_sha256: str, expected_outputs: int) -> dict[str, Any]:
    """Safely inspect CPU state-dictionary metadata without model construction."""
    if not path.is_file():
        raise EfficiencyEvidenceError(f"Checkpoint missing: {path}")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected_sha256:
        raise EfficiencyEvidenceError(f"Checkpoint hash mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state_dict"), dict):
        raise EfficiencyEvidenceError(f"Unsupported checkpoint schema: {path}")
    state = payload["model_state_dict"]
    weight = state.get("classifier.1.weight")
    bias = state.get("classifier.1.bias")
    if (
        not isinstance(weight, torch.Tensor)
        or not isinstance(bias, torch.Tensor)
        or tuple(weight.shape) != (expected_outputs, 1280)
        or tuple(bias.shape) != (expected_outputs,)
    ):
        raise EfficiencyEvidenceError(f"Classifier head mismatch: {path}")
    return {
        "path": path.as_posix(),
        "sha256": observed,
        "bytes": path.stat().st_size,
        "mib": bytes_to_mib(path.stat().st_size),
        "checkpoint_contains_training_state": any(
            key in payload for key in ("optimizer_state_dict", "scheduler_state_dict")
        ),
        "classifier_output_count": expected_outputs,
        **count_state_dictionary(state),
    }


def validate_grade(grade: str) -> None:
    if grade not in GRADES:
        raise EfficiencyEvidenceError(f"Unknown evidence grade: {grade}")


def unavailable(metric: str, system: str, note: str) -> dict[str, Any]:
    """Represent missing evidence explicitly, never as numeric zero."""
    return {
        "metric": metric,
        "value": "unavailable",
        "unit": "unavailable",
        "system": system,
        "source_path": "none",
        "source_field": "none",
        "evidence_grade": "U",
        "evidence_type": "unavailable",
        "hardware": "unavailable",
        "batch_size": "unavailable",
        "sample_count": "unavailable",
        "timing_scope": "unavailable",
        "data_loading": "unavailable",
        "warm_up": "unavailable",
        "precision_mode": "unavailable",
        "software_environment": "unavailable",
        "comparability": "evidence unavailable",
        "limitations": note,
    }


def generate_efficiency(repository: Path, generated: Path, reports: Path) -> list[Path]:
    """Generate all deterministic Gate 5A outputs from committed evidence."""
    checkpoints = {
        "stage1": inspect_checkpoint(
            repository / "runs/phase03_full/full__stage01_isic2019_efficientnet_b0_cross_entropy_seed42__20260724T190600Z/best_checkpoint.pt",
            "95e02c26b1ea4a0dba17016313c81f97c9c2635270a37b4debbee0f84e07ba3b",
            2,
        ),
        "stage2": inspect_checkpoint(
            repository / "runs/phase04_cb_focal_full/full__stage02_isic2019_efficientnet_b0_class_balanced_focal_loss_seed42__20260726T064808Z/best_checkpoint.pt",
            "10986d41b64a685fcd8fe166623c5b1c7fd2f21bdad7cf4d55dedc3967a397fd",
            3,
        ),
        "flat": inspect_checkpoint(
            repository / "runs/phase06_full/full__phase06_flat_four_class_isic2019_efficientnet_b0_cross_entropy_seed42__20260726T232308Z/best_checkpoint.pt",
            "f3d8b8b0e5ef42e3c287a2377b5570411d442246acd16cb874ccf903facdc7a7",
            4,
        ),
    }
    compute = routing_compute(3668, 1799)
    compute["flat_forward_passes_per_input"] = 1.0
    compute["hierarchical_installed_parameter_count"] = (
        checkpoints["stage1"]["parameter_elements"] + checkpoints["stage2"]["parameter_elements"]
    )
    compute["average_conditionally_active_parameter_pass_count"] = (
        checkpoints["stage1"]["parameter_elements"]
        + compute["stage2_invocation_rate"] * checkpoints["stage2"]["parameter_elements"]
    )
    compute["combined_hierarchical_checkpoint_bytes"] = (
        checkpoints["stage1"]["bytes"] + checkpoints["stage2"]["bytes"]
    )
    compute["combined_hierarchical_checkpoint_mib"] = bytes_to_mib(
        compute["combined_hierarchical_checkpoint_bytes"]
    )

    flat_elapsed = 30.6784900749999
    hierarchy_elapsed = 39.84738709799967
    common = {
        "hardware": "NVIDIA Tesla T4 (stored environment label: Tesla T4)",
        "batch_size": 64,
        "sample_count": 3668,
        "timing_scope": "wall clock around dataloader loop; includes batch loading iteration, host-to-device transfer, forward work, softmax/CPU collection and metadata accumulation; excludes post-loop metrics and file writing",
        "data_loading": "included within timed dataloader iteration",
        "warm_up": "none documented",
        "precision_mode": "CUDA autocast float16 enabled",
        "software_environment": "Python 3.12.3; torch 2.13.0+cu130; Linux Azure Tesla T4",
        "comparability": "comparable with limitations; no explicit CUDA synchronization or warm-up and different evaluator paths",
        "limitations": "Stored end-to-end evaluator-loop timing; no speed ratio authorized.",
    }
    inventory: list[dict[str, Any]] = []

    def add(metric: str, value: Any, unit: str, system: str, path: str, field: str, grade: str, kind: str, **details: Any) -> None:
        validate_grade(grade)
        inventory.append(
            {
                "metric": metric,
                "value": value,
                "unit": unit,
                "system": system,
                "source_path": path,
                "source_field": field,
                "evidence_grade": grade,
                "evidence_type": kind,
                "hardware": details.get("hardware", "not applicable"),
                "batch_size": details.get("batch_size", "not applicable"),
                "sample_count": details.get("sample_count", "not applicable"),
                "timing_scope": details.get("timing_scope", "not applicable"),
                "data_loading": details.get("data_loading", "not applicable"),
                "warm_up": details.get("warm_up", "not applicable"),
                "precision_mode": details.get("precision_mode", "not applicable"),
                "software_environment": details.get("software_environment", "not applicable"),
                "comparability": details.get("comparability", "not applicable"),
                "limitations": details.get("limitations", "none"),
            }
        )

    flat_source = "runs/backups/phase06c/phase06c_selected_flat_internal_test_550e7cdb1144.tar.gz::runs/phase06c/selected_flat_internal_test/locked_primary_evaluation/evaluation_summary.json"
    hierarchy_source = "runs/phase05_hierarchical_internal_test/locked_primary_evaluation/evaluation_summary.json"
    for system, elapsed, throughput, source in (
        ("flat", flat_elapsed, 119.56259877956239, flat_source),
        ("hierarchical", hierarchy_elapsed, 92.0512050383382, hierarchy_source),
    ):
        add("elapsed_evaluator_loop", elapsed, "seconds", system, source, "elapsed_seconds", "A", "direct locked measurement", **common)
        add("throughput_evaluator_loop", throughput, "samples/second", system, source, "samples_per_second", "A", "direct locked measurement", **common)
        add("mean_evaluator_loop_time_per_sample", mean_milliseconds(elapsed, 3668), "milliseconds/sample", system, source, "elapsed_seconds / sample_count * 1000", "B", "deterministic derivation", **common)

    routing_source = "reports/phase07/generated/gate04_evidence_review.json"
    add("stage2_invocation_count", 1799, "samples", "hierarchical", routing_source, "routing audit", "A", "direct committed routing count", sample_count=3668)
    add("stage2_invocation_rate", compute["stage2_invocation_rate"], "fraction", "hierarchical", routing_source, "1799 / 3668", "B", "deterministic derivation", sample_count=3668)
    add("stage2_bypass_rate", compute["stage2_bypass_rate"], "fraction", "hierarchical", routing_source, "1869 / 3668", "B", "deterministic derivation", sample_count=3668)
    add("average_forward_passes_per_input", compute["average_forward_passes_per_input"], "model passes/input", "hierarchical", routing_source, "1 + 1799 / 3668", "B", "deterministic derivation", sample_count=3668, limitations="Not FLOPs or measured latency.")
    for role in ("flat", "stage1", "stage2"):
        item = checkpoints[role]
        add("parameter_elements", item["parameter_elements"], "tensor elements", role, item["path"], "model_state_dict excluding standard BatchNorm buffers", "C", "static artifact measurement", limitations="Parameter count from state dictionary; no model instantiated.")
        add("checkpoint_size", item["bytes"], "bytes", role, item["path"], "filesystem size", "C", "static artifact measurement", limitations="Contains training state; not RAM, VRAM or deployment binary size.")
    for metric in ("FLOPs_or_MACs", "peak_GPU_memory", "CPU_memory", "energy_use", "power_draw", "model_loading_time"):
        inventory.extend([unavailable(metric, "flat", "No defensible stored measurement."), unavailable(metric, "hierarchical", "No defensible stored measurement.")])

    comparison = pd.DataFrame(
        [
            {"measure": "component models", "flat_system": 1, "hierarchical_system": 2, "unit": "models", "evidence_grade": "B", "comparability": "direct architecture accounting", "source_or_note": "selected flat checkpoint versus Stage 1 + Stage 2"},
            {"measure": "installed parameter count", "flat_system": checkpoints["flat"]["parameter_elements"], "hierarchical_system": compute["hierarchical_installed_parameter_count"], "unit": "parameter elements", "evidence_grade": "C", "comparability": "static state-dictionary audit", "source_or_note": "excludes standard BatchNorm buffers"},
            {"measure": "average conditionally active parameter-pass count", "flat_system": checkpoints["flat"]["parameter_elements"], "hierarchical_system": compute["average_conditionally_active_parameter_pass_count"], "unit": "parameter-passes/input", "evidence_grade": "B/C", "comparability": "workload proxy only", "source_or_note": "not FLOPs, latency or memory"},
            {"measure": "stored checkpoint size", "flat_system": checkpoints["flat"]["bytes"], "hierarchical_system": compute["combined_hierarchical_checkpoint_bytes"], "unit": "bytes", "evidence_grade": "C", "comparability": "system storage comparison", "source_or_note": "checkpoints contain training state"},
            {"measure": "forward passes per image", "flat_system": "1", "hierarchical_system": f"1 to 2; mean {compute['average_forward_passes_per_input']:.17g}", "unit": "model passes/input", "evidence_grade": "B", "comparability": "architecture accounting", "source_or_note": "does not establish latency"},
            {"measure": "Stage 2 invocation rate", "flat_system": "not applicable", "hierarchical_system": compute["stage2_invocation_rate"], "unit": "fraction", "evidence_grade": "B", "comparability": "hierarchical only", "source_or_note": "1799 / 3668"},
            {"measure": "stored measured throughput", "flat_system": 119.56259877956239, "hierarchical_system": 92.0512050383382, "unit": "samples/second", "evidence_grade": "A", "comparability": "comparable with limitations; no ratio", "source_or_note": "Tesla T4, batch 64, AMP; no explicit warm-up/synchronization"},
            {"measure": "stored mean evaluator-loop time", "flat_system": mean_milliseconds(flat_elapsed, 3668), "hierarchical_system": mean_milliseconds(hierarchy_elapsed, 3668), "unit": "ms/sample", "evidence_grade": "B", "comparability": "comparable with limitations; no ratio", "source_or_note": "derived from exact elapsed time"},
            {"measure": "peak memory", "flat_system": "unavailable", "hierarchical_system": "unavailable", "unit": "unavailable", "evidence_grade": "U", "comparability": "evidence unavailable", "source_or_note": "not estimated from checkpoint size"},
            {"measure": "FLOPs/MACs", "flat_system": "unavailable", "hierarchical_system": "unavailable", "unit": "unavailable", "evidence_grade": "U", "comparability": "evidence unavailable", "source_or_note": "no profiling or proven static calculation"},
        ]
    )
    artifact_rows = []
    for role in ("flat", "stage1", "stage2"):
        item = checkpoints[role]
        artifact_rows.append({"role": role, **item})
    artifact_rows.append(
        {
            "role": "hierarchical_combined",
            "path": "Stage 1 + Stage 2",
            "sha256": "not_applicable_combination",
            "bytes": compute["combined_hierarchical_checkpoint_bytes"],
            "mib": compute["combined_hierarchical_checkpoint_mib"],
            "checkpoint_contains_training_state": True,
            "classifier_output_count": "2 + 3",
            "state_tensor_count": checkpoints["stage1"]["state_tensor_count"] + checkpoints["stage2"]["state_tensor_count"],
            "state_tensor_elements": checkpoints["stage1"]["state_tensor_elements"] + checkpoints["stage2"]["state_tensor_elements"],
            "parameter_tensor_count": checkpoints["stage1"]["parameter_tensor_count"] + checkpoints["stage2"]["parameter_tensor_count"],
            "parameter_elements": compute["hierarchical_installed_parameter_count"],
            "buffer_tensor_count": checkpoints["stage1"]["buffer_tensor_count"] + checkpoints["stage2"]["buffer_tensor_count"],
            "buffer_elements": checkpoints["stage1"]["buffer_elements"] + checkpoints["stage2"]["buffer_elements"],
        }
    )

    claims = {
        "supported": [
            "The flat system uses one model decision path per image.",
            "The hierarchical system stores and coordinates two component models.",
            "Stage 2 was invoked for 1,799 of 3,668 samples on the locked split.",
            "The hierarchy required one to two model passes per image and a derived mean of 1 + 1799/3668.",
            "Static checkpoint and parameter footprints differ by the audited values.",
            "Stored evaluator-loop timing may be reported under its documented conditions.",
        ],
        "qualified": [
            "Timing values are comparable with limitations; evaluator paths differ and neither documents warm-up or explicit CUDA synchronization, so no speed ratio or faster-system claim is authorized.",
            "Average conditionally active parameter-pass count is an architecture workload proxy, not FLOPs, latency, memory, or energy.",
        ],
        "prohibited": PROHIBITED,
    }
    figures = [
        {"candidate": "conditional hierarchical architecture diagram", "research_value": "high", "evidence_source": "documented Stage 1/Stage 2 routing policy", "space_cost": "moderate", "two_column_readability": "high with simplified labels", "table_duplication": "low", "misleading_risk": "low if conditional path is explicit", "recommendation": "required"},
        {"candidate": "normalized confusion-matrix comparison", "research_value": "high", "evidence_source": "Gate 3 fixed confusion matrices", "space_cost": "moderate", "two_column_readability": "high as paired panels", "table_duplication": "moderate", "misleading_risk": "low with identical scales", "recommendation": "required"},
        {"candidate": "per-class F1 with 95% intervals", "research_value": "high", "evidence_source": "Gate 3 exploratory intervals", "space_cost": "moderate", "two_column_readability": "high", "table_duplication": "moderate", "misleading_risk": "medium unless exploratory status and SCC support are visible", "recommendation": "required"},
        {"candidate": "paired disagreement visualization", "research_value": "moderate", "evidence_source": "Gate 3 paired categories", "space_cost": "low", "two_column_readability": "high", "table_duplication": "high", "misleading_risk": "low", "recommendation": "optional"},
        {"candidate": "routing decomposition flow", "research_value": "moderate", "evidence_source": "Gate 4 routing dictionary", "space_cost": "high", "two_column_readability": "medium", "table_duplication": "high", "misleading_risk": "medium because structural missingness is not an error", "recommendation": "omit"},
    ]

    generated.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    outputs = [
        generated / "efficiency_evidence_inventory.csv",
        generated / "efficiency_comparison_table.csv",
        generated / "model_artifact_inventory.csv",
        generated / "conditional_compute_summary.json",
        generated / "efficiency_claims_lock.json",
        generated / "figure_candidate_review.json",
    ]
    write_csv(outputs[0], pd.DataFrame(inventory))
    write_csv(outputs[1], comparison)
    write_csv(outputs[2], pd.DataFrame(artifact_rows))
    write_json(outputs[3], compute)
    write_json(outputs[4], claims)
    write_json(outputs[5], figures)

    audit = f"""# Phase 07 Gate 5A — Efficiency Evidence Audit

## Decision and method

PASS. Evidence was collected without training, inference, evaluation, model construction, forward passes, benchmarking or GPU work. Parameter counts are Grade C static measurements from verified checkpoints loaded CPU-only with `weights_only=True`; standard BatchNorm buffers were separated. Checkpoints contain optimizer and scheduler state, so file size is storage evidence only.

## Static artifacts and conditional compute

| Role | Parameters | Checkpoint bytes | MiB |
|---|---:|---:|---:|
| Flat | {checkpoints['flat']['parameter_elements']:,} | {checkpoints['flat']['bytes']:,} | {checkpoints['flat']['mib']:.6f} |
| Stage 1 | {checkpoints['stage1']['parameter_elements']:,} | {checkpoints['stage1']['bytes']:,} | {checkpoints['stage1']['mib']:.6f} |
| Stage 2 | {checkpoints['stage2']['parameter_elements']:,} | {checkpoints['stage2']['bytes']:,} | {checkpoints['stage2']['mib']:.6f} |
| Hierarchy combined | {compute['hierarchical_installed_parameter_count']:,} | {compute['combined_hierarchical_checkpoint_bytes']:,} | {compute['combined_hierarchical_checkpoint_mib']:.6f} |

Stage 2 invocation was 1,799/3,668 ({compute['stage2_invocation_rate']:.6%}); bypass was 1,869/3,668 ({compute['stage2_bypass_rate']:.6%}). The hierarchy therefore used 1–2 passes and a derived mean of {compute['average_forward_passes_per_input']:.9f}. Its average conditionally active parameter-pass count was {compute['average_conditionally_active_parameter_pass_count']:.6f}; this is a workload proxy, not FLOPs, memory or latency.

## Stored timing

Flat: {flat_elapsed:.15g} seconds, 119.56259877956239 samples/s, {mean_milliseconds(flat_elapsed, 3668):.9f} ms/sample. Hierarchy: {hierarchy_elapsed:.15g} seconds, 92.0512050383382 samples/s, {mean_milliseconds(hierarchy_elapsed, 3668):.9f} ms/sample.

Both are stored Tesla T4, batch-64, CUDA-float16-autocast evaluator-loop measurements. The timer includes dataloader iteration, transfers, model work and CPU collection but excludes post-loop metrics/writes. Neither path documents warm-up or explicit CUDA synchronization, and the evaluator paths differ. Classification: comparable with limitations. No speed ratio or claim that either system is faster is authorized.

FLOPs/MACs, peak GPU/CPU memory, energy, power, and model-loading time are Grade U unavailable and were not estimated.
"""
    efficiency_claims = "# Phase 07 Efficiency Claims Lock\n\n## Supported\n\n" + "\n".join(f"- {x}" for x in claims["supported"]) + "\n\n## Qualified\n\n" + "\n".join(f"- {x}" for x in claims["qualified"]) + "\n\n## Prohibited formulations\n\n" + "\n".join(f"- `{x}`" for x in claims["prohibited"]) + "\n"
    figure_md = "# Phase 07 Figure Candidate Review\n\nRequired: conditional architecture diagram, normalized confusion-matrix comparison, and per-class F1 with intervals. Optional: compact paired-disagreement visualization. Omit the routing-decomposition flow by default because it costs space, duplicates supporting tables, and risks treating structural missingness as an error.\n\nMachine-readable candidate assessments are in `generated/figure_candidate_review.json`. No final figures were generated in Gate 5A.\n"
    human = [
        reports / "phase07_gate05a_efficiency_evidence_audit.md",
        reports / "phase07_efficiency_claims_lock.md",
        reports / "phase07_figure_candidate_review.md",
    ]
    for path, text in zip(human, (audit, efficiency_claims, figure_md)):
        path.write_text(text, encoding="utf-8", newline="\n")
    return outputs + human
