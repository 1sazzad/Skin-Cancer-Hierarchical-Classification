"""Audit Phase 08 evidence without loading datasets, checkpoints, or predictions."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Iterable

LOCKED_REQUIRED = (
    "reports/phase05/conditional_hierarchical_internal_evaluation.md",
    "reports/phase06/phase06c_selected_flat_internal_test_result.md",
    "reports/phase07/phase07_claims_lock.md",
    "reports/phase07/phase07_final_summary.md",
    "reports/phase07/phase07_iccit_artifact_index.md",
    "reports/phase07/generated/claims_lock.json",
    "reports/phase07/generated/statistical_analysis_results.json",
    "reports/phase07/generated/paired_prediction_manifest.csv",
    "reports/phase07/generated/bootstrap_confidence_intervals.csv",
    "reports/phase07/generated/efficiency_evidence_inventory.csv",
)

LOCALLY_REFERENCED = (
    (
        "runs/phase05_hierarchical_internal_test/locked_primary_evaluation/"
        "per_image_hierarchical_predictions.csv",
        "phase05",
        "prediction",
        True,
        "Locked Phase 05 prediction evidence; intentionally outside Git.",
    ),
    (
        "runs/phase06c/selected_flat_internal_test/locked_primary_evaluation/"
        "internal_test_predictions.csv",
        "phase06c",
        "prediction",
        True,
        "Canonical member is preserved in the verified local archive when not extracted.",
    ),
    (
        "runs/phase06_full/full__phase06_flat_four_class_isic2019_"
        "efficientnet_b0_cross_entropy_seed42__20260726T232308Z/best_checkpoint.pt",
        "phase06",
        "checkpoint",
        True,
        "Selected checkpoint referenced by the registry; intentionally outside Git.",
    ),
    (
        "runs/phase06b/full/full__phase06b_flat_four_class_isic2019_"
        "efficientnet_b0_class_balanced_focal_loss_seed42__20260727T120615Z/"
        "best_checkpoint.pt",
        "phase06b",
        "checkpoint",
        True,
        "Rejected candidate; absence is not a locked-evidence failure.",
    ),
)

INVENTORY_PREFIXES = (
    "configs/",
    "data/manifests/",
    "docs/",
    "experiments/",
    "reports/phase03/",
    "reports/phase04/",
    "reports/phase05/",
    "reports/phase06/",
    "reports/phase07/",
    "scripts/",
    "src/",
    "tests/",
)

OBJECTIVE_COLUMNS = (
    "objective_id",
    "original_objective",
    "planned_claim",
    "current_status",
    "supporting_phase",
    "supporting_artifact",
    "evidence_strength",
    "missing_evidence",
    "next_required_action",
    "requires_gpu",
    "manuscript_destination",
    "claim_allowed_now",
    "notes",
)

OBJECTIVES = (
    ("O01", "Malignant versus non-malignant classification", "Stage 1 performance on the frozen ISIC 2019 split", "complete", "phase03,phase05", "reports/phase05/conditional_hierarchical_internal_evaluation.md", "strong_locked_internal", "External performance and multi-seed uncertainty", "Preserve as locked comparator; externally evaluate only under a frozen protocol", "yes_future_only", "Results;Discussion", "yes_limited", "Single seed and internal dataset only."),
    ("O02", "Malignant cancer-type classification: melanoma, BCC, SCC", "Stage 2 subtype performance and imbalance response", "complete", "phase03,phase04,phase05", "reports/phase04/stage02_imbalance_aware_final_internal_evaluation.md", "strong_locked_internal", "External performance and broader uncertainty", "Preserve selected Stage 2 evidence", "yes_future_only", "Methodology;Results;Discussion", "yes_limited", "SCC support is 94 on the internal test."),
    ("O03", "Melanoma T-category or Breslow-thickness-group estimation", "A defensible Stage 3 severity classifier is feasible and evaluated", "blocked", "none", "configs/dataset_registry.yaml", "none", "EMB acquisition; license; integrity; label semantics; leakage-safe split; baseline", "Complete Phase 09 audit before any Stage 3 training", "yes_after_audit", "Methodology;Results;Limitations", "no", "Failure of feasibility must be reported rather than forcing labels."),
    ("O04", "Lightweight shared or parameter-efficient conditional framework", "A three-task design reduces resources without unacceptable performance loss", "missing", "phase07_partial", "reports/phase07/phase07_gate05a_efficiency_evidence_audit.md", "weak_partial", "No shared model; FLOPs; memory; matched latency; three-task results", "Design only after Stage 3 feasibility; compare with separate models", "yes", "Methodology;Results", "no", "Current system uses two separately trained EfficientNet-B0 models."),
    ("O05", "Partially labelled multi-dataset learning", "Masked supervision can train eligible heads without inventing labels", "missing", "none", "configs/dataset_registry.yaml", "none", "No masked-loss implementation; sampling protocol; training evidence", "Freeze dataset/task masks and leakage controls before implementation", "yes", "Methodology;Experimental Setup", "no", "Dataset roles exist, but learning support does not."),
    ("O06", "Comparison with flat classification", "Flat and hierarchical internal performance can be compared fairly", "partial", "phase06c,phase07", "reports/phase07/generated/statistical_analysis_results.json", "strong_locked_comparator_only", "No flat comparison with the unimplemented three-stage framework; no external comparison", "Reuse the locked two-stage comparator result; compare the final framework only under a frozen protocol", "yes_future_only", "Results;Discussion", "yes_two_stage_only", "No statistically distinguishable macro-F1 difference was established for the locked flat-versus-two-stage comparison."),
    ("O07", "Comparison with separate task-specific models", "Shared/conditional design is competitive with separate task-specific models", "partial", "phase03,phase04,phase05", "reports/phase05/conditional_hierarchical_internal_evaluation.md", "moderate_comparator_only", "No Stage 3 standalone model; no matched three-task shared comparison", "Train Stage 3 baseline then freeze matched comparison", "yes", "Results;Discussion", "no", "Two-task separate checkpoints are available only as a comparator."),
    ("O08", "Class-imbalance handling", "Imbalance-aware loss is evaluated across the proposed framework", "partial", "phase04", "reports/phase04/stage02_imbalance_aware_final_internal_evaluation.md", "moderate_stage2_only", "No Stage 3 or shared-framework imbalance evidence; no broader multi-seed evidence", "Preserve the locked Stage 2 result; select Stage 3 and shared-framework handling prospectively", "yes", "Methodology;Results;Limitations", "yes_stage2_only", "The completed Stage 2 sub-result does not complete framework-wide imbalance handling."),
    ("O09", "End-to-end error-propagation analysis", "Routing loss is quantified for the proposed three-stage framework", "partial", "phase05,phase07", "reports/phase07/generated/hierarchical_routing_decomposition.csv", "strong_locked_two_stage_only", "No Stage 3 routing or three-stage decomposition", "Extend the locked two-stage protocol prospectively in Phase 14", "yes", "Results;Discussion", "yes_two_stage_only", "The completed two-stage routing analysis cannot be described as complete end-to-end three-stage error propagation."),
    ("O10", "External-dataset evaluation", "Frozen models have zero-shot evidence under domain shift", "missing", "none", "configs/dataset_registry.yaml", "none", "HIBA acquisition; mapping; overlap audit; frozen evaluation", "Predeclare compatibility and evaluation rules before viewing results", "yes", "Experimental Setup;Results;Limitations", "no", "External validation cannot prove broad clinical generalisation."),
    ("O11", "XAI and Grad-CAM analysis", "Preselected cases support a descriptive saliency analysis", "missing", "none", "docs/01_scope_lock.md", "none", "No implementation; case-selection protocol; outputs; review", "Preregister representative-case selection before generating maps", "yes", "Methodology;Figures;Discussion", "no", "XAI must not be presented as proof of clinical correctness."),
    ("O12", "Parameters, size, FLOPs, latency, memory, and conditional compute", "Efficiency is measured comparably for all final systems", "partial", "phase05,phase06c,phase07", "reports/phase07/generated/efficiency_evidence_inventory.csv", "mixed_partial", "FLOPs/MACs; peak GPU/CPU memory; matched profiling; Stage 3/shared model", "Freeze a matched Tesla T4 profiling protocol", "yes", "Results;Tables;Limitations", "yes_restricted", "Parameters, file size, timing with limitations, and workload proxies only."),
)

INVENTORY_COLUMNS = (
    "artifact_path",
    "phase",
    "artifact_type",
    "tracked",
    "exists_locally",
    "locally_referenced",
    "locked",
    "reusable",
    "missing",
    "evidence_role",
    "notes",
)


class EvidenceAuditError(RuntimeError):
    """Raised when mandatory locked evidence is absent."""


def _type(path: str) -> str:
    name = Path(path).name.lower()
    suffix = Path(path).suffix.lower()
    if "config" in path or suffix in {".yaml", ".yml"}:
        return "config"
    if "prediction" in name:
        return "prediction"
    if "metric" in name or "statistical" in name or "bootstrap" in name:
        return "metric_or_statistical"
    if "figure" in path or suffix in {".png", ".pdf", ".svg"}:
        return "figure"
    if "table" in name:
        return "table"
    if path.startswith("scripts/") or path.startswith("src/"):
        return "code"
    if path.startswith("tests/"):
        return "test"
    if suffix == ".md":
        return "report_or_document"
    if "manifest" in name:
        return "manifest"
    return "supporting_artifact"


def _phase(path: str) -> str:
    for phase in ("phase03", "phase04", "phase05", "phase06c", "phase06b", "phase06", "phase07", "phase08"):
        if phase in path.lower():
            return phase
    return "cross_phase"


def git_tracked_paths(repository: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    return sorted(item for item in result.stdout.decode("utf-8").split("\0") if item)


def validate_locked_evidence(repository: Path) -> list[str]:
    missing = [path for path in LOCKED_REQUIRED if not (repository / path).is_file()]
    if missing:
        raise EvidenceAuditError(
            "Missing required locked evidence: " + ", ".join(missing)
        )
    return list(LOCKED_REQUIRED)


def build_inventory(repository: Path, tracked_paths: Iterable[str]) -> list[dict[str, str]]:
    tracked = {path.replace("\\", "/") for path in tracked_paths}
    rows: list[dict[str, str]] = []
    selected = sorted(path for path in tracked if path.startswith(INVENTORY_PREFIXES))
    for path in selected:
        phase = _phase(path)
        locked = phase in {"phase05", "phase06c", "phase07"}
        rows.append(
            {
                "artifact_path": path,
                "phase": phase,
                "artifact_type": _type(path),
                "tracked": "yes",
                "exists_locally": "yes" if (repository / path).is_file() else "no",
                "locally_referenced": "no",
                "locked": "yes" if locked else "no",
                "reusable": "yes",
                "missing": "no" if (repository / path).is_file() else "yes",
                "evidence_role": "repository evidence",
                "notes": "Tracked file; contents are not hashed by this audit.",
            }
        )
    for path, phase, artifact_type, locked, note in LOCALLY_REFERENCED:
        exists = (repository / path).is_file()
        rows.append(
            {
                "artifact_path": path,
                "phase": phase,
                "artifact_type": artifact_type,
                "tracked": "yes" if path in tracked else "no",
                "exists_locally": "yes" if exists else "no",
                "locally_referenced": "yes",
                "locked": "yes" if locked else "no",
                "reusable": "yes" if exists else "archive_or_restore_required",
                "missing": "no" if exists else "not_extracted_or_missing",
                "evidence_role": "registry or report reference",
                "notes": note,
            }
        )
    return sorted(rows, key=lambda row: row["artifact_path"])


def write_csv(path: Path, columns: Iterable[str], rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def generate_audit(repository: Path, output: Path, tracked_paths: Iterable[str] | None = None) -> list[Path]:
    repository = repository.resolve()
    validated = validate_locked_evidence(repository)
    tracked = list(tracked_paths) if tracked_paths is not None else git_tracked_paths(repository)
    inventory = build_inventory(repository, tracked)
    objective_rows = [dict(zip(OBJECTIVE_COLUMNS, row)) for row in OBJECTIVES]
    matrix_path = output / "phase08_objective_evidence_matrix.csv"
    inventory_path = output / "phase08_artifact_inventory.csv"
    json_path = output / "phase08_evidence_completeness_audit.json"
    write_csv(matrix_path, OBJECTIVE_COLUMNS, objective_rows)
    write_csv(inventory_path, INVENTORY_COLUMNS, inventory)
    payload = {
        "audit_version": 1,
        "execution_mode": "repository_metadata_only_no_model_or_dataset_access",
        "locked_required_count": len(LOCKED_REQUIRED),
        "locked_required_validated": validated,
        "objective_count": len(objective_rows),
        "inventory_count": len(inventory),
        "status_counts": {
            status: sum(row["current_status"] == status for row in objective_rows)
            for status in ("complete", "partial", "missing", "blocked", "obsolete")
        },
        "large_binary_policy": "paths and existence only; no checkpoint content or hash inspection",
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return [matrix_path, inventory_path, json_path]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/phase08/generated"),
    )
    args = parser.parse_args()
    try:
        outputs = generate_audit(args.repository, args.output)
    except (EvidenceAuditError, subprocess.CalledProcessError) as error:
        print(f"Phase 08 evidence audit failed: {error}")
        return 1
    print(f"Generated {len(outputs)} deterministic Phase 08 audit artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
