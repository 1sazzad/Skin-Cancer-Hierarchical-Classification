#!/usr/bin/env python3
"""ITT-04: run the frozen seven-CNN consensus analysis on locked ISIC predictions only.

This script performs no training, checkpoint loading, threshold tuning, or HIBA access.
It consumes the seven committed per-image Flat prediction CSVs from Gate 06D and writes
new ITT artifacts under results/extensions/itt2026_consensus/internal_isic/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.itt2026_consensus import (  # noqa: E402
    CLASS_NAMES,
    CohortSpec,
    FROZEN_MODELS,
    build_consensus,
    full_coverage_metrics,
    prediction_matrix,
    selective_referral_metrics,
    sha256_file,
    validate_and_align_prediction_frames,
)

INPUT_TEMPLATE = Path("results/paper/internal_isic/gate06d/{model}/paired_internal_test_predictions.csv")
DEFAULT_OUTPUT = Path("results/extensions/itt2026_consensus/internal_isic")
EXPECTED_ROWS = 3668


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _flatten_referral(rows: list[dict[str, object]]) -> pd.DataFrame:
    flat_rows: list[dict[str, object]] = []
    for row in rows:
        per_class = row["per_class"]
        coverage = row["class_specific_coverage"]
        flat = {
            "threshold_votes": row["threshold_votes"],
            "threshold_fraction": row["threshold_fraction"],
            "retained_count": row["retained_count"],
            "referred_count": row["referred_count"],
            "coverage": row["coverage"],
            "accuracy": row["accuracy"],
            "error_rate": row["error_rate"],
            "macro_f1": row["macro_f1"],
            "weighted_f1": row["weighted_f1"],
            "balanced_accuracy": row["balanced_accuracy"],
            "malignant_sensitivity": row["malignant_sensitivity"],
            "malignant_coverage": row["malignant_coverage"],
            "scc_sensitivity": row["scc_sensitivity"],
            "scc_f1": row["scc_f1"],
        }
        for name in CLASS_NAMES:
            flat[f"coverage_{name}"] = coverage[name]
            flat[f"precision_{name}"] = per_class[name]["precision"]
            flat[f"recall_{name}"] = per_class[name]["recall"]
            flat[f"f1_{name}"] = per_class[name]["f1"]
            flat[f"support_retained_{name}"] = per_class[name]["support"]
        flat_rows.append(flat)
    return pd.DataFrame(flat_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = args.output_directory
    if output.exists() and any(output.iterdir()):
        raise SystemExit(
            f"Refusing to overwrite existing ITT-04 output directory: {output}. "
            "Inspect/remove it manually only if the previous run did not complete."
        )

    frames: dict[str, pd.DataFrame] = {}
    input_manifest: dict[str, object] = {}
    for model in FROZEN_MODELS:
        path = Path(str(INPUT_TEMPLATE).format(model=model))
        if not path.is_file():
            raise SystemExit(f"Missing frozen prediction file: {path}")
        input_manifest[model] = {
            "path": path.as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        frames[model] = pd.read_csv(path)

    aligned = validate_and_align_prediction_frames(
        frames,
        CohortSpec(
            id_field="sample_id",
            target_field="true_label",
            prediction_field="flat_prediction",
            expected_rows=EXPECTED_ROWS,
        ),
    )
    target = aligned["target"].to_numpy(dtype=np.int64)
    matrix = prediction_matrix(aligned)
    consensus = build_consensus(matrix)

    individual_metrics = {
        model: full_coverage_metrics(target, matrix[:, index])
        for index, model in enumerate(FROZEN_MODELS)
    }
    majority_metrics = full_coverage_metrics(target, consensus.majority_prediction)
    weighted_metrics = full_coverage_metrics(target, consensus.weighted_prediction)
    referral_rows = selective_referral_metrics(
        target,
        consensus.majority_prediction,
        consensus.max_vote_count,
    )

    comparator = individual_metrics["densenet169"]
    summary = {
        "status": "PASS",
        "phase": "ITT-04",
        "cohort": "ISIC 2019 frozen internal test",
        "sample_count": int(target.size),
        "class_support": {
            name: int(np.sum(target == index))
            for index, name in enumerate(CLASS_NAMES)
        },
        "primary_endpoint": "macro_f1",
        "primary_individual_comparator": "densenet169",
        "primary_ensemble": "seven_model_hard_majority_vote",
        "secondary_ensemble": "validation_macro_f1_weighted_hard_vote",
        "majority": majority_metrics,
        "weighted": weighted_metrics,
        "individual_models": individual_metrics,
        "point_differences_macro_f1": {
            "majority_minus_densenet169": float(
                majority_metrics["macro_f1"] - comparator["macro_f1"]
            ),
            "weighted_minus_densenet169": float(
                weighted_metrics["macro_f1"] - comparator["macro_f1"]
            ),
            "weighted_minus_majority": float(
                weighted_metrics["macro_f1"] - majority_metrics["macro_f1"]
            ),
        },
        "tie_count": int(consensus.majority_tie_required.sum()),
        "tie_fraction": float(consensus.majority_tie_required.mean()),
        "agreement_distribution": {
            str(votes): int(np.sum(consensus.max_vote_count == votes))
            for votes in range(2, 8)
        },
        "selective_referral": referral_rows,
        "interpretation_boundary": (
            "No threshold is selected or optimized from this cohort. "
            "All four prospectively frozen agreement thresholds are reported."
        ),
    }

    per_sample = aligned.copy()
    per_sample["majority_prediction"] = consensus.majority_prediction
    per_sample["weighted_prediction"] = consensus.weighted_prediction
    per_sample["max_vote_count"] = consensus.max_vote_count
    per_sample["agreement"] = consensus.agreement
    per_sample["disagreement"] = consensus.disagreement
    per_sample["majority_tie_required"] = consensus.majority_tie_required
    for index, name in enumerate(CLASS_NAMES):
        per_sample[f"vote_count_{name}"] = consensus.vote_counts[:, index]
        per_sample[f"weighted_score_{name}"] = consensus.weighted_scores[:, index]
    per_sample["majority_correct"] = per_sample["majority_prediction"].to_numpy() == target
    per_sample["weighted_correct"] = per_sample["weighted_prediction"].to_numpy() == target

    agreement_rows: list[dict[str, object]] = []
    for votes in range(7, 1, -1):
        mask = consensus.max_vote_count == votes
        count = int(mask.sum())
        if count == 0:
            continue
        agreement_rows.append(
            {
                "max_vote_count": votes,
                "agreement_fraction": votes / 7.0,
                "count": count,
                "fraction_of_cohort": count / target.size,
                "majority_accuracy": float(np.mean(consensus.majority_prediction[mask] == target[mask])),
                "majority_error_rate": float(np.mean(consensus.majority_prediction[mask] != target[mask])),
                "weighted_accuracy": float(np.mean(consensus.weighted_prediction[mask] == target[mask])),
                "weighted_error_rate": float(np.mean(consensus.weighted_prediction[mask] != target[mask])),
                **{
                    f"true_{name}_count": int(np.sum(target[mask] == index))
                    for index, name in enumerate(CLASS_NAMES)
                },
            }
        )

    baseline_rows = []
    for model in FROZEN_MODELS:
        metrics = individual_metrics[model]
        baseline_rows.append(
            {
                "system": model,
                "type": "individual",
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "malignant_sensitivity": metrics["malignant_sensitivity"],
                "scc_sensitivity": metrics["scc_sensitivity"],
                "scc_f1": metrics["scc_f1"],
            }
        )
    for system, metrics, system_type in (
        ("majority_vote", majority_metrics, "primary_ensemble"),
        ("validation_weighted_vote", weighted_metrics, "secondary_ensemble"),
    ):
        baseline_rows.append(
            {
                "system": system,
                "type": system_type,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "malignant_sensitivity": metrics["malignant_sensitivity"],
                "scc_sensitivity": metrics["scc_sensitivity"],
                "scc_f1": metrics["scc_f1"],
            }
        )

    output.mkdir(parents=True, exist_ok=False)
    per_sample.to_csv(output / "per_sample_consensus_predictions.csv", index=False)
    pd.DataFrame(baseline_rows).to_csv(output / "full_coverage_metrics.csv", index=False)
    _flatten_referral(referral_rows).to_csv(output / "selective_referral_metrics.csv", index=False)
    pd.DataFrame(agreement_rows).to_csv(output / "agreement_strata.csv", index=False)
    _write_json(output / "itt04_internal_isic_results.json", summary)
    _write_json(
        output / "itt04_execution_manifest.json",
        {
            "phase": "ITT-04",
            "status": "PASS",
            "input_files": input_manifest,
            "sample_count": int(target.size),
            "models": list(FROZEN_MODELS),
            "outputs": [
                "per_sample_consensus_predictions.csv",
                "full_coverage_metrics.csv",
                "selective_referral_metrics.csv",
                "agreement_strata.csv",
                "itt04_internal_isic_results.json",
                "itt04_execution_manifest.json",
            ],
            "prohibitions_respected": {
                "training": True,
                "checkpoint_loading": True,
                "threshold_tuning": True,
                "hiba_access": True,
                "hierarchical_predictions_used": False,
            },
        },
    )

    concise = {
        "status": "PASS",
        "sample_count": int(target.size),
        "majority_macro_f1": majority_metrics["macro_f1"],
        "weighted_macro_f1": weighted_metrics["macro_f1"],
        "densenet169_macro_f1": comparator["macro_f1"],
        "majority_minus_densenet169": summary["point_differences_macro_f1"]["majority_minus_densenet169"],
        "weighted_minus_densenet169": summary["point_differences_macro_f1"]["weighted_minus_densenet169"],
        "tie_count": summary["tie_count"],
        "agreement_distribution": summary["agreement_distribution"],
        "referral": [
            {
                "threshold_votes": row["threshold_votes"],
                "coverage": row["coverage"],
                "macro_f1": row["macro_f1"],
                "accuracy": row["accuracy"],
            }
            for row in referral_rows
        ],
        "output_directory": output.as_posix(),
    }
    print(json.dumps(concise, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
