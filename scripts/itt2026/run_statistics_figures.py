#!/usr/bin/env python3
"""ITT-06: prespecified inference plus publication tables/figures.

Consumes only completed ITT-04 and ITT-05 per-sample/results artifacts.
No training, checkpoint loading, model selection, threshold tuning, or
prediction recomputation is performed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.itt2026_consensus import (  # noqa: E402
    FROZEN_MODELS,
    paired_correctness_mcnemar,
    sha256_file,
)
from src.analysis.stored_prediction_statistics import calculate_metrics

INTERNAL_DIR = Path("results/extensions/itt2026_consensus/internal_isic")
HIBA_DIR = Path("results/extensions/itt2026_consensus/external_hiba")
DEFAULT_OUTPUT = Path("results/extensions/itt2026_consensus/statistics_figures")
REPLICATES = 10000
SEED = 42
COMPARATOR_COLUMN = "pred__densenet169"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _metric(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(calculate_metrics(target, prediction).macro_f1)


def _bootstrap_three_systems_image(
    target: np.ndarray,
    majority: np.ndarray,
    weighted: np.ndarray,
    comparator: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = np.empty((replicates, 6), dtype=np.float64)
    for replicate in range(replicates):
        idx = rng.integers(0, target.size, size=target.size, dtype=np.int64)
        y = target[idx]
        maj = _metric(y, majority[idx])
        wei = _metric(y, weighted[idx])
        cmp = _metric(y, comparator[idx])
        values[replicate] = (
            maj,
            wei,
            cmp,
            maj - cmp,
            wei - cmp,
            wei - maj,
        )
    return pd.DataFrame(
        values,
        columns=[
            "majority_macro_f1",
            "weighted_macro_f1",
            "densenet169_macro_f1",
            "majority_minus_densenet169",
            "weighted_minus_densenet169",
            "weighted_minus_majority",
        ],
    )


def _bootstrap_three_systems_patient(
    target: np.ndarray,
    majority: np.ndarray,
    weighted: np.ndarray,
    comparator: np.ndarray,
    patient_id: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    patients = np.asarray(patient_id, dtype=str)
    unique = np.unique(patients)
    groups = {patient: np.flatnonzero(patients == patient) for patient in unique}
    rng = np.random.default_rng(seed)
    values = np.empty((replicates, 6), dtype=np.float64)

    for replicate in range(replicates):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        idx = np.concatenate([groups[patient] for patient in sampled]).astype(np.int64)
        y = target[idx]
        maj = _metric(y, majority[idx])
        wei = _metric(y, weighted[idx])
        cmp = _metric(y, comparator[idx])
        values[replicate] = (
            maj,
            wei,
            cmp,
            maj - cmp,
            wei - cmp,
            wei - maj,
        )
    return pd.DataFrame(
        values,
        columns=[
            "majority_macro_f1",
            "weighted_macro_f1",
            "densenet169_macro_f1",
            "majority_minus_densenet169",
            "weighted_minus_densenet169",
            "weighted_minus_majority",
        ],
    )


def _ci(values: pd.Series) -> tuple[float, float]:
    array = values.to_numpy(dtype=np.float64)
    lower, upper = np.quantile(array, [0.025, 0.975], method="linear")
    return float(lower), float(upper)


def _inference_rows(
    cohort: str,
    boot: pd.DataFrame,
    point: dict[str, float],
    method: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for column in (
        "majority_minus_densenet169",
        "weighted_minus_densenet169",
        "weighted_minus_majority",
    ):
        lower, upper = _ci(boot[column])
        rows.append(
            {
                "cohort": cohort,
                "comparison": column,
                "point_difference_macro_f1": point[column],
                "ci95_lower": lower,
                "ci95_upper": upper,
                "bootstrap_method": method,
                "replicates": REPLICATES,
                "seed": SEED,
                "ci_excludes_zero": bool(lower > 0.0 or upper < 0.0),
            }
        )
    return rows


def _full_coverage_table(internal: pd.DataFrame, hiba: pd.DataFrame) -> pd.DataFrame:
    a = internal.copy()
    b = hiba.copy()
    a.insert(0, "cohort", "ISIC internal test")
    b.insert(0, "cohort", "HIBA external")
    return pd.concat([a, b], ignore_index=True)


def _referral_table(internal: pd.DataFrame, hiba: pd.DataFrame) -> pd.DataFrame:
    a = internal.copy()
    b = hiba.copy()
    a.insert(0, "cohort", "ISIC internal test")
    b.insert(0, "cohort", "HIBA external")
    return pd.concat([a, b], ignore_index=True)


def _agreement_table(internal: pd.DataFrame, hiba: pd.DataFrame) -> pd.DataFrame:
    a = internal.copy()
    b = hiba.copy()
    a.insert(0, "cohort", "ISIC internal test")
    b.insert(0, "cohort", "HIBA external")
    return pd.concat([a, b], ignore_index=True)


def _figure_full_coverage(full: pd.DataFrame, path: Path) -> None:
    systems = list(FROZEN_MODELS) + ["majority_vote", "validation_weighted_vote"]
    labels = [
        "DenseNet121",
        "DenseNet169",
        "ResNet50",
        "MobileNetV3-L",
        "EffNet-B0",
        "EffNet-B2",
        "EffNet-B3",
        "Majority",
        "Weighted",
    ]
    isic = full[full["cohort"] == "ISIC internal test"].set_index("system").loc[systems, "macro_f1"]
    hiba = full[full["cohort"] == "HIBA external"].set_index("system").loc[systems, "macro_f1"]
    x = np.arange(len(systems), dtype=np.float64)
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.bar(x - width / 2, isic.to_numpy(), width, label="ISIC internal test")
    ax.bar(x + width / 2, hiba.to_numpy(), width, label="HIBA external")
    ax.set_ylabel("Macro-F1")
    ax.set_xlabel("System")
    ax.set_title("Full-Coverage Macro-F1 Across Individual CNNs and Consensus")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=32, ha="right")
    ax.set_ylim(0.0, 0.8)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _figure_risk_coverage(
    internal_ref: pd.DataFrame,
    hiba_ref: pd.DataFrame,
    internal_full: pd.DataFrame,
    hiba_full: pd.DataFrame,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    for cohort, ref, full in (
        ("ISIC internal test", internal_ref, internal_full),
        ("HIBA external", hiba_ref, hiba_full),
    ):
        majority_accuracy = float(
            full.loc[full["system"] == "majority_vote", "accuracy"].iloc[0]
        )
        x = [1.0] + ref.sort_values("coverage", ascending=False)["coverage"].tolist()
        y = [1.0 - majority_accuracy] + ref.sort_values("coverage", ascending=False)["error_rate"].tolist()
        ax.plot(x, y, marker="o", label=cohort)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Retained error rate")
    ax.set_title("Selective Referral Risk-Coverage Curve")
    ax.set_xlim(0.48, 1.01)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _figure_agreement(internal: pd.DataFrame, hiba: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    for cohort, frame in (
        ("ISIC internal test", internal),
        ("HIBA external", hiba),
    ):
        ordered = frame.sort_values("max_vote_count")
        ax.plot(
            ordered["max_vote_count"],
            ordered["fraction_of_cohort"],
            marker="o",
            label=cohort,
        )
    ax.set_xlabel("Maximum agreeing votes (out of 7)")
    ax.set_ylabel("Fraction of cohort")
    ax.set_title("Cross-Backbone Agreement Distribution")
    ax.set_xticks([2, 3, 4, 5, 6, 7])
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _figure_macro_f1_coverage(
    internal_ref: pd.DataFrame,
    hiba_ref: pd.DataFrame,
    internal_full: pd.DataFrame,
    hiba_full: pd.DataFrame,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    for cohort, ref, full in (
        ("ISIC internal test", internal_ref, internal_full),
        ("HIBA external", hiba_ref, hiba_full),
    ):
        majority_f1 = float(
            full.loc[full["system"] == "majority_vote", "macro_f1"].iloc[0]
        )
        ordered = ref.sort_values("coverage", ascending=False)
        x = [1.0] + ordered["coverage"].tolist()
        y = [majority_f1] + ordered["macro_f1"].tolist()
        ax.plot(x, y, marker="o", label=cohort)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Retained macro-F1")
    ax.set_title("Macro-F1 Across the Frozen Referral Series")
    ax.set_xlim(0.48, 1.01)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = args.output_directory
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty ITT-06 output: {output}")

    internal_pred_path = INTERNAL_DIR / "per_sample_consensus_predictions.csv"
    hiba_pred_path = HIBA_DIR / "per_sample_consensus_predictions.csv"
    required = [
        internal_pred_path,
        HIBA_DIR / "per_sample_consensus_predictions.csv",
        INTERNAL_DIR / "full_coverage_metrics.csv",
        HIBA_DIR / "full_coverage_metrics.csv",
        INTERNAL_DIR / "selective_referral_metrics.csv",
        HIBA_DIR / "selective_referral_metrics.csv",
        INTERNAL_DIR / "agreement_strata.csv",
        HIBA_DIR / "agreement_strata.csv",
    ]
    for path in required:
        _require(path.is_file(), f"Missing prerequisite ITT artifact: {path}")

    internal = pd.read_csv(internal_pred_path)
    hiba = pd.read_csv(hiba_pred_path)
    _require(len(internal) == 3668, f"Expected 3668 internal rows, got {len(internal)}")
    _require(len(hiba) == 1232, f"Expected 1232 HIBA rows, got {len(hiba)}")
    _require(hiba["patient_id"].astype(str).str.strip().ne("").all(), "Missing HIBA patient IDs.")
    _require(hiba["patient_id"].nunique() == 568, "Expected 568 HIBA patients.")

    target_i = internal["target"].to_numpy(dtype=np.int64)
    majority_i = internal["majority_prediction"].to_numpy(dtype=np.int64)
    weighted_i = internal["weighted_prediction"].to_numpy(dtype=np.int64)
    comparator_i = internal[COMPARATOR_COLUMN].to_numpy(dtype=np.int64)

    target_h = hiba["target"].to_numpy(dtype=np.int64)
    majority_h = hiba["majority_prediction"].to_numpy(dtype=np.int64)
    weighted_h = hiba["weighted_prediction"].to_numpy(dtype=np.int64)
    comparator_h = hiba[COMPARATOR_COLUMN].to_numpy(dtype=np.int64)
    patient_h = hiba["patient_id"].astype(str).to_numpy()

    boot_i = _bootstrap_three_systems_image(
        target_i, majority_i, weighted_i, comparator_i,
        replicates=REPLICATES, seed=SEED,
    )
    boot_h = _bootstrap_three_systems_patient(
        target_h, majority_h, weighted_h, comparator_h, patient_h,
        replicates=REPLICATES, seed=SEED,
    )

    point_i = {
        "majority_minus_densenet169": _metric(target_i, majority_i) - _metric(target_i, comparator_i),
        "weighted_minus_densenet169": _metric(target_i, weighted_i) - _metric(target_i, comparator_i),
        "weighted_minus_majority": _metric(target_i, weighted_i) - _metric(target_i, majority_i),
    }
    point_h = {
        "majority_minus_densenet169": _metric(target_h, majority_h) - _metric(target_h, comparator_h),
        "weighted_minus_densenet169": _metric(target_h, weighted_h) - _metric(target_h, comparator_h),
        "weighted_minus_majority": _metric(target_h, weighted_h) - _metric(target_h, majority_h),
    }

    inference = pd.DataFrame(
        _inference_rows(
            "ISIC internal test", boot_i, point_i, "paired image bootstrap"
        )
        + _inference_rows(
            "HIBA external", boot_h, point_h, "patient-cluster bootstrap"
        )
    )

    mcnemar_rows: list[dict[str, object]] = []
    for name, a, b in (
        ("majority_vs_densenet169", majority_i, comparator_i),
        ("weighted_vs_densenet169", weighted_i, comparator_i),
        ("weighted_vs_majority", weighted_i, majority_i),
    ):
        row = paired_correctness_mcnemar(target_i, a, b)
        mcnemar_rows.append({"comparison": name, **row})
    mcnemar = pd.DataFrame(mcnemar_rows)

    internal_full = pd.read_csv(INTERNAL_DIR / "full_coverage_metrics.csv")
    hiba_full = pd.read_csv(HIBA_DIR / "full_coverage_metrics.csv")
    internal_ref = pd.read_csv(INTERNAL_DIR / "selective_referral_metrics.csv")
    hiba_ref = pd.read_csv(HIBA_DIR / "selective_referral_metrics.csv")
    internal_agree = pd.read_csv(INTERNAL_DIR / "agreement_strata.csv")
    hiba_agree = pd.read_csv(HIBA_DIR / "agreement_strata.csv")

    full = _full_coverage_table(internal_full, hiba_full)
    referral = _referral_table(internal_ref, hiba_ref)
    agreement = _agreement_table(internal_agree, hiba_agree)

    output.mkdir(parents=True, exist_ok=False)
    boot_i.to_csv(output / "bootstrap_internal_isic_10000.csv", index=False)
    boot_h.to_csv(output / "bootstrap_hiba_patient_cluster_10000.csv", index=False)
    inference.to_csv(output / "table_inferential_macro_f1.csv", index=False)
    mcnemar.to_csv(output / "table_internal_mcnemar.csv", index=False)
    full.to_csv(output / "table_full_coverage_metrics.csv", index=False)
    referral.to_csv(output / "table_selective_referral.csv", index=False)
    agreement.to_csv(output / "table_agreement_strata.csv", index=False)

    _figure_full_coverage(full, output / "fig1_full_coverage_macro_f1.png")
    _figure_risk_coverage(
        internal_ref, hiba_ref, internal_full, hiba_full,
        output / "fig2_risk_coverage.png",
    )
    _figure_agreement(
        internal_agree, hiba_agree,
        output / "fig3_agreement_distribution.png",
    )
    _figure_macro_f1_coverage(
        internal_ref, hiba_ref, internal_full, hiba_full,
        output / "fig4_macro_f1_coverage.png",
    )

    primary_internal = inference[
        (inference["cohort"] == "ISIC internal test")
        & (inference["comparison"] == "majority_minus_densenet169")
    ].iloc[0]
    primary_hiba = inference[
        (inference["cohort"] == "HIBA external")
        & (inference["comparison"] == "majority_minus_densenet169")
    ].iloc[0]
    primary_mcnemar = mcnemar[
        mcnemar["comparison"] == "majority_vs_densenet169"
    ].iloc[0]

    manifest = {
        "phase": "ITT-06",
        "status": "PASS",
        "replicates": REPLICATES,
        "seed": SEED,
        "internal_bootstrap_unit": "image",
        "hiba_bootstrap_unit": "patient",
        "hiba_patient_count": 568,
        "input_hashes": {path.as_posix(): sha256_file(path) for path in required},
        "outputs": sorted(path.name for path in output.iterdir()),
        "interpretation_boundary": (
            "Bootstrap intervals quantify the prespecified macro-F1 differences. "
            "Internal McNemar assesses paired correctness and is a different estimand. "
            "Selective-referral threshold results remain descriptive; no threshold is selected."
        ),
    }
    _write_json(output / "itt06_execution_manifest.json", manifest)

    concise = {
        "status": "PASS",
        "primary_internal": {
            "macro_f1_difference": float(primary_internal["point_difference_macro_f1"]),
            "ci95": [
                float(primary_internal["ci95_lower"]),
                float(primary_internal["ci95_upper"]),
            ],
            "ci_excludes_zero": bool(primary_internal["ci_excludes_zero"]),
            "mcnemar_p": float(primary_mcnemar["exact_two_sided_p_value"]),
            "a_correct_b_wrong": int(primary_mcnemar["a_correct_b_wrong"]),
            "a_wrong_b_correct": int(primary_mcnemar["a_wrong_b_correct"]),
        },
        "primary_hiba": {
            "macro_f1_difference": float(primary_hiba["point_difference_macro_f1"]),
            "patient_cluster_ci95": [
                float(primary_hiba["ci95_lower"]),
                float(primary_hiba["ci95_upper"]),
            ],
            "ci_excludes_zero": bool(primary_hiba["ci_excludes_zero"]),
        },
        "weighted_equals_majority_point_macro_f1": {
            "internal_difference": point_i["weighted_minus_majority"],
            "hiba_difference": point_h["weighted_minus_majority"],
        },
        "output_directory": output.as_posix(),
        "figures": [
            "fig1_full_coverage_macro_f1.png",
            "fig2_risk_coverage.png",
            "fig3_agreement_distribution.png",
            "fig4_macro_f1_coverage.png",
        ],
    }
    print(json.dumps(concise, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
