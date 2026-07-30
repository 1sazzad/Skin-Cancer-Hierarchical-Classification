"""Paired stored-prediction statistics for the Phase 11 DenseNet-121 baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.analysis.stored_prediction_statistics import (
    CLASSES,
    EXPECTED_SUPPORT,
    LABELS,
    StatisticalAnalysisError,
    calculate_metrics,
    exact_mcnemar,
    linear_quantile,
    stratified_bootstrap_indices,
)


METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StatisticalAnalysisError(message)


def _normalize_source(
    frame: pd.DataFrame,
    *,
    name: str,
    id_column: str,
    target_index_column: str,
    target_label_column: str,
    prediction_index_column: str,
    prediction_label_column: str,
    expected_count: int,
) -> pd.DataFrame:
    required = {
        id_column,
        target_index_column,
        target_label_column,
        prediction_index_column,
        prediction_label_column,
    }
    _require(required.issubset(frame.columns), f"{name}: missing required columns.")
    _require(len(frame) == expected_count, f"{name}: expected {expected_count} rows.")
    identifiers = frame[id_column]
    _require(not identifiers.isna().any(), f"{name}: missing sample identifier.")
    identifiers = identifiers.astype(str).str.strip()
    _require(identifiers.ne("").all(), f"{name}: missing sample identifier.")
    _require(not identifiers.duplicated().any(), f"{name}: duplicate sample identifier.")

    result = pd.DataFrame({"sample_id": identifiers})
    mapping = dict(enumerate(CLASSES))
    for role, index_column, label_column in (
        ("target", target_index_column, target_label_column),
        ("prediction", prediction_index_column, prediction_label_column),
    ):
        numeric = pd.to_numeric(frame[index_column], errors="coerce")
        _require(
            numeric.notna().all() and np.isfinite(numeric.to_numpy(dtype=float)).all(),
            f"{name}: non-finite or invalid {role} index.",
        )
        integers = numeric.to_numpy(dtype=np.int64)
        _require(
            np.equal(numeric.to_numpy(dtype=float), integers).all()
            and np.isin(integers, LABELS).all(),
            f"{name}: unsupported {role} index.",
        )
        labels = frame[label_column].astype(str).str.strip().str.lower()
        _require(
            labels.isin(CLASSES).all(), f"{name}: unsupported {role} label."
        )
        expected_labels = pd.Series(integers, index=frame.index).map(mapping)
        _require(
            np.array_equal(labels.to_numpy(), expected_labels.to_numpy()),
            f"{name}: {role} label/index mismatch.",
        )
        result[f"{role}_index"] = integers
        result[f"{role}_label"] = labels.to_numpy()
    return result.sort_values("sample_id", kind="stable").reset_index(drop=True)


def align_predictions(
    densenet: pd.DataFrame,
    manifest: pd.DataFrame,
    hierarchy: pd.DataFrame,
    *,
    expected_count: int = 3668,
    expected_support: np.ndarray = EXPECTED_SUPPORT,
) -> pd.DataFrame:
    """Validate and align all systems by sample ID, never input row order."""
    probability_columns = [
        column for column in densenet.columns if column.startswith("probability_")
    ]
    _require(len(probability_columns) == 4, "DenseNet: probability columns are incomplete.")
    probabilities = densenet[probability_columns].apply(pd.to_numeric, errors="coerce")
    _require(
        np.isfinite(probabilities.to_numpy(dtype=float)).all(),
        "DenseNet: non-finite probability value.",
    )
    dense = _normalize_source(
        densenet,
        name="DenseNet",
        id_column="image_id",
        target_index_column="target_index",
        target_label_column="target_label",
        prediction_index_column="predicted_index",
        prediction_label_column="predicted_label",
        expected_count=expected_count,
    )
    flat = _normalize_source(
        manifest,
        name="paired manifest flat",
        id_column="sample_id",
        target_index_column="target_index",
        target_label_column="target_label",
        prediction_index_column="flat_predicted_index",
        prediction_label_column="flat_predicted_label",
        expected_count=expected_count,
    )
    hierarchical_manifest = _normalize_source(
        manifest,
        name="paired manifest hierarchy",
        id_column="sample_id",
        target_index_column="target_index",
        target_label_column="target_label",
        prediction_index_column="hierarchical_predicted_index",
        prediction_label_column="hierarchical_predicted_label",
        expected_count=expected_count,
    )
    hierarchical_raw = _normalize_source(
        hierarchy,
        name="hierarchy",
        id_column="image_id",
        target_index_column="final_target_index",
        target_label_column="final_target_label",
        prediction_index_column="predicted_gate_predicted_index",
        prediction_label_column="predicted_gate_predicted_label",
        expected_count=expected_count,
    )
    id_sets = [set(x["sample_id"]) for x in (dense, flat, hierarchical_manifest, hierarchical_raw)]
    _require(all(ids == id_sets[0] for ids in id_sets[1:]), "Prediction sample-ID sets differ.")
    for source_name, source in (
        ("flat", flat),
        ("hierarchy manifest", hierarchical_manifest),
        ("hierarchy raw", hierarchical_raw),
    ):
        _require(
            np.array_equal(dense["target_index"], source["target_index"])
            and np.array_equal(dense["target_label"], source["target_label"]),
            f"Target-label mismatch between DenseNet and {source_name}.",
        )
    _require(
        np.array_equal(
            hierarchical_manifest["prediction_index"],
            hierarchical_raw["prediction_index"],
        ),
        "Hierarchy predictions differ between manifest and locked raw artifact.",
    )
    support = np.bincount(dense["target_index"].to_numpy(), minlength=4)
    _require(np.array_equal(support, expected_support), "Ground-truth support changed.")
    return pd.DataFrame(
        {
            "sample_id": dense["sample_id"],
            "target_index": dense["target_index"],
            "target_label": dense["target_label"],
            "densenet_predicted_index": dense["prediction_index"],
            "flat_predicted_index": flat["prediction_index"],
            "hierarchy_predicted_index": hierarchical_raw["prediction_index"],
        }
    )


def paired_comparison(
    target: np.ndarray,
    densenet_prediction: np.ndarray,
    comparator_prediction: np.ndarray,
    *,
    comparator_name: str,
    replicate_count: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Calculate point estimates, paired percentile CIs, and exact McNemar."""
    dense_metrics = calculate_metrics(target, densenet_prediction)
    comparator_metrics = calculate_metrics(target, comparator_prediction)
    indices = stratified_bootstrap_indices(target, replicate_count, seed)
    differences = np.empty((replicate_count, len(METRICS)), dtype=np.float64)
    for replicate, selected in enumerate(indices):
        left = calculate_metrics(target[selected], densenet_prediction[selected])
        right = calculate_metrics(target[selected], comparator_prediction[selected])
        differences[replicate] = [
            getattr(left, metric) - getattr(right, metric) for metric in METRICS
        ]
    _require(np.isfinite(differences).all(), "Non-finite bootstrap result.")
    difference_rows = []
    for column, metric in enumerate(METRICS):
        point = getattr(dense_metrics, metric) - getattr(comparator_metrics, metric)
        lower, upper = linear_quantile(differences[:, column], [0.025, 0.975])
        difference_rows.append(
            {
                "comparison": f"densenet121_vs_{comparator_name}",
                "metric": metric,
                "densenet121": getattr(dense_metrics, metric),
                "comparator": getattr(comparator_metrics, metric),
                "difference_densenet121_minus_comparator": point,
                "ci_lower": lower,
                "ci_upper": upper,
                "confidence_level": 0.95,
                "replicate_count": replicate_count,
                "seed": seed,
            }
        )
    dense_correct = densenet_prediction == target
    comparator_correct = comparator_prediction == target
    raw_mcnemar = exact_mcnemar(dense_correct, comparator_correct)
    mcnemar = {
        "comparison": f"densenet121_vs_{comparator_name}",
        "both_correct": raw_mcnemar["both_correct"],
        "densenet121_only_correct": raw_mcnemar["flat_correct_hierarchy_wrong"],
        "comparator_only_correct": raw_mcnemar["flat_wrong_hierarchy_correct"],
        "both_incorrect": raw_mcnemar["both_wrong"],
        "discordant_pairs": raw_mcnemar["discordant_pairs"],
        "exact_two_sided_p_value": raw_mcnemar["exact_two_sided_p_value"],
    }
    return {
        "densenet_metrics": dense_metrics,
        "comparator_metrics": comparator_metrics,
        "differences": difference_rows,
        "mcnemar": mcnemar,
    }
