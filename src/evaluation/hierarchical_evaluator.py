"""Pure routing and metric logic for the locked conditional hierarchy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.evaluation.classification_metrics import (
    compute_classification_metrics,
)


STAGE_1_CLASS_NAMES: tuple[str, ...] = (
    "non_malignant",
    "malignant",
)

STAGE_2_CLASS_NAMES: tuple[str, ...] = (
    "melanoma",
    "bcc",
    "scc",
)

FINAL_CLASS_NAMES: tuple[str, ...] = (
    "non_malignant",
    "melanoma",
    "bcc",
    "scc",
)


@dataclass(frozen=True, slots=True)
class HierarchicalRoutingOutcome:
    """Derived predictions and routing counts for one locked evaluation."""

    final_targets: np.ndarray
    oracle_gate_predictions: np.ndarray
    predicted_gate_predictions: np.ndarray
    stage_2_execution_mask: np.ndarray
    routing_counts: dict[str, int | float]


def _integer_vector(
    values: Sequence[int] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")

    return array


def _validate_matching_shapes(
    arrays: dict[str, np.ndarray],
) -> None:
    shapes = {name: value.shape for name, value in arrays.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(
            f"All routing arrays must have matching shapes: {shapes}"
        )


def _validate_allowed_values(
    values: np.ndarray,
    *,
    mask: np.ndarray,
    allowed: set[int],
    name: str,
) -> None:
    observed = set(values[mask].tolist())
    unsupported = sorted(observed - allowed)

    if unsupported:
        raise ValueError(
            f"{name} contains unsupported values: {unsupported}"
        )


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def build_hierarchical_routing(
    stage_1_targets: Sequence[int] | np.ndarray,
    stage_1_predictions: Sequence[int] | np.ndarray,
    stage_2_targets: Sequence[int] | np.ndarray,
    stage_2_predictions: Sequence[int] | np.ndarray,
) -> HierarchicalRoutingOutcome:
    """Compose oracle-gated and predicted-gate four-class predictions.

    Stage 1 class indices:
        0 = non_malignant
        1 = malignant

    Stage 2 class indices:
        0 = melanoma
        1 = bcc
        2 = scc

    Non-malignant rows must use ``stage_2_target == -1``. Stage 2 predictions
    are required only for the union of true-malignant and predicted-malignant
    rows.
    """

    stage_1_true = _integer_vector(
        stage_1_targets,
        name="stage_1_targets",
    )
    stage_1_pred = _integer_vector(
        stage_1_predictions,
        name="stage_1_predictions",
    )
    stage_2_true = _integer_vector(
        stage_2_targets,
        name="stage_2_targets",
    )
    stage_2_pred = _integer_vector(
        stage_2_predictions,
        name="stage_2_predictions",
    )

    _validate_matching_shapes(
        {
            "stage_1_targets": stage_1_true,
            "stage_1_predictions": stage_1_pred,
            "stage_2_targets": stage_2_true,
            "stage_2_predictions": stage_2_pred,
        }
    )

    all_rows = np.ones(stage_1_true.shape, dtype=bool)

    _validate_allowed_values(
        stage_1_true,
        mask=all_rows,
        allowed={0, 1},
        name="stage_1_targets",
    )
    _validate_allowed_values(
        stage_1_pred,
        mask=all_rows,
        allowed={0, 1},
        name="stage_1_predictions",
    )

    true_malignant = stage_1_true == 1
    true_non_malignant = ~true_malignant
    predicted_malignant = stage_1_pred == 1
    predicted_non_malignant = ~predicted_malignant

    _validate_allowed_values(
        stage_2_true,
        mask=true_malignant,
        allowed={0, 1, 2},
        name="stage_2_targets for true malignant rows",
    )

    invalid_non_malignant_targets = (
        true_non_malignant & (stage_2_true != -1)
    )
    if invalid_non_malignant_targets.any():
        raise ValueError(
            "True non-malignant rows must use stage_2_target == -1."
        )

    stage_2_execution_mask = (
        true_malignant | predicted_malignant
    )

    _validate_allowed_values(
        stage_2_pred,
        mask=stage_2_execution_mask,
        allowed={0, 1, 2},
        name="stage_2_predictions for executed rows",
    )

    invalid_unexecuted_predictions = (
        ~stage_2_execution_mask & (stage_2_pred != -1)
    )
    if invalid_unexecuted_predictions.any():
        raise ValueError(
            "Rows outside the Stage 2 execution union must use "
            "stage_2_prediction == -1."
        )

    final_targets = np.zeros_like(stage_1_true)
    final_targets[true_malignant] = (
        stage_2_true[true_malignant] + 1
    )

    oracle_gate_predictions = np.zeros_like(stage_1_true)
    oracle_gate_predictions[true_malignant] = (
        stage_2_pred[true_malignant] + 1
    )

    predicted_gate_predictions = np.zeros_like(stage_1_true)
    predicted_gate_predictions[predicted_malignant] = (
        stage_2_pred[predicted_malignant] + 1
    )

    malignant_blocked = (
        true_malignant & predicted_non_malignant
    )
    non_malignant_incorrectly_routed = (
        true_non_malignant & predicted_malignant
    )
    correctly_routed_malignant = (
        true_malignant & predicted_malignant
    )

    subtype_correct = (
        correctly_routed_malignant
        & (stage_2_true == stage_2_pred)
    )
    subtype_error = (
        correctly_routed_malignant
        & (stage_2_true != stage_2_pred)
    )

    sample_count = int(stage_1_true.size)
    true_malignant_count = int(true_malignant.sum())
    true_non_malignant_count = int(true_non_malignant.sum())
    correctly_routed_count = int(
        correctly_routed_malignant.sum()
    )
    stage_2_execution_count = int(
        stage_2_execution_mask.sum()
    )
    blocked_count = int(malignant_blocked.sum())
    incorrect_route_count = int(
        non_malignant_incorrectly_routed.sum()
    )
    subtype_correct_count = int(subtype_correct.sum())
    subtype_error_count = int(subtype_error.sum())

    routing_counts: dict[str, int | float] = {
        "sample_count": sample_count,
        "true_non_malignant_count": true_non_malignant_count,
        "true_malignant_count": true_malignant_count,
        "predicted_malignant_count": int(
            predicted_malignant.sum()
        ),
        "stage_2_execution_count": stage_2_execution_count,
        "stage_2_execution_fraction": _safe_rate(
            stage_2_execution_count,
            sample_count,
        ),
        "malignant_blocked_by_stage_1": blocked_count,
        "malignant_block_rate": _safe_rate(
            blocked_count,
            true_malignant_count,
        ),
        "non_malignant_incorrectly_routed_to_stage_2": (
            incorrect_route_count
        ),
        "non_malignant_incorrect_route_rate": _safe_rate(
            incorrect_route_count,
            true_non_malignant_count,
        ),
        "correctly_routed_malignant": correctly_routed_count,
        "subtype_correct_after_correct_route": (
            subtype_correct_count
        ),
        "subtype_error_after_correct_route": (
            subtype_error_count
        ),
        "subtype_error_rate_after_correct_route": _safe_rate(
            subtype_error_count,
            correctly_routed_count,
        ),
    }

    return HierarchicalRoutingOutcome(
        final_targets=final_targets,
        oracle_gate_predictions=oracle_gate_predictions,
        predicted_gate_predictions=predicted_gate_predictions,
        stage_2_execution_mask=stage_2_execution_mask,
        routing_counts=routing_counts,
    )


def compute_hierarchical_evaluation(
    stage_1_targets: Sequence[int] | np.ndarray,
    stage_1_predictions: Sequence[int] | np.ndarray,
    stage_2_targets: Sequence[int] | np.ndarray,
    stage_2_predictions: Sequence[int] | np.ndarray,
) -> dict[str, object]:
    """Compute all required standalone, oracle, and end-to-end metrics."""

    routing = build_hierarchical_routing(
        stage_1_targets,
        stage_1_predictions,
        stage_2_targets,
        stage_2_predictions,
    )

    stage_1_true = np.asarray(
        stage_1_targets,
        dtype=np.int64,
    )
    stage_1_pred = np.asarray(
        stage_1_predictions,
        dtype=np.int64,
    )
    stage_2_true = np.asarray(
        stage_2_targets,
        dtype=np.int64,
    )
    stage_2_pred = np.asarray(
        stage_2_predictions,
        dtype=np.int64,
    )

    true_malignant = stage_1_true == 1

    return {
        "protocol": {
            "gate_policy": "argmax",
            "stage_2_execution_policy": (
                "union_of_true_and_predicted_malignant"
            ),
            "stage_1_class_names": list(
                STAGE_1_CLASS_NAMES
            ),
            "stage_2_class_names": list(
                STAGE_2_CLASS_NAMES
            ),
            "final_class_names": list(FINAL_CLASS_NAMES),
        },
        "standalone_stage_1": (
            compute_classification_metrics(
                stage_1_true,
                stage_1_pred,
                STAGE_1_CLASS_NAMES,
            )
        ),
        "oracle_gated_stage_2": (
            compute_classification_metrics(
                stage_2_true[true_malignant],
                stage_2_pred[true_malignant],
                STAGE_2_CLASS_NAMES,
            )
        ),
        "oracle_gate_four_class": (
            compute_classification_metrics(
                routing.final_targets,
                routing.oracle_gate_predictions,
                FINAL_CLASS_NAMES,
            )
        ),
        "predicted_gate_end_to_end": (
            compute_classification_metrics(
                routing.final_targets,
                routing.predicted_gate_predictions,
                FINAL_CLASS_NAMES,
            )
        ),
        "routing": routing.routing_counts,
    }
