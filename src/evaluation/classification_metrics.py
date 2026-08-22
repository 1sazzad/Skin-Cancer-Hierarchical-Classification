"""Imbalance-sensitive classification metrics with JSON-safe outputs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_fscore_support,
    recall_score,
)


def compute_classification_metrics(
    targets: Sequence[int] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
    class_names: Sequence[str],
) -> dict[str, object]:
    """Compute primary project metrics for one classification stage."""

    true_values = np.asarray(targets, dtype=np.int64)
    predicted_values = np.asarray(predictions, dtype=np.int64)
    names = list(class_names)

    if true_values.ndim != 1 or predicted_values.ndim != 1:
        raise ValueError("targets and predictions must be one-dimensional.")
    if true_values.size == 0:
        raise ValueError("targets must not be empty.")
    if true_values.shape != predicted_values.shape:
        raise ValueError("targets and predictions must have matching shapes.")
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("class_names must contain at least two unique names.")

    labels = np.arange(len(names), dtype=np.int64)
    observed = set(true_values.tolist()) | set(predicted_values.tolist())
    unsupported = sorted(observed - set(labels.tolist()))
    if unsupported:
        raise ValueError(f"Observed class indices outside class_names: {unsupported}")

    precision, recall, class_f1, support = precision_recall_fscore_support(
        true_values,
        predicted_values,
        labels=labels,
        zero_division=0,
    )
    matrix = confusion_matrix(true_values, predicted_values, labels=labels)

    return {
        "sample_count": int(true_values.size),
        "accuracy": float(accuracy_score(true_values, predicted_values)),
        "balanced_accuracy": float(
            balanced_accuracy_score(true_values, predicted_values)
        ),
        "macro_f1": float(
            f1_score(true_values, predicted_values, average="macro", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(
                true_values,
                predicted_values,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                true_values,
                predicted_values,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                true_values,
                predicted_values,
                average="weighted",
                zero_division=0,
            )
        ),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(class_f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(names)
        },
        "confusion_matrix": matrix.astype(int).tolist(),
        "class_names": names,
    }
