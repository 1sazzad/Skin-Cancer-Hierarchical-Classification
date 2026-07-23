from __future__ import annotations

import pytest

from src.evaluation.classification_metrics import compute_classification_metrics


def test_classification_metrics_return_project_primary_metrics() -> None:
    metrics = compute_classification_metrics(
        targets=[0, 0, 1, 1],
        predictions=[0, 1, 1, 1],
        class_names=["non_malignant", "malignant"],
    )

    assert metrics["sample_count"] == 4
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert metrics["per_class"]["malignant"]["recall"] == pytest.approx(1.0)
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]


def test_classification_metrics_reject_out_of_range_class_index() -> None:
    with pytest.raises(ValueError, match="outside class_names"):
        compute_classification_metrics(
            targets=[0, 2],
            predictions=[0, 1],
            class_names=["a", "b"],
        )
