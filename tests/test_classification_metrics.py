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
    assert metrics["macro_precision"] == pytest.approx((1.0 + 2 / 3) / 2)
    assert metrics["macro_recall"] == pytest.approx(0.75)
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert metrics["weighted_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert metrics["per_class"]["malignant"]["recall"] == pytest.approx(1.0)
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["class_names"] == ["non_malignant", "malignant"]


def test_classification_metrics_use_fixed_class_order_and_zero_for_absent_class() -> None:
    metrics = compute_classification_metrics(
        targets=[0, 0, 1],
        predictions=[0, 1, 1],
        class_names=["non_malignant", "melanoma", "bcc", "scc"],
    )

    assert list(metrics["per_class"]) == [
        "non_malignant",
        "melanoma",
        "bcc",
        "scc",
    ]
    assert metrics["macro_precision"] == pytest.approx((1.0 + 0.5 + 0.0 + 0.0) / 4)
    assert metrics["macro_recall"] == pytest.approx((0.5 + 1.0 + 0.0 + 0.0) / 4)
    assert metrics["per_class"]["bcc"] == {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "support": 0,
    }
    assert metrics["confusion_matrix"] == [
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]


def test_classification_metrics_reject_out_of_range_class_index() -> None:
    with pytest.raises(ValueError, match="outside class_names"):
        compute_classification_metrics(
            targets=[0, 2],
            predictions=[0, 1],
            class_names=["a", "b"],
        )
