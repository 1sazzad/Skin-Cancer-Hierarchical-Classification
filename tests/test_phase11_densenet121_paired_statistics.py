from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.densenet121_paired_statistics import align_predictions, paired_comparison
from src.analysis.stored_prediction_statistics import CLASSES, StatisticalAnalysisError


def _sources():
    ids = ["d", "a", "c", "b"]
    target = [3, 0, 2, 1]
    dense_pred = [3, 0, 0, 1]
    flat_pred = [0, 0, 2, 2]
    hierarchy_pred = [3, 1, 2, 1]
    dense = pd.DataFrame({
        "image_id": ids, "target_index": target,
        "target_label": [CLASSES[x] for x in target],
        "predicted_index": dense_pred,
        "predicted_label": [CLASSES[x] for x in dense_pred],
        **{f"probability_{name}": [0.25] * 4 for name in CLASSES},
    })
    manifest = pd.DataFrame({
        "sample_id": ids, "target_index": target,
        "target_label": [CLASSES[x] for x in target],
        "flat_predicted_index": flat_pred,
        "flat_predicted_label": [CLASSES[x] for x in flat_pred],
        "hierarchical_predicted_index": hierarchy_pred,
        "hierarchical_predicted_label": [CLASSES[x] for x in hierarchy_pred],
    })
    hierarchy = pd.DataFrame({
        "image_id": ids, "final_target_index": target,
        "final_target_label": [CLASSES[x] for x in target],
        "predicted_gate_predicted_index": hierarchy_pred,
        "predicted_gate_predicted_label": [CLASSES[x] for x in hierarchy_pred],
    })
    return dense, manifest, hierarchy


def test_alignment_is_by_id_and_targets_are_consistent():
    dense, manifest, hierarchy = _sources()
    aligned = align_predictions(
        dense.sample(frac=1, random_state=1),
        manifest.sample(frac=1, random_state=2),
        hierarchy.sample(frac=1, random_state=3),
        expected_count=4,
        expected_support=np.ones(4, dtype=int),
    )
    assert aligned["sample_id"].tolist() == ["a", "b", "c", "d"]


def test_duplicate_missing_and_target_mismatch_fail():
    dense, manifest, hierarchy = _sources()
    dense.loc[1, "image_id"] = dense.loc[0, "image_id"]
    with pytest.raises(StatisticalAnalysisError, match="duplicate"):
        align_predictions(dense, manifest, hierarchy, expected_count=4,
                          expected_support=np.ones(4, dtype=int))
    dense, manifest, hierarchy = _sources()
    hierarchy.loc[0, ["final_target_index", "final_target_label"]] = [0, "non_malignant"]
    with pytest.raises(StatisticalAnalysisError, match="Target-label mismatch"):
        align_predictions(dense, manifest, hierarchy, expected_count=4,
                          expected_support=np.ones(4, dtype=int))


def test_bootstrap_is_deterministic_and_difference_direction_is_dense_minus_comparator():
    target = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    dense = np.array([0, 0, 1, 1, 2, 2, 3, 0])
    comparator = np.array([0, 1, 1, 2, 2, 0, 0, 0])
    first = paired_comparison(target, dense, comparator, comparator_name="test",
                              replicate_count=50, seed=42)
    second = paired_comparison(target, dense, comparator, comparator_name="test",
                               replicate_count=50, seed=42)
    assert first["differences"] == second["differences"]
    accuracy = next(row for row in first["differences"] if row["metric"] == "accuracy")
    assert accuracy["difference_densenet121_minus_comparator"] == pytest.approx(4 / 8)


def test_exact_mcnemar_counts():
    target = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    dense = np.array([0, 0, 0, 1, 2, 2, 3, 3])
    comparator = np.array([0, 1, 1, 0, 2, 2, 3, 3])
    result = paired_comparison(target, dense, comparator, comparator_name="test",
                               replicate_count=2, seed=42)["mcnemar"]
    assert result == {
        "comparison": "densenet121_vs_test",
        "both_correct": 5,
        "densenet121_only_correct": 2,
        "comparator_only_correct": 1,
        "both_incorrect": 0,
        "discordant_pairs": 3,
        "exact_two_sided_p_value": 1.0,
    }
