from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.itt2026_consensus import (
    CohortSpec,
    FROZEN_MODELS,
    FROZEN_VALIDATION_RANKING,
    FROZEN_VALIDATION_WEIGHTS,
    ITTConsensusError,
    build_consensus,
    full_coverage_metrics,
    paired_correctness_mcnemar,
    paired_image_bootstrap_macro_f1_difference,
    patient_cluster_bootstrap_macro_f1_difference,
    selective_referral_metrics,
    validate_and_align_prediction_frames,
)


def _frames() -> dict[str, pd.DataFrame]:
    targets = [0, 1, 2, 3]
    frames: dict[str, pd.DataFrame] = {}
    for index, model in enumerate(FROZEN_MODELS):
        frames[model] = pd.DataFrame(
            {
                "sample_id": ["a", "b", "c", "d"],
                "target": targets,
                "prediction": [
                    0,
                    1 if index < 5 else 2,
                    2 if index < 4 else 1,
                    3 if index < 6 else 0,
                ],
            }
        )
    return frames


def test_validate_and_align_prediction_frames_accepts_matching_synthetic_frames() -> None:
    aligned = validate_and_align_prediction_frames(
        _frames(),
        CohortSpec(
            id_field="sample_id",
            target_field="target",
            prediction_field="prediction",
            expected_rows=4,
        ),
    )
    assert list(aligned["sample_id"]) == ["a", "b", "c", "d"]
    assert aligned.shape == (4, 9)
    assert [column for column in aligned.columns if column.startswith("pred__")] == [
        f"pred__{model}" for model in FROZEN_MODELS
    ]


def test_alignment_rejects_duplicate_ids() -> None:
    frames = _frames()
    frames["densenet121"].loc[1, "sample_id"] = "a"
    with pytest.raises(ITTConsensusError, match="duplicate"):
        validate_and_align_prediction_frames(
            frames,
            CohortSpec("sample_id", "target", "prediction", expected_rows=4),
        )


def test_alignment_rejects_cross_model_target_mismatch() -> None:
    frames = _frames()
    frames["resnet50"].loc[2, "target"] = 1
    with pytest.raises(ITTConsensusError, match="ground-truth"):
        validate_and_align_prediction_frames(
            frames,
            CohortSpec("sample_id", "target", "prediction", expected_rows=4),
        )


def test_majority_vote_and_agreement_are_frozen_unweighted_hard_votes() -> None:
    matrix = np.array(
        [
            [1, 1, 2, 1, 1, 2, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [3, 3, 3, 3, 2, 2, 2],
        ],
        dtype=np.int64,
    )
    output = build_consensus(matrix)
    assert output.majority_prediction.tolist() == [1, 0, 3]
    assert output.max_vote_count.tolist() == [5, 7, 4]
    assert output.agreement.tolist() == pytest.approx([5 / 7, 1.0, 4 / 7])
    assert output.disagreement.tolist() == pytest.approx([2 / 7, 0.0, 3 / 7])
    assert not output.majority_tie_required.any()


def test_majority_2221_tie_uses_validation_weight_sum() -> None:
    # class 0: DenseNet121 + DenseNet169
    # class 1: ResNet50 + MobileNetV3
    # class 2: EfficientNet-B0 + EfficientNet-B2
    # class 3: EfficientNet-B3
    # Frozen validation sums make class 1 the winner among the three tied 2-vote classes.
    matrix = np.array([[0, 0, 1, 1, 2, 2, 3]], dtype=np.int64)
    output = build_consensus(matrix)
    assert output.vote_counts.tolist() == [[2, 2, 2, 1]]
    assert output.majority_tie_required.tolist() == [True]
    assert output.majority_prediction.tolist() == [1]
    assert output.max_vote_count.tolist() == [2]


def test_exact_weight_tie_falls_back_to_validation_model_ranking() -> None:
    weights = {model: 1.0 for model in FROZEN_MODELS}
    matrix = np.array([[0, 0, 1, 1, 2, 2, 3]], dtype=np.int64)
    output = build_consensus(
        matrix,
        weights=weights,
        ranking=FROZEN_VALIDATION_RANKING,
    )
    # DenseNet169 is highest-ranked and votes class 0.
    assert output.majority_prediction.tolist() == [0]
    assert output.weighted_prediction.tolist() == [0]


def test_weighted_vote_can_differ_from_unweighted_majority() -> None:
    custom = dict(FROZEN_VALIDATION_WEIGHTS)
    custom["densenet121"] = 0.50
    custom["densenet169"] = 0.20
    for model in FROZEN_MODELS[2:]:
        custom[model] = 0.06
    # Five models vote class 1, while two high-weight DenseNets vote class 2.
    matrix = np.array([[2, 2, 1, 1, 1, 1, 1]], dtype=np.int64)
    output = build_consensus(matrix, weights=custom)
    assert output.majority_prediction.tolist() == [1]
    assert output.weighted_prediction.tolist() == [2]


def test_full_metrics_keep_all_four_classes_even_when_some_are_absent() -> None:
    metrics = full_coverage_metrics([0, 0, 1], [0, 1, 1])
    assert list(metrics["per_class"]) == [
        "non_malignant",
        "melanoma",
        "bcc",
        "scc",
    ]
    assert metrics["per_class"]["bcc"]["support"] == 0
    assert metrics["per_class"]["bcc"]["f1"] == 0.0
    assert metrics["macro_f1"] == pytest.approx(((2 / 3) + (2 / 3)) / 4)


def test_selective_referral_reports_coverage_and_class_specific_coverage() -> None:
    target = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    prediction = np.array([0, 0, 1, 0, 2, 1, 3, 2])
    max_votes = np.array([7, 6, 7, 4, 6, 4, 5, 4])
    rows = selective_referral_metrics(target, prediction, max_votes)
    by_threshold = {row["threshold_votes"]: row for row in rows}

    assert by_threshold[7]["retained_count"] == 2
    assert by_threshold[7]["coverage"] == pytest.approx(0.25)
    assert by_threshold[6]["retained_count"] == 4
    assert by_threshold[5]["retained_count"] == 5
    assert by_threshold[4]["retained_count"] == 8
    assert by_threshold[6]["class_specific_coverage"]["non_malignant"] == pytest.approx(1.0)
    assert by_threshold[6]["class_specific_coverage"]["scc"] == pytest.approx(0.0)
    assert by_threshold[6]["malignant_coverage"] == pytest.approx(2 / 6)


def test_selective_referral_rejects_unfrozen_threshold() -> None:
    with pytest.raises(ITTConsensusError, match="Unfrozen"):
        selective_referral_metrics(
            [0, 1],
            [0, 1],
            [7, 6],
            thresholds=[3],
        )


def test_image_bootstrap_is_deterministic_and_paired() -> None:
    target = [0, 0, 1, 1, 2, 2, 3, 3]
    a = [0, 0, 1, 1, 2, 2, 3, 2]
    b = [0, 1, 1, 0, 2, 1, 3, 0]
    first = paired_image_bootstrap_macro_f1_difference(
        target, a, b, replicate_count=40, seed=42
    )
    second = paired_image_bootstrap_macro_f1_difference(
        target, a, b, replicate_count=40, seed=42
    )
    assert first == second
    assert first["point_difference_a_minus_b"] > 0


def test_patient_cluster_bootstrap_is_deterministic_and_uses_patient_clusters() -> None:
    target = [0, 0, 1, 1, 2, 2, 3, 3]
    a = [0, 0, 1, 1, 2, 2, 3, 2]
    b = [0, 1, 1, 0, 2, 1, 3, 0]
    patient = ["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"]
    first = patient_cluster_bootstrap_macro_f1_difference(
        target, a, b, patient, replicate_count=40, seed=42
    )
    second = patient_cluster_bootstrap_macro_f1_difference(
        target, a, b, patient, replicate_count=40, seed=42
    )
    assert first == second
    assert first["patient_count"] == 4
    assert first["point_difference_a_minus_b"] > 0


def test_mcnemar_wrapper_reports_a_minus_b_direction() -> None:
    result = paired_correctness_mcnemar(
        target=[0, 1, 2, 3],
        prediction_a=[0, 1, 2, 0],
        prediction_b=[0, 0, 2, 0],
    )
    assert result["both_correct"] == 2
    assert result["a_correct_b_wrong"] == 1
    assert result["a_wrong_b_correct"] == 0
    assert result["paired_accuracy_difference_a_minus_b"] == pytest.approx(0.25)
