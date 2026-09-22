"""ITT 2026 consensus, disagreement, selective-referral, and paired inference utilities.

This module is deliberately dataset-agnostic. Phase ITT-02 unit tests use only
synthetic fixtures; locked ISIC/HIBA evaluation is performed in later phases.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.analysis.stored_prediction_statistics import calculate_metrics, exact_mcnemar

CLASS_NAMES = ("non_malignant", "melanoma", "bcc", "scc")
LABELS = np.arange(4, dtype=np.int64)
FROZEN_MODELS = (
    "densenet121",
    "densenet169",
    "resnet50",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "efficientnet_b2",
    "efficientnet_b3",
)
FROZEN_VALIDATION_MACRO_F1 = {
    "densenet121": 0.6449820791,
    "densenet169": 0.6659647727,
    "resnet50": 0.6533194436,
    "mobilenet_v3_large": 0.6582407229,
    "efficientnet_b0": 0.6535716654,
    "efficientnet_b2": 0.6544148913,
    "efficientnet_b3": 0.6583008001,
}
FROZEN_VALIDATION_RANKING = (
    "densenet169",
    "efficientnet_b3",
    "mobilenet_v3_large",
    "efficientnet_b2",
    "efficientnet_b0",
    "resnet50",
    "densenet121",
)
FROZEN_VALIDATION_WEIGHTS = {
    model: score / sum(FROZEN_VALIDATION_MACRO_F1.values())
    for model, score in FROZEN_VALIDATION_MACRO_F1.items()
}
REFERRAL_THRESHOLDS = (7, 6, 5, 4)


class ITTConsensusError(ValueError):
    """Raised when a frozen ITT protocol invariant is violated."""


@dataclass(frozen=True)
class CohortSpec:
    """Schema and size invariants for one stored-prediction cohort."""

    id_field: str
    target_field: str
    prediction_field: str
    expected_rows: int | None = None
    patient_field: str | None = None
    expected_patients: int | None = None


@dataclass(frozen=True)
class ConsensusOutput:
    """Per-sample outputs from the two frozen consensus methods."""

    majority_prediction: np.ndarray
    weighted_prediction: np.ndarray
    max_vote_count: np.ndarray
    agreement: np.ndarray
    disagreement: np.ndarray
    majority_tie_required: np.ndarray
    vote_counts: np.ndarray
    weighted_scores: np.ndarray


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ITTConsensusError(message)


def sha256_file(path: Path) -> str:
    """Return SHA-256 for one immutable evidence file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coerce_labels(values: pd.Series, description: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    _require(not numeric.isna().any(), f"Missing or invalid labels in {description}.")
    array = numeric.to_numpy(dtype=np.int64)
    _require(np.isin(array, LABELS).all(), f"Unsupported labels in {description}.")
    return array


def validate_and_align_prediction_frames(
    frames: Mapping[str, pd.DataFrame],
    spec: CohortSpec,
    *,
    model_order: Sequence[str] = FROZEN_MODELS,
) -> pd.DataFrame:
    """Validate seven model files and return one row-aligned prediction table."""
    models = tuple(model_order)
    _require(models == FROZEN_MODELS, "Model order/set differs from the frozen seven-model set.")
    _require(set(frames) == set(models), "Prediction frames do not match the frozen model set.")

    normalized: dict[str, pd.DataFrame] = {}
    canonical_ids: set[str] | None = None
    canonical_target: dict[str, int] | None = None
    canonical_patient: dict[str, str] | None = None

    for model in models:
        frame = frames[model].copy()
        required = {spec.id_field, spec.target_field, spec.prediction_field}
        if spec.patient_field is not None:
            required.add(spec.patient_field)
        _require(required.issubset(frame.columns), f"{model}: required prediction columns are missing.")
        if spec.expected_rows is not None:
            _require(len(frame) == spec.expected_rows, f"{model}: expected {spec.expected_rows} rows, observed {len(frame)}.")

        ids = frame[spec.id_field].astype(str).str.strip()
        _require(ids.ne("").all(), f"{model}: blank sample/image identifier.")
        _require(not ids.duplicated().any(), f"{model}: duplicate sample/image identifier.")

        target = _coerce_labels(frame[spec.target_field], f"{model}.{spec.target_field}")
        prediction = _coerce_labels(frame[spec.prediction_field], f"{model}.{spec.prediction_field}")

        current = pd.DataFrame(
            {
                "sample_id": ids,
                "target": target,
                f"pred__{model}": prediction,
            }
        )
        if spec.patient_field is not None:
            patients = frame[spec.patient_field].astype(str).str.strip()
            _require(patients.ne("").all(), f"{model}: missing patient identifier.")
            current["patient_id"] = patients

        current = current.sort_values("sample_id", kind="stable").reset_index(drop=True)
        id_set = set(current["sample_id"])
        target_map = dict(zip(current["sample_id"], current["target"], strict=True))

        if canonical_ids is None:
            canonical_ids = id_set
            canonical_target = target_map
            if spec.patient_field is not None:
                canonical_patient = dict(
                    zip(current["sample_id"], current["patient_id"], strict=True)
                )
        else:
            _require(id_set == canonical_ids, f"{model}: identifier set differs from the other models.")
            assert canonical_target is not None
            _require(
                all(canonical_target[sample_id] == value for sample_id, value in target_map.items()),
                f"{model}: ground-truth labels disagree across model files.",
            )
            if spec.patient_field is not None:
                assert canonical_patient is not None
                patient_map = dict(zip(current["sample_id"], current["patient_id"], strict=True))
                _require(
                    all(canonical_patient[sample_id] == value for sample_id, value in patient_map.items()),
                    f"{model}: patient IDs disagree across model files.",
                )
        normalized[model] = current

    first = normalized[models[0]][["sample_id", "target"]].copy()
    if spec.patient_field is not None:
        first["patient_id"] = normalized[models[0]]["patient_id"]
    for model in models:
        first[f"pred__{model}"] = normalized[model][f"pred__{model}"].to_numpy(dtype=np.int64)

    if spec.expected_patients is not None:
        _require(spec.patient_field is not None, "expected_patients requires patient_field.")
        observed_patients = int(first["patient_id"].nunique())
        _require(
            observed_patients == spec.expected_patients,
            f"Expected {spec.expected_patients} unique patients, observed {observed_patients}.",
        )
    return first


def prediction_matrix(
    aligned: pd.DataFrame,
    *,
    model_order: Sequence[str] = FROZEN_MODELS,
) -> np.ndarray:
    """Extract the frozen Nx7 prediction matrix from an aligned cohort."""
    columns = [f"pred__{model}" for model in model_order]
    _require(all(column in aligned.columns for column in columns), "Aligned prediction columns are incomplete.")
    matrix = aligned[columns].to_numpy(dtype=np.int64)
    _require(matrix.ndim == 2 and matrix.shape[1] == 7, "Prediction matrix must be Nx7.")
    _require(np.isin(matrix, LABELS).all(), "Prediction matrix contains unsupported labels.")
    return matrix


def _ranking_indices(model_order: Sequence[str], ranking: Sequence[str]) -> tuple[int, ...]:
    models = tuple(model_order)
    ranked = tuple(ranking)
    _require(set(ranked) == set(models) and len(ranked) == len(models), "Tie-break ranking must contain every model exactly once.")
    return tuple(models.index(model) for model in ranked)


def build_consensus(
    predictions: np.ndarray,
    *,
    model_order: Sequence[str] = FROZEN_MODELS,
    weights: Mapping[str, float] = FROZEN_VALIDATION_WEIGHTS,
    ranking: Sequence[str] = FROZEN_VALIDATION_RANKING,
) -> ConsensusOutput:
    """Apply the prospectively frozen majority and validation-weighted votes."""
    matrix = np.asarray(predictions, dtype=np.int64)
    models = tuple(model_order)
    _require(models == FROZEN_MODELS, "Consensus model order/set differs from the frozen protocol.")
    _require(matrix.ndim == 2 and matrix.shape[1] == len(models), "Predictions must have shape Nx7.")
    _require(matrix.shape[0] > 0, "Predictions cannot be empty.")
    _require(np.isin(matrix, LABELS).all(), "Predictions contain unsupported class labels.")
    _require(set(weights) == set(models), "Weight keys differ from the frozen model set.")

    weight_vector = np.asarray([float(weights[model]) for model in models], dtype=np.float64)
    _require(np.isfinite(weight_vector).all() and (weight_vector > 0).all(), "Weights must be finite and positive.")
    rank_indices = _ranking_indices(models, ranking)

    count_matrix = np.stack(
        [np.bincount(row, minlength=4) for row in matrix], axis=0
    ).astype(np.int64)
    weighted_scores = np.zeros((matrix.shape[0], 4), dtype=np.float64)
    for model_index in range(matrix.shape[1]):
        weighted_scores[np.arange(matrix.shape[0]), matrix[:, model_index]] += weight_vector[model_index]

    majority = np.empty(matrix.shape[0], dtype=np.int64)
    weighted = np.empty(matrix.shape[0], dtype=np.int64)
    tie_required = np.zeros(matrix.shape[0], dtype=bool)

    for row_index in range(matrix.shape[0]):
        counts = count_matrix[row_index]
        max_count = int(counts.max())
        tied = np.flatnonzero(counts == max_count)
        if tied.size == 1:
            majority[row_index] = int(tied[0])
        else:
            tie_required[row_index] = True
            tied_scores = weighted_scores[row_index, tied]
            max_tied_score = float(tied_scores.max())
            weighted_tied = tied[np.isclose(tied_scores, max_tied_score, rtol=0.0, atol=1e-15)]
            if weighted_tied.size == 1:
                majority[row_index] = int(weighted_tied[0])
            else:
                allowed = set(int(value) for value in weighted_tied)
                majority[row_index] = next(
                    int(matrix[row_index, model_index])
                    for model_index in rank_indices
                    if int(matrix[row_index, model_index]) in allowed
                )

        scores = weighted_scores[row_index]
        top_score = float(scores.max())
        tied_weighted = np.flatnonzero(
            np.isclose(scores, top_score, rtol=0.0, atol=1e-15)
        )
        if tied_weighted.size == 1:
            weighted[row_index] = int(tied_weighted[0])
        else:
            allowed = set(int(value) for value in tied_weighted)
            weighted[row_index] = next(
                int(matrix[row_index, model_index])
                for model_index in rank_indices
                if int(matrix[row_index, model_index]) in allowed
            )

    max_votes = count_matrix.max(axis=1).astype(np.int64)
    agreement = max_votes.astype(np.float64) / 7.0
    disagreement = 1.0 - agreement
    return ConsensusOutput(
        majority_prediction=majority,
        weighted_prediction=weighted,
        max_vote_count=max_votes,
        agreement=agreement,
        disagreement=disagreement,
        majority_tie_required=tie_required,
        vote_counts=count_matrix,
        weighted_scores=weighted_scores,
    )


def _metrics_to_dict(target: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    metrics = calculate_metrics(target, prediction)
    per_class = {
        name: {
            "precision": float(metrics.precision[index]),
            "recall": float(metrics.recall[index]),
            "f1": float(metrics.f1[index]),
            "support": int(metrics.support[index]),
        }
        for index, name in enumerate(CLASS_NAMES)
    }
    malignant_mask = target > 0
    malignant_sensitivity = (
        float(np.mean(prediction[malignant_mask] > 0))
        if malignant_mask.any()
        else 0.0
    )
    return {
        "sample_count": int(target.size),
        "accuracy": float(metrics.accuracy),
        "balanced_accuracy": float(metrics.balanced_accuracy),
        "macro_f1": float(metrics.macro_f1),
        "weighted_f1": float(metrics.weighted_f1),
        "per_class": per_class,
        "confusion_matrix": metrics.confusion_matrix.astype(int).tolist(),
        "malignant_sensitivity": malignant_sensitivity,
        "scc_sensitivity": float(metrics.recall[3]),
        "scc_f1": float(metrics.f1[3]),
    }


def full_coverage_metrics(target: Sequence[int], prediction: Sequence[int]) -> dict[str, object]:
    """Return fixed-four-class metrics at 100% coverage."""
    target_array = np.asarray(target, dtype=np.int64)
    pred_array = np.asarray(prediction, dtype=np.int64)
    _require(target_array.shape == pred_array.shape and target_array.ndim == 1, "Target/prediction shape mismatch.")
    _require(target_array.size > 0, "Target/prediction arrays cannot be empty.")
    _require(np.isin(target_array, LABELS).all() and np.isin(pred_array, LABELS).all(), "Unsupported class label.")
    return _metrics_to_dict(target_array, pred_array)


def selective_referral_metrics(
    target: Sequence[int],
    majority_prediction: Sequence[int],
    max_vote_count: Sequence[int],
    *,
    thresholds: Sequence[int] = REFERRAL_THRESHOLDS,
) -> list[dict[str, object]]:
    """Evaluate all predeclared majority-agreement referral thresholds."""
    target_array = np.asarray(target, dtype=np.int64)
    prediction = np.asarray(majority_prediction, dtype=np.int64)
    max_votes = np.asarray(max_vote_count, dtype=np.int64)
    _require(target_array.shape == prediction.shape == max_votes.shape, "Selective-referral arrays must align.")
    _require(target_array.ndim == 1 and target_array.size > 0, "Selective-referral arrays must be non-empty 1D.")
    _require(np.isin(target_array, LABELS).all() and np.isin(prediction, LABELS).all(), "Unsupported class label.")
    _require(np.isin(max_votes, [2, 3, 4, 5, 6, 7]).all(), "Invalid maximum vote count for seven four-class voters.")

    rows: list[dict[str, object]] = []
    total_by_class = np.bincount(target_array, minlength=4)
    malignant_total = int(np.sum(target_array > 0))

    for threshold in thresholds:
        _require(int(threshold) in REFERRAL_THRESHOLDS, f"Unfrozen referral threshold: {threshold}")
        accepted = max_votes >= int(threshold)
        retained_count = int(accepted.sum())
        _require(retained_count > 0, f"Referral threshold {threshold}/7 retains no samples.")
        retained_target = target_array[accepted]
        retained_prediction = prediction[accepted]
        metrics = _metrics_to_dict(retained_target, retained_prediction)
        accepted_by_class = np.bincount(retained_target, minlength=4)
        class_coverage = {
            CLASS_NAMES[index]: (
                float(accepted_by_class[index] / total_by_class[index])
                if total_by_class[index] > 0
                else 0.0
            )
            for index in range(4)
        }
        malignant_accepted = int(np.sum(retained_target > 0))
        row = {
            "threshold_votes": int(threshold),
            "threshold_fraction": float(threshold / 7.0),
            "retained_count": retained_count,
            "referred_count": int(target_array.size - retained_count),
            "coverage": float(retained_count / target_array.size),
            "error_rate": float(1.0 - metrics["accuracy"]),
            "class_specific_coverage": class_coverage,
            "malignant_coverage": (
                float(malignant_accepted / malignant_total)
                if malignant_total > 0
                else 0.0
            ),
            **metrics,
        }
        rows.append(row)
    return rows


def _bootstrap_difference(
    target: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    sampled_indices: Sequence[np.ndarray],
) -> np.ndarray:
    values = np.empty(len(sampled_indices), dtype=np.float64)
    for replicate, indices in enumerate(sampled_indices):
        metrics_a = calculate_metrics(target[indices], prediction_a[indices])
        metrics_b = calculate_metrics(target[indices], prediction_b[indices])
        values[replicate] = metrics_a.macro_f1 - metrics_b.macro_f1
    _require(np.isfinite(values).all(), "Non-finite bootstrap macro-F1 difference.")
    return values


def paired_image_bootstrap_macro_f1_difference(
    target: Sequence[int],
    prediction_a: Sequence[int],
    prediction_b: Sequence[int],
    *,
    replicate_count: int = 10000,
    seed: int = 42,
) -> dict[str, object]:
    """Paired image-level bootstrap for macro-F1(A)-macro-F1(B)."""
    target_array = np.asarray(target, dtype=np.int64)
    a = np.asarray(prediction_a, dtype=np.int64)
    b = np.asarray(prediction_b, dtype=np.int64)
    _require(target_array.shape == a.shape == b.shape and target_array.ndim == 1, "Paired bootstrap arrays must align.")
    _require(target_array.size > 0 and replicate_count > 0, "Paired bootstrap inputs must be non-empty.")
    rng = np.random.default_rng(seed)
    samples = [
        rng.integers(0, target_array.size, size=target_array.size, dtype=np.int64)
        for _ in range(replicate_count)
    ]
    values = _bootstrap_difference(target_array, a, b, samples)
    point = _metrics_to_dict(target_array, a)["macro_f1"] - _metrics_to_dict(target_array, b)["macro_f1"]
    lower, upper = np.quantile(values, [0.025, 0.975], method="linear")
    return {
        "point_difference_a_minus_b": float(point),
        "paired_bootstrap_95ci": [float(lower), float(upper)],
        "replicate_count": int(replicate_count),
        "seed": int(seed),
    }


def patient_cluster_bootstrap_macro_f1_difference(
    target: Sequence[int],
    prediction_a: Sequence[int],
    prediction_b: Sequence[int],
    patient_id: Sequence[str],
    *,
    replicate_count: int = 10000,
    seed: int = 42,
) -> dict[str, object]:
    """Patient-cluster bootstrap for HIBA macro-F1(A)-macro-F1(B)."""
    target_array = np.asarray(target, dtype=np.int64)
    a = np.asarray(prediction_a, dtype=np.int64)
    b = np.asarray(prediction_b, dtype=np.int64)
    patients = np.asarray(patient_id, dtype=str)
    _require(target_array.shape == a.shape == b.shape == patients.shape and target_array.ndim == 1, "Cluster-bootstrap arrays must align.")
    _require(target_array.size > 0 and replicate_count > 0, "Cluster-bootstrap inputs must be non-empty.")
    _require(np.all(np.char.str_len(patients) > 0), "Patient IDs cannot be blank.")

    unique_patients = np.unique(patients)
    groups = {patient: np.flatnonzero(patients == patient) for patient in unique_patients}
    rng = np.random.default_rng(seed)
    sampled_indices: list[np.ndarray] = []
    for _ in range(replicate_count):
        sampled_patients = rng.choice(unique_patients, size=unique_patients.size, replace=True)
        sampled_indices.append(
            np.concatenate([groups[patient] for patient in sampled_patients]).astype(np.int64)
        )
    values = _bootstrap_difference(target_array, a, b, sampled_indices)
    point = _metrics_to_dict(target_array, a)["macro_f1"] - _metrics_to_dict(target_array, b)["macro_f1"]
    lower, upper = np.quantile(values, [0.025, 0.975], method="linear")
    return {
        "point_difference_a_minus_b": float(point),
        "patient_cluster_bootstrap_95ci": [float(lower), float(upper)],
        "replicate_count": int(replicate_count),
        "seed": int(seed),
        "patient_count": int(unique_patients.size),
    }


def paired_correctness_mcnemar(
    target: Sequence[int],
    prediction_a: Sequence[int],
    prediction_b: Sequence[int],
) -> dict[str, object]:
    """Exact two-sided McNemar table with A-minus-B naming."""
    target_array = np.asarray(target, dtype=np.int64)
    a = np.asarray(prediction_a, dtype=np.int64)
    b = np.asarray(prediction_b, dtype=np.int64)
    _require(target_array.shape == a.shape == b.shape and target_array.ndim == 1, "McNemar arrays must align.")
    raw = exact_mcnemar(a == target_array, b == target_array)
    return {
        "both_correct": raw["both_correct"],
        "a_correct_b_wrong": raw["flat_correct_hierarchy_wrong"],
        "a_wrong_b_correct": raw["flat_wrong_hierarchy_correct"],
        "both_wrong": raw["both_wrong"],
        "discordant_pairs": raw["discordant_pairs"],
        "exact_two_sided_p_value": raw["exact_two_sided_p_value"],
        "paired_accuracy_difference_a_minus_b": raw["paired_accuracy_difference"],
        "raw_discordant_pair_odds_ratio_a_over_b": raw["raw_discordant_pair_odds_ratio"],
    }
