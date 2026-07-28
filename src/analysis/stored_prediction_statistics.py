"""Frozen Phase 07 statistics over paired stored predictions only."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import binomtest


CLASSES = ("non_malignant", "melanoma", "bcc", "scc")
LABELS = np.arange(4, dtype=np.int64)
EXPECTED_SUPPORT = np.array([2398, 678, 498, 94], dtype=np.int64)
METRIC_NAMES = ("accuracy", "balanced_accuracy", "macro_f1")
BOOTSTRAP_COLUMNS = (
    "replicate_id",
    "flat_accuracy",
    "hierarchical_accuracy",
    "difference_accuracy",
    "flat_balanced_accuracy",
    "hierarchical_balanced_accuracy",
    "difference_balanced_accuracy",
    "flat_macro_f1",
    "hierarchical_macro_f1",
    "difference_macro_f1",
    *(f"flat_f1_{name}" for name in CLASSES),
    *(f"hierarchical_f1_{name}" for name in CLASSES),
    *(f"difference_f1_{name}" for name in CLASSES),
)


class StatisticalAnalysisError(ValueError):
    """Raised when frozen analysis invariants are violated."""


@dataclass(frozen=True)
class ModelMetrics:
    """Metrics derived from a fixed-label confusion matrix."""

    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    support: np.ndarray
    confusion_matrix: np.ndarray


def sha256_file(path: Path) -> str:
    """Hash one file without modifying it."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StatisticalAnalysisError(message)


def verify_hash(path: Path, expected: str, description: str) -> str:
    """Fail closed unless a file has the frozen digest."""
    _require(path.is_file(), f"{description} does not exist: {path}")
    observed = sha256_file(path)
    _require(
        observed == expected,
        f"{description} SHA-256 mismatch: expected {expected}, observed {observed}.",
    )
    return observed


def verify_archive_member(
    archive_path: Path,
    expected_archive_sha256: str,
    canonical_member: str,
    expected_member_sha256: str,
) -> bytes:
    """Safely read exactly one regular canonical member from a verified tar."""
    verify_hash(archive_path, expected_archive_sha256, "Phase 06C archive")
    canonical_matches: list[tarfile.TarInfo] = []
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            _require(not path.is_absolute(), f"Unsafe absolute archive member: {member.name}")
            _require(
                ".." not in path.parts,
                f"Unsafe path-traversal archive member: {member.name}",
            )
            _require(
                not member.issym() and not member.islnk(),
                f"Archive links are prohibited: {member.name}",
            )
            if member.name == canonical_member:
                canonical_matches.append(member)
        _require(
            len(canonical_matches) == 1,
            "Expected exactly one canonical Phase 06C member, "
            f"observed {len(canonical_matches)}.",
        )
        member = canonical_matches[0]
        _require(member.isfile(), "Canonical Phase 06C member is not a regular file.")
        handle = archive.extractfile(member)
        _require(handle is not None, "Canonical Phase 06C member is unreadable.")
        content = handle.read()
    observed = hashlib.sha256(content).hexdigest()
    _require(
        observed == expected_member_sha256,
        "Phase 06C member SHA-256 mismatch: "
        f"expected {expected_member_sha256}, observed {observed}.",
    )
    return content


def confusion_matrix_fixed(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Return a fixed 4x4 integer confusion matrix."""
    target = np.asarray(target)
    prediction = np.asarray(prediction)
    _require(target.shape == prediction.shape, "Target and prediction shapes differ.")
    _require(target.ndim == 1, "Targets and predictions must be one-dimensional.")
    _require(target.size > 0, "Targets and predictions cannot be empty.")
    _require(
        np.isin(target, LABELS).all() and np.isin(prediction, LABELS).all(),
        "Targets and predictions must contain only labels [0, 1, 2, 3].",
    )
    counts = np.bincount(
        target.astype(np.int64) * 4 + prediction.astype(np.int64), minlength=16
    )
    return counts.reshape(4, 4).astype(np.int64, copy=False)


def metrics_from_confusion(confusion: np.ndarray) -> ModelMetrics:
    """Derive frozen zero-division=0 metrics from a 4x4 confusion matrix."""
    matrix = np.asarray(confusion, dtype=np.int64)
    _require(matrix.shape == (4, 4), "Confusion matrix must be 4x4.")
    _require((matrix >= 0).all(), "Confusion matrix counts cannot be negative.")
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    total = int(support.sum())
    _require(total > 0, "Confusion matrix cannot be empty.")
    true_positive = np.diag(matrix).astype(np.float64)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros(4, dtype=np.float64),
        where=predicted != 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(4, dtype=np.float64),
        where=support != 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(4, dtype=np.float64),
        where=(precision + recall) != 0,
    )
    values = np.concatenate((precision, recall, f1))
    _require(np.isfinite(values).all(), "A non-finite metric was produced.")
    return ModelMetrics(
        accuracy=float(true_positive.sum() / total),
        balanced_accuracy=float(recall.mean(dtype=np.float64)),
        macro_f1=float(f1.mean(dtype=np.float64)),
        weighted_f1=float(np.average(f1, weights=support)),
        precision=precision,
        recall=recall,
        f1=f1,
        support=support.astype(np.int64),
        confusion_matrix=matrix,
    )


def calculate_metrics(target: np.ndarray, prediction: np.ndarray) -> ModelMetrics:
    """Calculate all frozen metrics for one complete prediction vector."""
    return metrics_from_confusion(confusion_matrix_fixed(target, prediction))


def validate_paired_manifest(
    frame: pd.DataFrame,
    *,
    expected_count: int = 3668,
    expected_support: np.ndarray = EXPECTED_SUPPORT,
) -> pd.DataFrame:
    """Validate and normalize the committed paired manifest."""
    required = {
        "sample_id",
        "target_label",
        "target_index",
        "hierarchical_predicted_label",
        "hierarchical_predicted_index",
        "flat_predicted_label",
        "flat_predicted_index",
    }
    _require(set(frame.columns) == required, "Paired-manifest schema changed.")
    _require(len(frame) == expected_count, f"Expected {expected_count} paired rows.")
    _require(not frame["sample_id"].isna().any(), "Missing sample identifier.")
    _require(
        frame["sample_id"].astype(str).str.strip().ne("").all(),
        "Missing sample identifier.",
    )
    _require(not frame["sample_id"].duplicated().any(), "Duplicate sample identifier.")
    normalized = frame.copy()
    index_columns = (
        "target_index",
        "hierarchical_predicted_index",
        "flat_predicted_index",
    )
    for column in index_columns:
        numeric = pd.to_numeric(normalized[column], errors="coerce")
        _require(not numeric.isna().any(), f"Missing or invalid values in {column}.")
        values = numeric.to_numpy(dtype=np.int64)
        _require(np.isin(values, LABELS).all(), f"Unsupported labels in {column}.")
        normalized[column] = values
    for prefix in ("target", "hierarchical_predicted", "flat_predicted"):
        expected_labels = normalized[f"{prefix}_index"].map(dict(enumerate(CLASSES)))
        _require(
            expected_labels.equals(normalized[f"{prefix}_label"]),
            f"{prefix} label/index mapping changed.",
        )
    support = np.bincount(
        normalized["target_index"].to_numpy(dtype=np.int64), minlength=4
    )
    _require(np.array_equal(support, expected_support), "Frozen class support changed.")
    return normalized.sort_values("sample_id", kind="stable").reset_index(drop=True)


def stratified_bootstrap_indices(
    target: np.ndarray, replicate_count: int, seed: int
) -> np.ndarray:
    """Generate identical paired indices with preserved within-class support."""
    _require(replicate_count > 0, "Replicate count must be positive.")
    target = np.asarray(target, dtype=np.int64)
    _require(np.isin(target, LABELS).all(), "Unsupported target label.")
    strata = [np.flatnonzero(target == label) for label in LABELS]
    _require(all(indices.size > 0 for indices in strata), "Every class must be present.")
    generator = np.random.default_rng(seed)
    sampled = np.empty((replicate_count, target.size), dtype=np.int64)
    offsets = np.cumsum([0, *(indices.size for indices in strata)])
    for replicate in range(replicate_count):
        for position, indices in enumerate(strata):
            sampled[replicate, offsets[position] : offsets[position + 1]] = (
                generator.choice(indices, size=indices.size, replace=True)
            )
    return sampled


def _bootstrap_metric_values(
    target: np.ndarray,
    flat: np.ndarray,
    hierarchical: np.ndarray,
    *,
    replicate_count: int,
    seed: int,
) -> np.ndarray:
    indices = stratified_bootstrap_indices(target, replicate_count, seed)
    values = np.empty((replicate_count, len(BOOTSTRAP_COLUMNS) - 1), dtype=np.float64)
    for replicate, selected in enumerate(indices):
        flat_metrics = calculate_metrics(target[selected], flat[selected])
        hierarchical_metrics = calculate_metrics(
            target[selected], hierarchical[selected]
        )
        row = [
            flat_metrics.accuracy,
            hierarchical_metrics.accuracy,
            flat_metrics.accuracy - hierarchical_metrics.accuracy,
            flat_metrics.balanced_accuracy,
            hierarchical_metrics.balanced_accuracy,
            flat_metrics.balanced_accuracy - hierarchical_metrics.balanced_accuracy,
            flat_metrics.macro_f1,
            hierarchical_metrics.macro_f1,
            flat_metrics.macro_f1 - hierarchical_metrics.macro_f1,
            *flat_metrics.f1,
            *hierarchical_metrics.f1,
            *(flat_metrics.f1 - hierarchical_metrics.f1),
        ]
        values[replicate] = np.asarray(row, dtype=np.float64)
    _require(np.isfinite(values).all(), "Non-finite bootstrap metric encountered.")
    _require(values.shape[0] == replicate_count, "A bootstrap replicate was skipped.")
    return values


def bootstrap_table(
    target: np.ndarray,
    flat: np.ndarray,
    hierarchical: np.ndarray,
    *,
    replicate_count: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """Return every frozen paired bootstrap replicate."""
    values = _bootstrap_metric_values(
        target, flat, hierarchical, replicate_count=replicate_count, seed=seed
    )
    frame = pd.DataFrame(values, columns=BOOTSTRAP_COLUMNS[1:])
    frame.insert(0, "replicate_id", np.arange(replicate_count, dtype=np.int64))
    return frame


def linear_quantile(values: np.ndarray, probabilities: Sequence[float]) -> np.ndarray:
    """Apply the explicit amended quantile method to finite float64 values."""
    array = np.asarray(values)
    _require(array.dtype == np.float64, "Quantile values must have dtype float64.")
    _require(array.ndim == 1 and array.size > 0, "Quantile input must be non-empty 1D.")
    _require(np.isfinite(array).all(), "Quantile input contains NaN or infinity.")
    result = np.quantile(array, probabilities, method="linear")
    _require(np.isfinite(result).all(), "Quantile result is non-finite.")
    return np.asarray(result, dtype=np.float64)


def exact_mcnemar(flat_correct: np.ndarray, hierarchical_correct: np.ndarray) -> dict[str, Any]:
    """Calculate the frozen paired correctness table and exact test."""
    flat_correct = np.asarray(flat_correct, dtype=bool)
    hierarchical_correct = np.asarray(hierarchical_correct, dtype=bool)
    _require(flat_correct.shape == hierarchical_correct.shape, "Correctness shapes differ.")
    both_correct = int(np.sum(flat_correct & hierarchical_correct))
    flat_only = int(np.sum(flat_correct & ~hierarchical_correct))
    hierarchical_only = int(np.sum(~flat_correct & hierarchical_correct))
    both_wrong = int(np.sum(~flat_correct & ~hierarchical_correct))
    discordant = flat_only + hierarchical_only
    p_value = (
        1.0
        if discordant == 0
        else float(binomtest(flat_only, discordant, 0.5, alternative="two-sided").pvalue)
    )
    if hierarchical_only == 0 and flat_only == 0:
        odds = {"numeric_value": None, "status": "undefined"}
    elif hierarchical_only == 0:
        odds = {"numeric_value": None, "status": "positive_infinity"}
    else:
        odds = {"numeric_value": float(flat_only / hierarchical_only), "status": "finite"}
    total = flat_correct.size
    return {
        "both_correct": both_correct,
        "flat_correct_hierarchy_wrong": flat_only,
        "flat_wrong_hierarchy_correct": hierarchical_only,
        "both_wrong": both_wrong,
        "discordant_pairs": discordant,
        "exact_two_sided_p_value": p_value,
        "paired_accuracy_difference": float((flat_only - hierarchical_only) / total),
        "net_paired_correctness_advantage": float((flat_only - hierarchical_only) / total),
        "raw_discordant_pair_odds_ratio": odds,
    }


def correctness_category(flat_correct: bool, hierarchical_correct: bool) -> str:
    """Name one mutually exclusive paired correctness category."""
    if flat_correct and hierarchical_correct:
        return "both_correct"
    if flat_correct:
        return "correct_only_flat"
    if hierarchical_correct:
        return "correct_only_hierarchy"
    return "both_wrong"


def routing_decomposition(frame: pd.DataFrame) -> pd.DataFrame:
    """Describe proven routing states from stored Phase 05 columns."""
    required = {
        "image_id",
        "final_target_index",
        "stage_1_predicted_index",
        "stage_2_executed",
        "stage_2_predicted_index",
        "predicted_gate_correct",
    }
    _require(required.issubset(frame.columns), "Stored routing columns are incomplete.")
    target = pd.to_numeric(frame["final_target_index"], errors="raise").astype(int)
    stage1 = pd.to_numeric(frame["stage_1_predicted_index"], errors="raise").astype(int)
    executed = pd.to_numeric(frame["stage_2_executed"], errors="raise").astype(int)
    _require(executed.isin([0, 1]).all(), "Invalid stage_2_executed value.")
    structural_missing = executed.eq(0) & frame["stage_2_predicted_index"].isna()
    anomalous_missing = executed.eq(1) & frame["stage_2_predicted_index"].isna()
    _require(not anomalous_missing.any(), "Anomalous Stage 2 missingness detected.")
    categories = [
        ("true_malignant_routed_non_malignant", target.gt(0) & stage1.eq(0)),
        ("true_non_malignant_routed_stage2", target.eq(0) & stage1.eq(1)),
        (
            "correct_malignant_route_wrong_subtype",
            target.gt(0)
            & stage1.eq(1)
            & pd.to_numeric(frame["predicted_gate_correct"], errors="raise").eq(0),
        ),
        (
            "correct_route_correct_subtype",
            target.gt(0)
            & stage1.eq(1)
            & pd.to_numeric(frame["predicted_gate_correct"], errors="raise").eq(1),
        ),
        ("structural_stage2_missing_not_invoked", structural_missing),
        ("anomalous_stage2_missing", anomalous_missing),
    ]
    return pd.DataFrame(
        [
            {
                "category": name,
                "count": int(mask.sum()),
                "percent_of_all_samples": float(mask.mean() * 100.0),
            }
            for name, mask in categories
        ]
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        _require(math.isfinite(number), "JSON payload contains NaN or Infinity.")
        return number
    return value


def write_json(path: Path, payload: Any) -> None:
    """Write deterministic finite JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    """Write deterministic UTF-8 CSV with frozen float formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")


def environment_versions() -> dict[str, str]:
    """Return execution dependency versions without volatile timestamps."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }


def metric_rows(model: str, metrics: ModelMetrics) -> list[dict[str, Any]]:
    """Flatten model metrics in deterministic order."""
    rows = [
        {"model": model, "metric": "accuracy", "class": "overall", "value": metrics.accuracy},
        {
            "model": model,
            "metric": "balanced_accuracy",
            "class": "overall",
            "value": metrics.balanced_accuracy,
        },
        {"model": model, "metric": "macro_f1", "class": "overall", "value": metrics.macro_f1},
        {
            "model": model,
            "metric": "weighted_f1",
            "class": "overall",
            "value": metrics.weighted_f1,
        },
    ]
    for index, name in enumerate(CLASSES):
        rows.extend(
            [
                {"model": model, "metric": "precision", "class": name, "value": metrics.precision[index]},
                {"model": model, "metric": "recall", "class": name, "value": metrics.recall[index]},
                {"model": model, "metric": "f1", "class": name, "value": metrics.f1[index]},
                {"model": model, "metric": "support", "class": name, "value": int(metrics.support[index])},
            ]
        )
    return rows


def confusion_frame(metrics: ModelMetrics) -> pd.DataFrame:
    """Format a fixed confusion matrix."""
    frame = pd.DataFrame(metrics.confusion_matrix, columns=[f"predicted_{x}" for x in CLASSES])
    frame.insert(0, "true_label", CLASSES)
    return frame


def build_analysis(
    paired: pd.DataFrame,
    routing: pd.DataFrame,
    *,
    replicate_count: int = 10000,
    seed: int = 42,
    expected_count: int = 3668,
    expected_support: np.ndarray = EXPECTED_SUPPORT,
) -> dict[str, Any]:
    """Calculate the complete frozen deterministic analysis in memory."""
    paired = validate_paired_manifest(
        paired,
        expected_count=expected_count,
        expected_support=expected_support,
    )
    target = paired["target_index"].to_numpy(dtype=np.int64)
    flat = paired["flat_predicted_index"].to_numpy(dtype=np.int64)
    hierarchical = paired["hierarchical_predicted_index"].to_numpy(dtype=np.int64)
    flat_metrics = calculate_metrics(target, flat)
    hierarchical_metrics = calculate_metrics(target, hierarchical)
    replicates = bootstrap_table(
        target,
        flat,
        hierarchical,
        replicate_count=replicate_count,
        seed=seed,
    )
    flat_correct = flat == target
    hierarchical_correct = hierarchical == target
    mcnemar = exact_mcnemar(flat_correct, hierarchical_correct)

    sample_rows = paired.rename(columns={"sample_id": "image_id"}).copy()
    sample_rows["flat_correct"] = flat_correct
    sample_rows["hierarchical_correct"] = hierarchical_correct
    sample_rows["correctness_category"] = [
        correctness_category(f, h)
        for f, h in zip(flat_correct, hierarchical_correct)
    ]
    sample_rows["predictions_agree"] = flat == hierarchical
    sample_rows = sample_rows[
        [
            "image_id",
            "target_index",
            "target_label",
            "flat_predicted_index",
            "flat_predicted_label",
            "hierarchical_predicted_index",
            "hierarchical_predicted_label",
            "flat_correct",
            "hierarchical_correct",
            "correctness_category",
            "predictions_agree",
        ]
    ]

    point_lookup = {
        "flat_accuracy": flat_metrics.accuracy,
        "hierarchical_accuracy": hierarchical_metrics.accuracy,
        "difference_accuracy": flat_metrics.accuracy - hierarchical_metrics.accuracy,
        "flat_balanced_accuracy": flat_metrics.balanced_accuracy,
        "hierarchical_balanced_accuracy": hierarchical_metrics.balanced_accuracy,
        "difference_balanced_accuracy": flat_metrics.balanced_accuracy
        - hierarchical_metrics.balanced_accuracy,
        "flat_macro_f1": flat_metrics.macro_f1,
        "hierarchical_macro_f1": hierarchical_metrics.macro_f1,
        "difference_macro_f1": flat_metrics.macro_f1 - hierarchical_metrics.macro_f1,
    }
    for index, name in enumerate(CLASSES):
        point_lookup[f"flat_f1_{name}"] = float(flat_metrics.f1[index])
        point_lookup[f"hierarchical_f1_{name}"] = float(hierarchical_metrics.f1[index])
        point_lookup[f"difference_f1_{name}"] = float(
            flat_metrics.f1[index] - hierarchical_metrics.f1[index]
        )
    ci_rows = []
    for column in BOOTSTRAP_COLUMNS[1:]:
        lower, upper = linear_quantile(
            replicates[column].to_numpy(dtype=np.float64), [0.025, 0.975]
        )
        ci_rows.append(
            {
                "estimand": column,
                "point_estimate": point_lookup[column],
                "lower": lower,
                "upper": upper,
                "confidence_level": 0.95,
                "replicate_count": replicate_count,
                "seed": seed,
                "ci_method": "percentile_numpy_quantile_method_linear",
                "direction": (
                    "flat_minus_hierarchical"
                    if column.startswith("difference_")
                    else "model_specific"
                ),
            }
        )
    confidence_intervals = pd.DataFrame(ci_rows)

    agreement = pd.DataFrame(
        [
            {"category": "both_correct", "count": mcnemar["both_correct"]},
            {"category": "both_wrong", "count": mcnemar["both_wrong"]},
            {"category": "correct_only_flat", "count": mcnemar["flat_correct_hierarchy_wrong"]},
            {"category": "correct_only_hierarchy", "count": mcnemar["flat_wrong_hierarchy_correct"]},
            {"category": "exact_prediction_agreement", "count": int(np.sum(flat == hierarchical))},
            {"category": "prediction_disagreement", "count": int(np.sum(flat != hierarchical))},
        ]
    )
    agreement["percent"] = agreement["count"] / len(paired) * 100.0

    transitions = pd.crosstab(
        pd.Categorical(flat, categories=LABELS),
        pd.Categorical(hierarchical, categories=LABELS),
        dropna=False,
    )
    transitions.index = CLASSES
    transitions.columns = [f"hierarchical_{name}" for name in CLASSES]
    transition_frame = transitions.reset_index(names="flat_prediction")

    error_transitions = (
        sample_rows.groupby(
            [
                "target_label",
                "flat_predicted_label",
                "hierarchical_predicted_label",
                "correctness_category",
            ],
            sort=True,
            observed=True,
        )
        .size()
        .reset_index(name="count")
    )

    scc = target == 3
    scc_rows = []
    for model, prediction, metrics in (
        ("flat", flat, flat_metrics),
        ("hierarchical", hierarchical, hierarchical_metrics),
    ):
        counts = np.bincount(prediction[scc], minlength=4)
        scc_rows.append(
            {
                "model": model,
                "support": int(scc.sum()),
                "true_positives": int(counts[3]),
                "precision": metrics.precision[3],
                "recall": metrics.recall[3],
                "f1": metrics.f1[3],
                **{f"predicted_{name}": int(counts[i]) for i, name in enumerate(CLASSES)},
            }
        )
    scc_rows.append(
        {
            "model": "paired_categories",
            "support": int(scc.sum()),
            "correct_only_flat": int(np.sum(scc & flat_correct & ~hierarchical_correct)),
            "correct_only_hierarchy": int(np.sum(scc & ~flat_correct & hierarchical_correct)),
            "both_wrong": int(np.sum(scc & ~flat_correct & ~hierarchical_correct)),
            "both_correct": int(np.sum(scc & flat_correct & hierarchical_correct)),
            "prediction_agreement": int(np.sum(scc & (flat == hierarchical))),
            "prediction_disagreement": int(np.sum(scc & (flat != hierarchical))),
        }
    )

    return {
        "flat_metrics": flat_metrics,
        "hierarchical_metrics": hierarchical_metrics,
        "replicates": replicates,
        "confidence_intervals": confidence_intervals,
        "mcnemar": mcnemar,
        "sample_categories": sample_rows,
        "agreement": agreement,
        "transitions": transition_frame,
        "error_transitions": error_transitions,
        "scc": pd.DataFrame(scc_rows),
        "routing": routing_decomposition(routing),
        "point_lookup": point_lookup,
    }
