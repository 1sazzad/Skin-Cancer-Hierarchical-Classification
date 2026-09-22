"""COM-02 reusable primitives; importing this module performs no study execution.

Callers supply already verified frozen models and aligned arrays. This module
does not load checkpoints, construct real datasets, or launch an evaluation.
Future production callers must verify the COM-01B checkpoint/cohort locks first.
"""
from __future__ import annotations

import json
import time
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from src.analysis.stored_prediction_statistics import (
    calculate_metrics, exact_mcnemar, linear_quantile,
)

CLASS_NAMES = ("non_malignant", "melanoma", "bcc", "scc")
LABELS = (0, 1, 2, 3)
SEEDS = (42, 123, 2026)
FEATURE_ORDER = (
    "p_task1_malignant", "p_task2_melanoma", "p_task2_bcc", "p_task2_scc",
)
PROTOCOL_PATH = "configs/extensions/comesyso2026_probability_fusion/comesyso2026_protocol.yaml"
PROTOCOL_ID = "comesyso2026_probability_fusion/COM-01B/47fa88a"
FIT_PARTITION = "isic2019_validation_all_four_classes"
CLASSIFIER_SPEC = {
    "model": "multinomial_logistic_regression",
    "objective": "joint_four_class_multinomial_softmax", "penalty": "l2",
    "C": 1.0, "solver": "lbfgs", "fit_intercept": True,
    "max_iter": 1000, "class_weight": None, "tol": 1e-4,
}
CONTRASTS = {
    "fusion_minus_hard": ("fusion", "hard"),
    "path_soft_minus_hard": ("path_soft", "hard"),
    "fusion_minus_flat": ("fusion", "flat"),
    "oracle_minus_fusion": ("oracle", "fusion"),
}


def _labels(values, *, count=None, allowed=LABELS, name="labels"):
    array = np.asarray(values)
    if (array.ndim != 1 or not array.size or array.dtype.kind not in "iu"
            or not np.isin(array, allowed).all()
            or (count is not None and len(array) != count)):
        raise ValueError(f"{name} must be a nonempty integer vector in {allowed} of matching length")
    return array.astype(np.int64, copy=False)


def _ids(values, count, *, unique=True, nullable=False):
    if values is None or isinstance(values, (str, bytes)):
        raise ValueError("IDs must be a sequence, not a scalar string")
    values = tuple(values)
    if len(values) != count or not count:
        raise ValueError("ID count must match the nonempty sample count")
    if any(not (nullable and v is None) and (not isinstance(v, str) or not v.strip()) for v in values):
        raise ValueError("IDs must be nonempty strings (or explicitly permitted nulls)")
    if unique and len(set(values)) != count:
        raise ValueError("Duplicate sample IDs")
    return values


def _aligned_ids(left, right, count):
    left = _ids(left, count)
    if left != _ids(right, count):
        raise ValueError("Sample IDs must have identical stable order; no dropping or reordering")
    return left


def _probabilities(values, columns, *, count=None):
    array = np.asarray(values, dtype=np.float64)
    if (array.ndim != 2 or array.shape[1] != columns or not len(array)
            or (count is not None and len(array) != count)
            or not np.isfinite(array).all() or np.any((array < 0) | (array > 1))
            or not np.allclose(array.sum(axis=1), 1.0, rtol=0, atol=1e-6)):
        raise ValueError(f"Expected finite [N,{columns}] probabilities in [0,1] summing to one")
    return array


def _heads(task1, task2):
    task1 = _probabilities(task1, 2)
    return task1, _probabilities(task2, 3, count=len(task1))


def _features(values):
    array = np.asarray(values, dtype=np.float64)
    if (array.ndim != 2 or array.shape[1] != 4 or not len(array)
            or not np.isfinite(array).all() or np.any((array < 0) | (array > 1))):
        raise ValueError("Fusion features must be finite [N,4] values in [0,1]")
    _probabilities(array[:, 1:], 3)
    return array


def _seed(seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("Expected one frozen seed: 42, 123, or 2026")


@dataclass(frozen=True)
class SharedPredictions:
    """Historical field names, with unmasked Task-2 outputs on EVERY row."""
    sample_ids: tuple[str, ...]
    flat_targets: np.ndarray
    stage1_targets: np.ndarray
    stage1_predictions: np.ndarray
    stage1_probabilities: np.ndarray
    stage2_targets: np.ndarray
    stage2_predictions: np.ndarray
    stage2_probabilities: np.ndarray
    patient_ids: tuple[str | None, ...]
    elapsed_seconds: float


def collect_shared_predictions(model, dataloader: Iterable[Mapping], *, device="cpu"):
    """Single frozen forward per batch; Task 3 is deliberately never accessed.

    Uses historical autocast/float32-softmax behavior, but never applies the
    historical Task-2 execution mask. Batch keys match existing ISIC/HIBA loaders.
    """
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only the historical CPU/CUDA device paths are supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model.to(device).eval()
    ids, patients, targets, first, second = [], [], [], [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in dataloader:
            images, target = batch.get("image"), batch.get("target")
            if not isinstance(images, torch.Tensor) or images.ndim != 4 or not len(images):
                raise ValueError("Expected nonempty image tensor [N,C,H,W]")
            n = len(images)
            if not isinstance(target, torch.Tensor):
                raise ValueError("Expected target tensor")
            truth = _labels(target.detach().cpu().numpy(), count=n)
            batch_ids = batch.get("image_id", batch.get("sample_id"))
            if isinstance(batch_ids, str):
                batch_ids = [batch_ids]
            if batch_ids is None:
                raise ValueError("Every batch requires image_id or sample_id")
            ids.extend(_ids(batch_ids, n))
            patient_ids = batch.get("patient_id", [None] * n)
            if isinstance(patient_ids, str):
                patient_ids = [patient_ids]
            # Historical ISIC manifests may express missing patient IDs as ''.
            patients.extend(_ids([None if p == "" else p for p in patient_ids], n,
                                 unique=False, nullable=True))
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                outputs = model(images.to(device, non_blocking=True))
            if not isinstance(outputs, Mapping):
                raise ValueError("Shared model must return a task-logit mapping")
            probabilities = []
            for task, width in (("task1", 2), ("task2", 3)):
                logits = outputs.get(task)
                if (not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (n, width)
                        or not torch.isfinite(logits).all()):
                    raise ValueError(f"Invalid {task} logits")
                probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            targets.append(truth)
            first.append(probabilities[0])
            second.append(probabilities[1])
    _ids(ids, len(ids))
    truth = np.concatenate(targets)
    p1, p2 = _heads(np.concatenate(first), np.concatenate(second))
    return SharedPredictions(
        tuple(ids), truth, (truth != 0).astype(np.int64), p1.argmax(axis=1), p1,
        np.where(truth == 0, -1, truth - 1), p2.argmax(axis=1), p2,
        tuple(patients), time.perf_counter() - started,
    )


def hard_routing(task1, task2):
    p1, p2 = _heads(task1, task2)
    return np.where(p1.argmax(axis=1) == 0, 0, p2.argmax(axis=1) + 1)


def path_soft(task1, task2):
    p1, p2 = _heads(task1, task2)
    probabilities = np.column_stack((p1[:, 0], p1[:, 1, None] * p2))
    return probabilities, probabilities.argmax(axis=1)


def oracle_routing(task1_targets, task2):
    """Diagnostic only: root truth, never predicted Task-1 routing."""
    p2 = _probabilities(task2, 3)
    truth = _labels(task1_targets, count=len(p2), allowed=(0, 1))
    return np.where(truth == 0, 0, p2.argmax(axis=1) + 1)


def build_fusion_features(task1, task2, *, sample_ids, task2_sample_ids):
    p1, p2 = _heads(task1, task2)
    _aligned_ids(sample_ids, task2_sample_ids, len(p1))
    return _features(np.column_stack((p1[:, 1], p2)))


def _new_classifier():
    # sklearn 1.9: penalty is deprecated; multi_class has been removed.
    # l1_ratio=0 is L2 and lbfgs uses joint multinomial loss for four classes.
    return LogisticRegression(l1_ratio=0.0, C=1.0, solver="lbfgs",
                              fit_intercept=True, max_iter=1000,
                              class_weight=None, tol=1e-4)


def _class_order(classifier, *, canonical=False):
    if not hasattr(classifier, "classes_"):
        raise ValueError("Classifier is not fitted: classes_ is missing")
    classes = _labels(classifier.classes_, count=4)
    if set(classes.tolist()) != set(LABELS):
        raise ValueError("Classifier must contain exactly four distinct classes")
    if canonical and not np.array_equal(classes, LABELS):
        raise ValueError("Fitted classifier classes_ must equal [0,1,2,3]")
    return classes


@dataclass(frozen=True)
class FusionModel:
    classifier: LogisticRegression
    seed: int
    fitting_partition: str
    fitting_sample_count: int
    sklearn_version: str


def fit_fusion(features, targets, *, seed, fitting_partition,
               sample_ids, target_sample_ids):
    """Fit exactly one seed; explicit partition prevents accidental test fitting.

    'synthetic' is supported for tests and is persisted honestly as synthetic.
    A partition declaration is not proof of provenance: future production
    orchestration must verify the frozen validation manifest before calling.
    """
    _seed(seed)
    if fitting_partition not in (FIT_PARTITION, "synthetic"):
        raise ValueError("Fusion may fit only ISIC validation or synthetic fixtures")
    features = _features(features)
    _aligned_ids(sample_ids, target_sample_ids, len(features))
    targets = _labels(targets, count=len(features))
    if set(targets.tolist()) != set(LABELS):
        raise ValueError("Training targets must contain exactly [0,1,2,3]")
    classifier = _new_classifier()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            classifier.fit(features, targets)
    except ConvergenceWarning as exc:
        raise RuntimeError("Fusion convergence failed; frozen specification unchanged") from exc
    _class_order(classifier, canonical=True)
    if np.any(classifier.n_iter_ >= CLASSIFIER_SPEC["max_iter"]):
        raise RuntimeError("Fusion iteration limit reached; stop without retuning")
    model = FusionModel(classifier, seed, fitting_partition, len(features), sklearn.__version__)
    _model_metadata(model)
    return model


def predict_fusion(classifier, features):
    """Return probabilities and argmax aligned to final order, including ties."""
    if isinstance(classifier, FusionModel):
        classifier = classifier.classifier
    features = _features(features)
    classes = _class_order(classifier)
    raw = _probabilities(classifier.predict_proba(features), 4, count=len(features))
    indices = [int(np.flatnonzero(classes == label)[0]) for label in LABELS]
    probabilities = raw[:, indices]
    return probabilities, probabilities.argmax(axis=1)


def _model_metadata(model):
    _seed(model.seed)
    if model.fitting_partition not in (FIT_PARTITION, "synthetic"):
        raise ValueError("Invalid fitting partition")
    if type(model.fitting_sample_count) is not int or model.fitting_sample_count < 4:
        raise ValueError("Invalid fitting sample count")
    if not isinstance(model.sklearn_version, str) or not model.sklearn_version:
        raise ValueError("Missing sklearn version")
    classifier = model.classifier
    if not isinstance(classifier, LogisticRegression):
        raise ValueError("Expected the frozen logistic-regression estimator")
    _class_order(classifier, canonical=True)
    if classifier.get_params() != _new_classifier().get_params():
        raise ValueError("Classifier parameters differ from frozen specification")
    coefficients = np.asarray(classifier.coef_, dtype=np.float64)
    intercepts = np.asarray(classifier.intercept_, dtype=np.float64)
    iterations = np.asarray(classifier.n_iter_)
    if (coefficients.shape != (4, 4) or intercepts.shape != (4,)
            or not np.isfinite(coefficients).all() or not np.isfinite(intercepts).all()
            or classifier.n_features_in_ != 4 or iterations.shape != (1,)
            or iterations.dtype.kind not in "iu" or np.any(iterations < 0)
            or np.any(iterations >= 1000)):
        raise ValueError("Invalid or unconverged classifier state")
    return {
        "schema_version": 1, "seed": model.seed, "feature_order": list(FEATURE_ORDER),
        "classes": list(LABELS), "coefficients": coefficients.tolist(),
        "intercepts": intercepts.tolist(), "n_iter": iterations.tolist(),
        "classifier_specification": dict(CLASSIFIER_SPEC),
        "fitting_partition": model.fitting_partition,
        "fitting_sample_count": model.fitting_sample_count,
        "protocol_identity": PROTOCOL_ID, "protocol_path": PROTOCOL_PATH,
        "sklearn_version": model.sklearn_version,
    }


def save_fusion(model, path):
    """Deterministic transparent JSON, no pickle, no overwrite of existing files.

    Production callers must use the protocol's CoMeSySo output root. Synthetic
    tests may use an isolated temporary directory. No performance is serialized.
    """
    payload = json.dumps(_model_metadata(model), sort_keys=True, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)


def load_fusion(path, *, expected_seed):
    _seed(expected_seed)
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Malformed fusion metadata")
    for key, expected in (
        ("schema_version", 1), ("seed", expected_seed),
        ("feature_order", list(FEATURE_ORDER)), ("classes", list(LABELS)),
        ("classifier_specification", CLASSIFIER_SPEC),
        ("protocol_identity", PROTOCOL_ID), ("protocol_path", PROTOCOL_PATH),
    ):
        if value.get(key) != expected:
            raise ValueError(f"Fusion metadata mismatch: {key}")
    try:
        classifier = _new_classifier()
        classifier.classes_ = np.asarray(value["classes"])
        classifier.coef_ = np.asarray(value["coefficients"], dtype=np.float64)
        classifier.intercept_ = np.asarray(value["intercepts"], dtype=np.float64)
        classifier.n_iter_ = np.asarray(value["n_iter"])
        classifier.n_features_in_ = 4
        model = FusionModel(classifier, value["seed"], value["fitting_partition"],
                            value["fitting_sample_count"], value["sklearn_version"])
        if _model_metadata(model) != value:
            raise ValueError("Unexpected or inconsistent fusion metadata")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Incomplete fusion metadata") from exc
    return model


def four_class_metrics(targets, predictions):
    targets = _labels(targets)
    predictions = _labels(predictions, count=len(targets))
    result = calculate_metrics(targets, predictions)
    return {
        "sample_count": len(targets), "class_names": list(CLASS_NAMES),
        "labels": list(LABELS), "accuracy": result.accuracy,
        "balanced_accuracy": result.balanced_accuracy, "macro_f1": result.macro_f1,
        "weighted_f1": result.weighted_f1,
        "confusion_matrix": result.confusion_matrix.tolist(),
        "per_class": {name: {"precision": float(result.precision[i]),
                             "recall": float(result.recall[i]), "f1": float(result.f1[i]),
                             "support": int(result.support[i])}
                      for i, name in enumerate(CLASS_NAMES)},
    }


def routing_rescue(targets, task1_predictions, hard_predictions, fusion_predictions):
    truth = _labels(targets)
    n = len(truth)
    root = _labels(task1_predictions, count=n, allowed=(0, 1))
    hard = _labels(hard_predictions, count=n)
    fusion = _labels(fusion_predictions, count=n)
    if not np.array_equal(hard == 0, root == 0):
        raise ValueError("Hard predictions contradict Task-1 routing")
    hc, fc = hard == truth, fusion == truth
    same = hard == fusion
    malignant = (truth != 0) & (root == 0)
    nm = (truth == 0) & (root == 1)
    count = lambda mask: int(np.count_nonzero(mask))
    partition = {
        "hard_wrong_fusion_correct": count(~hc & fc),
        "hard_correct_fusion_wrong": count(hc & ~fc),
        "both_correct": count(hc & fc),
        "both_wrong_same_prediction": count(~hc & ~fc & same),
        "both_wrong_different_predictions": count(~hc & ~fc & ~same),
    }
    if sum(partition.values()) != n:
        raise RuntimeError("Hard/Fusion partition does not cover the cohort")
    return {
        "sample_count": n,
        "malignant_routed_nm": {
            "total": count(malignant), "fusion_correct": count(malignant & fc),
            "fusion_still_incorrect": count(malignant & ~fc),
            "rescued_true_subtype_counts": {CLASS_NAMES[i]: count(malignant & fc & (truth == i))
                                             for i in (1, 2, 3)},
        },
        "nm_routed_malignant": {
            "total": count(nm), "fusion_correct": count(nm & fc),
            "fusion_still_incorrect": count(nm & ~fc),
            "unchanged_incorrect_prediction": count(nm & ~fc & same),
            "changed_to_another_incorrect_class": count(nm & ~fc & ~same),
            "made_worse_correct_to_wrong_count": 0,
        },
        "pairwise_partition": partition,
        "hard_wrong_fusion_wrong_prediction_changed": partition["both_wrong_different_predictions"],
    }


def build_routing_rescue_rows(shared, *, dataset, seed, flat_predictions,
                             flat_sample_ids, fusion_predictions, fusion_sample_ids):
    """Paired export records with explicit ID checks for independently run systems."""
    _seed(seed)
    if dataset not in ("isic", "hiba"):
        raise ValueError("dataset must be 'isic' or 'hiba'")
    truth = _labels(shared.flat_targets)
    n = len(truth)
    ids = _aligned_ids(shared.sample_ids, flat_sample_ids, n)
    _aligned_ids(ids, fusion_sample_ids, n)
    patients = _ids(shared.patient_ids, n, unique=False, nullable=dataset == "isic")
    p1, p2 = _heads(shared.stage1_probabilities, shared.stage2_probabilities)
    if len(p1) != n:
        raise ValueError("Shared target/probability counts differ")
    t1, pred1 = (truth != 0).astype(np.int64), p1.argmax(axis=1)
    for actual, expected in ((shared.stage1_targets, t1), (shared.stage1_predictions, pred1),
                             (shared.stage2_targets, np.where(truth == 0, -1, truth - 1)),
                             (shared.stage2_predictions, p2.argmax(axis=1))):
        if not np.array_equal(actual, expected):
            raise ValueError("Shared collection fields disagree with targets/probabilities")
    predictions = {
        "flat": _labels(flat_predictions, count=n), "hard": hard_routing(p1, p2),
        "path_soft": path_soft(p1, p2)[1],
        "fusion": _labels(fusion_predictions, count=n), "oracle": oracle_routing(t1, p2),
    }
    rows = []
    for i, sample_id in enumerate(ids):
        row = {"dataset": dataset, "seed": seed, "sample_id": sample_id,
               "patient_id": patients[i], "true_final_class": int(truth[i]),
               "task1_target": int(t1[i]), "task1_prediction": int(pred1[i]),
               "p_task1_non_malignant": float(p1[i, 0]), "p_task1_malignant": float(p1[i, 1]),
               "p_task2_melanoma": float(p2[i, 0]), "p_task2_bcc": float(p2[i, 1]),
               "p_task2_scc": float(p2[i, 2])}
        row.update({f"{name}_prediction": int(pred[i]) for name, pred in predictions.items()})
        row.update({f"{name}_correct": bool(predictions[name][i] == truth[i])
                    for name in ("hard", "path_soft", "fusion", "oracle")})
        rows.append(row)
    summary = routing_rescue(truth, pred1, predictions["hard"], predictions["fusion"])
    summary.update(dataset=dataset, seed=seed)
    return rows, summary


def _systems(targets, predictions):
    targets = _labels(targets)
    expected = {name for pair in CONTRASTS.values() for name in pair}
    if set(predictions) != expected:
        raise ValueError("Supply exactly Flat, Hard, Path-Soft, Fusion, and Oracle")
    return targets, {name: _labels(predictions[name], count=len(targets)) for name in sorted(expected)}


def _replicates(replicates):
    if type(replicates) is not int or replicates <= 0:
        raise ValueError("Replicate count must be a positive integer")


def patient_cluster_indices(patient_ids, *, replicates=5000, seed=42):
    """Historical HIBA draw algorithm, preserving every repeated cluster copy.

    Kept local because the historical CI function hardcodes a two-system
    contrast and uses a metric that omits absent classes. Here only the draws
    are reproduced; fixed-four-class metrics are reused from Phase 07.
    """
    _replicates(replicates)
    ids = _ids(patient_ids, len(patient_ids), unique=False)
    patients = np.array(sorted(set(ids)), dtype=object)
    by_patient = {p: np.flatnonzero(np.asarray(ids, dtype=object) == p) for p in patients}
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        yield np.concatenate([by_patient[p] for p in sampled])


def _bootstrap(targets, predictions, draws, *, replicates, seed, unit):
    point = {name: calculate_metrics(targets, pred).macro_f1 for name, pred in predictions.items()}
    differences = {name: np.empty(replicates, dtype=np.float64) for name in CONTRASTS}
    for i, indices in enumerate(draws):
        scores = {name: calculate_metrics(targets[indices], pred[indices]).macro_f1
                  for name, pred in predictions.items()}
        for name, (minuend, subtrahend) in CONTRASTS.items():
            differences[name][i] = scores[minuend] - scores[subtrahend]
    return {
        "metric": "four_class_macro_f1", "unit": unit, "paired": True,
        "replicates": replicates, "seed": seed, "rng": "numpy_default_rng",
        "confidence_level": 0.95, "interval": "percentile", "quantile_method": "linear",
        "contrasts": {name: {"point_difference": point[a] - point[b],
                              "confidence_interval": linear_quantile(differences[name], [.025, .975]).tolist()}
                      for name, (a, b) in CONTRASTS.items()},
    }


def paired_image_bootstrap(targets, predictions, *, replicates=10000, seed=42):
    """Unstratified paired image bootstrap; all systems share each index draw.

    Historical bootstrap_table is stratified and is therefore not reused.
    Reduced replicate counts are exposed for synthetic unit tests only.
    """
    _replicates(replicates)
    targets, predictions = _systems(targets, predictions)
    rng = np.random.default_rng(seed)
    draws = (rng.integers(0, len(targets), size=len(targets)) for _ in range(replicates))
    return _bootstrap(targets, predictions, draws, replicates=replicates, seed=seed, unit="image")


def patient_cluster_bootstrap(targets, predictions, patient_ids, *, replicates=5000, seed=42):
    """Generic synthetic-capable primitive; production HIBA requires 568 patients."""
    _replicates(replicates)
    targets, predictions = _systems(targets, predictions)
    ids = _ids(patient_ids, len(targets), unique=False)
    draws = patient_cluster_indices(ids, replicates=replicates, seed=seed)
    result = _bootstrap(targets, predictions, draws, replicates=replicates, seed=seed,
                        unit="patient_cluster")
    result["patients_per_replicate"] = len(set(ids))
    return result


def mcnemar_comparisons(targets, predictions):
    targets, predictions = _systems(targets, predictions)
    result = {}
    for name, (a, b) in CONTRASTS.items():
        test = exact_mcnemar(predictions[a] == targets, predictions[b] == targets)
        result[name] = {
            "method": "exact_two_sided", "unit": "image",
            "interpretation": "paired_correctness_not_macro_f1",
            "minuend_correct_subtrahend_wrong": test["flat_correct_hierarchy_wrong"],
            "minuend_wrong_subtrahend_correct": test["flat_wrong_hierarchy_correct"],
            "p_value": test["exact_two_sided_p_value"],
        }
    return result


def protocol_statistics(targets, predictions, *, dataset, patient_ids=None):
    """Frozen production settings without loading data or invoking a runner."""
    if dataset == "isic":
        if len(targets) != 3668:
            raise ValueError("Frozen ISIC internal test requires 3668 images")
        return {"bootstrap": paired_image_bootstrap(targets, predictions),
                "mcnemar": mcnemar_comparisons(targets, predictions)}
    if dataset == "hiba":
        if len(targets) != 1232 or patient_ids is None:
            raise ValueError("Frozen HIBA requires 1232 images with patient IDs")
        ids = _ids(patient_ids, len(targets), unique=False)
        if len(set(ids)) != 568:
            raise ValueError("Frozen HIBA requires 568 unique patients")
        return {"bootstrap": patient_cluster_bootstrap(targets, predictions, ids)}
    raise ValueError("dataset must be 'isic' or 'hiba'")


def summarize_three_seeds(values):
    """Summarize one scalar metric/contrast without pooling seed-image pairs."""
    if set(values) != set(SEEDS) or any(type(seed) is not int for seed in values):
        raise ValueError("Exactly seeds 42, 123, and 2026 are required")
    array = np.asarray([values[seed] for seed in SEEDS], dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError("Expected one finite scalar per seed")
    return {"individual_seed_values": {str(seed): float(values[seed]) for seed in SEEDS},
            "mean": float(array.mean()), "sample_standard_deviation": float(array.std(ddof=1))}
