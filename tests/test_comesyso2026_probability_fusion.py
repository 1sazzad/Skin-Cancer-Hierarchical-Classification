"""COM-02 synthetic contracts only: no datasets, checkpoints, or study results."""
from dataclasses import replace
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import pytest
import torch
import yaml
from sklearn.exceptions import ConvergenceWarning

from src.evaluation import comesyso2026_probability_fusion as ev


class SyntheticShared(torch.nn.Module):
    def __init__(self, task3_value=0.0):
        super().__init__()
        self.task3_value = task3_value
        self.calls = 0

    def forward(self, images):
        assert not self.training
        assert not torch.is_grad_enabled()
        self.calls += 1
        n = len(images)
        return {
            "task1": torch.tensor([[2.0, 0.0]], device=images.device).repeat(n, 1),
            "task2": torch.tensor([[0.0, 1.0, 3.0]], device=images.device).repeat(n, 1),
            "task3": torch.full((n, 5), self.task3_value, device=images.device),
        }


def batch(ids=("nm", "mel"), targets=(0, 1)):
    return {"image": torch.zeros(len(ids), 3, 2, 2),
            "target": torch.tensor(targets), "image_id": list(ids)}


def features_from(shared):
    return ev.build_fusion_features(
        shared.stage1_probabilities, shared.stage2_probabilities,
        sample_ids=shared.sample_ids, task2_sample_ids=shared.sample_ids,
    )


def test_unmasked_single_forward_and_task3_exclusion():
    models = [SyntheticShared(), SyntheticShared(float("nan"))]
    collections = [ev.collect_shared_predictions(m, [batch(), batch(("bcc", "scc"), (2, 3))])
                   for m in models]
    a, b = collections
    assert [m.calls for m in models] == [2, 2]
    assert a.sample_ids == ("nm", "mel", "bcc", "scc")
    np.testing.assert_array_equal(a.stage1_predictions, [0, 0, 0, 0])
    np.testing.assert_array_equal(a.stage1_targets, [0, 1, 1, 1])
    np.testing.assert_array_equal(a.stage2_targets, [-1, 0, 1, 2])
    np.testing.assert_array_equal(a.stage2_predictions, [2, 2, 2, 2])
    expected = torch.softmax(torch.tensor([0.0, 1.0, 3.0]), dim=0).numpy()
    np.testing.assert_allclose(a.stage2_probabilities, np.tile(expected, (4, 1)))
    assert np.isfinite(a.stage2_probabilities).all()
    np.testing.assert_array_equal(features_from(a), features_from(b))


def test_collector_preserves_patient_ids_and_sample_id_alias():
    value = batch()
    value["sample_id"] = value.pop("image_id")
    value["patient_id"] = ["p1", "p1"]
    shared = ev.collect_shared_predictions(SyntheticShared(), [value])
    assert shared.patient_ids == ("p1", "p1")


@pytest.mark.parametrize("invalid", ["empty", "duplicate", "fractional", "missing_id", "shape", "nan"])
def test_collector_fails_closed(invalid):
    value, model = batch(), SyntheticShared()
    loader = [value]
    if invalid == "empty":
        loader = []
    elif invalid == "duplicate":
        value["image_id"] = ["same", "same"]
    elif invalid == "fractional":
        value["target"] = torch.tensor([0.0, 1.5])
    elif invalid == "missing_id":
        del value["image_id"]
    elif invalid == "shape":
        value["image"] = torch.zeros(2, 3)
    else:
        model.forward = lambda images: {"task1": torch.full((2, 2), float("nan")),
                                         "task2": torch.zeros(2, 3)}
    with pytest.raises(ValueError):
        ev.collect_shared_predictions(model, loader)


def test_hard_path_soft_oracle_and_tie_order():
    p1 = np.array([[.8, .2], [.3, .7], [.5, .5], [.1, .9]])
    p2 = np.array([[0, 0, 1], [.4, .4, .2], [1, 0, 0], [.1, .2, .7]])
    np.testing.assert_array_equal(ev.hard_routing(p1, p2), [0, 1, 0, 3])
    probabilities, predictions = ev.path_soft(p1, p2)
    np.testing.assert_allclose(probabilities, [[.8, 0, 0, .2], [.3, .28, .28, .14],
                                               [.5, .5, 0, 0], [.1, .09, .18, .63]])
    np.testing.assert_array_equal(predictions, [0, 0, 0, 3])
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    # Truth routing differs from predicted routing in the first two rows.
    np.testing.assert_array_equal(ev.oracle_routing([1, 0, 1, 1], p2), [3, 0, 1, 3])


def test_path_soft_cannot_overturn_strict_nm_root_preference():
    rng = np.random.default_rng(123)
    nm = np.linspace(.500001, 1, 100)
    p1 = np.column_stack((nm, 1 - nm))
    p2 = rng.dirichlet(np.ones(3), size=100)
    p2[:3] = np.eye(3)
    probabilities, predictions = ev.path_soft(p1, p2)
    assert np.all(probabilities[:, 0, None] > probabilities[:, 1:])
    assert np.all(predictions == 0)


def test_feature_order_and_stable_alignment():
    p1, p2 = [[.7, .3], [.4, .6]], [[.1, .2, .7], [.6, .3, .1]]
    x = ev.build_fusion_features(p1, p2, sample_ids=["a", "b"], task2_sample_ids=["a", "b"])
    np.testing.assert_allclose(x, [[.3, .1, .2, .7], [.6, .6, .3, .1]])
    for ids in (["b", "a"], ["a", "a"], ["a"]):
        with pytest.raises(ValueError):
            ev.build_fusion_features(p1, p2, sample_ids=["a", "b"], task2_sample_ids=ids)


@pytest.mark.parametrize("p1,p2", [
    ([[.5, .5]], [[np.nan, .2, .8]]),
    ([[.5, .5]], [[.2, .8]]),
    ([[.5, .5]], [[-.1, .5, .6]]),
    ([[.5, .5]], [[.2, .2, .2]]),
    ([[.5, .5], [.1, .9]], [[.2, .3, .5]]),
    ([[np.inf, 0]], [[.2, .3, .5]]),
])
def test_routing_rejects_invalid_head_probabilities(p1, p2):
    with pytest.raises(ValueError):
        ev.hard_routing(p1, p2)
    with pytest.raises(ValueError):
        ev.path_soft(p1, p2)


@pytest.fixture
def synthetic_training():
    x = np.tile([[.1, .2, .3, .5], [.8, .8, .1, .1],
                 [.8, .1, .8, .1], [.8, .1, .1, .8]], (20, 1))
    y = np.tile(np.arange(4), 20)
    ids = tuple(f"synthetic_{i}" for i in range(len(y)))
    return x, y, ids


def fit_fixture(data, **overrides):
    x, y, ids = data
    options = dict(seed=42, fitting_partition="synthetic", sample_ids=ids, target_sample_ids=ids)
    options.update(overrides)
    return ev.fit_fusion(x, y, **options)


def test_synthetic_multinomial_fit(synthetic_training):
    model = fit_fixture(synthetic_training)
    classifier = model.classifier
    np.testing.assert_array_equal(classifier.classes_, [0, 1, 2, 3])
    assert classifier.coef_.shape == (4, 4)
    assert classifier.l1_ratio == 0.0
    assert classifier.C == 1.0 and classifier.solver == "lbfgs"
    assert classifier.fit_intercept is True and classifier.max_iter == 1000
    assert classifier.class_weight is None and classifier.tol == 1e-4
    probabilities, prediction = ev.predict_fusion(model, synthetic_training[0])
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    np.testing.assert_array_equal(prediction, synthetic_training[1])


def test_prediction_aligns_classifier_columns_before_argmax():
    classifier = SimpleNamespace(classes_=np.array([3, 1, 0, 2]),
                                 predict_proba=lambda x: np.array([[.4, .1, .4, .1]]))
    probabilities, prediction = ev.predict_fusion(classifier, [[.3, .2, .3, .5]])
    np.testing.assert_allclose(probabilities, [[.4, .1, .1, .4]])
    np.testing.assert_array_equal(prediction, [0])


@pytest.mark.parametrize("classes", [[0, 1, 2], [0, 1, 2, 2], [0, 1, 2, 4], [0., 1., 2., 3.]])
def test_invalid_classifier_classes(classes):
    classifier = SimpleNamespace(classes_=np.asarray(classes))
    with pytest.raises(ValueError):
        ev.predict_fusion(classifier, [[.3, .2, .3, .5]])


@pytest.mark.parametrize("invalid", ["nan", "inf", "shape", "range", "simplex", "count", "missing_class", "float_labels"])
def test_invalid_fit_inputs_fail_closed(synthetic_training, invalid):
    x, y, ids = synthetic_training
    x, y = x.copy(), y.copy()
    if invalid == "nan":
        x[0, 0] = np.nan
    elif invalid == "inf":
        x[0, 0] = np.inf
    elif invalid == "shape":
        x = x[:, :3]
    elif invalid == "range":
        x[0, 0] = -0.1
    elif invalid == "simplex":
        x[0, 1:] = .1
    elif invalid == "count":
        y = y[:-1]
    elif invalid == "missing_class":
        y[y == 3] = 2
    else:
        y = y.astype(float)
    with pytest.raises(ValueError):
        fit_fixture((x, y, ids))


@pytest.mark.parametrize("partition", ["internal_test", "hiba", "train", "validation"])
def test_rejects_undeclared_fit_partition(synthetic_training, partition):
    with pytest.raises(ValueError):
        fit_fixture(synthetic_training, fitting_partition=partition)


def test_fit_rejects_target_order_and_undeclared_seed(synthetic_training):
    with pytest.raises(ValueError):
        fit_fixture(synthetic_training, target_sample_ids=synthetic_training[2][::-1])
    with pytest.raises(ValueError):
        fit_fixture(synthetic_training, seed=7)


@pytest.mark.parametrize("mode", ["warning", "limit", "classes"])
def test_fit_failure_does_not_retry_or_retune(synthetic_training, monkeypatch, mode):
    calls = []

    class FakeClassifier:
        def fit(self, x, y):
            calls.append(1)
            if mode == "warning":
                warnings.warn("synthetic nonconvergence", ConvergenceWarning)
            self.classes_ = np.array([0, 2, 1, 3] if mode == "classes" else [0, 1, 2, 3])
            self.n_iter_ = np.array([1000])

    monkeypatch.setattr(ev, "_new_classifier", FakeClassifier)
    with pytest.raises(ValueError if mode == "classes" else RuntimeError):
        fit_fixture(synthetic_training)
    assert calls == [1]


def test_deterministic_json_roundtrip(synthetic_training, tmp_path):
    model = fit_fixture(synthetic_training)
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    ev.save_fusion(model, a)
    restored = ev.load_fusion(a, expected_seed=42)
    ev.save_fusion(restored, b)
    assert a.read_bytes() == b.read_bytes()
    np.testing.assert_allclose(ev.predict_fusion(restored, synthetic_training[0])[0],
                               ev.predict_fusion(model, synthetic_training[0])[0], rtol=0, atol=0)
    payload = json.loads(a.read_text())
    assert payload["feature_order"] == list(ev.FEATURE_ORDER)
    assert payload["fitting_partition"] == "synthetic"
    assert payload["fitting_sample_count"] == 80
    assert payload["protocol_identity"] == ev.PROTOCOL_ID
    assert payload["sklearn_version"]
    with pytest.raises(FileExistsError):
        ev.save_fusion(model, a)
    with pytest.raises(ValueError):
        ev.load_fusion(a, expected_seed=123)


@pytest.mark.parametrize("field,value", [
    ("feature_order", ["wrong"]), ("classes", [0, 2, 1, 3]),
    ("coefficients", [[0]]), ("intercepts", [float("nan")] * 4),
    ("fitting_partition", "hiba"), ("n_iter", [1000]),
    ("fitting_sample_count", 0), ("protocol_identity", "different"),
    ("classifier_specification", {}),
])
def test_load_rejects_invalid_metadata(synthetic_training, tmp_path, field, value):
    path = tmp_path / "fusion.json"
    ev.save_fusion(fit_fixture(synthetic_training), path)
    payload = json.loads(path.read_text())
    payload[field] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        ev.load_fusion(path, expected_seed=42)


def test_metrics_keep_absent_classes_and_zero_division():
    metrics = ev.four_class_metrics([0, 0], [0, 0])
    assert metrics["accuracy"] == 1
    assert metrics["balanced_accuracy"] == metrics["macro_f1"] == .25
    assert metrics["weighted_f1"] == 1
    assert len(metrics["per_class"]) == 4
    assert np.asarray(metrics["confusion_matrix"]).shape == (4, 4)
    assert metrics["per_class"]["scc"] == {"precision": 0., "recall": 0., "f1": 0., "support": 0}


def test_confusion_matrix_orientation_and_known_metrics():
    metrics = ev.four_class_metrics([0, 0, 1, 2], [0, 1, 1, 1])
    assert metrics["confusion_matrix"] == [[1, 1, 0, 0], [0, 1, 0, 0],
                                            [0, 1, 0, 0], [0, 0, 0, 0]]
    assert metrics["accuracy"] == .5
    assert metrics["balanced_accuracy"] == .375
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + .5) / 4)
    assert metrics["weighted_f1"] == pytest.approx((2 * 2 / 3 + .5) / 4)


def test_rescue_requires_exact_subtype_and_five_disjoint_categories():
    truth = [1, 2, 0, 0, 0, 1, 2, 3]
    hard = [0, 0, 1, 2, 3, 1, 2, 1]
    fusion = [1, 1, 0, 2, 1, 0, 2, 1]
    result = ev.routing_rescue(truth, [0, 0, 1, 1, 1, 1, 1, 1], hard, fusion)
    assert result["malignant_routed_nm"] == {
        "total": 2, "fusion_correct": 1, "fusion_still_incorrect": 1,
        "rescued_true_subtype_counts": {"melanoma": 1, "bcc": 0, "scc": 0},
    }
    assert result["nm_routed_malignant"] == {
        "total": 3, "fusion_correct": 1, "fusion_still_incorrect": 2,
        "unchanged_incorrect_prediction": 1, "changed_to_another_incorrect_class": 1,
        "made_worse_correct_to_wrong_count": 0,
    }
    assert result["pairwise_partition"] == {
        "hard_wrong_fusion_correct": 2, "hard_correct_fusion_wrong": 1,
        "both_correct": 1, "both_wrong_same_prediction": 2, "both_wrong_different_predictions": 2,
    }
    assert sum(result["pairwise_partition"].values()) == len(truth)
    assert result["hard_wrong_fusion_wrong_prediction_changed"] == 2


def test_rescue_rows_require_alignment_and_hiba_patient_ids():
    shared = ev.collect_shared_predictions(SyntheticShared(), [batch()])
    kwargs = dict(dataset="isic", seed=42, flat_predictions=[0, 1],
                  flat_sample_ids=shared.sample_ids, fusion_predictions=[0, 1],
                  fusion_sample_ids=shared.sample_ids)
    rows, summary = ev.build_routing_rescue_rows(shared, **kwargs)
    assert rows[0]["patient_id"] is None
    assert rows[1]["fusion_correct"] and not rows[1]["hard_correct"]
    assert summary["dataset"] == "isic" and summary["seed"] == 42
    protocol = yaml.safe_load((Path(__file__).resolve().parents[1] / ev.PROTOCOL_PATH).read_text())
    assert set(rows[0]) == set(protocol["routing_rescue"]["fields"])
    with pytest.raises(ValueError):
        ev.build_routing_rescue_rows(shared, **{**kwargs, "flat_sample_ids": shared.sample_ids[::-1]})
    with pytest.raises(ValueError):
        ev.build_routing_rescue_rows(shared, **{**kwargs, "dataset": "hiba"})
    hiba = replace(shared, patient_ids=("patient", "patient"))
    hiba_rows, _ = ev.build_routing_rescue_rows(hiba, **{**kwargs, "dataset": "hiba"})
    assert hiba_rows[0]["patient_id"] == "patient"


@pytest.fixture
def statistical_arrays():
    target = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    return target, {
        "flat": np.array([0, 1, 1, 2, 2, 0, 3, 1]),
        "hard": np.array([0, 0, 0, 0, 2, 2, 2, 2]),
        "path_soft": np.array([0, 0, 0, 0, 2, 2, 2, 2]),
        "fusion": target.copy(), "oracle": target.copy(),
    }


def test_image_bootstrap_determinism_and_paired_manual_reference(statistical_arrays):
    target, predictions = statistical_arrays
    a = ev.paired_image_bootstrap(target, predictions, replicates=19)
    assert a == ev.paired_image_bootstrap(target, predictions, replicates=19)
    rng = np.random.default_rng(42)
    values = {name: [] for name in ev.CONTRASTS}
    for _ in range(19):
        indices = rng.integers(0, len(target), size=len(target))
        scores = {name: ev.four_class_metrics(target[indices], pred[indices])["macro_f1"]
                  for name, pred in predictions.items()}
        for name, (left, right) in ev.CONTRASTS.items():
            values[name].append(scores[left] - scores[right])
    for name, (left, right) in ev.CONTRASTS.items():
        result = a["contrasts"][name]
        np.testing.assert_allclose(result["confidence_interval"],
                                   np.quantile(values[name], [.025, .975], method="linear"))
        assert result["point_difference"] == (ev.four_class_metrics(target, predictions[left])["macro_f1"]
                                               - ev.four_class_metrics(target, predictions[right])["macro_f1"])
    assert a["contrasts"]["path_soft_minus_hard"]["confidence_interval"] == [0, 0]


def test_exact_mcnemar_known_discordance(statistical_arrays):
    target, predictions = statistical_arrays
    tests = ev.mcnemar_comparisons(target, predictions)
    primary = tests["fusion_minus_hard"]
    assert primary["minuend_correct_subtrahend_wrong"] == 4
    assert primary["minuend_wrong_subtrahend_correct"] == 0
    assert primary["p_value"] == pytest.approx(.125)
    assert tests["path_soft_minus_hard"]["p_value"] == 1
    assert tests["oracle_minus_fusion"]["p_value"] == 1
    reverse = {**predictions, "fusion": predictions["hard"], "hard": target}
    primary_reverse = ev.mcnemar_comparisons(target, reverse)["fusion_minus_hard"]
    assert primary_reverse["minuend_wrong_subtrahend_correct"] == 4
    assert primary_reverse["p_value"] == primary["p_value"]


def test_cluster_draws_preserve_multiplicity_sorted_order_and_all_images():
    ids = ["b", "a", "b", "c", "c", "c"]
    actual = list(ev.patient_cluster_indices(ids, replicates=20))
    again = list(ev.patient_cluster_indices(ids, replicates=20))
    rng = np.random.default_rng(42)
    repeated = False
    for indices, second in zip(actual, again):
        patients = rng.choice(np.array(["a", "b", "c"], dtype=object), size=3, replace=True)
        repeated |= len(set(patients)) < 3
        expected = np.concatenate([np.flatnonzero(np.asarray(ids) == p) for p in patients])
        np.testing.assert_array_equal(indices, expected)
        np.testing.assert_array_equal(indices, second)
    assert repeated
    assert len({len(indices) for indices in actual}) > 1


def test_patient_bootstrap_determinism_and_reference(statistical_arrays):
    target, predictions = statistical_arrays
    ids = ["b", "a", "b", "b", "c", "c", "d", "d"]
    result = ev.patient_cluster_bootstrap(target, predictions, ids, replicates=17)
    assert result == ev.patient_cluster_bootstrap(target, predictions, ids, replicates=17)
    assert result["patients_per_replicate"] == 4
    values = {name: [] for name in ev.CONTRASTS}
    for indices in ev.patient_cluster_indices(ids, replicates=17):
        scores = {name: ev.four_class_metrics(target[indices], pred[indices])["macro_f1"]
                  for name, pred in predictions.items()}
        for name, (a, b) in ev.CONTRASTS.items():
            values[name].append(scores[a] - scores[b])
    for name in ev.CONTRASTS:
        np.testing.assert_allclose(result["contrasts"][name]["confidence_interval"],
                                   np.quantile(values[name], [.025, .975], method="linear"))


def test_invalid_statistics_and_production_counts(statistical_arrays):
    target, predictions = statistical_arrays
    for replicates in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            ev.paired_image_bootstrap(target, predictions, replicates=replicates)
    with pytest.raises(ValueError):
        ev.paired_image_bootstrap(target, {**predictions, "flat": [np.nan] * 8}, replicates=2)
    with pytest.raises(ValueError):
        ev.patient_cluster_bootstrap(target, predictions, [None] * 8, replicates=2)
    for dataset in ("isic", "hiba"):
        with pytest.raises(ValueError):
            ev.protocol_statistics(target, predictions, dataset=dataset)


def test_three_seed_summary_requires_all_seeds():
    result = ev.summarize_three_seeds({42: 1., 123: 2., 2026: 3.})
    assert result["mean"] == 2 and result["sample_standard_deviation"] == 1
    assert len(result["individual_seed_values"]) == 3
    with pytest.raises(ValueError):
        ev.summarize_three_seeds({42: 1., 123: 2.})


def test_constants_and_defaults_match_frozen_protocol():
    protocol = yaml.safe_load((Path(__file__).resolve().parents[1] / ev.PROTOCOL_PATH).read_text())
    fusion = protocol["systems"]["fusion"]
    assert fusion["classifier"] == ev.CLASSIFIER_SPEC
    assert fusion["feature_order"] == list(ev.FEATURE_ORDER)
    assert fusion["fit_partition"] == ev.FIT_PARTITION
    assert protocol["models"]["seeds"] == list(ev.SEEDS)
    contrasts = [protocol["endpoints"]["primary_comparison"], *protocol["endpoints"]["secondary_comparisons"]]
    assert {c["id"]: (c["minuend"], c["subtrahend"]) for c in contrasts} == ev.CONTRASTS
    for function, key in ((ev.paired_image_bootstrap, "internal_isic"),
                          (ev.patient_cluster_bootstrap, "hiba")):
        params = inspect.signature(function).parameters
        assert params["replicates"].default == protocol["statistics"][key]["bootstrap"]["replicates"]
        assert params["seed"].default == 42
