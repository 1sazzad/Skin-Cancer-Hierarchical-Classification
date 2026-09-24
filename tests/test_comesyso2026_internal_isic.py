"""Synthetic COM-04 tests; no real images, checkpoints, or remote execution."""
import ast
import copy
import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from src.evaluation import comesyso2026_internal_isic as internal
from src.evaluation import comesyso2026_probability_fusion as fusion
from src.evaluation import comesyso2026_validation_fit as fit

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bundle(monkeypatch):
    monkeypatch.setattr(internal, "COUNTS", (1, 1, 1, 1))
    truth = np.arange(4)
    ids = ("a", "b", "c", "d")
    p1 = np.asarray([[.2, .8], [.8, .2], [.3, .7], [.1, .9]])
    p2 = np.asarray([[.7, .2, .1], [.8, .1, .1], [.1, .8, .1], [.1, .1, .8]])
    shared = fusion.SharedPredictions(ids, truth, (truth != 0).astype(int), p1.argmax(1), p1,
        np.where(truth == 0, -1, truth - 1), p2.argmax(1), p2, (None,) * 4, 0.)
    flat = SimpleNamespace(sample_ids=ids, targets=truth, probabilities=np.eye(4), predictions=truth)
    # Explicit fitted-state fixture: no fitting, even in tests for COM-04.
    classifier = fusion._new_classifier()
    classifier.classes_ = np.arange(4)
    classifier.coef_ = np.asarray([[0., 0., 0., 0.], [0., 5., 0., 0.],
                                  [0., 0., 5., 0.], [0., 0., 0., 5.]])
    classifier.intercept_ = np.zeros(4)
    classifier.n_iter_ = np.asarray([1])
    classifier.n_features_in_ = 4
    model = fusion.FusionModel(classifier, 42, fusion.FIT_PARTITION, 4, fit.sklearn.__version__)
    cohort = internal.cohort_identity(ids, truth)
    lock = {"cohort": cohort, "input_sha256": {}, "phase": "COM-04"}
    real_bootstrap = fusion.paired_image_bootstrap
    def small_bootstrap(targets, predictions, *, replicates, seed):
        assert (replicates, seed) == (10000, 42)
        return real_bootstrap(targets, predictions, replicates=8, seed=seed)
    monkeypatch.setattr(fusion, "paired_image_bootstrap", small_bootstrap)
    monkeypatch.setattr(fusion, "fit_fusion", Mock(side_effect=AssertionError("COM-04 must never fit")))
    return flat, shared, model, lock


def evaluated(bundle, seed=42):
    flat, shared, model, lock = bundle
    model = fusion.FusionModel(model.classifier, seed, model.fitting_partition,
                               model.fitting_sample_count, model.sklearn_version)
    return internal.evaluate_collections(flat, shared, model, seed=seed, expected_cohort=lock["cohort"])


def test_exact_selections():
    assert [(s.seed, s.epoch) for s in internal.FLAT_SPECS] == [(42, 9), (123, 10), (2026, 17)]
    assert [(s.seed, s.epoch) for s in internal.SHARED_SPECS] == [(42, 10), (123, 1), (2026, 26)]
    assert internal.SHARED_SPECS == fit.SPECS


def test_five_system_predictions_and_partition(bundle):
    result, rescue, rows = evaluated(bundle)
    assert [r["hard_prediction"] for r in rows] == [1, 0, 2, 3]
    assert [r["path_soft_prediction"] for r in rows] == [1, 0, 2, 3]
    assert [r["fusion_prediction"] for r in rows] == [1, 1, 2, 3]
    assert [r["oracle_prediction"] for r in rows] == [0, 1, 2, 3]
    assert [r["flat_prediction"] for r in rows] == [0, 1, 2, 3]
    assert sum(rescue["pairwise_partition"].values()) == 4
    assert rescue["malignant_routed_nm"]["rescued"] == 1
    assert rescue["malignant_routed_nm"]["by_class"]["melanoma"]["rescued"] == 1
    assert rescue["nm_routed_malignant"]["new_errors"] == 0
    assert set(rescue["pairwise_partition"]) == set(internal.PARTITIONS)
    assert all(result["systems"][s]["labels"] == [0, 1, 2, 3] for s in internal.SYSTEMS)
    for name, (a, b) in fusion.CONTRASTS.items():
        assert result["contrasts"][name]["macro_f1"] == (
            result["systems"][a]["macro_f1"] - result["systems"][b]["macro_f1"])
    assert result["statistics"]["mcnemar"]["fusion_minus_hard"]["p_value"] == 1.0
    assert all("p_flat_scc" in r and "p_fusion_scc" in r and "flat_correct" in r for r in rows)
    fusion.fit_fusion.assert_not_called()


def test_unmasked_task2_for_true_and_predicted_nm():
    class Model(torch.nn.Module):
        def forward(self, images):
            return {"task1": torch.tensor([[5., 0.]]).repeat(4, 1),
                    "task2": torch.tensor([[1., 2., 3.]]).repeat(4, 1)}
    shared = fusion.collect_shared_predictions(Model(), [{"image": torch.zeros(4, 3, 2, 2),
        "target": torch.arange(4), "image_id": ["a", "b", "c", "d"]}])
    assert np.all(shared.stage1_predictions == 0)
    assert np.isfinite(shared.stage2_probabilities).all()
    np.testing.assert_allclose(shared.stage2_probabilities.sum(1), 1)


def test_bootstrap_ordinary_paired_indices(monkeypatch):
    truth = np.arange(4)
    predictions = {s: truth.copy() for s in internal.SYSTEMS}
    captured = {}
    def capture(targets, preds, draws, **kwargs):
        captured["draws"] = list(draws)
        captured.update(kwargs)
        return captured
    monkeypatch.setattr(fusion, "_bootstrap", capture)
    fusion.paired_image_bootstrap(truth, predictions, replicates=8, seed=42)
    rng = np.random.default_rng(42)
    expected = [rng.integers(0, 4, size=4) for _ in range(8)]
    for actual, wanted in zip(captured["draws"], expected):
        np.testing.assert_array_equal(actual, wanted)
    assert any(len(set(draw)) < 4 for draw in captured["draws"])
    assert captured["unit"] == "image"


def test_bootstrap_linear_percentile_and_mcnemar():
    truth = np.arange(4)
    predictions = {s: truth.copy() for s in internal.SYSTEMS}
    predictions["hard"] = np.zeros(4, dtype=int)
    stats = fusion.paired_image_bootstrap(truth, predictions, replicates=8, seed=42)
    assert stats["quantile_method"] == "linear"
    assert stats["interval"] == "percentile"
    test = fusion.mcnemar_comparisons(truth, predictions)["fusion_minus_hard"]
    assert test["minuend_correct_subtrahend_wrong"] == 3
    assert test["minuend_wrong_subtrahend_correct"] == 0
    assert test["p_value"] == .25


def test_three_seed_sample_sd(bundle):
    results = {s: evaluated(bundle, s)[0] for s in internal.SEEDS}
    for s, value in zip(internal.SEEDS, (.1, .2, .3)):
        results[s]["systems"]["fusion"]["macro_f1"] = value
    summary = internal.three_seed_summary(results)
    scalar = summary["systems"]["fusion"]["macro_f1"]
    assert scalar["mean"] == pytest.approx(.2)
    assert scalar["sample_standard_deviation"] == pytest.approx(.1)
    assert set(scalar["individual_seed_values"]) == {"42", "123", "2026"}
    with pytest.raises(ValueError):
        internal.three_seed_summary({42: results[42]})


def test_frozen_model_load_does_not_fit(bundle, tmp_path):
    model = bundle[2]
    path = tmp_path / "fusion.json"
    fusion.save_fusion(model, path)
    restored = fusion.load_fusion(path, expected_seed=42)
    np.testing.assert_array_equal(restored.classifier.coef_, model.classifier.coef_)
    fusion.fit_fusion.assert_not_called()
    with pytest.raises(ValueError):
        fusion.load_fusion(path, expected_seed=123)


@pytest.mark.parametrize("field,value", [("feature_order", ["wrong"]), ("classes", [3, 2, 1, 0]),
                                          ("protocol_identity", "wrong")])
def test_fusion_identity_drift(bundle, tmp_path, field, value):
    path = tmp_path / "fusion.json"
    fusion.save_fusion(bundle[2], path)
    payload = internal._read_json(path)
    payload[field] = value
    path.write_bytes(fit._json_bytes(payload))
    with pytest.raises(ValueError):
        fusion.load_fusion(path, expected_seed=42)


def mock_production(monkeypatch, tmp_path, bundle):
    lock = bundle[3]
    output = internal.output_directory(tmp_path)
    internal._publish_json(output / internal.LOCK_NAME, lock)
    monkeypatch.setattr(internal, "verify_preflight_lock", lambda root: lock)
    monkeypatch.setattr(internal, "verify_input_bytes", lambda *a, **k: None)
    evaluate = Mock(side_effect=lambda root, seed, lock, **kw: evaluated(bundle, seed))
    monkeypatch.setattr(internal, "evaluate_one_seed", evaluate)
    return output, evaluate


def test_atomic_seed_reuse_and_volume_commits(monkeypatch, tmp_path, bundle):
    output, evaluate = mock_production(monkeypatch, tmp_path, bundle)
    result, rescue, rows = evaluated(bundle)
    internal._publish_seed(output, 42, bundle[3], result, rescue, rows)
    before = (output / "seed42/paired_predictions.csv").read_bytes()
    commit = Mock()
    summary = internal.run_internal_isic_production(tmp_path, device="cpu", persist_callback=commit)
    assert [call.args[1] for call in evaluate.call_args_list] == [123, 2026]
    assert commit.call_count == 3  # Two new seeds and final completion.
    assert (output / "seed42/paired_predictions.csv").read_bytes() == before
    assert internal.run_internal_isic_production(tmp_path, device="cpu") == summary
    assert evaluate.call_count == 2
    with pytest.raises(FileExistsError):
        internal._publish_seed(output, 42, bundle[3], result, rescue, rows)


@pytest.mark.parametrize("partial", ["seed2026", "three_seed_summary.json", "internal_isic_complete.json", "unknown.json"])
def test_partial_outputs_block_all_inference(monkeypatch, tmp_path, bundle, partial):
    output, evaluate = mock_production(monkeypatch, tmp_path, bundle)
    path = output / partial
    if partial.startswith("seed"):
        path.mkdir()
    else:
        path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        internal.run_internal_isic_production(tmp_path, device="cpu")
    evaluate.assert_not_called()


def test_corrupt_completed_seed_blocks_missing_seed(monkeypatch, tmp_path, bundle):
    output, evaluate = mock_production(monkeypatch, tmp_path, bundle)
    internal._publish_seed(output, 42, bundle[3], *evaluated(bundle))
    (output / "seed42/rescue_analysis.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        internal.run_internal_isic_production(tmp_path, device="cpu")
    evaluate.assert_not_called()


def test_preflight_no_loader_or_inference(monkeypatch, tmp_path, bundle):
    monkeypatch.setattr(internal, "inspect_inputs", lambda root: bundle[3])
    loader = Mock(side_effect=AssertionError("no dataset in preflight"))
    evaluate = Mock(side_effect=AssertionError("no inference in preflight"))
    monkeypatch.setattr(internal.historical, "build_locked_isic_loader", loader)
    monkeypatch.setattr(internal, "evaluate_one_seed", evaluate)
    assert internal.run_internal_isic_preflight(tmp_path) == bundle[3]
    assert internal.run_internal_isic_preflight(tmp_path) == bundle[3]
    loader.assert_not_called()
    evaluate.assert_not_called()
    fusion.fit_fusion.assert_not_called()


def test_preflight_rejects_partial_before_lock(monkeypatch, tmp_path):
    output = internal.output_directory(tmp_path)
    (output / "seed42").mkdir(parents=True)
    inspector = Mock(side_effect=AssertionError("too late"))
    monkeypatch.setattr(internal, "inspect_inputs", inspector)
    with pytest.raises(ValueError, match="before-inference"):
        internal.run_internal_isic_preflight(tmp_path)
    inspector.assert_not_called()


def test_missing_lock_blocks_loader(monkeypatch, tmp_path):
    loader = Mock(side_effect=AssertionError("no dataset before lock"))
    monkeypatch.setattr(internal.historical, "build_locked_isic_loader", loader)
    with pytest.raises(FileNotFoundError):
        internal.run_internal_isic_production(tmp_path, device="cpu")
    loader.assert_not_called()


def test_input_hash_drift(tmp_path):
    path = tmp_path / "synthetic.txt"
    path.write_text("original", encoding="utf-8")
    lock = {"input_sha256": {"synthetic.txt": internal.sha256_file(path)}}
    internal.verify_input_bytes(tmp_path, lock, require_lock=False)
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="drift"):
        internal.verify_input_bytes(tmp_path, lock, require_lock=False)


def test_com03_completion_required(monkeypatch, tmp_path):
    monkeypatch.setattr(fit, "verify_preflight_lock", lambda root: {})
    output = fit.output_directory(tmp_path)
    output.mkdir(parents=True)
    with pytest.raises(ValueError, match="COM-03 completed file set"):
        internal._completed_fits(tmp_path)


@pytest.mark.parametrize("dataset,split", [("hiba", "internal_test"), ("isic2019", "validation"),
                                           ("isic2019", "train")])
def test_scope_rejects_external_and_non_test(dataset, split):
    with pytest.raises(ValueError):
        internal.require_scope(dataset=dataset, split=split)


@pytest.mark.parametrize("alternative", ["../outside", "results/extensions/paul2026_swin/internal_isic",
                                          "results/paper/internal_isic"])
def test_output_boundary(monkeypatch, tmp_path, alternative):
    monkeypatch.setattr(internal, "OUTPUT", Path(alternative))
    with pytest.raises(ValueError):
        internal.output_directory(tmp_path)


def test_root_traversal(tmp_path):
    with pytest.raises(ValueError):
        internal.output_directory(tmp_path / ".." / "other")


def test_modal_cpu_and_single_use_t4_configuration():
    path = ROOT / "scripts/comesyso2026/modal_comesyso2026_internal_isic.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name, fn in functions.items():
        options = {k.arg: k.value for k in fn.decorator_list[0].keywords}
        assert ast.literal_eval(options["max_containers"]) == 1
        assert ast.literal_eval(options["retries"]) == 0
        assert ast.literal_eval(options["single_use_containers"]) is True
        if name.endswith("preflight"):
            assert "gpu" not in options
            assert "/root/project/data/raw" not in ast.unparse(options["volumes"])
        else:
            assert ast.literal_eval(options["gpu"]) == "T4"
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr == "App" for n in ast.walk(tree))


def test_source_has_no_training_or_external_execution():
    source = (ROOT / "src/evaluation/comesyso2026_internal_isic.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert not any(isinstance(n.func, ast.Attribute) and n.func.attr in
                   {"fit", "fit_fusion", "train", "patient_cluster_bootstrap", "bootstrap_table"} for n in calls)
    assert "hiba_evaluation" not in source
    assert "collect_shared_isic_predictions" not in source
    assert "collect_shared_predictions" in source


def test_launcher_checks_all_functions_and_rejects_modes(monkeypatch):
    functions = {}
    def lookup(app, name):
        assert app == "comesyso2026-probability-fusion"
        fn = Mock()
        fn.get_current_stats.return_value = SimpleNamespace(num_running_inputs=0, backlog=0)
        fn.spawn.return_value = SimpleNamespace(object_id="synthetic-call")
        functions[name] = fn
        return fn
    modal = SimpleNamespace(App=lambda name: SimpleNamespace(local_entrypoint=lambda: lambda f: f),
                            Function=SimpleNamespace(from_name=lookup))
    monkeypatch.setitem(__import__("sys").modules, "modal", modal)
    path = ROOT / "scripts/comesyso2026/launch_comesyso2026_internal_isic.py"
    spec = importlib.util.spec_from_file_location("com04_launcher_test", path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    for mode in ("fit", "train", "hiba"):
        with pytest.raises(ValueError):
            launcher.main(mode)
    launcher.main("preflight")
    assert len(functions) == 4
    functions["run_internal_isic_preflight"].spawn.assert_called_once_with()
    assert sum(f.spawn.call_count for f in functions.values()) == 1


@pytest.mark.parametrize("failure", ["epoch", "checkpoint_hash", "manifest_hash", "missing_selection"])
def test_preflight_checkpoint_anchor_drift(monkeypatch, tmp_path, bundle, failure):
    import yaml
    com03 = {"protocol_sha256": "protocol", "validation_manifest_sha256": "manifest",
             "checkpoints": [{"seed": s, "checkpoint_sha256": "shared",
                              "config_path": f"config{s}", "config_sha256": "config"}
                             for s in internal.SEEDS]}
    monkeypatch.setattr(internal, "_completed_fits", lambda root: (com03, {}))
    monkeypatch.setattr(fit, "validate_protocol", lambda protocol: None)
    protocol = {"models": {"selections": [
        {"flat_run": s.run_name, "flat_selected_epoch": s.epoch} for s in internal.FLAT_SPECS]},
        "endpoints": {"primary_metric": "four_class_macro_f1",
                      "primary_comparison": {"id": "fusion_minus_hard", "minuend": "fusion", "subtrahend": "hard"},
                      "secondary_comparisons": [{"id": n, "minuend": a, "subtrahend": b}
                                                for n, (a, b) in fusion.CONTRASTS.items()
                                                if n != "fusion_minus_hard"]}}
    path = tmp_path / fusion.PROTOCOL_PATH
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(protocol), encoding="utf-8")
    records = [{"system": s.system, "seed": s.seed, "run_name": s.run_name,
                "selected_epoch": s.epoch, "sha256": "flat" if s.system == "flat" else "shared",
                "run_summary_sha256": "summary"} for s in (*internal.FLAT_SPECS, *internal.SHARED_SPECS)]
    old = {"status": "PASS", "manifest_sha256": "manifest", "checkpoints": records}
    if failure == "epoch":
        records[0]["selected_epoch"] = 99
    elif failure == "checkpoint_hash":
        records[3]["sha256"] = "substituted"
    elif failure == "manifest_hash":
        old["manifest_sha256"] = "changed"
    else:
        records.pop(0)
    internal._publish_json(tmp_path / internal.HISTORICAL_LOCK, old)
    monkeypatch.setattr(internal.historical, "load_selected_model", Mock())
    dataset = Mock(side_effect=AssertionError("Preflight must not construct a test dataset"))
    monkeypatch.setattr(internal.historical, "build_locked_isic_loader", dataset)
    with pytest.raises(ValueError):
        internal.inspect_inputs(tmp_path)
    dataset.assert_not_called()
    fusion.fit_fusion.assert_not_called()


def test_com03_final_marker_drift(monkeypatch, tmp_path, bundle):
    lock = {"phase": "COM-03"}
    monkeypatch.setattr(fit, "verify_preflight_lock", lambda root: lock)
    output = fit.output_directory(tmp_path)
    output.mkdir(parents=True)
    for name in (fit.LOCK_NAME, fit.SUMMARY_NAME, fit.COMPLETE_NAME):
        internal._publish_json(output / name, {})
    for seed in internal.SEEDS:
        directory = output / f"seed{seed}"
        directory.mkdir()
        internal._publish_json(directory / "seed_complete.json", {})
    monkeypatch.setattr(fit, "validate_completed_seed", lambda *a, **k: (bundle[2], {}, {}))
    monkeypatch.setattr(fit, "three_seed_summary", lambda results: {})
    with pytest.raises(ValueError, match="COM-03 completion marker"):
        internal._completed_fits(tmp_path)


def test_changed_preflight_lock_not_overwritten(monkeypatch, tmp_path, bundle):
    lock = bundle[3]
    monkeypatch.setattr(internal, "inspect_inputs", lambda root: lock)
    internal.run_internal_isic_preflight(tmp_path)
    path = internal.output_directory(tmp_path) / internal.LOCK_NAME
    before = path.read_bytes()
    monkeypatch.setattr(internal, "inspect_inputs", lambda root: {**lock, "phase": "changed"})
    with pytest.raises(ValueError, match="drift"):
        internal.run_internal_isic_preflight(tmp_path)
    assert path.read_bytes() == before


def test_preflight_checks_existing_partial_seed(monkeypatch, tmp_path, bundle):
    output = internal.output_directory(tmp_path)
    internal._publish_json(output / internal.LOCK_NAME, bundle[3])
    (output / "seed2026").mkdir()
    monkeypatch.setattr(internal, "inspect_inputs", lambda root: bundle[3])
    with pytest.raises(ValueError, match="file set"):
        internal.run_internal_isic_preflight(tmp_path)


def test_redirected_internal_output_rejected(monkeypatch, tmp_path):
    output = tmp_path / internal.OUTPUT
    original = Path.resolve
    def redirected(path, *args, **kwargs):
        if path == output:
            return tmp_path / "results/extensions/paul2026_swin/foreign"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "resolve", redirected)
    with pytest.raises(ValueError, match="Redirected"):
        internal.output_directory(tmp_path)


@pytest.mark.parametrize("image_path", ["data/external/hiba/image.jpg", "../foreign.jpg", "data/raw/hiba/image.jpg"])
def test_manifest_rejects_external_paths_before_image_access(tmp_path, image_path):
    path = tmp_path / fit.MANIFEST
    path.parent.mkdir(parents=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "image_id", "image_path", "split_included",
                                                   "split", "include_stage_1", "diagnosis_canonical"])
        writer.writeheader()
        writer.writerow({"dataset": "isic2019", "image_id": "synthetic", "image_path": image_path,
                         "split_included": "1", "split": "internal_test", "include_stage_1": "1",
                         "diagnosis_canonical": "melanoma"})
    with pytest.raises(ValueError):
        internal.manifest_cohort(tmp_path)
