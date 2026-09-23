"""Synthetic COM-05 tests: no real cohort, checkpoint, or remote execution."""
import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from src.evaluation import comesyso2026_external_hiba as external
from src.evaluation import comesyso2026_probability_fusion as fusion
from src.evaluation import comesyso2026_validation_fit as fit

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bundle(monkeypatch, tmp_path):
    monkeypatch.setattr(external, "COUNTS", (1, 1, 1, 1))
    monkeypatch.setattr(external, "PATIENT_COUNT", 3)
    truth = np.arange(4)
    ids, patients = ("a", "b", "c", "d"), ("p2", "p1", "p2", "p3")
    p1 = np.asarray([[.2, .8], [.8, .2], [.3, .7], [.1, .9]])
    p2 = np.asarray([[.7, .2, .1], [.8, .1, .1], [.1, .8, .1], [.1, .1, .8]])
    shared = fusion.SharedPredictions(ids, truth, (truth != 0).astype(int), p1.argmax(1), p1,
        np.where(truth == 0, -1, truth - 1), p2.argmax(1), p2, patients, 0.)
    cohort = external.cohort_identity(ids, patients, truth)
    lock = {"schema_version": 1, "phase": "COM-05", "status": "PASS", "zero_shot": True,
            "seeds": list(external.SEEDS), "cohort": cohort, "cohort_sha256": fit._digest(cohort),
            "checkpoints": [], "common_input_sha256": {}, "inference_input_sha256": {},
            "fusion_paths": {}, "feature_order": list(fusion.FEATURE_ORDER),
            "class_order": list(fusion.CLASS_NAMES)}
    lock.update(protocol_identity=fusion.PROTOCOL_ID, inference_performed=False, dataset_constructed=False,
                manifest_canonical_sha256=external.MANIFEST_CANONICAL_SHA256)
    for spec in (*external.com04.FLAT_SPECS, *external.com04.SHARED_SPECS):
        checkpoint = spec.path(tmp_path)
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"synthetic checkpoint bytes, never loaded")
        relative = checkpoint.relative_to(tmp_path).as_posix()
        digest = external.sha256_file(checkpoint)
        lock["checkpoints"].append({"system": spec.system, "seed": spec.seed, "epoch": spec.epoch,
                                    "path": relative, "sha256": digest})
        lock["inference_input_sha256"][relative] = digest
    # Explicit frozen-state fixture; these tests never train a classifier.
    classifier = fusion._new_classifier()
    classifier.classes_ = np.arange(4)
    classifier.coef_ = np.asarray([[0., 0., 0., 0.], [0., 5., 0., 0.],
                                  [0., 0., 5., 0.], [0., 0., 0., 5.]])
    classifier.intercept_ = np.zeros(4)
    classifier.n_iter_ = np.asarray([1])
    classifier.n_features_in_ = 4
    for seed in external.SEEDS:
        path = fit.OUTPUT / f"seed{seed}/fusion_model.json"
        (tmp_path / path).parent.mkdir(parents=True)
        fusion.save_fusion(fusion.FusionModel(classifier, seed, fusion.FIT_PARTITION, 4,
                                             fit.sklearn.__version__), tmp_path / path)
        lock["fusion_paths"][str(seed)] = path.as_posix()
        lock["common_input_sha256"][path.as_posix()] = external.sha256_file(tmp_path / path)
    real_bootstrap = fusion.patient_cluster_bootstrap
    def small_bootstrap(targets, predictions, patient_ids, *, replicates, seed):
        assert (replicates, seed) == (5000, 42)
        return real_bootstrap(targets, predictions, patient_ids, replicates=8, seed=seed)
    bootstrap = Mock(side_effect=small_bootstrap)
    monkeypatch.setattr(fusion, "patient_cluster_bootstrap", bootstrap)
    monkeypatch.setattr(fusion, "fit_fusion", Mock(side_effect=AssertionError("No HIBA fitting")))
    monkeypatch.setattr(fusion, "paired_image_bootstrap", Mock(side_effect=AssertionError("No image bootstrap")))
    monkeypatch.setattr(fusion, "mcnemar_comparisons", Mock(side_effect=AssertionError("No added HIBA test")))
    return SimpleNamespace(shared=shared, lock=lock, bootstrap=bootstrap, root=tmp_path)


def rows_for(bundle, seed=42):
    model = external.load_frozen_fusion(bundle.root, seed, bundle.lock)
    shared = bundle.shared
    features = fusion.build_fusion_features(shared.stage1_probabilities, shared.stage2_probabilities,
        sample_ids=shared.sample_ids, task2_sample_ids=shared.sample_ids)
    probabilities, _ = fusion.predict_fusion(model, features)
    return external.prediction_rows(shared, np.eye(4), probabilities, seed=seed,
                                    expected_cohort=bundle.lock["cohort"])


def install_lock(bundle):
    output = external.output_directory(bundle.root)
    external._publish_json(output / external.LOCK_NAME, bundle.lock)
    return output


def completed_inference(bundle):
    output = install_lock(bundle)
    for seed in external.SEEDS:
        external._publish_seed(output, seed, bundle.lock, "inference", rows=rows_for(bundle, seed))
    external._publish_json(output / "inference" / external.INFERENCE_COMPLETE,
                           external._inference_complete(output, bundle.lock))
    return output


def test_frozen_cohort_counts_and_identifiers():
    assert external.COUNTS == (696, 196, 229, 111)
    assert external.PATIENT_COUNT == 568
    targets = np.repeat(np.arange(4), external.COUNTS)
    ids = [f"image-{i}" for i in range(1232)]
    patients = [f"patient-{i % 568:04d}" for i in range(1232)]
    cohort = external.cohort_identity(ids, patients, targets)
    assert cohort["patient_ids"] == patients
    assert len(cohort["targets"]) == 1232
    for bad in (patients[:-1], ["one"] * 1232, [""] + patients[1:]):
        with pytest.raises(ValueError):
            external.cohort_identity(ids, bad, targets)
    with pytest.raises(ValueError):
        external.cohort_identity(ids, patients, np.zeros(1232, dtype=int))
    with pytest.raises(ValueError):
        external.cohort_identity(ids[:-1], patients[:-1], targets[:-1])


def test_exact_checkpoints():
    assert [(s.seed, s.epoch) for s in external.com04.FLAT_SPECS] == [(42, 9), (123, 10), (2026, 17)]
    assert [(s.seed, s.epoch) for s in external.com04.SHARED_SPECS] == [(42, 10), (123, 1), (2026, 26)]


def test_frozen_protocol_source():
    import yaml
    protocol = yaml.safe_load((ROOT / fusion.PROTOCOL_PATH).read_text(encoding="utf-8"))
    external.validate_protocol(protocol)
    for keys, value in [(("statistics", "hiba", "bootstrap", "replicates"), 4999),
                        (("statistics", "hiba", "bootstrap", "unit"), "image"),
                        (("datasets", "hiba", "zero_shot"), False)]:
        changed = copy.deepcopy(protocol)
        current = changed
        for key in keys[:-1]:
            current = current[key]
        current[keys[-1]] = value
        with pytest.raises(ValueError):
            external.validate_protocol(changed)


def test_unmasked_collector():
    class Model(torch.nn.Module):
        def forward(self, images):
            return {"task1": torch.tensor([[5., 0.]]).repeat(4, 1),
                    "task2": torch.tensor([[1., 2., 3.]]).repeat(4, 1)}
    shared = fusion.collect_shared_predictions(Model(), [{"image": torch.zeros(4, 3, 2, 2),
        "target": torch.arange(4), "image_id": ["a", "b", "c", "d"],
        "patient_id": ["p1", "p1", "p2", "p3"]}])
    assert np.all(shared.stage1_predictions == 0)
    assert np.isfinite(shared.stage2_probabilities).all()
    np.testing.assert_allclose(shared.stage2_probabilities.sum(1), 1)
    assert shared.patient_ids == ("p1", "p1", "p2", "p3")


def test_five_systems_and_rescue(bundle):
    rows = rows_for(bundle)
    assert [r["hard_prediction"] for r in rows] == [1, 0, 2, 3]
    assert [r["path_soft_prediction"] for r in rows] == [1, 0, 2, 3]
    assert [r["fusion_prediction"] for r in rows] == [1, 1, 2, 3]
    assert [r["oracle_prediction"] for r in rows] == [0, 1, 2, 3]
    result, rescue = external.analyze_rows(rows, 42, bundle.lock)
    assert sum(rescue["pairwise_partition"].values()) == 4
    assert rescue["malignant_routed_nm"]["rescued"] == 1
    assert rescue["malignant_routed_nm"]["by_class"]["melanoma"]["rescued"] == 1
    assert rescue["nm_routed_malignant"]["new_errors"] == 0
    assert set(rescue["pairwise_partition"]) == set(external.com04.PARTITIONS)
    for name, (a, b) in fusion.CONTRASTS.items():
        assert result["contrasts"][name]["macro_f1"] == (
            result["systems"][a]["macro_f1"] - result["systems"][b]["macro_f1"])
    assert set(result["statistics"]) == {"bootstrap"}
    assert all(result["systems"][s]["labels"] == [0, 1, 2, 3] for s in external.SYSTEMS)
    fusion.fit_fusion.assert_not_called()


def test_cluster_draws_duplicate_whole_patient_clusters():
    patients = ["z", "a", "z", "b", "a"]
    rng = np.random.default_rng(42)
    draws = list(fusion.patient_cluster_indices(patients, replicates=20, seed=42))
    repeated = False
    for indices in draws:
        selected = rng.choice(np.array(["a", "b", "z"], dtype=object), size=3, replace=True)
        expected = np.concatenate([np.flatnonzero(np.asarray(patients) == p) for p in selected])
        np.testing.assert_array_equal(indices, expected)
        repeated |= len(set(selected)) < 3
        assert np.count_nonzero(indices == 0) == np.count_nonzero(indices == 2)
        assert np.count_nonzero(indices == 1) == np.count_nonzero(indices == 4)
    assert repeated


def test_cluster_bootstrap_fixed_four_labels_without_all_classes(monkeypatch):
    predictions = {s: np.arange(4) for s in external.SYSTEMS}
    monkeypatch.setattr(fusion, "patient_cluster_indices",
                        lambda *a, **k: iter([np.asarray([0, 0])] * k["replicates"]))
    seen = []
    original = fusion.calculate_metrics
    def tracked(targets, preds):
        result = original(targets, preds)
        seen.append(result.macro_f1)
        return result
    monkeypatch.setattr(fusion, "calculate_metrics", tracked)
    stats = fusion.patient_cluster_bootstrap(np.arange(4), predictions, ["a", "b", "c", "d"],
                                             replicates=2, seed=42)
    assert .25 in seen  # One perfect present class and three retained absent classes.
    assert stats["unit"] == "patient_cluster"
    assert stats["interval"] == "percentile"
    assert stats["quantile_method"] == "linear"
    assert stats["confidence_level"] == .95


def test_fusion_loaded_never_fitted_and_hash_bound(bundle):
    model = external.load_frozen_fusion(bundle.root, 123, bundle.lock)
    assert model.seed == 123
    fusion.fit_fusion.assert_not_called()
    path = bundle.root / bundle.lock["fusion_paths"]["123"]
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        external.load_frozen_fusion(bundle.root, 123, bundle.lock)


@pytest.mark.parametrize("field,value", [("classes", [3, 2, 1, 0]), ("feature_order", ["wrong"]),
                                          ("fitting_partition", "hiba")])
def test_fusion_schema_rejected_even_with_matching_hash(bundle, field, value):
    path = bundle.root / bundle.lock["fusion_paths"]["42"]
    data = external._read_json(path)
    data[field] = value
    path.write_bytes(fit._json_bytes(data))
    bundle.lock["common_input_sha256"][bundle.lock["fusion_paths"]["42"]] = external.sha256_file(path)
    with pytest.raises(ValueError):
        external.load_frozen_fusion(bundle.root, 42, bundle.lock)


def test_preflight_no_dataset_or_inference(monkeypatch, bundle):
    monkeypatch.setattr(external, "inspect_inputs", lambda root: bundle.lock)
    trap = Mock(side_effect=AssertionError("Preflight must not infer"))
    monkeypatch.setattr(external, "infer_one_seed", trap)
    monkeypatch.setattr(external.hiba, "build_hiba_loader", trap)
    assert external.run_external_hiba_preflight(bundle.root) == bundle.lock
    assert external.run_external_hiba_preflight(bundle.root) == bundle.lock
    trap.assert_not_called()
    bundle.bootstrap.assert_not_called()


def test_inference_stage_no_statistics_and_resume(monkeypatch, bundle):
    output = install_lock(bundle)
    external._publish_seed(output, 42, bundle.lock, "inference", rows=rows_for(bundle))
    infer = Mock(side_effect=lambda root, seed, lock, **kwargs: rows_for(bundle, seed))
    monkeypatch.setattr(external, "infer_one_seed", infer)
    stats = Mock(side_effect=AssertionError("GPU function must not calculate statistics"))
    monkeypatch.setattr(external, "analyze_rows", stats)
    monkeypatch.setattr(external, "three_seed_summary", stats)
    monkeypatch.setattr(fusion, "four_class_metrics", stats)
    commits = Mock()
    marker = external.run_external_hiba_inference(bundle.root, device="cpu", persist_callback=commits)
    assert [c.args[1] for c in infer.call_args_list] == [123, 2026]
    assert commits.call_count == 3
    assert external.run_external_hiba_inference(bundle.root, device="cpu") == marker
    assert infer.call_count == 2
    bundle.bootstrap.assert_not_called()
    stats.assert_not_called()


def test_cpu_analysis_no_inference_and_resume(monkeypatch, bundle):
    output = completed_inference(bundle)
    trap = Mock(side_effect=AssertionError("CPU analysis must not load/forward neural models"))
    monkeypatch.setattr(external, "infer_one_seed", trap)
    monkeypatch.setattr(external.hiba, "build_hiba_loader", trap)
    monkeypatch.setattr(external.com04.historical, "load_selected_model", trap)
    monkeypatch.setattr(fusion, "collect_shared_predictions", trap)
    commits = Mock()
    result, rescue = external.analyze_rows(rows_for(bundle), 42, bundle.lock)
    external._publish_seed(output, 42, bundle.lock, "analysis", result=result, rescue=rescue)
    bundle.bootstrap.reset_mock()
    summary = external.run_external_hiba_analysis(bundle.root, persist_callback=commits)
    assert bundle.bootstrap.call_count == 2
    assert commits.call_count == 3
    assert summary["phase"] == "COM-05"
    assert summary["pooled"] is False
    assert external.run_external_hiba_analysis(bundle.root) == summary
    assert bundle.bootstrap.call_count == 2
    trap.assert_not_called()


def test_three_seed_sample_sd_no_pooling(bundle):
    results = {s: external.analyze_rows(rows_for(bundle, s), s, bundle.lock)[0] for s in external.SEEDS}
    for seed, value in zip(external.SEEDS, (.1, .2, .3)):
        results[seed]["systems"]["fusion"]["macro_f1"] = value
    summary = external.three_seed_summary(results)
    scalar = summary["systems"]["fusion"]["macro_f1"]
    assert scalar["mean"] == pytest.approx(.2)
    assert scalar["sample_standard_deviation"] == pytest.approx(.1)
    assert set(scalar["individual_seed_values"]) == {"42", "123", "2026"}
    with pytest.raises(ValueError):
        external.three_seed_summary({42: results[42]})


@pytest.mark.parametrize("stage", ["inference", "analysis"])
def test_partial_seed_blocks_all_work(monkeypatch, bundle, stage):
    output = completed_inference(bundle) if stage == "analysis" else install_lock(bundle)
    (output / stage / "seed2026").mkdir(parents=True)
    trap = Mock(side_effect=AssertionError("No work after partial artifact"))
    monkeypatch.setattr(external, "infer_one_seed", trap)
    monkeypatch.setattr(external, "analyze_rows", trap)
    with pytest.raises(ValueError, match="file set"):
        if stage == "inference":
            external.run_external_hiba_inference(bundle.root, device="cpu")
        else:
            external.run_external_hiba_analysis(bundle.root)
    trap.assert_not_called()


@pytest.mark.parametrize("name", [external.SUMMARY_NAME, external.COMPLETE_NAME])
def test_partial_final_analysis_rejected(bundle, name):
    output = completed_inference(bundle)
    external._publish_json(output / "analysis" / name, {})
    with pytest.raises(ValueError, match="Partial"):
        external.run_external_hiba_analysis(bundle.root)
    bundle.bootstrap.assert_not_called()


def test_corrupted_inference_and_no_overwrite(bundle):
    output = completed_inference(bundle)
    with pytest.raises(FileExistsError):
        external._publish_seed(output, 42, bundle.lock, "inference", rows=rows_for(bundle))
    path = output / "inference/seed42/paired_predictions.csv"
    path.write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="drift"):
        external.run_external_hiba_analysis(bundle.root)
    bundle.bootstrap.assert_not_called()


def test_patient_pairing_not_remapped(bundle):
    rows = rows_for(bundle)
    assert [r["patient_id"] for r in rows] == list(bundle.shared.patient_ids)
    changed = copy.deepcopy(bundle.lock["cohort"])
    changed["patient_ids"][0] = "p1"
    with pytest.raises(ValueError, match="pairing"):
        external.prediction_rows(bundle.shared, np.eye(4), np.eye(4), seed=42, expected_cohort=changed)


def test_missing_lock_and_analysis_requires_inference(bundle):
    with pytest.raises(FileNotFoundError):
        external.run_external_hiba_inference(bundle.root, device="cpu")
    install_lock(bundle)
    with pytest.raises(ValueError, match="all completed inference"):
        external.run_external_hiba_analysis(bundle.root)


def test_late_preflight_and_incompatible_lock(monkeypatch, bundle):
    output = install_lock(bundle)
    before = (output / external.LOCK_NAME).read_bytes()
    monkeypatch.setattr(external, "inspect_inputs", lambda root: {**bundle.lock, "phase": "changed"})
    with pytest.raises(ValueError, match="drift"):
        external.run_external_hiba_preflight(bundle.root)
    assert (output / external.LOCK_NAME).read_bytes() == before


def test_output_boundary_and_unknown_stage(monkeypatch, bundle):
    output = external.output_directory(bundle.root)
    assert output == bundle.root / external.OUTPUT
    with pytest.raises(ValueError):
        external._publish_seed(output, 42, bundle.lock, "../../outside")
    with pytest.raises(ValueError):
        external.output_directory(bundle.root / ".." / "outside")
    monkeypatch.setattr(external, "OUTPUT", Path("results/extensions/paul2026_swin/external_hiba"))
    with pytest.raises(ValueError):
        external.output_directory(bundle.root)


def test_redirected_stage_rejected(monkeypatch, bundle):
    output = install_lock(bundle)
    (output / "inference").mkdir()
    original = Path.resolve
    def redirected(path, *args, **kwargs):
        if path == output / "inference":
            return bundle.root / "foreign"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "resolve", redirected)
    with pytest.raises(ValueError, match="Redirected"):
        external.validate_outputs(output, bundle.lock)



def test_modal_stage_architecture():
    path = ROOT / "scripts/comesyso2026/modal_comesyso2026_external_hiba.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert set(functions) == {"run_external_hiba_preflight", "run_external_hiba_inference", "run_external_hiba_analysis"}
    for name, fn in functions.items():
        options = {k.arg: k.value for k in fn.decorator_list[0].keywords}
        assert ast.literal_eval(options["max_containers"]) == 1
        assert ast.literal_eval(options["retries"]) == 0
        assert ast.literal_eval(options["single_use_containers"]) is True
        if name.endswith("inference"):
            assert ast.literal_eval(options["gpu"]) == "T4"
        else:
            assert "gpu" not in options
        namespace = {"hiba_volume": object(), "checkpoint_volume": object(), "output_volume": object(),
                     "HIBA_SOURCE_MOUNT": "/mnt/comesyso2026-hiba-data",
                     "INPUT_CHECKPOINT_MOUNT": "/root/project/results/extensions/paul2026_swin",
                     "OUTPUT_MOUNT": "/root/project/results/extensions/comesyso2026_probability_fusion"}
        mounts = eval(compile(ast.Expression(options["volumes"]), str(path), "eval"), namespace)
        assert len({id(volume) for volume in mounts.values()}) == len(mounts)
        if name.endswith("analysis"):
            assert ast.unparse(options["volumes"]) == "{OUTPUT_MOUNT: output_volume}"
            assert "stage_hiba_images" not in ast.unparse(fn)
        else:
            assert mounts[namespace["HIBA_SOURCE_MOUNT"]] is namespace["hiba_volume"]
            assert mounts[namespace["INPUT_CHECKPOINT_MOUNT"]] is namespace["checkpoint_volume"]
            assert mounts[namespace["OUTPUT_MOUNT"]] is namespace["output_volume"]
            assert "/root/project/data/raw" not in mounts
            assert "/root/project/data/external/hiba/extracted" not in mounts
            calls = [n.value for n in fn.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)]
            assert calls[0].func.id == "stage_hiba_images"
            assert isinstance(fn.body[-1], ast.Return)
    assert "comesyso2026-hiba-data" in source
    assert 'HIBA_SOURCE_MOUNT = "/mnt/comesyso2026-hiba-data"' in source
    assert "data/raw/emb/images/isic" not in source
    assert "prepare_hiba_volume_paths" not in source
    assert "modal.App(" not in source


def test_no_fit_training_or_stage_c_neural_calls_in_source():
    source = (ROOT / "src/evaluation/comesyso2026_external_hiba.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    forbidden = {"fit", "fit_fusion", "train", "paired_image_bootstrap", "mcnemar_comparisons"}
    assert not any(isinstance(n.func, ast.Attribute) and n.func.attr in forbidden for n in calls)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name in ("run_external_hiba_analysis", "analyze_rows"):
        text = ast.unparse(functions[name])
        assert "load_selected_model" not in text and "collect_shared_predictions" not in text
    assert "bootstrap" not in ast.unparse(functions["run_external_hiba_inference"])
    assert "build_hiba_loader" not in ast.unparse(functions["inspect_inputs"])


def test_launcher_modes_busy_checks_and_dispatch(monkeypatch):
    functions = {}
    busy = {"name": None}
    def lookup(app, name):
        assert app == "comesyso2026-probability-fusion"
        fn = Mock()
        fn.get_current_stats.return_value = SimpleNamespace(num_running_inputs=int(name == busy["name"]), backlog=0)
        fn.spawn.return_value = SimpleNamespace(object_id="synthetic-call")
        functions[name] = fn
        return fn
    modal = SimpleNamespace(App=lambda name: SimpleNamespace(local_entrypoint=lambda: lambda fn: fn),
                            Function=SimpleNamespace(from_name=lookup))
    monkeypatch.setitem(__import__("sys").modules, "modal", modal)
    path = ROOT / "scripts/comesyso2026/launch_comesyso2026_external_hiba.py"
    spec = importlib.util.spec_from_file_location("com05_launcher_test", path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    assert set(launcher.FUNCTION_BY_MODE) == {"preflight", "infer", "analyze"}
    for mode in ("fit", "train", "evaluate", "analysis", "inference"):
        with pytest.raises(ValueError, match="COM-05 mode must be preflight, infer, or analyze"):
            launcher.main(mode)
    for mode in launcher.FUNCTION_BY_MODE:
        functions.clear()
        launcher.main(mode)
        assert len(functions) == 7
        functions[launcher.FUNCTION_BY_MODE[mode]].spawn.assert_called_once_with()
        assert sum(f.spawn.call_count for f in functions.values()) == 1
    functions.clear()
    busy["name"] = "run_internal_isic_production"
    with pytest.raises(SystemExit, match="running or queued"):
        launcher.main("infer")
    assert sum(f.spawn.call_count for f in functions.values()) == 0


@pytest.mark.parametrize("field,value", [("epoch", 999), ("seed", 999), ("system", "replacement"),
                                          ("path", "foreign.pt"), ("sha256", "invalid")])
def test_checkpoint_record_mismatch_fails_closed(bundle, field, value):
    changed = copy.deepcopy(bundle.lock["checkpoints"])
    changed[0][field] = value
    with pytest.raises(ValueError):
        external.validate_checkpoint_records(changed, bundle.root)


def test_checkpoint_hash_anchor_mismatch(bundle):
    bundle.lock["checkpoints"][0]["sha256"] = "f" * 64
    install_lock(bundle)
    with pytest.raises(ValueError, match="hash anchor"):
        external.verify_preflight_lock(bundle.root)


def test_cpu_analysis_needs_no_checkpoint_or_image_files(bundle):
    completed_inference(bundle)
    for relative in bundle.lock["inference_input_sha256"]:
        (bundle.root / relative).unlink()
    # Stage C intentionally verifies common source/provenance + persisted bytes,
    # not unavailable neural weights/raw images. Stage B still rejects absence.
    summary = external.run_external_hiba_analysis(bundle.root)
    assert summary["zero_shot"] is True
    with pytest.raises(FileNotFoundError):
        external.verify_preflight_lock(bundle.root, inference=True)


def test_one_flat_and_one_unmasked_shared_forward_collection(monkeypatch, bundle):
    shared = bundle.shared
    rows = [{"isic_id": sid, "patient_id": pid, "target_index": str(target)}
            for sid, pid, target in zip(shared.sample_ids, shared.patient_ids, shared.flat_targets)]
    loader = SimpleNamespace(dataset=SimpleNamespace(rows=rows))
    monkeypatch.setattr(external.hiba, "build_hiba_loader", Mock(return_value=loader))
    model_loader = Mock(return_value=Mock())
    monkeypatch.setattr(external.com04.historical, "load_selected_model", model_loader)
    flat = SimpleNamespace(sample_ids=shared.sample_ids, targets=shared.flat_targets,
                           probabilities=np.eye(4), predictions=np.arange(4))
    flat_collector = Mock(return_value=flat)
    shared_collector = Mock(return_value=shared)
    monkeypatch.setattr(external.com04.historical, "collect_single_task_predictions", flat_collector)
    monkeypatch.setattr(fusion, "collect_shared_predictions", shared_collector)
    output = external.infer_one_seed(bundle.root, 42, bundle.lock, device="cpu")
    assert len(output) == 4
    assert model_loader.call_count == 2
    flat_collector.assert_called_once()
    shared_collector.assert_called_once()
    assert [call.args[1].system for call in model_loader.call_args_list] == ["flat", "shared_hard"]
    bundle.bootstrap.assert_not_called()
    fusion.fit_fusion.assert_not_called()


def test_late_preflight_fails_without_inspection(monkeypatch, bundle):
    output = external.output_directory(bundle.root)
    (output / "inference/seed42").mkdir(parents=True)
    inspector = Mock(side_effect=AssertionError("No late lock"))
    monkeypatch.setattr(external, "inspect_inputs", inspector)
    with pytest.raises(ValueError, match="before-inference"):
        external.run_external_hiba_preflight(bundle.root)
    inspector.assert_not_called()


def test_preflight_rejects_partial_existing_inference(monkeypatch, bundle):
    output = install_lock(bundle)
    (output / "inference/seed123").mkdir(parents=True)
    monkeypatch.setattr(external, "inspect_inputs", lambda root: bundle.lock)
    with pytest.raises(ValueError, match="file set"):
        external.run_external_hiba_preflight(bundle.root)


def test_corrupt_completed_analysis_blocks_remaining_statistics(bundle):
    output = completed_inference(bundle)
    result, rescue = external.analyze_rows(rows_for(bundle), 42, bundle.lock)
    external._publish_seed(output, 42, bundle.lock, "analysis", result=result, rescue=rescue)
    (output / "analysis/seed42/rescue_analysis.json").write_text("{}", encoding="utf-8")
    bundle.bootstrap.reset_mock()
    with pytest.raises(ValueError, match="drift"):
        external.run_external_hiba_analysis(bundle.root)
    bundle.bootstrap.assert_not_called()


def test_incomplete_inference_completion_marker_rejected(bundle):
    output = install_lock(bundle)
    external._publish_json(output / "inference" / external.INFERENCE_COMPLETE, {})
    with pytest.raises(ValueError, match="Partial inference"):
        external.validate_outputs(output, bundle.lock)


def test_common_provenance_drift_blocks_cpu_analysis(bundle):
    completed_inference(bundle)
    path = bundle.root / bundle.lock["fusion_paths"]["42"]
    path.write_text("changed model", encoding="utf-8")
    with pytest.raises(ValueError, match="Locked input drift"):
        external.run_external_hiba_analysis(bundle.root)
    bundle.bootstrap.assert_not_called()




def test_hiba_staging_copies_real_manifest_bound_files(monkeypatch, tmp_path):
    source_root = tmp_path / "hiba-volume"
    source_images = source_root / "images"
    source_images.mkdir(parents=True)
    payload = b"synthetic hiba image bytes"
    source = source_images / "ISIC_TEST.jpg"
    source.write_bytes(payload)
    digest = external.sha256_file(source)

    manifest = tmp_path / "data/external/hiba/manifests/frozen.csv"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "isic_id,image_path,image_sha256\n"
        f"ISIC_TEST,data/external/hiba/extracted/images/ISIC_TEST.jpg,{digest}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(external, "MANIFEST", manifest.relative_to(tmp_path))

    assert external.stage_hiba_images(tmp_path, source_root) == 1
    destination = tmp_path / "data/external/hiba/extracted/images/ISIC_TEST.jpg"
    assert destination.is_file()
    assert not destination.is_symlink()
    assert destination.read_bytes() == payload
    assert external.stage_hiba_images(tmp_path, source_root) == 1
    assert external.sha256_file(destination) == digest


def test_hiba_staging_fails_closed_on_missing_or_wrong_source(monkeypatch, tmp_path):
    source_root = tmp_path / "hiba-volume"
    source_images = source_root / "images"
    source_images.mkdir(parents=True)
    manifest = tmp_path / "data/external/hiba/manifests/frozen.csv"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "isic_id,image_path,image_sha256\n"
        "ISIC_TEST,data/external/hiba/extracted/images/ISIC_TEST.jpg," + "0" * 64 + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(external, "MANIFEST", manifest.relative_to(tmp_path))
    with pytest.raises(FileNotFoundError, match="source image is missing"):
        external.stage_hiba_images(tmp_path, source_root)
    (source_images / "ISIC_TEST.jpg").write_bytes(b"wrong bytes")
    with pytest.raises(ValueError):
        external.stage_hiba_images(tmp_path, source_root)


def test_no_emb_alias_or_symlink_creation_in_com05_sources():
    evaluator = (ROOT / "src/evaluation/comesyso2026_external_hiba.py").read_text(encoding="utf-8")
    modal_source = (ROOT / "scripts/comesyso2026/modal_comesyso2026_external_hiba.py").read_text(encoding="utf-8")
    combined = evaluator + "\n" + modal_source
    assert "data/raw/emb/images/isic" not in combined
    assert "prepare_hiba_volume_paths" not in combined
    assert ".symlink_to(" not in combined
    assert "comesyso2026-hiba-data" in modal_source
    assert "/mnt/comesyso2026-hiba-data" in modal_source
