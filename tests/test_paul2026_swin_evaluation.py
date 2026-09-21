"""Offline contracts using synthetic checkpoints/predictions; no test images."""

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation import paul2026_swin_evaluation as ev
from src.evaluation.phase04_comparative_harness import PredictionCollection, SharedISICPredictionCollection

ROOT = Path(__file__).resolve().parents[1]


def summary_for(spec):
    return {"run_name": spec.run_name, "seed": spec.seed,
            "best_epoch": spec.epoch, "reportable_as_full_result": True}


def payload_for(spec):
    return {"epoch": spec.epoch, "model_state_dict": {}, "config": {
        "experiment": {"run_name": spec.run_name, "seed": spec.seed, "model": "swin_t"},
        "model": {"architecture": "swin_t", "dropout_probability": 0.2},
    }}


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    manifest = tmp_path / ev.MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_text("manifest bytes only; never parsed in preflight", encoding="utf-8")
    payloads = {}
    for spec in ev.CHECKPOINTS:
        path = spec.path(tmp_path)
        path.parent.mkdir(parents=True)
        path.write_bytes(spec.run_name.encode())
        (path.parent / "run_summary.json").write_text(json.dumps(summary_for(spec)), encoding="utf-8")
        payloads[str(path)] = payload_for(spec)
    load = Mock(side_effect=lambda path, **kw: deepcopy(payloads[str(path)]))
    monkeypatch.setattr(ev.torch, "load", load)
    models = []

    def model(*args, **kwargs):
        result = Mock()
        result.parameters.return_value = [torch.zeros(2, 3)]
        result.eval.return_value = result
        models.append(result)
        return result

    flat_builder = Mock(side_effect=model)
    shared_builder = Mock(side_effect=model)
    monkeypatch.setattr(ev, "build_swin_t_classification_model", flat_builder)
    monkeypatch.setattr(ev, "build_swin_t_shared_three_task_model", shared_builder)
    return SimpleNamespace(root=tmp_path, payloads=payloads, load=load, models=models,
                           flat_builder=flat_builder, shared_builder=shared_builder)


def forbid(*args, **kwargs):
    pytest.fail("Forbidden dataset, prediction, training, or CUDA action")


def test_frozen_selections():
    assert ev.SEEDS == (42, 123, 2026)
    assert [(s.system, s.seed, s.epoch) for s in ev.CHECKPOINTS] == [
        ("flat", 42, 9), ("flat", 123, 10), ("flat", 2026, 17),
        ("shared_hard", 42, 10), ("shared_hard", 123, 1), ("shared_hard", 2026, 26),
    ]
    for spec in ev.CHECKPOINTS:
        assert spec.run_name == f"paul2026_{spec.system}_swin_t_seed{spec.seed}"
        assert spec.path(ROOT).name == "best_checkpoint.pt"
    assert ev.EXPECTED_N == 3668
    assert ev.CLASS_NAMES == ("non_malignant", "melanoma", "bcc", "scc")
    assert tuple(ev.FINAL_CLASS_NAMES) == ev.CLASS_NAMES


@pytest.mark.parametrize("key,value", [
    ("reportable_as_full_result", False), ("reportable_as_full_result", 1),
    ("seed", 43), ("seed", None), ("run_name", "wrong"), ("best_epoch", 10),
])
def test_summary_mismatch_rejected(key, value):
    spec = ev.CHECKPOINTS[0]
    summary = summary_for(spec)
    summary[key] = value
    with pytest.raises(ValueError, match=key):
        ev.validate_checkpoint_and_summary(payload_for(spec), summary, spec)


@pytest.mark.parametrize("field,value", [
    ("epoch", 10), ("seed", 43), ("run_name", "wrong"),
    ("class_names", ["melanoma", "non_malignant", "bcc", "scc"]),
    ("class_mappings", {}), ("architecture", "resnet50"),
])
def test_checkpoint_identity_mismatch(field, value):
    spec = ev.CHECKPOINTS[0]
    payload = payload_for(spec)
    payload[field] = value
    with pytest.raises(ValueError, match=field):
        ev.validate_checkpoint_and_summary(payload, summary_for(spec), spec)


def test_nested_shared_metadata_and_flat_config():
    for spec in ev.CHECKPOINTS:
        payload = payload_for(spec)
        payload["config_metadata"] = {"experiment": {"seed": spec.seed, "run_name": spec.run_name}}
        payload["model_metadata"] = {"architecture": "swin_t"}
        ev.validate_checkpoint_and_summary(payload, summary_for(spec), spec)
        payload["config_metadata"]["experiment"]["seed"] = 99
        with pytest.raises(ValueError, match="seed"):
            ev.validate_checkpoint_and_summary(payload, summary_for(spec), spec)


@pytest.mark.parametrize("summary_seed_present", [False, True])
@pytest.mark.parametrize("spec", ev.CHECKPOINTS[:3], ids=lambda spec: spec.run_name)
def test_flat_summary_seed_optional_with_matching_checkpoint(spec, summary_seed_present):
    summary = summary_for(spec)
    if not summary_seed_present:
        del summary["seed"]
    ev.validate_checkpoint_and_summary(payload_for(spec), summary, spec)


@pytest.mark.parametrize("spec", ev.CHECKPOINTS[:3], ids=lambda spec: spec.run_name)
def test_flat_summary_conflicting_seed_rejected(spec):
    summary = summary_for(spec)
    summary["seed"] = spec.seed + 1
    with pytest.raises(ValueError, match=r"run_summary\.seed"):
        ev.validate_checkpoint_and_summary(payload_for(spec), summary, spec)


@pytest.mark.parametrize("summary_seed_present", [False, True])
@pytest.mark.parametrize("spec", ev.CHECKPOINTS[:3], ids=lambda spec: spec.run_name)
@pytest.mark.parametrize("field,value", [
    ("seed", None), ("seed", 99), ("run_name", None), ("run_name", "wrong"),
])
def test_flat_checkpoint_experiment_identity_required(spec, summary_seed_present, field, value):
    summary = summary_for(spec)
    if not summary_seed_present:
        del summary["seed"]
    payload = payload_for(spec)
    if value is None:
        del payload["config"]["experiment"][field]
    else:
        payload["config"]["experiment"][field] = value
    with pytest.raises(ValueError, match=rf"checkpoint\.config\.experiment\.{field}"):
        ev.validate_checkpoint_and_summary(payload, summary, spec)


@pytest.mark.parametrize("field", ["config", "experiment"])
def test_flat_checkpoint_config_identity_container_required(field):
    spec = ev.CHECKPOINTS[0]
    payload = payload_for(spec)
    if field == "config":
        del payload["config"]
        label = r"checkpoint\.config"
    else:
        del payload["config"]["experiment"]
        label = r"checkpoint\.config\.experiment"
    with pytest.raises(ValueError, match=label):
        ev.validate_checkpoint_and_summary(payload, summary_for(spec), spec)


@pytest.mark.parametrize("spec", ev.CHECKPOINTS[3:], ids=lambda spec: spec.run_name)
@pytest.mark.parametrize("summary_seed_present", [False, True])
def test_shared_summary_seed_required_and_matching(spec, summary_seed_present):
    summary = summary_for(spec)
    if summary_seed_present:
        summary["seed"] = spec.seed + 1
    else:
        del summary["seed"]
    with pytest.raises(ValueError, match=r"run_summary\.seed"):
        ev.validate_checkpoint_and_summary(payload_for(spec), summary, spec)


@pytest.mark.parametrize("flat_summary_seed_present", [False, True])
def test_preflight_only_hashes_and_strict_loads(artifacts, monkeypatch, flat_summary_seed_present):
    if not flat_summary_seed_present:
        for spec in ev.CHECKPOINTS[:3]:
            path = spec.path(artifacts.root).parent / "run_summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            del summary["seed"]
            path.write_text(json.dumps(summary), encoding="utf-8")
    for name in ("ISIC2019HierarchicalDataset", "DataLoader", "build_locked_isic_loader",
                 "collect_single_task_predictions", "collect_shared_isic_predictions",
                 "compute_classification_metrics"):
        monkeypatch.setattr(ev, name, forbid)
    monkeypatch.setattr(torch.cuda, "is_available", forbid)
    callback = Mock()
    lock = ev.checkpoint_only_preflight(artifacts.root, persist_callback=callback)
    assert lock["status"] == "PASS"
    assert lock["created_before_test_execution"] is True
    assert lock["internal_test_dataset_constructed"] is False
    assert len(lock["checkpoints"]) == 6
    expected_seeds = [42, 123, 2026, 42, 123, 2026]
    assert [record["seed"] for record in lock["checkpoints"]] == expected_seeds
    persisted = json.loads((artifacts.root / ev.EVALUATION / ev.LOCK_NAME).read_text(encoding="utf-8"))
    assert [record["seed"] for record in persisted["checkpoints"]] == expected_seeds
    for spec, record in zip(ev.CHECKPOINTS, lock["checkpoints"]):
        assert record["system"] == spec.system
        assert record["run_name"] == spec.run_name
        assert record["sha256"] == hashlib.sha256(spec.path(artifacts.root).read_bytes()).hexdigest()
        assert record["parameter_count"] == 6
        assert record["byte_size"] == spec.path(artifacts.root).stat().st_size
    for call in artifacts.load.call_args_list:
        assert call.kwargs == {"map_location": "cpu", "weights_only": False}
    for model in artifacts.models:
        model.load_state_dict.assert_called_once_with({}, strict=True)
    assert artifacts.flat_builder.call_args.args == (4,)
    assert artifacts.flat_builder.call_args.kwargs == {"pretrained": "none", "dropout_probability": 0.2}
    assert artifacts.shared_builder.call_args.kwargs == {"pretrained": "none", "dropout_probability": 0.2}
    callback.assert_called_once_with()


def test_strict_loading_failure_never_publishes_lock(artifacts):
    artifacts.flat_builder.side_effect = None
    artifacts.flat_builder.return_value.load_state_dict.side_effect = RuntimeError("missing key")
    with pytest.raises(RuntimeError, match="missing key"):
        ev.checkpoint_only_preflight(artifacts.root)
    assert not (artifacts.root / ev.EVALUATION / ev.LOCK_NAME).exists()


@pytest.mark.parametrize("missing", ["best_checkpoint.pt", "run_summary.json"])
def test_missing_artifact_is_not_replaced_by_last_checkpoint(artifacts, missing):
    directory = ev.CHECKPOINTS[0].path(artifacts.root).parent
    (directory / missing).unlink()
    (directory / "last_checkpoint.pt").write_bytes(b"must not be used")
    with pytest.raises(FileNotFoundError):
        ev.checkpoint_only_preflight(artifacts.root)
    assert not (artifacts.root / ev.EVALUATION / ev.LOCK_NAME).exists()


def test_existing_identical_lock_is_immutable(artifacts):
    ev.checkpoint_only_preflight(artifacts.root)
    path = artifacts.root / ev.EVALUATION / ev.LOCK_NAME
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    ev.checkpoint_only_preflight(artifacts.root)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    lock = json.loads(path.read_text())
    lock["created_before_test_execution"] = False
    path.write_text(json.dumps(lock))
    incompatible = path.read_bytes()
    with pytest.raises(ValueError, match="Incompatible"):
        ev.checkpoint_only_preflight(artifacts.root)
    assert path.read_bytes() == incompatible


def test_missing_lock_before_any_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "ISIC2019HierarchicalDataset", forbid)
    monkeypatch.setattr(ev, "_inspect_checkpoints", forbid)
    with pytest.raises(FileNotFoundError, match="preflight lock missing"):
        ev.run_isic_evaluation(tmp_path, device="cpu")


@pytest.mark.parametrize("drift", ["checkpoint", "manifest", "summary"])
def test_drift_rejected_before_dataset(artifacts, monkeypatch, drift):
    ev.checkpoint_only_preflight(artifacts.root)
    monkeypatch.setattr(ev, "ISIC2019HierarchicalDataset", forbid)
    if drift == "checkpoint":
        ev.CHECKPOINTS[0].path(artifacts.root).write_bytes(b"changed checkpoint")
    elif drift == "manifest":
        (artifacts.root / ev.MANIFEST).write_bytes(b"changed manifest")
    else:
        path = ev.CHECKPOINTS[0].path(artifacts.root).parent / "run_summary.json"
        summary = json.loads(path.read_text())
        summary["seed"] = 7
        path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        ev.run_isic_evaluation(artifacts.root, device="cpu")


def test_cuda_is_checked_only_when_requested(artifacts, monkeypatch):
    available = Mock(return_value=False)
    monkeypatch.setattr(torch.cuda, "is_available", available)
    with pytest.raises(RuntimeError, match="CUDA"):
        ev.checkpoint_only_preflight(artifacts.root, require_cuda=True)
    available.assert_called_once_with()


def test_loader_and_transform_contract(tmp_path, monkeypatch):
    dataset = Mock(return_value=range(3668))
    monkeypatch.setattr(ev, "ISIC2019HierarchicalDataset", dataset)
    loader = ev.build_locked_isic_loader(tmp_path)
    assert dataset.call_args.args[:4] == (tmp_path / ev.MANIFEST, tmp_path, "internal_test", "flat_four_class")
    transform = dataset.call_args.args[4]
    assert [type(t).__name__ for t in transform.transforms] == ["ToImage", "Resize", "CenterCrop", "ToDtype", "Normalize"]
    assert transform.transforms[1].size == [256]
    assert tuple(transform.transforms[2].size) == (224, 224)
    assert tuple(transform.transforms[-1].mean) == (0.485, 0.456, 0.406)
    assert tuple(transform.transforms[-1].std) == (0.229, 0.224, 0.225)
    assert loader.batch_size == 64 and loader.num_workers == 4
    assert loader.drop_last is False
    assert isinstance(loader.sampler, torch.utils.data.SequentialSampler)
    dataset.return_value = range(3667)
    with pytest.raises(ValueError, match="sample count"):
        ev.build_locked_isic_loader(tmp_path)


def synthetic_collections():
    # Gate FP, gate FN, correct subtype, incorrect subtype: all routing cases.
    target = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    s1 = np.array([0, 1, 0, 1, 1, 1, 0, 1])
    s2 = np.array([-1, 0, 0, 0, 1, 2, 2, 0])
    ids = tuple(f"synthetic-{i}" for i in range(8))
    flat = PredictionCollection(ids, target, target.copy(), np.eye(4)[target], 0.0)
    shared = SharedISICPredictionCollection(ids, target, (target != 0).astype(int), s1,
        np.eye(2)[s1], np.where(target == 0, -1, target - 1), s2,
        np.full((8, 3), np.nan), 0.0)
    return flat, shared


def test_historical_routing_metrics_and_paired_rows(monkeypatch):
    flat, shared = synthetic_collections()
    monkeypatch.setattr(ev, "EXPECTED_N", 8)
    monkeypatch.setattr(ev, "paired_statistics", lambda *a: {"macro_f1": {}, "accuracy": {}})
    result, rows = ev.evaluate_collections(flat, shared)
    assert [r["shared_predicted_gate"] for r in rows] == [0, 1, 0, 1, 2, 3, 0, 1]
    assert [r["shared_oracle_gate"] for r in rows] == [0, 0, 1, 1, 2, 3, 3, 1]
    assert result["task1"]["sample_count"] == 8
    assert result["task2_malignant_subset"]["sample_count"] == 6
    assert result["routing_counts"]
    assert result["routing_loss_macro_f1"] == result["oracle"]["macro_f1"] - result["hard"]["macro_f1"]
    assert result["delta_hard_minus_flat_macro_f1"] == result["hard"]["macro_f1"] - result["flat"]["macro_f1"]
    assert all("per_class" in result[s] and "confusion_matrix" in result[s] for s in ("flat", "hard", "oracle"))
    assert rows[0]["stage2_probability_0"] == ""


def test_sample_order_mismatch_rejected():
    flat, shared = synthetic_collections()
    shared = SharedISICPredictionCollection(tuple(reversed(shared.sample_ids)), *[
        getattr(shared, name) for name in shared.__dataclass_fields__ if name != "sample_ids"])
    with pytest.raises(ValueError, match="sample order"):
        ev.evaluate_collections(flat, shared)


def test_statistics_sign_seed_replicates_and_mcnemar(monkeypatch):
    table = pd.DataFrame({"difference_macro_f1": [0.1, 0.2, 0.3], "difference_accuracy": [-0.2, 0.0, 0.2]})
    bootstrap = Mock(return_value=table)
    monkeypatch.setattr(ev, "bootstrap_table", bootstrap)
    target = np.array([0, 1, 2, 3])
    flat = np.array([0, 0, 2, 3])
    hard = np.array([0, 1, 0, 3])
    result = ev.paired_statistics(target, flat, hard)
    assert bootstrap.call_args.kwargs == {"replicate_count": 10000, "seed": 42}
    assert result["macro_f1"]["paired_bootstrap_95ci"] == pytest.approx([-0.295, -0.105])
    assert result["mcnemar"] == ev.exact_mcnemar(flat == target, hard == target)


def results_for(seed, lock):
    return {"seed": seed, "sample_count": ev.EXPECTED_N, "protocol": ev.PROTOCOL,
            "checkpoint_provenance": ev._seed_provenance(lock, seed),
            **{s: {m: float(ev.SEEDS.index(seed)) for m in ev.METRIC_NAMES} for s in ("flat", "hard", "oracle")},
            "delta_hard_minus_flat_macro_f1": 0.0, "routing_loss_macro_f1": 0.0}


def test_aggregation_sample_std_and_no_selection():
    result = ev.aggregate_three_seeds({s: results_for(s, {"checkpoints": []}) for s in ev.SEEDS})
    for system in result["systems"].values():
        for values in system.values():
            assert values == {"individual_seed_values": {"42": 0.0, "123": 1.0, "2026": 2.0}, "mean": 1.0, "sample_std": 1.0}
    assert result["protocol"]["ensemble"] is False
    assert result["protocol"]["model_selection_from_test"] is False
    assert result["protocol"]["tuning_from_test"] is False
    with pytest.raises(ValueError, match="exactly"):
        ev.aggregate_three_seeds({42: {}})


@pytest.fixture
def resumable(artifacts, monkeypatch):
    lock = ev.checkpoint_only_preflight(artifacts.root)
    monkeypatch.setattr(ev, "EXPECTED_N", 4)
    monkeypatch.setattr(ev, "ISIC2019HierarchicalDataset", forbid)
    rows = [{"sample_id": f"fake-{i}"} for i in range(4)]
    evaluator = Mock(side_effect=lambda root, seed, lock, **kw: (results_for(seed, lock), rows))
    monkeypatch.setattr(ev, "evaluate_one_seed", evaluator)
    return artifacts.root, lock, evaluator


def test_resume_skips_completed_seed_and_persists(resumable):
    root, lock, evaluator = resumable
    output = root / ev.EVALUATION / "internal_isic"
    output.mkdir()
    first, rows = evaluator(root, 42, lock)
    ev._publish_seed(output, 42, first, rows, lock)
    before = {p.name: p.read_bytes() for p in (output / "seed42").iterdir()}
    evaluator.reset_mock()
    durable = []

    def persist():
        durable.append([s for s in ev.SEEDS if (output / f"seed{s}" / "seed_complete.json").exists()])

    ev.run_isic_evaluation(root, device="cpu", persist_callback=persist)
    assert [c.args[1] for c in evaluator.call_args_list] == [123, 2026]
    assert durable == [[42, 123], [42, 123, 2026], [42, 123, 2026]]
    assert {p.name: p.read_bytes() for p in (output / "seed42").iterdir()} == before
    evaluator.reset_mock()
    with pytest.raises(RuntimeError, match="rerun refused"):
        ev.run_isic_evaluation(root, device="cpu")
    evaluator.assert_not_called()


def test_failed_seed_does_not_publish_completion(resumable):
    root, lock, evaluator = resumable
    evaluator.side_effect = RuntimeError("interrupted")
    callback = Mock()
    with pytest.raises(RuntimeError, match="interrupted"):
        ev.run_isic_evaluation(root, device="cpu", persist_callback=callback)
    output = root / ev.EVALUATION / "internal_isic"
    assert not (output / "seed42").exists()
    assert not (output / "evaluation_complete.json").exists()
    callback.assert_not_called()


def test_artifact_publication_failure_never_exposes_completed_seed(resumable, monkeypatch):
    root, lock, evaluator = resumable
    output = root / ev.EVALUATION / "internal_isic"
    output.mkdir()
    result, rows = evaluator(root, 42, lock)
    original = ev._publish_json

    def fail_marker(path, value):
        if Path(path).name == "seed_complete.json":
            raise OSError("simulated publication failure")
        return original(path, value)

    monkeypatch.setattr(ev, "_publish_json", fail_marker)
    with pytest.raises(OSError, match="publication failure"):
        ev._publish_seed(output, 42, result, rows, lock)
    assert not (output / "seed42").exists()
    monkeypatch.setattr(ev, "_publish_json", original)
    ev.run_isic_evaluation(root, device="cpu")
    assert (output / "seed42" / "seed_complete.json").exists()


def test_preflight_cannot_recreate_lost_lock_after_execution(artifacts):
    (artifacts.root / ev.EVALUATION / "internal_isic").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="before-test lock"):
        ev.checkpoint_only_preflight(artifacts.root)


@pytest.mark.parametrize("name", ["metrics_and_statistics.json", "paired_internal_test_predictions.csv", "seed_complete.json"])
def test_completed_seed_tampering_fails_before_missing_seeds(resumable, name):
    root, lock, evaluator = resumable
    output = root / ev.EVALUATION / "internal_isic"
    output.mkdir()
    result, rows = evaluator(root, 42, lock)
    ev._publish_seed(output, 42, result, rows, lock)
    (output / "seed42" / name).write_text("{}")
    evaluator.reset_mock()
    with pytest.raises(ValueError):
        ev.run_isic_evaluation(root, device="cpu")
    evaluator.assert_not_called()


def test_completion_after_last_seed_does_not_repeat_inference(resumable):
    root, lock, evaluator = resumable
    output = root / ev.EVALUATION / "internal_isic"
    output.mkdir()
    for seed in ev.SEEDS:
        result, rows = evaluator(root, seed, lock)
        ev._publish_seed(output, seed, result, rows, lock)
    evaluator.reset_mock()
    ev.run_isic_evaluation(root, device="cpu")
    evaluator.assert_not_called()
    assert (output / "evaluation_complete.json").is_file()


def test_extension_only_no_training_imports_or_last_checkpoint():
    path = ROOT / "src/evaluation/paul2026_swin_evaluation.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("src.training")
    assert "last_checkpoint.pt" not in source
    assert "build_train_transform" not in source
    assert "results/paper" not in source
    assert ev.EVALUATION == Path("results/extensions/paul2026_swin/evaluation")
    from tests.test_paul2026_swin_protocol import test_paper_configs_and_registry_remain_unchanged
    test_paper_configs_and_registry_remain_unchanged()


def test_modal_evaluation_resource_and_import_contract():
    tree = ast.parse((ROOT / "scripts/modal_paul2026_swin.py").read_text(encoding="utf-8"))
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name in ("run_isic_evaluation_preflight", "run_isic_evaluation_production"):
        node = functions[name]
        kwargs = {k.arg: k.value for k in node.decorator_list[0].keywords}
        assert ast.literal_eval(kwargs["max_containers"]) == 1
        assert ast.literal_eval(kwargs["retries"]) == 0
        assert ast.literal_eval(kwargs["single_use_containers"]) is True
        if name.endswith("preflight"):
            assert "gpu" not in kwargs
        else:
            assert ast.literal_eval(kwargs["gpu"]) == "T4"
        for child in ast.walk(node):
            if isinstance(child, ast.ImportFrom):
                assert not (child.module or "").startswith("src.training")
        assert any(isinstance(k, ast.keyword) and k.arg == "persist_callback" for k in ast.walk(node))


@pytest.fixture
def launcher(monkeypatch):
    app = Mock()
    app.local_entrypoint.return_value = lambda f: f
    functions = {}

    def lookup(app_name, name):
        assert app_name == "paul2026-swin"
        if name not in functions:
            function = Mock()
            function.get_current_stats.return_value = SimpleNamespace(num_running_inputs=0, backlog=0)
            function.spawn.return_value.object_id = "fc-offline"
            functions[name] = function
        return functions[name]

    modal = SimpleNamespace(App=Mock(return_value=app), Function=SimpleNamespace(from_name=Mock(side_effect=lookup)))
    monkeypatch.setitem(sys.modules, "modal", modal)
    namespace = runpy.run_path(str(ROOT / "scripts/launch_paul2026_swin_evaluation.py"))
    return namespace, functions, lookup


@pytest.mark.parametrize("mode", ["preflight", "isic", "hiba-preflight", "hiba"])
def test_launcher_spawn_only(launcher, mode, capsys):
    namespace, functions, _ = launcher
    namespace["main"](mode)
    selected = namespace["FUNCTION_BY_MODE"][mode]
    for name, function in functions.items():
        if name == selected:
            function.spawn.assert_called_once_with()
            function.spawn.return_value.get.assert_not_called()
        else:
            function.spawn.assert_not_called()
        function.remote.assert_not_called()
    assert "function_call_id: fc-offline" in capsys.readouterr().out


@pytest.mark.parametrize("field", ["num_running_inputs", "backlog"])
def test_launcher_busy_guard(launcher, field):
    namespace, functions, lookup = launcher
    busy = lookup("paul2026-swin", "run_shared_hard_production")
    setattr(busy.get_current_stats.return_value, field, 1)
    with pytest.raises(SystemExit, match="running or queued"):
        namespace["main"]("isic")
    for function in functions.values():
        function.spawn.assert_not_called()


@pytest.mark.parametrize("mode", ["", "hiba-invalid", "train", None, [], True])
def test_launcher_invalid_mode_never_looks_up(launcher, mode):
    namespace, functions, _ = launcher
    with pytest.raises(ValueError, match="mode"):
        namespace["main"](mode)
    assert not functions
