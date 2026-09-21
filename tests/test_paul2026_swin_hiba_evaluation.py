"""Synthetic regression contracts; never run a real HIBA evaluation."""
import ast
import csv
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from src.evaluation import paul2026_swin_hiba_evaluation as ev
from tests.test_paul2026_swin_evaluation import (
    artifacts, forbid, payload_for, summary_for, synthetic_collections,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def locked_inputs(artifacts, monkeypatch):
    root = artifacts.root
    config = root / ev.CONFIG
    config.parent.mkdir(parents=True)
    config.write_bytes((ROOT / ev.CONFIG).read_bytes())
    for name in ev.SOURCE_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    cfg = ev._config(root)
    manifest = root / cfg["hiba"]["manifest_path"]
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes((ROOT / cfg["hiba"]["manifest_path"]).read_bytes())
    original_hash = ev.sha256_file
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    image_hashes = {str((root / r["image_path"]).resolve()): r["image_sha256"] for r in rows}
    monkeypatch.setattr(ev, "sha256_file", lambda p: image_hashes[str(p)] if str(p) in image_hashes else original_hash(p))
    return root


def test_preflight_cpu_no_inference_or_performance(locked_inputs, monkeypatch):
    for name in ("HIBADataset", "collect_single_task_predictions", "collect_shared_isic_predictions", "patient_cluster_ci"):
        monkeypatch.setattr(ev.historical, name, forbid)
    monkeypatch.setattr(ev, "compute_classification_metrics", forbid)
    monkeypatch.setattr(ev.torch.cuda, "is_available", forbid)
    original = ev.internal._read_json

    def metadata_only(path):
        assert Path(path).name == "run_summary.json"
        return original(path)

    monkeypatch.setattr(ev.internal, "_read_json", metadata_only)
    lock = ev.hiba_preflight(locked_inputs)
    assert len(lock["checkpoints"]) == 6
    assert {r["seed"] for r in lock["checkpoints"]} == {42, 123, 2026}
    assert lock["inference_performed"] is lock["performance_calculated"] is False
    assert len(lock["cohort"]) == 1232
    assert len({r["patient_id"] for r in lock["cohort"]}) == 568
    assert lock["source_sha256"] and lock["manifest_sha256"]
    assert (locked_inputs / ev.OUTPUT / ev.LOCK_NAME).is_file()


@pytest.mark.parametrize("system", ["flat", "shared_hard"])
def test_summary_seed_policy(system):
    spec = next(s for s in ev.internal.CHECKPOINTS if s.system == system)
    summary = summary_for(spec)
    del summary["seed"]
    if system == "flat":
        ev.internal.validate_checkpoint_and_summary(payload_for(spec), summary, spec)
        payload = payload_for(spec)
        del payload["config"]["experiment"]["seed"]
        with pytest.raises(ValueError):
            ev.internal.validate_checkpoint_and_summary(payload, summary, spec)
    else:
        with pytest.raises(ValueError):
            ev.internal.validate_checkpoint_and_summary(payload_for(spec), summary, spec)
    summary["seed"] = 999
    with pytest.raises(ValueError):
        ev.internal.validate_checkpoint_and_summary(payload_for(spec), summary, spec)


def test_missing_lock_precedes_inference(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "evaluate_one_seed", forbid)
    with pytest.raises(FileNotFoundError, match="preflight lock"):
        ev.run_hiba_evaluation(tmp_path, device="cpu")


@pytest.mark.parametrize("change", ["lock", "checkpoint", "manifest", "source", "config", "image"])
def test_locked_identity_drift(locked_inputs, monkeypatch, change):
    ev.hiba_preflight(locked_inputs)
    monkeypatch.setattr(ev, "evaluate_one_seed", forbid)
    if change == "lock":
        path = locked_inputs / ev.OUTPUT / ev.LOCK_NAME
        value = json.loads(path.read_text())
        value["protocol"]["ensemble"] = True
        path.write_text(json.dumps(value))
    elif change == "image":
        original = ev.sha256_file
        monkeypatch.setattr(ev, "sha256_file", lambda p: "changed" if str(p).endswith(".jpg") else original(p))
    else:
        path = {"checkpoint": ev.internal.CHECKPOINTS[0].path(locked_inputs),
            "manifest": locked_inputs / ev._config(locked_inputs)["hiba"]["manifest_path"],
            "source": locked_inputs / ev.SOURCE_FILES[0], "config": locked_inputs / ev.CONFIG}[change]
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        ev.run_hiba_evaluation(locked_inputs, device="cpu")


def test_historical_loader_reused(monkeypatch):
    cfg = ev._config(ROOT)
    dataset = Mock()
    loader = Mock()
    monkeypatch.setattr(ev.historical, "HIBADataset", dataset)
    monkeypatch.setattr(ev, "DataLoader", loader)
    ev.build_hiba_loader(ROOT)
    dataset.assert_called_once_with(ROOT / cfg["hiba"]["manifest_path"], ROOT, verify_hashes=True)
    assert loader.call_args.kwargs["shuffle"] is False
    assert loader.call_args.kwargs["worker_init_fn"] is ev.seed_worker
    assert loader.call_args.kwargs["generator"].initial_seed() == 42


def test_historical_patient_bootstrap_both_pairs(monkeypatch):
    flat, shared = synthetic_collections()
    cohort = [{"isic_id": sid, "patient_id": f"p{i // 2}", "target_index": str(flat.targets[i])}
              for i, sid in enumerate(flat.sample_ids)]
    bootstrap = Mock(return_value=[-0.2, 0.1])
    monkeypatch.setattr(ev, "EXPECTED_N", 8)
    monkeypatch.setattr(ev.historical, "patient_cluster_ci", bootstrap)
    monkeypatch.setattr(ev.internal, "bootstrap_table", forbid)
    monkeypatch.setattr(ev.internal, "exact_mcnemar", forbid)
    result, rows = ev.evaluate_collections(flat, shared, cohort)
    assert bootstrap.call_count == 2
    for call in bootstrap.call_args_list:
        assert call.args[3] == [r["patient_id"] for r in cohort]
        assert call.args[4:] == (5000, 42)
    np.testing.assert_array_equal(bootstrap.call_args_list[0].args[2], bootstrap.call_args_list[1].args[1])
    assert result["task1"]["sample_count"] == 8
    assert result["routing_counts"]
    assert all("per_class" in result[s] for s in ("flat", "hard", "oracle"))
    assert [r["patient_id"] for r in rows] == [r["patient_id"] for r in cohort]


def result_for(seed, lock):
    return {**ev._identity(seed, lock), "sample_count": ev.EXPECTED_N,
        **{s: {m: ev.SEEDS.index(seed) / 4 for m in ev.internal.METRIC_NAMES} for s in ("flat", "hard", "oracle")},
        "delta_hard_minus_flat_macro_f1": 0.0, "routing_loss_macro_f1": 0.0}


def test_aggregation_all_seeds_mean_sample_sd():
    results = {s: result_for(s, {"checkpoints": []}) for s in ev.SEEDS}
    result = ev.aggregate_three_seeds(results)
    for system in result["systems"].values():
        for metric in system.values():
            assert metric["mean"] == 0.25
            assert metric["sample_std"] == 0.25
            assert metric["individual_seed_values"] == {"42": 0.0, "123": 0.25, "2026": 0.5}
    assert result["protocol"]["ensemble"] is False
    assert result["protocol"]["model_selection_from_test"] is False
    for seed in ev.SEEDS:
        with pytest.raises(ValueError):
            ev.aggregate_three_seeds({s: r for s, r in results.items() if s != seed})


@pytest.fixture
def resumable(locked_inputs, monkeypatch):
    lock = ev.hiba_preflight(locked_inputs)
    rows = [{"sample_id": r["isic_id"], "patient_id": r["patient_id"]} for r in lock["cohort"]]
    evaluator = Mock(side_effect=lambda root, seed, lock, **kw: (result_for(seed, lock), rows))
    monkeypatch.setattr(ev, "evaluate_one_seed", evaluator)
    return locked_inputs, lock, rows, evaluator


def test_resume_and_final_completion(resumable):
    root, lock, rows, evaluator = resumable
    output = root / ev.OUTPUT
    ev._publish_seed(output, 42, result_for(42, lock), rows, lock)
    ev.run_hiba_evaluation(root, device="cpu")
    assert [c.args[1] for c in evaluator.call_args_list] == [123, 2026]
    evaluator.reset_mock()
    ev.run_hiba_evaluation(root, device="cpu")
    evaluator.assert_not_called()


def test_partial_seed_is_never_complete(resumable):
    root, lock, rows, evaluator = resumable
    directory = root / ev.OUTPUT / "seed42"
    directory.mkdir()
    (directory / "metrics_and_statistics.json").write_text("{}")
    with pytest.raises(FileNotFoundError):
        ev.run_hiba_evaluation(root, device="cpu")
    evaluator.assert_not_called()


def test_failed_publication_can_resume(resumable, monkeypatch):
    root, lock, rows, evaluator = resumable
    original = ev.publish_json

    def fail(path, value):
        if path.name == "seed_complete.json":
            raise OSError("interrupted")
        return original(path, value)

    monkeypatch.setattr(ev, "publish_json", fail)
    with pytest.raises(OSError):
        ev.run_hiba_evaluation(root, device="cpu")
    assert not (root / ev.OUTPUT / "seed42").exists()
    monkeypatch.setattr(ev, "publish_json", original)
    ev.run_hiba_evaluation(root, device="cpu")
    assert (root / ev.OUTPUT / "evaluation_complete.json").is_file()


@pytest.mark.parametrize("name", [*ev.ARTIFACTS, "seed_complete.json"])
def test_completed_artifact_tampering(resumable, name):
    root, lock, rows, evaluator = resumable
    ev._publish_seed(root / ev.OUTPUT, 42, result_for(42, lock), rows, lock)
    (root / ev.OUTPUT / "seed42" / name).write_text("{}")
    with pytest.raises(ValueError):
        ev.run_hiba_evaluation(root, device="cpu")
    evaluator.assert_not_called()


def test_modal_cpu_gpu_resource_contract():
    tree = ast.parse((ROOT / "scripts/modal_paul2026_swin.py").read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for suffix in ("preflight", "production"):
        node = functions[f"run_hiba_evaluation_{suffix}"]
        kwargs = {k.arg: k.value for k in node.decorator_list[0].keywords}
        assert ast.literal_eval(kwargs["max_containers"]) == 1
        assert ast.literal_eval(kwargs["retries"]) == 0
        assert ast.literal_eval(kwargs["single_use_containers"]) is True
        if suffix == "preflight":
            assert "gpu" not in kwargs
        else:
            assert ast.literal_eval(kwargs["gpu"]) == "T4"
