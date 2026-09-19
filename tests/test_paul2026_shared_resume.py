"""Offline Shared resume contracts; epochs and model construction are mocked."""

import ast
from copy import deepcopy
import json
from pathlib import Path
import random
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import yaml

from scripts import train_phase03_shared_three_task as phase03
from src.training import paul2026_swin as paul
from src.training import shared_resume as resume
from src.training.shared_resume_loader import ResumableTrainLoader

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def harness(monkeypatch, tmp_path):
    config = paul.load_paul_config(ROOT / "configs/extensions/paul2026_swin/shared_hard_seed42.yaml")
    # Tiny offline control horizon; no real data, model forward, or GPU work.
    config["training"]["epochs"] = 5
    config["training"]["early_stopping_patience"] = 2
    config["loader"]["num_workers"] = 0
    state = {"fail": None, "calls": 0, "scores": iter([.6, .5, .4])}
    monkeypatch.setattr(phase03, "_git_commit", lambda _: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    scaler = Mock()
    scaler.state_dict.return_value = {"scale": 123}
    monkeypatch.setattr(torch.amp, "GradScaler", lambda *a, **k: scaler)
    components = []

    def build(*args, **kwargs):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5)
        loader = torch.utils.data.DataLoader([0, 1, 2], shuffle=True, generator=torch.Generator().manual_seed(42))
        result = (SimpleNamespace(train=loader, validation_task1=[], validation_task2=[], validation_task3=[]), model, None, optimizer, scheduler)
        components.append(result)
        return result

    def epoch(model, loader, criterion, optimizer, device, **kwargs):
        state["calls"] += 1
        if state["calls"] == state["fail"]:
            raise RuntimeError("interrupted epoch")
        list(loader)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        return SimpleNamespace(train_total_loss=1., train_task1_loss=1., train_task2_loss=1., train_task3_loss=1.)

    def validate(model, loaders, device):
        assert set(loaders) == {"task1", "task2", "task3"}
        score = next(state["scores"])
        return {"shared_validation_score": score, **{f"val_task{i}_macro_f1": score for i in (1, 2, 3)}}

    monkeypatch.setattr(phase03, "_build_components", build)
    monkeypatch.setattr(phase03, "run_shared_training_epoch", epoch)
    monkeypatch.setattr(phase03, "validate_shared_model", validate)
    callback = Mock()

    def run(directory=tmp_path, resumable=True):
        directory.mkdir(parents=True, exist_ok=True)
        return phase03._run_training(config, config_path=directory / "resolved_config.yaml", project_root=ROOT,
                                     run_directory=directory, device=torch.device("cpu"),
                                     resume=resumable, persist_callback=callback if resumable else None)

    return SimpleNamespace(config=config, state=state, run=run, directory=tmp_path,
                           callback=callback, scaler=scaler, components=components)


def test_bootstrap_restart_and_failed_epoch_is_not_persisted(harness):
    h = harness
    (h.directory / "logs").mkdir()
    (h.directory / "logs/train_console.log").write_text("interrupted setup")
    (h.directory / "resolved_config.yaml").write_text(yaml.safe_dump(h.config))
    h.state["fail"] = 2
    with pytest.raises(RuntimeError, match="interrupted epoch"):
        h.run()
    saved = resume.load_shared_resume(h.directory / "last_checkpoint.pt", h.config)
    assert saved["completed_epoch"] == 1
    assert [row["epoch"] for row in saved["history"]] == [1]
    assert h.callback.call_count == 1
    assert saved["scaler_state_dict"] == {"scale": 123}
    assert saved["optimizer_state_dict"]["state"]
    assert saved["scheduler_state_dict"]["last_epoch"] == 1


def test_resume_restores_history_patience_best_and_final_guard(harness):
    h = harness
    h.state["fail"] = 3
    with pytest.raises(RuntimeError, match="interrupted epoch"):
        h.run()
    saved = resume.load_shared_resume(h.directory / "last_checkpoint.pt", h.config)
    assert saved["epochs_without_improvement"] == 1
    h.state["fail"] = None
    h.run()
    final = resume.load_shared_resume(h.directory / "last_checkpoint.pt", h.config)
    assert [row["epoch"] for row in final["history"]] == [1, 2, 3]
    assert final["history"][:2] == saved["history"]
    assert final["best_shared_validation_score"] == .6
    assert final["best_epoch"] == 1
    assert final["epochs_without_improvement"] == 2
    assert final["scheduler_state_dict"]["last_epoch"] == 3
    assert final["cumulative_training_seconds"] >= saved["cumulative_training_seconds"]
    h.scaler.load_state_dict.assert_called_once_with({"scale": 123})
    assert h.callback.call_count == 4  # Three epochs and the final summary.
    before = (h.directory / "last_checkpoint.pt").read_bytes()
    with pytest.raises(RuntimeError, match="already completed"):
        h.run()
    assert (h.directory / "last_checkpoint.pt").read_bytes() == before


def test_restore_rng_optimizer_scheduler_scaler_and_generator(harness):
    h = harness
    h.state["fail"] = 2
    with pytest.raises(RuntimeError):
        h.run()
    saved = resume.load_shared_resume(h.directory / "last_checkpoint.pt", h.config)
    calls = Mock()
    control = phase03.SharedValidationEarlyStopping(2)
    generator = torch.Generator()
    resume.restore_shared_resume(saved, model=calls.model, optimizer=calls.optimizer,
                                 scheduler=calls.scheduler, scaler=calls.scaler,
                                 control=control, train_generator=generator)
    assert [call[0] for call in calls.mock_calls] == [
        "model.load_state_dict", "optimizer.load_state_dict", "scheduler.load_state_dict", "scaler.load_state_dict",
    ]
    assert random.getstate() == saved["python_rng_state"]
    assert np.array_equal(np.random.get_state()[1], saved["numpy_rng_state"][1])
    assert torch.equal(torch.get_rng_state(), saved["torch_cpu_rng_state"])
    assert torch.equal(generator.get_state(), saved["train_generator_state"])
    assert (control.best_score, control.best_epoch, control.epochs_without_improvement) == (.6, 1, 0)


@pytest.mark.parametrize("field,value", [("seed", 123), ("completed_epoch", 9), ("history", []), ("best_epoch", 2), ("epochs_without_improvement", 6)])
def test_inconsistent_checkpoint_rejected(harness, field, value):
    h = harness
    h.state["fail"] = 2
    with pytest.raises(RuntimeError):
        h.run()
    path = h.directory / "last_checkpoint.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    saved[field] = value
    torch.save(saved, path)
    with pytest.raises(ValueError):
        h.run()


def test_orphan_artifacts_fail_loudly(harness):
    (harness.directory / "history.json").write_text("[]")
    with pytest.raises(ValueError, match="lacks last_checkpoint"):
        harness.run()
    harness.callback.assert_not_called()


def test_cpu_checkpoint_loading_contract():
    tree = ast.parse((ROOT / "src/training/shared_resume.py").read_text())
    loads = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "load"]
    assert len(loads) == 1
    assert {kw.arg: ast.literal_eval(kw.value) for kw in loads[0].keywords} == {
        "map_location": "cpu", "weights_only": False,
    }


def test_historical_default_has_no_resume_artifacts(harness):
    harness.run(resumable=False)
    assert not (harness.directory / "last_checkpoint.pt").exists()
    assert (harness.directory / "best_checkpoint.pt").exists()
    summary = json.loads((harness.directory / "run_summary.json").read_text())
    assert "reportable_as_full_result" not in summary
    assert summary["epochs_completed"] == 3
    assert summary["internal_test_evaluated"] is False
    harness.callback.assert_not_called()


def test_final_epoch_checkpoint_finalizes_without_another_epoch(harness):
    h = harness
    h.run()
    (h.directory / "run_summary.json").write_text('{"completion_status": "running"}')
    calls = h.state["calls"]
    h.run()
    assert h.state["calls"] == calls
    assert json.loads((h.directory / "run_summary.json").read_text())["reportable_as_full_result"] is True


def test_uninterrupted_and_resumed_states_match(harness):
    h = harness
    continuous = h.directory / "continuous"
    h.run(continuous)
    expected = resume.load_shared_resume(continuous / "last_checkpoint.pt", h.config)
    h.state.update(calls=0, fail=3, scores=iter([.6, .5, .4]))
    interrupted = h.directory / "interrupted"
    with pytest.raises(RuntimeError):
        h.run(interrupted)
    h.state["fail"] = None
    h.run(interrupted)
    actual = resume.load_shared_resume(interrupted / "last_checkpoint.pt", h.config)
    for key in ("history", "scheduler_state_dict", "scaler_state_dict", "best_epoch", "epochs_without_improvement"):
        assert actual[key] == expected[key]
    for key, value in expected["model_state_dict"].items():
        assert torch.equal(actual["model_state_dict"][key], value)
    for parameter, values in expected["optimizer_state_dict"]["state"].items():
        for key, value in values.items():
            assert torch.equal(actual["optimizer_state_dict"]["state"][parameter][key], value)
    assert torch.equal(actual["train_generator_state"], expected["train_generator_state"])


class RandomWorkerDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, index):
        return index, random.random(), float(np.random.random()), float(torch.rand(()))


def test_persistent_worker_rng_and_shuffle_survive_restart():
    def loader():
        return torch.utils.data.DataLoader(
            RandomWorkerDataset(), batch_size=2, shuffle=True,
            num_workers=2, persistent_workers=True,
            generator=torch.Generator().manual_seed(42),
        )

    continuous = ResumableTrainLoader(loader())
    list(continuous)
    generator_state = continuous.generator.get_state()
    worker_states = deepcopy(continuous.worker_states)
    expected = list(continuous)
    fresh = loader()
    fresh.generator.set_state(generator_state)
    restarted = ResumableTrainLoader(fresh, worker_states, resumed=True)
    actual = list(restarted)
    for expected_batch, actual_batch in zip(expected, actual, strict=True):
        for expected_column, actual_column in zip(expected_batch, actual_batch, strict=True):
            assert torch.equal(expected_column, actual_column)
    assert torch.equal(continuous.generator.get_state(), restarted.generator.get_state())


def test_shared_adapter_passes_resume_and_callback(monkeypatch, tmp_path):
    path = ROOT / "configs/extensions/paul2026_swin/shared_hard_seed123.yaml"
    runner = Mock()
    monkeypatch.setattr(paul.phase03, "_run_training", runner)
    callback = Mock()
    paul.run_paul_experiment(path, output_root=tmp_path, resume=True, persist_callback=callback)
    assert runner.call_args.kwargs["resume"] is True
    assert runner.call_args.kwargs["persist_callback"] is callback
    assert runner.call_args.kwargs["run_directory"].name == "paul2026_shared_hard_swin_t_seed123"


def test_atomic_save_failure_retains_previous_checkpoint(harness, monkeypatch):
    h = harness
    h.state["fail"] = 2
    with pytest.raises(RuntimeError):
        h.run()
    path = h.directory / "last_checkpoint.pt"
    before = path.read_bytes()
    original = torch.save

    def fail_save(payload, destination, *args, **kwargs):
        if Path(destination).name == ".last_checkpoint.pt.tmp":
            Path(destination).write_bytes(b"incomplete")
            raise OSError("interrupted save")
        return original(payload, destination, *args, **kwargs)

    monkeypatch.setattr(torch, "save", fail_save)
    h.state["fail"] = None
    h.callback.reset_mock()
    with pytest.raises(OSError, match="interrupted save"):
        h.run()
    assert path.read_bytes() == before
    h.callback.assert_not_called()


def test_failed_first_epoch_has_no_durable_epoch(harness):
    harness.state["fail"] = 1
    with pytest.raises(RuntimeError, match="interrupted epoch"):
        harness.run()
    assert not (harness.directory / "last_checkpoint.pt").exists()
    harness.callback.assert_not_called()


def test_cuda_rng_restore_uses_cpu_bytes_without_gpu(harness, monkeypatch):
    h = harness
    h.state["fail"] = 2
    with pytest.raises(RuntimeError):
        h.run()
    state = resume.load_shared_resume(h.directory / "last_checkpoint.pt", h.config)
    state["torch_cuda_rng_states"] = [torch.get_rng_state()]
    setter = Mock()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", setter)
    calls = Mock()
    resume.restore_shared_resume(
        state, model=calls.model, optimizer=calls.optimizer, scheduler=calls.scheduler,
        scaler=calls.scaler, control=phase03.SharedValidationEarlyStopping(2),
        train_generator=torch.Generator(),
    )
    setter.assert_called_once()
    restored = setter.call_args.args[0][0]
    assert restored.device.type == "cpu"
    assert restored.dtype == torch.uint8


def test_shared_modal_production_resource_contract():
    tree = ast.parse((ROOT / "scripts/modal_paul2026_swin.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "run_shared_hard_production")
    names = {"GPU": "T4", "MAX_CONTAINERS": 1, "RETRIES": 0,
             "PRODUCTION_TIMEOUT_SECONDS": 50400, "SINGLE_USE_CONTAINERS": True}
    keywords = {kw.arg: kw.value for kw in function.decorator_list[0].keywords}
    for key, value in {"gpu": "T4", "max_containers": 1, "retries": 0,
                       "timeout": 50400, "single_use_containers": True}.items():
        assert names[keywords[key].id] == value
    volumes = keywords["volumes"]
    assert {ast.literal_eval(key) for key in volumes.keys} == {
        "/root/project/data/raw", "/root/project/results/extensions/paul2026_swin",
    }
