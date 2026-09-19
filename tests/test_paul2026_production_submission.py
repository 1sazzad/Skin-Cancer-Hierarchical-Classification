"""Exercise submission with a fake Modal module: no SDK or network required."""

import ast
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def production(monkeypatch, tmp_path):
    from src.training import paul2026_swin as paul

    app = Mock()
    app.function.return_value = lambda function: function
    app.local_entrypoint.return_value = lambda function: function
    modal = SimpleNamespace(App=Mock(return_value=app), Image=Mock(), Volume=Mock())
    monkeypatch.setitem(sys.modules, "modal", modal)
    namespace = runpy.run_path(str(ROOT / "scripts/modal_paul2026_swin.py"))
    run = namespace["run_flat_production"]
    run.__globals__.update(
        PROJECT_ROOT=str(ROOT), PRODUCTION_OUTPUT_ROOT=str(tmp_path),
        ISIC_DATA_PATH=str(tmp_path), EMB_DATA_PATH=str(tmp_path),
    )
    # Stop at dispatch, before training or any CUDA inspection.
    runner = Mock(side_effect=RuntimeError("offline dispatch reached"))
    monkeypatch.setattr(paul, "run_paul_experiment", runner)
    return run, namespace, runner, tmp_path


@pytest.mark.parametrize("seed", [42, 123, 2026])
def test_production_resolves_declared_config_and_identity(production, seed, capsys):
    from src.training import paul2026_swin as paul

    run, namespace, runner, output = production
    expected = f"configs/extensions/paul2026_swin/flat_seed{seed}.yaml"
    assert namespace["FLAT_CONFIG_BY_SEED"][seed] == expected
    with pytest.raises(RuntimeError, match="offline dispatch reached"):
        run(seed)
    runner.assert_called_once_with(
        ROOT / expected, project_root=ROOT, output_root=output, device="cuda",
        epoch_limit=None, max_train_batches=None, max_validation_batches=None,
        resume=True, persist_callback=namespace["results_volume"].commit,
    )
    config = paul.load_paul_config(runner.call_args.args[0])
    assert config["experiment"]["run_name"] == f"paul2026_flat_swin_t_seed{seed}"
    assert config["experiment"]["seed"] == seed
    assert config["data"]["internal_test"] == paul.INTERNAL_TEST_POLICY
    assert capsys.readouterr().out.splitlines() == [
        "mode: production", "resume_capable: true", "system: flat",
        "backbone: swin_t", f"seed: {seed}", f"config: {expected}",
        "gpu: T4", "max_epochs: 30", "early_stopping_patience: 7",
        "bounded_batches: false",
    ]


@pytest.mark.parametrize("seed", [0, 43, True, False, "42", "123", None, 42.0, [], {}])
def test_invalid_production_seed_never_dispatches(production, seed):
    run, _, runner, _ = production
    with pytest.raises(ValueError, match="production seed"):
        run(seed)
    runner.assert_not_called()


@pytest.mark.parametrize("seed", [42, 123, 2026])
def test_completed_run_refuses_training_without_modifying_files(production, seed):
    run, namespace, runner, output = production
    name = f"paul2026_flat_swin_t_seed{seed}"
    directory = output / name
    directory.mkdir()
    summary = directory / "run_summary.json"
    summary.write_text(json.dumps({"run_name": name, "reportable_as_full_result": True}))
    checkpoint = directory / "last_checkpoint.pt"
    checkpoint.write_bytes(b"preserve checkpoint")
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(RuntimeError, match=f"production run already completed: {name}"):
        run(seed)
    runner.assert_not_called()
    namespace["results_volume"].commit.assert_not_called()
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


@pytest.mark.parametrize("summary", [None, {"run_name": "paul2026_flat_swin_t_seed123", "reportable_as_full_result": False}])
def test_partial_run_remains_resumable(production, summary):
    run, _, runner, output = production
    directory = output / "paul2026_flat_swin_t_seed123"
    directory.mkdir()
    checkpoint = directory / "last_checkpoint.pt"
    checkpoint.write_bytes(b"partial checkpoint")
    if summary is not None:
        (directory / "run_summary.json").write_text(json.dumps(summary))
    with pytest.raises(RuntimeError, match="offline dispatch reached"):
        run(123)
    assert runner.call_args.kwargs["resume"] is True
    assert checkpoint.read_bytes() == b"partial checkpoint"


@pytest.mark.parametrize("seed", [0, 43, True, False, "42", "123", None, 42.0, [], {}])
def test_invalid_launcher_seed_rejected_before_lookup(launcher, seed):
    main, modal, function, _ = launcher
    with pytest.raises(ValueError, match="production seed"):
        main(seed)
    modal.Function.from_name.assert_not_called()
    function.spawn.assert_not_called()


@pytest.mark.parametrize("seed", [42, 2026])
def test_launcher_accepts_other_declared_seeds(launcher, seed):
    main, _, function, call = launcher
    main(seed)
    function.spawn.assert_called_once_with(seed)
    function.remote.assert_not_called()
    call.get.assert_not_called()


@pytest.fixture
def launcher(monkeypatch):
    call = Mock(object_id="fc-offline-test")
    function = Mock()
    function.spawn.return_value = call
    function.get_current_stats.return_value = SimpleNamespace(
        num_running_inputs=0, backlog=0,
    )
    app = Mock()
    app.local_entrypoint.return_value = lambda main: main
    modal = SimpleNamespace(App=Mock(return_value=app), Function=Mock())
    modal.Function.from_name.return_value = function
    monkeypatch.setitem(sys.modules, "modal", modal)
    namespace = runpy.run_path(
        str(ROOT / "scripts/launch_paul2026_swin_production.py")
    )
    modal.Function.from_name.assert_not_called()
    return namespace["main"], modal, function, call


def test_idle_submission_returns_call_id_without_waiting(launcher, capsys):
    main, modal, function, call = launcher
    main(123)
    modal.Function.from_name.assert_called_once_with(
        "paul2026-swin", "run_flat_production",
    )
    function.get_current_stats.assert_called_once_with()
    function.spawn.assert_called_once_with(123)
    assert [c[0] for c in function.mock_calls] == ["get_current_stats", "spawn"]
    function.remote.assert_not_called()
    call.get.assert_not_called()
    assert capsys.readouterr().out.splitlines() == [
        "launch_status: submitted", "app: paul2026-swin",
        "function: run_flat_production", "seed: 123", "function_call_id: fc-offline-test",
        "local_process_required: false",
    ]
    modal.App.return_value.function.assert_not_called()


@pytest.mark.parametrize("running,backlog", [(1, 0), (0, 1), (2, 3)])
def test_existing_input_refuses_submission(launcher, running, backlog, capsys):
    main, modal, function, call = launcher
    function.get_current_stats.return_value = SimpleNamespace(
        num_running_inputs=running, backlog=backlog,
    )
    with pytest.raises(SystemExit, match="production launch refused") as error:
        main(123)
    assert error.value.code != 0
    modal.Function.from_name.assert_called_once_with(
        "paul2026-swin", "run_flat_production",
    )
    function.spawn.assert_not_called()
    function.remote.assert_not_called()
    call.get.assert_not_called()
    assert "submitted" not in capsys.readouterr().out


def test_stats_failure_does_not_spawn(launcher):
    main, _, function, call = launcher
    function.get_current_stats.side_effect = RuntimeError("stats unavailable")
    with pytest.raises(RuntimeError, match="stats unavailable"):
        main(123)
    function.spawn.assert_not_called()
    call.get.assert_not_called()


def test_spawn_failure_is_not_retried_or_reported_as_submitted(launcher, capsys):
    main, _, function, _ = launcher
    function.spawn.side_effect = RuntimeError("submission failed")
    with pytest.raises(RuntimeError, match="submission failed"):
        main(123)
    function.spawn.assert_called_once_with(123)
    assert "submitted" not in capsys.readouterr().out


def test_old_entrypoint_fails_with_dedicated_launcher_instructions():
    tree = ast.parse((ROOT / "scripts/modal_paul2026_swin.py").read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    main.decorator_list = []
    namespace = {"run_flat_production": Mock()}
    exec(compile(ast.Module(body=[main], type_ignores=[]), "entrypoint", "exec"), namespace)
    with pytest.raises(SystemExit, match="launch_paul2026_swin_production.py"):
        namespace["main"]()
    namespace["run_flat_production"].remote.assert_not_called()


def test_submission_has_no_training_or_internal_test_imports():
    tree = ast.parse((ROOT / "scripts/launch_paul2026_swin_production.py").read_text())
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert len(imports) == 1
    assert isinstance(imports[0], ast.Import)
    assert [alias.name for alias in imports[0].names] == ["modal"]
