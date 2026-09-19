"""Exercise submission with a fake Modal module: no SDK or network required."""

import ast
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[1]


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
    main()
    modal.Function.from_name.assert_called_once_with(
        "paul2026-swin", "run_flat_production",
    )
    function.get_current_stats.assert_called_once_with()
    function.spawn.assert_called_once_with()
    assert [c[0] for c in function.mock_calls] == ["get_current_stats", "spawn"]
    function.remote.assert_not_called()
    call.get.assert_not_called()
    assert capsys.readouterr().out.splitlines() == [
        "launch_status: submitted", "app: paul2026-swin",
        "function: run_flat_production", "function_call_id: fc-offline-test",
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
        main()
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
        main()
    function.spawn.assert_not_called()
    call.get.assert_not_called()


def test_spawn_failure_is_not_retried_or_reported_as_submitted(launcher, capsys):
    main, _, function, _ = launcher
    function.spawn.side_effect = RuntimeError("submission failed")
    with pytest.raises(RuntimeError, match="submission failed"):
        main()
    function.spawn.assert_called_once_with()
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
