"""Lightweight tests for the Gate 03E executable launcher."""

from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.training.shared_three_task import write_training_history

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs/experiments/phase03_shared_three_task_hierarchical_baseline.yaml"
)
LAUNCHER_PATH = ROOT / "scripts/train_phase03_shared_three_task.py"


def _load_launcher():
    specification = importlib.util.spec_from_file_location(
        "train_phase03_shared_three_task",
        LAUNCHER_PATH,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_launcher_accepts_only_frozen_config() -> None:
    launcher = _load_launcher()
    launcher._validate_frozen_config(_config())

    drifted = deepcopy(_config())
    drifted["training"]["optimizer"]["learning_rate"] = 0.001
    with pytest.raises(ValueError, match="Frozen Phase 03 protocol mismatch"):
        launcher._validate_frozen_config(drifted)


def test_launcher_rejects_internal_test_enablement() -> None:
    launcher = _load_launcher()
    drifted = deepcopy(_config())
    drifted["data"]["internal_test"]["construct_loader"] = True
    with pytest.raises(ValueError, match="internal test must remain prohibited"):
        launcher._validate_frozen_config(drifted)


def test_launcher_refuses_to_overwrite_nonempty_run(tmp_path: Path) -> None:
    launcher = _load_launcher()
    run_directory = tmp_path / "seed_42"
    run_directory.mkdir()
    (run_directory / "evidence.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        launcher._prepare_run_directory(run_directory)


def test_training_history_supports_required_csv_name(tmp_path: Path) -> None:
    record = {
        "epoch": 1,
        "train_total_loss": 1.0,
        "train_task1_loss": 1.0,
        "train_task2_loss": 1.0,
        "train_task3_loss": 1.0,
        "val_task1_macro_f1": 0.1,
        "val_task2_macro_f1": 0.2,
        "val_task3_macro_f1": 0.3,
        "shared_validation_score": 0.2,
        "learning_rate": 0.0003,
    }
    write_training_history(
        tmp_path,
        [record],
        {"completion_status": "running"},
        csv_filename="training_history.csv",
    )
    assert (tmp_path / "training_history.csv").is_file()
    assert not (tmp_path / "history.csv").exists()
    assert (tmp_path / "run_summary.json").is_file()
