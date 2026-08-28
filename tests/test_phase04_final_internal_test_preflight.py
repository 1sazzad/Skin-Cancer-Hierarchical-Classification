from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from scripts.run_phase04_final_internal_test import (
    REQUIRED_CHECKPOINTS,
    REQUIRED_MANIFEST_KEYS,
    _load_config,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_phase04_final_internal_test.py"
VALIDATION_CONFIG = ROOT / "configs/evaluation/phase04_controlled_comparative_validation.yaml"
INTERNAL_TEST_CONFIG = (
    ROOT / "configs/evaluation/phase04_controlled_comparative_internal_test.yaml"
)
PAIRED_COLUMNS = [
    "sample_id",
    "true_label",
    "shared_predicted_gate",
    "shared_oracle_gate",
    "flat_prediction",
    "shared_correct",
    "flat_correct",
    "stage1_target",
    "stage1_prediction",
    "stage2_target",
    "stage2_prediction",
]


def _yaml(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_yaml(path: Path, value: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def test_final_config_differs_from_validation_only_by_authorization_fields() -> None:
    validation = _yaml(VALIDATION_CONFIG)
    final = _yaml(INTERNAL_TEST_CONFIG)
    assert validation["execution_split"] == "validation"
    assert validation["internal_test_execution_allowed"] is False
    assert final["execution_split"] == "internal_test"
    assert final["internal_test_execution_allowed"] is True
    validation["execution_split"] = "internal_test"
    validation["internal_test_execution_allowed"] = True
    assert final == validation


def test_final_config_preserves_frozen_keys_and_paired_schema() -> None:
    config = _load_config(INTERNAL_TEST_CONFIG)
    assert set(config["checkpoints"]) == REQUIRED_CHECKPOINTS
    assert set(config["isic"]["manifest_paths"]) == REQUIRED_MANIFEST_KEYS
    assert config["statistics_export"]["paired_columns"] == PAIRED_COLUMNS


@pytest.mark.parametrize(
    ("split", "allowed", "message"),
    [
        ("validation", True, "execution_split=internal_test"),
        ("internal_test", False, "explicit internal-test authorization"),
    ],
)
def test_final_runner_refuses_unauthorized_configs(
    tmp_path: Path, split: str, allowed: bool, message: str
) -> None:
    config = deepcopy(_yaml(INTERNAL_TEST_CONFIG))
    config["execution_split"] = split
    config["internal_test_execution_allowed"] = allowed
    path = tmp_path / "unauthorized.yaml"
    _write_yaml(path, config)
    with pytest.raises(ValueError, match=message):
        _load_config(path)


def test_runner_constructs_only_four_internal_test_datasets() -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    dataset_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"ISIC2019HierarchicalDataset", "EMBStage03Dataset"}
    ]
    assert len(dataset_calls) == 4
    split_literals = [
        arg.value
        for call in dataset_calls
        for arg in call.args
        if isinstance(arg, ast.Constant)
        and isinstance(arg.value, str)
        and arg.value in {"train", "training", "validation", "internal_test"}
    ]
    assert split_literals == ["internal_test"] * 4


def test_runner_has_no_training_or_validation_loader_construction() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_validation_loader" not in called_names
    assert "_train_loader" not in called_names
    assert "train" not in called_names
    assert '"validation"' not in source
    assert "'validation'" not in source


def test_internal_test_execution_flag_is_only_in_success_payload() -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "payload" for target in node.targets)
    ]
    assert len(assignments) == 1
    payload = assignments[0].value
    assert isinstance(payload, ast.Dict)
    values = {
        key.value: value
        for key, value in zip(payload.keys, payload.values)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    flag = values["internal_test_executed"]
    assert isinstance(flag, ast.Constant) and flag.value is True
