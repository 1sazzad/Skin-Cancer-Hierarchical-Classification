from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

import src.training.baseline_experiment as baseline_experiment
from src.training.baseline_experiment import (
    load_experiment_config,
    run_baseline_experiment,
)


class TinyDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self) -> None:
        self.images = torch.tensor(
            [
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[1.0, 1.0], [1.0, 1.0]]],
                [[[0.2, 0.2], [0.2, 0.2]]],
                [[[0.8, 0.8], [0.8, 0.8]]],
            ],
            dtype=torch.float32,
        )
        self.targets = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "image": self.images[index],
            "target": self.targets[index],
        }


def _config() -> dict[str, object]:
    return {
        "experiment": {
            "status": "ready_for_training",
            "research_stage": "stage01",
            "run_name": "tiny_stage01_seed42",
            "dataset": "isic2019",
            "model": "efficientnet_b0",
            "variant": "cross_entropy",
            "seed": 42,
        },
        "data": {
            "split_manifest": "data/manifests/frozen.csv",
            "task": "stage_1",
            "class_to_index": {"non_malignant": 0, "malignant": 1},
            "verify_image_paths": False,
        },
        "loader": {
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": 2,
            "drop_last_train": False,
        },
        "model": {
            "architecture": "efficientnet_b0",
            "pretrained_weights": "imagenet",
            "number_of_classes": 2,
            "dropout_probability": 0.2,
        },
        "training": {
            "epochs": 2,
            "loss": "cross_entropy",
            "weighted_sampler": False,
            "class_weights": None,
            "focal_loss": False,
            "amp": True,
            "optimizer": {
                "name": "adamw",
                "learning_rate": 0.001,
                "weight_decay": 0.0,
            },
            "scheduler": {
                "name": "cosine_annealing",
                "minimum_learning_rate": 0.000001,
            },
            "selection_metric": "macro_f1",
            "early_stopping_patience": 2,
            "full_training_allowed": True,
        },
    }


def _write_config(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_runnable_config_preserves_clean_baseline_policy(tmp_path: Path) -> None:
    path = _write_config(tmp_path, _config())
    loaded = load_experiment_config(path)

    assert loaded["experiment"]["status"] == "ready_for_training"
    assert loaded["training"]["loss"] == "cross_entropy"
    assert loaded["training"]["weighted_sampler"] is False
    assert loaded["training"]["class_weights"] is None
    assert loaded["training"]["focal_loss"] is False
    assert loaded["training"]["selection_metric"] == "macro_f1"


def test_runnable_config_rejects_weighted_sampler(tmp_path: Path) -> None:
    payload = _config()
    payload["training"]["weighted_sampler"] = True
    path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match="weighted_sampler"):
        load_experiment_config(path)



@pytest.mark.parametrize(
    ("config_name", "task", "number_of_classes"),
    [
        (
            "phase03_stage01_isic2019_efficientnet_b0_cross_entropy.yaml",
            "stage_1",
            2,
        ),
        (
            "phase03_stage02_isic2019_efficientnet_b0_cross_entropy.yaml",
            "stage_2",
            3,
        ),
    ],
)
def test_repository_phase03_configs_are_runnable(
    config_name: str,
    task: str,
    number_of_classes: int,
) -> None:
    path = Path("configs/experiments") / config_name
    loaded = load_experiment_config(path)

    assert loaded["data"]["task"] == task
    assert loaded["model"]["number_of_classes"] == number_of_classes
    assert loaded["training"]["selection_metric"] == "macro_f1"
    assert loaded["training"]["full_training_allowed"] is True


def test_sanity_run_writes_non_reportable_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path, _config())
    loader = DataLoader(TinyDataset(), batch_size=2, shuffle=False)

    monkeypatch.setattr(
        baseline_experiment,
        "build_stage_dataloaders",
        lambda *args, **kwargs: {
            "train": loader,
            "validation": loader,
            "internal_test": loader,
        },
    )
    monkeypatch.setattr(
        baseline_experiment,
        "build_efficientnet_b0",
        lambda *args, **kwargs: nn.Sequential(nn.Flatten(), nn.Linear(4, 2)),
    )

    outcome = run_baseline_experiment(
        config_path,
        project_root=tmp_path,
        output_root=tmp_path / "runs",
        device="cpu",
        max_train_batches=1,
        max_validation_batches=1,
        epoch_limit=1,
    )

    summary = json.loads(
        (outcome.run_directory / "run_summary.json").read_text(encoding="utf-8")
    )
    assert summary["sanity_run"] is True
    assert summary["reportable_as_full_result"] is False
    assert summary["best_epoch"] == 1
    assert outcome.best_checkpoint_path.is_file()
    assert outcome.last_checkpoint_path.is_file()
    assert (outcome.run_directory / "history.csv").is_file()
    assert (outcome.run_directory / "resolved_config.yaml").is_file()
