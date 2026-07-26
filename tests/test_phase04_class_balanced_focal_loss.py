from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import yaml
from torch.nn import functional as F

from src.training.baseline_experiment import (
    _build_criterion,
    load_experiment_config,
)
from src.training.losses import ClassBalancedFocalLoss


CONFIG_PATH = Path(
    "configs/experiments/"
    "phase04_stage02_isic2019_efficientnet_b0_class_balanced_focal_loss.yaml"
)


def _write_config(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_class_balanced_focal_config_is_runnable() -> None:
    config = load_experiment_config(CONFIG_PATH)

    assert config["data"]["task"] == "stage_2"
    assert config["training"]["loss"] == "class_balanced_focal_loss"
    assert config["training"]["focal_loss"] is True
    assert config["training"]["focal_gamma"] == 2.0
    assert config["training"]["weighted_sampler"] is False
    assert config["training"]["selection_metric"] == "macro_f1"


def test_effective_number_weights_match_training_counts() -> None:
    config = load_experiment_config(CONFIG_PATH)
    training = config["training"]
    source = training["class_weight_source"]

    beta = float(source["beta"])
    counts = source["class_counts"]

    raw_weights = {
        name: (1.0 - beta) / (1.0 - beta ** int(count))
        for name, count in counts.items()
    }
    scale = len(raw_weights) / sum(raw_weights.values())

    expected = {
        name: value * scale
        for name, value in raw_weights.items()
    }

    for class_name, expected_weight in expected.items():
        assert float(training["class_weights"][class_name]) == pytest.approx(
            expected_weight,
            rel=1e-12,
            abs=1e-12,
        )


def test_criterion_uses_class_order_and_gamma() -> None:
    config = load_experiment_config(CONFIG_PATH)
    criterion = _build_criterion(config, "cpu")

    assert isinstance(criterion, ClassBalancedFocalLoss)
    assert criterion.gamma == 2.0
    assert criterion.class_weights.tolist() == pytest.approx(
        [
            0.3485376280807543,
            0.4553489231324597,
            2.196113448786786,
        ]
    )


def test_focal_loss_matches_manual_formula() -> None:
    weights = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32)
    criterion = ClassBalancedFocalLoss(weights, gamma=2.0)

    logits = torch.tensor(
        [
            [2.0, 0.5, -1.0],
            [0.2, 0.4, 1.5],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 2], dtype=torch.long)

    probabilities = torch.softmax(logits, dim=1)
    target_probabilities = probabilities[
        torch.arange(targets.numel()),
        targets,
    ]
    cross_entropy = F.cross_entropy(
        logits,
        targets,
        reduction="none",
    )

    expected = (
        weights[targets]
        * (1.0 - target_probabilities).pow(2.0)
        * cross_entropy
    ).mean()

    actual = criterion(logits, targets)

    assert actual == pytest.approx(expected.item(), rel=1e-6)


def test_gamma_zero_reduces_to_weighted_sample_mean() -> None:
    weights = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32)
    criterion = ClassBalancedFocalLoss(weights, gamma=0.0)

    logits = torch.tensor(
        [
            [1.0, 0.0, -1.0],
            [0.0, 0.5, 1.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 2], dtype=torch.long)

    expected = (
        weights[targets]
        * F.cross_entropy(logits, targets, reduction="none")
    ).mean()

    assert criterion(logits, targets) == pytest.approx(
        expected.item(),
        rel=1e-6,
    )


def test_config_rejects_negative_gamma(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["training"]["focal_gamma"] = -1.0

    path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match="focal_gamma"):
        load_experiment_config(path)


def test_config_rejects_non_training_weight_source(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["training"]["class_weight_source"]["partition"] = "validation"

    path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match="training partition"):
        load_experiment_config(path)


def test_config_rejects_formula_weight_mismatch(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["training"]["class_weights"]["scc"] = 3.0

    path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match="effective-number formula"):
        load_experiment_config(path)
