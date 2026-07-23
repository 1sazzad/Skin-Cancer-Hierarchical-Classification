from __future__ import annotations

from pathlib import Path

import pytest
import yaml


@pytest.mark.parametrize(
    ("config_name", "task", "number_of_classes"),
    [
        ("stage01_isic2019_efficientnet_b0_cross_entropy.yaml", "stage_1", 2),
        ("stage02_isic2019_efficientnet_b0_cross_entropy.yaml", "stage_2", 3),
    ],
)
def test_baseline_config_remains_clean_and_not_trained(
    config_name: str,
    task: str,
    number_of_classes: int,
) -> None:
    config_path = Path("configs/experiments") / config_name
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))

    assert config["experiment"]["status"] == "prepared_not_trained"
    assert config["data"]["task"] == task
    assert config["model"]["number_of_classes"] == number_of_classes
    assert config["training"]["loss"] == "cross_entropy"
    assert config["training"]["weighted_sampler"] is False
    assert config["training"]["focal_loss"] is False
    assert config["training"]["full_training_allowed"] is False
