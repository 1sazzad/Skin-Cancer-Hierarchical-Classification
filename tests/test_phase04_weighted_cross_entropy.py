from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from src.training.baseline_experiment import (
    _build_criterion,
    load_experiment_config,
)


PHASE03_CONFIG = Path(
    "configs/experiments/"
    "phase03_stage02_isic2019_efficientnet_b0_cross_entropy.yaml"
)
PHASE04_CONFIG = Path(
    "configs/experiments/"
    "phase04_stage02_isic2019_efficientnet_b0_weighted_cross_entropy.yaml"
)


def _write_config(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_phase03_plain_cross_entropy_remains_unweighted() -> None:
    config = load_experiment_config(PHASE03_CONFIG)
    criterion = _build_criterion(config, "cpu")

    assert isinstance(criterion, nn.CrossEntropyLoss)
    assert criterion.weight is None
    assert config["training"]["class_weights"] is None


def test_phase04_weighted_config_is_runnable_and_traceable() -> None:
    config = load_experiment_config(PHASE04_CONFIG)

    assert config["data"]["task"] == "stage_2"
    assert config["training"]["loss"] == "weighted_cross_entropy"
    assert config["training"]["weighted_sampler"] is False
    assert config["training"]["focal_loss"] is False
    assert config["training"]["selection_metric"] == "macro_f1"

    source = config["training"]["class_weight_source"]
    assert source["partition"] == "train"
    assert source["total_samples"] == 5931
    assert source["class_counts"] == {
        "melanoma": 3164,
        "bcc": 2327,
        "scc": 440,
    }


def test_weighted_criterion_follows_class_to_index_order() -> None:
    config = load_experiment_config(PHASE04_CONFIG)
    criterion = _build_criterion(config, torch.device("cpu"))

    assert isinstance(criterion, nn.CrossEntropyLoss)
    assert criterion.weight is not None
    assert criterion.weight.device.type == "cpu"
    assert criterion.weight.tolist() == pytest.approx(
        [
            0.6248419721871049,
            0.8495917490330898,
            4.493181818181818,
        ]
    )


def test_weighted_config_rejects_missing_class_weight(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load(PHASE04_CONFIG.read_text(encoding="utf-8"))
    del payload["training"]["class_weights"]["scc"]

    path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match="exactly match"):
        load_experiment_config(path)


@pytest.mark.parametrize("invalid_weight", [0.0, -1.0])
def test_weighted_config_rejects_nonpositive_weight(
    tmp_path: Path,
    invalid_weight: float,
) -> None:
    payload = yaml.safe_load(PHASE04_CONFIG.read_text(encoding="utf-8"))
    payload["training"]["class_weights"]["scc"] = invalid_weight

    path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match="finite and positive"):
        load_experiment_config(path)


def test_weighted_cross_entropy_is_restricted_to_stage2(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load(PHASE04_CONFIG.read_text(encoding="utf-8"))
    payload["data"]["task"] = "stage_1"
    payload["data"]["class_to_index"] = {
        "non_malignant": 0,
        "malignant": 1,
    }
    payload["model"]["number_of_classes"] = 2
    payload["training"]["class_weights"] = {
        "non_malignant": 1.0,
        "malignant": 1.0,
    }

    path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match="restricted to Stage 2"):
        load_experiment_config(path)
