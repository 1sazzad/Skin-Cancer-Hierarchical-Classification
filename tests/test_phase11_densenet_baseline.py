from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

import src.evaluation.internal_test_evaluator as internal_test_evaluator
from src.evaluation.internal_test_evaluator import evaluate_frozen_internal_test
from src.models.classification_backbone import (
    SUPPORTED_CLASSIFICATION_ARCHITECTURES,
    build_classification_model,
)
from src.models.densenet_baseline import build_densenet121
from src.training.baseline_experiment import load_experiment_config


EFFICIENTNET_CONFIG = Path(
    "configs/experiments/"
    "phase06_flat_four_class_isic2019_efficientnet_b0_cross_entropy.yaml"
)
DENSENET_CONFIG = Path(
    "configs/experiments/"
    "phase11_flat_four_class_isic2019_densenet121_cross_entropy.yaml"
)


def _load_yaml(path: Path) -> dict[str, object]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    assert isinstance(loaded, dict)
    return loaded


def test_phase11_config_changes_only_approved_backbone_fields() -> None:
    efficientnet = _load_yaml(EFFICIENTNET_CONFIG)
    densenet = _load_yaml(DENSENET_CONFIG)

    expected = deepcopy(efficientnet)
    expected["experiment"]["research_stage"] = "phase11_final_densenet_baseline"
    expected["experiment"]["run_name"] = (
        "phase11_flat_four_class_isic2019_densenet121_cross_entropy_seed42"
    )
    expected["experiment"]["model"] = "densenet121"
    expected["model"]["architecture"] = "densenet121"

    assert densenet == expected


def test_phase11_config_is_runnable_under_locked_policy() -> None:
    loaded = load_experiment_config(DENSENET_CONFIG)

    assert loaded["experiment"]["seed"] == 42
    assert loaded["data"]["task"] == "flat_four_class"
    assert loaded["model"]["architecture"] == "densenet121"
    assert loaded["model"]["number_of_classes"] == 4
    assert loaded["training"]["loss"] == "cross_entropy"
    assert loaded["training"]["selection_metric"] == "macro_f1"
    assert loaded["training"]["full_training_allowed"] is True


def test_supported_architectures_are_explicitly_locked() -> None:
    assert SUPPORTED_CLASSIFICATION_ARCHITECTURES == (
        "efficientnet_b0",
        "densenet121",
    )


def test_densenet121_output_shape() -> None:
    model = build_densenet121(4, pretrained="none")
    model.eval()

    with torch.inference_mode():
        output = model(torch.rand(1, 3, 64, 64))

    assert output.shape == (1, 4)


def test_generic_factory_builds_densenet121() -> None:
    model = build_classification_model(
        "densenet121",
        4,
        pretrained="none",
        dropout_probability=0.2,
    )
    model.eval()

    with torch.inference_mode():
        output = model(torch.rand(1, 3, 64, 64))

    assert output.shape == (1, 4)


def test_generic_factory_rejects_unknown_architecture() -> None:
    with pytest.raises(ValueError, match="Unsupported classification architecture"):
        build_classification_model(
            "unknown_model",
            4,
            pretrained="none",
        )


def test_densenet_is_restricted_to_flat_four_class(tmp_path: Path) -> None:
    payload = _load_yaml(DENSENET_CONFIG)
    payload["data"]["task"] = "stage_1"

    config_path = tmp_path / "invalid_densenet_stage1.yaml"
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="only.*flat_four_class"):
        load_experiment_config(config_path)


class TinyInternalTestDataset(Dataset[dict[str, object]]):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, object]:
        value = float(index) / 3.0
        return {
            "image": torch.full((1, 2, 2), value, dtype=torch.float32),
            "target": torch.tensor(index, dtype=torch.long),
            "image_id": f"image_{index}",
            "image_path": f"data/image_{index}.jpg",
            "split_group_id": f"group_{index}",
            "file_sha256": f"sha256_{index}",
        }


def test_internal_evaluator_accepts_frozen_densenet_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class_names = ["non_malignant", "melanoma", "bcc", "scc"]
    tiny_model = nn.Sequential(nn.Flatten(), nn.Linear(4, 4))

    config = {
        "experiment": {
            "run_name": (
                "phase11_flat_four_class_isic2019_"
                "densenet121_cross_entropy_seed42"
            ),
            "seed": 42,
        },
        "data": {
            "split_manifest": "data/manifests/frozen.csv",
            "task": "flat_four_class",
            "class_to_index": {
                "non_malignant": 0,
                "melanoma": 1,
                "bcc": 2,
                "scc": 3,
            },
            "verify_image_paths": False,
        },
        "loader": {
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": 2,
        },
        "model": {
            "architecture": "densenet121",
            "number_of_classes": 4,
            "dropout_probability": 0.2,
        },
        "runtime": {
            "sanity_run": False,
        },
    }

    checkpoint_directory = tmp_path / "full_run"
    checkpoint_directory.mkdir()
    checkpoint_path = checkpoint_directory / "best_checkpoint.pt"

    torch.save(
        {
            "epoch": 1,
            "model_state_dict": tiny_model.state_dict(),
            "validation_metrics": {"macro_f1": 0.25},
            "config": config,
            "class_names": class_names,
        },
        checkpoint_path,
    )

    (checkpoint_directory / "run_summary.json").write_text(
        json.dumps(
            {
                "reportable_as_full_result": True,
                "best_epoch": 1,
            }
        ),
        encoding="utf-8",
    )

    loader = DataLoader(
        TinyInternalTestDataset(),
        batch_size=2,
        shuffle=False,
    )
    captured: dict[str, object] = {}

    def fake_builder(
        architecture: str,
        number_of_classes: int,
        *,
        pretrained: str,
        dropout_probability: float,
    ) -> nn.Module:
        captured["architecture"] = architecture
        captured["number_of_classes"] = number_of_classes
        captured["pretrained"] = pretrained
        captured["dropout_probability"] = dropout_probability
        return nn.Sequential(nn.Flatten(), nn.Linear(4, 4))

    monkeypatch.setattr(
        internal_test_evaluator,
        "build_stage_dataloaders",
        lambda *args, **kwargs: {"internal_test": loader},
    )
    monkeypatch.setattr(
        internal_test_evaluator,
        "build_classification_model",
        fake_builder,
    )

    outcome = evaluate_frozen_internal_test(
        checkpoint_path,
        project_root=tmp_path,
        output_directory=tmp_path / "evaluation",
        device="cpu",
        batch_size=2,
        num_workers=0,
    )

    assert captured == {
        "architecture": "densenet121",
        "number_of_classes": 4,
        "pretrained": "none",
        "dropout_probability": 0.2,
    }
    assert outcome.task == "flat_four_class"
    assert outcome.checkpoint_epoch == 1
    assert outcome.metrics_path.is_file()
    assert outcome.predictions_path.is_file()
    assert outcome.summary_path.is_file()
