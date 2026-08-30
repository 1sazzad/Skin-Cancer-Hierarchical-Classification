from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

import src.training.baseline_experiment as baseline_experiment
from src.models.classification_backbone import (
    SUPPORTED_CLASSIFICATION_ARCHITECTURES,
    build_classification_model,
)
from src.training.baseline_experiment import (
    load_experiment_config,
    run_baseline_experiment,
)
from src.training.engine import EpochResult

NEW_ARCHITECTURES = (
    "densenet169",
    "resnet50",
    "mobilenet_v3_large",
    "efficientnet_b2",
    "efficientnet_b3",
)
ALL_ARCHITECTURES = (
    "efficientnet_b0",
    "densenet121",
    *NEW_ARCHITECTURES,
)
CONFIG_PATHS = {
    architecture: Path("configs/experiments")
    / f"phase02_flat_four_class_isic2019_{architecture}_cross_entropy.yaml"
    for architecture in NEW_ARCHITECTURES
}


@pytest.mark.parametrize("architecture", ALL_ARCHITECTURES)
def test_all_phase02_backbones_construct_offline_with_four_class_head(
    architecture: str,
) -> None:
    model = build_classification_model(
        architecture,
        4,
        pretrained="none",
        dropout_probability=0.2,
    )
    model.eval()

    with torch.inference_mode():
        output = model(torch.rand(1, 3, 64, 64))

    assert output.shape == (1, 4)
    linear_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
    dropout_layers = [module for module in model.modules() if isinstance(module, nn.Dropout)]
    assert linear_layers[-1].out_features == 4
    assert any(layer.p == pytest.approx(0.2) for layer in dropout_layers)


def test_supported_architecture_identifiers_and_invalid_failure_are_explicit() -> None:
    assert SUPPORTED_CLASSIFICATION_ARCHITECTURES == ALL_ARCHITECTURES
    with pytest.raises(ValueError, match="Unsupported classification architecture"):
        build_classification_model("not_a_backbone", 4, pretrained="none")


@pytest.mark.parametrize("architecture", NEW_ARCHITECTURES)
def test_phase02_new_backbone_configs_are_runnable_and_quarantined(
    architecture: str,
) -> None:
    config = load_experiment_config(CONFIG_PATHS[architecture])

    assert config["model"]["architecture"] == architecture
    assert config["training"]["evaluate_internal_test_after_training"] is False
    assert config["data"]["class_to_index"] == {
        "non_malignant": 0,
        "melanoma": 1,
        "bcc": 2,
        "scc": 3,
    }


def test_phase02_configs_differ_only_in_identity_and_architecture() -> None:
    normalized = []
    for architecture, path in CONFIG_PATHS.items():
        payload = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
        payload["experiment"]["run_name"] = "<run_name>"
        payload["experiment"]["model"] = "<architecture>"
        payload["model"]["architecture"] = "<architecture>"
        normalized.append(payload)

    assert all(payload == normalized[0] for payload in normalized[1:])


@pytest.mark.parametrize(
    ("section", "field", "changed_value"),
    [
        ("data", "split_manifest", "data/manifests/other.csv"),
        ("loader", "batch_size", 32),
        ("model", "dropout_probability", 0.3),
        ("preprocessing", "input_size", [256, 256]),
        ("training", "epochs", 31),
        ("training", "evaluate_internal_test_after_training", True),
    ],
)
def test_phase02_frozen_fields_cannot_drift(
    tmp_path: Path,
    section: str,
    field: str,
    changed_value: object,
) -> None:
    payload = yaml.safe_load(
        CONFIG_PATHS["resnet50"].read_text(encoding="utf-8-sig")
    )
    payload[section][field] = changed_value
    path = tmp_path / "drifted.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Phase 02 frozen protocol mismatch"):
        load_experiment_config(path)


def test_phase02_training_runner_never_iterates_internal_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = yaml.safe_load(
        CONFIG_PATHS["resnet50"].read_text(encoding="utf-8-sig")
    )
    payload["training"]["epochs"] = 1
    payload["training"]["early_stopping_patience"] = 1
    # Preserve validation of the repository config, then use the in-memory
    # short run only to exercise partition routing without doing optimization.
    load_experiment_config(CONFIG_PATHS["resnet50"])
    config_path = tmp_path / "phase02_short.yaml"
    payload["experiment"]["research_stage"] = "phase02_test_harness"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    train_loader = object()
    validation_loader = object()

    class ForbiddenInternalTest:
        def __iter__(self):
            raise AssertionError("internal test must remain quarantined")

    internal_test_loader = ForbiddenInternalTest()
    monkeypatch.setattr(
        baseline_experiment,
        "build_stage_dataloaders",
        lambda *args, **kwargs: {
            "train": train_loader,
            "validation": validation_loader,
            "internal_test": internal_test_loader,
        },
    )
    monkeypatch.setattr(
        baseline_experiment,
        "build_classification_model",
        lambda *args, **kwargs: nn.Linear(1, 4),
    )

    class NoOpScheduler:
        def step(self) -> None:
            pass

        def state_dict(self) -> dict[str, object]:
            return {}

    monkeypatch.setattr(
        baseline_experiment,
        "_build_scheduler",
        lambda *args, **kwargs: NoOpScheduler(),
    )
    visited: list[object] = []

    def fake_epoch(*args, **kwargs) -> EpochResult:
        visited.append(args[1])
        return EpochResult(
            mean_loss=1.0,
            sample_count=4,
            targets=torch.tensor([0, 1, 2, 3]),
            predictions=torch.tensor([0, 1, 2, 3]),
            probabilities=torch.eye(4),
        )

    monkeypatch.setattr(baseline_experiment, "run_classification_epoch", fake_epoch)
    run_baseline_experiment(
        config_path,
        project_root=tmp_path,
        output_root=tmp_path / "runs",
        device="cpu",
    )

    assert visited == [train_loader, validation_loader]
    assert internal_test_loader not in visited
