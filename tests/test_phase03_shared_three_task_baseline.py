"""Gate 03B tests for the frozen shared three-task baseline."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from torch.nn import functional as F

from src.data.shared_three_task import (
    MISSING_TARGET,
    encode_isic_shared_targets,
    encode_stage3_shared_targets,
)
from src.models.shared_three_task import (
    TASK_CLASS_MAPPINGS,
    build_shared_three_task_efficientnet_b0,
)
from src.training.shared_three_task import (
    MaskedThreeTaskLoss,
    SharedValidationEarlyStopping,
    build_optimizer_and_scheduler,
    shared_validation_score,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs/experiments/phase03_shared_three_task_hierarchical_baseline.yaml"
)


def _criterion() -> MaskedThreeTaskLoss:
    return MaskedThreeTaskLoss(
        torch.tensor([0.35, 0.45, 2.20]),
        torch.tensor([0.06, 0.12, 0.68, 2.25, 1.88]),
    )


def _logits(batch_size: int = 4) -> dict[str, torch.Tensor]:
    return {
        "task1": torch.randn(batch_size, 2, requires_grad=True),
        "task2": torch.randn(batch_size, 3, requires_grad=True),
        "task3": torch.randn(batch_size, 5, requires_grad=True),
    }


def test_model_output_shapes_and_one_encoder_forward() -> None:
    model = build_shared_three_task_efficientnet_b0(pretrained="none")
    calls = 0

    def count_encoder_call(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        nonlocal calls
        calls += 1

    handle = model.encoder.register_forward_hook(count_encoder_call)
    try:
        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 64, 64))
    finally:
        handle.remove()

    assert outputs["task1"].shape == (2, 2)
    assert outputs["task2"].shape == (2, 3)
    assert outputs["task3"].shape == (2, 5)
    assert calls == 1
    for head in (model.task1_head, model.task2_head, model.task3_head):
        assert isinstance(head[0], torch.nn.Dropout)
        assert head[0].p == pytest.approx(0.2)
    assert model.task1_head[1].out_features == 2
    assert model.task2_head[1].out_features == 3
    assert model.task3_head[1].out_features == 5


@pytest.mark.parametrize(
    ("flat_target", "expected_targets", "expected_mask"),
    [
        (0, (0, MISSING_TARGET, MISSING_TARGET), (True, False, False)),
        (1, (1, 0, MISSING_TARGET), (True, True, False)),
        (2, (1, 1, MISSING_TARGET), (True, True, False)),
        (3, (1, 2, MISSING_TARGET), (True, True, False)),
    ],
)
def test_isic_masks_and_targets(
    flat_target: int,
    expected_targets: tuple[int, int, int],
    expected_mask: tuple[bool, bool, bool],
) -> None:
    targets, mask = encode_isic_shared_targets(flat_target)
    assert tuple(targets.tolist()) == expected_targets
    assert tuple(mask.tolist()) == expected_mask


@pytest.mark.parametrize("target", range(5))
def test_stage3_mask_never_invents_upstream_labels(target: int) -> None:
    targets, mask = encode_stage3_shared_targets(target)
    assert tuple(targets.tolist()) == (MISSING_TARGET, MISSING_TARGET, target)
    assert tuple(mask.tolist()) == (False, False, True)


def test_missing_labels_are_not_valid_class_zero() -> None:
    isic_targets, _ = encode_isic_shared_targets(0)
    stage3_targets, _ = encode_stage3_shared_targets(0)
    assert isic_targets[1:].tolist() == [MISSING_TARGET, MISSING_TARGET]
    assert stage3_targets[:2].tolist() == [MISSING_TARGET, MISSING_TARGET]
    assert MISSING_TARGET not in {0, 1, 2, 3, 4}


@pytest.mark.parametrize(
    ("task_index", "task_name", "class_count"),
    [(0, "task1", 2), (1, "task2", 3), (2, "task3", 5)],
)
def test_masked_task_loss_uses_only_valid_samples(
    task_index: int,
    task_name: str,
    class_count: int,
) -> None:
    criterion = _criterion()
    logits = _logits()
    targets = torch.full((4, 3), MISSING_TARGET, dtype=torch.long)
    task_mask = torch.zeros((4, 3), dtype=torch.bool)
    active = torch.tensor([True, False, True, False])
    task_mask[:, task_index] = active
    targets[active, task_index] = torch.tensor([0, class_count - 1])

    result = criterion(logits, targets, task_mask)

    selected_logits = logits[task_name][active]
    selected_targets = targets[active, task_index]
    if task_name == "task1":
        expected = F.cross_entropy(selected_logits, selected_targets)
    elif task_name == "task2":
        expected = criterion.task2_criterion(selected_logits, selected_targets)
    else:
        expected = criterion.task3_criterion(selected_logits, selected_targets)
    assert result.total_loss.detach().item() == pytest.approx(
        expected.detach().item()
    )
    assert result.active_counts[task_name] == 2
    task_loss = result.task_losses[task_name]
    assert task_loss is not None
    assert task_loss.detach().item() == pytest.approx(expected.detach().item())


def test_zero_active_task_is_skipped_and_not_in_denominator() -> None:
    criterion = _criterion()
    logits = _logits()
    targets = torch.tensor(
        [
            [0, MISSING_TARGET, MISSING_TARGET],
            [1, 2, MISSING_TARGET],
            [0, MISSING_TARGET, MISSING_TARGET],
            [1, 1, MISSING_TARGET],
        ],
        dtype=torch.long,
    )
    task_mask = torch.tensor(
        [
            [True, False, False],
            [True, True, False],
            [True, False, False],
            [True, True, False],
        ]
    )
    result = criterion(logits, targets, task_mask)

    task1 = criterion.task1_criterion(logits["task1"], targets[:, 0])
    active2 = task_mask[:, 1]
    task2 = criterion.task2_criterion(
        logits["task2"][active2], targets[active2, 1]
    )
    assert result.task_losses["task3"] is None
    assert result.active_counts["task3"] == 0
    expected = (task1 + task2) / 2
    assert result.total_loss.detach().item() == pytest.approx(
        expected.detach().item()
    )


def test_inactive_heads_do_not_receive_invalid_targets_or_gradients() -> None:
    criterion = _criterion()
    logits = _logits(batch_size=2)
    targets = torch.tensor(
        [[0, MISSING_TARGET, MISSING_TARGET], [1, MISSING_TARGET, MISSING_TARGET]]
    )
    task_mask = torch.tensor(
        [[True, False, False], [True, False, False]]
    )
    result = criterion(logits, targets, task_mask)
    result.total_loss.backward()

    assert logits["task1"].grad is not None
    assert logits["task2"].grad is None
    assert logits["task3"].grad is None


def test_active_task_gradient_reaches_shared_encoder() -> None:
    model = build_shared_three_task_efficientnet_b0(pretrained="none")
    criterion = _criterion()
    outputs = model(torch.randn(2, 3, 64, 64))
    targets = torch.tensor(
        [[0, MISSING_TARGET, MISSING_TARGET], [1, MISSING_TARGET, MISSING_TARGET]]
    )
    task_mask = torch.tensor(
        [[True, False, False], [True, False, False]]
    )
    criterion(outputs, targets, task_mask).total_loss.backward()

    first_parameter = next(model.encoder.parameters())
    assert first_parameter.grad is not None
    assert torch.isfinite(first_parameter.grad).all()
    assert model.task1_head[1].weight.grad is not None
    assert model.task2_head[1].weight.grad is None
    assert model.task3_head[1].weight.grad is None


def test_shared_validation_score_is_arithmetic_mean() -> None:
    assert shared_validation_score(0.6, 0.3, 0.9) == pytest.approx(0.6)


def test_early_stopping_maximizes_shared_score_with_patience_seven() -> None:
    control = SharedValidationEarlyStopping(patience=7)
    assert control.update(0.5, 1) == (True, False)
    for epoch in range(2, 8):
        assert control.update(0.5, epoch) == (False, False)
    assert control.update(0.49, 8) == (False, True)
    assert control.best_epoch == 1


def test_frozen_config_and_seed_are_deterministic() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["experiment"]["seed"] == 42
    assert config["loader"]["batch_size"] == 64
    assert config["loader"]["train_shuffle"] is True
    assert config["data"]["training_sources"] == {
        "isic2019_train": 17124,
        "isic_stage3_train": 594,
        "combined_natural_pool": 17718,
    }
    assert config["data"]["internal_test"] == {
        "allowed": False,
        "construct_loader": False,
        "influence_selection": False,
    }
    assert config["model"]["historical_checkpoint_initialization"] is False
    assert config["task_losses"]["lambda_task1"] == 1.0
    assert config["task_losses"]["lambda_task2"] == 1.0
    assert config["task_losses"]["lambda_task3"] == 1.0
    assert config["training"]["epochs"] == 30
    assert config["training"]["early_stopping_patience"] == 7
    assert config["validation"]["checkpoint_selection"] == (
        "highest_shared_validation_score"
    )
    assert TASK_CLASS_MAPPINGS["task1"] == {
        "non_malignant": 0,
        "malignant": 1,
    }

    torch.manual_seed(42)
    first = torch.rand(4)
    torch.manual_seed(42)
    second = torch.rand(4)
    assert torch.equal(first, second)


def test_optimizer_and_scheduler_use_frozen_horizon() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    model = torch.nn.Linear(2, 2)
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["lr"] == pytest.approx(3e-4)
    assert optimizer.defaults["weight_decay"] == pytest.approx(1e-4)
    assert scheduler.T_max == 30
    assert scheduler.eta_min == pytest.approx(1e-6)
