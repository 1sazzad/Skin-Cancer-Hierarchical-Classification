"""Gate 06B coverage for architecture-selectable shared three-task models."""

from __future__ import annotations

import pytest
import torch

from src.models.shared_three_task import (
    SUPPORTED_SHARED_ARCHITECTURES,
    build_shared_three_task_efficientnet_b0,
    build_shared_three_task_model,
)

EXPECTED_ARCHITECTURES = (
    "efficientnet_b0",
    "densenet121",
    "densenet169",
    "resnet50",
    "mobilenet_v3_large",
    "efficientnet_b2",
    "efficientnet_b3",
)


@pytest.mark.parametrize("architecture", EXPECTED_ARCHITECTURES)
def test_all_shared_backbones_construct_offline_and_preserve_head_shapes(
    architecture: str,
) -> None:
    model = build_shared_three_task_model(
        architecture,
        pretrained="none",
        dropout_probability=0.2,
    )
    model.eval()

    calls = 0

    def count_encoder_call(_module, _inputs, _output) -> None:
        nonlocal calls
        calls += 1

    handle = model.encoder.register_forward_hook(count_encoder_call)
    try:
        with torch.inference_mode():
            outputs = model(torch.rand(1, 3, 64, 64))
    finally:
        handle.remove()

    assert outputs["task1"].shape == (1, 2)
    assert outputs["task2"].shape == (1, 3)
    assert outputs["task3"].shape == (1, 5)
    assert calls == 1
    assert model.task1_head[0].p == pytest.approx(0.2)
    assert model.task2_head[0].p == pytest.approx(0.2)
    assert model.task3_head[0].p == pytest.approx(0.2)
    assert model.task1_head[1].out_features == 2
    assert model.task2_head[1].out_features == 3
    assert model.task3_head[1].out_features == 5


def test_supported_shared_architecture_registry_is_frozen() -> None:
    assert SUPPORTED_SHARED_ARCHITECTURES == EXPECTED_ARCHITECTURES


def test_invalid_shared_architecture_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="Unsupported shared architecture"):
        build_shared_three_task_model("not_a_backbone", pretrained="none")


def test_legacy_efficientnet_b0_builder_remains_compatible() -> None:
    model = build_shared_three_task_efficientnet_b0(pretrained="none")
    model.eval()
    with torch.inference_mode():
        outputs = model(torch.rand(1, 3, 64, 64))
    assert outputs["task1"].shape == (1, 2)
    assert outputs["task2"].shape == (1, 3)
    assert outputs["task3"].shape == (1, 5)
