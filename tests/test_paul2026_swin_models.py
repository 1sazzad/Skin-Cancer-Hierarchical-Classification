"""Offline CPU contract checks for the PAUL-2026 transformer extension."""

import pytest
import torch
from torch import nn

from src.models import transformer_extension as extension
from src.models.classification_backbone import SUPPORTED_CLASSIFICATION_ARCHITECTURES
from src.models.shared_three_task import SUPPORTED_SHARED_ARCHITECTURES


@pytest.fixture(autouse=True)
def offline_swin(monkeypatch):
    """Fail before construction if any test requests pretrained weights."""
    original = extension.swin_t

    def construct(*, weights):
        assert weights is None, "Tests must never request ImageNet weights"
        return original(weights=weights)

    monkeypatch.setattr(extension, "swin_t", construct)


def test_flat_swin_constructs_offline_and_has_four_class_output():
    model = extension.build_swin_t_classification_model(4, pretrained="none").eval()
    assert isinstance(model.head[0], nn.Dropout)
    assert model.head[0].p == pytest.approx(0.2)
    assert model.head[1].in_features == 768
    with torch.inference_mode():
        outputs = model(torch.rand(1, 3, 64, 64))
    assert outputs.shape == (1, 4)


def test_shared_swin_preserves_full_feature_path_and_calls_encoder_once(monkeypatch):
    # Capture the actual torchvision modules to guard against skipping or
    # replacing the pretrained normalization/permutation/pooling path.
    original = extension.swin_t
    captured = {}

    def capture(*, weights):
        backbone = original(weights=weights)
        captured["backbone"] = backbone
        return backbone

    monkeypatch.setattr(extension, "swin_t", capture)
    model = extension.build_swin_t_shared_three_task_model(pretrained="none").eval()
    assert isinstance(model, extension.SwinTSharedThreeTaskModel)
    assert model.feature_dimension == 768
    path = ("features", "norm", "permute", "avgpool", "flatten")
    assert tuple(model.encoder._modules) == path
    for name in path:
        assert getattr(model.encoder, name) is getattr(captured["backbone"], name)

    encoder_outputs = []
    handle = model.encoder.register_forward_hook(
        lambda _module, _inputs, output: encoder_outputs.append(output)
    )
    try:
        with torch.inference_mode():
            outputs = model(torch.rand(1, 3, 64, 64))
    finally:
        handle.remove()
    assert len(encoder_outputs) == 1
    assert encoder_outputs[0].shape == (1, 768)
    assert set(outputs) == {"task1", "task2", "task3"}
    for task, classes in (("task1", 2), ("task2", 3), ("task3", 5)):
        assert outputs[task].shape == (1, classes)
        head = getattr(model, f"{task}_head")
        assert len(head) == 2
        assert isinstance(head[0], nn.Dropout)
        assert head[0].p == pytest.approx(0.2)
        assert isinstance(head[1], nn.Linear)
        assert head[1].in_features == 768
        assert head[1].out_features == classes


@pytest.mark.parametrize("builder", [
    lambda **kwargs: extension.build_swin_t_classification_model(4, **kwargs),
    extension.build_swin_t_shared_three_task_model,
])
def test_invalid_pretrained_mode(builder):
    with pytest.raises(ValueError, match="pretrained"):
        builder(pretrained="invalid")


@pytest.mark.parametrize("dropout", [-0.1, 1.0, float("nan")])
@pytest.mark.parametrize("builder", [
    lambda **kwargs: extension.build_swin_t_classification_model(4, **kwargs),
    extension.build_swin_t_shared_three_task_model,
])
def test_invalid_dropout(builder, dropout):
    with pytest.raises(ValueError, match="dropout_probability"):
        builder(pretrained="none", dropout_probability=dropout)


@pytest.mark.parametrize("classes", [-1, 0, 1])
def test_invalid_flat_class_count(classes):
    with pytest.raises(ValueError, match="number_of_classes"):
        extension.build_swin_t_classification_model(classes, pretrained="none")


def test_swin_is_excluded_from_frozen_historical_registries():
    assert "swin_t" not in SUPPORTED_CLASSIFICATION_ARCHITECTURES
    assert "swin_t" not in SUPPORTED_SHARED_ARCHITECTURES
    assert len(SUPPORTED_CLASSIFICATION_ARCHITECTURES) == 7
    assert len(SUPPORTED_SHARED_ARCHITECTURES) == 7
