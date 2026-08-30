"""Torchvision model builders added for the controlled Phase 02 benchmark."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from torch import nn
from torchvision.models import (
    DenseNet169_Weights,
    EfficientNet_B2_Weights,
    EfficientNet_B3_Weights,
    MobileNet_V3_Large_Weights,
    ResNet50_Weights,
    densenet169,
    efficientnet_b2,
    efficientnet_b3,
    mobilenet_v3_large,
    resnet50,
)

PretrainedMode = Literal["none", "imagenet"]


def _validate_arguments(
    number_of_classes: int,
    pretrained: PretrainedMode,
    dropout_probability: float,
) -> None:
    if number_of_classes < 2:
        raise ValueError("number_of_classes must be at least 2.")
    if not 0.0 <= dropout_probability < 1.0:
        raise ValueError("dropout_probability must be in [0, 1).")
    if pretrained not in {"none", "imagenet"}:
        raise ValueError("pretrained must be either 'none' or 'imagenet'.")


def _build_densenet169(
    number_of_classes: int,
    pretrained: PretrainedMode,
    dropout_probability: float,
) -> nn.Module:
    model = densenet169(
        weights=DenseNet169_Weights.DEFAULT if pretrained == "imagenet" else None
    )
    classifier = model.classifier
    if not isinstance(classifier, nn.Linear):
        raise TypeError("Expected the DenseNet classifier to be Linear.")
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_probability, inplace=True),
        nn.Linear(classifier.in_features, number_of_classes),
    )
    return model


def _build_resnet50(
    number_of_classes: int,
    pretrained: PretrainedMode,
    dropout_probability: float,
) -> nn.Module:
    model = resnet50(
        weights=ResNet50_Weights.DEFAULT if pretrained == "imagenet" else None
    )
    classifier = model.fc
    if not isinstance(classifier, nn.Linear):
        raise TypeError("Expected the ResNet classifier to be Linear.")
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_probability),
        nn.Linear(classifier.in_features, number_of_classes),
    )
    return model


def _build_mobilenet_v3_large(
    number_of_classes: int,
    pretrained: PretrainedMode,
    dropout_probability: float,
) -> nn.Module:
    model = mobilenet_v3_large(
        weights=(
            MobileNet_V3_Large_Weights.DEFAULT
            if pretrained == "imagenet"
            else None
        )
    )
    classifier = model.classifier
    if not isinstance(classifier, nn.Sequential):
        raise TypeError("Unexpected MobileNetV3 classifier structure.")
    dropout = classifier[-2]
    linear = classifier[-1]
    if not isinstance(dropout, nn.Dropout) or not isinstance(linear, nn.Linear):
        raise TypeError("Unexpected MobileNetV3 final classifier layers.")
    dropout.p = dropout_probability
    classifier[-1] = nn.Linear(linear.in_features, number_of_classes)
    return model


def _build_efficientnet(
    constructor: Callable[..., nn.Module],
    weights: object | None,
    number_of_classes: int,
    dropout_probability: float,
) -> nn.Module:
    model = constructor(weights=weights)
    classifier = model.classifier
    if not isinstance(classifier, nn.Sequential):
        raise TypeError("Unexpected EfficientNet classifier structure.")
    linear = classifier[-1]
    if not isinstance(linear, nn.Linear):
        raise TypeError("Expected the final EfficientNet classifier layer to be Linear.")
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_probability, inplace=True),
        nn.Linear(linear.in_features, number_of_classes),
    )
    return model


def build_phase02_backbone(
    architecture: str,
    number_of_classes: int,
    *,
    pretrained: PretrainedMode = "imagenet",
    dropout_probability: float = 0.2,
) -> nn.Module:
    """Build one of the five new Phase 02 backbones."""

    _validate_arguments(number_of_classes, pretrained, dropout_probability)
    if architecture == "densenet169":
        return _build_densenet169(
            number_of_classes, pretrained, dropout_probability
        )
    if architecture == "resnet50":
        return _build_resnet50(number_of_classes, pretrained, dropout_probability)
    if architecture == "mobilenet_v3_large":
        return _build_mobilenet_v3_large(
            number_of_classes, pretrained, dropout_probability
        )
    if architecture == "efficientnet_b2":
        return _build_efficientnet(
            efficientnet_b2,
            EfficientNet_B2_Weights.DEFAULT if pretrained == "imagenet" else None,
            number_of_classes,
            dropout_probability,
        )
    if architecture == "efficientnet_b3":
        return _build_efficientnet(
            efficientnet_b3,
            EfficientNet_B3_Weights.DEFAULT if pretrained == "imagenet" else None,
            number_of_classes,
            dropout_probability,
        )
    raise ValueError(f"Unsupported Phase 02 architecture {architecture!r}.")
