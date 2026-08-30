"""Shared one-encoder three-task model across the frozen Phase 06 backbones."""

from __future__ import annotations

from types import MappingProxyType
from typing import Literal, Mapping

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import (
    DenseNet121_Weights,
    DenseNet169_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_B2_Weights,
    EfficientNet_B3_Weights,
    MobileNet_V3_Large_Weights,
    ResNet50_Weights,
    densenet121,
    densenet169,
    efficientnet_b0,
    efficientnet_b2,
    efficientnet_b3,
    mobilenet_v3_large,
    resnet50,
)

PretrainedMode = Literal["none", "imagenet"]
ArchitectureName = Literal[
    "efficientnet_b0",
    "densenet121",
    "densenet169",
    "resnet50",
    "mobilenet_v3_large",
    "efficientnet_b2",
    "efficientnet_b3",
]

SUPPORTED_SHARED_ARCHITECTURES: tuple[ArchitectureName, ...] = (
    "efficientnet_b0",
    "densenet121",
    "densenet169",
    "resnet50",
    "mobilenet_v3_large",
    "efficientnet_b2",
    "efficientnet_b3",
)

TASK_CLASS_MAPPINGS: Mapping[str, Mapping[str, int]] = MappingProxyType(
    {
        "task1": MappingProxyType({"non_malignant": 0, "malignant": 1}),
        "task2": MappingProxyType({"melanoma": 0, "bcc": 1, "scc": 2}),
        "task3": MappingProxyType(
            {"Tis": 0, "T1": 1, "T2": 2, "T3": 3, "T4": 4}
        ),
    }
)


class _DenseNetEncoder(nn.Module):
    """DenseNet feature extractor matching torchvision's pre-classifier path."""

    def __init__(self, features: nn.Module) -> None:
        super().__init__()
        self.features = features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        return F.relu(features, inplace=True)


class SharedThreeTaskModel(nn.Module):
    """One shared encoder with frozen Task-1/Task-2/Task-3 head semantics."""

    def __init__(
        self,
        architecture: str,
        *,
        pretrained: PretrainedMode = "imagenet",
        dropout_probability: float = 0.2,
    ) -> None:
        super().__init__()
        if pretrained not in {"none", "imagenet"}:
            raise ValueError("pretrained must be either 'none' or 'imagenet'.")
        if not 0.0 <= dropout_probability < 1.0:
            raise ValueError("dropout_probability must be in [0, 1).")
        if architecture not in SUPPORTED_SHARED_ARCHITECTURES:
            supported = ", ".join(SUPPORTED_SHARED_ARCHITECTURES)
            raise ValueError(
                f"Unsupported shared architecture {architecture!r}. "
                f"Supported architectures: {supported}."
            )

        self.architecture = architecture
        self.encoder, self.pool, feature_dimension = self._build_encoder(
            architecture, pretrained
        )
        self.task1_head = self._make_head(
            feature_dimension, 2, dropout_probability
        )
        self.task2_head = self._make_head(
            feature_dimension, 3, dropout_probability
        )
        self.task3_head = self._make_head(
            feature_dimension, 5, dropout_probability
        )

    @staticmethod
    def _build_encoder(
        architecture: str,
        pretrained: PretrainedMode,
    ) -> tuple[nn.Module, nn.Module, int]:
        use_imagenet = pretrained == "imagenet"

        if architecture == "efficientnet_b0":
            backbone = efficientnet_b0(
                weights=EfficientNet_B0_Weights.DEFAULT if use_imagenet else None
            )
            final_linear = backbone.classifier[-1]
            if not isinstance(final_linear, nn.Linear):
                raise TypeError("Expected EfficientNet-B0 classifier to end in Linear.")
            return backbone.features, backbone.avgpool, int(final_linear.in_features)

        if architecture == "efficientnet_b2":
            backbone = efficientnet_b2(
                weights=EfficientNet_B2_Weights.DEFAULT if use_imagenet else None
            )
            final_linear = backbone.classifier[-1]
            if not isinstance(final_linear, nn.Linear):
                raise TypeError("Expected EfficientNet-B2 classifier to end in Linear.")
            return backbone.features, backbone.avgpool, int(final_linear.in_features)

        if architecture == "efficientnet_b3":
            backbone = efficientnet_b3(
                weights=EfficientNet_B3_Weights.DEFAULT if use_imagenet else None
            )
            final_linear = backbone.classifier[-1]
            if not isinstance(final_linear, nn.Linear):
                raise TypeError("Expected EfficientNet-B3 classifier to end in Linear.")
            return backbone.features, backbone.avgpool, int(final_linear.in_features)

        if architecture == "densenet121":
            backbone = densenet121(
                weights=DenseNet121_Weights.DEFAULT if use_imagenet else None
            )
            classifier = backbone.classifier
            if not isinstance(classifier, nn.Linear):
                raise TypeError("Expected DenseNet-121 classifier to be Linear.")
            return (
                _DenseNetEncoder(backbone.features),
                nn.AdaptiveAvgPool2d((1, 1)),
                int(classifier.in_features),
            )

        if architecture == "densenet169":
            backbone = densenet169(
                weights=DenseNet169_Weights.DEFAULT if use_imagenet else None
            )
            classifier = backbone.classifier
            if not isinstance(classifier, nn.Linear):
                raise TypeError("Expected DenseNet-169 classifier to be Linear.")
            return (
                _DenseNetEncoder(backbone.features),
                nn.AdaptiveAvgPool2d((1, 1)),
                int(classifier.in_features),
            )

        if architecture == "resnet50":
            backbone = resnet50(
                weights=ResNet50_Weights.DEFAULT if use_imagenet else None
            )
            classifier = backbone.fc
            if not isinstance(classifier, nn.Linear):
                raise TypeError("Expected ResNet-50 classifier to be Linear.")
            encoder = nn.Sequential(*list(backbone.children())[:-2])
            return encoder, backbone.avgpool, int(classifier.in_features)

        if architecture == "mobilenet_v3_large":
            backbone = mobilenet_v3_large(
                weights=(
                    MobileNet_V3_Large_Weights.DEFAULT if use_imagenet else None
                )
            )
            classifier = backbone.classifier
            if not isinstance(classifier, nn.Sequential):
                raise TypeError("Unexpected MobileNetV3-Large classifier structure.")
            first_linear = classifier[0]
            if not isinstance(first_linear, nn.Linear):
                raise TypeError("Expected MobileNetV3-Large classifier to start in Linear.")
            return backbone.features, backbone.avgpool, int(first_linear.in_features)

        raise AssertionError("Architecture validation should make this unreachable.")

    @staticmethod
    def _make_head(
        feature_dimension: int,
        number_of_classes: int,
        dropout_probability: float,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Dropout(p=dropout_probability),
            nn.Linear(feature_dimension, number_of_classes),
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return all task logits from exactly one shared encoder pass."""

        features = self.encoder(images)
        features = self.pool(features)
        features = torch.flatten(features, 1)
        return {
            "task1": self.task1_head(features),
            "task2": self.task2_head(features),
            "task3": self.task3_head(features),
        }


class SharedThreeTaskEfficientNetB0(SharedThreeTaskModel):
    """Backward-compatible frozen Phase 03 EfficientNet-B0 shared model."""

    def __init__(
        self,
        *,
        pretrained: PretrainedMode = "imagenet",
        dropout_probability: float = 0.2,
    ) -> None:
        super().__init__(
            "efficientnet_b0",
            pretrained=pretrained,
            dropout_probability=dropout_probability,
        )


def build_shared_three_task_model(
    architecture: str,
    *,
    pretrained: PretrainedMode = "imagenet",
    dropout_probability: float = 0.2,
) -> SharedThreeTaskModel:
    """Build one approved Phase 06 shared three-task architecture."""

    return SharedThreeTaskModel(
        architecture,
        pretrained=pretrained,
        dropout_probability=dropout_probability,
    )


def build_shared_three_task_efficientnet_b0(
    *,
    pretrained: PretrainedMode = "imagenet",
    dropout_probability: float = 0.2,
) -> SharedThreeTaskEfficientNetB0:
    """Build the frozen Phase 03 shared baseline with state-dict compatibility."""

    return SharedThreeTaskEfficientNetB0(
        pretrained=pretrained,
        dropout_probability=dropout_probability,
    )
