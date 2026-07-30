"""Controlled architecture-selectable classification model factory."""

from __future__ import annotations

from typing import Literal

from torch import nn

from src.models.densenet_baseline import build_densenet121
from src.models.efficientnet_baseline import build_efficientnet_b0


PretrainedMode = Literal["none", "imagenet"]
ArchitectureName = Literal["efficientnet_b0", "densenet121"]

SUPPORTED_CLASSIFICATION_ARCHITECTURES: tuple[ArchitectureName, ...] = (
    "efficientnet_b0",
    "densenet121",
)


def build_classification_model(
    architecture: str,
    number_of_classes: int,
    *,
    pretrained: PretrainedMode = "imagenet",
    dropout_probability: float = 0.2,
) -> nn.Module:
    """Build one approved classification architecture."""

    if architecture == "efficientnet_b0":
        return build_efficientnet_b0(
            number_of_classes,
            pretrained=pretrained,
            dropout_probability=dropout_probability,
        )

    if architecture == "densenet121":
        return build_densenet121(
            number_of_classes,
            pretrained=pretrained,
            dropout_probability=dropout_probability,
        )

    supported = ", ".join(SUPPORTED_CLASSIFICATION_ARCHITECTURES)
    raise ValueError(
        f"Unsupported classification architecture {architecture!r}. "
        f"Supported architectures: {supported}."
    )
