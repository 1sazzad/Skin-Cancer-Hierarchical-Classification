"""PAUL-2026 Swin-T extension, separate from the frozen paper CNN registries."""

from __future__ import annotations

from collections import OrderedDict
from typing import Literal

import torch
from torch import nn
from torchvision.models import Swin_T_Weights, swin_t

PretrainedMode = Literal["none", "imagenet"]


def _validate_options(pretrained: str, dropout_probability: float) -> None:
    if pretrained not in {"none", "imagenet"}:
        raise ValueError("pretrained must be either 'none' or 'imagenet'.")
    if not 0.0 <= dropout_probability < 1.0:
        raise ValueError("dropout_probability must be in [0, 1).")


def _make_head(feature_dimension: int, classes: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Dropout(p=dropout), nn.Linear(feature_dimension, classes))


def build_swin_t_classification_model(
    number_of_classes: int,
    pretrained: PretrainedMode = "imagenet",
    dropout_probability: float = 0.2,
) -> nn.Module:
    """Replace only Swin-T's classifier, preserving its full feature path."""
    _validate_options(pretrained, dropout_probability)
    if number_of_classes < 2:
        raise ValueError("number_of_classes must be at least 2.")
    backbone = swin_t(
        weights=Swin_T_Weights.DEFAULT if pretrained == "imagenet" else None
    )
    feature_dimension = backbone.head.in_features
    backbone.head = _make_head(
        feature_dimension, number_of_classes, dropout_probability
    )
    return backbone


class SwinTSharedThreeTaskModel(nn.Module):
    """One complete Swin-T encoder pass feeding the three hierarchical heads."""

    def __init__(
        self,
        pretrained: PretrainedMode = "imagenet",
        dropout_probability: float = 0.2,
    ) -> None:
        super().__init__()
        _validate_options(pretrained, dropout_probability)
        backbone = swin_t(
            weights=Swin_T_Weights.DEFAULT if pretrained == "imagenet" else None
        )
        self.feature_dimension = backbone.head.in_features
        self.encoder = nn.Sequential(OrderedDict([
            ("features", backbone.features),
            ("norm", backbone.norm),
            ("permute", backbone.permute),
            ("avgpool", backbone.avgpool),
            ("flatten", backbone.flatten),
        ]))
        self.task1_head = _make_head(self.feature_dimension, 2, dropout_probability)
        self.task2_head = _make_head(self.feature_dimension, 3, dropout_probability)
        self.task3_head = _make_head(self.feature_dimension, 5, dropout_probability)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(images)
        return {
            "task1": self.task1_head(features),
            "task2": self.task2_head(features),
            "task3": self.task3_head(features),
        }


def build_swin_t_shared_three_task_model(
    pretrained: PretrainedMode = "imagenet",
    dropout_probability: float = 0.2,
) -> SwinTSharedThreeTaskModel:
    """Build the PAUL-2026 extension without historical registry integration."""
    return SwinTSharedThreeTaskModel(pretrained, dropout_probability)
