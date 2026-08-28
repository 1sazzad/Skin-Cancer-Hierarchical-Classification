"""One-encoder EfficientNet-B0 model for the shared three-task baseline."""

from __future__ import annotations

from types import MappingProxyType
from typing import Literal, Mapping

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

PretrainedMode = Literal["none", "imagenet"]

TASK_CLASS_MAPPINGS: Mapping[str, Mapping[str, int]] = MappingProxyType(
    {
        "task1": MappingProxyType({"non_malignant": 0, "malignant": 1}),
        "task2": MappingProxyType({"melanoma": 0, "bcc": 1, "scc": 2}),
        "task3": MappingProxyType(
            {"Tis": 0, "T1": 1, "T2": 2, "T3": 3, "T4": 4}
        ),
    }
)


class SharedThreeTaskEfficientNetB0(nn.Module):
    """EfficientNet-B0 encoder with three fresh dropout-linear heads."""

    def __init__(
        self,
        *,
        pretrained: PretrainedMode = "imagenet",
        dropout_probability: float = 0.2,
    ) -> None:
        super().__init__()
        if pretrained not in {"none", "imagenet"}:
            raise ValueError("pretrained must be either 'none' or 'imagenet'.")
        if not 0.0 <= dropout_probability < 1.0:
            raise ValueError("dropout_probability must be in [0, 1).")

        weights = (
            EfficientNet_B0_Weights.DEFAULT
            if pretrained == "imagenet"
            else None
        )
        backbone = efficientnet_b0(weights=weights)
        classifier = backbone.classifier
        if not isinstance(classifier, nn.Sequential):
            raise TypeError("Unexpected EfficientNet classifier structure.")
        final_linear = classifier[-1]
        if not isinstance(final_linear, nn.Linear):
            raise TypeError("Expected EfficientNet classifier to end in Linear.")

        self.encoder = backbone.features
        self.pool = backbone.avgpool
        feature_dimension = int(final_linear.in_features)
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


def build_shared_three_task_efficientnet_b0(
    *,
    pretrained: PretrainedMode = "imagenet",
    dropout_probability: float = 0.2,
) -> SharedThreeTaskEfficientNetB0:
    """Build the frozen Phase 03 shared baseline without historical checkpoints."""

    return SharedThreeTaskEfficientNetB0(
        pretrained=pretrained,
        dropout_probability=dropout_probability,
    )
