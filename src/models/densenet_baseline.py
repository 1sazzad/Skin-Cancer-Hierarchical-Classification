"""DenseNet-121 baseline model construction."""

from __future__ import annotations

from typing import Literal

import torch.nn as nn
from torchvision.models import DenseNet121_Weights, densenet121


PretrainedMode = Literal["none", "imagenet"]


def build_densenet121(
    number_of_classes: int,
    *,
    pretrained: PretrainedMode = "imagenet",
    dropout_probability: float = 0.2,
) -> nn.Module:
    """Build DenseNet-121 with the controlled classification head."""

    if number_of_classes < 2:
        raise ValueError("number_of_classes must be at least 2.")
    if not 0.0 <= dropout_probability < 1.0:
        raise ValueError("dropout_probability must be in [0, 1).")
    if pretrained not in {"none", "imagenet"}:
        raise ValueError("pretrained must be either 'none' or 'imagenet'.")

    weights = (
        DenseNet121_Weights.DEFAULT
        if pretrained == "imagenet"
        else None
    )
    model = densenet121(weights=weights)

    classifier = model.classifier
    if not isinstance(classifier, nn.Linear):
        raise TypeError("Expected the DenseNet classifier to be Linear.")

    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_probability, inplace=True),
        nn.Linear(classifier.in_features, number_of_classes),
    )
    return model
