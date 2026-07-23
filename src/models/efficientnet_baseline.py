"""EfficientNet-B0 baseline model construction."""

from __future__ import annotations

from typing import Literal

import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

PretrainedMode = Literal["none", "imagenet"]


def build_efficientnet_b0(
    number_of_classes: int,
    *,
    pretrained: PretrainedMode = "imagenet",
    dropout_probability: float = 0.2,
) -> nn.Module:
    """Build EfficientNet-B0 with a stage-specific classification head.

    Parameters
    ----------
    number_of_classes:
        Number of output logits required by the classification task.
    pretrained:
        ``"imagenet"`` loads torchvision's default ImageNet weights;
        ``"none"`` avoids any download and is suitable for offline tests.
    dropout_probability:
        Dropout used before the replacement linear classifier.
    """

    if number_of_classes < 2:
        raise ValueError("number_of_classes must be at least 2.")
    if not 0.0 <= dropout_probability < 1.0:
        raise ValueError("dropout_probability must be in [0, 1).")
    if pretrained not in {"none", "imagenet"}:
        raise ValueError("pretrained must be either 'none' or 'imagenet'.")

    weights = (
        EfficientNet_B0_Weights.DEFAULT
        if pretrained == "imagenet"
        else None
    )
    model = efficientnet_b0(weights=weights)

    classifier = model.classifier
    if not isinstance(classifier, nn.Sequential):
        raise TypeError("Unexpected EfficientNet classifier structure.")

    linear_layer = classifier[-1]
    if not isinstance(linear_layer, nn.Linear):
        raise TypeError("Expected the final EfficientNet classifier layer to be Linear.")

    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_probability, inplace=True),
        nn.Linear(linear_layer.in_features, number_of_classes),
    )
    return model
