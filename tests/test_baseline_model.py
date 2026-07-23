from __future__ import annotations

import pytest
import torch

from src.models.efficientnet_baseline import build_efficientnet_b0


@pytest.mark.parametrize("number_of_classes", [2, 3])
def test_efficientnet_baseline_output_shape(number_of_classes: int) -> None:
    model = build_efficientnet_b0(number_of_classes, pretrained="none")
    model.eval()

    with torch.inference_mode():
        output = model(torch.rand(1, 3, 64, 64))

    assert output.shape == (1, number_of_classes)


def test_efficientnet_baseline_rejects_invalid_class_count() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        build_efficientnet_b0(1, pretrained="none")
