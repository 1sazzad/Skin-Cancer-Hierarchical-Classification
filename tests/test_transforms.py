from __future__ import annotations

import torch
from PIL import Image

from src.data.transforms import build_eval_transform, build_train_transform


def test_eval_transform_is_deterministic_and_has_expected_shape() -> None:
    image = Image.new("RGB", (320, 280), color=(90, 120, 150))
    transform = build_eval_transform()

    first = transform(image)
    second = transform(image)

    assert first.shape == (3, 224, 224)
    assert first.dtype == torch.float32
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()


def test_train_transform_has_expected_shape_and_dtype() -> None:
    image = Image.new("RGB", (320, 280), color=(90, 120, 150))
    output = build_train_transform()(image)

    assert output.shape == (3, 224, 224)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
