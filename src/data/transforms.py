"""Deterministic evaluation and moderate train-only image transforms."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(frozen=True, slots=True)
class TransformConfig:
    """Configuration shared by Stage 1, Stage 2, and flat baselines."""

    image_size: int = 224
    eval_resize_size: int = 256
    random_resized_crop_scale: tuple[float, float] = (0.85, 1.0)
    random_resized_crop_ratio: tuple[float, float] = (0.90, 1.10)
    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.5
    rotation_degrees: float = 15.0
    brightness_jitter: float = 0.10
    contrast_jitter: float = 0.10
    saturation_jitter: float = 0.10
    hue_jitter: float = 0.02
    normalization_mean: tuple[float, float, float] = IMAGENET_MEAN
    normalization_std: tuple[float, float, float] = IMAGENET_STD

    def __post_init__(self) -> None:
        if self.image_size <= 0 or self.eval_resize_size <= 0:
            raise ValueError("Image sizes must be positive integers.")
        if self.eval_resize_size < self.image_size:
            raise ValueError("eval_resize_size must be at least image_size.")
        for name, probability in (
            ("horizontal_flip_probability", self.horizontal_flip_probability),
            ("vertical_flip_probability", self.vertical_flip_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")


def build_train_transform(config: TransformConfig | None = None) -> v2.Compose:
    """Build the locked moderate train-only augmentation pipeline."""

    cfg = config or TransformConfig()
    return v2.Compose(
        [
            v2.ToImage(),
            v2.RandomResizedCrop(
                size=(cfg.image_size, cfg.image_size),
                scale=cfg.random_resized_crop_scale,
                ratio=cfg.random_resized_crop_ratio,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            v2.RandomHorizontalFlip(p=cfg.horizontal_flip_probability),
            v2.RandomVerticalFlip(p=cfg.vertical_flip_probability),
            v2.RandomRotation(
                degrees=cfg.rotation_degrees,
                interpolation=InterpolationMode.BILINEAR,
            ),
            v2.ColorJitter(
                brightness=cfg.brightness_jitter,
                contrast=cfg.contrast_jitter,
                saturation=cfg.saturation_jitter,
                hue=cfg.hue_jitter,
            ),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=cfg.normalization_mean, std=cfg.normalization_std),
        ]
    )


def build_eval_transform(config: TransformConfig | None = None) -> v2.Compose:
    """Build deterministic validation and test preprocessing."""

    cfg = config or TransformConfig()
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(
                size=cfg.eval_resize_size,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            v2.CenterCrop(size=(cfg.image_size, cfg.image_size)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=cfg.normalization_mean, std=cfg.normalization_std),
        ]
    )
