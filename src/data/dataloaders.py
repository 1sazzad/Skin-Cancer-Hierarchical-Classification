"""Reproducible DataLoader construction for Stage 1 and Stage 2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.data.isic2019_dataset import ISIC2019HierarchicalDataset, TaskName
from src.data.transforms import build_eval_transform, build_train_transform
from src.utils.reproducibility import make_generator, seed_worker


@dataclass(frozen=True, slots=True)
class DataLoaderConfig:
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int = 2
    drop_last_train: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
        if self.prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be positive.")
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")
        if self.persistent_workers and self.num_workers == 0:
            raise ValueError("persistent_workers requires num_workers > 0.")


def _make_loader(
    dataset: ISIC2019HierarchicalDataset,
    *,
    split: str,
    config: DataLoaderConfig,
) -> DataLoader:
    is_train = split == "train"
    kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_size": config.batch_size,
        "shuffle": is_train,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "drop_last": config.drop_last_train if is_train else False,
        "worker_init_fn": seed_worker,
        "generator": make_generator(config.seed),
        "persistent_workers": config.persistent_workers,
    }
    if config.num_workers > 0:
        kwargs["prefetch_factor"] = config.prefetch_factor
    return DataLoader(**kwargs)


def build_stage_dataloaders(
    manifest_path: str | Path,
    project_root: str | Path,
    stage: TaskName,
    *,
    config: DataLoaderConfig | None = None,
    train_transform: Callable[[Image.Image], torch.Tensor] | None = None,
    eval_transform: Callable[[Image.Image], torch.Tensor] | None = None,
    verify_image_paths: bool = False,
) -> dict[str, DataLoader]:
    """Build train, validation, and internal-test loaders from one frozen manifest."""

    loader_config = config or DataLoaderConfig()
    train_transform = train_transform or build_train_transform()
    eval_transform = eval_transform or build_eval_transform()

    datasets = {
        "train": ISIC2019HierarchicalDataset(
            manifest_path,
            project_root,
            "train",
            stage,
            train_transform,
            verify_image_paths=verify_image_paths,
        ),
        "validation": ISIC2019HierarchicalDataset(
            manifest_path,
            project_root,
            "validation",
            stage,
            eval_transform,
            verify_image_paths=verify_image_paths,
        ),
        "internal_test": ISIC2019HierarchicalDataset(
            manifest_path,
            project_root,
            "internal_test",
            stage,
            eval_transform,
            verify_image_paths=verify_image_paths,
        ),
    }

    return {
        split: _make_loader(dataset, split=split, config=loader_config)
        for split, dataset in datasets.items()
    }
