"""Deterministic DataLoader for locked Phase 05 hierarchical inference."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from src.data.dataloaders import DataLoaderConfig
from src.data.hierarchical_inference_dataset import (
    ISIC2019HierarchicalInferenceDataset,
)
from src.data.transforms import build_eval_transform
from src.utils.reproducibility import make_generator, seed_worker


def build_hierarchical_inference_dataloader(
    manifest_path: str | Path,
    project_root: str | Path,
    *,
    config: DataLoaderConfig,
    verify_image_paths: bool = True,
) -> DataLoader:
    """Build the non-shuffled internal-test loader for the hierarchy."""

    dataset = ISIC2019HierarchicalInferenceDataset(
        manifest_path=manifest_path,
        project_root=project_root,
        split="internal_test",
        transform=build_eval_transform(),
        verify_image_paths=verify_image_paths,
    )

    kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_size": config.batch_size,
        "shuffle": False,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "drop_last": False,
        "worker_init_fn": seed_worker,
        "generator": make_generator(config.seed),
        "persistent_workers": config.persistent_workers,
    }

    if config.num_workers > 0:
        kwargs["prefetch_factor"] = config.prefetch_factor

    return DataLoader(**kwargs)
