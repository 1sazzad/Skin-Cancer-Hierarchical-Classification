from __future__ import annotations

from pathlib import Path

import torch

from src.data.dataloaders import DataLoaderConfig, build_stage_dataloaders


def test_build_stage_dataloaders_returns_all_locked_splits(
    synthetic_project: tuple[Path, Path],
) -> None:
    project_root, manifest_path = synthetic_project
    loaders = build_stage_dataloaders(
        manifest_path,
        project_root,
        "stage_1",
        config=DataLoaderConfig(batch_size=2, num_workers=0, seed=42),
    )

    assert set(loaders) == {"train", "validation", "internal_test"}
    assert len(loaders["train"].dataset) == 3
    assert len(loaders["validation"].dataset) == 2
    assert len(loaders["internal_test"].dataset) == 2

    validation_batch = next(iter(loaders["validation"]))
    assert validation_batch["image"].shape == (2, 3, 224, 224)
    assert validation_batch["target"].dtype == torch.int64
    assert validation_batch["image_id"] == ["val_nv", "val_scc"]
