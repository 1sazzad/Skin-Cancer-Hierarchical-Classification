"""Train/validation-safe data plumbing for the shared three-task baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from src.data.dataloaders import DataLoaderConfig
from src.data.emb_stage03 import EMBStage03Dataset
from src.data.isic2019_dataset import ISIC2019HierarchicalDataset
from src.data.transforms import build_eval_transform, build_train_transform
from src.utils.reproducibility import make_generator, seed_worker

MISSING_TARGET = -100
TaskSource = Literal["isic2019", "isic_stage03"]


def encode_isic_shared_targets(
    flat_four_class_target: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode safe Task-1/Task-2 targets and masks from a flat ISIC target."""

    target = int(flat_four_class_target)
    if target not in range(4):
        raise ValueError("ISIC flat target must be in [0, 3].")
    if target == 0:
        targets = (0, MISSING_TARGET, MISSING_TARGET)
        mask = (True, False, False)
    else:
        targets = (1, target - 1, MISSING_TARGET)
        mask = (True, True, False)
    return (
        torch.tensor(targets, dtype=torch.long),
        torch.tensor(mask, dtype=torch.bool),
    )


def encode_stage3_shared_targets(
    stage3_target: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a Stage-3 target without inventing Task-1 or Task-2 labels."""

    target = int(stage3_target)
    if target not in range(5):
        raise ValueError("Stage-3 target must be in [0, 4].")
    return (
        torch.tensor(
            (MISSING_TARGET, MISSING_TARGET, target),
            dtype=torch.long,
        ),
        torch.tensor((False, False, True), dtype=torch.bool),
    )


class SharedTaskDataset(Dataset[dict[str, Any]]):
    """Adapt one eligible source dataset to explicit targets and task masks."""

    def __init__(
        self,
        dataset: ISIC2019HierarchicalDataset | EMBStage03Dataset,
        source: TaskSource,
    ) -> None:
        if source == "isic2019":
            if not isinstance(dataset, ISIC2019HierarchicalDataset):
                raise TypeError("isic2019 source requires ISIC2019HierarchicalDataset.")
            if dataset.stage != "flat_four_class":
                raise ValueError("ISIC shared data must use flat_four_class targets.")
        elif source == "isic_stage03":
            if not isinstance(dataset, EMBStage03Dataset):
                raise TypeError("isic_stage03 source requires EMBStage03Dataset.")
        else:
            raise ValueError(f"Unsupported shared-task source: {source!r}.")
        self.dataset = dataset
        self.source = source

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        if self.source == "isic2019":
            targets, task_mask = encode_isic_shared_targets(sample["target"])
        else:
            targets, task_mask = encode_stage3_shared_targets(sample["target"])
        return {
            **sample,
            "targets": targets,
            "task_mask": task_mask,
            "source": self.source,
        }

    @property
    def task_masks(self) -> list[tuple[bool, bool, bool]]:
        """Return masks without loading image data."""

        if self.source == "isic2019":
            return [
                tuple(bool(value) for value in encode_isic_shared_targets(target)[1])
                for target in self.dataset.targets
            ]
        return [(False, False, True)] * len(self.dataset)


@dataclass(frozen=True, slots=True)
class SharedThreeTaskDataLoaders:
    """Natural training pool and task-specific validation cohorts."""

    train: DataLoader
    validation_task1: DataLoader
    validation_task2: DataLoader
    validation_task3: DataLoader
    train_source_counts: dict[str, int]
    task3_train_class_counts: dict[str, int]


def _make_loader(
    dataset: Dataset,
    *,
    train: bool,
    config: DataLoaderConfig,
) -> DataLoader:
    kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_size": config.batch_size,
        "shuffle": train,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "drop_last": config.drop_last_train if train else False,
        "worker_init_fn": seed_worker,
        "generator": make_generator(config.seed),
        "persistent_workers": config.persistent_workers,
    }
    if config.num_workers > 0:
        kwargs["prefetch_factor"] = config.prefetch_factor
    return DataLoader(**kwargs)


def build_shared_three_task_dataloaders(
    isic_manifest_path: str | Path,
    stage3_manifest_path: str | Path,
    project_root: str | Path,
    *,
    config: DataLoaderConfig | None = None,
    train_transform: Callable[[Image.Image], torch.Tensor] | None = None,
    eval_transform: Callable[[Image.Image], torch.Tensor] | None = None,
    verify_image_paths: bool = False,
) -> SharedThreeTaskDataLoaders:
    """Build natural train and validation-only loaders; never construct tests."""

    loader_config = config or DataLoaderConfig(
        batch_size=64,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        seed=42,
    )
    train_transform = train_transform or build_train_transform()
    eval_transform = eval_transform or build_eval_transform()

    isic_train_base = ISIC2019HierarchicalDataset(
        isic_manifest_path,
        project_root,
        "train",
        "flat_four_class",
        train_transform,
        verify_image_paths=verify_image_paths,
    )
    stage3_train_base = EMBStage03Dataset(
        stage3_manifest_path,
        project_root,
        "train",
        train_transform,
        verify_image_paths=verify_image_paths,
    )
    isic_validation_base = ISIC2019HierarchicalDataset(
        isic_manifest_path,
        project_root,
        "validation",
        "flat_four_class",
        eval_transform,
        verify_image_paths=verify_image_paths,
    )
    stage3_validation_base = EMBStage03Dataset(
        stage3_manifest_path,
        project_root,
        "validation",
        eval_transform,
        verify_image_paths=verify_image_paths,
    )

    isic_train = SharedTaskDataset(isic_train_base, "isic2019")
    stage3_train = SharedTaskDataset(stage3_train_base, "isic_stage03")
    isic_validation = SharedTaskDataset(isic_validation_base, "isic2019")
    stage3_validation = SharedTaskDataset(
        stage3_validation_base, "isic_stage03"
    )

    combined_train = ConcatDataset([isic_train, stage3_train])
    malignant_indices = [
        index
        for index, target in enumerate(isic_validation_base.targets)
        if int(target) != 0
    ]
    task2_validation = Subset(isic_validation, malignant_indices)

    return SharedThreeTaskDataLoaders(
        train=_make_loader(combined_train, train=True, config=loader_config),
        validation_task1=_make_loader(
            isic_validation, train=False, config=loader_config
        ),
        validation_task2=_make_loader(
            task2_validation, train=False, config=loader_config
        ),
        validation_task3=_make_loader(
            stage3_validation, train=False, config=loader_config
        ),
        train_source_counts={
            "isic2019": len(isic_train),
            "isic_stage03": len(stage3_train),
            "combined": len(combined_train),
        },
        task3_train_class_counts=stage3_train_base.class_counts(),
    )
