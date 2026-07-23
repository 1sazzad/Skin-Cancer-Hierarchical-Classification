from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from src.data.isic2019_dataset import ISIC2019HierarchicalDataset
from src.data.transforms import build_eval_transform


def test_stage_1_filters_split_and_inclusion_flags(
    synthetic_project: tuple[Path, Path],
) -> None:
    project_root, manifest_path = synthetic_project
    dataset = ISIC2019HierarchicalDataset(
        manifest_path,
        project_root,
        "train",
        "stage_1",
        build_eval_transform(),
    )

    assert len(dataset) == 3
    assert dataset.targets == [0, 1, 1]
    assert dataset.class_counts() == {"non_malignant": 1, "malignant": 2}
    assert "excluded_mel" not in set(dataset.selected_frame["image_id"])
    assert "train_ak" not in set(dataset.selected_frame["image_id"])


def test_stage_2_filters_non_malignant_rows(
    synthetic_project: tuple[Path, Path],
) -> None:
    project_root, manifest_path = synthetic_project
    dataset = ISIC2019HierarchicalDataset(
        manifest_path,
        project_root,
        "train",
        "stage_2",
        build_eval_transform(),
    )

    assert len(dataset) == 2
    assert dataset.targets == [0, 1]
    assert dataset.class_counts() == {"melanoma": 1, "bcc": 1, "scc": 0}


def test_dataset_item_contains_tensor_and_audit_identifiers(
    synthetic_project: tuple[Path, Path],
) -> None:
    project_root, manifest_path = synthetic_project
    dataset = ISIC2019HierarchicalDataset(
        manifest_path,
        project_root,
        "validation",
        "stage_1",
        build_eval_transform(),
    )

    sample = dataset[0]
    assert sample["image"].shape == (3, 224, 224)
    assert sample["image"].dtype == torch.float32
    assert sample["target"].dtype == torch.long
    assert sample["image_id"] == "val_nv"
    assert sample["split"] == "validation"
    assert sample["stage"] == "stage_1"
    assert sample["split_group_id"] == "group_val_nv"


def test_unknown_selected_label_is_rejected(
    synthetic_project: tuple[Path, Path],
) -> None:
    project_root, manifest_path = synthetic_project
    frame = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    frame.loc[frame["image_id"] == "train_nv", "stage_1_label"] = "unknown_label"
    frame.to_csv(manifest_path, index=False)

    with pytest.raises(ValueError, match="Unknown stage_1 labels"):
        ISIC2019HierarchicalDataset(
            manifest_path,
            project_root,
            "train",
            "stage_1",
        )

def test_internal_test_accepts_manifest_test_partition_name(
    synthetic_project: tuple[Path, Path],
) -> None:
    project_root, manifest_path = synthetic_project
    frame = pd.read_csv(
        manifest_path,
        dtype=str,
        keep_default_na=False,
    )
    frame.loc[frame["split"] == "internal_test", "split"] = "test"
    frame.to_csv(manifest_path, index=False)

    dataset = ISIC2019HierarchicalDataset(
        manifest_path,
        project_root,
        "internal_test",
        "stage_1",
        build_eval_transform(),
    )

    assert len(dataset) == 2
    assert dataset.targets == [0, 1]
    assert dataset[0]["split"] == "internal_test"
