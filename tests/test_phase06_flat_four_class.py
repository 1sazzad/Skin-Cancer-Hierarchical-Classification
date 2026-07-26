from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.flat_four_class_audit import audit_flat_four_class_manifest
from src.data.isic2019_dataset import (
    FLAT_DIAGNOSIS_TO_CLASS,
    FLAT_FOUR_CLASS_TO_INDEX,
    ISIC2019HierarchicalDataset,
    map_flat_diagnosis,
)
from src.training.baseline_experiment import load_experiment_config


@pytest.mark.parametrize(
    ("diagnosis", "expected"),
    list(FLAT_DIAGNOSIS_TO_CLASS.items()),
)
def test_supported_diagnoses_map_to_expected_flat_class(
    diagnosis: str,
    expected: str,
) -> None:
    assert map_flat_diagnosis(diagnosis) == expected


def test_flat_class_order_is_locked() -> None:
    assert dict(FLAT_FOUR_CLASS_TO_INDEX) == {
        "non_malignant": 0,
        "melanoma": 1,
        "bcc": 2,
        "scc": 3,
    }


@pytest.mark.parametrize("diagnosis", ["", None, pd.NA])
def test_missing_flat_diagnosis_is_rejected(diagnosis: object) -> None:
    with pytest.raises(ValueError, match="Missing diagnosis_canonical"):
        map_flat_diagnosis(diagnosis)


def test_unknown_flat_diagnosis_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown diagnosis_canonical"):
        map_flat_diagnosis("unknown_diagnosis")


def test_flat_dataset_returns_four_class_integer_targets(
    synthetic_project: tuple[Path, Path],
) -> None:
    project_root, manifest_path = synthetic_project
    dataset = ISIC2019HierarchicalDataset(
        manifest_path,
        project_root,
        "train",
        "flat_four_class",
    )

    assert dataset.targets == [0, 1, 2]
    assert dataset.class_counts() == {
        "non_malignant": 1,
        "melanoma": 1,
        "bcc": 1,
        "scc": 0,
    }


def test_phase06_config_preserves_task_metadata_and_selection_policy() -> None:
    config = load_experiment_config(
        "configs/experiments/"
        "phase06_flat_four_class_isic2019_efficientnet_b0_cross_entropy.yaml"
    )

    assert config["data"]["task"] == "flat_four_class"
    assert config["data"]["label_source"] == "diagnosis_canonical"
    assert config["data"]["label_mapping_strategy"] == "phase06_flat_four_class_v1"
    assert config["model"]["number_of_classes"] == 4
    assert config["training"]["selection_metric"] == "macro_f1"


def test_repository_manifest_audit_reconciles_without_leakage() -> None:
    audit = audit_flat_four_class_manifest(
        "data/manifests/isic2019_train_val_test_split_seed42.csv"
    )

    assert audit["manifest_row_count"] == 25331
    assert audit["mapped_row_count"] == 24460
    assert audit["excluded_rows"] == 871
    assert audit["reconciled"] is True
    assert audit["leakage"]["passed"] is True
    assert audit["counts"]["train"]["total"] == 17124
    assert audit["counts"]["validation"]["total"] == 3668
    assert audit["counts"]["test"]["total"] == 3668
