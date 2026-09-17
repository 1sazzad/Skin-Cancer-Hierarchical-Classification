from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image
import yaml


@pytest.fixture()
def loss_config_factory(tmp_path: Path):
    """Build isolated loss regression inputs from the canonical Flat config.

    The counts and expected weights are fixed regression vectors, not a new
    experiment or recomputation of dataset statistics.
    """
    def make(task: str, loss: str) -> Path:
        config = yaml.safe_load(
            Path("configs/paper/flat/efficientnet_b0.yaml").read_text(encoding="utf-8")
        )
        counts = {"melanoma": 3164, "bcc": 2327, "scc": 440}
        focal_weights = {
            "melanoma": 0.3485376280807543,
            "bcc": 0.4553489231324597,
            "scc": 2.196113448786786,
        }
        if task == "flat_four_class":
            counts = {"non_malignant": 11193, **counts}
            focal_weights = {
                "non_malignant": 0.1787906368601743,
                "melanoma": 0.4439450826095749,
                "bcc": 0.579994522856464,
                "scc": 2.797269757673787,
            }
        config["experiment"]["research_stage"] = "synthetic_loss_regression"
        config["data"]["task"] = task
        config["data"]["class_to_index"] = dict(zip(counts, range(len(counts))))
        config["model"]["number_of_classes"] = len(counts)
        training = config["training"]
        training["loss"] = loss
        if loss == "class_balanced_focal_loss":
            training.update(focal_loss=True, focal_gamma=2.0, class_weights=focal_weights)
            training["class_weight_source"] = {
                "partition": "train", "method": "effective_number",
                "normalization": "sum_to_number_of_classes", "beta": 0.9999,
                "class_counts": counts,
            }
        elif loss == "weighted_cross_entropy":
            training["class_weights"] = {
                "melanoma": 0.6248419721871049,
                "bcc": 0.8495917490330898,
                "scc": 4.493181818181818,
            }
            training["class_weight_source"] = {
                "partition": "train", "total_samples": 5931, "class_counts": counts,
            }
        path = tmp_path / f"{task}_{loss}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return path
    return make


@pytest.fixture()
def stage2_focal_config(loss_config_factory):
    return loss_config_factory("stage_2", "class_balanced_focal_loss")


@pytest.fixture()
def flat_focal_config(loss_config_factory):
    return loss_config_factory("flat_four_class", "class_balanced_focal_loss")


@pytest.fixture()
def stage2_weighted_config(loss_config_factory):
    return loss_config_factory("stage_2", "weighted_cross_entropy")


@pytest.fixture()
def stage2_ce_config(loss_config_factory):
    return loss_config_factory("stage_2", "cross_entropy")


def _row(
    image_id: str,
    image_path: str,
    split: str,
    stage_1_label: str,
    stage_2_label: str,
    include_stage_1: str,
    include_stage_2: str,
    *,
    split_included: str = "1",
) -> dict[str, str]:
    canonical_by_labels = {
        ("non_malignant", ""): "melanocytic_nevus",
        ("malignant", "melanoma"): "melanoma",
        ("malignant", "bcc"): "basal_cell_carcinoma",
        ("malignant", "scc"): "squamous_cell_carcinoma",
        ("", ""): "actinic_keratosis",
    }
    return {
        "dataset": "isic2019",
        "image_id": image_id,
        "image_path": image_path,
        "source_split": "official_training_pool",
        "diagnosis_original": "",
        "diagnosis_canonical": canonical_by_labels[(stage_1_label, stage_2_label)],
        "stage_1_label": stage_1_label,
        "stage_2_label": stage_2_label,
        "stage_3_label": "",
        "patient_id": "",
        "lesion_id": "",
        "age_approx": "",
        "sex": "",
        "anatom_site_general": "",
        "include_stage_1": include_stage_1,
        "include_stage_2": include_stage_2,
        "include_stage_3": "0",
        "exclusion_reason": "",
        "file_extension": ".jpg",
        "file_size_bytes": "1",
        "file_sha256": f"hash_{image_id}",
        "split_group_id": f"group_{image_id}",
        "split": split,
        "split_seed": "42",
        "split_ratio": "0.700000/0.150000/0.150000",
        "split_included": split_included,
        "split_exclusion_reason": "" if split_included == "1" else "conflict",
    }


@pytest.fixture()
def synthetic_project(tmp_path: Path) -> tuple[Path, Path]:
    image_dir = tmp_path / "data/raw/isic2019/images"
    image_dir.mkdir(parents=True)

    rows = [
        _row("train_nv", "data/raw/isic2019/images/train_nv.jpg", "train", "non_malignant", "", "1", "0"),
        _row("train_mel", "data/raw/isic2019/images/train_mel.jpg", "train", "malignant", "melanoma", "1", "1"),
        _row("train_bcc", "data/raw/isic2019/images/train_bcc.jpg", "train", "malignant", "bcc", "1", "1"),
        _row("train_ak", "data/raw/isic2019/images/train_ak.jpg", "train", "", "", "0", "0"),
        _row("val_nv", "data/raw/isic2019/images/val_nv.jpg", "validation", "non_malignant", "", "1", "0"),
        _row("val_scc", "data/raw/isic2019/images/val_scc.jpg", "validation", "malignant", "scc", "1", "1"),
        _row("test_nv", "data/raw/isic2019/images/test_nv.jpg", "internal_test", "non_malignant", "", "1", "0"),
        _row("test_mel", "data/raw/isic2019/images/test_mel.jpg", "internal_test", "malignant", "melanoma", "1", "1"),
        _row("excluded_mel", "data/raw/isic2019/images/excluded_mel.jpg", "train", "malignant", "melanoma", "1", "1", split_included="0"),
    ]

    for index, row in enumerate(rows):
        image_path = tmp_path / row["image_path"]
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new(
            "RGB",
            (320 + index, 280 + index),
            color=(20 + index, 40 + index, 60 + index),
        ).save(image_path)

    manifest_path = tmp_path / "data/manifests/split.csv"
    manifest_path.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return tmp_path, manifest_path
