from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image


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
    return {
        "dataset": "isic2019",
        "image_id": image_id,
        "image_path": image_path,
        "source_split": "official_training_pool",
        "diagnosis_original": "",
        "diagnosis_canonical": "",
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
