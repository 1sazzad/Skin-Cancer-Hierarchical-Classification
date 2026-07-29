from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from scripts.build_emb_stage03_split import build_split
from src.data.emb_stage03 import (
    EMB_STAGE03_CLASS_TO_INDEX,
    inverse_frequency_class_weights,
    map_stage_ajcc,
)
from src.models.efficientnet_baseline import build_efficientnet_b0
from src.training.baseline_experiment import load_experiment_config


def fixture_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stage in range(5):
        for index in range(20):
            rows.append(
                {
                    "image_id": f"stage{stage}_{index}",
                    "image_path": f"data/raw/emb/images/stage{stage}_{index}.jpg",
                    "stage_ajcc": stage,
                    "modality": "dermoscopic",
                    "file_sha256": f"{stage:02x}{index:062x}",
                }
            )
        rows.append(
            {
                "image_id": f"clinical_{stage}",
                "image_path": f"data/raw/emb/images/clinical_{stage}.jpg",
                "stage_ajcc": stage,
                "modality": "clinical",
                "file_sha256": f"ff{stage:062x}",
            }
        )
    return pd.DataFrame(rows)


def test_official_stage_mapping_and_rejection() -> None:
    assert [map_stage_ajcc(value) for value in range(5)] == ["Tis", "T1", "T2", "T3", "T4"]
    for invalid in (None, "", -1, 5, 1.5, "unknown"):
        with pytest.raises(ValueError):
            map_stage_ajcc(invalid)


def test_split_is_deterministic_stratified_dermoscopic_and_disjoint() -> None:
    first, weights, limitations = build_split(fixture_frame(), seed=42)
    second, second_weights, _ = build_split(fixture_frame(), seed=42)
    pd.testing.assert_frame_equal(first, second)
    assert weights == second_weights
    assert set(first["modality"]) == {"dermoscopic"}
    assert set(first["split"]) == {"train", "validation", "test"}
    assert first.groupby("split")["image_id"].nunique().sum() == first["image_id"].nunique()
    assert first.groupby("file_sha256")["split"].nunique().max() == 1
    assert set(first.groupby("split")["t_category"].nunique()) == {5}
    assert limitations


def test_duplicate_hash_stays_in_one_split() -> None:
    frame = fixture_frame()
    duplicate = frame.iloc[[0]].copy()
    duplicate["image_id"] = "duplicate_copy"
    frame = pd.concat([frame, duplicate], ignore_index=True)
    manifest, _, _ = build_split(frame)
    assert manifest.loc[manifest["file_sha256"] == frame.iloc[0]["file_sha256"], "split"].nunique() == 1


def test_train_only_class_weights() -> None:
    labels = ["Tis"] * 10 + ["T1"] * 5 + ["T2"] * 4 + ["T3"] * 2 + ["T4"]
    weights = inverse_frequency_class_weights(labels)
    assert list(weights) == list(EMB_STAGE03_CLASS_TO_INDEX)
    assert weights["T4"] > weights["Tis"]
    with pytest.raises(ValueError):
        inverse_frequency_class_weights(["Tis", "T1", "T2", "T3"])


def test_five_class_output_shape() -> None:
    model = build_efficientnet_b0(5, pretrained="none")
    model.eval()
    with torch.inference_mode():
        logits = model(torch.zeros(2, 3, 224, 224))
    assert logits.shape == (2, 5)


def test_phase09_config_resolves() -> None:
    config = load_experiment_config(
        Path("configs/experiments/phase09_stage03_emb_efficientnet_b0_cross_entropy.yaml")
    )
    assert config["data"]["class_to_index"] == {"Tis": 0, "T1": 1, "T2": 2, "T3": 3, "T4": 4}
    assert config["data"]["modality"] == "dermoscopic"
    assert config["training"]["epochs"] == 30
