from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.class_statistics import compute_class_statistics


def test_class_statistics_use_frozen_filters(
    synthetic_project: tuple[Path, Path],
) -> None:
    _, manifest_path = synthetic_project
    statistics = compute_class_statistics(manifest_path)

    stage_1_train = statistics.loc[
        (statistics["stage"] == "stage_1")
        & (statistics["split"] == "train")
    ].set_index("label")
    assert int(stage_1_train.loc["non_malignant", "count"]) == 1
    assert int(stage_1_train.loc["malignant", "count"]) == 2
    assert int(stage_1_train["split_total"].iloc[0]) == 3

    stage_2_validation = statistics.loc[
        (statistics["stage"] == "stage_2")
        & (statistics["split"] == "validation")
    ].set_index("label")
    assert int(stage_2_validation.loc["melanoma", "count"]) == 0
    assert int(stage_2_validation.loc["bcc", "count"]) == 0
    assert int(stage_2_validation.loc["scc", "count"]) == 1

def test_class_statistics_accept_manifest_test_partition_name(
    synthetic_project: tuple[Path, Path],
) -> None:
    _, manifest_path = synthetic_project
    frame = pd.read_csv(
        manifest_path,
        dtype=str,
        keep_default_na=False,
    )
    frame.loc[frame["split"] == "internal_test", "split"] = "test"
    frame.to_csv(manifest_path, index=False)

    statistics = compute_class_statistics(manifest_path)

    stage_1_test = statistics.loc[
        (statistics["stage"] == "stage_1")
        & (statistics["split"] == "internal_test")
    ].set_index("label")

    assert int(stage_1_test.loc["non_malignant", "count"]) == 1
    assert int(stage_1_test.loc["malignant", "count"]) == 1
    assert int(stage_1_test["split_total"].iloc[0]) == 2
