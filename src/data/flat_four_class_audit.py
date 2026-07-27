"""Static Phase 06 label and leakage audit for the frozen ISIC 2019 manifest."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.isic2019_dataset import (
    FLAT_DIAGNOSIS_TO_CLASS,
    FLAT_FOUR_CLASS_TO_INDEX,
    map_flat_diagnosis,
)

SPLITS = ("train", "validation", "test")


def audit_flat_four_class_manifest(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path).expanduser().resolve()
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {
        "image_id",
        "file_sha256",
        "split_group_id",
        "split",
        "split_included",
        "include_stage_1",
        "diagnosis_canonical",
        "stage_1_label",
        "stage_2_label",
    }
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"Split manifest is missing audit columns: {missing}")

    eligible = frame.loc[
        (frame["split_included"] == "1") & (frame["include_stage_1"] == "1")
    ].copy()
    eligible["flat_class"] = eligible["diagnosis_canonical"].map(map_flat_diagnosis)
    if eligible["image_id"].duplicated().any():
        raise ValueError("Eligible Phase 06 rows contain duplicate image_id values.")

    expected_from_hierarchy = eligible.apply(
        lambda row: (
            "non_malignant"
            if row["stage_1_label"] == "non_malignant"
            else row["stage_2_label"]
        ),
        axis=1,
    )
    mismatches = eligible.loc[eligible["flat_class"] != expected_from_hierarchy]
    if not mismatches.empty:
        raise ValueError(
            "Canonical diagnosis mapping disagrees with frozen hierarchy labels "
            f"for {len(mismatches)} rows."
        )

    counts: dict[str, Any] = {}
    partitions = [("full_dataset", eligible)] + [
        (split, eligible.loc[eligible["split"] == split]) for split in SPLITS
    ]
    for name, subset in partitions:
        class_counts = Counter(subset["flat_class"])
        if name != "full_dataset" and set(class_counts) != set(FLAT_FOUR_CLASS_TO_INDEX):
            raise ValueError(f"Not all four classes are present in split {name!r}.")
        total = len(subset)
        counts[name] = {
            "total": total,
            "classes": {
                label: {
                    "count": int(class_counts.get(label, 0)),
                    "percentage": 100.0 * class_counts.get(label, 0) / total,
                }
                for label in FLAT_FOUR_CLASS_TO_INDEX
            },
        }

    leakage: dict[str, Any] = {}
    for column in ("split_group_id", "file_sha256"):
        split_counts = eligible.groupby(column)["split"].nunique()
        leakage[f"{column}_cross_split_count"] = int((split_counts > 1).sum())
    leakage["passed"] = all(value == 0 for value in leakage.values())
    if not leakage["passed"]:
        raise ValueError(f"Cross-split leakage detected: {leakage}")

    excluded = frame.loc[
        ~((frame["split_included"] == "1") & (frame["include_stage_1"] == "1"))
    ]
    exclusions = []
    for keys, group in excluded.groupby(
        ["split_included", "include_stage_1", "diagnosis_canonical"],
        dropna=False,
    ):
        split_included, _include_stage_1, diagnosis = keys
        reason = (
            "frozen_split_exclusion"
            if split_included != "1"
            else "outside_locked_stage_1_task_scope"
        )
        exclusions.append(
            {
                "diagnosis_canonical": diagnosis,
                "count": len(group),
                "reason": reason,
            }
        )

    return {
        "schema_version": 1,
        "task": "flat_four_class",
        "manifest": str(path),
        "manifest_row_count": len(frame),
        "mapped_row_count": len(eligible),
        "reconciled": len(eligible) + len(excluded) == len(frame),
        "class_order": list(FLAT_FOUR_CLASS_TO_INDEX),
        "class_to_index": dict(FLAT_FOUR_CLASS_TO_INDEX),
        "diagnosis_to_class": dict(FLAT_DIAGNOSIS_TO_CLASS),
        "selection_policy": "split_included=1 and include_stage_1=1",
        "counts": counts,
        "excluded_rows": len(excluded),
        "exclusions": exclusions,
        "leakage": leakage,
    }
