"""Class-distribution summaries from the frozen ISIC 2019 split manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import pandas as pd

from src.data.isic2019_dataset import (
    STAGE_1_CLASS_TO_INDEX,
    STAGE_2_CLASS_TO_INDEX,
)

_SPLIT_FILTERS: dict[str, tuple[str, ...]] = {
    "train": ("train",),
    "validation": ("validation",),
    "internal_test": ("internal_test", "test"),
}
_SPLITS = tuple(_SPLIT_FILTERS)
_STAGE_DEFINITIONS: dict[str, tuple[str, str, Mapping[str, int]]] = {
    "stage_1": ("include_stage_1", "stage_1_label", STAGE_1_CLASS_TO_INDEX),
    "stage_2": ("include_stage_2", "stage_2_label", STAGE_2_CLASS_TO_INDEX),
}


def compute_class_statistics(manifest_path: str | Path) -> pd.DataFrame:
    """Return counts and proportions after all locked inclusion filters."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    frame = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    required = {
        "dataset",
        "split",
        "split_included",
        "include_stage_1",
        "include_stage_2",
        "stage_1_label",
        "stage_2_label",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Split manifest is missing required columns: {missing}")
    if set(frame["dataset"]) != {"isic2019"}:
        raise ValueError("Statistics input must contain only ISIC 2019 rows.")

    records: list[dict[str, object]] = []
    for stage, (include_column, label_column, class_to_index) in _STAGE_DEFINITIONS.items():
        for split in _SPLITS:
            selected = frame.loc[
                (frame["split_included"] == "1")
                & (frame["split"].isin(_SPLIT_FILTERS[split]))
                & (frame[include_column] == "1")
            ]
            total = len(selected)
            if total == 0:
                raise ValueError(f"No eligible rows for {stage}/{split}.")

            counts = selected[label_column].value_counts()
            unknown_labels = sorted(set(counts.index) - set(class_to_index))
            if unknown_labels:
                raise ValueError(f"Unknown labels for {stage}: {unknown_labels}")

            for label, class_index in class_to_index.items():
                count = int(counts.get(label, 0))
                records.append(
                    {
                        "stage": stage,
                        "split": split,
                        "class_index": class_index,
                        "label": label,
                        "count": count,
                        "split_total": total,
                        "proportion": count / total,
                    }
                )

    return pd.DataFrame.from_records(records).sort_values(
        ["stage", "split", "class_index"], ignore_index=True
    )


def save_class_statistics(
    statistics: pd.DataFrame,
    csv_path: str | Path,
    json_path: str | Path,
) -> None:
    """Save a machine-readable table and a compact nested audit summary."""

    csv_path = Path(csv_path)
    json_path = Path(json_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    statistics.to_csv(csv_path, index=False, float_format="%.8f")

    summary: dict[str, dict[str, object]] = {}
    for stage in statistics["stage"].unique():
        stage_rows = statistics.loc[statistics["stage"] == stage]
        summary[stage] = {}
        for split in _SPLITS:
            split_rows = stage_rows.loc[stage_rows["split"] == split]
            summary[stage][split] = {
                "total": int(split_rows["split_total"].iloc[0]),
                "class_counts": {
                    str(row.label): int(row.count)
                    for row in split_rows.itertuples(index=False)
                },
                "class_proportions": {
                    str(row.label): float(row.proportion)
                    for row in split_rows.itertuples(index=False)
                },
            }

    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
