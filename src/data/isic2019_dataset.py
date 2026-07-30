"""Manifest-driven ISIC 2019 datasets for the locked hierarchy."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

TaskName = Literal["stage_1", "stage_2", "flat_four_class", "emb_stage03"]
StageName = TaskName
SplitName = Literal["train", "validation", "internal_test", "test"]

STAGE_1_CLASS_TO_INDEX: Mapping[str, int] = MappingProxyType(
    {"non_malignant": 0, "malignant": 1}
)
STAGE_2_CLASS_TO_INDEX: Mapping[str, int] = MappingProxyType(
    {"melanoma": 0, "bcc": 1, "scc": 2}
)
FLAT_FOUR_CLASS_TO_INDEX: Mapping[str, int] = MappingProxyType(
    {"non_malignant": 0, "melanoma": 1, "bcc": 2, "scc": 3}
)
FLAT_DIAGNOSIS_TO_CLASS: Mapping[str, str] = MappingProxyType(
    {
        "melanocytic_nevus": "non_malignant",
        "benign_keratosis_like_lesion": "non_malignant",
        "dermatofibroma": "non_malignant",
        "vascular_lesion": "non_malignant",
        "melanoma": "melanoma",
        "basal_cell_carcinoma": "bcc",
        "squamous_cell_carcinoma": "scc",
    }
)

_TASK_POLICIES: dict[str, tuple[str, str, Mapping[str, int]]] = {
    "stage_1": ("include_stage_1", "stage_1_label", STAGE_1_CLASS_TO_INDEX),
    "stage_2": ("include_stage_2", "stage_2_label", STAGE_2_CLASS_TO_INDEX),
    "flat_four_class": (
        "include_stage_1",
        "diagnosis_canonical",
        FLAT_FOUR_CLASS_TO_INDEX,
    ),
}
_SPLIT_ALIASES: dict[str, tuple[str, ...]] = {
    "train": ("train",),
    "validation": ("validation",),
    "internal_test": ("internal_test", "test"),
    "test": ("internal_test", "test"),
}
_ALLOWED_SPLITS = set(_SPLIT_ALIASES)
_REQUIRED_COLUMNS = {
    "dataset",
    "image_id",
    "image_path",
    "diagnosis_canonical",
    "split",
    "split_included",
    "split_group_id",
    "include_stage_1",
    "include_stage_2",
    "stage_1_label",
    "stage_2_label",
    "file_sha256",
}


def map_flat_diagnosis(diagnosis: object) -> str:
    """Map one canonical ISIC diagnosis to the locked Phase 06 class."""

    if diagnosis is None or pd.isna(diagnosis):
        raise ValueError("Missing diagnosis_canonical value for flat_four_class.")
    canonical = str(diagnosis).strip()
    if not canonical:
        raise ValueError("Missing diagnosis_canonical value for flat_four_class.")
    try:
        return FLAT_DIAGNOSIS_TO_CLASS[canonical]
    except KeyError as exc:
        raise ValueError(
            f"Unknown diagnosis_canonical {canonical!r} for flat_four_class; "
            f"expected one of {list(FLAT_DIAGNOSIS_TO_CLASS)}."
        ) from exc


class ISIC2019HierarchicalDataset(Dataset[dict[str, Any]]):
    """Load one hierarchy stage from the frozen leakage-aware split manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        project_root: str | Path,
        split: SplitName,
        stage: StageName,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        *,
        verify_image_paths: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.project_root = Path(project_root).expanduser().resolve()
        self.split = str(split)
        self.stage = str(stage)
        self.transform = transform

        if self.split not in _ALLOWED_SPLITS:
            raise ValueError(
                f"Unsupported split {self.split!r}; expected one of "
                f"{sorted(_ALLOWED_SPLITS)}."
            )

        requested_split = self.split
        self._manifest_split_values = _SPLIT_ALIASES[requested_split]

        # Keep the public pipeline name semantically explicit even when the
        # frozen manifest stores the partition as "test".
        if requested_split == "test":
            self.split = "internal_test"
        if self.stage not in _TASK_POLICIES:
            raise ValueError(
                f"Unsupported task {self.stage!r}; expected one of "
                f"{list(_TASK_POLICIES)}."
            )
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Split manifest not found: {self.manifest_path}")

        frame = pd.read_csv(
            self.manifest_path,
            dtype=str,
            keep_default_na=False,
        )
        self._validate_manifest_columns(frame)
        self._validate_dataset_identity(frame)

        include_column, label_column, class_to_index = _TASK_POLICIES[self.stage]
        selected = frame.loc[
            (frame["split_included"] == "1")
            & (frame["split"].isin(self._manifest_split_values))
            & (frame[include_column] == "1")
        ].copy()

        if selected.empty:
            raise ValueError(
                f"No eligible rows found for stage={self.stage!r}, split={self.split!r}."
            )
        if selected["image_id"].duplicated().any():
            duplicate_ids = sorted(
                selected.loc[selected["image_id"].duplicated(keep=False), "image_id"]
                .unique()
                .tolist()
            )
            raise ValueError(f"Duplicate image_id values in selected rows: {duplicate_ids[:5]}")

        source_labels = selected[label_column].str.strip()
        labels = (
            source_labels.map(map_flat_diagnosis)
            if self.stage == "flat_four_class"
            else source_labels
        )
        blank_count = int((labels == "").sum())
        if blank_count:
            raise ValueError(
                f"{blank_count} selected rows have an empty {label_column} value."
            )
        unknown_labels = sorted(set(labels) - set(class_to_index))
        if unknown_labels:
            raise ValueError(
                f"Unknown {self.stage} labels in manifest: {unknown_labels}; "
                f"expected {list(class_to_index)}."
            )

        selected = selected.reset_index(drop=True)
        selected["_label"] = labels.reset_index(drop=True)
        selected["_target"] = selected["_label"].map(class_to_index).astype("int64")
        selected["_resolved_image_path"] = selected["image_path"].map(
            self._resolve_image_path
        )

        if verify_image_paths:
            missing_paths = [
                path for path in selected["_resolved_image_path"] if not Path(path).is_file()
            ]
            if missing_paths:
                preview = ", ".join(str(path) for path in missing_paths[:3])
                raise FileNotFoundError(
                    f"{len(missing_paths)} selected image files are missing. "
                    f"Examples: {preview}"
                )

        self._frame = selected
        self.class_to_index = class_to_index
        self.index_to_class = MappingProxyType(
            {index: label for label, index in class_to_index.items()}
        )
        self.targets: list[int] = selected["_target"].tolist()

    @staticmethod
    def _validate_manifest_columns(frame: pd.DataFrame) -> None:
        missing = sorted(_REQUIRED_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(f"Split manifest is missing required columns: {missing}")

    @staticmethod
    def _validate_dataset_identity(frame: pd.DataFrame) -> None:
        identities = sorted(set(frame["dataset"].str.strip()) - {""})
        if identities != ["isic2019"]:
            raise ValueError(
                f"Expected only dataset='isic2019'; found {identities or ['<blank>']}."
            )

    def _resolve_image_path(self, raw_path: str) -> str:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        return str(candidate.resolve())

    def __len__(self) -> int:
        return len(self._frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._frame.iloc[index]
        image_path = Path(row["_resolved_image_path"])
        try:
            with Image.open(image_path) as opened_image:
                image = opened_image.convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(
                f"Unable to load image_id={row['image_id']!r} from {image_path}: {exc}"
            ) from exc

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "target": torch.tensor(int(row["_target"]), dtype=torch.long),
            "label": str(row["_label"]),
            "image_id": str(row["image_id"]),
            "image_path": str(image_path),
            "split_group_id": str(row["split_group_id"]),
            "file_sha256": str(row["file_sha256"]),
            "split": self.split,
            "stage": self.stage,
        }

    @property
    def selected_frame(self) -> pd.DataFrame:
        """Return a defensive copy for audits without exposing mutable state."""

        public_columns = [
            column for column in self._frame.columns if not column.startswith("_")
        ]
        return self._frame.loc[:, public_columns].copy()

    def class_counts(self) -> dict[str, int]:
        counts = Counter(self._frame["_label"].tolist())
        return {label: int(counts.get(label, 0)) for label in self.class_to_index}
