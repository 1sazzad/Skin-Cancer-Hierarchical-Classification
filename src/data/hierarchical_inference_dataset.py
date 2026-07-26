"""Manifest-driven dataset for locked end-to-end hierarchical inference."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset


HierarchicalSplitName = Literal["internal_test", "test"]

STAGE_1_CLASS_TO_INDEX: Mapping[str, int] = MappingProxyType(
    {
        "non_malignant": 0,
        "malignant": 1,
    }
)

STAGE_2_CLASS_TO_INDEX: Mapping[str, int] = MappingProxyType(
    {
        "melanoma": 0,
        "bcc": 1,
        "scc": 2,
    }
)

FINAL_CLASS_TO_INDEX: Mapping[str, int] = MappingProxyType(
    {
        "non_malignant": 0,
        "melanoma": 1,
        "bcc": 2,
        "scc": 3,
    }
)

_REQUIRED_COLUMNS = {
    "dataset",
    "image_id",
    "image_path",
    "split",
    "split_included",
    "split_group_id",
    "include_stage_1",
    "include_stage_2",
    "stage_1_label",
    "stage_2_label",
    "file_sha256",
}

_INTERNAL_TEST_ALIASES = {"internal_test", "test"}


class ISIC2019HierarchicalInferenceDataset(Dataset[dict[str, Any]]):
    """Load every Stage 1-eligible image from the frozen internal-test split.

    Non-malignant images receive ``stage_2_target == -1`` because Stage 2 has no
    valid subtype target for them. Final targets use the locked four-class order:

    0: non_malignant
    1: melanoma
    2: bcc
    3: scc
    """

    def __init__(
        self,
        manifest_path: str | Path,
        project_root: str | Path,
        split: HierarchicalSplitName = "internal_test",
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        *,
        verify_image_paths: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.project_root = Path(project_root).expanduser().resolve()
        self.split = str(split)
        self.transform = transform

        if self.split not in _INTERNAL_TEST_ALIASES:
            raise ValueError(
                "Hierarchical inference is locked to internal_test/test only."
            )

        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Split manifest not found: {self.manifest_path}"
            )

        frame = pd.read_csv(
            self.manifest_path,
            dtype=str,
            keep_default_na=False,
        )

        self._validate_manifest_columns(frame)
        self._validate_dataset_identity(frame)

        selected = frame.loc[
            (frame["split_included"] == "1")
            & (frame["split"].isin(_INTERNAL_TEST_ALIASES))
            & (frame["include_stage_1"] == "1")
        ].copy()

        if selected.empty:
            raise ValueError(
                "No Stage 1-eligible internal-test rows were found."
            )

        if selected["image_id"].duplicated().any():
            duplicate_ids = sorted(
                selected.loc[
                    selected["image_id"].duplicated(keep=False),
                    "image_id",
                ]
                .unique()
                .tolist()
            )
            raise ValueError(
                "Duplicate image_id values in hierarchical dataset: "
                f"{duplicate_ids[:5]}"
            )

        stage_1_labels = selected["stage_1_label"].str.strip()
        unknown_stage_1 = sorted(
            set(stage_1_labels) - set(STAGE_1_CLASS_TO_INDEX)
        )
        if unknown_stage_1:
            raise ValueError(
                f"Unknown Stage 1 labels: {unknown_stage_1}"
            )

        malignant_mask = stage_1_labels == "malignant"
        non_malignant_mask = ~malignant_mask

        invalid_stage_2_inclusion = malignant_mask & (
            selected["include_stage_2"] != "1"
        )
        if invalid_stage_2_inclusion.any():
            invalid_ids = selected.loc[
                invalid_stage_2_inclusion,
                "image_id",
            ].tolist()
            raise ValueError(
                "Malignant Stage 1 rows must be Stage 2-eligible. "
                f"Examples: {invalid_ids[:5]}"
            )

        stage_2_labels = selected["stage_2_label"].str.strip()
        unknown_stage_2 = sorted(
            set(stage_2_labels.loc[malignant_mask])
            - set(STAGE_2_CLASS_TO_INDEX)
        )
        if unknown_stage_2:
            raise ValueError(
                f"Unknown malignant Stage 2 labels: {unknown_stage_2}"
            )

        selected = selected.reset_index(drop=True)
        stage_1_labels = stage_1_labels.reset_index(drop=True)
        stage_2_labels = stage_2_labels.reset_index(drop=True)
        malignant_mask = malignant_mask.reset_index(drop=True)
        non_malignant_mask = non_malignant_mask.reset_index(drop=True)

        selected["_stage_1_label"] = stage_1_labels
        selected["_stage_1_target"] = (
            stage_1_labels.map(STAGE_1_CLASS_TO_INDEX).astype("int64")
        )

        selected["_stage_2_label"] = ""
        selected.loc[
            malignant_mask,
            "_stage_2_label",
        ] = stage_2_labels.loc[malignant_mask]

        selected["_stage_2_target"] = -1
        selected.loc[
            malignant_mask,
            "_stage_2_target",
        ] = (
            stage_2_labels.loc[malignant_mask]
            .map(STAGE_2_CLASS_TO_INDEX)
            .astype("int64")
        )
        selected["_stage_2_target"] = selected[
            "_stage_2_target"
        ].astype("int64")

        selected["_final_label"] = ""
        selected.loc[
            non_malignant_mask,
            "_final_label",
        ] = "non_malignant"
        selected.loc[
            malignant_mask,
            "_final_label",
        ] = stage_2_labels.loc[malignant_mask]

        selected["_final_target"] = (
            selected["_final_label"]
            .map(FINAL_CLASS_TO_INDEX)
            .astype("int64")
        )

        selected["_resolved_image_path"] = selected["image_path"].map(
            self._resolve_image_path
        )

        if verify_image_paths:
            missing_paths = [
                path
                for path in selected["_resolved_image_path"]
                if not Path(path).is_file()
            ]
            if missing_paths:
                preview = ", ".join(
                    str(path) for path in missing_paths[:3]
                )
                raise FileNotFoundError(
                    f"{len(missing_paths)} image files are missing. "
                    f"Examples: {preview}"
                )

        self._frame = selected
        self.stage_1_class_to_index = STAGE_1_CLASS_TO_INDEX
        self.stage_2_class_to_index = STAGE_2_CLASS_TO_INDEX
        self.final_class_to_index = FINAL_CLASS_TO_INDEX

    @staticmethod
    def _validate_manifest_columns(frame: pd.DataFrame) -> None:
        missing = sorted(_REQUIRED_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(
                f"Split manifest is missing required columns: {missing}"
            )

    @staticmethod
    def _validate_dataset_identity(frame: pd.DataFrame) -> None:
        identities = sorted(
            set(frame["dataset"].str.strip()) - {""}
        )
        if identities != ["isic2019"]:
            raise ValueError(
                "Expected only dataset='isic2019'; found "
                f"{identities or ['<blank>']}."
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
        except (
            FileNotFoundError,
            UnidentifiedImageError,
            OSError,
        ) as exc:
            raise RuntimeError(
                f"Unable to load image_id={row['image_id']!r} "
                f"from {image_path}: {exc}"
            ) from exc

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "image_id": str(row["image_id"]),
            "image_path": str(image_path),
            "split_group_id": str(row["split_group_id"]),
            "file_sha256": str(row["file_sha256"]),
            "stage_1_target": torch.tensor(
                int(row["_stage_1_target"]),
                dtype=torch.long,
            ),
            "stage_1_label": str(row["_stage_1_label"]),
            "stage_2_target": torch.tensor(
                int(row["_stage_2_target"]),
                dtype=torch.long,
            ),
            "stage_2_label": str(row["_stage_2_label"]),
            "final_target": torch.tensor(
                int(row["_final_target"]),
                dtype=torch.long,
            ),
            "final_label": str(row["_final_label"]),
            "split": "internal_test",
        }

    @property
    def selected_frame(self) -> pd.DataFrame:
        """Return a defensive copy without private derived columns."""

        public_columns = [
            column
            for column in self._frame.columns
            if not column.startswith("_")
        ]
        return self._frame.loc[:, public_columns].copy()

    def final_class_counts(self) -> dict[str, int]:
        counts = Counter(self._frame["_final_label"].tolist())
        return {
            label: int(counts.get(label, 0))
            for label in FINAL_CLASS_TO_INDEX
        }
