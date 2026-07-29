"""ISIC-derived melanoma T-category labels, manifests, and dataset support."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

EMB_STAGE03_CLASS_TO_INDEX: Mapping[str, int] = MappingProxyType(
    {"Tis": 0, "T1": 1, "T2": 2, "T3": 3, "T4": 4}
)


def map_stage_ajcc(value: object) -> str:
    """Map a derived broad numeric T-category to its label."""

    if value is None or pd.isna(value) or str(value).strip() == "":
        raise ValueError("Missing derived_stage_ajcc value.")
    raw = str(value).strip()
    try:
        numeric = float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid derived_stage_ajcc value: {value!r}.") from exc
    if not numeric.is_integer() or int(numeric) not in range(5):
        raise ValueError(
            f"Invalid derived_stage_ajcc value: {value!r}; expected 0-4."
        )
    return ("Tis", "T1", "T2", "T3", "T4")[int(numeric)]


def derive_t_category_from_isic_metadata(
    diagnosis_3: object,
    mel_thick_mm: object,
) -> tuple[int, str]:
    """Derive broad AJCC8 T-category from official ISIC diagnosis/thickness."""

    if diagnosis_3 is None or pd.isna(diagnosis_3):
        raise ValueError("Missing official ISIC diagnosis_3.")
    diagnosis = " ".join(str(diagnosis_3).strip().lower().replace("_", " ").split())
    if not diagnosis:
        raise ValueError("Missing official ISIC diagnosis_3.")
    contains_in_situ = "melanoma in situ" in diagnosis
    contains_invasive = (
        "melanoma invasive" in diagnosis or "invasive melanoma" in diagnosis
    )
    if contains_in_situ and contains_invasive:
        raise ValueError("Contradictory melanoma in-situ and invasive metadata.")
    if contains_in_situ:
        if mel_thick_mm is not None and not pd.isna(mel_thick_mm):
            raw = str(mel_thick_mm).strip()
            if raw:
                try:
                    if float(raw) > 0:
                        raise ValueError(
                            "Contradictory in-situ melanoma with positive thickness."
                        )
                except ValueError as exc:
                    if "Contradictory" in str(exc):
                        raise
                    raise ValueError("Invalid in-situ melanoma thickness.") from exc
        return 0, "Tis"
    if not contains_invasive:
        if "melanoma" in diagnosis:
            raise ValueError("Melanoma NOS cannot produce an official T-category.")
        raise ValueError("Official diagnosis is not invasive or in-situ melanoma.")
    if mel_thick_mm is None or pd.isna(mel_thick_mm) or str(mel_thick_mm).strip() == "":
        raise ValueError("Invasive melanoma requires Breslow thickness.")
    try:
        thickness = float(str(mel_thick_mm).strip())
    except ValueError as exc:
        raise ValueError("Invasive melanoma thickness must be numeric.") from exc
    if not 0 < thickness < float("inf"):
        raise ValueError("Invasive melanoma thickness must be finite and positive.")
    if thickness <= 1.0:
        return 1, "T1"
    if thickness <= 2.0:
        return 2, "T2"
    if thickness <= 4.0:
        return 3, "T3"
    return 4, "T4"


def inverse_frequency_class_weights(
    labels: list[str], class_order: tuple[str, ...] = ("Tis", "T1", "T2", "T3", "T4")
) -> dict[str, float]:
    """Return mean-one inverse-frequency weights from training labels only."""

    counts = Counter(labels)
    if any(counts[name] <= 0 for name in class_order):
        raise ValueError("Every class must occur in the training split.")
    raw = {name: len(labels) / (len(class_order) * counts[name]) for name in class_order}
    scale = len(class_order) / sum(raw.values())
    return {name: raw[name] * scale for name in class_order}


class EMBStage03Dataset(Dataset[dict[str, Any]]):
    """Load one split from the VM-generated ISIC Stage-3 manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        project_root: str | Path,
        split: str,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        *,
        verify_image_paths: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.project_root = Path(project_root).expanduser().resolve()
        self.split = "test" if split == "internal_test" else split
        self.stage = "emb_stage03"
        self.transform = transform
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("EMB split must be train, validation, or test.")
        frame = pd.read_csv(self.manifest_path, dtype=str, keep_default_na=False)
        required = {
            "dataset", "image_id", "image_path", "derived_stage_ajcc", "t_category",
            "modality", "split", "split_group_id", "file_sha256",
        }
        missing = sorted(required - set(frame))
        if missing:
            raise ValueError(f"ISIC Stage-3 manifest is missing columns: {missing}")
        if set(frame["dataset"].str.strip()) != {"isic_stage03"}:
            raise ValueError(
                "Stage-3 manifest must contain only dataset='isic_stage03'."
            )
        selected = frame.loc[
            (frame["split"] == self.split)
            & (frame["modality"].str.strip().str.lower() == "dermoscopic")
        ].copy()
        if selected.empty:
            raise ValueError(f"No dermoscopic ISIC Stage-3 rows for split={self.split!r}.")
        mapped = selected["derived_stage_ajcc"].map(map_stage_ajcc)
        if not mapped.equals(selected["t_category"].str.strip()):
            raise ValueError("t_category disagrees with derived_stage_ajcc.")
        if selected["image_id"].duplicated().any():
            raise ValueError("Duplicate image_id in selected EMB split.")
        selected["_target"] = mapped.map(EMB_STAGE03_CLASS_TO_INDEX).astype("int64")
        selected["_label"] = mapped
        selected["_resolved_image_path"] = selected["image_path"].map(
            lambda value: str(
                (Path(value) if Path(value).is_absolute() else self.project_root / value)
                .resolve()
            )
        )
        if verify_image_paths:
            missing_paths = [
                path for path in selected["_resolved_image_path"] if not Path(path).is_file()
            ]
            if missing_paths:
                raise FileNotFoundError(f"{len(missing_paths)} EMB images are missing.")
        self._frame = selected.reset_index(drop=True)
        self.class_to_index = EMB_STAGE03_CLASS_TO_INDEX
        self.index_to_class = MappingProxyType(
            {value: key for key, value in EMB_STAGE03_CLASS_TO_INDEX.items()}
        )
        self.targets = self._frame["_target"].tolist()

    def __len__(self) -> int:
        return len(self._frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._frame.iloc[index]
        path = Path(row["_resolved_image_path"])
        try:
            with Image.open(path) as opened:
                image = opened.convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(f"Unable to load EMB image {row['image_id']!r}: {exc}") from exc
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "target": torch.tensor(int(row["_target"]), dtype=torch.long),
            "label": str(row["_label"]),
            "image_id": str(row["image_id"]),
            "image_path": str(path),
            "split_group_id": str(row["split_group_id"]),
            "file_sha256": str(row["file_sha256"]),
            "split": self.split,
            "stage": self.stage,
        }

    def class_counts(self) -> dict[str, int]:
        counts = Counter(self._frame["_label"])
        return {name: counts.get(name, 0) for name in EMB_STAGE03_CLASS_TO_INDEX}
