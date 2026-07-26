from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import src.evaluation.hierarchical_preflight as preflight
from src.data.dataloaders import DataLoaderConfig
from src.evaluation.hierarchical_preflight import (
    run_hierarchical_preflight,
)
from src.evaluation.hierarchical_protocol import (
    FrozenCheckpointSpec,
    HierarchicalEvaluationProtocol,
)


class TinyOutputModel(nn.Module):
    def __init__(self, output_count: int) -> None:
        super().__init__()
        self.output_count = output_count

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros(
            (images.shape[0], self.output_count),
            dtype=images.dtype,
            device=images.device,
        )


def _protocol(
    tmp_path: Path,
) -> HierarchicalEvaluationProtocol:
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "dataset,image_id\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "phase05.yaml"
    config_path.write_text(
        "evaluation:\n  phase: phase05\n",
        encoding="utf-8",
    )

    stage_1 = FrozenCheckpointSpec(
        task="stage_1",
        path=tmp_path / "stage1.pt",
        sha256="1" * 64,
        epoch=5,
        class_to_index={
            "non_malignant": 0,
            "malignant": 1,
        },
        class_names=(
            "non_malignant",
            "malignant",
        ),
    )

    stage_2 = FrozenCheckpointSpec(
        task="stage_2",
        path=tmp_path / "stage2.pt",
        sha256="2" * 64,
        epoch=8,
        class_to_index={
            "melanoma": 0,
            "bcc": 1,
            "scc": 2,
        },
        class_names=(
            "melanoma",
            "bcc",
            "scc",
        ),
    )

    return HierarchicalEvaluationProtocol(
        config_path=config_path,
        project_root=tmp_path,
        manifest_path=manifest_path,
        output_directory=tmp_path / "locked_output",
        seed=42,
        verify_image_paths=True,
        loader_config=DataLoaderConfig(
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            seed=42,
        ),
        stage_1=stage_1,
        stage_2=stage_2,
        final_class_to_index={
            "non_malignant": 0,
            "melanoma": 1,
            "bcc": 2,
            "scc": 3,
        },
        reporting={
            "save_per_image_predictions": True,
            "save_probabilities": True,
            "save_confusion_matrices": True,
            "save_per_class_metrics": True,
            "save_routing_analysis": True,
            "save_environment": True,
        },
    )


def test_preflight_uses_only_synthetic_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _protocol(tmp_path)

    monkeypatch.setattr(
        preflight,
        "load_hierarchical_evaluation_protocol",
        lambda *args, **kwargs: protocol,
    )

    def fake_builder(
        specification: FrozenCheckpointSpec,
        *,
        device: str | torch.device,
    ) -> tuple[nn.Module, dict[str, object]]:
        output_count = (
            2 if specification.task == "stage_1" else 3
        )
        return TinyOutputModel(output_count), {}

    monkeypatch.setattr(
        preflight,
        "build_verified_frozen_model",
        fake_builder,
    )

    outcome = run_hierarchical_preflight(
        protocol.config_path,
        project_root=tmp_path,
        device="cpu",
        dummy_batch_size=3,
    )

    assert outcome.stage_1_output_shape == (3, 2)
    assert outcome.stage_2_output_shape == (3, 3)
    assert outcome.stage_1_epoch == 5
    assert outcome.stage_2_epoch == 8
    assert not protocol.output_directory.exists()


def test_preflight_rejects_existing_locked_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _protocol(tmp_path)
    protocol.output_directory.mkdir()

    monkeypatch.setattr(
        preflight,
        "load_hierarchical_evaluation_protocol",
        lambda *args, **kwargs: protocol,
    )

    with pytest.raises(
        FileExistsError,
        match="already exists",
    ):
        run_hierarchical_preflight(
            protocol.config_path,
            project_root=tmp_path,
            device="cpu",
        )
