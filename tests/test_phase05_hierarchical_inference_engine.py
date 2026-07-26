from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import src.evaluation.hierarchical_inference_engine as engine
from src.data.dataloaders import DataLoaderConfig
from src.evaluation.hierarchical_inference_engine import (
    collect_hierarchical_predictions,
    run_locked_hierarchical_evaluation,
)
from src.evaluation.hierarchical_protocol import (
    FrozenCheckpointSpec,
    HierarchicalEvaluationProtocol,
)


class TinyHierarchicalDataset(
    Dataset[dict[str, object]]
):
    def __init__(self) -> None:
        self.stage_1_targets = [0, 0, 1, 1, 1, 1]
        self.stage_2_targets = [-1, -1, 0, 1, 2, 0]
        self.final_targets = [0, 0, 1, 2, 3, 1]

    def __len__(self) -> int:
        return 6

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, object]:
        image = torch.full(
            (1, 2, 2),
            float(index),
            dtype=torch.float32,
        )

        return {
            "image": image,
            "image_id": f"image_{index}",
            "image_path": f"/images/image_{index}.jpg",
            "split_group_id": f"group_{index}",
            "file_sha256": f"hash_{index}",
            "stage_1_target": torch.tensor(
                self.stage_1_targets[index],
                dtype=torch.long,
            ),
            "stage_2_target": torch.tensor(
                self.stage_2_targets[index],
                dtype=torch.long,
            ),
            "final_target": torch.tensor(
                self.final_targets[index],
                dtype=torch.long,
            ),
        }


class IndexLogitModel(nn.Module):
    def __init__(
        self,
        logits_by_index: list[list[float]],
    ) -> None:
        super().__init__()
        self.register_buffer(
            "lookup",
            torch.tensor(
                logits_by_index,
                dtype=torch.float32,
            ),
        )

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        indices = images[:, 0, 0, 0].long()
        return self.lookup[indices]


def _loader() -> DataLoader:
    return DataLoader(
        TinyHierarchicalDataset(),
        batch_size=2,
        shuffle=False,
    )


def _stage_1_model() -> nn.Module:
    # Predictions:
    # non_malignant, malignant, non_malignant,
    # malignant, malignant, malignant
    return IndexLogitModel(
        [
            [5.0, 0.0],
            [0.0, 5.0],
            [5.0, 0.0],
            [0.0, 5.0],
            [0.0, 5.0],
            [0.0, 5.0],
        ]
    )


def _stage_2_model() -> nn.Module:
    # Stage 2 executes for indices 1 through 5.
    # Predictions: bcc, bcc, bcc, scc, melanoma.
    return IndexLogitModel(
        [
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
            [5.0, 0.0, 0.0],
        ]
    )


def test_collect_hierarchical_predictions_uses_locked_union() -> None:
    collection = collect_hierarchical_predictions(
        _stage_1_model(),
        _stage_2_model(),
        _loader(),
        device="cpu",
    )

    assert collection.sample_count == 6
    assert collection.stage_1_predictions.tolist() == [
        0,
        1,
        0,
        1,
        1,
        1,
    ]
    assert collection.stage_2_execution_mask.tolist() == [
        False,
        True,
        True,
        True,
        True,
        True,
    ]
    assert collection.stage_2_predictions.tolist() == [
        -1,
        1,
        1,
        1,
        2,
        0,
    ]
    assert np.isnan(
        collection.stage_2_probabilities[0]
    ).all()
    assert collection.metadata["image_id"] == [
        "image_0",
        "image_1",
        "image_2",
        "image_3",
        "image_4",
        "image_5",
    ]


def _protocol(
    tmp_path: Path,
) -> HierarchicalEvaluationProtocol:
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
        manifest_path=tmp_path / "manifest.csv",
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


def test_locked_engine_writes_required_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _protocol(tmp_path)

    monkeypatch.setattr(
        engine,
        "load_hierarchical_evaluation_protocol",
        lambda *args, **kwargs: protocol,
    )
    monkeypatch.setattr(
        engine,
        "build_hierarchical_inference_dataloader",
        lambda *args, **kwargs: _loader(),
    )

    def fake_model_builder(
        specification: FrozenCheckpointSpec,
        *,
        device: str | torch.device,
    ) -> tuple[nn.Module, dict[str, object]]:
        if specification.task == "stage_1":
            model = _stage_1_model()
        else:
            model = _stage_2_model()

        payload: dict[str, object] = {
            "validation_metrics": {
                "macro_f1": 0.8,
            }
        }
        return model, payload

    monkeypatch.setattr(
        engine,
        "build_verified_frozen_model",
        fake_model_builder,
    )

    outcome = run_locked_hierarchical_evaluation(
        protocol.config_path,
        project_root=tmp_path,
        device="cpu",
        batch_size=2,
        num_workers=0,
    )

    assert outcome.output_directory.is_dir()
    assert outcome.metrics_path.is_file()
    assert outcome.predictions_path.is_file()
    assert outcome.routing_path.is_file()
    assert outcome.summary_path.is_file()

    required_files = {
        "hierarchical_metrics.json",
        "routing_analysis.json",
        "error_propagation.json",
        "per_image_hierarchical_predictions.csv",
        "standalone_stage_1_confusion_matrix.csv",
        "standalone_stage_1_per_class_metrics.csv",
        "oracle_gated_stage_2_confusion_matrix.csv",
        "oracle_gated_stage_2_per_class_metrics.csv",
        "oracle_gate_four_class_confusion_matrix.csv",
        "oracle_gate_four_class_per_class_metrics.csv",
        "predicted_gate_end_to_end_confusion_matrix.csv",
        "predicted_gate_end_to_end_per_class_metrics.csv",
        "checkpoint_provenance.json",
        "environment.json",
        "locked_protocol.yaml",
        "evaluation_summary.json",
    }

    assert required_files.issubset(
        {
            path.name
            for path in outcome.output_directory.iterdir()
        }
    )

    summary = json.loads(
        outcome.summary_path.read_text(
            encoding="utf-8"
        )
    )
    assert summary["sample_count"] == 6
    assert (
        summary[
            "union_stage_2_execution_count"
        ]
        == 5
    )
    assert summary["rerun_permitted"] is False

    with outcome.predictions_path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 6
    assert rows[0]["stage_2_executed"] == "0"
    assert rows[0]["stage_2_predicted_label"] == ""
    assert rows[1]["routing_status"] == (
        "non_malignant_incorrectly_routed"
    )
    assert rows[2]["routing_status"] == (
        "malignant_blocked_by_stage_1"
    )


def test_locked_engine_rejects_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _protocol(tmp_path)
    protocol.output_directory.mkdir()

    monkeypatch.setattr(
        engine,
        "load_hierarchical_evaluation_protocol",
        lambda *args, **kwargs: protocol,
    )

    with pytest.raises(
        FileExistsError,
        match="already exists",
    ):
        run_locked_hierarchical_evaluation(
            protocol.config_path,
            project_root=tmp_path,
            device="cpu",
        )
