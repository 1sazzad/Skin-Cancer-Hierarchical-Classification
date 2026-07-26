from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.data.hierarchical_inference_dataset import (
    ISIC2019HierarchicalInferenceDataset,
)
from src.evaluation.hierarchical_evaluator import (
    build_hierarchical_routing,
    compute_hierarchical_evaluation,
)


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(
        "RGB",
        (8, 8),
        color=(100, 120, 140),
    ).save(path)


def _manifest_row(
    *,
    image_id: str,
    image_path: str,
    stage_1_label: str,
    stage_2_label: str,
    include_stage_1: str = "1",
    include_stage_2: str = "0",
    split: str = "internal_test",
) -> dict[str, str]:
    return {
        "dataset": "isic2019",
        "image_id": image_id,
        "image_path": image_path,
        "split": split,
        "split_included": "1",
        "split_group_id": f"group_{image_id}",
        "include_stage_1": include_stage_1,
        "include_stage_2": include_stage_2,
        "stage_1_label": stage_1_label,
        "stage_2_label": stage_2_label,
        "file_sha256": f"hash_{image_id}",
    }


def test_hierarchical_dataset_builds_locked_four_class_targets(
    tmp_path: Path,
) -> None:
    rows: list[dict[str, str]] = []

    cases = [
        ("nm", "non_malignant", "", "0"),
        ("mel", "malignant", "melanoma", "1"),
        ("bcc", "malignant", "bcc", "1"),
        ("scc", "malignant", "scc", "1"),
    ]

    for image_id, stage_1, stage_2, include_stage_2 in cases:
        relative_path = Path("images") / f"{image_id}.jpg"
        _write_image(tmp_path / relative_path)

        rows.append(
            _manifest_row(
                image_id=image_id,
                image_path=str(relative_path),
                stage_1_label=stage_1,
                stage_2_label=stage_2,
                include_stage_2=include_stage_2,
            )
        )

    rows.append(
        _manifest_row(
            image_id="excluded_ak",
            image_path="images/excluded_ak.jpg",
            stage_1_label="",
            stage_2_label="",
            include_stage_1="0",
            include_stage_2="0",
        )
    )

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(
        manifest_path,
        index=False,
    )

    dataset = ISIC2019HierarchicalInferenceDataset(
        manifest_path,
        tmp_path,
        transform=lambda image: torch.zeros(
            (3, 8, 8),
            dtype=torch.float32,
        ),
        verify_image_paths=True,
    )

    assert len(dataset) == 4
    assert dataset.final_class_counts() == {
        "non_malignant": 1,
        "melanoma": 1,
        "bcc": 1,
        "scc": 1,
    }

    samples = [dataset[index] for index in range(len(dataset))]

    assert [
        sample["stage_1_target"].item()
        for sample in samples
    ] == [0, 1, 1, 1]

    assert [
        sample["stage_2_target"].item()
        for sample in samples
    ] == [-1, 0, 1, 2]

    assert [
        sample["final_target"].item()
        for sample in samples
    ] == [0, 1, 2, 3]

    assert [
        sample["final_label"]
        for sample in samples
    ] == [
        "non_malignant",
        "melanoma",
        "bcc",
        "scc",
    ]


def test_hierarchical_dataset_rejects_malignant_row_without_stage_2(
    tmp_path: Path,
) -> None:
    relative_path = Path("images") / "mel.jpg"
    _write_image(tmp_path / relative_path)

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            _manifest_row(
                image_id="mel",
                image_path=str(relative_path),
                stage_1_label="malignant",
                stage_2_label="melanoma",
                include_stage_2="0",
            )
        ]
    ).to_csv(
        manifest_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="must be Stage 2-eligible",
    ):
        ISIC2019HierarchicalInferenceDataset(
            manifest_path,
            tmp_path,
        )


def test_hierarchical_routing_quantifies_error_propagation() -> None:
    stage_1_targets = np.array(
        [0, 0, 1, 1, 1, 1],
        dtype=np.int64,
    )
    stage_1_predictions = np.array(
        [0, 1, 0, 1, 1, 1],
        dtype=np.int64,
    )
    stage_2_targets = np.array(
        [-1, -1, 0, 1, 2, 2],
        dtype=np.int64,
    )
    stage_2_predictions = np.array(
        [-1, 2, 1, 1, 0, 2],
        dtype=np.int64,
    )

    outcome = build_hierarchical_routing(
        stage_1_targets,
        stage_1_predictions,
        stage_2_targets,
        stage_2_predictions,
    )

    assert outcome.final_targets.tolist() == [
        0,
        0,
        1,
        2,
        3,
        3,
    ]
    assert outcome.oracle_gate_predictions.tolist() == [
        0,
        0,
        2,
        2,
        1,
        3,
    ]
    assert outcome.predicted_gate_predictions.tolist() == [
        0,
        3,
        0,
        2,
        1,
        3,
    ]
    assert outcome.stage_2_execution_mask.tolist() == [
        False,
        True,
        True,
        True,
        True,
        True,
    ]

    routing = outcome.routing_counts

    assert routing["stage_2_execution_count"] == 5
    assert routing["malignant_blocked_by_stage_1"] == 1
    assert (
        routing[
            "non_malignant_incorrectly_routed_to_stage_2"
        ]
        == 1
    )
    assert routing["correctly_routed_malignant"] == 3
    assert routing["subtype_correct_after_correct_route"] == 2
    assert routing["subtype_error_after_correct_route"] == 1


def test_hierarchical_metrics_include_all_required_views() -> None:
    results = compute_hierarchical_evaluation(
        stage_1_targets=[
            0, 0,
            1, 1, 1, 1, 1, 1,
        ],
        stage_1_predictions=[
            0, 1,
            0, 1, 1, 1, 1, 1,
        ],
        stage_2_targets=[
            -1, -1,
            0, 0, 1, 1, 2, 2,
        ],
        stage_2_predictions=[
            -1, 1,
            1, 0, 1, 2, 2, 0,
        ],
    )

    assert set(results) == {
        "protocol",
        "standalone_stage_1",
        "oracle_gated_stage_2",
        "oracle_gate_four_class",
        "predicted_gate_end_to_end",
        "routing",
    }

    assert results["protocol"]["final_class_names"] == [
        "non_malignant",
        "melanoma",
        "bcc",
        "scc",
    ]

    predicted_gate = results["predicted_gate_end_to_end"]
    assert len(predicted_gate["confusion_matrix"]) == 4
    assert set(predicted_gate["per_class"]) == {
        "non_malignant",
        "melanoma",
        "bcc",
        "scc",
    }


def test_routing_requires_stage_2_prediction_for_union_rows() -> None:
    with pytest.raises(
        ValueError,
        match="stage_2_predictions for executed rows",
    ):
        build_hierarchical_routing(
            stage_1_targets=[0, 1],
            stage_1_predictions=[1, 0],
            stage_2_targets=[-1, 0],
            stage_2_predictions=[-1, -1],
        )
