from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.evaluation.phase04_comparative_harness import (
    FINAL_CLASS_NAMES,
    FrozenCheckpointSpec,
    PredictionCollection,
    SharedISICPredictionCollection,
    benchmark_inference,
    build_paired_four_class_rows,
    count_parameters,
    evaluate_prediction_collection,
    evaluate_shared_isic,
    sha256_file,
    verify_checkpoint_artifact,
    write_paired_prediction_csv,
)


def _shared_collection() -> SharedISICPredictionCollection:
    # Flat truth: non-malignant, melanoma, BCC, SCC.
    flat = np.asarray([0, 1, 2, 3], dtype=np.int64)
    stage1_true = np.asarray([0, 1, 1, 1], dtype=np.int64)
    # One malignant sample is blocked at Task1 (the SCC row).
    stage1_pred = np.asarray([0, 1, 1, 0], dtype=np.int64)
    stage2_true = np.asarray([-1, 0, 1, 2], dtype=np.int64)
    # True malignant rows are always in the execution union, including blocked SCC.
    stage2_pred = np.asarray([-1, 0, 2, 2], dtype=np.int64)
    return SharedISICPredictionCollection(
        sample_ids=("a", "b", "c", "d"),
        flat_targets=flat,
        stage1_targets=stage1_true,
        stage1_predictions=stage1_pred,
        stage1_probabilities=np.zeros((4, 2), dtype=np.float32),
        stage2_targets=stage2_true,
        stage2_predictions=stage2_pred,
        stage2_probabilities=np.zeros((4, 3), dtype=np.float32),
        elapsed_seconds=0.1,
    )


def test_shared_metrics_include_oracle_predicted_gate_and_routing_loss() -> None:
    result = evaluate_shared_isic(_shared_collection())
    assert set(result) == {
        "task1",
        "task2_malignant_subset",
        "predicted_gate_four_class",
        "oracle_gate_four_class",
        "routing_loss_macro_f1",
        "routing",
    }
    assert result["routing"]["malignant_blocked_by_stage_1"] == 1
    assert result["routing"]["malignant_block_rate"] == pytest.approx(1 / 3)
    assert result["routing_loss_macro_f1"] == pytest.approx(
        result["oracle_gate_four_class"]["macro_f1"]
        - result["predicted_gate_four_class"]["macro_f1"]
    )


def test_paired_export_is_keyed_and_mcnemar_compatible(tmp_path: Path) -> None:
    shared = _shared_collection()
    flat = PredictionCollection(
        sample_ids=shared.sample_ids,
        targets=shared.flat_targets.copy(),
        predictions=np.asarray([0, 1, 1, 3], dtype=np.int64),
        probabilities=np.zeros((4, 4), dtype=np.float32),
        elapsed_seconds=0.1,
    )
    rows = build_paired_four_class_rows(shared, flat)
    assert [row["sample_id"] for row in rows] == ["a", "b", "c", "d"]
    assert {"true_label", "shared_correct", "flat_correct"}.issubset(rows[0])

    destination = tmp_path / "paired.csv"
    write_paired_prediction_csv(destination, rows)
    with destination.open(newline="", encoding="utf-8") as handle:
        stored = list(csv.DictReader(handle))
    assert len(stored) == 4
    assert stored[0]["sample_id"] == "a"


def test_paired_export_rejects_mismatched_sample_order() -> None:
    shared = _shared_collection()
    flat = PredictionCollection(
        sample_ids=("b", "a", "c", "d"),
        targets=shared.flat_targets.copy(),
        predictions=shared.flat_targets.copy(),
        probabilities=np.zeros((4, 4), dtype=np.float32),
        elapsed_seconds=0.1,
    )
    with pytest.raises(ValueError, match="identical stable sample order"):
        build_paired_four_class_rows(shared, flat)


def test_prediction_collection_exposes_locked_secondary_metrics() -> None:
    collection = PredictionCollection(
        sample_ids=("a", "b", "c", "d"),
        targets=np.asarray([0, 1, 2, 3]),
        predictions=np.asarray([0, 1, 1, 3]),
        probabilities=np.zeros((4, 4), dtype=np.float32),
        elapsed_seconds=0.1,
    )
    metrics = evaluate_prediction_collection(collection, FINAL_CLASS_NAMES)
    assert {
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "per_class",
        "confusion_matrix",
    }.issubset(metrics)


def test_checkpoint_sha_verification(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"frozen-checkpoint")
    spec = FrozenCheckpointSpec(
        name="synthetic",
        path="checkpoint.pt",
        sha256=sha256_file(path),
        expected_epoch=1,
        model_kind="single_task",
        class_names=("a", "b"),
    )
    assert verify_checkpoint_artifact(spec, tmp_path) == path

    bad = FrozenCheckpointSpec(
        name="synthetic",
        path="checkpoint.pt",
        sha256="0" * 64,
        expected_epoch=1,
        model_kind="single_task",
        class_names=("a", "b"),
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_checkpoint_artifact(bad, tmp_path)


def test_parameter_and_cpu_efficiency_hooks_are_executable() -> None:
    model = nn.Sequential(nn.Flatten(), nn.Linear(12, 2))
    counts = count_parameters(model)
    assert counts["total_parameters"] == 26
    benchmark = benchmark_inference(
        model,
        device="cpu",
        input_shape=(3, 2, 2),
        warmup_iterations=1,
        measured_iterations=2,
    )
    assert benchmark["throughput_images_per_second"] > 0
    assert benchmark["latency_ms_per_image"] >= 0
    assert benchmark["peak_cuda_memory_bytes"] is None
