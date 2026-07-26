"""Dual-model inference and artifact writing for the locked Phase 05 hierarchy."""

from __future__ import annotations

import csv
import json
import platform
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from src.data.dataloaders import DataLoaderConfig
from src.data.hierarchical_dataloader import (
    build_hierarchical_inference_dataloader,
)
from src.evaluation.hierarchical_evaluator import (
    FINAL_CLASS_NAMES,
    STAGE_1_CLASS_NAMES,
    STAGE_2_CLASS_NAMES,
    HierarchicalRoutingOutcome,
    build_hierarchical_routing,
    compute_hierarchical_evaluation,
)
from src.evaluation.hierarchical_protocol import (
    HierarchicalEvaluationProtocol,
    build_verified_frozen_model,
    load_hierarchical_evaluation_protocol,
)
from src.utils.reproducibility import seed_everything


@dataclass(frozen=True, slots=True)
class HierarchicalPredictionCollection:
    """All predictions, probabilities, targets, and metadata from one pass."""

    metadata: dict[str, list[str]]
    stage_1_targets: np.ndarray
    stage_1_predictions: np.ndarray
    stage_1_probabilities: np.ndarray
    stage_2_targets: np.ndarray
    stage_2_predictions: np.ndarray
    stage_2_probabilities: np.ndarray
    final_targets: np.ndarray
    stage_2_execution_mask: np.ndarray
    sample_count: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class HierarchicalEvaluationOutcome:
    """Primary files produced by the locked Phase 05 evaluation."""

    output_directory: Path
    metrics_path: Path
    predictions_path: Path
    routing_path: Path
    summary_path: Path
    predicted_gate_macro_f1: float


def _json_dump(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _environment_payload(device: torch.device) -> dict[str, object]:
    payload: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
    }

    if device.type == "cuda":
        payload.update(
            {
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_memory_bytes": int(
                    torch.cuda.get_device_properties(device).total_memory
                ),
            }
        )

    return payload


def _string_batch(
    batch: Mapping[str, object],
    key: str,
    count: int,
) -> list[str]:
    value = batch.get(key)

    if isinstance(value, str):
        values = [value]
    elif isinstance(value, Sequence):
        values = [str(item) for item in value]
    else:
        raise TypeError(f"batch[{key!r}] must contain strings.")

    if len(values) != count:
        raise ValueError(
            f"batch[{key!r}] length does not match the batch size."
        )

    return values


def _tensor_vector(
    batch: Mapping[str, object],
    key: str,
    count: int,
) -> torch.Tensor:
    value = batch.get(key)

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"batch[{key!r}] must be a tensor.")
    if value.ndim != 1 or int(value.shape[0]) != count:
        raise ValueError(
            f"batch[{key!r}] must have shape ({count},)."
        )

    return value.to(dtype=torch.long)


def _validate_logits(
    logits: torch.Tensor,
    *,
    row_count: int,
    class_count: int,
    name: str,
) -> None:
    expected = (row_count, class_count)

    if logits.ndim != 2 or tuple(logits.shape) != expected:
        raise ValueError(
            f"{name} logits must have shape {expected}; "
            f"observed {tuple(logits.shape)}."
        )


def collect_hierarchical_predictions(
    stage_1_model: nn.Module,
    stage_2_model: nn.Module,
    dataloader: Iterable[Mapping[str, object]],
    *,
    device: str | torch.device,
) -> HierarchicalPredictionCollection:
    """Run Stage 1 on all rows and Stage 2 on the locked execution union."""

    resolved_device = torch.device(device)

    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    stage_1_model.to(resolved_device)
    stage_2_model.to(resolved_device)
    stage_1_model.eval()
    stage_2_model.eval()

    metadata: dict[str, list[str]] = {
        "image_id": [],
        "image_path": [],
        "split_group_id": [],
        "file_sha256": [],
    }

    stage_1_targets_all: list[torch.Tensor] = []
    stage_1_predictions_all: list[torch.Tensor] = []
    stage_1_probabilities_all: list[torch.Tensor] = []

    stage_2_targets_all: list[torch.Tensor] = []
    stage_2_predictions_all: list[torch.Tensor] = []
    stage_2_probabilities_all: list[torch.Tensor] = []

    final_targets_all: list[torch.Tensor] = []
    execution_masks_all: list[torch.Tensor] = []

    total_samples = 0
    use_amp = resolved_device.type == "cuda"
    started_at = time.perf_counter()

    with torch.inference_mode():
        for batch in dataloader:
            images = batch.get("image")

            if not isinstance(images, torch.Tensor):
                raise TypeError("Each batch must contain tensor images.")
            if images.ndim != 4:
                raise ValueError(
                    "Each image batch must have shape "
                    "(batch, channels, height, width)."
                )

            batch_count = int(images.shape[0])
            if batch_count <= 0:
                raise ValueError("Empty batches are not supported.")

            stage_1_targets = _tensor_vector(
                batch,
                "stage_1_target",
                batch_count,
            )
            stage_2_targets = _tensor_vector(
                batch,
                "stage_2_target",
                batch_count,
            )
            final_targets = _tensor_vector(
                batch,
                "final_target",
                batch_count,
            )

            images_device = images.to(
                resolved_device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type=resolved_device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                stage_1_logits = stage_1_model(images_device)

            _validate_logits(
                stage_1_logits,
                row_count=batch_count,
                class_count=len(STAGE_1_CLASS_NAMES),
                name="Stage 1",
            )

            stage_1_probabilities = torch.softmax(
                stage_1_logits.float(),
                dim=1,
            ).cpu()
            stage_1_predictions = stage_1_probabilities.argmax(
                dim=1
            )

            true_malignant = stage_1_targets.cpu() == 1
            predicted_malignant = stage_1_predictions == 1
            execution_mask = true_malignant | predicted_malignant

            stage_2_probabilities = torch.full(
                (batch_count, len(STAGE_2_CLASS_NAMES)),
                float("nan"),
                dtype=torch.float32,
            )
            stage_2_predictions = torch.full(
                (batch_count,),
                -1,
                dtype=torch.long,
            )

            execution_count = int(execution_mask.sum())

            if execution_count:
                device_mask = execution_mask.to(resolved_device)

                with torch.autocast(
                    device_type=resolved_device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    stage_2_logits = stage_2_model(
                        images_device[device_mask]
                    )

                _validate_logits(
                    stage_2_logits,
                    row_count=execution_count,
                    class_count=len(STAGE_2_CLASS_NAMES),
                    name="Stage 2",
                )

                executed_probabilities = torch.softmax(
                    stage_2_logits.float(),
                    dim=1,
                ).cpu()
                executed_predictions = (
                    executed_probabilities.argmax(dim=1)
                )

                stage_2_probabilities[execution_mask] = (
                    executed_probabilities
                )
                stage_2_predictions[execution_mask] = (
                    executed_predictions
                )

            stage_1_targets_all.append(stage_1_targets.cpu())
            stage_1_predictions_all.append(
                stage_1_predictions.cpu()
            )
            stage_1_probabilities_all.append(
                stage_1_probabilities
            )

            stage_2_targets_all.append(stage_2_targets.cpu())
            stage_2_predictions_all.append(
                stage_2_predictions
            )
            stage_2_probabilities_all.append(
                stage_2_probabilities
            )

            final_targets_all.append(final_targets.cpu())
            execution_masks_all.append(execution_mask.cpu())

            for key in metadata:
                metadata[key].extend(
                    _string_batch(batch, key, batch_count)
                )

            total_samples += batch_count

    elapsed_seconds = time.perf_counter() - started_at

    if total_samples == 0:
        raise ValueError(
            "Hierarchical inference loader produced no samples."
        )

    return HierarchicalPredictionCollection(
        metadata=metadata,
        stage_1_targets=torch.cat(
            stage_1_targets_all
        ).numpy(),
        stage_1_predictions=torch.cat(
            stage_1_predictions_all
        ).numpy(),
        stage_1_probabilities=torch.cat(
            stage_1_probabilities_all
        ).numpy(),
        stage_2_targets=torch.cat(
            stage_2_targets_all
        ).numpy(),
        stage_2_predictions=torch.cat(
            stage_2_predictions_all
        ).numpy(),
        stage_2_probabilities=torch.cat(
            stage_2_probabilities_all
        ).numpy(),
        final_targets=torch.cat(
            final_targets_all
        ).numpy(),
        stage_2_execution_mask=torch.cat(
            execution_masks_all
        ).numpy(),
        sample_count=total_samples,
        elapsed_seconds=elapsed_seconds,
    )


def _write_confusion_matrix_csv(
    path: Path,
    class_names: Sequence[str],
    matrix: Sequence[Sequence[int]],
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["actual\\predicted", *class_names]
        )

        for class_name, row in zip(
            class_names,
            matrix,
            strict=True,
        ):
            writer.writerow([class_name, *row])


def _write_per_class_metrics_csv(
    path: Path,
    class_names: Sequence[str],
    per_class: Mapping[str, Any],
) -> None:
    fieldnames = [
        "class_name",
        "precision",
        "recall",
        "f1",
        "support",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for class_name in class_names:
            values = per_class[class_name]
            writer.writerow(
                {
                    "class_name": class_name,
                    **values,
                }
            )


def _write_metric_artifacts(
    output_directory: Path,
    prefix: str,
    metrics: Mapping[str, Any],
) -> None:
    class_names = [
        str(name) for name in metrics["class_names"]
    ]

    _write_confusion_matrix_csv(
        output_directory
        / f"{prefix}_confusion_matrix.csv",
        class_names,
        metrics["confusion_matrix"],
    )

    _write_per_class_metrics_csv(
        output_directory
        / f"{prefix}_per_class_metrics.csv",
        class_names,
        metrics["per_class"],
    )


def _routing_status(
    stage_1_target: int,
    stage_1_prediction: int,
    stage_2_target: int,
    stage_2_prediction: int,
) -> str:
    if stage_1_target == 0 and stage_1_prediction == 0:
        return "correctly_not_routed"

    if stage_1_target == 0 and stage_1_prediction == 1:
        return "non_malignant_incorrectly_routed"

    if stage_1_target == 1 and stage_1_prediction == 0:
        return "malignant_blocked_by_stage_1"

    if stage_2_target == stage_2_prediction:
        return "correctly_routed_subtype_correct"

    return "correctly_routed_subtype_error"


def _write_predictions_csv(
    path: Path,
    collection: HierarchicalPredictionCollection,
    routing: HierarchicalRoutingOutcome,
) -> None:
    stage_1_probability_fields = [
        f"stage_1_probability_{name}"
        for name in STAGE_1_CLASS_NAMES
    ]
    stage_2_probability_fields = [
        f"stage_2_probability_{name}"
        for name in STAGE_2_CLASS_NAMES
    ]

    fieldnames = [
        "image_id",
        "image_path",
        "split_group_id",
        "file_sha256",
        "stage_1_target_index",
        "stage_1_target_label",
        "stage_1_predicted_index",
        "stage_1_predicted_label",
        "stage_1_correct",
        *stage_1_probability_fields,
        "stage_2_executed",
        "stage_2_target_index",
        "stage_2_target_label",
        "stage_2_predicted_index",
        "stage_2_predicted_label",
        "stage_2_correct_on_true_malignant",
        *stage_2_probability_fields,
        "final_target_index",
        "final_target_label",
        "oracle_gate_predicted_index",
        "oracle_gate_predicted_label",
        "oracle_gate_correct",
        "predicted_gate_predicted_index",
        "predicted_gate_predicted_label",
        "predicted_gate_correct",
        "routing_status",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for index in range(collection.sample_count):
            stage_1_target = int(
                collection.stage_1_targets[index]
            )
            stage_1_prediction = int(
                collection.stage_1_predictions[index]
            )
            stage_2_target = int(
                collection.stage_2_targets[index]
            )
            stage_2_prediction = int(
                collection.stage_2_predictions[index]
            )
            final_target = int(
                collection.final_targets[index]
            )
            oracle_prediction = int(
                routing.oracle_gate_predictions[index]
            )
            predicted_gate_prediction = int(
                routing.predicted_gate_predictions[index]
            )
            executed = bool(
                collection.stage_2_execution_mask[index]
            )

            row: dict[str, object] = {
                key: values[index]
                for key, values in collection.metadata.items()
            }

            row.update(
                {
                    "stage_1_target_index": stage_1_target,
                    "stage_1_target_label": (
                        STAGE_1_CLASS_NAMES[stage_1_target]
                    ),
                    "stage_1_predicted_index": (
                        stage_1_prediction
                    ),
                    "stage_1_predicted_label": (
                        STAGE_1_CLASS_NAMES[
                            stage_1_prediction
                        ]
                    ),
                    "stage_1_correct": int(
                        stage_1_target
                        == stage_1_prediction
                    ),
                    "stage_2_executed": int(executed),
                    "stage_2_target_index": (
                        stage_2_target
                        if stage_2_target >= 0
                        else ""
                    ),
                    "stage_2_target_label": (
                        STAGE_2_CLASS_NAMES[
                            stage_2_target
                        ]
                        if stage_2_target >= 0
                        else ""
                    ),
                    "stage_2_predicted_index": (
                        stage_2_prediction
                        if stage_2_prediction >= 0
                        else ""
                    ),
                    "stage_2_predicted_label": (
                        STAGE_2_CLASS_NAMES[
                            stage_2_prediction
                        ]
                        if stage_2_prediction >= 0
                        else ""
                    ),
                    "stage_2_correct_on_true_malignant": (
                        int(
                            stage_2_target
                            == stage_2_prediction
                        )
                        if stage_2_target >= 0
                        else ""
                    ),
                    "final_target_index": final_target,
                    "final_target_label": (
                        FINAL_CLASS_NAMES[final_target]
                    ),
                    "oracle_gate_predicted_index": (
                        oracle_prediction
                    ),
                    "oracle_gate_predicted_label": (
                        FINAL_CLASS_NAMES[
                            oracle_prediction
                        ]
                    ),
                    "oracle_gate_correct": int(
                        final_target
                        == oracle_prediction
                    ),
                    "predicted_gate_predicted_index": (
                        predicted_gate_prediction
                    ),
                    "predicted_gate_predicted_label": (
                        FINAL_CLASS_NAMES[
                            predicted_gate_prediction
                        ]
                    ),
                    "predicted_gate_correct": int(
                        final_target
                        == predicted_gate_prediction
                    ),
                    "routing_status": _routing_status(
                        stage_1_target,
                        stage_1_prediction,
                        stage_2_target,
                        stage_2_prediction,
                    ),
                }
            )

            for class_index, field_name in enumerate(
                stage_1_probability_fields
            ):
                row[field_name] = float(
                    collection.stage_1_probabilities[
                        index,
                        class_index,
                    ]
                )

            for class_index, field_name in enumerate(
                stage_2_probability_fields
            ):
                row[field_name] = (
                    float(
                        collection.stage_2_probabilities[
                            index,
                            class_index,
                        ]
                    )
                    if executed
                    else ""
                )

            writer.writerow(row)


def _error_propagation_payload(
    oracle_metrics: Mapping[str, Any],
    predicted_metrics: Mapping[str, Any],
) -> dict[str, float]:
    payload: dict[str, float] = {}

    for metric_name in (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
    ):
        oracle_value = float(
            oracle_metrics[metric_name]
        )
        predicted_value = float(
            predicted_metrics[metric_name]
        )

        payload[f"{metric_name}_oracle_gate"] = (
            oracle_value
        )
        payload[
            f"{metric_name}_predicted_gate"
        ] = predicted_value
        payload[
            f"{metric_name}_delta_predicted_minus_oracle"
        ] = predicted_value - oracle_value
        payload[
            f"{metric_name}_loss_from_stage_1_propagation"
        ] = oracle_value - predicted_value

    return payload


def _routing_by_final_class(
    collection: HierarchicalPredictionCollection,
    routing: HierarchicalRoutingOutcome,
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}

    for class_index, class_name in enumerate(
        FINAL_CLASS_NAMES
    ):
        mask = collection.final_targets == class_index

        result[class_name] = {
            "sample_count": int(mask.sum()),
            "stage_1_predicted_malignant": int(
                (
                    mask
                    & (
                        collection.stage_1_predictions
                        == 1
                    )
                ).sum()
            ),
            "stage_1_predicted_non_malignant": int(
                (
                    mask
                    & (
                        collection.stage_1_predictions
                        == 0
                    )
                ).sum()
            ),
            "stage_2_executed": int(
                (
                    mask
                    & collection.stage_2_execution_mask
                ).sum()
            ),
            "oracle_gate_correct": int(
                (
                    mask
                    & (
                        routing.oracle_gate_predictions
                        == collection.final_targets
                    )
                ).sum()
            ),
            "predicted_gate_correct": int(
                (
                    mask
                    & (
                        routing.predicted_gate_predictions
                        == collection.final_targets
                    )
                ).sum()
            ),
        }

    return result


def run_locked_hierarchical_evaluation(
    config_path: str | Path,
    *,
    project_root: str | Path,
    device: str | torch.device = "cuda",
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> HierarchicalEvaluationOutcome:
    """Execute the one locked Phase 05 hierarchical internal-test evaluation."""

    protocol = load_hierarchical_evaluation_protocol(
        config_path,
        project_root=project_root,
    )

    output = protocol.output_directory

    if output.exists():
        raise FileExistsError(
            f"Locked Phase 05 output already exists: {output}"
        )

    resolved_device = torch.device(device)
    if (
        resolved_device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is not available."
        )

    effective_batch_size = int(
        protocol.loader_config.batch_size
        if batch_size is None
        else batch_size
    )
    effective_num_workers = int(
        protocol.loader_config.num_workers
        if num_workers is None
        else num_workers
    )

    loader_config = DataLoaderConfig(
        batch_size=effective_batch_size,
        num_workers=effective_num_workers,
        pin_memory=(
            protocol.loader_config.pin_memory
            if resolved_device.type == "cuda"
            else False
        ),
        persistent_workers=(
            protocol.loader_config.persistent_workers
            and effective_num_workers > 0
        ),
        prefetch_factor=(
            protocol.loader_config.prefetch_factor
        ),
        drop_last_train=False,
        seed=protocol.seed,
    )

    seed_everything(protocol.seed)

    dataloader = build_hierarchical_inference_dataloader(
        protocol.manifest_path,
        protocol.project_root,
        config=loader_config,
        verify_image_paths=protocol.verify_image_paths,
    )

    stage_1_model, stage_1_payload = (
        build_verified_frozen_model(
            protocol.stage_1,
            device=resolved_device,
        )
    )
    stage_2_model, stage_2_payload = (
        build_verified_frozen_model(
            protocol.stage_2,
            device=resolved_device,
        )
    )

    collection = collect_hierarchical_predictions(
        stage_1_model,
        stage_2_model,
        dataloader,
        device=resolved_device,
    )

    routing = build_hierarchical_routing(
        collection.stage_1_targets,
        collection.stage_1_predictions,
        collection.stage_2_targets,
        collection.stage_2_predictions,
    )

    if not np.array_equal(
        collection.final_targets,
        routing.final_targets,
    ):
        raise ValueError(
            "Dataset final targets do not match derived hierarchy targets."
        )

    metrics = compute_hierarchical_evaluation(
        collection.stage_1_targets,
        collection.stage_1_predictions,
        collection.stage_2_targets,
        collection.stage_2_predictions,
    )

    oracle_four_class = metrics[
        "oracle_gate_four_class"
    ]
    predicted_four_class = metrics[
        "predicted_gate_end_to_end"
    ]

    error_propagation = _error_propagation_payload(
        oracle_four_class,
        predicted_four_class,
    )
    metrics["error_propagation"] = (
        error_propagation
    )

    routing_payload = {
        "aggregate": metrics["routing"],
        "by_true_final_class": _routing_by_final_class(
            collection,
            routing,
        ),
    }

    output.mkdir(
        parents=True,
        exist_ok=False,
    )

    metrics_path = output / "hierarchical_metrics.json"
    predictions_path = (
        output
        / "per_image_hierarchical_predictions.csv"
    )
    routing_path = output / "routing_analysis.json"
    error_propagation_path = (
        output
        / "error_propagation.json"
    )
    summary_path = output / "evaluation_summary.json"
    environment_path = output / "environment.json"
    provenance_path = (
        output
        / "checkpoint_provenance.json"
    )
    protocol_snapshot_path = (
        output
        / "locked_protocol.yaml"
    )

    _json_dump(metrics_path, metrics)
    _json_dump(routing_path, routing_payload)
    _json_dump(
        error_propagation_path,
        error_propagation,
    )

    _write_metric_artifacts(
        output,
        "standalone_stage_1",
        metrics["standalone_stage_1"],
    )
    _write_metric_artifacts(
        output,
        "oracle_gated_stage_2",
        metrics["oracle_gated_stage_2"],
    )
    _write_metric_artifacts(
        output,
        "oracle_gate_four_class",
        oracle_four_class,
    )
    _write_metric_artifacts(
        output,
        "predicted_gate_end_to_end",
        predicted_four_class,
    )

    _write_predictions_csv(
        predictions_path,
        collection,
        routing,
    )

    environment = _environment_payload(
        resolved_device
    )
    _json_dump(environment_path, environment)

    provenance = {
        "stage_1": {
            "checkpoint_path": str(
                protocol.stage_1.path
            ),
            "checkpoint_sha256": (
                protocol.stage_1.sha256
            ),
            "checkpoint_epoch": (
                protocol.stage_1.epoch
            ),
            "checkpoint_validation_metrics": (
                stage_1_payload[
                    "validation_metrics"
                ]
            ),
            "class_names": list(
                protocol.stage_1.class_names
            ),
        },
        "stage_2": {
            "checkpoint_path": str(
                protocol.stage_2.path
            ),
            "checkpoint_sha256": (
                protocol.stage_2.sha256
            ),
            "checkpoint_epoch": (
                protocol.stage_2.epoch
            ),
            "checkpoint_validation_metrics": (
                stage_2_payload[
                    "validation_metrics"
                ]
            ),
            "class_names": list(
                protocol.stage_2.class_names
            ),
        },
    }
    _json_dump(provenance_path, provenance)

    shutil.copyfile(
        protocol.config_path,
        protocol_snapshot_path,
    )

    summary = {
        "evaluation_scope": (
            "isic2019_internal_test_conditional_hierarchy"
        ),
        "reportable_internal_test_result": True,
        "rerun_permitted": False,
        "protocol_config": str(
            protocol.config_path
        ),
        "output_directory": str(output),
        "sample_count": collection.sample_count,
        "oracle_gate_stage_2_execution_count": int(
            (
                collection.stage_1_targets
                == 1
            ).sum()
        ),
        "predicted_gate_stage_2_execution_count": int(
            (
                collection.stage_1_predictions
                == 1
            ).sum()
        ),
        "union_stage_2_execution_count": int(
            collection.stage_2_execution_mask.sum()
        ),
        "batch_size": effective_batch_size,
        "num_workers": effective_num_workers,
        "elapsed_seconds": (
            collection.elapsed_seconds
        ),
        "samples_per_second": (
            collection.sample_count
            / collection.elapsed_seconds
        ),
        "headline_metrics": {
            "standalone_stage_1_macro_f1": float(
                metrics[
                    "standalone_stage_1"
                ]["macro_f1"]
            ),
            "oracle_gated_stage_2_macro_f1": float(
                metrics[
                    "oracle_gated_stage_2"
                ]["macro_f1"]
            ),
            "oracle_gate_four_class_macro_f1": float(
                oracle_four_class["macro_f1"]
            ),
            "predicted_gate_end_to_end_macro_f1": float(
                predicted_four_class["macro_f1"]
            ),
        },
        "error_propagation": error_propagation,
        "environment": environment,
    }
    _json_dump(summary_path, summary)

    return HierarchicalEvaluationOutcome(
        output_directory=output,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        routing_path=routing_path,
        summary_path=summary_path,
        predicted_gate_macro_f1=float(
            predicted_four_class["macro_f1"]
        ),
    )
