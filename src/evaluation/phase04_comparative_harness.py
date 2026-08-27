"""Frozen Phase 04 comparative evaluation and efficiency primitives.

This module is deliberately split-agnostic. It never constructs an internal-test
loader. Gate 04D may pass validation loaders; Gate 04E may later pass the locked
internal-test loaders after the protocol gate is explicitly opened.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from src.evaluation.classification_metrics import compute_classification_metrics
from src.evaluation.hierarchical_evaluator import (
    FINAL_CLASS_NAMES,
    STAGE_1_CLASS_NAMES,
    STAGE_2_CLASS_NAMES,
    build_hierarchical_routing,
)
from src.models.classification_backbone import build_classification_model
from src.models.shared_three_task import (
    TASK_CLASS_MAPPINGS,
    build_shared_three_task_efficientnet_b0,
)

TASK3_CLASS_NAMES: tuple[str, ...] = ("Tis", "T1", "T2", "T3", "T4")
PHASE04_ARCHITECTURE = "efficientnet_b0"
PHASE04_DROPOUT = 0.2


@dataclass(frozen=True, slots=True)
class FrozenCheckpointSpec:
    name: str
    path: str
    sha256: str
    expected_epoch: int
    model_kind: str
    class_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PredictionCollection:
    sample_ids: tuple[str, ...]
    targets: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class SharedISICPredictionCollection:
    sample_ids: tuple[str, ...]
    flat_targets: np.ndarray
    stage1_targets: np.ndarray
    stage1_predictions: np.ndarray
    stage1_probabilities: np.ndarray
    stage2_targets: np.ndarray
    stage2_predictions: np.ndarray
    stage2_probabilities: np.ndarray
    elapsed_seconds: float


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint_artifact(spec: FrozenCheckpointSpec, project_root: str | Path) -> Path:
    path = Path(project_root) / spec.path
    if not path.is_file():
        raise FileNotFoundError(f"Frozen checkpoint not found: {path}")
    observed = sha256_file(path)
    if observed.lower() != spec.sha256.lower():
        raise ValueError(
            f"Checkpoint SHA-256 mismatch for {spec.name}: expected {spec.sha256}, got {observed}."
        )
    return path


def _torch_load(path: Path, device: torch.device) -> Mapping[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Checkpoint payload must be a mapping: {path}")
    if "model_state_dict" not in payload:
        raise ValueError(f"Checkpoint lacks model_state_dict: {path}")
    return payload


def _validate_epoch(payload: Mapping[str, Any], spec: FrozenCheckpointSpec) -> None:
    if int(payload.get("epoch", -1)) != spec.expected_epoch:
        raise ValueError(
            f"Frozen epoch mismatch for {spec.name}: expected {spec.expected_epoch}, "
            f"got {payload.get('epoch')!r}."
        )


def load_frozen_shared_model(
    spec: FrozenCheckpointSpec,
    project_root: str | Path,
    device: str | torch.device,
) -> nn.Module:
    if spec.model_kind != "shared_three_task":
        raise ValueError("Shared loader requires model_kind='shared_three_task'.")
    resolved_device = torch.device(device)
    path = verify_checkpoint_artifact(spec, project_root)
    payload = _torch_load(path, resolved_device)
    _validate_epoch(payload, spec)
    mappings = payload.get("class_mappings")
    expected = {task: dict(mapping) for task, mapping in TASK_CLASS_MAPPINGS.items()}
    if mappings is not None and dict(mappings) != expected:
        raise ValueError("Shared checkpoint class mappings do not match the frozen Phase 03 mappings.")
    model = build_shared_three_task_efficientnet_b0(
        pretrained="none", dropout_probability=PHASE04_DROPOUT
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(resolved_device).eval()
    return model


def load_frozen_single_task_model(
    spec: FrozenCheckpointSpec,
    project_root: str | Path,
    device: str | torch.device,
) -> nn.Module:
    if spec.model_kind != "single_task":
        raise ValueError("Single-task loader requires model_kind='single_task'.")
    resolved_device = torch.device(device)
    path = verify_checkpoint_artifact(spec, project_root)
    payload = _torch_load(path, resolved_device)
    _validate_epoch(payload, spec)
    stored_names = payload.get("class_names")
    if stored_names is not None and tuple(stored_names) != spec.class_names:
        raise ValueError(
            f"Checkpoint class order mismatch for {spec.name}: {stored_names!r}."
        )
    model = build_classification_model(
        PHASE04_ARCHITECTURE,
        len(spec.class_names),
        pretrained="none",
        dropout_probability=PHASE04_DROPOUT,
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(resolved_device).eval()
    return model


def _sample_ids(batch: Mapping[str, object], count: int) -> list[str]:
    for key in ("image_id", "sample_id"):
        value = batch.get(key)
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            values = [str(item) for item in value]
        else:
            continue
        if len(values) != count:
            raise ValueError(f"batch[{key!r}] does not match batch size.")
        return values
    raise KeyError("Every Phase04 batch must provide a stable image_id or sample_id.")


def _target_vector(batch: Mapping[str, object], count: int) -> torch.Tensor:
    value = batch.get("target")
    if not isinstance(value, torch.Tensor) or value.ndim != 1 or len(value) != count:
        raise ValueError("Every Phase04 batch must provide a 1-D tensor target.")
    return value.to(torch.long)


def collect_single_task_predictions(
    model: nn.Module,
    dataloader: Iterable[Mapping[str, object]],
    *,
    class_names: Sequence[str],
    device: str | torch.device,
) -> PredictionCollection:
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    model.to(resolved_device).eval()
    ids: list[str] = []
    targets: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in dataloader:
            images = batch.get("image")
            if not isinstance(images, torch.Tensor) or images.ndim != 4:
                raise ValueError("Each batch must provide image tensor [N,C,H,W].")
            count = int(images.shape[0])
            target = _target_vector(batch, count)
            with torch.autocast(
                device_type=resolved_device.type,
                dtype=torch.float16,
                enabled=resolved_device.type == "cuda",
            ):
                logits = model(images.to(resolved_device, non_blocking=True))
            if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (count, len(class_names)):
                raise ValueError("Unexpected single-task logit shape.")
            probs = torch.softmax(logits.float(), dim=1).cpu()
            ids.extend(_sample_ids(batch, count))
            targets.append(target.cpu())
            probabilities.append(probs)
            predictions.append(probs.argmax(dim=1))
    if not ids:
        raise ValueError("Evaluation loader produced no samples.")
    return PredictionCollection(
        tuple(ids),
        torch.cat(targets).numpy(),
        torch.cat(predictions).numpy(),
        torch.cat(probabilities).numpy(),
        time.perf_counter() - started,
    )


def collect_shared_isic_predictions(
    model: nn.Module,
    dataloader: Iterable[Mapping[str, object]],
    *,
    device: str | torch.device,
) -> SharedISICPredictionCollection:
    """Collect Task1/Task2 outputs from one encoder pass over flat ISIC rows."""
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    model.to(resolved_device).eval()
    ids: list[str] = []
    flat_targets_all: list[torch.Tensor] = []
    s1_targets_all: list[torch.Tensor] = []
    s1_preds_all: list[torch.Tensor] = []
    s1_probs_all: list[torch.Tensor] = []
    s2_targets_all: list[torch.Tensor] = []
    s2_preds_all: list[torch.Tensor] = []
    s2_probs_all: list[torch.Tensor] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in dataloader:
            images = batch.get("image")
            if not isinstance(images, torch.Tensor) or images.ndim != 4:
                raise ValueError("Each batch must provide image tensor [N,C,H,W].")
            count = int(images.shape[0])
            flat_targets = _target_vector(batch, count).cpu()
            if torch.any((flat_targets < 0) | (flat_targets > 3)):
                raise ValueError("Flat ISIC targets must be in [0,3].")
            with torch.autocast(
                device_type=resolved_device.type,
                dtype=torch.float16,
                enabled=resolved_device.type == "cuda",
            ):
                outputs = model(images.to(resolved_device, non_blocking=True))
            if not isinstance(outputs, Mapping):
                raise ValueError("Shared model must return a task-logit mapping.")
            s1_logits = outputs["task1"]
            s2_logits = outputs["task2"]
            if tuple(s1_logits.shape) != (count, 2) or tuple(s2_logits.shape) != (count, 3):
                raise ValueError("Unexpected shared Task1/Task2 logit shape.")
            s1_probs = torch.softmax(s1_logits.float(), dim=1).cpu()
            s2_probs = torch.softmax(s2_logits.float(), dim=1).cpu()
            s1_targets = (flat_targets != 0).to(torch.long)
            s1_preds = s1_probs.argmax(dim=1)
            s2_targets = torch.where(flat_targets == 0, -1, flat_targets - 1)
            execution = (s1_targets == 1) | (s1_preds == 1)
            s2_preds = torch.full((count,), -1, dtype=torch.long)
            s2_preds[execution] = s2_probs.argmax(dim=1)[execution]
            masked_s2_probs = torch.full_like(s2_probs, float("nan"))
            masked_s2_probs[execution] = s2_probs[execution]
            ids.extend(_sample_ids(batch, count))
            flat_targets_all.append(flat_targets)
            s1_targets_all.append(s1_targets)
            s1_preds_all.append(s1_preds)
            s1_probs_all.append(s1_probs)
            s2_targets_all.append(s2_targets)
            s2_preds_all.append(s2_preds)
            s2_probs_all.append(masked_s2_probs)
    if not ids:
        raise ValueError("Evaluation loader produced no samples.")
    return SharedISICPredictionCollection(
        tuple(ids),
        torch.cat(flat_targets_all).numpy(),
        torch.cat(s1_targets_all).numpy(),
        torch.cat(s1_preds_all).numpy(),
        torch.cat(s1_probs_all).numpy(),
        torch.cat(s2_targets_all).numpy(),
        torch.cat(s2_preds_all).numpy(),
        torch.cat(s2_probs_all).numpy(),
        time.perf_counter() - started,
    )


def evaluate_shared_isic(collection: SharedISICPredictionCollection) -> dict[str, object]:
    routing = build_hierarchical_routing(
        collection.stage1_targets,
        collection.stage1_predictions,
        collection.stage2_targets,
        collection.stage2_predictions,
    )
    malignant = collection.stage1_targets == 1
    task1 = compute_classification_metrics(
        collection.stage1_targets, collection.stage1_predictions, STAGE_1_CLASS_NAMES
    )
    task2 = compute_classification_metrics(
        collection.stage2_targets[malignant],
        collection.stage2_predictions[malignant],
        STAGE_2_CLASS_NAMES,
    )
    predicted_gate = compute_classification_metrics(
        routing.final_targets, routing.predicted_gate_predictions, FINAL_CLASS_NAMES
    )
    oracle_gate = compute_classification_metrics(
        routing.final_targets, routing.oracle_gate_predictions, FINAL_CLASS_NAMES
    )
    return {
        "task1": task1,
        "task2_malignant_subset": task2,
        "predicted_gate_four_class": predicted_gate,
        "oracle_gate_four_class": oracle_gate,
        "routing_loss_macro_f1": float(oracle_gate["macro_f1"]) - float(predicted_gate["macro_f1"]),
        "routing": routing.routing_counts,
    }


def evaluate_prediction_collection(
    collection: PredictionCollection, class_names: Sequence[str]
) -> dict[str, object]:
    return compute_classification_metrics(collection.targets, collection.predictions, class_names)


def build_paired_four_class_rows(
    shared: SharedISICPredictionCollection,
    flat: PredictionCollection,
) -> list[dict[str, object]]:
    if shared.sample_ids != flat.sample_ids:
        raise ValueError("Shared and flat predictions must have identical stable sample order.")
    routing = build_hierarchical_routing(
        shared.stage1_targets,
        shared.stage1_predictions,
        shared.stage2_targets,
        shared.stage2_predictions,
    )
    if not np.array_equal(routing.final_targets, flat.targets):
        raise ValueError("Shared and flat ground-truth labels do not match.")
    rows: list[dict[str, object]] = []
    for index, sample_id in enumerate(shared.sample_ids):
        true_value = int(routing.final_targets[index])
        shared_pred = int(routing.predicted_gate_predictions[index])
        oracle_pred = int(routing.oracle_gate_predictions[index])
        flat_pred = int(flat.predictions[index])
        rows.append(
            {
                "sample_id": sample_id,
                "true_label": true_value,
                "shared_predicted_gate": shared_pred,
                "shared_oracle_gate": oracle_pred,
                "flat_prediction": flat_pred,
                "shared_correct": int(shared_pred == true_value),
                "flat_correct": int(flat_pred == true_value),
                "stage1_target": int(shared.stage1_targets[index]),
                "stage1_prediction": int(shared.stage1_predictions[index]),
                "stage2_target": int(shared.stage2_targets[index]),
                "stage2_prediction": int(shared.stage2_predictions[index]),
            }
        )
    return rows


def write_paired_prediction_csv(path: str | Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("Paired prediction export cannot be empty.")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def count_parameters(model: nn.Module) -> dict[str, int]:
    return {
        "total_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    }


def checkpoint_bytes(spec: FrozenCheckpointSpec, project_root: str | Path) -> int:
    return int(verify_checkpoint_artifact(spec, project_root).stat().st_size)


def benchmark_inference(
    model: nn.Module,
    *,
    device: str | torch.device,
    input_shape: tuple[int, int, int] = (3, 224, 224),
    batch_size: int = 1,
    warmup_iterations: int = 20,
    measured_iterations: int = 100,
) -> dict[str, object]:
    """Matched-hardware latency/throughput/peak-memory hook for Gate04D/E."""
    if batch_size <= 0 or warmup_iterations < 0 or measured_iterations <= 0:
        raise ValueError("Invalid benchmark iteration settings.")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    model.to(resolved_device).eval()
    sample = torch.zeros((batch_size, *input_shape), device=resolved_device)
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            model(sample)
        if resolved_device.type == "cuda":
            torch.cuda.synchronize(resolved_device)
            torch.cuda.reset_peak_memory_stats(resolved_device)
        started = time.perf_counter()
        for _ in range(measured_iterations):
            model(sample)
        if resolved_device.type == "cuda":
            torch.cuda.synchronize(resolved_device)
        elapsed = time.perf_counter() - started
    images = batch_size * measured_iterations
    result: dict[str, object] = {
        "device": str(resolved_device),
        "batch_size": batch_size,
        "warmup_iterations": warmup_iterations,
        "measured_iterations": measured_iterations,
        "latency_ms_per_batch": 1000.0 * elapsed / measured_iterations,
        "latency_ms_per_image": 1000.0 * elapsed / images,
        "throughput_images_per_second": images / elapsed,
        "peak_cuda_memory_bytes": None,
    }
    if resolved_device.type == "cuda":
        result["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated(resolved_device))
    return result


def optional_macs_flops(model: nn.Module, input_shape: tuple[int, int, int] = (3, 224, 224)) -> dict[str, object]:
    """Return MACs/FLOPs when fvcore is installed; otherwise record unsupported."""
    try:
        from fvcore.nn import FlopCountAnalysis  # type: ignore
    except ImportError:
        return {"supported": False, "backend": None, "flops": None, "macs": None}
    model_cpu = model.to("cpu").eval()
    sample = torch.zeros((1, *input_shape), dtype=torch.float32)
    try:
        flops = int(FlopCountAnalysis(model_cpu, sample).total())
    except Exception as exc:  # pragma: no cover - backend/model-version dependent
        return {"supported": False, "backend": "fvcore", "flops": None, "macs": None, "error": str(exc)}
    return {"supported": True, "backend": "fvcore", "flops": flops, "macs": None}


def environment_provenance(device: str | torch.device) -> dict[str, object]:
    resolved_device = torch.device(device)
    payload: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(resolved_device),
    }
    if resolved_device.type == "cuda" and torch.cuda.is_available():
        payload["gpu_name"] = torch.cuda.get_device_name(resolved_device)
        payload["gpu_total_memory_bytes"] = int(torch.cuda.get_device_properties(resolved_device).total_memory)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    payload["git_commit"] = commit
    return payload


def write_json(path: str | Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
