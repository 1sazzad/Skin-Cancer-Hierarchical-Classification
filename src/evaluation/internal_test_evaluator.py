"""Frozen-checkpoint evaluation on the untouched ISIC 2019 internal test split."""

from __future__ import annotations

import csv
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from src.data.dataloaders import DataLoaderConfig, build_stage_dataloaders
from src.evaluation.classification_metrics import compute_classification_metrics
from src.models.efficientnet_baseline import build_efficientnet_b0
from src.utils.reproducibility import seed_everything


@dataclass(frozen=True, slots=True)
class InternalTestEvaluationOutcome:
    """Artifacts produced by one frozen internal-test evaluation."""

    output_directory: Path
    metrics_path: Path
    predictions_path: Path
    summary_path: Path
    task: str
    checkpoint_epoch: int
    test_macro_f1: float


def _mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Expected {key!r} to be a mapping.")
    return value


def _json_dump(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _ordered_class_names(config: Mapping[str, Any]) -> list[str]:
    data = _mapping(config, "data")
    class_to_index = data.get("class_to_index")
    if not isinstance(class_to_index, dict) or len(class_to_index) < 2:
        raise ValueError("Checkpoint config must contain data.class_to_index.")
    ordered = sorted(class_to_index.items(), key=lambda item: int(item[1]))
    indices = [int(index) for _, index in ordered]
    if indices != list(range(len(indices))):
        raise ValueError("Checkpoint class indices must be contiguous from zero.")
    return [str(name) for name, _ in ordered]


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


def _load_frozen_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict):
        raise ValueError("Checkpoint payload must be a mapping.")

    required = {"epoch", "model_state_dict", "validation_metrics", "config", "class_names"}
    missing = sorted(required - set(loaded))
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {missing}")

    config = loaded["config"]
    if not isinstance(config, dict):
        raise ValueError("Checkpoint config must be a mapping.")
    runtime = config.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("sanity_run") is not False:
        raise ValueError("Internal-test evaluation requires a non-sanity full-run checkpoint.")

    run_summary_path = checkpoint_path.parent / "run_summary.json"
    if not run_summary_path.is_file():
        raise FileNotFoundError(
            "run_summary.json must be present beside the frozen checkpoint."
        )
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    if not isinstance(run_summary, dict) or not bool(
        run_summary.get("reportable_as_full_result")
    ):
        raise ValueError("Checkpoint run_summary.json is not marked as a full result.")
    if int(run_summary.get("best_epoch", -1)) != int(loaded["epoch"]):
        raise ValueError("best_checkpoint.pt epoch does not match run_summary.json.")

    expected_names = _ordered_class_names(config)
    checkpoint_names = [str(name) for name in loaded["class_names"]]
    if checkpoint_names != expected_names:
        raise ValueError("Checkpoint class_names do not match data.class_to_index.")

    return loaded


def _string_batch(batch: Mapping[str, object], key: str, count: int) -> list[str]:
    value = batch.get(key)
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, Sequence):
        values = [str(item) for item in value]
    else:
        raise TypeError(f"batch[{key!r}] must contain strings.")
    if len(values) != count:
        raise ValueError(f"batch[{key!r}] length does not match the batch size.")
    return values


def _write_confusion_matrix_csv(
    path: Path,
    class_names: Sequence[str],
    matrix: Sequence[Sequence[int]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", *class_names])
        for class_name, row in zip(class_names, matrix, strict=True):
            writer.writerow([class_name, *row])


def _write_per_class_metrics_csv(
    path: Path,
    class_names: Sequence[str],
    per_class: Mapping[str, Any],
) -> None:
    fieldnames = ["class_name", "precision", "recall", "f1", "support"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for class_name in class_names:
            values = per_class[class_name]
            writer.writerow({"class_name": class_name, **values})


def evaluate_frozen_internal_test(
    checkpoint_path: str | Path,
    *,
    project_root: str | Path,
    output_directory: str | Path,
    device: str | torch.device = "cuda",
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> InternalTestEvaluationOutcome:
    """Evaluate one frozen full-run best checkpoint exactly on internal_test.

    The output directory must not already exist. This intentionally prevents an
    accidental overwrite or silent repetition of the locked internal-test run.
    """

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    project = Path(project_root).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    payload = _load_frozen_checkpoint(checkpoint)
    config = payload["config"]
    experiment = _mapping(config, "experiment")
    data = _mapping(config, "data")
    loader = _mapping(config, "loader")
    model_config = _mapping(config, "model")

    task = str(data.get("task"))
    if task not in {"stage_1", "stage_2"}:
        raise ValueError("Checkpoint task must be stage_1 or stage_2.")
    if model_config.get("architecture") != "efficientnet_b0":
        raise ValueError("Internal evaluator currently supports efficientnet_b0 only.")

    class_names = [str(name) for name in payload["class_names"]]
    seed = int(experiment["seed"])
    seed_everything(seed)

    effective_batch_size = int(loader["batch_size"] if batch_size is None else batch_size)
    effective_workers = int(loader["num_workers"] if num_workers is None else num_workers)
    if effective_batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if effective_workers < 0:
        raise ValueError("num_workers must be non-negative.")

    loader_config = DataLoaderConfig(
        batch_size=effective_batch_size,
        num_workers=effective_workers,
        pin_memory=bool(loader.get("pin_memory", resolved_device.type == "cuda")),
        persistent_workers=bool(loader.get("persistent_workers", False))
        and effective_workers > 0,
        prefetch_factor=int(loader.get("prefetch_factor", 2)),
        drop_last_train=False,
        seed=seed,
    )
    dataloaders = build_stage_dataloaders(
        project / str(data["split_manifest"]),
        project,
        task,
        config=loader_config,
        verify_image_paths=bool(data.get("verify_image_paths", False)),
    )
    test_loader = dataloaders["internal_test"]

    model = build_efficientnet_b0(
        int(model_config["number_of_classes"]),
        pretrained="none",
        dropout_probability=float(model_config.get("dropout_probability", 0.2)),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(resolved_device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    use_amp = resolved_device.type == "cuda"
    total_loss = 0.0
    total_samples = 0
    targets_all: list[torch.Tensor] = []
    probabilities_all: list[torch.Tensor] = []
    metadata: dict[str, list[str]] = {
        "image_id": [],
        "image_path": [],
        "split_group_id": [],
        "file_sha256": [],
    }

    started_at = time.perf_counter()
    with torch.inference_mode():
        for batch in test_loader:
            images = batch.get("image")
            targets = batch.get("target")
            if not isinstance(images, torch.Tensor) or not isinstance(targets, torch.Tensor):
                raise TypeError("Each test batch must contain tensor image and target values.")
            if images.ndim != 4 or targets.ndim != 1 or images.shape[0] != targets.shape[0]:
                raise ValueError("Invalid image/target shapes in internal-test batch.")

            batch_count = int(targets.shape[0])
            images = images.to(resolved_device, non_blocking=True)
            targets_device = targets.to(resolved_device, non_blocking=True)
            with torch.autocast(
                device_type=resolved_device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)
                loss = criterion(logits, targets_device)

            probabilities = torch.softmax(logits, dim=1).cpu()
            total_loss += float(loss.item()) * batch_count
            total_samples += batch_count
            targets_all.append(targets.cpu())
            probabilities_all.append(probabilities)
            for key in metadata:
                metadata[key].extend(_string_batch(batch, key, batch_count))

    elapsed_seconds = time.perf_counter() - started_at
    if total_samples == 0:
        raise ValueError("Internal-test loader produced no samples.")

    targets_tensor = torch.cat(targets_all)
    probabilities_tensor = torch.cat(probabilities_all)
    predictions_tensor = probabilities_tensor.argmax(dim=1)
    metrics = compute_classification_metrics(
        targets_tensor.numpy(),
        predictions_tensor.numpy(),
        class_names,
    )
    metrics["mean_loss"] = total_loss / total_samples

    output.mkdir(parents=True, exist_ok=False)
    metrics_path = output / "internal_test_metrics.json"
    predictions_path = output / "internal_test_predictions.csv"
    summary_path = output / "evaluation_summary.json"
    _json_dump(metrics_path, metrics)
    _write_confusion_matrix_csv(
        output / "confusion_matrix.csv",
        class_names,
        metrics["confusion_matrix"],
    )
    _write_per_class_metrics_csv(
        output / "per_class_metrics.csv",
        class_names,
        metrics["per_class"],
    )

    probability_fields = [f"probability_{name}" for name in class_names]
    prediction_fields = [
        "image_id",
        "image_path",
        "split_group_id",
        "file_sha256",
        "target_index",
        "target_label",
        "predicted_index",
        "predicted_label",
        "correct",
        *probability_fields,
    ]
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_fields)
        writer.writeheader()
        for index in range(total_samples):
            target_index = int(targets_tensor[index])
            predicted_index = int(predictions_tensor[index])
            row: dict[str, object] = {
                key: values[index] for key, values in metadata.items()
            }
            row.update(
                {
                    "target_index": target_index,
                    "target_label": class_names[target_index],
                    "predicted_index": predicted_index,
                    "predicted_label": class_names[predicted_index],
                    "correct": int(target_index == predicted_index),
                }
            )
            for class_index, field_name in enumerate(probability_fields):
                row[field_name] = float(probabilities_tensor[index, class_index])
            writer.writerow(row)

    summary = {
        "evaluation_scope": "isic2019_internal_test",
        "reportable_internal_test_result": True,
        "checkpoint_path": str(checkpoint),
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_validation_metrics": payload["validation_metrics"],
        "run_name": experiment["run_name"],
        "task": task,
        "seed": seed,
        "class_names": class_names,
        "sample_count": total_samples,
        "batch_size": effective_batch_size,
        "num_workers": effective_workers,
        "elapsed_seconds": elapsed_seconds,
        "samples_per_second": total_samples / elapsed_seconds,
        "test_metrics": metrics,
        "environment": _environment_payload(resolved_device),
    }
    _json_dump(summary_path, summary)

    return InternalTestEvaluationOutcome(
        output_directory=output,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        summary_path=summary_path,
        task=task,
        checkpoint_epoch=int(payload["epoch"]),
        test_macro_f1=float(metrics["macro_f1"]),
    )
