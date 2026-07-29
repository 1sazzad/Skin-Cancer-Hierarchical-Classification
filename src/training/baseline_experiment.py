"""Configuration-driven baseline and controlled imbalance-aware training."""

from __future__ import annotations

import csv
import json
import math
import platform
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

from src.data.dataloaders import DataLoaderConfig, build_stage_dataloaders
from src.evaluation.classification_metrics import compute_classification_metrics
from src.models.efficientnet_baseline import build_efficientnet_b0
from src.training.engine import EpochResult, run_classification_epoch
from src.training.losses import ClassBalancedFocalLoss
from src.utils.reproducibility import seed_everything


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    """Locations and selection information for one completed training run."""

    run_directory: Path
    best_checkpoint_path: Path
    last_checkpoint_path: Path
    best_epoch: int
    best_validation_macro_f1: float
    stopped_early: bool


def load_experiment_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate one runnable controlled experiment configuration."""

    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Experiment configuration not found: {path}")

    loaded = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(loaded, dict):
        raise ValueError("Experiment configuration must be a YAML mapping.")

    required_sections = {
        "experiment",
        "data",
        "loader",
        "model",
        "training",
    }
    missing = sorted(required_sections - set(loaded))
    if missing:
        raise ValueError(f"Experiment configuration is missing sections: {missing}")

    experiment = _mapping(loaded, "experiment")
    data = _mapping(loaded, "data")
    model = _mapping(loaded, "model")
    training = _mapping(loaded, "training")

    if experiment.get("status") != "ready_for_training":
        raise ValueError(
            "Runnable experiment configs must use status='ready_for_training'."
        )
    if not bool(training.get("full_training_allowed")):
        raise ValueError("full_training_allowed must be true for a runnable config.")

    task = data.get("task")
    if task not in {"stage_1", "stage_2", "flat_four_class", "emb_stage03"}:
        raise ValueError(
            "data.task must be stage_1, stage_2, flat_four_class, or emb_stage03."
        )

    if model.get("architecture") != "efficientnet_b0":
        raise ValueError("The controlled runner currently supports efficientnet_b0 only.")
    if model.get("pretrained_weights") != "imagenet":
        raise ValueError("Experiments must use declared ImageNet pretrained weights.")

    class_to_index = data.get("class_to_index")
    if not isinstance(class_to_index, dict) or len(class_to_index) < 2:
        raise ValueError("data.class_to_index must map at least two classes.")

    indices = sorted(int(value) for value in class_to_index.values())
    if indices != list(range(len(indices))):
        raise ValueError("class indices must be contiguous and start at zero.")

    number_of_classes = int(model.get("number_of_classes", -1))
    if number_of_classes != len(class_to_index):
        raise ValueError("model.number_of_classes does not match class_to_index.")

    if task == "flat_four_class":
        expected_mapping = {
            "non_malignant": 0,
            "melanoma": 1,
            "bcc": 2,
            "scc": 3,
        }
        if class_to_index != expected_mapping:
            raise ValueError(
                "flat_four_class requires exact class order "
                "[non_malignant, melanoma, bcc, scc]."
            )
        if data.get("label_source") != "diagnosis_canonical":
            raise ValueError(
                "flat_four_class requires data.label_source='diagnosis_canonical'."
            )
        if data.get("label_mapping_strategy") != "phase06_flat_four_class_v1":
            raise ValueError(
                "flat_four_class requires the phase06_flat_four_class_v1 mapping."
            )

    if task == "emb_stage03":
        expected_mapping = {"Tis": 0, "T1": 1, "T2": 2, "T3": 3, "T4": 4}
        if class_to_index != expected_mapping:
            raise ValueError(
                "emb_stage03 requires exact class order [Tis, T1, T2, T3, T4]."
            )
        if data.get("dataset") != "isic_stage03":
            raise ValueError("emb_stage03 requires data.dataset='isic_stage03'.")
        if data.get("label_source") != "official_isic_metadata":
            raise ValueError(
                "emb_stage03 requires data.label_source='official_isic_metadata'."
            )
        if data.get("label_mapping_strategy") != (
            "official_isic_diagnosis_and_breslow_ajcc8_broad_t_category"
        ):
            raise ValueError("emb_stage03 requires official ISIC label derivation.")
        if data.get("modality") != "dermoscopic":
            raise ValueError(
                "emb_stage03 primary training must be dermoscopic only."
            )

    loss_name = training.get("loss")
    supported_losses = {
        "cross_entropy",
        "weighted_cross_entropy",
        "class_balanced_focal_loss",
    }
    if loss_name not in supported_losses:
        raise ValueError(
            "training.loss must be cross_entropy, weighted_cross_entropy, "
            "or class_balanced_focal_loss."
        )

    if bool(training.get("weighted_sampler")):
        raise ValueError(
            "weighted_sampler must remain false for these controlled variants."
        )
    if training.get("selection_metric") != "macro_f1":
        raise ValueError("Checkpoint selection must use validation macro_f1.")

    class_weights = training.get("class_weights")
    focal_enabled = bool(training.get("focal_loss"))

    def validate_class_weight_mapping() -> dict[str, Any]:
        if not isinstance(class_weights, dict):
            raise ValueError(
                "class_weights must be a class-name-to-weight mapping."
            )

        expected_classes = set(class_to_index)
        provided_classes = set(class_weights)

        if provided_classes != expected_classes:
            missing_classes = sorted(expected_classes - provided_classes)
            unexpected_classes = sorted(provided_classes - expected_classes)
            raise ValueError(
                "class_weights keys must exactly match data.class_to_index; "
                f"missing={missing_classes}, unexpected={unexpected_classes}."
            )

        for class_name, raw_weight in class_weights.items():
            weight = float(raw_weight)
            if not math.isfinite(weight) or weight <= 0.0:
                raise ValueError(
                    f"class weight for {class_name!r} must be finite and positive."
                )

        return class_weights

    if loss_name == "cross_entropy":
        if class_weights is not None:
            raise ValueError(
                "class_weights must remain null for ordinary cross_entropy."
            )
        if focal_enabled:
            raise ValueError(
                "focal_loss must remain false for ordinary cross_entropy."
            )

    elif loss_name == "weighted_cross_entropy":
        if task != "stage_2":
            raise ValueError(
                "weighted_cross_entropy is currently restricted to Stage 2."
            )
        validate_class_weight_mapping()
        if focal_enabled:
            raise ValueError(
                "focal_loss must remain false for weighted_cross_entropy."
            )

    elif loss_name == "class_balanced_focal_loss":
        if task not in {"stage_2", "flat_four_class"}:
            raise ValueError(
                "class_balanced_focal_loss is restricted to Stage 2 and the "
                "flat four-class task."
            )

        validated_weights = validate_class_weight_mapping()

        if not focal_enabled:
            raise ValueError(
                "focal_loss must be true for class_balanced_focal_loss."
            )

        gamma = float(training.get("focal_gamma", -1.0))
        if not math.isfinite(gamma) or gamma < 0.0:
            raise ValueError(
                "focal_gamma must be finite and non-negative."
            )

        source = _mapping(training, "class_weight_source")
        if source.get("partition") != "train":
            raise ValueError(
                "class-balanced weights must use the training partition only."
            )
        if source.get("method") != "effective_number":
            raise ValueError(
                "class_weight_source.method must be effective_number."
            )
        if source.get("normalization") != "sum_to_number_of_classes":
            raise ValueError(
                "class-balanced weights must use sum_to_number_of_classes."
            )

        beta = float(source.get("beta", -1.0))
        if not math.isfinite(beta) or not 0.0 < beta < 1.0:
            raise ValueError("class-balance beta must be between zero and one.")

        class_counts = source.get("class_counts")
        if not isinstance(class_counts, dict):
            raise ValueError(
                "class_weight_source.class_counts must be a mapping."
            )
        if set(class_counts) != set(class_to_index):
            raise ValueError(
                "class count keys must exactly match data.class_to_index."
            )

        calculated_weights = compute_effective_number_class_weights(
            class_counts,
            list(class_to_index),
            beta=beta,
        )

        for class_name, calculated_weight in calculated_weights.items():
            configured_weight = float(validated_weights[class_name])
            if not math.isclose(
                configured_weight,
                calculated_weight,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "Configured class-balanced weight does not match "
                    f"the effective-number formula for {class_name!r}."
                )

    epochs = int(training.get("epochs", 0))
    patience = int(training.get("early_stopping_patience", 0))

    if epochs <= 0:
        raise ValueError("training.epochs must be positive.")
    if task == "emb_stage03" and epochs > 30:
        raise ValueError("emb_stage03 is capped at 30 epochs.")
    if task == "emb_stage03" and int(experiment.get("seed", -1)) != 42:
        raise ValueError("emb_stage03 requires seed 42.")
    if patience <= 0:
        raise ValueError("training.early_stopping_patience must be positive.")

    return deepcopy(loaded)


def compute_effective_number_class_weights(
    class_counts: Mapping[str, Any],
    class_order: list[str],
    *,
    beta: float,
) -> dict[str, float]:
    """Compute the repository-standard sum-to-class-count effective weights."""

    if len(class_counts) != len(class_order) or set(class_counts) != set(class_order):
        raise ValueError(
            "class count keys and class order must contain the same classes."
        )
    resolved_beta = float(beta)
    if not math.isfinite(resolved_beta) or not 0.0 < resolved_beta < 1.0:
        raise ValueError("class-balance beta must be between zero and one.")

    raw_weights: dict[str, float] = {}
    for class_name in class_order:
        count = int(class_counts[class_name])
        if count <= 0:
            raise ValueError("All training class counts must be positive.")
        raw_weights[class_name] = (
            (1.0 - resolved_beta) / (1.0 - resolved_beta ** count)
        )

    normalizer = len(raw_weights) / sum(raw_weights.values())
    return {
        class_name: raw_weights[class_name] * normalizer
        for class_name in class_order
    }


def _mapping(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section {key!r} must be a mapping.")
    return value


def _ordered_class_names(config: Mapping[str, Any]) -> list[str]:
    data = _mapping(config, "data")
    class_to_index = data["class_to_index"]
    assert isinstance(class_to_index, dict)
    return [
        name
        for name, _ in sorted(
            class_to_index.items(),
            key=lambda item: int(item[1]),
        )
    ]


def _build_criterion(
    config: Mapping[str, Any],
    device: torch.device | str,
) -> nn.Module:
    """Build the declared controlled classification loss."""

    training = _mapping(config, "training")
    loss_name = str(training["loss"])
    resolved_device = torch.device(device)

    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()

    if loss_name in {
        "weighted_cross_entropy",
        "class_balanced_focal_loss",
    }:
        class_weights = training["class_weights"]
        assert isinstance(class_weights, dict)

        ordered_class_names = _ordered_class_names(config)
        weight_tensor = torch.tensor(
            [float(class_weights[name]) for name in ordered_class_names],
            dtype=torch.float32,
            device=resolved_device,
        )

        if loss_name == "weighted_cross_entropy":
            return nn.CrossEntropyLoss(weight=weight_tensor)

        return ClassBalancedFocalLoss(
            weight_tensor,
            gamma=float(training["focal_gamma"]),
        )

    raise ValueError(f"Unsupported loss configuration: {loss_name!r}.")


def _build_optimizer(model: nn.Module, config: Mapping[str, Any]) -> Optimizer:
    training = _mapping(config, "training")
    optimizer_config = _mapping(training, "optimizer")
    if optimizer_config.get("name") != "adamw":
        raise ValueError("The Phase 03 clean baseline currently supports adamw only.")
    learning_rate = float(optimizer_config.get("learning_rate", 0.0))
    weight_decay = float(optimizer_config.get("weight_decay", -1.0))
    if learning_rate <= 0.0:
        raise ValueError("optimizer.learning_rate must be positive.")
    if weight_decay < 0.0:
        raise ValueError("optimizer.weight_decay must be non-negative.")
    return AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def _build_scheduler(
    optimizer: Optimizer,
    config: Mapping[str, Any],
) -> LRScheduler:
    training = _mapping(config, "training")
    scheduler_config = _mapping(training, "scheduler")
    if scheduler_config.get("name") != "cosine_annealing":
        raise ValueError("The Phase 03 clean baseline supports cosine_annealing only.")
    epochs = int(training["epochs"])
    minimum_learning_rate = float(
        scheduler_config.get("minimum_learning_rate", 0.0)
    )
    if minimum_learning_rate < 0.0:
        raise ValueError("scheduler.minimum_learning_rate must be non-negative.")
    return CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=minimum_learning_rate,
    )


def _make_run_directory(
    output_root: Path,
    run_name: str,
    *,
    sanity_run: bool,
) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    prefix = "sanity" if sanity_run else "full"
    run_directory = output_root / f"{prefix}__{run_name}__{timestamp}"
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


def _json_dump(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_history(run_directory: Path, history: list[dict[str, Any]]) -> None:
    _json_dump(run_directory / "history.json", history)
    if not history:
        return
    with (run_directory / "history.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


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


def _checkpoint_payload(
    *,
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    validation_metrics: Mapping[str, Any],
    config: Mapping[str, Any],
    class_names: list[str],
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "validation_metrics": dict(validation_metrics),
        "config": deepcopy(dict(config)),
        "class_names": list(class_names),
    }


def _epoch_record(
    *,
    epoch: int,
    learning_rate: float,
    train_result: EpochResult,
    validation_result: EpochResult,
    train_metrics: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
    epoch_seconds: float,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "learning_rate": learning_rate,
        "train_loss": train_result.mean_loss,
        "train_accuracy": train_metrics["accuracy"],
        "train_balanced_accuracy": train_metrics["balanced_accuracy"],
        "train_macro_f1": train_metrics["macro_f1"],
        "validation_loss": validation_result.mean_loss,
        "validation_accuracy": validation_metrics["accuracy"],
        "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
        "validation_macro_f1": validation_metrics["macro_f1"],
        "epoch_seconds": epoch_seconds,
    }


def run_baseline_experiment(
    config_path: str | Path,
    *,
    project_root: str | Path,
    output_root: str | Path,
    device: str | torch.device = "cuda",
    max_train_batches: int | None = None,
    max_validation_batches: int | None = None,
    epoch_limit: int | None = None,
) -> TrainingOutcome:
    """Train one Stage 1 or Stage 2 clean baseline using validation only.

    ``max_*_batches`` and ``epoch_limit`` create a clearly labelled sanity run.
    A sanity run is never eligible for paper reporting or checkpoint freezing.
    The internal-test loader is never iterated by this function.
    """

    config = load_experiment_config(config_path)
    project_root_path = Path(project_root).expanduser().resolve()
    output_root_path = Path(output_root).expanduser().resolve()
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    experiment = _mapping(config, "experiment")
    data = _mapping(config, "data")
    loader = _mapping(config, "loader")
    model_config = _mapping(config, "model")
    training = _mapping(config, "training")

    seed = int(experiment["seed"])
    seed_everything(seed)
    sanity_run = any(
        value is not None
        for value in (max_train_batches, max_validation_batches, epoch_limit)
    )
    run_directory = _make_run_directory(
        output_root_path,
        str(experiment["run_name"]),
        sanity_run=sanity_run,
    )

    config["runtime"] = {
        "config_path": str(Path(config_path).expanduser().resolve()),
        "project_root": str(project_root_path),
        "output_root": str(output_root_path),
        "device": str(resolved_device),
        "sanity_run": sanity_run,
        "max_train_batches": max_train_batches,
        "max_validation_batches": max_validation_batches,
        "epoch_limit": epoch_limit,
    }
    (run_directory / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    _json_dump(run_directory / "environment.json", _environment_payload(resolved_device))

    loader_config = DataLoaderConfig(
        batch_size=int(loader["batch_size"]),
        num_workers=int(loader["num_workers"]),
        pin_memory=bool(loader["pin_memory"]),
        persistent_workers=bool(loader["persistent_workers"]),
        prefetch_factor=int(loader.get("prefetch_factor", 2)),
        drop_last_train=bool(loader.get("drop_last_train", False)),
        seed=seed,
    )
    dataloaders = build_stage_dataloaders(
        project_root_path / str(data["split_manifest"]),
        project_root_path,
        str(data["task"]),
        config=loader_config,
        verify_image_paths=bool(data.get("verify_image_paths", False)),
    )

    class_names = _ordered_class_names(config)
    model = build_efficientnet_b0(
        int(model_config["number_of_classes"]),
        pretrained="imagenet",
        dropout_probability=float(model_config.get("dropout_probability", 0.2)),
    ).to(resolved_device)
    criterion = _build_criterion(config, resolved_device)
    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, config)
    amp_enabled = bool(training.get("amp", True))
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled and resolved_device.type == "cuda",
    )

    configured_epochs = int(training["epochs"])
    epochs = configured_epochs if epoch_limit is None else min(configured_epochs, epoch_limit)
    if epochs <= 0:
        raise ValueError("epoch_limit must be positive when provided.")
    patience = int(training["early_stopping_patience"])

    history: list[dict[str, Any]] = []
    best_metric = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    best_checkpoint_path = run_directory / "best_checkpoint.pt"
    last_checkpoint_path = run_directory / "last_checkpoint.pt"
    started_at = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_started_at = time.perf_counter()
        learning_rate = float(optimizer.param_groups[0]["lr"])

        train_result = run_classification_epoch(
            model,
            dataloaders["train"],
            criterion,
            resolved_device,
            optimizer=optimizer,
            gradient_scaler=scaler,
            amp_enabled=amp_enabled,
            max_batches=max_train_batches,
        )
        validation_result = run_classification_epoch(
            model,
            dataloaders["validation"],
            criterion,
            resolved_device,
            amp_enabled=amp_enabled,
            max_batches=max_validation_batches,
        )

        train_metrics = compute_classification_metrics(
            train_result.targets.numpy(),
            train_result.predictions.numpy(),
            class_names,
        )
        validation_metrics = compute_classification_metrics(
            validation_result.targets.numpy(),
            validation_result.predictions.numpy(),
            class_names,
        )
        epoch_seconds = time.perf_counter() - epoch_started_at
        record = _epoch_record(
            epoch=epoch,
            learning_rate=learning_rate,
            train_result=train_result,
            validation_result=validation_result,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            epoch_seconds=epoch_seconds,
        )
        history.append(record)
        _write_history(run_directory, history)
        _json_dump(
            run_directory / f"validation_metrics_epoch_{epoch:03d}.json",
            validation_metrics,
        )

        current_metric = float(validation_metrics["macro_f1"])
        improved = current_metric > best_metric
        if improved:
            best_metric = current_metric
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                _checkpoint_payload(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    validation_metrics=validation_metrics,
                    config=config,
                    class_names=class_names,
                ),
                best_checkpoint_path,
            )
            _json_dump(
                run_directory / "best_validation_metrics.json",
                validation_metrics,
            )
        else:
            epochs_without_improvement += 1

        scheduler.step()
        torch.save(
            _checkpoint_payload(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                validation_metrics=validation_metrics,
                config=config,
                class_names=class_names,
            ),
            last_checkpoint_path,
        )

        print(
            f"epoch={epoch:03d}/{epochs:03d} "
            f"train_loss={train_result.mean_loss:.6f} "
            f"val_loss={validation_result.mean_loss:.6f} "
            f"val_macro_f1={current_metric:.6f} "
            f"best={best_metric:.6f}"
        )

        if not sanity_run and epochs_without_improvement >= patience:
            stopped_early = True
            break

    total_seconds = time.perf_counter() - started_at
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    summary = {
        "run_name": experiment["run_name"],
        "task": data["task"],
        "class_names": class_names,
        "class_to_index": deepcopy(data["class_to_index"]),
        "label_source": data.get("label_source"),
        "label_mapping_strategy": data.get("label_mapping_strategy"),
        "selection_metric": training["selection_metric"],
        "loss": training["loss"],
        "class_weights": deepcopy(training.get("class_weights")),
        "class_weight_source": deepcopy(training.get("class_weight_source")),
        "focal_gamma": training.get("focal_gamma"),
        "sanity_run": sanity_run,
        "reportable_as_full_result": not sanity_run,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_metric,
        "completed_epochs": len(history),
        "configured_epochs": configured_epochs,
        "stopped_early": stopped_early,
        "total_training_seconds": total_seconds,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "best_checkpoint_bytes": best_checkpoint_path.stat().st_size,
        "last_checkpoint_bytes": last_checkpoint_path.stat().st_size,
    }
    _json_dump(run_directory / "run_summary.json", summary)

    return TrainingOutcome(
        run_directory=run_directory,
        best_checkpoint_path=best_checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
        best_epoch=best_epoch,
        best_validation_macro_f1=best_metric,
        stopped_early=stopped_early,
    )
