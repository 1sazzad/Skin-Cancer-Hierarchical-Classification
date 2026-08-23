"""Masked optimization and validation primitives for the shared baseline."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

from src.evaluation.classification_metrics import compute_classification_metrics
from src.models.shared_three_task import TASK_CLASS_MAPPINGS
from src.training.losses import ClassBalancedFocalLoss

TASK_NAMES = ("task1", "task2", "task3")


@dataclass(frozen=True, slots=True)
class MaskedLossResult:
    """Normalized total loss plus optional per-task mean losses."""

    total_loss: torch.Tensor
    task_losses: dict[str, torch.Tensor | None]
    active_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class SharedTrainEpochResult:
    """Mean losses recorded from one natural-pool training epoch."""

    train_total_loss: float
    train_task1_loss: float | None
    train_task2_loss: float | None
    train_task3_loss: float | None
    sample_count: int
    active_counts: dict[str, int]


@dataclass(slots=True)
class SharedValidationEarlyStopping:
    """Maximize shared validation score with the frozen patience."""

    patience: int = 7
    best_score: float | None = None
    best_epoch: int | None = None
    epochs_without_improvement: int = 0

    def __post_init__(self) -> None:
        if self.patience <= 0:
            raise ValueError("patience must be positive.")

    def update(self, score: float, epoch: int) -> tuple[bool, bool]:
        """Return improved and should-stop flags for one validation."""

        resolved_score = float(score)
        if not 0.0 <= resolved_score <= 1.0:
            raise ValueError("shared validation score must be in [0, 1].")
        if epoch <= 0:
            raise ValueError("epoch must be positive.")
        improved = self.best_score is None or resolved_score > self.best_score
        if improved:
            self.best_score = resolved_score
            self.best_epoch = int(epoch)
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        return improved, self.epochs_without_improvement >= self.patience


class MaskedThreeTaskLoss(nn.Module):
    """Compute each loss only after selecting samples with an active mask."""

    def __init__(
        self,
        task2_class_weights: torch.Tensor,
        task3_class_weights: torch.Tensor,
        *,
        task2_gamma: float = 2.0,
        task_weights: Sequence[float] = (1.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        resolved_weights = tuple(float(value) for value in task_weights)
        if len(resolved_weights) != 3 or any(value <= 0 for value in resolved_weights):
            raise ValueError("task_weights must contain three positive values.")
        if task3_class_weights.shape != (5,):
            raise ValueError("task3_class_weights must contain five values.")
        if not torch.isfinite(task3_class_weights).all() or not torch.all(
            task3_class_weights > 0
        ):
            raise ValueError("task3_class_weights must be finite and positive.")

        self.task1_criterion = nn.CrossEntropyLoss()
        self.task2_criterion = ClassBalancedFocalLoss(
            task2_class_weights,
            gamma=task2_gamma,
        )
        self.task3_criterion = nn.CrossEntropyLoss(
            weight=task3_class_weights.detach().clone().to(torch.float32)
        )
        self.task_weights = resolved_weights

    def forward(
        self,
        logits: Mapping[str, torch.Tensor],
        targets: torch.Tensor,
        task_mask: torch.Tensor,
    ) -> MaskedLossResult:
        if targets.ndim != 2 or targets.shape[1] != 3:
            raise ValueError("targets must have shape [batch, 3].")
        if task_mask.shape != targets.shape or task_mask.dtype != torch.bool:
            raise ValueError("task_mask must be boolean with shape [batch, 3].")
        if targets.dtype != torch.long:
            raise ValueError("targets must use torch.long dtype.")
        if set(logits) != set(TASK_NAMES):
            raise ValueError(f"logits must contain exactly {TASK_NAMES}.")

        criteria = (
            self.task1_criterion,
            self.task2_criterion,
            self.task3_criterion,
        )
        expected_classes = (2, 3, 5)
        task_losses: dict[str, torch.Tensor | None] = {}
        active_counts: dict[str, int] = {}
        weighted_losses: list[torch.Tensor] = []
        active_weight_sum = 0.0

        for index, (task_name, criterion, class_count, task_weight) in enumerate(
            zip(
                TASK_NAMES,
                criteria,
                expected_classes,
                self.task_weights,
                strict=True,
            )
        ):
            task_logits = logits[task_name]
            if task_logits.ndim != 2 or task_logits.shape != (
                targets.shape[0],
                class_count,
            ):
                raise ValueError(
                    f"{task_name} logits must have shape "
                    f"[{targets.shape[0]}, {class_count}]."
                )
            active = task_mask[:, index]
            active_count = int(active.sum().item())
            active_counts[task_name] = active_count
            if active_count == 0:
                task_losses[task_name] = None
                continue

            # Selection occurs before the criterion sees targets. Missing
            # sentinels in inactive rows can therefore never become class 0.
            active_logits = task_logits[active]
            active_targets = targets[active, index]
            if torch.any(active_targets < 0) or torch.any(
                active_targets >= class_count
            ):
                raise ValueError(f"{task_name} has an invalid active target.")
            loss = criterion(active_logits, active_targets)
            task_losses[task_name] = loss
            weighted_losses.append(loss * task_weight)
            active_weight_sum += task_weight

        if not weighted_losses:
            raise ValueError("A batch must activate at least one task.")
        total_loss = torch.stack(weighted_losses).sum() / active_weight_sum
        return MaskedLossResult(total_loss, task_losses, active_counts)


def run_shared_training_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: MaskedThreeTaskLoss,
    optimizer: Optimizer,
    device: torch.device | str,
    *,
    gradient_scaler: torch.amp.GradScaler | None = None,
    amp_enabled: bool = False,
) -> SharedTrainEpochResult:
    """Run one training epoch over the naturally shuffled combined pool."""

    resolved_device = torch.device(device)
    use_amp = bool(amp_enabled and resolved_device.type == "cuda")
    if use_amp and gradient_scaler is None:
        raise ValueError("CUDA AMP training requires a gradient scaler.")
    model.train()
    total_loss_sum = 0.0
    total_samples = 0
    task_loss_sums = {task: 0.0 for task in TASK_NAMES}
    active_totals = {task: 0 for task in TASK_NAMES}

    for batch in dataloader:
        images = batch["image"].to(resolved_device, non_blocking=True)
        targets = batch["targets"].to(resolved_device, non_blocking=True)
        task_mask = batch["task_mask"].to(resolved_device, non_blocking=True)
        if images.ndim != 4 or images.shape[0] != targets.shape[0]:
            raise ValueError("Invalid shared training batch shapes.")

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=resolved_device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            result = criterion(model(images), targets, task_mask)
            if not torch.isfinite(result.total_loss):
                raise ValueError("Shared total loss must be finite.")

        if use_amp:
            assert gradient_scaler is not None
            gradient_scaler.scale(result.total_loss).backward()
            gradient_scaler.step(optimizer)
            gradient_scaler.update()
        else:
            result.total_loss.backward()
            optimizer.step()

        batch_size = int(images.shape[0])
        total_loss_sum += float(result.total_loss.detach().item()) * batch_size
        total_samples += batch_size
        for task_name in TASK_NAMES:
            count = result.active_counts[task_name]
            active_totals[task_name] += count
            task_loss = result.task_losses[task_name]
            if task_loss is not None:
                task_loss_sums[task_name] += float(task_loss.detach().item()) * count

    if total_samples == 0:
        raise ValueError("The shared training dataloader produced no samples.")

    task_means = {
        task_name: (
            task_loss_sums[task_name] / active_totals[task_name]
            if active_totals[task_name] > 0
            else None
        )
        for task_name in TASK_NAMES
    }
    return SharedTrainEpochResult(
        train_total_loss=total_loss_sum / total_samples,
        train_task1_loss=task_means["task1"],
        train_task2_loss=task_means["task2"],
        train_task3_loss=task_means["task3"],
        sample_count=total_samples,
        active_counts=active_totals,
    )


def shared_validation_score(
    task1_macro_f1: float,
    task2_macro_f1: float,
    task3_macro_f1: float,
) -> float:
    """Return the unweighted arithmetic mean of three validation Macro-F1s."""

    values = (
        float(task1_macro_f1),
        float(task2_macro_f1),
        float(task3_macro_f1),
    )
    if not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError("Validation Macro-F1 values must be in [0, 1].")
    return sum(values) / 3.0


def build_masked_loss_from_config(
    config: Mapping[str, Any],
    device: torch.device | str,
) -> MaskedThreeTaskLoss:
    """Build the three frozen criteria from explicit config weights."""

    loss_config = config["task_losses"]
    task2 = loss_config["task2"]
    task3 = loss_config["task3"]
    resolved_device = torch.device(device)
    task2_weights = torch.tensor(
        [float(task2["class_weights"][name]) for name in ("melanoma", "bcc", "scc")],
        dtype=torch.float32,
        device=resolved_device,
    )
    task3_weights = torch.tensor(
        [
            float(task3["class_weights"][name])
            for name in ("Tis", "T1", "T2", "T3", "T4")
        ],
        dtype=torch.float32,
        device=resolved_device,
    )
    return MaskedThreeTaskLoss(
        task2_weights,
        task3_weights,
        task2_gamma=float(task2["gamma"]),
        task_weights=(
            float(loss_config["lambda_task1"]),
            float(loss_config["lambda_task2"]),
            float(loss_config["lambda_task3"]),
        ),
    )


def build_optimizer_and_scheduler(
    model: nn.Module,
    config: Mapping[str, Any],
) -> tuple[Optimizer, LRScheduler]:
    """Build frozen AdamW and 30-horizon cosine annealing components."""

    training = config["training"]
    optimizer_config = training["optimizer"]
    scheduler_config = training["scheduler"]
    if optimizer_config["name"] != "adamw":
        raise ValueError("Shared baseline optimizer must be adamw.")
    if scheduler_config["name"] != "cosine_annealing":
        raise ValueError("Shared baseline scheduler must be cosine_annealing.")
    epochs = int(training["epochs"])
    optimizer = AdamW(
        model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(scheduler_config["minimum_learning_rate"]),
    )
    return optimizer, scheduler


def validate_task_macro_f1(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    task_name: str,
    device: torch.device | str,
) -> float:
    """Evaluate one validation cohort, filtering by its task mask."""

    if task_name not in TASK_NAMES:
        raise ValueError(f"Unknown task name: {task_name!r}.")
    task_index = TASK_NAMES.index(task_name)
    class_names = tuple(TASK_CLASS_MAPPINGS[task_name])
    targets: list[int] = []
    predictions: list[int] = []
    resolved_device = torch.device(device)
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(resolved_device)
            batch_targets = batch["targets"]
            active = batch["task_mask"][:, task_index]
            if not bool(active.any()):
                continue
            task_logits = model(images)[task_name].detach().cpu()
            targets.extend(batch_targets[active, task_index].tolist())
            predictions.extend(task_logits[active].argmax(dim=1).tolist())
    metrics = compute_classification_metrics(targets, predictions, class_names)
    return float(metrics["macro_f1"])


def validate_shared_model(
    model: nn.Module,
    validation_loaders: Mapping[str, torch.utils.data.DataLoader],
    device: torch.device | str,
) -> dict[str, float]:
    """Calculate three validation Macro-F1s and their shared score."""

    scores = {
        task_name: validate_task_macro_f1(
            model,
            validation_loaders[task_name],
            task_name,
            device,
        )
        for task_name in TASK_NAMES
    }
    return {
        "val_task1_macro_f1": scores["task1"],
        "val_task2_macro_f1": scores["task2"],
        "val_task3_macro_f1": scores["task3"],
        "shared_validation_score": shared_validation_score(
            scores["task1"], scores["task2"], scores["task3"]
        ),
    }


def build_checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    epoch: int,
    validation_metrics: Mapping[str, float],
    seed: int,
    model_metadata: Mapping[str, Any],
    config_metadata: Mapping[str, Any],
    task_loss_configuration: Mapping[str, Any],
    task_mask_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the complete best-checkpoint payload for later Gate 03 execution."""

    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": int(epoch),
        "shared_validation_score": float(
            validation_metrics["shared_validation_score"]
        ),
        "task1_val_macro_f1": float(validation_metrics["val_task1_macro_f1"]),
        "task2_val_macro_f1": float(validation_metrics["val_task2_macro_f1"]),
        "task3_val_macro_f1": float(validation_metrics["val_task3_macro_f1"]),
        "seed": int(seed),
        "model_metadata": dict(model_metadata),
        "config_metadata": dict(config_metadata),
        "class_mappings": {
            task: dict(mapping) for task, mapping in TASK_CLASS_MAPPINGS.items()
        },
        "task_loss_configuration": dict(task_loss_configuration),
        "task_mask_policy": dict(task_mask_policy),
    }


def save_best_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write the selected checkpoint to the expected filename."""

    resolved = Path(path)
    if resolved.name != "best_checkpoint.pt":
        raise ValueError("Primary shared checkpoint must be named best_checkpoint.pt.")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), resolved)


def write_training_history(
    run_directory: str | Path,
    history: Sequence[Mapping[str, Any]],
    run_summary: Mapping[str, Any],
) -> None:
    """Write the required machine-readable history and run summary."""

    directory = Path(run_directory)
    directory.mkdir(parents=True, exist_ok=True)
    records = [dict(record) for record in history]
    required = {
        "train_total_loss",
        "train_task1_loss",
        "train_task2_loss",
        "train_task3_loss",
        "val_task1_macro_f1",
        "val_task2_macro_f1",
        "val_task3_macro_f1",
        "shared_validation_score",
        "learning_rate",
    }
    if records and any(not required.issubset(record) for record in records):
        raise ValueError("Every history record must contain required shared metrics.")
    (directory / "history.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if records:
        with (directory / "history.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    (directory / "run_summary.json").write_text(
        json.dumps(dict(run_summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
