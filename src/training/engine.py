"""Reusable classification training and evaluation engine."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
import torch.nn as nn
from torch.optim import Optimizer


@dataclass(frozen=True, slots=True)
class EpochResult:
    """CPU outputs from one finite training or evaluation pass."""

    mean_loss: float
    sample_count: int
    targets: torch.Tensor
    predictions: torch.Tensor
    probabilities: torch.Tensor


def _validate_batch(batch: Mapping[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
    if "image" not in batch or "target" not in batch:
        raise KeyError("Each batch must contain 'image' and 'target'.")

    images = batch["image"]
    targets = batch["target"]
    if not isinstance(images, torch.Tensor):
        raise TypeError("batch['image'] must be a torch.Tensor.")
    if not isinstance(targets, torch.Tensor):
        raise TypeError("batch['target'] must be a torch.Tensor.")
    if images.ndim != 4:
        raise ValueError("batch['image'] must have shape [N, C, H, W].")
    if targets.ndim != 1:
        raise ValueError("batch['target'] must have shape [N].")
    if images.shape[0] != targets.shape[0]:
        raise ValueError("Image and target batch sizes do not match.")
    return images, targets


def run_classification_epoch(
    model: nn.Module,
    dataloader: Iterable[Mapping[str, object]],
    criterion: nn.Module,
    device: torch.device | str,
    *,
    optimizer: Optimizer | None = None,
    gradient_scaler: torch.amp.GradScaler | None = None,
    amp_enabled: bool = False,
    max_batches: int | None = None,
) -> EpochResult:
    """Run one classification pass.

    Passing an optimizer enables training. Omitting it performs inference-only
    evaluation. CUDA automatic mixed precision is enabled only when requested
    and the resolved device is CUDA. ``max_batches`` is reserved for smoke and
    sanity runs and must not be used for reported full-training results.
    """

    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when provided.")

    resolved_device = torch.device(device)
    training = optimizer is not None
    use_amp = bool(amp_enabled and resolved_device.type == "cuda")
    if gradient_scaler is not None and not training:
        raise ValueError("gradient_scaler is only valid when optimizer is provided.")
    if use_amp and training and gradient_scaler is None:
        raise ValueError("CUDA AMP training requires a gradient_scaler.")

    model.train(training)

    total_loss = 0.0
    total_samples = 0
    target_chunks: list[torch.Tensor] = []
    probability_chunks: list[torch.Tensor] = []

    gradient_context = nullcontext() if training else torch.inference_mode()
    with gradient_context:
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break

            images, targets = _validate_batch(batch)
            images = images.to(resolved_device, non_blocking=True)
            targets = targets.to(resolved_device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=resolved_device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)
                if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
                    raise ValueError("Model output must be a rank-2 logits tensor.")
                if logits.shape[0] != targets.shape[0]:
                    raise ValueError("Logit and target batch sizes do not match.")

                loss = criterion(logits, targets)
                if loss.ndim != 0 or not torch.isfinite(loss):
                    raise ValueError("Criterion must return one finite scalar loss.")

            if training:
                if use_amp:
                    assert gradient_scaler is not None
                    gradient_scaler.scale(loss).backward()
                    gradient_scaler.step(optimizer)
                    gradient_scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            batch_size = int(targets.shape[0])
            total_loss += float(loss.detach().item()) * batch_size
            total_samples += batch_size
            target_chunks.append(targets.detach().cpu())
            probability_chunks.append(torch.softmax(logits.detach(), dim=1).cpu())

    if total_samples == 0:
        raise ValueError("The dataloader produced no samples.")

    all_targets = torch.cat(target_chunks)
    all_probabilities = torch.cat(probability_chunks)
    predictions = all_probabilities.argmax(dim=1)
    return EpochResult(
        mean_loss=total_loss / total_samples,
        sample_count=total_samples,
        targets=all_targets,
        predictions=predictions,
        probabilities=all_probabilities,
    )
