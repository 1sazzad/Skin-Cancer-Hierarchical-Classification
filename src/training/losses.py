"""Loss functions for controlled classification experiments."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ClassBalancedFocalLoss(nn.Module):
    """Multiclass focal loss with explicit class-balanced weights.

    For each sample:

        loss = -alpha_y * (1 - p_y) ** gamma * log(p_y)

    The supplied class weights must already be calculated and normalized from
    training-partition class counts.
    """

    def __init__(
        self,
        class_weights: torch.Tensor,
        *,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()

        if class_weights.ndim != 1 or class_weights.numel() < 2:
            raise ValueError(
                "class_weights must be a rank-1 tensor with at least two values."
            )
        if not torch.isfinite(class_weights).all():
            raise ValueError("class_weights must contain only finite values.")
        if not torch.all(class_weights > 0):
            raise ValueError("class_weights must contain only positive values.")

        resolved_gamma = float(gamma)
        if not math.isfinite(resolved_gamma) or resolved_gamma < 0.0:
            raise ValueError("gamma must be finite and non-negative.")

        self.register_buffer(
            "class_weights",
            class_weights.detach().clone().to(dtype=torch.float32),
        )
        self.gamma = resolved_gamma

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError("logits must be a rank-2 tensor.")
        if targets.ndim != 1:
            raise ValueError("targets must be a rank-1 tensor.")
        if logits.shape[0] != targets.shape[0]:
            raise ValueError("logits and targets must have matching batch sizes.")
        if logits.shape[1] != self.class_weights.numel():
            raise ValueError(
                "logit class dimension does not match class_weights."
            )
        if targets.dtype != torch.long:
            raise ValueError("targets must use torch.long dtype.")

        log_probabilities = F.log_softmax(logits, dim=1)
        probabilities = log_probabilities.exp()

        target_indices = targets.unsqueeze(1)
        target_log_probabilities = (
            log_probabilities.gather(1, target_indices).squeeze(1)
        )
        target_probabilities = (
            probabilities.gather(1, target_indices).squeeze(1)
        )
        target_weights = self.class_weights[targets]

        modulation = (1.0 - target_probabilities).pow(self.gamma)
        losses = (
            -target_weights
            * modulation
            * target_log_probabilities
        )

        return losses.mean()
