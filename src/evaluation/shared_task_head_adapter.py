"""Expose one frozen shared-model head through the ordinary classifier interface."""

from __future__ import annotations

import torch
from torch import nn


class SharedTaskHeadAdapter(nn.Module):
    """Return one task's logits without modifying or copying model parameters."""

    def __init__(self, shared_model: nn.Module, task_name: str) -> None:
        super().__init__()
        if task_name not in {"task1", "task2", "task3"}:
            raise ValueError("task_name must be task1, task2, or task3.")
        self.shared_model = shared_model
        self.task_name = task_name

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.shared_model(images)
        if not isinstance(outputs, dict) or self.task_name not in outputs:
            raise ValueError("Shared model did not return the requested task logits.")
        logits = outputs[self.task_name]
        if not isinstance(logits, torch.Tensor):
            raise TypeError("Requested shared task output must be a tensor.")
        return logits
