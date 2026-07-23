"""Randomness controls for repeatable local and Azure experiments."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch and configure deterministic execution."""

    if seed < 0:
        raise ValueError("seed must be non-negative.")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy inside a PyTorch DataLoader worker."""

    del worker_id  # The worker-specific seed is already encoded by PyTorch.
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Create a seeded CPU generator for DataLoader sampling order."""

    if seed < 0:
        raise ValueError("seed must be non-negative.")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
