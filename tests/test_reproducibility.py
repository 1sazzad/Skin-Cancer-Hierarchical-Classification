from __future__ import annotations

import random

import numpy as np
import torch

from src.utils.reproducibility import make_generator, seed_everything


def test_seed_everything_repeats_python_numpy_and_torch() -> None:
    seed_everything(42)
    first = (random.random(), np.random.rand(), torch.rand(1).item())
    seed_everything(42)
    second = (random.random(), np.random.rand(), torch.rand(1).item())
    assert first == second


def test_make_generator_repeats_sequence() -> None:
    first = torch.randperm(10, generator=make_generator(42))
    second = torch.randperm(10, generator=make_generator(42))
    assert torch.equal(first, second)
