from __future__ import annotations

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset

from src.training.engine import run_classification_epoch


class TinyDictionaryDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self) -> None:
        self.images = torch.tensor(
            [
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[1.0, 1.0], [1.0, 1.0]]],
                [[[0.2, 0.2], [0.2, 0.2]]],
                [[[0.8, 0.8], [0.8, 0.8]]],
            ],
            dtype=torch.float32,
        )
        self.targets = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "image": self.images[index],
            "target": self.targets[index],
        }


def test_training_engine_returns_finite_outputs() -> None:
    loader = DataLoader(TinyDictionaryDataset(), batch_size=2, shuffle=False)
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
    optimizer = SGD(model.parameters(), lr=0.1)

    result = run_classification_epoch(
        model,
        loader,
        nn.CrossEntropyLoss(),
        "cpu",
        optimizer=optimizer,
    )

    assert result.sample_count == 4
    assert result.targets.shape == (4,)
    assert result.predictions.shape == (4,)
    assert result.probabilities.shape == (4, 2)
    assert torch.isfinite(result.probabilities).all()
    assert result.mean_loss > 0.0


def test_evaluation_engine_limits_smoke_batches() -> None:
    loader = DataLoader(TinyDictionaryDataset(), batch_size=2, shuffle=False)
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))

    result = run_classification_epoch(
        model,
        loader,
        nn.CrossEntropyLoss(),
        "cpu",
        max_batches=1,
    )

    assert result.sample_count == 2
