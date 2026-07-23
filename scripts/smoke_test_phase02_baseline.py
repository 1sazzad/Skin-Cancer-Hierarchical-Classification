"""Run one real-data forward/backward step for both prepared baselines."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.optim import SGD

from src.data.dataloaders import DataLoaderConfig, build_stage_dataloaders
from src.models.efficientnet_baseline import build_efficientnet_b0
from src.training.engine import run_classification_epoch
from src.utils.reproducibility import seed_everything

_STAGE_CLASS_COUNTS = {"stage_1": 2, "stage_2": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/isic2019_train_val_test_split_seed42.csv"),
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")

    seed_everything(args.seed)
    device = torch.device("cpu")
    print("Phase 02 baseline one-batch CPU smoke test")
    print("------------------------------------------")

    for stage, number_of_classes in _STAGE_CLASS_COUNTS.items():
        loaders = build_stage_dataloaders(
            args.manifest,
            args.project_root,
            stage,
            config=DataLoaderConfig(
                batch_size=args.batch_size,
                num_workers=0,
                pin_memory=False,
                seed=args.seed,
            ),
            verify_image_paths=True,
        )
        model = build_efficientnet_b0(
            number_of_classes,
            pretrained="none",
        ).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr=1e-3)

        result = run_classification_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            max_batches=1,
        )

        if result.probabilities.shape != (
            result.sample_count,
            number_of_classes,
        ):
            raise RuntimeError(f"Unexpected probability shape for {stage}.")

        print(
            f"{stage}: samples={result.sample_count} "
            f"loss={result.mean_loss:.6f} "
            f"probabilities={tuple(result.probabilities.shape)}"
        )

        del loaders, model, criterion, optimizer, result
        gc.collect()

    print("Phase 02 baseline smoke test passed.")
    print("No checkpoint or training run was created.")


if __name__ == "__main__":
    main()
