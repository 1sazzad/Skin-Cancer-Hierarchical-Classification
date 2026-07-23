"""CPU smoke test for the real Stage 1 and Stage 2 data pipelines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data.dataloaders import DataLoaderConfig, build_stage_dataloaders
from src.utils.reproducibility import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/isic2019_train_val_test_split_seed42.csv"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(42)
    config = DataLoaderConfig(
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=False,
        seed=42,
    )

    for stage in ("stage_1", "stage_2"):
        loaders = build_stage_dataloaders(
            args.manifest,
            args.project_root,
            stage,
            config=config,
            verify_image_paths=False,
        )
        print(f"\n{stage}")
        for split, loader in loaders.items():
            batch = next(iter(loader))
            images = batch["image"]
            targets = batch["target"]
            if images.ndim != 4 or images.shape[1:] != (3, 224, 224):
                raise RuntimeError(
                    f"Unexpected image batch shape for {stage}/{split}: "
                    f"{tuple(images.shape)}"
                )
            if images.dtype != torch.float32 or targets.dtype != torch.int64:
                raise RuntimeError(
                    f"Unexpected dtypes for {stage}/{split}: "
                    f"images={images.dtype}, targets={targets.dtype}"
                )
            print(
                f"  {split:13s} samples={len(loader.dataset):5d} "
                f"batch={tuple(images.shape)} targets={targets.tolist()}"
            )

    print("\nPhase 02 data-pipeline smoke test passed.")


if __name__ == "__main__":
    main()
