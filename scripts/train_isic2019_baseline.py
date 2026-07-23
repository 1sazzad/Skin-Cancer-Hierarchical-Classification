"""Train one configuration-driven ISIC 2019 clean baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.baseline_experiment import run_baseline_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/runs"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--epoch-limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outcome = run_baseline_experiment(
        args.config,
        project_root=args.project_root,
        output_root=args.output_root,
        device=args.device,
        max_train_batches=args.max_train_batches,
        max_validation_batches=args.max_validation_batches,
        epoch_limit=args.epoch_limit,
    )
    print("Training run completed.")
    print(f"run_directory: {outcome.run_directory}")
    print(f"best_epoch: {outcome.best_epoch}")
    print(
        "best_validation_macro_f1: "
        f"{outcome.best_validation_macro_f1:.6f}"
    )
    print(f"best_checkpoint: {outcome.best_checkpoint_path}")
    print(f"last_checkpoint: {outcome.last_checkpoint_path}")


if __name__ == "__main__":
    main()
