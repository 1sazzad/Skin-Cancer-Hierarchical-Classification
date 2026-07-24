"""Evaluate one frozen ISIC 2019 best checkpoint on the internal test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.internal_test_evaluator import evaluate_frozen_internal_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outcome = evaluate_frozen_internal_test(
        args.checkpoint,
        project_root=args.project_root,
        output_directory=args.output_directory,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print("Internal-test evaluation completed.")
    print(f"task: {outcome.task}")
    print(f"checkpoint_epoch: {outcome.checkpoint_epoch}")
    print(f"test_macro_f1: {outcome.test_macro_f1:.6f}")
    print(f"output_directory: {outcome.output_directory}")
    print(f"metrics: {outcome.metrics_path}")
    print(f"predictions: {outcome.predictions_path}")
    print(f"summary: {outcome.summary_path}")


if __name__ == "__main__":
    main()
