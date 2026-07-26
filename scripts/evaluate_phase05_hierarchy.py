"""Run the one locked Phase 05 conditional hierarchical internal evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.evaluation.hierarchical_inference_engine import (
    run_locked_hierarchical_evaluation,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "evaluation"
    / "phase05_hierarchical_internal_test.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Stage 1 + Stage 2 hierarchy exactly once "
            "on the locked ISIC 2019 internal-test partition."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    outcome = run_locked_hierarchical_evaluation(
        args.config,
        project_root=args.project_root,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("Locked Phase 05 evaluation completed.")
    print("rerun_permitted: false")
    print(
        "predicted_gate_macro_f1: "
        f"{outcome.predicted_gate_macro_f1:.6f}"
    )
    print(f"output_directory: {outcome.output_directory}")
    print(f"metrics: {outcome.metrics_path}")
    print(f"predictions: {outcome.predictions_path}")
    print(f"routing: {outcome.routing_path}")
    print(f"summary: {outcome.summary_path}")


if __name__ == "__main__":
    main()
