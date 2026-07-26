"""Verify Phase 05 checkpoints and CUDA compatibility without dataset inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.evaluation.hierarchical_preflight import (
    run_hierarchical_preflight,
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
            "Verify the frozen Phase 05 models using synthetic input. "
            "No ISIC images are evaluated."
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
        "--dummy-batch-size",
        type=int,
        default=2,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    outcome = run_hierarchical_preflight(
        args.config,
        project_root=args.project_root,
        device=args.device,
        dummy_batch_size=args.dummy_batch_size,
    )

    print("Phase 05 preflight passed.")
    print("Internal-test images evaluated: 0")
    print(f"device: {outcome.device}")
    print(f"manifest: {outcome.manifest_path}")
    print(f"locked_output: {outcome.output_directory}")
    print(
        "stage_1: "
        f"epoch={outcome.stage_1_epoch}, "
        f"sha256={outcome.stage_1_sha256}, "
        f"output_shape={outcome.stage_1_output_shape}"
    )
    print(
        "stage_2: "
        f"epoch={outcome.stage_2_epoch}, "
        f"sha256={outcome.stage_2_sha256}, "
        f"output_shape={outcome.stage_2_output_shape}"
    )


if __name__ == "__main__":
    main()
