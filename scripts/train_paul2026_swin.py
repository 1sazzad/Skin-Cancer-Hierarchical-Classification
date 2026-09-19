"""Launch PAUL-2026 Swin-T runs or perform offline CPU preflight checks."""

import argparse
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.paul2026_swin import (
    load_paul_config,
    preflight,
    run_paul_experiment,
    validate_smoke_limits,
)


def positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--epoch-limit", type=positive_integer)
    parser.add_argument("--max-train-batches", type=positive_integer)
    parser.add_argument("--max-validation-batches", type=positive_integer)
    args = parser.parse_args()
    try:
        validate_smoke_limits(
            epoch_limit=args.epoch_limit,
            max_train_batches=args.max_train_batches,
            max_validation_batches=args.max_validation_batches,
        )
    except ValueError as error:
        parser.error(str(error))
    config = load_paul_config(args.config)
    if args.preflight:
        if any(
            value is not None
            for value in (
                args.epoch_limit,
                args.max_train_batches,
                args.max_validation_batches,
            )
        ):
            parser.error("smoke limits cannot be used with --preflight")
        report = preflight(config, project_root=args.project_root, device=args.device)
        print(yaml.safe_dump(report, sort_keys=False), end="")
    else:
        run_paul_experiment(
            args.config, project_root=args.project_root,
            output_root=args.output_root, device=args.device,
            epoch_limit=args.epoch_limit,
            max_train_batches=args.max_train_batches,
            max_validation_batches=args.max_validation_batches,
        )


if __name__ == "__main__":
    main()
