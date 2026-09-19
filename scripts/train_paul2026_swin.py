"""Launch PAUL-2026 Swin-T runs or perform offline CPU preflight checks."""

import argparse
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.paul2026_swin import load_paul_config, preflight, run_paul_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config = load_paul_config(args.config)
    if args.preflight:
        report = preflight(config, project_root=args.project_root, device=args.device)
        print(yaml.safe_dump(report, sort_keys=False), end="")
    else:
        run_paul_experiment(
            args.config, project_root=args.project_root,
            output_root=args.output_root, device=args.device,
        )


if __name__ == "__main__":
    main()
