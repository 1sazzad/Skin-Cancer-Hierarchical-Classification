"""Train one Phase 06 shared three-task backbone under the frozen Phase 03 protocol.

This adapter deliberately reuses the validated Phase 03 training loop. The only
scientific variable it permits is the shared encoder architecture.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.train_phase03_shared_three_task as phase03_launcher
from src.models.shared_three_task import (
    SUPPORTED_SHARED_ARCHITECTURES,
    build_shared_three_task_model,
)
from src.utils.reproducibility import seed_everything

DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/experiments/phase03_shared_three_task_hierarchical_baseline.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one approved Phase 06 shared three-task backbone while "
            "preserving the frozen Phase 03 protocol. No internal-test path exists."
        )
    )
    parser.add_argument(
        "--architecture",
        required=True,
        choices=SUPPORTED_SHARED_ARCHITECTURES,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def load_phase06_config(path: Path, architecture: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Shared training config not found: {resolved}")
    config = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Shared training config must be a mapping.")

    # Reuse the strict Phase 03 scientific drift guard for every field except
    # architecture, which is the one predeclared Phase 06 experimental variable.
    frozen_view = deepcopy(config)
    frozen_view["model"]["architecture"] = "efficientnet_b0"
    phase03_launcher._validate_frozen_config(frozen_view)

    if architecture not in SUPPORTED_SHARED_ARCHITECTURES:
        raise ValueError(f"Unsupported shared architecture {architecture!r}.")

    config["model"]["architecture"] = architecture
    config["experiment"]["research_stage"] = (
        "phase06_multi_backbone_shared_hierarchical_benchmark"
    )
    config["experiment"]["run_name"] = (
        f"phase06_shared_three_task_{architecture}_seed42"
    )
    config["experiment"]["output_root"] = (
        f"runs/phase06_shared_three_task/{architecture}/"
    )
    return config


def install_architecture_builder(architecture: str) -> None:
    """Redirect the legacy launcher builder without modifying training semantics."""

    def build_selected_shared_model(
        *,
        pretrained: str = "imagenet",
        dropout_probability: float = 0.2,
    ) -> torch.nn.Module:
        return build_shared_three_task_model(
            architecture,
            pretrained=pretrained,
            dropout_probability=dropout_probability,
        )

    phase03_launcher.build_shared_three_task_efficientnet_b0 = (
        build_selected_shared_model
    )


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    config = load_phase06_config(config_path, args.architecture)
    install_architecture_builder(args.architecture)

    device = phase03_launcher._resolve_device(args.device)
    seed_everything(int(config["experiment"]["seed"]))

    if args.preflight:
        dataloaders, model, criterion, optimizer, scheduler = (
            phase03_launcher._build_components(
                config,
                project_root=project_root,
                device=device,
            )
        )
        del criterion, optimizer, scheduler
        print("Phase 06 shared multi-backbone preflight PASS.")
        print(f"architecture: {args.architecture}")
        print(f"device: {device}")
        print(f"train_samples: {len(dataloaders.train.dataset)}")
        print(
            f"validation_task1: {len(dataloaders.validation_task1.dataset)}"
        )
        print(
            f"validation_task2: {len(dataloaders.validation_task2.dataset)}"
        )
        print(
            f"validation_task3: {len(dataloaders.validation_task3.dataset)}"
        )
        print(f"parameters: {sum(p.numel() for p in model.parameters())}")
        print("internal_test_loader_constructed: false")
        return

    run_directory = phase03_launcher._prepare_run_directory(args.run_directory)
    resolved_config_path = run_directory / "resolved_config.yaml"
    resolved_config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    log_path = run_directory / "logs/train_console.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        tee = phase03_launcher.Tee(sys.stdout, log_handle)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            phase03_launcher._run_training(
                config,
                config_path=resolved_config_path,
                project_root=project_root,
                run_directory=run_directory,
                device=device,
            )


if __name__ == "__main__":
    main()
