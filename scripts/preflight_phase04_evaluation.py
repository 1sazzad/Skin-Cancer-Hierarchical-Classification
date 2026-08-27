#!/usr/bin/env python3
"""Gate 04C static/checkpoint preflight. No dataset loader or inference is constructed."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from src.evaluation.phase04_comparative_harness import (
    FrozenCheckpointSpec,
    checkpoint_bytes,
    count_parameters,
    environment_provenance,
    load_frozen_shared_model,
    load_frozen_single_task_model,
    optional_macs_flops,
    write_json,
)


def _spec(name: str, raw: dict[str, object]) -> FrozenCheckpointSpec:
    return FrozenCheckpointSpec(
        name=name,
        path=str(raw["path"]),
        sha256=str(raw["sha256"]),
        expected_epoch=int(raw["expected_epoch"]),
        model_kind=str(raw["model_kind"]),
        class_names=tuple(str(value) for value in raw["class_names"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evaluation/phase04_controlled_comparative_validation.yaml"),
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/phase04_controlled_comparative/gate04c_preflight.json"),
    )
    args = parser.parse_args()

    root = args.project_root.expanduser().resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Phase04 config must be a mapping.")
    if config.get("execution_split") != "validation":
        raise ValueError("Gate04C preflight config must remain validation-only.")
    if bool(config.get("internal_test_execution_allowed")):
        raise ValueError("Gate04C must not allow internal-test execution.")

    raw_checkpoints = config.get("checkpoints")
    if not isinstance(raw_checkpoints, dict) or set(raw_checkpoints) != {
        "shared", "task1", "task2", "task3", "flat"
    }:
        raise ValueError("Phase04 requires exactly shared/task1/task2/task3/flat checkpoints.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    specs = {name: _spec(name, value) for name, value in raw_checkpoints.items()}
    models = {
        "shared": load_frozen_shared_model(specs["shared"], root, device),
        "task1": load_frozen_single_task_model(specs["task1"], root, device),
        "task2": load_frozen_single_task_model(specs["task2"], root, device),
        "task3": load_frozen_single_task_model(specs["task3"], root, device),
        "flat": load_frozen_single_task_model(specs["flat"], root, device),
    }

    artifacts: dict[str, object] = {}
    for name, model in models.items():
        artifacts[name] = {
            "checkpoint_path": specs[name].path,
            "checkpoint_sha256": specs[name].sha256,
            "expected_epoch": specs[name].expected_epoch,
            "checkpoint_bytes": checkpoint_bytes(specs[name], root),
            **count_parameters(model),
            "compute": optional_macs_flops(model),
        }

    payload = {
        "gate": "04C",
        "internal_test_executed": False,
        "checkpoint_interface_validation": "PASS",
        "config_path": str(config_path),
        "environment": environment_provenance(device),
        "artifacts": artifacts,
    }
    output = args.output if args.output.is_absolute() else root / args.output
    write_json(output, payload)
    print(f"Gate04C checkpoint/interface preflight PASS: {output}")


if __name__ == "__main__":
    main()
