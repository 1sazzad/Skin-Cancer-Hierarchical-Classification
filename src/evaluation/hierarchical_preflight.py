"""Safe preflight checks before the one-time Phase 05 internal evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from src.evaluation.hierarchical_protocol import (
    build_verified_frozen_model,
    load_hierarchical_evaluation_protocol,
)


@dataclass(frozen=True, slots=True)
class HierarchicalPreflightOutcome:
    """Validated identities and synthetic-forward results."""

    device: str
    manifest_path: Path
    output_directory: Path
    stage_1_checkpoint: Path
    stage_2_checkpoint: Path
    stage_1_epoch: int
    stage_2_epoch: int
    stage_1_sha256: str
    stage_2_sha256: str
    stage_1_output_shape: tuple[int, int]
    stage_2_output_shape: tuple[int, int]


def _validate_output(
    logits: torch.Tensor,
    *,
    batch_size: int,
    class_count: int,
    stage_name: str,
) -> tuple[int, int]:
    expected_shape = (batch_size, class_count)

    if logits.ndim != 2 or tuple(logits.shape) != expected_shape:
        raise ValueError(
            f"{stage_name} synthetic output must have shape "
            f"{expected_shape}; observed {tuple(logits.shape)}."
        )

    if not bool(torch.isfinite(logits).all()):
        raise ValueError(
            f"{stage_name} synthetic output contains non-finite values."
        )

    return expected_shape


def run_hierarchical_preflight(
    config_path: str | Path,
    *,
    project_root: str | Path,
    device: str | torch.device = "cuda",
    dummy_batch_size: int = 2,
) -> HierarchicalPreflightOutcome:
    """Verify the frozen models without consulting internal-test images."""

    if dummy_batch_size <= 0:
        raise ValueError("dummy_batch_size must be positive.")

    protocol = load_hierarchical_evaluation_protocol(
        config_path,
        project_root=project_root,
    )

    if not protocol.manifest_path.is_file():
        raise FileNotFoundError(
            f"Frozen split manifest not found: {protocol.manifest_path}"
        )

    if protocol.output_directory.exists():
        raise FileExistsError(
            "Locked Phase 05 output directory already exists: "
            f"{protocol.output_directory}"
        )

    resolved_device = torch.device(device)

    if (
        resolved_device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested for preflight but is not available."
        )

    stage_1_model, _ = build_verified_frozen_model(
        protocol.stage_1,
        device=resolved_device,
    )
    stage_2_model, _ = build_verified_frozen_model(
        protocol.stage_2,
        device=resolved_device,
    )

    synthetic_images = torch.zeros(
        (dummy_batch_size, 3, 224, 224),
        dtype=torch.float32,
        device=resolved_device,
    )

    use_amp = resolved_device.type == "cuda"

    with torch.inference_mode():
        with torch.autocast(
            device_type=resolved_device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            stage_1_logits = stage_1_model(
                synthetic_images
            )
            stage_2_logits = stage_2_model(
                synthetic_images
            )

    stage_1_shape = _validate_output(
        stage_1_logits,
        batch_size=dummy_batch_size,
        class_count=2,
        stage_name="Stage 1",
    )
    stage_2_shape = _validate_output(
        stage_2_logits,
        batch_size=dummy_batch_size,
        class_count=3,
        stage_name="Stage 2",
    )

    return HierarchicalPreflightOutcome(
        device=str(resolved_device),
        manifest_path=protocol.manifest_path,
        output_directory=protocol.output_directory,
        stage_1_checkpoint=protocol.stage_1.path,
        stage_2_checkpoint=protocol.stage_2.path,
        stage_1_epoch=protocol.stage_1.epoch,
        stage_2_epoch=protocol.stage_2.epoch,
        stage_1_sha256=protocol.stage_1.sha256,
        stage_2_sha256=protocol.stage_2.sha256,
        stage_1_output_shape=stage_1_shape,
        stage_2_output_shape=stage_2_shape,
    )
