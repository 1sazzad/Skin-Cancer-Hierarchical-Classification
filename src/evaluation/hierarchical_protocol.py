"""Locked Phase 05 protocol and frozen-checkpoint identity validation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from torch import nn

from src.data.dataloaders import DataLoaderConfig
from src.data.hierarchical_inference_dataset import (
    FINAL_CLASS_TO_INDEX,
    STAGE_1_CLASS_TO_INDEX,
    STAGE_2_CLASS_TO_INDEX,
)
from src.models.efficientnet_baseline import build_efficientnet_b0


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_REQUIRED_CHECKPOINT_FIELDS = {
    "epoch",
    "model_state_dict",
    "validation_metrics",
    "config",
    "class_names",
}

_REQUIRED_REPORTING_FLAGS = {
    "save_per_image_predictions",
    "save_probabilities",
    "save_confusion_matrices",
    "save_per_class_metrics",
    "save_routing_analysis",
    "save_environment",
}


@dataclass(frozen=True, slots=True)
class FrozenCheckpointSpec:
    """Expected identity and class contract for one frozen checkpoint."""

    task: str
    path: Path
    sha256: str
    epoch: int
    class_to_index: dict[str, int]
    class_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HierarchicalEvaluationProtocol:
    """Fully validated, path-resolved Phase 05 evaluation protocol."""

    config_path: Path
    project_root: Path
    manifest_path: Path
    output_directory: Path
    seed: int
    verify_image_paths: bool
    loader_config: DataLoaderConfig
    stage_1: FrozenCheckpointSpec
    stage_2: FrozenCheckpointSpec
    final_class_to_index: dict[str, int]
    reporting: dict[str, bool]


def _mapping(
    payload: Mapping[str, Any],
    key: str,
) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Expected {key!r} to be a mapping.")
    return value


def _resolve_path(
    raw_path: object,
    project_root: Path,
    *,
    name: str,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"{name} must be a non-empty path string.")

    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate

    return candidate.resolve()


def _ordered_class_names(
    class_to_index: Mapping[str, int],
) -> tuple[str, ...]:
    ordered = sorted(
        class_to_index.items(),
        key=lambda item: int(item[1]),
    )
    indices = [int(index) for _, index in ordered]

    if indices != list(range(len(indices))):
        raise ValueError(
            "Class indices must be contiguous and start from zero."
        )

    return tuple(str(name) for name, _ in ordered)


def _normalized_class_mapping(
    raw_mapping: object,
    *,
    name: str,
) -> dict[str, int]:
    if not isinstance(raw_mapping, dict):
        raise ValueError(f"{name} must be a mapping.")

    try:
        normalized = {
            str(class_name): int(index)
            for class_name, index in raw_mapping.items()
        }
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must map class names to integer indices."
        ) from exc

    _ordered_class_names(normalized)
    return normalized


def _validate_exact_mapping(
    actual: Mapping[str, int],
    expected: Mapping[str, int],
    *,
    name: str,
) -> None:
    if dict(actual) != dict(expected):
        raise ValueError(
            f"{name} must equal {dict(expected)}; found {dict(actual)}."
        )


def _validate_sha256(value: object, *, name: str) -> str:
    normalized = str(value).strip().lower()

    if not _SHA256_PATTERN.fullmatch(normalized):
        raise ValueError(
            f"{name} must be a lowercase 64-character SHA-256 digest."
        )

    return normalized


def _checkpoint_spec(
    section: Mapping[str, Any],
    *,
    project_root: Path,
    task: str,
    expected_mapping: Mapping[str, int],
) -> FrozenCheckpointSpec:
    class_to_index = _normalized_class_mapping(
        section.get("class_to_index"),
        name=f"{task}.class_to_index",
    )
    _validate_exact_mapping(
        class_to_index,
        expected_mapping,
        name=f"{task}.class_to_index",
    )

    epoch = int(section.get("checkpoint_epoch", -1))
    if epoch <= 0:
        raise ValueError(
            f"{task}.checkpoint_epoch must be a positive integer."
        )

    return FrozenCheckpointSpec(
        task=task,
        path=_resolve_path(
            section.get("checkpoint"),
            project_root,
            name=f"{task}.checkpoint",
        ),
        sha256=_validate_sha256(
            section.get("checkpoint_sha256"),
            name=f"{task}.checkpoint_sha256",
        ),
        epoch=epoch,
        class_to_index=class_to_index,
        class_names=_ordered_class_names(class_to_index),
    )


def load_hierarchical_evaluation_protocol(
    config_path: str | Path,
    *,
    project_root: str | Path,
) -> HierarchicalEvaluationProtocol:
    """Load and validate the locked Phase 05 protocol without running inference."""

    project = Path(project_root).expanduser().resolve()
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = project / path
    path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Phase 05 evaluation config not found: {path}"
        )

    loaded = yaml.safe_load(
        path.read_text(encoding="utf-8-sig")
    )
    if not isinstance(loaded, dict):
        raise ValueError(
            "Phase 05 evaluation config must contain a mapping."
        )

    evaluation = _mapping(loaded, "evaluation")
    data = _mapping(loaded, "data")
    stage_1_section = _mapping(loaded, "stage_1")
    stage_2_section = _mapping(loaded, "stage_2")
    hierarchy = _mapping(loaded, "hierarchy")
    loader = _mapping(loaded, "loader")
    output = _mapping(loaded, "output")
    reporting = _mapping(loaded, "reporting")

    expected_evaluation_values = {
        "phase": "phase05",
        "dataset": "isic2019",
        "split": "internal_test",
        "protocol_status": "locked_before_internal_test",
        "rerun_permitted": False,
    }
    for key, expected_value in expected_evaluation_values.items():
        if evaluation.get(key) != expected_value:
            raise ValueError(
                f"evaluation.{key} must be {expected_value!r}."
            )

    seed = int(evaluation.get("seed", -1))
    if seed != 42:
        raise ValueError(
            "The primary Phase 05 evaluation seed must remain 42."
        )

    if data.get("verify_image_paths") is not True:
        raise ValueError(
            "data.verify_image_paths must remain true."
        )

    if hierarchy.get("gate_policy") != "argmax":
        raise ValueError(
            "hierarchy.gate_policy must remain 'argmax'."
        )

    if int(hierarchy.get("stage_1_malignant_index", -1)) != 1:
        raise ValueError(
            "hierarchy.stage_1_malignant_index must remain 1."
        )

    if (
        hierarchy.get("stage_2_execution_policy")
        != "union_of_true_and_predicted_malignant"
    ):
        raise ValueError(
            "Unsupported Stage 2 execution policy."
        )

    final_mapping = _normalized_class_mapping(
        hierarchy.get("final_class_to_index"),
        name="hierarchy.final_class_to_index",
    )
    _validate_exact_mapping(
        final_mapping,
        FINAL_CLASS_TO_INDEX,
        name="hierarchy.final_class_to_index",
    )

    loader_config = DataLoaderConfig(
        batch_size=int(loader.get("batch_size", 0)),
        num_workers=int(loader.get("num_workers", -1)),
        pin_memory=bool(loader.get("pin_memory")),
        persistent_workers=bool(
            loader.get("persistent_workers")
        ),
        prefetch_factor=int(loader.get("prefetch_factor", 0)),
        drop_last_train=False,
        seed=seed,
    )

    if output.get("refuse_existing_directory") is not True:
        raise ValueError(
            "output.refuse_existing_directory must remain true."
        )

    missing_reporting = sorted(
        _REQUIRED_REPORTING_FLAGS - set(reporting)
    )
    if missing_reporting:
        raise ValueError(
            "Missing Phase 05 reporting flags: "
            f"{missing_reporting}"
        )

    normalized_reporting = {
        key: bool(reporting[key])
        for key in _REQUIRED_REPORTING_FLAGS
    }
    disabled_reporting = sorted(
        key
        for key, enabled in normalized_reporting.items()
        if not enabled
    )
    if disabled_reporting:
        raise ValueError(
            "All primary Phase 05 reporting outputs must remain enabled: "
            f"{disabled_reporting}"
        )

    return HierarchicalEvaluationProtocol(
        config_path=path,
        project_root=project,
        manifest_path=_resolve_path(
            data.get("split_manifest"),
            project,
            name="data.split_manifest",
        ),
        output_directory=_resolve_path(
            output.get("directory"),
            project,
            name="output.directory",
        ),
        seed=seed,
        verify_image_paths=True,
        loader_config=loader_config,
        stage_1=_checkpoint_spec(
            stage_1_section,
            project_root=project,
            task="stage_1",
            expected_mapping=STAGE_1_CLASS_TO_INDEX,
        ),
        stage_2=_checkpoint_spec(
            stage_2_section,
            project_root=project,
            task="stage_2",
            expected_mapping=STAGE_2_CLASS_TO_INDEX,
        ),
        final_class_to_index=final_mapping,
        reporting=normalized_reporting,
    )


def compute_file_sha256(path: str | Path) -> str:
    """Compute SHA-256 without loading the complete checkpoint into memory."""

    candidate = Path(path)
    digest = hashlib.sha256()

    with candidate.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def verify_frozen_checkpoint(
    specification: FrozenCheckpointSpec,
) -> dict[str, Any]:
    """Verify checkpoint bytes, metadata, task, class order, and epoch."""

    checkpoint = specification.path

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Frozen {specification.task} checkpoint not found: "
            f"{checkpoint}"
        )

    observed_sha256 = compute_file_sha256(checkpoint)
    if observed_sha256 != specification.sha256:
        raise ValueError(
            f"{specification.task} checkpoint SHA-256 mismatch: "
            f"expected {specification.sha256}, "
            f"observed {observed_sha256}."
        )

    loaded = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(loaded, dict):
        raise ValueError(
            f"{specification.task} checkpoint must contain a mapping."
        )

    missing_fields = sorted(
        _REQUIRED_CHECKPOINT_FIELDS - set(loaded)
    )
    if missing_fields:
        raise ValueError(
            f"{specification.task} checkpoint is missing fields: "
            f"{missing_fields}"
        )

    if int(loaded["epoch"]) != specification.epoch:
        raise ValueError(
            f"{specification.task} checkpoint epoch mismatch."
        )

    checkpoint_class_names = tuple(
        str(name) for name in loaded["class_names"]
    )
    if checkpoint_class_names != specification.class_names:
        raise ValueError(
            f"{specification.task} checkpoint class order mismatch: "
            f"expected {specification.class_names}, "
            f"observed {checkpoint_class_names}."
        )

    config = loaded["config"]
    if not isinstance(config, dict):
        raise ValueError(
            f"{specification.task} checkpoint config must be a mapping."
        )

    runtime = _mapping(config, "runtime")
    if runtime.get("sanity_run") is not False:
        raise ValueError(
            f"{specification.task} must use a full non-sanity checkpoint."
        )

    data = _mapping(config, "data")
    if data.get("task") != specification.task:
        raise ValueError(
            f"{specification.task} checkpoint task mismatch."
        )

    checkpoint_mapping = _normalized_class_mapping(
        data.get("class_to_index"),
        name=f"{specification.task} checkpoint class_to_index",
    )
    _validate_exact_mapping(
        checkpoint_mapping,
        specification.class_to_index,
        name=f"{specification.task} checkpoint class_to_index",
    )

    model_config = _mapping(config, "model")
    if model_config.get("architecture") != "efficientnet_b0":
        raise ValueError(
            "Phase 05 supports the frozen EfficientNet-B0 models only."
        )
    if int(model_config.get("number_of_classes", -1)) != len(
        specification.class_names
    ):
        raise ValueError(
            f"{specification.task} number_of_classes mismatch."
        )

    run_summary_path = checkpoint.parent / "run_summary.json"
    if not run_summary_path.is_file():
        raise FileNotFoundError(
            f"Missing run_summary.json beside {checkpoint}."
        )

    run_summary = json.loads(
        run_summary_path.read_text(encoding="utf-8-sig")
    )
    if not isinstance(run_summary, dict):
        raise ValueError(
            f"{specification.task} run summary must be a mapping."
        )
    if run_summary.get("reportable_as_full_result") is not True:
        raise ValueError(
            f"{specification.task} run is not reportable as a full result."
        )
    if bool(run_summary.get("sanity_run")):
        raise ValueError(
            f"{specification.task} run summary is marked as sanity."
        )
    if int(run_summary.get("best_epoch", -1)) != specification.epoch:
        raise ValueError(
            f"{specification.task} run-summary epoch mismatch."
        )
    if run_summary.get("task") != specification.task:
        raise ValueError(
            f"{specification.task} run-summary task mismatch."
        )

    return loaded


def build_verified_frozen_model(
    specification: FrozenCheckpointSpec,
    *,
    device: str | torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Verify and construct one frozen EfficientNet-B0 inference model."""

    payload = verify_frozen_checkpoint(specification)
    model_config = _mapping(payload["config"], "model")

    model = build_efficientnet_b0(
        len(specification.class_names),
        pretrained="none",
        dropout_probability=float(
            model_config.get("dropout_probability", 0.2)
        ),
    )
    model.load_state_dict(
        payload["model_state_dict"],
        strict=True,
    )
    model.to(torch.device(device))
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model, payload
