"""Execute the frozen Phase 03 shared three-task training protocol."""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, TextIO

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/experiments/phase03_shared_three_task_hierarchical_baseline.yaml"
)
DEFAULT_RUN_DIRECTORY = (
    PROJECT_ROOT / "runs/phase03_shared_three_task/seed_42"
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloaders import DataLoaderConfig
from src.data.shared_three_task import build_shared_three_task_dataloaders
from src.models.shared_three_task import (
    TASK_CLASS_MAPPINGS,
    build_shared_three_task_efficientnet_b0,
)
from src.training.shared_three_task import (
    SharedValidationEarlyStopping,
    build_checkpoint_payload,
    build_masked_loss_from_config,
    build_optimizer_and_scheduler,
    run_shared_training_epoch,
    save_best_checkpoint,
    validate_shared_model,
    write_training_history,
)
from src.utils.reproducibility import seed_everything


class Tee:
    """Write console text to both the terminal and the run log."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the frozen Phase 03 shared EfficientNet-B0 baseline. "
            "This launcher has no internal-test path."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--run-directory",
        type=Path,
        default=DEFAULT_RUN_DIRECTORY,
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Build and verify config, train/validation loaders, model, losses, "
            "optimizer, and scheduler without iterating data or writing a run."
        ),
    )
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Shared training config not found: {resolved}")
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Shared training config must be a mapping.")
    _validate_frozen_config(payload)
    return payload


def _validate_frozen_config(config: Mapping[str, Any]) -> None:
    """Reject scientific drift and any internal-test enablement."""

    experiment = config["experiment"]
    data = config["data"]
    loader = config["loader"]
    model = config["model"]
    preprocessing = config["preprocessing"]
    losses = config["task_losses"]
    training = config["training"]
    validation = config["validation"]
    expected = {
        "experiment.seed": (experiment["seed"], 42),
        "loader.batch_size": (loader["batch_size"], 64),
        "loader.num_workers": (loader["num_workers"], 4),
        "loader.drop_last_train": (loader["drop_last_train"], False),
        "loader.train_shuffle": (loader["train_shuffle"], True),
        "model.architecture": (model["architecture"], "efficientnet_b0"),
        "model.encoder_sharing": (
            model["encoder_sharing"],
            "one_shared_encoder_pass",
        ),
        "model.pretrained_weights": (model["pretrained_weights"], "imagenet"),
        "model.dropout_probability": (model["dropout_probability"], 0.2),
        "model.historical_checkpoint_initialization": (
            model["historical_checkpoint_initialization"],
            False,
        ),
        "preprocessing.input_size": (preprocessing["input_size"], [224, 224]),
        "preprocessing.normalization": (
            preprocessing["normalization"],
            "imagenet",
        ),
        "preprocessing.train_transform": (
            preprocessing["train_transform"],
            "locked_moderate_baseline",
        ),
        "preprocessing.validation_transform": (
            preprocessing["validation_transform"],
            "deterministic_resize_256_center_crop_224",
        ),
        "task_losses.lambda_task1": (losses["lambda_task1"], 1.0),
        "task_losses.lambda_task2": (losses["lambda_task2"], 1.0),
        "task_losses.lambda_task3": (losses["lambda_task3"], 1.0),
        "task1.name": (losses["task1"]["name"], "cross_entropy"),
        "task2.name": (losses["task2"]["name"], "class_balanced_focal_loss"),
        "task2.beta": (losses["task2"]["beta"], 0.9999),
        "task2.gamma": (losses["task2"]["gamma"], 2.0),
        "task3.name": (losses["task3"]["name"], "weighted_cross_entropy"),
        "training.epochs": (training["epochs"], 30),
        "training.early_stopping_patience": (
            training["early_stopping_patience"],
            7,
        ),
        "training.amp": (training["amp"], True),
        "optimizer.name": (training["optimizer"]["name"], "adamw"),
        "optimizer.learning_rate": (
            training["optimizer"]["learning_rate"],
            0.0003,
        ),
        "optimizer.weight_decay": (
            training["optimizer"]["weight_decay"],
            0.0001,
        ),
        "scheduler.name": (
            training["scheduler"]["name"],
            "cosine_annealing",
        ),
        "scheduler.t_max": (training["scheduler"]["t_max"], 30),
        "scheduler.minimum_learning_rate": (
            training["scheduler"]["minimum_learning_rate"],
            0.000001,
        ),
        "validation.task1_cohort": (
            validation["task1_cohort"],
            "isic2019_validation_all",
        ),
        "validation.task2_cohort": (
            validation["task2_cohort"],
            "isic2019_validation_malignant_only",
        ),
        "validation.task3_cohort": (
            validation["task3_cohort"],
            "isic_stage3_validation",
        ),
        "validation.task_metric": (validation["task_metric"], "macro_f1"),
        "validation.shared_score": (
            validation["shared_score"],
            "arithmetic_mean_task1_task2_task3_macro_f1",
        ),
        "validation.checkpoint_selection": (
            validation["checkpoint_selection"],
            "highest_shared_validation_score",
        ),
    }
    mismatches = [
        f"{name}: expected {wanted!r}, got {actual!r}"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    expected_heads = {
        "task1": {"type": "dropout_linear", "logits": 2},
        "task2": {"type": "dropout_linear", "logits": 3},
        "task3": {"type": "dropout_linear", "logits": 5},
    }
    if model["heads"] != expected_heads:
        mismatches.append("model.heads must remain the frozen 2/3/5 heads")
    expected_masks = {
        "isic_non_malignant": [1, 0, 0],
        "isic_melanoma_bcc_scc": [1, 1, 0],
        "isic_derived_stage3": [0, 0, 1],
    }
    if data["task_mask_policy"] != expected_masks:
        mismatches.append("data.task_mask_policy differs from the frozen masks")
    expected_class_mappings = {
        task: dict(mapping) for task, mapping in TASK_CLASS_MAPPINGS.items()
    }
    if data["class_mappings"] != expected_class_mappings:
        mismatches.append("data.class_mappings differs from the frozen mappings")
    if data["missing_target_sentinel"] != -100:
        mismatches.append("data.missing_target_sentinel must remain -100")
    if data["sampling"] != {
        "policy": "natural_concat_shuffle_each_epoch",
        "weighted_sampler": False,
        "stage3_oversampling": False,
        "forced_source_mixing": False,
    }:
        mismatches.append("data.sampling differs from the frozen natural policy")
    if data["internal_test"] != {
        "allowed": False,
        "construct_loader": False,
        "influence_selection": False,
    }:
        mismatches.append("internal test must remain prohibited")
    if losses["task2"]["class_weights"] != {
        "melanoma": 0.3485376280807543,
        "bcc": 0.4553489231324597,
        "scc": 2.196113448786786,
    }:
        mismatches.append("Task-2 frozen train-only weights changed")
    if losses["task2"]["class_weight_source"] != {
        "partition": "isic2019_train",
        "method": "effective_number",
        "normalization": "sum_to_number_of_classes",
        "class_counts": {"melanoma": 3164, "bcc": 2327, "scc": 440},
    }:
        mismatches.append("Task-2 weight provenance changed")
    if losses["task3"]["class_weights"] != {
        "Tis": 0.063475735584673,
        "T1": 0.12246677245955931,
        "T2": 0.682845034319967,
        "T3": 2.253388613255891,
        "T4": 1.8778238443799093,
    }:
        mismatches.append("Task-3 frozen train-only weights changed")
    if losses["task3"]["class_weight_source"] != {
        "partition": "isic_stage3_train",
        "method": "inverse_frequency",
        "normalization": "sum_to_number_of_classes",
        "class_counts": {"Tis": 355, "T1": 184, "T2": 33, "T3": 10, "T4": 12},
    }:
        mismatches.append("Task-3 weight provenance changed")
    if mismatches:
        raise ValueError("Frozen Phase 03 protocol mismatch: " + "; ".join(mismatches))


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(requested)


def _build_components(
    config: Mapping[str, Any],
    *,
    project_root: Path,
    device: torch.device,
) -> tuple[Any, torch.nn.Module, Any, Any, Any]:
    loader = config["loader"]
    isic_manifest = Path(config["data"]["isic2019_manifest"])
    stage3_manifest = Path(config["data"]["stage3_manifest"])
    if not isic_manifest.is_absolute():
        isic_manifest = project_root / isic_manifest
    if not stage3_manifest.is_absolute():
        stage3_manifest = project_root / stage3_manifest
    dataloaders = build_shared_three_task_dataloaders(
        isic_manifest,
        stage3_manifest,
        project_root,
        config=DataLoaderConfig(
            batch_size=int(loader["batch_size"]),
            num_workers=int(loader["num_workers"]),
            pin_memory=bool(loader["pin_memory"]),
            persistent_workers=bool(loader["persistent_workers"]),
            prefetch_factor=int(loader["prefetch_factor"]),
            drop_last_train=bool(loader["drop_last_train"]),
            seed=int(config["experiment"]["seed"]),
        ),
        verify_image_paths=True,
    )
    expected_counts = config["data"]["training_sources"]
    if dataloaders.train_source_counts != {
        "isic2019": int(expected_counts["isic2019_train"]),
        "isic_stage03": int(expected_counts["isic_stage3_train"]),
        "combined": int(expected_counts["combined_natural_pool"]),
    }:
        raise ValueError("Resolved training-source counts differ from config.")

    model = build_shared_three_task_efficientnet_b0(
        pretrained="imagenet",
        dropout_probability=float(config["model"]["dropout_probability"]),
    ).to(device)
    criterion = build_masked_loss_from_config(config, device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)
    return dataloaders, model, criterion, optimizer, scheduler


def _git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _prepare_run_directory(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(
            f"Run directory is not empty; refusing to overwrite: {resolved}"
        )
    (resolved / "logs").mkdir(parents=True, exist_ok=True)
    return resolved


def _format_loss(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def _run_training(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    project_root: Path,
    run_directory: Path,
    device: torch.device,
) -> None:
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    dataloaders, model, criterion, optimizer, scheduler = _build_components(
        config,
        project_root=project_root,
        device=device,
    )
    use_amp = bool(config["training"]["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    validation_loaders = {
        "task1": dataloaders.validation_task1,
        "task2": dataloaders.validation_task2,
        "task3": dataloaders.validation_task3,
    }
    control = SharedValidationEarlyStopping(
        patience=int(config["training"]["early_stopping_patience"])
    )
    history: list[dict[str, Any]] = []
    best_metrics: dict[str, float] | None = None
    stopped_early = False
    started = time.perf_counter()
    checkpoint_path = run_directory / "best_checkpoint.pt"
    git_commit = _git_commit(project_root)

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_result = run_shared_training_epoch(
            model,
            dataloaders.train,
            criterion,
            optimizer,
            device,
            gradient_scaler=scaler,
            amp_enabled=use_amp,
        )
        validation_metrics = validate_shared_model(
            model,
            validation_loaders,
            device,
        )
        scheduler.step()
        improved, should_stop = control.update(
            validation_metrics["shared_validation_score"],
            epoch,
        )
        if improved:
            best_metrics = dict(validation_metrics)
            payload = build_checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                validation_metrics=validation_metrics,
                seed=seed,
                model_metadata=deepcopy(config["model"]),
                config_metadata={
                    "config_path": str(config_path),
                    "experiment": deepcopy(config["experiment"]),
                    "training": deepcopy(config["training"]),
                    "validation": deepcopy(config["validation"]),
                    "git_commit": git_commit,
                },
                task_loss_configuration=deepcopy(config["task_losses"]),
                task_mask_policy=deepcopy(
                    config["data"]["task_mask_policy"]
                ),
            )
            save_best_checkpoint(checkpoint_path, payload)

        record = {
            "epoch": epoch,
            "train_total_loss": train_result.train_total_loss,
            "train_task1_loss": train_result.train_task1_loss,
            "train_task2_loss": train_result.train_task2_loss,
            "train_task3_loss": train_result.train_task3_loss,
            **validation_metrics,
            "best_shared_validation_score": control.best_score,
            "learning_rate": learning_rate,
            "patience_counter": control.epochs_without_improvement,
        }
        history.append(record)
        running_summary = {
            "completion_status": "running",
            "best_epoch": control.best_epoch,
            "best_shared_validation_score": control.best_score,
            "best_validation_metrics": best_metrics,
            "early_stopping": False,
            "epochs_completed": len(history),
            "seed": seed,
            "config_path": str(config_path),
            "git_commit": git_commit,
            "best_checkpoint_path": str(checkpoint_path),
        }
        write_training_history(
            run_directory,
            history,
            running_summary,
            csv_filename="training_history.csv",
        )
        print(
            f"epoch={epoch:02d} "
            f"train_total_loss={train_result.train_total_loss:.6f} "
            f"train_task1_loss={_format_loss(train_result.train_task1_loss)} "
            f"train_task2_loss={_format_loss(train_result.train_task2_loss)} "
            f"train_task3_loss={_format_loss(train_result.train_task3_loss)} "
            f"val_task1_macro_f1={validation_metrics['val_task1_macro_f1']:.6f} "
            f"val_task2_macro_f1={validation_metrics['val_task2_macro_f1']:.6f} "
            f"val_task3_macro_f1={validation_metrics['val_task3_macro_f1']:.6f} "
            f"shared_validation_score="
            f"{validation_metrics['shared_validation_score']:.6f} "
            f"best_shared_validation_score={control.best_score:.6f} "
            f"learning_rate={learning_rate:.8f} "
            f"patience_counter={control.epochs_without_improvement}"
        )
        if should_stop:
            stopped_early = True
            break

    if best_metrics is None or control.best_epoch is None:
        raise RuntimeError("Training completed without selecting a checkpoint.")
    summary = {
        "completion_status": (
            "completed_early_stopping"
            if stopped_early
            else "completed_max_epochs"
        ),
        "best_epoch": control.best_epoch,
        "best_shared_validation_score": control.best_score,
        "best_task1_val_macro_f1": best_metrics["val_task1_macro_f1"],
        "best_task2_val_macro_f1": best_metrics["val_task2_macro_f1"],
        "best_task3_val_macro_f1": best_metrics["val_task3_macro_f1"],
        "early_stopping": stopped_early,
        "epochs_completed": len(history),
        "configured_epochs": int(config["training"]["epochs"]),
        "seed": seed,
        "config_path": str(config_path),
        "git_commit": git_commit,
        "best_checkpoint_path": str(checkpoint_path),
        "training_seconds": time.perf_counter() - started,
        "internal_test_evaluated": False,
    }
    write_training_history(
        run_directory,
        history,
        summary,
        csv_filename="training_history.csv",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    config = _load_config(config_path)
    device = _resolve_device(args.device)
    seed_everything(int(config["experiment"]["seed"]))

    if args.preflight:
        dataloaders, model, criterion, optimizer, scheduler = _build_components(
            config,
            project_root=project_root,
            device=device,
        )
        del criterion, optimizer, scheduler
        print("Phase 03 shared training preflight PASS.")
        print(f"device: {device}")
        print(f"train_samples: {len(dataloaders.train.dataset)}")
        print(f"validation_task1: {len(dataloaders.validation_task1.dataset)}")
        print(f"validation_task2: {len(dataloaders.validation_task2.dataset)}")
        print(f"validation_task3: {len(dataloaders.validation_task3.dataset)}")
        print(f"parameters: {sum(p.numel() for p in model.parameters())}")
        print("internal_test_loader_constructed: false")
        return

    run_directory = _prepare_run_directory(args.run_directory)
    config_copy = run_directory / "resolved_config.yaml"
    config_copy.write_text(
        yaml.safe_dump(dict(config), sort_keys=False),
        encoding="utf-8",
    )
    log_path = run_directory / "logs/train_console.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        tee = Tee(sys.stdout, log_handle)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            _run_training(
                config,
                config_path=config_path,
                project_root=project_root,
                run_directory=run_directory,
                device=device,
            )


if __name__ == "__main__":
    main()
