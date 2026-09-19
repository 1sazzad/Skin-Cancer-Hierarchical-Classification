"""Epoch-boundary persistence for opt-in Shared-Hard production runs."""

from copy import deepcopy
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
import yaml

from src.training.baseline_experiment import (
    _atomic_torch_save,
    _require_cpu_byte_rng_state,
)
from src.training.shared_three_task import SharedValidationEarlyStopping


def refuse_completed_run(directory, config):
    path = Path(directory) / "run_summary.json"
    if path.exists():
        summary = json.loads(path.read_text(encoding="utf-8"))
        name = config["experiment"]["run_name"]
        if summary.get("run_name") == name and (
            summary.get("reportable_as_full_result") is True
            or summary.get("completion_status") in {
                "completed_early_stopping", "completed_max_epochs",
            }
        ):
            raise RuntimeError(f"production run already completed: {name}")


def prepare_resumable_directory(directory, config):
    """Accept only known bootstrap artifacts when no epoch checkpoint exists."""
    directory = Path(directory).expanduser().resolve()
    refuse_completed_run(directory, config)
    resolved = directory / "resolved_config.yaml"
    if resolved.exists() and yaml.safe_load(resolved.read_text(encoding="utf-8")) != config:
        raise ValueError("Shared resume config identity does not match")
    if not (directory / "last_checkpoint.pt").exists() and directory.exists():
        allowed = {"resolved_config.yaml", "logs", ".last_checkpoint.pt.tmp"}
        unexpected = {p.name for p in directory.iterdir()} - allowed
        if unexpected:
            raise ValueError(f"Shared resume state lacks last_checkpoint.pt: {sorted(unexpected)}")
        logs = directory / "logs"
        if logs.exists() and any(p.name != "train_console.log" for p in logs.iterdir()):
            raise ValueError("Unknown Shared bootstrap log artifacts")
    (directory / "logs").mkdir(parents=True, exist_ok=True)
    return directory


def save_shared_resume(path, *, config, model, optimizer, scheduler, scaler,
                       control, best_metrics, history, train_generator,
                       cumulative_seconds, best_payload, worker_states):
    payload = {
        "format_version": 1,
        "run_name": config["experiment"]["run_name"],
        "seed": config["experiment"]["seed"],
        "config_identity": deepcopy(dict(config)),
        "completed_epoch": len(history),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_shared_validation_score": control.best_score,
        "best_epoch": control.best_epoch,
        "epochs_without_improvement": control.epochs_without_improvement,
        "best_validation_metrics": deepcopy(best_metrics),
        # Keep the selected model recoverable if preemption interrupts a later
        # best-checkpoint write. This payload retains the historical schema.
        "best_checkpoint_payload": deepcopy(best_payload),
        "history": deepcopy(history),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "train_generator_state": train_generator.get_state(),
        "worker_rng_states": deepcopy(worker_states),
        "cumulative_training_seconds": cumulative_seconds,
    }
    _atomic_torch_save(payload, Path(path))


def load_shared_resume(path, config):
    """Validate the durable epoch before modifying any training artifacts."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    try:
        if state["format_version"] != 1 or any(
            state[key] != config["experiment"][key] for key in ("run_name", "seed")
        ) or state["config_identity"] != dict(config):
            raise ValueError("Shared checkpoint identity mismatch")
        epoch, history = state["completed_epoch"], state["history"]
        if type(epoch) is not int or not 1 <= epoch <= config["training"]["epochs"]:
            raise ValueError("Invalid Shared completed epoch")
        if not isinstance(history, list) or len(history) != epoch or [
            row["epoch"] for row in history
        ] != list(range(1, epoch + 1)):
            raise ValueError("Shared history is incomplete or duplicated")
        replay = SharedValidationEarlyStopping(config["training"]["early_stopping_patience"])
        for row in history:
            _, stop = replay.update(row["shared_validation_score"], row["epoch"])
            if stop and row["epoch"] != epoch:
                raise ValueError("Shared history continues past early stopping")
        if (replay.best_score, replay.best_epoch, replay.epochs_without_improvement) != (
            state["best_shared_validation_score"], state["best_epoch"], state["epochs_without_improvement"],
        ):
            raise ValueError("Shared early stopping state disagrees with history")
        metrics = state["best_validation_metrics"]
        for key in ("val_task1_macro_f1", "val_task2_macro_f1", "val_task3_macro_f1", "shared_validation_score"):
            if metrics[key] != history[replay.best_epoch - 1][key]:
                raise ValueError("Shared best metrics disagree with history")
        best = state["best_checkpoint_payload"]
        if best["epoch"] != replay.best_epoch or best["seed"] != state["seed"] or best["shared_validation_score"] != replay.best_score:
            raise ValueError("Shared best checkpoint disagrees with history")
        seconds = state["cumulative_training_seconds"]
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid Shared cumulative training time")
        for key in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict"):
            if not isinstance(state[key], dict):
                raise ValueError(f"Invalid Shared {key}")
        if state["scheduler_state_dict"].get("last_epoch") != epoch:
            raise ValueError("Shared scheduler disagrees with completed epoch")
        if best["config_metadata"]["experiment"] != config["experiment"]:
            raise ValueError("Shared best checkpoint identity mismatch")
        for key in ("torch_cpu_rng_state", "train_generator_state"):
            state[key] = _require_cpu_byte_rng_state(state[key], key)
        if not isinstance(state["torch_cuda_rng_states"], (list, tuple)):
            raise ValueError("Invalid Shared CUDA RNG states")
        state["torch_cuda_rng_states"] = [
            _require_cpu_byte_rng_state(value, "torch_cuda_rng_states")
            for value in state["torch_cuda_rng_states"]
        ]
        random.Random().setstate(state["python_rng_state"])
        np.random.RandomState().set_state(state["numpy_rng_state"])
        torch.Generator().set_state(state["train_generator_state"])
        torch.Generator().set_state(state["torch_cpu_rng_state"])
        workers = state["worker_rng_states"]
        if not isinstance(workers, dict) or set(workers) != set(range(config["loader"]["num_workers"])):
            raise ValueError("Shared worker RNG states are incomplete")
        for worker in workers.values():
            random.Random().setstate(worker["python"])
            np.random.RandomState().set_state(worker["numpy"])
            worker["torch"] = _require_cpu_byte_rng_state(worker["torch"], "worker_rng_state")
            torch.Generator().set_state(worker["torch"])
    except (KeyError, TypeError, IndexError, AttributeError, RuntimeError) as error:
        raise ValueError(f"Invalid Shared resume state: {error}") from error
    return state


def restore_shared_resume(state, *, model, optimizer, scheduler, scaler, control, train_generator):
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    scaler.load_state_dict(state["scaler_state_dict"])
    control.best_score = state["best_shared_validation_score"]
    control.best_epoch = state["best_epoch"]
    control.epochs_without_improvement = state["epochs_without_improvement"]
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(_require_cpu_byte_rng_state(state["torch_cpu_rng_state"], "torch_cpu_rng_state"))
    if torch.cuda.is_available():
        if len(state["torch_cuda_rng_states"]) != torch.cuda.device_count():
            raise ValueError("Shared checkpoint CUDA RNG device count differs")
        torch.cuda.set_rng_state_all([
            _require_cpu_byte_rng_state(value, "torch_cuda_rng_states")
            for value in state["torch_cuda_rng_states"]
        ])
    train_generator.set_state(_require_cpu_byte_rng_state(state["train_generator_state"], "train_generator_state"))
