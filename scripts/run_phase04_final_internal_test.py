#!/usr/bin/env python3
"""Frozen final internal-test comparative evaluation.

This entry point deliberately constructs internal-test datasets/loaders directly.
It executes the untouched internal-test split once under the frozen protocol.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src.data.emb_stage03 import EMBStage03Dataset
from src.data.isic2019_dataset import ISIC2019HierarchicalDataset
from src.data.transforms import build_eval_transform
from src.evaluation.phase04_comparative_harness import (
    FrozenCheckpointSpec,
    benchmark_inference,
    build_paired_four_class_rows,
    checkpoint_bytes,
    collect_shared_isic_predictions,
    collect_single_task_predictions,
    count_parameters,
    environment_provenance,
    evaluate_prediction_collection,
    evaluate_shared_isic,
    load_frozen_shared_model,
    load_frozen_single_task_model,
    optional_macs_flops,
    write_json,
    write_paired_prediction_csv,
)
from src.evaluation.shared_task_head_adapter import SharedTaskHeadAdapter
from src.utils.reproducibility import make_generator, seed_worker


REQUIRED_CHECKPOINTS = {"shared", "task1", "task2", "task3", "flat"}
REQUIRED_MANIFEST_KEYS = {"isic2019", "task3"}


def _spec(name: str, raw: Mapping[str, object]) -> FrozenCheckpointSpec:
    return FrozenCheckpointSpec(
        name=name,
        path=str(raw["path"]),
        sha256=str(raw["sha256"]),
        expected_epoch=int(raw["expected_epoch"]),
        model_kind=str(raw["model_kind"]),
        class_names=tuple(str(value) for value in raw["class_names"]),
    )


def _load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Phase04 config must be a mapping.")
    if raw.get("execution_split") != "internal_test":
        raise ValueError("Final runner requires execution_split=internal_test.")
    if not bool(raw.get("internal_test_execution_allowed")):
        raise ValueError("Final runner requires explicit internal-test authorization.")

    isic = raw.get("isic")
    if not isinstance(isic, dict):
        raise ValueError("Phase04 config must contain an isic mapping.")
    manifests = isic.get("manifest_paths")
    if not isinstance(manifests, dict) or set(manifests) != REQUIRED_MANIFEST_KEYS:
        raise ValueError(
            "Final test requires exactly isic2019/task3 authoritative manifest paths."
        )

    checkpoints = raw.get("checkpoints")
    if not isinstance(checkpoints, dict) or set(checkpoints) != REQUIRED_CHECKPOINTS:
        raise ValueError("Phase04 requires exactly shared/task1/task2/task3/flat checkpoints.")
    return raw


def _resolve_existing(root: Path, value: object, description: str) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def _test_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    seed: int,
) -> DataLoader:
    if batch_size <= 0 or num_workers < 0 or prefetch_factor <= 0:
        raise ValueError("Invalid final-test loader settings.")
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers requires num_workers > 0.")
    kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
        "worker_init_fn": seed_worker,
        "generator": make_generator(seed),
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


def _release_cuda(model: torch.nn.Module, device: torch.device) -> None:
    if device.type == "cuda":
        model.to("cpu")
        torch.cuda.empty_cache()


def _assert_unique_sample_ids(rows: list[dict[str, object]]) -> None:
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Paired internal-test export contains duplicate sample IDs.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evaluation/phase04_controlled_comparative_internal_test.yaml"),
    )
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/phase04_controlled_comparative/final_internal_test"),
    )
    args = parser.parse_args()

    root = args.project_root.expanduser().resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config_path = config_path.resolve()
    config = _load_config(config_path)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    isic_cfg = config["isic"]
    manifests_cfg = isic_cfg["manifest_paths"]
    isic_manifest = _resolve_existing(
        root, manifests_cfg["isic2019"], "Frozen ISIC2019 split manifest"
    )
    task3_manifest = _resolve_existing(
        root, manifests_cfg["task3"], "Frozen ISIC-derived Task3 split manifest"
    )

    loader_cfg = config["loader"]
    seed = int(config["seed"])
    loader_kwargs = {
        "batch_size": int(loader_cfg["batch_size"]),
        "num_workers": int(loader_cfg["num_workers"]),
        "pin_memory": bool(loader_cfg["pin_memory"]),
        "persistent_workers": bool(loader_cfg["persistent_workers"]),
        "prefetch_factor": int(loader_cfg["prefetch_factor"]),
        "seed": seed,
    }

    # Frozen internal-test datasets. No training or validation dataset is constructed here.
    eval_transform = build_eval_transform()
    flat_dataset = ISIC2019HierarchicalDataset(
        isic_manifest, root, "internal_test", "flat_four_class", eval_transform
    )
    task1_dataset = ISIC2019HierarchicalDataset(
        isic_manifest, root, "internal_test", "stage_1", eval_transform
    )
    task2_dataset = ISIC2019HierarchicalDataset(
        isic_manifest, root, "internal_test", "stage_2", eval_transform
    )
    task3_dataset = EMBStage03Dataset(
        task3_manifest, root, "internal_test", eval_transform
    )

    flat_loader = _test_loader(flat_dataset, **loader_kwargs)
    task1_loader = _test_loader(task1_dataset, **loader_kwargs)
    task2_loader = _test_loader(task2_dataset, **loader_kwargs)
    task3_loader = _test_loader(task3_dataset, **loader_kwargs)

    raw_checkpoints = config["checkpoints"]
    specs = {name: _spec(name, raw_checkpoints[name]) for name in REQUIRED_CHECKPOINTS}

    # Load checkpoint payloads on CPU first so evaluation/benchmark GPU residency can
    # be controlled model-by-model.
    models = {
        "shared": load_frozen_shared_model(specs["shared"], root, "cpu"),
        "task1": load_frozen_single_task_model(specs["task1"], root, "cpu"),
        "task2": load_frozen_single_task_model(specs["task2"], root, "cpu"),
        "task3": load_frozen_single_task_model(specs["task3"], root, "cpu"),
        "flat": load_frozen_single_task_model(specs["flat"], root, "cpu"),
    }

    shared_isic = collect_shared_isic_predictions(models["shared"], flat_loader, device=device)
    shared_task3 = collect_single_task_predictions(
        SharedTaskHeadAdapter(models["shared"], "task3"),
        task3_loader,
        class_names=tuple(isic_cfg["task3_class_names"]),
        device=device,
    )
    shared_metrics = evaluate_shared_isic(shared_isic)
    shared_metrics["task3"] = evaluate_prediction_collection(
        shared_task3, tuple(isic_cfg["task3_class_names"])
    )
    _release_cuda(models["shared"], device)

    task1_predictions = collect_single_task_predictions(
        models["task1"],
        task1_loader,
        class_names=tuple(isic_cfg["task1_class_names"]),
        device=device,
    )
    task1_metrics = evaluate_prediction_collection(
        task1_predictions, tuple(isic_cfg["task1_class_names"])
    )
    _release_cuda(models["task1"], device)

    task2_predictions = collect_single_task_predictions(
        models["task2"],
        task2_loader,
        class_names=tuple(isic_cfg["task2_class_names"]),
        device=device,
    )
    task2_metrics = evaluate_prediction_collection(
        task2_predictions, tuple(isic_cfg["task2_class_names"])
    )
    _release_cuda(models["task2"], device)

    task3_predictions = collect_single_task_predictions(
        models["task3"],
        task3_loader,
        class_names=tuple(isic_cfg["task3_class_names"]),
        device=device,
    )
    task3_metrics = evaluate_prediction_collection(
        task3_predictions, tuple(isic_cfg["task3_class_names"])
    )
    _release_cuda(models["task3"], device)

    flat_predictions = collect_single_task_predictions(
        models["flat"],
        flat_loader,
        class_names=tuple(isic_cfg["flat_class_names"]),
        device=device,
    )
    flat_metrics = evaluate_prediction_collection(
        flat_predictions, tuple(isic_cfg["flat_class_names"])
    )
    _release_cuda(models["flat"], device)

    paired_rows = build_paired_four_class_rows(shared_isic, flat_predictions)
    _assert_unique_sample_ids(paired_rows)
    required_paired_columns = [str(value) for value in config["statistics_export"]["paired_columns"]]
    if list(paired_rows[0]) != required_paired_columns:
        raise ValueError(
            "Paired export schema differs from the frozen final-test config: "
            f"expected {required_paired_columns}, got {list(paired_rows[0])}."
        )

    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir = output_dir.resolve()
    paired_path = output_dir / "paired_internal_test_predictions.csv"
    write_paired_prediction_csv(paired_path, paired_rows)

    efficiency_cfg = config["efficiency"]
    benchmark_kwargs = {
        "device": device,
        "input_shape": tuple(int(value) for value in efficiency_cfg["input_shape"]),
        "batch_size": int(efficiency_cfg["batch_size"]),
        "warmup_iterations": int(efficiency_cfg["warmup_iterations"]),
        "measured_iterations": int(efficiency_cfg["measured_iterations"]),
    }
    efficiency: dict[str, object] = {}
    for name in ("shared", "task1", "task2", "task3", "flat"):
        model = models[name]
        benchmark = benchmark_inference(model, **benchmark_kwargs)
        _release_cuda(model, device)
        efficiency[name] = {
            **count_parameters(model),
            "checkpoint_bytes": checkpoint_bytes(specs[name], root),
            "benchmark": benchmark,
            "compute": optional_macs_flops(model),
        }

    paired_sample_ids = [str(row["sample_id"]) for row in paired_rows]
    payload = {
        "gate": "FINAL_INTERNAL_TEST",
        "status": "PASS",
        "execution_split": "internal_test",
        "internal_test_executed": True,
        "config_path": str(config_path),
        "environment": environment_provenance(device),
        "manifests": {
            "isic2019": str(isic_manifest),
            "task3": str(task3_manifest),
        },
        "internal_test_sample_counts": {
            "flat_four_class": len(flat_dataset),
            "task1": len(task1_dataset),
            "task2_malignant_subset": len(task2_dataset),
            "task3": len(task3_dataset),
        },
        "metrics": {
            "shared": shared_metrics,
            "standalone_task1": task1_metrics,
            "standalone_task2": task2_metrics,
            "standalone_task3": task3_metrics,
            "flat_four_class": flat_metrics,
        },
        "collection_elapsed_seconds": {
            "shared_isic": shared_isic.elapsed_seconds,
            "shared_task3": shared_task3.elapsed_seconds,
            "standalone_task1": task1_predictions.elapsed_seconds,
            "standalone_task2": task2_predictions.elapsed_seconds,
            "standalone_task3": task3_predictions.elapsed_seconds,
            "flat_four_class": flat_predictions.elapsed_seconds,
        },
        "paired_export": {
            "path": str(paired_path),
            "row_count": len(paired_rows),
            "columns": required_paired_columns,
            "unique_sample_ids": len(set(paired_sample_ids)) == len(paired_sample_ids),
            "shared_flat_order_identical": shared_isic.sample_ids == flat_predictions.sample_ids,
            "ground_truth_identical": all(
                int(row["true_label"]) == int(flat_predictions.targets[index])
                for index, row in enumerate(paired_rows)
            ),
        },
        "efficiency": efficiency,
    }
    summary_path = output_dir / "final_internal_test_summary.json"
    write_json(summary_path, payload)
    print(f"Frozen internal-test evaluation PASS: {summary_path}")
    print("internal_test_executed=true")


if __name__ == "__main__":
    main()
