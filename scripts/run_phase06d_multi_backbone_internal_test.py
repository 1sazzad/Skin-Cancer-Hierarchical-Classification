#!/usr/bin/env python3
"""Gate 06D: one locked, architecture-aware internal-test comparison.

The script evaluates every frozen flat/shared backbone pair on the same untouched
ISIC 2019 internal-test split. It performs no training, tuning, threshold search,
or test-driven model selection. A --preflight-only mode verifies configuration,
checkpoint existence/hashes/epochs, model loadability, and CUDA availability
without constructing the internal-test dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.stored_prediction_statistics import bootstrap_table, exact_mcnemar, linear_quantile
from src.data.isic2019_dataset import ISIC2019HierarchicalDataset
from src.data.transforms import build_eval_transform
from src.evaluation.classification_metrics import compute_classification_metrics
from src.evaluation.hierarchical_evaluator import FINAL_CLASS_NAMES, build_hierarchical_routing
from src.evaluation.phase04_comparative_harness import (
    build_paired_four_class_rows,
    collect_shared_isic_predictions,
    collect_single_task_predictions,
)
from src.models.classification_backbone import SUPPORTED_CLASSIFICATION_ARCHITECTURES, build_classification_model
from src.models.shared_three_task import SUPPORTED_SHARED_ARCHITECTURES, build_shared_three_task_model
from src.utils.reproducibility import make_generator, seed_worker

EXPECTED_BACKBONES = (
    "efficientnet_b0",
    "densenet121",
    "densenet169",
    "resnet50",
    "mobilenet_v3_large",
    "efficientnet_b2",
    "efficientnet_b3",
)
CLASS_NAMES = ("non_malignant", "melanoma", "bcc", "scc")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Gate 06D config must be a mapping.")
    if raw.get("execution_split") != "internal_test" or raw.get("internal_test_execution_allowed") is not True:
        raise ValueError("Gate 06D config must explicitly authorize internal_test execution.")
    protocol = raw.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("Missing Gate 06D protocol mapping.")
    required_false = ("candidate_switching_after_test", "retraining_after_test", "test_tuning_allowed")
    if any(protocol.get(key) is not False for key in required_false):
        raise ValueError("Gate 06D protocol must prohibit switching, retraining, and test tuning.")
    if protocol.get("report_all_backbones") is not True:
        raise ValueError("Gate 06D must report all frozen backbones.")
    backbones = raw.get("backbones")
    if not isinstance(backbones, dict) or tuple(backbones) != EXPECTED_BACKBONES:
        raise ValueError(f"Backbone order/set must be exactly {EXPECTED_BACKBONES!r}.")
    return raw


def resolve_checkpoint(project_root: Path, phase02_root: Path, item: Mapping[str, object]) -> Path:
    root_name = str(item["root"])
    base = project_root if root_name == "project" else phase02_root if root_name == "phase02" else None
    if base is None:
        raise ValueError(f"Unsupported checkpoint root {root_name!r}.")
    path = (base / str(item["path"])).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    expected = str(item["sha256"]).lower()
    if observed.lower() != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {observed}")
    return path


def checkpoint_payload(path: Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "model_state_dict" not in payload:
        raise ValueError(f"Invalid checkpoint payload: {path}")
    return payload


def validate_epoch(payload: Mapping[str, Any], expected: int, path: Path) -> None:
    observed = int(payload.get("epoch", -1))
    if observed != expected:
        raise ValueError(f"Epoch mismatch for {path}: expected {expected}, got {observed}")


def load_flat(architecture: str, path: Path, expected_epoch: int) -> torch.nn.Module:
    payload = checkpoint_payload(path)
    validate_epoch(payload, expected_epoch, path)
    stored_names = payload.get("class_names")
    if stored_names is not None and tuple(stored_names) != CLASS_NAMES:
        raise ValueError(f"Flat class order mismatch in {path}: {stored_names!r}")
    model = build_classification_model(
        architecture, 4, pretrained="none", dropout_probability=0.2
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.eval()


def load_shared(architecture: str, path: Path, expected_epoch: int) -> torch.nn.Module:
    payload = checkpoint_payload(path)
    validate_epoch(payload, expected_epoch, path)
    model = build_shared_three_task_model(
        architecture, pretrained="none", dropout_probability=0.2
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.eval()


def test_loader(dataset: ISIC2019HierarchicalDataset, cfg: Mapping[str, object], seed: int) -> DataLoader:
    num_workers = int(cfg["num_workers"])
    kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_size": int(cfg["batch_size"]),
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": bool(cfg["pin_memory"]),
        "drop_last": False,
        "worker_init_fn": seed_worker,
        "generator": make_generator(seed),
        "persistent_workers": bool(cfg["persistent_workers"]),
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(cfg["prefetch_factor"])
    return DataLoader(**kwargs)


def metrics_dict(target: np.ndarray, pred: np.ndarray) -> dict[str, object]:
    return compute_classification_metrics(target, pred, FINAL_CLASS_NAMES)


def statistical_comparison(target: np.ndarray, flat: np.ndarray, hierarchy: np.ndarray, *, seed: int) -> dict[str, object]:
    boot = bootstrap_table(target, flat, hierarchy, replicate_count=10000, seed=seed)
    result: dict[str, object] = {}
    flat_metrics = metrics_dict(target, flat)
    hierarchy_metrics = metrics_dict(target, hierarchy)
    for metric in ("macro_f1", "accuracy"):
        values = -boot[f"difference_{metric}"].to_numpy(dtype=np.float64)
        lower, upper = linear_quantile(values, [0.025, 0.975])
        result[metric] = {
            "flat": float(flat_metrics[metric]),
            "hierarchy": float(hierarchy_metrics[metric]),
            "delta_hierarchy_minus_flat": float(hierarchy_metrics[metric]) - float(flat_metrics[metric]),
            "paired_bootstrap_95ci": [float(lower), float(upper)],
        }
    raw = exact_mcnemar(flat == target, hierarchy == target)
    result["mcnemar"] = raw
    return result


def preflight(config: Mapping[str, Any], project_root: Path, phase02_root: Path, device: torch.device) -> dict[str, object]:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    if set(EXPECTED_BACKBONES) - set(SUPPORTED_CLASSIFICATION_ARCHITECTURES):
        raise RuntimeError("Flat model factory does not support all Gate 06D architectures.")
    if set(EXPECTED_BACKBONES) - set(SUPPORTED_SHARED_ARCHITECTURES):
        raise RuntimeError("Shared model factory does not support all Gate 06D architectures.")
    records = {}
    for architecture in EXPECTED_BACKBONES:
        pair = config["backbones"][architecture]
        flat_path = resolve_checkpoint(project_root, phase02_root, pair["flat"])
        shared_path = resolve_checkpoint(project_root, phase02_root, pair["shared"])
        flat = load_flat(architecture, flat_path, int(pair["flat"]["expected_epoch"]))
        shared = load_shared(architecture, shared_path, int(pair["shared"]["expected_epoch"]))
        records[architecture] = {
            "flat_path": str(flat_path),
            "flat_sha256": str(pair["flat"]["sha256"]),
            "shared_path": str(shared_path),
            "shared_sha256": str(pair["shared"]["sha256"]),
            "flat_parameters": int(sum(p.numel() for p in flat.parameters())),
            "shared_parameters": int(sum(p.numel() for p in shared.parameters())),
        }
        del flat, shared
    return {
        "status": "PASS",
        "internal_test_constructed": False,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "backbones": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--phase02-root", type=Path, default=Path.home() / "projects/Skin-Cancer-Hierarchical-Classification-phase02")
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation/phase06d_multi_backbone_locked_internal_test.yaml"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    phase02_root = args.phase02_root.expanduser().resolve()
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    config = load_config(config_path.resolve())
    device = torch.device(args.device)

    audit = preflight(config, project_root, phase02_root, device)
    if args.preflight_only:
        print(json.dumps(audit, indent=2, sort_keys=True))
        return

    manifest = (project_root / str(config["isic"]["manifest_path"])).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    dataset = ISIC2019HierarchicalDataset(
        manifest, project_root, "internal_test", "flat_four_class", build_eval_transform()
    )
    loader = test_loader(dataset, config["loader"], int(config["seed"]))
    output_root = project_root / str(config["outputs"]["directory"])
    output_root.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, object] = {}
    for architecture in EXPECTED_BACKBONES:
        print(f"=== {architecture}: locked internal-test inference ===", flush=True)
        pair = config["backbones"][architecture]
        flat_path = resolve_checkpoint(project_root, phase02_root, pair["flat"])
        shared_path = resolve_checkpoint(project_root, phase02_root, pair["shared"])

        flat_model = load_flat(architecture, flat_path, int(pair["flat"]["expected_epoch"]))
        flat_collection = collect_single_task_predictions(
            flat_model, loader, class_names=CLASS_NAMES, device=device
        )
        flat_model.to("cpu")
        del flat_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        shared_model = load_shared(architecture, shared_path, int(pair["shared"]["expected_epoch"]))
        shared_collection = collect_shared_isic_predictions(shared_model, loader, device=device)
        shared_model.to("cpu")
        del shared_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if flat_collection.sample_ids != shared_collection.sample_ids:
            raise ValueError(f"Sample order mismatch for {architecture}")
        routing = build_hierarchical_routing(
            shared_collection.stage1_targets,
            shared_collection.stage1_predictions,
            shared_collection.stage2_targets,
            shared_collection.stage2_predictions,
        )
        if not np.array_equal(flat_collection.targets, routing.final_targets):
            raise ValueError(f"Ground-truth mismatch for {architecture}")

        flat_metrics = metrics_dict(flat_collection.targets, flat_collection.predictions)
        predicted_metrics = metrics_dict(routing.final_targets, routing.predicted_gate_predictions)
        oracle_metrics = metrics_dict(routing.final_targets, routing.oracle_gate_predictions)
        rows = build_paired_four_class_rows(shared_collection, flat_collection)
        paired_path = output_root / architecture / "paired_internal_test_predictions.csv"
        paired_path.parent.mkdir(parents=True, exist_ok=True)
        with paired_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        stats = statistical_comparison(
            routing.final_targets,
            flat_collection.predictions,
            routing.predicted_gate_predictions,
            seed=int(config["seed"]),
        )
        architecture_result = {
            "architecture": architecture,
            "sample_count": len(rows),
            "flat": flat_metrics,
            "shared_predicted_gate": predicted_metrics,
            "shared_oracle_gate": oracle_metrics,
            "routing_loss_macro_f1": float(oracle_metrics["macro_f1"]) - float(predicted_metrics["macro_f1"]),
            "statistics_hierarchy_minus_flat": stats,
            "checkpoint_provenance": {
                "flat": {"path": str(flat_path), "sha256": str(pair["flat"]["sha256"]), "epoch": int(pair["flat"]["expected_epoch"])},
                "shared": {"path": str(shared_path), "sha256": str(pair["shared"]["sha256"]), "epoch": int(pair["shared"]["expected_epoch"])},
            },
            "paired_predictions": str(paired_path),
            "paired_predictions_sha256": sha256_file(paired_path),
        }
        write_json(output_root / architecture / "metrics_and_statistics.json", architecture_result)
        all_results[architecture] = architecture_result
        print(
            f"{architecture}: flat={flat_metrics['macro_f1']:.6f} "
            f"shared={predicted_metrics['macro_f1']:.6f} "
            f"oracle={oracle_metrics['macro_f1']:.6f}",
            flush=True,
        )

    summary = {
        "gate": "06D",
        "status": "PASS",
        "execution_split": "internal_test",
        "internal_test_executed": True,
        "selection_basis": "validation_only",
        "candidate_switching_after_test": False,
        "test_tuning_allowed": False,
        "report_all_backbones": True,
        "seed": int(config["seed"]),
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path.resolve()),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "preflight": audit,
        "results": all_results,
    }
    write_json(output_root / "gate06d_locked_internal_test_summary.json", summary)
    print(f"PASS: Gate 06D evaluated all {len(EXPECTED_BACKBONES)} frozen backbone pairs.")


if __name__ == "__main__":
    main()
