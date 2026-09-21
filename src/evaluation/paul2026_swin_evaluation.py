"""Locked, resumable PAUL 2026 ISIC evaluation. No training entry points.

Preflight reads checkpoints, summaries, and manifest bytes only. Production
verifies that immutable lock before constructing any internal-test dataset.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.analysis.stored_prediction_statistics import bootstrap_table, exact_mcnemar, linear_quantile
from src.data.isic2019_dataset import ISIC2019HierarchicalDataset
from src.data.transforms import TransformConfig, build_eval_transform
from src.evaluation.classification_metrics import compute_classification_metrics
from src.evaluation.hierarchical_evaluator import (
    FINAL_CLASS_NAMES, STAGE_1_CLASS_NAMES, STAGE_2_CLASS_NAMES, build_hierarchical_routing,
)
from src.evaluation.phase04_comparative_harness import (
    build_paired_four_class_rows, collect_shared_isic_predictions,
    collect_single_task_predictions,
)
from src.models.shared_three_task import TASK_CLASS_MAPPINGS
from src.models.transformer_extension import (
    build_swin_t_classification_model, build_swin_t_shared_three_task_model,
)

SEEDS = (42, 123, 2026)
CLASS_NAMES = ("non_malignant", "melanoma", "bcc", "scc")
EXPECTED_N = 3668
MANIFEST = Path("data/manifests/isic2019_train_val_test_split_seed42.csv")
EXTENSION = Path("results/extensions/paul2026_swin")
EVALUATION = EXTENSION / "evaluation"
LOCK_NAME = "preflight_checkpoint_lock.json"
METRIC_NAMES = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
PROTOCOL = {
    "mode": "locked_internal_test_evaluation", "system": "flat_shared_hard_oracle",
    "backbone": "swin_t", "seeds": list(SEEDS), "dataset": "ISIC2019",
    "split": "internal_test", "task": "flat_four_class", "sample_count_expected": EXPECTED_N,
    "class_names": list(CLASS_NAMES), "resize": 256, "center_crop": 224,
    "normalization_mean": [0.485, 0.456, 0.406], "normalization_std": [0.229, 0.224, 0.225],
    "augmentation": False, "batch_size": 64, "num_workers": 4,
    "shuffle": False, "drop_last": False, "bootstrap_replicates": 10000,
    "statistical_seed": 42, "confidence_level": 0.95, "std_ddof": 1,
    "model_selection_from_test": False, "tuning_from_test": False, "ensemble": False,
    "inference_autocast": "float16_on_cuda_float32_on_cpu",
}


@dataclass(frozen=True)
class CheckpointSpec:
    system: str
    seed: int
    epoch: int

    @property
    def run_name(self):
        return f"paul2026_{self.system}_swin_t_seed{self.seed}"

    def path(self, root):
        return Path(root) / EXTENSION / "runs" / self.run_name / "best_checkpoint.pt"


CHECKPOINTS = tuple(
    CheckpointSpec(system, seed, epoch)
    for system, epochs in (("flat", (9, 10, 17)), ("shared_hard", (10, 1, 26)))
    for seed, epoch in zip(SEEDS, epochs)
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _publish_json(path, value):
    """Atomically publish under the deployed function's single-writer contract.

    Existing compatible artifacts are verified without rewriting. Callers must
    not run concurrent local evaluations; the launcher busy check is advisory.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require_equal(_read_json(path), value, f"Incompatible existing artifact: {path}")
        return
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            _require_equal(_read_json(path), value, f"Incompatible existing artifact: {path}")
            return
        temporary.rename(path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_equal(actual, expected, label):
    if actual != expected or (type(expected) in (int, bool) and type(actual) is not type(expected)):
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def _validate_metadata(metadata, spec, label):
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Malformed {label}")
    expected = {
        "run_name": spec.run_name, "seed": spec.seed, "architecture": "swin_t",
        "backbone": "swin_t", "system": spec.system, "dropout_probability": 0.2,
        "research_stage": "paul2026_swin_extension",
    }
    for key, value in expected.items():
        if key in metadata:
            _require_equal(metadata[key], value, f"{label}.{key}")
    if spec.system == "flat":
        for key, value in (("task", "flat_four_class"), ("number_of_classes", 4),
                           ("split_manifest", MANIFEST.as_posix())):
            if key in metadata:
                _require_equal(metadata[key], value, f"{label}.{key}")
    elif "isic2019_manifest" in metadata:
        _require_equal(metadata["isic2019_manifest"], MANIFEST.as_posix(), f"{label}.isic2019_manifest")
    if "class_names" in metadata:
        _require_equal(list(metadata["class_names"]), list(CLASS_NAMES), f"{label}.class_names")
    if "class_to_index" in metadata:
        _require_equal(metadata["class_to_index"], dict(zip(CLASS_NAMES, range(4))), f"{label}.class_to_index")
    if "class_mappings" in metadata:
        _require_equal(metadata["class_mappings"], {k: dict(v) for k, v in TASK_CLASS_MAPPINGS.items()}, f"{label}.class_mappings")
    for key in ("config", "config_metadata", "config_identity", "scientific_identity",
                "model_metadata", "model", "experiment", "data", "resume_state"):
        if key in metadata:
            if key == "model" and isinstance(metadata[key], str):
                _require_equal(metadata[key], "swin_t", f"{label}.model")
            else:
                _validate_metadata(metadata[key], spec, f"{label}.{key}")


def validate_checkpoint_and_summary(payload, summary, spec):
    if spec not in CHECKPOINTS:
        raise ValueError("Checkpoint is not one of the six predeclared selections")
    if not isinstance(payload, Mapping) or "model_state_dict" not in payload:
        raise ValueError("Checkpoint must contain model_state_dict")
    _require_equal(payload.get("epoch"), spec.epoch, "checkpoint.epoch")
    for key, expected in (("reportable_as_full_result", True), ("run_name", spec.run_name),
                          ("best_epoch", spec.epoch)):
        _require_equal(summary.get(key), expected, f"run_summary.{key}")
    # Historical Flat summaries omit seed; Shared-Hard summaries must include it.
    if spec.system != "flat" or "seed" in summary:
        _require_equal(summary.get("seed"), spec.seed, "run_summary.seed")
    _validate_metadata(payload, spec, "checkpoint")
    _validate_metadata(summary, spec, "run_summary")
    if spec.system == "flat":
        # Require checkpoint identity even when the summary also supplies a seed.
        config = payload.get("config")
        if not isinstance(config, Mapping):
            raise ValueError("Missing or malformed checkpoint.config")
        experiment = config.get("experiment")
        if not isinstance(experiment, Mapping):
            raise ValueError("Missing or malformed checkpoint.config.experiment")
        for key, expected in (("seed", spec.seed), ("run_name", spec.run_name)):
            _require_equal(experiment.get(key), expected, f"checkpoint.config.experiment.{key}")


def load_selected_model(root, spec, *, expected_sha256=None):
    path = spec.path(root)
    before = sha256_file(path)
    if expected_sha256 is not None:
        _require_equal(before, expected_sha256, f"Checkpoint hash drift: {path}")
    summary = _read_json(path.parent / "run_summary.json")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_checkpoint_and_summary(payload, summary, spec)
    if spec.system == "flat":
        model = build_swin_t_classification_model(4, pretrained="none", dropout_probability=0.2)
    else:
        model = build_swin_t_shared_three_task_model(pretrained="none", dropout_probability=0.2)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    _require_equal(sha256_file(path), before, "Checkpoint changed during loading")
    return model.eval()


def _inspect_checkpoints(root, *, require_cuda=False):
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if tuple(FINAL_CLASS_NAMES) != CLASS_NAMES:
        raise ValueError("Historical four-class ordering changed")
    manifest = root / MANIFEST
    manifest_hash = sha256_file(manifest)
    records = []
    for spec in CHECKPOINTS:
        path = spec.path(root)
        summary_path = path.parent / "run_summary.json"
        summary_hash = sha256_file(summary_path)
        checkpoint_hash = sha256_file(path)
        model = load_selected_model(root, spec, expected_sha256=checkpoint_hash)
        _require_equal(sha256_file(summary_path), summary_hash, "Summary changed during preflight")
        records.append({
            "system": spec.system, "seed": spec.seed, "run_name": spec.run_name,
            "selected_epoch": spec.epoch, "checkpoint_path": str(path),
            "sha256": checkpoint_hash, "byte_size": path.stat().st_size,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "run_summary_sha256": summary_hash,
        })
        del model
    _require_equal(sha256_file(manifest), manifest_hash, "Manifest changed during preflight")
    return {
        "schema_version": 1, "status": "PASS", "created_before_test_execution": True,
        "internal_test_dataset_constructed": False, "manifest_path": str(manifest),
        "manifest_sha256": manifest_hash, "class_names": list(CLASS_NAMES),
        "protocol": PROTOCOL, "checkpoints": records,
    }


def checkpoint_only_preflight(project_root, *, require_cuda=False, persist_callback=None):
    root = Path(project_root).resolve()
    lock_path = root / EVALUATION / LOCK_NAME
    if not lock_path.exists() and (root / EVALUATION / "internal_isic").exists():
        raise RuntimeError("Cannot create a before-test lock after evaluation artifacts exist")
    lock = _inspect_checkpoints(root, require_cuda=require_cuda)
    _publish_json(lock_path, lock)
    if persist_callback is not None:
        persist_callback()
    return lock


def verify_preflight_lock(project_root, *, require_cuda=False):
    root = Path(project_root).resolve()
    path = root / EVALUATION / LOCK_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Required checkpoint-only preflight lock missing: {path}")
    lock = _read_json(path)
    observed = _inspect_checkpoints(root, require_cuda=require_cuda)
    _require_equal(observed, lock, "Preflight lock drift (checkpoint, summary, manifest, or protocol)")
    return lock


def build_locked_isic_loader(project_root):
    root = Path(project_root).resolve()
    dataset = ISIC2019HierarchicalDataset(
        root / MANIFEST, root, "internal_test", "flat_four_class",
        build_eval_transform(TransformConfig(image_size=224, eval_resize_size=256)),
    )
    _require_equal(len(dataset), EXPECTED_N, "Internal-test sample count")
    return DataLoader(dataset, batch_size=64, num_workers=4, shuffle=False,
                      drop_last=False, generator=torch.Generator().manual_seed(42))


def paired_statistics(targets, flat, hard):
    boot = bootstrap_table(targets, flat, hard, replicate_count=10000, seed=42)
    result = {"replicates": 10000, "seed": 42, "confidence_level": 0.95,
              "direction": "hard_minus_flat", "reporting_diagnostics_only": True}
    for metric in ("macro_f1", "accuracy"):
        # Historical utility stores flat minus hierarchy. Reverse before quantiles.
        values = -boot[f"difference_{metric}"].to_numpy(dtype=np.float64)
        result[metric] = {"paired_bootstrap_95ci": linear_quantile(values, [0.025, 0.975]).tolist()}
    result["mcnemar"] = exact_mcnemar(flat == targets, hard == targets)
    return result


def evaluate_collections(flat, shared):
    rows = build_paired_four_class_rows(shared, flat)
    _require_equal(len(rows), EXPECTED_N, "Prediction sample count")
    if len(set(flat.sample_ids)) != EXPECTED_N:
        raise ValueError("Duplicate sample IDs")
    routing = build_hierarchical_routing(
        shared.stage1_targets, shared.stage1_predictions,
        shared.stage2_targets, shared.stage2_predictions,
    )
    flat_metrics = compute_classification_metrics(flat.targets, flat.predictions, CLASS_NAMES)
    hard = compute_classification_metrics(routing.final_targets, routing.predicted_gate_predictions, CLASS_NAMES)
    oracle = compute_classification_metrics(routing.final_targets, routing.oracle_gate_predictions, CLASS_NAMES)
    malignant = shared.stage1_targets == 1
    stats = paired_statistics(flat.targets, flat.predictions, routing.predicted_gate_predictions)
    for metric in ("macro_f1", "accuracy"):
        stats[metric].update(flat=flat_metrics[metric], hard=hard[metric],
                             delta_hard_minus_flat=hard[metric] - flat_metrics[metric])
    for index, row in enumerate(rows):
        row["hard_correct"] = row["shared_correct"]
        for prefix, probabilities in (("flat", flat.probabilities),
                                      ("stage1", shared.stage1_probabilities),
                                      ("stage2", shared.stage2_probabilities)):
            for class_index, value in enumerate(probabilities[index]):
                row[f"{prefix}_probability_{class_index}"] = float(value) if np.isfinite(value) else ""
    return {
        "sample_count": len(rows), "flat": flat_metrics, "hard": hard, "oracle": oracle,
        "task1": compute_classification_metrics(shared.stage1_targets, shared.stage1_predictions, STAGE_1_CLASS_NAMES),
        "task2_malignant_subset": compute_classification_metrics(
            shared.stage2_targets[malignant], shared.stage2_predictions[malignant], STAGE_2_CLASS_NAMES),
        "routing_counts": routing.routing_counts,
        "routing_loss_macro_f1": oracle["macro_f1"] - hard["macro_f1"],
        "delta_hard_minus_flat_macro_f1": hard["macro_f1"] - flat_metrics["macro_f1"],
        "statistics_hard_minus_flat": stats,
    }, rows


def _seed_provenance(lock, seed):
    return [record for record in lock["checkpoints"] if record["seed"] == seed]


def evaluate_one_seed(root, seed, lock, *, device):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("Undeclared seed")
    loader = build_locked_isic_loader(root)
    specs = {spec.system: spec for spec in CHECKPOINTS if spec.seed == seed}
    hashes = {record["system"]: record["sha256"] for record in _seed_provenance(lock, seed)}
    model = load_selected_model(root, specs["flat"], expected_sha256=hashes["flat"])
    flat = collect_single_task_predictions(model, loader, class_names=CLASS_NAMES, device=device)
    model.cpu()
    del model
    model = load_selected_model(root, specs["shared_hard"], expected_sha256=hashes["shared_hard"])
    shared = collect_shared_isic_predictions(model, loader, device=device)
    model.cpu()
    del model
    result, rows = evaluate_collections(flat, shared)
    result.update(seed=seed, checkpoint_provenance=_seed_provenance(lock, seed), protocol=PROTOCOL)
    return result, rows


def aggregate_three_seeds(results):
    if set(results) != set(SEEDS):
        raise ValueError("Aggregation requires exactly seeds 42, 123, 2026")

    def summarize(values):
        array = np.asarray(values, dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError("Non-finite seed metric")
        return {"individual_seed_values": dict(zip(map(str, SEEDS), map(float, array))),
                "mean": float(array.mean()), "sample_std": float(array.std(ddof=1))}

    return {
        "protocol": PROTOCOL,
        "systems": {system: {metric: summarize([results[s][system][metric] for s in SEEDS])
                              for metric in METRIC_NAMES} for system in ("flat", "hard", "oracle")},
        "hard_minus_flat_macro_f1": summarize([results[s]["delta_hard_minus_flat_macro_f1"] for s in SEEDS]),
        "oracle_minus_hard_macro_f1": summarize([results[s]["routing_loss_macro_f1"] for s in SEEDS]),
    }


def _validate_completed_seed(directory, seed, lock):
    marker = _read_json(directory / "seed_complete.json")
    expected_files = {"metrics_and_statistics.json", "paired_internal_test_predictions.csv"}
    _require_equal(marker.get("seed"), seed, "Completed seed identity")
    _require_equal(marker.get("checkpoint_provenance"), _seed_provenance(lock, seed), "Completed checkpoint provenance")
    _require_equal(marker.get("protocol"), PROTOCOL, "Completed protocol")
    _require_equal(marker.get("manifest_sha256"), lock["manifest_sha256"], "Completed manifest")
    _require_equal(set(marker.get("artifacts", {})), expected_files, "Completed artifact set")
    for name, digest in marker["artifacts"].items():
        _require_equal(sha256_file(directory / name), digest, f"Completed artifact drift: {name}")
    result = _read_json(directory / "metrics_and_statistics.json")
    _require_equal(result.get("seed"), seed, "Result seed")
    _require_equal(result.get("checkpoint_provenance"), _seed_provenance(lock, seed), "Result provenance")
    _require_equal(result.get("protocol"), PROTOCOL, "Result protocol")
    _require_equal(result.get("sample_count"), EXPECTED_N, "Result sample count")
    with (directory / "paired_internal_test_predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    _require_equal(len(rows), EXPECTED_N, "Stored CSV sample count")
    if len({row["sample_id"] for row in rows}) != EXPECTED_N:
        raise ValueError("Stored CSV sample IDs are not unique")
    return result


def _publish_seed(output, seed, result, rows, lock):
    destination = output / f"seed{seed}"
    if destination.exists():
        raise FileExistsError(f"Refusing to replace seed artifacts: {destination}")
    # A hidden staging directory may survive a crash. Only atomic directory
    # publication makes it a completed seed; abandoned staging is never resumed.
    staging = Path(tempfile.mkdtemp(prefix=f".seed{seed}.", dir=output))
    _publish_json(staging / "metrics_and_statistics.json", result)
    with (staging / "paired_internal_test_predictions.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    _publish_json(staging / "seed_complete.json", {
        "seed": seed, "checkpoint_provenance": _seed_provenance(lock, seed),
        "manifest_sha256": lock["manifest_sha256"], "protocol": PROTOCOL,
        "artifacts": {name: sha256_file(staging / name) for name in
                      ("metrics_and_statistics.json", "paired_internal_test_predictions.csv")},
    })
    _validate_completed_seed(staging, seed, lock)
    staging.rename(destination)


def run_isic_evaluation(project_root, *, device="cuda", persist_callback=None):
    root = Path(project_root).resolve()
    resolved_device = torch.device(device)
    if resolved_device.type not in ("cuda", "cpu"):
        raise ValueError("Evaluation device must be cpu or cuda")
    lock = verify_preflight_lock(root, require_cuda=resolved_device.type == "cuda")
    output = root / EVALUATION / "internal_isic"
    final_path = output / "evaluation_complete.json"
    results = {}
    # Validate ALL durable results before executing any missing seed.
    for seed in SEEDS:
        directory = output / f"seed{seed}"
        if directory.exists():
            results[seed] = _validate_completed_seed(directory, seed, lock)
    if final_path.exists():
        final = _read_json(final_path)
        _require_equal(final, _final_marker(output, lock), "Final completion marker")
        _require_equal(_read_json(output / "three_seed_summary.json"), aggregate_three_seeds(results), "Final aggregation")
        raise RuntimeError("All three seeds already completed; complete evaluation rerun refused")
    output.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = False
    for seed in SEEDS:
        if seed in results:
            continue
        result, rows = evaluate_one_seed(root, seed, lock, device=resolved_device)
        _publish_seed(output, seed, result, rows, lock)
        if persist_callback is not None:
            persist_callback()
        results[seed] = result
    summary = aggregate_three_seeds(results)
    _publish_json(output / "three_seed_summary.json", summary)
    _publish_json(final_path, _final_marker(output, lock))
    if persist_callback is not None:
        persist_callback()
    return summary


def _final_marker(output, lock):
    return {"status": "PASS", "seeds": list(SEEDS), "protocol": PROTOCOL,
            "preflight_lock_sha256": sha256_file(output.parent / LOCK_NAME),
            "manifest_sha256": lock["manifest_sha256"],
            "summary_sha256": sha256_file(output / "three_seed_summary.json"),
            "seed_completion_sha256": {str(seed): sha256_file(output / f"seed{seed}" / "seed_complete.json") for seed in SEEDS}}
