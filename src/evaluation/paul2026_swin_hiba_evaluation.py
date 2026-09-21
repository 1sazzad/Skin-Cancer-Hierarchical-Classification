"""Frozen three-seed Swin HIBA evaluation, reusing the paper's Gate 06F code.

Preflight hashes inputs and strictly loads CPU checkpoints, without collecting
predictions or reading performance artifacts. No historical outputs are written.
"""
from __future__ import annotations

import csv
import hashlib
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from scripts import run_phase06f_hiba_external as historical
from src.evaluation import paul2026_swin_evaluation as internal
from src.evaluation.classification_metrics import compute_classification_metrics
from src.evaluation.hierarchical_evaluator import (
    STAGE_1_CLASS_NAMES, STAGE_2_CLASS_NAMES, build_hierarchical_routing,
)
from src.evaluation.phase04_comparative_harness import build_paired_four_class_rows
from src.utils.reproducibility import make_generator, seed_worker

SEEDS = internal.SEEDS
OUTPUT = internal.EVALUATION / "hiba_external"
LOCK_NAME = "preflight_hiba_lock.json"
CONFIG = Path("configs/paper/evaluation/phase06f_hiba_external.yaml")
EXPECTED_N = 1232
PROTOCOL = {
    "dataset": "HIBA", "mode": "external_zero_shot", "backbone": "swin_t",
    "seeds": list(SEEDS), "systems": ["flat", "hard", "oracle"],
    "bootstrap_unit": "patient_cluster", "bootstrap_replicates": 5000,
    "statistical_seed": 42, "confidence_level": 0.95, "std_ddof": 1,
    "training": False, "finetuning": False, "threshold_fitting": False,
    "calibration_fitting": False, "preprocessing_changes": False,
    "model_selection_from_test": False, "tuning_from_test": False,
    "performance_driven_filtering": False, "ensemble": False,
    "checkpoint": "best_checkpoint.pt", "oracle": "historical_true_task1_gate",
}
SOURCE_FILES = (
    "scripts/run_phase06f_hiba_external.py",
    "src/data/transforms.py",
    "src/evaluation/classification_metrics.py",
    "src/evaluation/hierarchical_evaluator.py",
    "src/evaluation/phase04_comparative_harness.py",
    "src/evaluation/paul2026_swin_evaluation.py",
    "src/evaluation/paul2026_swin_hiba_evaluation.py",
    "src/models/transformer_extension.py",
    "src/models/shared_three_task.py",
    "src/utils/reproducibility.py",
)
eq = internal._require_equal
read_json = internal._read_json
publish_json = internal._publish_json
sha256_file = internal.sha256_file


def _config(root):
    cfg = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    for key in ("external_training_allowed", "external_finetuning_allowed",
                "external_threshold_tuning_allowed", "external_checkpoint_selection_allowed",
                "preprocessing_changes_allowed"):
        eq(cfg["protocol"].get(key), False, f"Zero-shot constraint {key}")
    eq(cfg["protocol"]["selection_basis"], "frozen_internal_validation_only", "Selection basis")
    eq(cfg["uncertainty"], {"unit": "patient_cluster", "bootstrap_replicates": 5000,
                           "confidence_level": 0.95, "seed": 42}, "Historical statistics")
    eq(cfg["preprocessing"], {"input_size": [224, 224],
        "transform": "deterministic_resize_256_center_crop_224",
        "normalization": "imagenet"}, "Historical preprocessing")
    eq(cfg["seed"], 42, "Historical loader seed")
    eq(cfg["loader"], {"batch_size": 64, "num_workers": 4, "pin_memory": True,
        "persistent_workers": True, "prefetch_factor": 2}, "Historical loader")
    eq(cfg["hiba"], {
        "manifest_path": "data/external/hiba/manifests/hiba_external_dermoscopic_4class_final.csv",
        "manifest_sha256": "a2f30f14a249d8acb2bd9f03884e5e41c773ddbee19910d5b739407521e3706a",
        "expected_rows": EXPECTED_N, "class_names": list(internal.CLASS_NAMES),
        "expected_class_counts": {"non_malignant": 696, "melanoma": 196, "bcc": 229, "scc": 111},
        "expected_unique_patients": 568, "verify_image_sha256": True,
    }, "Frozen HIBA cohort specification")
    return cfg


def _cohort(root, cfg):
    """Read the final frozen cohort directly; never refilter or remap diagnoses."""
    manifest = root / cfg["hiba"]["manifest_path"]
    eq(historical.sha256_csv_canonical_crlf(manifest), cfg["hiba"]["manifest_sha256"], "Frozen manifest hash")
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    eq(len(rows), EXPECTED_N, "HIBA sample count")
    eq(dict(Counter(r["target_label"] for r in rows)), cfg["hiba"]["expected_class_counts"], "Class counts")
    eq(len({r["patient_id"] for r in rows}), cfg["hiba"]["expected_unique_patients"], "Patient count")
    eq(len({r["isic_id"] for r in rows}), EXPECTED_N, "Unique sample IDs")
    for row in rows:
        if not row["patient_id"].strip() or not row["isic_id"].strip():
            raise ValueError("Empty patient or image identifier")
        eq(int(row["target_index"]), internal.CLASS_NAMES.index(row["target_label"]), "Label mapping")
        path = (root / row["image_path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("HIBA image path escapes project root")
        eq(sha256_file(path).lower(), row["image_sha256"].lower(), f"HIBA image hash: {path}")
    return rows


def _inspect(root):
    cfg_hash = sha256_file(root / CONFIG)
    cfg = _config(root)
    manifest = root / cfg["hiba"]["manifest_path"]
    manifest_hash = sha256_file(manifest)
    sources = {name: sha256_file(root / name) for name in SOURCE_FILES}
    rows = _cohort(root, cfg)
    # Reuse the exact completed ISIC checkpoint governance, including optional
    # Flat summary seed, mandatory config seed, and strict state_dict loading.
    checkpoints = internal._inspect_checkpoints(root, require_cuda=False)
    eq(sha256_file(root / CONFIG), cfg_hash, "Config changed during preflight")
    eq(sha256_file(manifest), manifest_hash, "Manifest changed during preflight")
    eq({name: sha256_file(root / name) for name in SOURCE_FILES}, sources, "Source drift")
    return {
        "schema_version": 1, "status": "PASS", "protocol": PROTOCOL,
        "inference_performed": False, "performance_calculated": False,
        "checkpoints": checkpoints["checkpoints"],
        "training_manifest_sha256": checkpoints["manifest_sha256"],
        "historical_config_sha256": cfg_hash, "source_sha256": sources,
        "manifest_sha256": cfg["hiba"]["manifest_sha256"],
        "manifest_checkout_sha256": manifest_hash,
        "cohort": rows, "historical_preprocessing": cfg["preprocessing"],
        "historical_loader": cfg["loader"], "historical_uncertainty": cfg["uncertainty"],
    }


def hiba_preflight(project_root, *, persist_callback=None):
    root = Path(project_root).resolve()
    output = root / OUTPUT
    path = output / LOCK_NAME
    if not path.exists() and output.exists() and any(output.iterdir()):
        raise RuntimeError("Cannot recreate before-evaluation lock after HIBA artifacts exist")
    lock = _inspect(root)
    publish_json(path, lock)
    if persist_callback is not None:
        persist_callback()
    return lock


def verify_hiba_lock(project_root):
    root = Path(project_root).resolve()
    path = root / OUTPUT / LOCK_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Required HIBA preflight lock missing: {path}")
    lock = read_json(path)
    eq(_inspect(root), lock, "HIBA lock/checkpoint/input/protocol drift")
    return lock


def build_hiba_loader(root):
    cfg = _config(root)
    dataset = historical.HIBADataset(root / cfg["hiba"]["manifest_path"], root, verify_hashes=True)
    lc = cfg["loader"]
    return DataLoader(dataset, batch_size=int(lc["batch_size"]), shuffle=False,
        num_workers=int(lc["num_workers"]), pin_memory=bool(lc["pin_memory"]),
        persistent_workers=bool(lc["persistent_workers"]), prefetch_factor=int(lc["prefetch_factor"]),
        worker_init_fn=seed_worker, generator=make_generator(int(cfg["seed"])))


def paired_statistics(targets, first, second, patient_ids, direction):
    return {"direction": direction, "unit": "patient_cluster", "replicates": 5000,
        "seed": 42, "confidence_level": 0.95,
        "macro_f1_delta_95ci": historical.patient_cluster_ci(
            targets, first, second, patient_ids, 5000, 42)}


def evaluate_collections(flat, shared, cohort):
    rows = build_paired_four_class_rows(shared, flat)
    eq(list(flat.sample_ids), [r["isic_id"] for r in cohort], "Frozen sample order")
    eq(flat.targets.tolist(), [int(r["target_index"]) for r in cohort], "Frozen targets")
    eq(len(rows), EXPECTED_N, "HIBA prediction count")
    routing = build_hierarchical_routing(shared.stage1_targets, shared.stage1_predictions,
                                         shared.stage2_targets, shared.stage2_predictions)
    eq(routing.final_targets.tolist(), flat.targets.tolist(), "Paired targets")
    predictions = {"flat": flat.predictions, "hard": routing.predicted_gate_predictions,
                   "oracle": routing.oracle_gate_predictions}
    result = {name: compute_classification_metrics(flat.targets, pred, internal.CLASS_NAMES)
              for name, pred in predictions.items()}
    patients = [r["patient_id"] for r in cohort]
    for i, row in enumerate(rows):
        row["patient_id"] = patients[i]
        for prefix, probabilities in (("flat", flat.probabilities),
                ("stage1", shared.stage1_probabilities), ("stage2", shared.stage2_probabilities)):
            for j, value in enumerate(probabilities[i]):
                row[f"{prefix}_probability_{j}"] = float(value) if np.isfinite(value) else ""
    malignant = shared.stage1_targets == 1
    result.update(sample_count=len(rows), patient_count=len(set(patients)),
        task1=compute_classification_metrics(shared.stage1_targets, shared.stage1_predictions, STAGE_1_CLASS_NAMES),
        task2_malignant_subset=compute_classification_metrics(shared.stage2_targets[malignant],
            shared.stage2_predictions[malignant], STAGE_2_CLASS_NAMES), routing_counts=routing.routing_counts,
        delta_hard_minus_flat_macro_f1=result["hard"]["macro_f1"] - result["flat"]["macro_f1"],
        routing_loss_macro_f1=result["oracle"]["macro_f1"] - result["hard"]["macro_f1"])
    for first, second in (("flat", "hard"), ("hard", "oracle")):
        direction = f"{second}_minus_{first}"
        result[f"statistics_{direction}"] = paired_statistics(flat.targets,
            predictions[first], predictions[second], patients, direction)
    return result, rows


def _lock_digest(lock):
    return hashlib.sha256(internal._json_bytes(lock)).hexdigest()


def _identity(seed, lock):
    return {"seed": seed, "protocol": PROTOCOL, "preflight_identity_sha256": _lock_digest(lock),
            "checkpoint_provenance": internal._seed_provenance(lock, seed)}


def evaluate_one_seed(root, seed, lock, *, device):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("Undeclared seed")
    loader = build_hiba_loader(root)
    collections = {}
    for spec in (s for s in internal.CHECKPOINTS if s.seed == seed):
        record = next(r for r in lock["checkpoints"] if r["seed"] == seed and r["system"] == spec.system)
        model = internal.load_selected_model(root, spec, expected_sha256=record["sha256"])
        if spec.system == "flat":
            collections["flat"] = historical.collect_single_task_predictions(
                model, loader, class_names=internal.CLASS_NAMES, device=device)
        else:
            collections["shared"] = historical.collect_shared_isic_predictions(model, loader, device=device)
        model.cpu()
        del model
    result, rows = evaluate_collections(collections["flat"], collections["shared"], lock["cohort"])
    result.update(_identity(seed, lock))
    return result, rows


def aggregate_three_seeds(results):
    summary = internal.aggregate_three_seeds(results)
    summary["protocol"] = PROTOCOL
    return summary


ARTIFACTS = ("metrics_and_statistics.json", "paired_hiba_predictions.csv")


def _validate_completed_seed(directory, seed, lock):
    marker = read_json(directory / "seed_complete.json")
    for key, value in _identity(seed, lock).items():
        eq(marker.get(key), value, f"Completed {key}")
    eq(marker.get("status"), "PASS", "Completed status")
    eq(set(marker.get("artifacts", {})), set(ARTIFACTS), "Completed artifact set")
    for name, digest in marker["artifacts"].items():
        eq(sha256_file(directory / name), digest, f"Completed artifact drift: {name}")
    result = read_json(directory / ARTIFACTS[0])
    for key, value in _identity(seed, lock).items():
        eq(result.get(key), value, f"Result {key}")
    eq(result.get("sample_count"), EXPECTED_N, "Completed sample count")
    with (directory / ARTIFACTS[1]).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    eq([(r["sample_id"], r["patient_id"]) for r in rows],
       [(r["isic_id"], r["patient_id"]) for r in lock["cohort"]], "Completed cohort pairing")
    return result


def _publish_seed(output, seed, result, rows, lock):
    destination = output / f"seed{seed}"
    if destination.exists():
        raise FileExistsError(f"Refusing to replace seed artifacts: {destination}")
    staging = Path(tempfile.mkdtemp(prefix=f".seed{seed}.", dir=output))
    publish_json(staging / ARTIFACTS[0], result)
    with (staging / ARTIFACTS[1]).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    publish_json(staging / "seed_complete.json", {**_identity(seed, lock), "status": "PASS",
        "artifacts": {name: sha256_file(staging / name) for name in ARTIFACTS}})
    _validate_completed_seed(staging, seed, lock)
    staging.rename(destination)


def _final_marker(output, lock):
    return {"status": "PASS", "seeds": list(SEEDS), "protocol": PROTOCOL,
        "preflight_identity_sha256": _lock_digest(lock),
        "preflight_lock_sha256": sha256_file(output / LOCK_NAME),
        "summary_sha256": sha256_file(output / "three_seed_summary.json"),
        "seed_completion_sha256": {str(s): sha256_file(output / f"seed{s}" / "seed_complete.json") for s in SEEDS}}


def run_hiba_evaluation(project_root, *, device="cuda", persist_callback=None):
    root = Path(project_root).resolve()
    resolved_device = torch.device(device)
    if resolved_device.type not in ("cpu", "cuda"):
        raise ValueError("Evaluation device must be cpu or cuda")
    lock = verify_hiba_lock(root)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    output = root / OUTPUT
    results = {}
    # Validate every published seed before doing any new inference. A partial
    # visible directory fails closed; abandoned hidden staging is ignored.
    for seed in SEEDS:
        directory = output / f"seed{seed}"
        if directory.exists():
            results[seed] = _validate_completed_seed(directory, seed, lock)
    final_path = output / "evaluation_complete.json"
    if final_path.exists():
        eq(read_json(final_path), _final_marker(output, lock), "Final completion identity")
        summary = aggregate_three_seeds(results)
        eq(read_json(output / "three_seed_summary.json"), summary, "Final summary")
        return summary
    torch.backends.cudnn.benchmark = False
    for seed in SEEDS:
        if seed in results:
            continue
        result, rows = evaluate_one_seed(root, seed, lock, device=resolved_device)
        _publish_seed(output, seed, result, rows, lock)
        results[seed] = result
        if persist_callback is not None:
            persist_callback()
    summary = aggregate_three_seeds(results)
    publish_json(output / "three_seed_summary.json", summary)
    publish_json(final_path, _final_marker(output, lock))
    if persist_callback is not None:
        persist_callback()
    return summary
