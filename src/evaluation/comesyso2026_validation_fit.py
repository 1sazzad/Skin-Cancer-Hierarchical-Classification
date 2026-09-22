"""COM-03 validation-only orchestration. Importing this module executes no study.

Preflight and production are separate, explicit entry points. Only the new
CoMeSySo tree is writable. Callers must serialize writers (including preflight);
the Modal launcher's busy check is advisory, as in the historical evaluator.
"""
from __future__ import annotations

import csv
import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np
import sklearn
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.isic2019_dataset import ISIC2019HierarchicalDataset
from src.data.transforms import TransformConfig, build_eval_transform
from src.evaluation import comesyso2026_probability_fusion as fusion
from src.evaluation.paul2026_swin_evaluation import (
    CheckpointSpec, load_selected_model, sha256_file, validate_checkpoint_and_summary,
    _json_bytes, _publish_json, _read_json, _require_equal,
)

SEEDS = (42, 123, 2026)
EPOCHS = (10, 1, 26)
SPECS = tuple(CheckpointSpec("shared_hard", s, e) for s, e in zip(SEEDS, EPOCHS))
MANIFEST = Path("data/manifests/isic2019_train_val_test_split_seed42.csv")
OUTPUT = Path("results/extensions/comesyso2026_probability_fusion/validation_fit")
LOCK_NAME = "preflight_lock.json"
SUMMARY_NAME = "three_seed_fit_summary.json"
COMPLETE_NAME = "validation_fit_complete.json"
CSV_FIELDS = (
    "sample_id", "true_final_class", "task1_target", "task1_prediction",
    "p_task1_non_malignant", *fusion.FEATURE_ORDER,
)
SEED_FILES = {"validation_probabilities.csv", "fusion_model.json", "fit_metadata.json"}
SOURCE_PATHS = (
    "src/evaluation/comesyso2026_validation_fit.py",
    "src/evaluation/comesyso2026_probability_fusion.py",
    "src/evaluation/paul2026_swin_evaluation.py",
    "src/data/isic2019_dataset.py", "src/data/transforms.py",
    "src/models/transformer_extension.py", "src/models/shared_three_task.py",
    "scripts/comesyso2026/modal_comesyso2026_probability_fusion.py",
    "scripts/comesyso2026/launch_comesyso2026_validation_fit.py",
    "requirements.txt",
)


def _digest(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def require_validation_scope(*, dataset="isic2019", split="validation", stage="flat_four_class"):
    if (dataset, split, stage) != ("isic2019", "validation", "flat_four_class"):
        raise ValueError("COM-03 accepts only ISIC2019 validation / flat_four_class")


def output_directory(project_root):
    root = Path(project_root)
    frozen = Path("results/extensions/comesyso2026_probability_fusion/validation_fit")
    if ".." in root.parts or OUTPUT != frozen:
        raise ValueError("CoMeSySo output must use the exact frozen relative path without traversal")
    # Validate the requested namespace, not a mounted volume's backing path.
    root = root.absolute()
    path = root / OUTPUT
    if path.relative_to(root) != frozen:
        raise ValueError("CoMeSySo output must remain under the supplied project root")
    return path


def _visible(directory):
    if not directory.exists():
        return set()
    if not directory.is_dir():
        raise ValueError(f"Expected artifact directory: {directory}")
    entries = list(directory.iterdir())
    for path in entries:
        if path.resolve() != directory.resolve() / path.name:
            raise ValueError(f"Redirected artifact: {path}")
    return {p.name for p in entries if not p.name.startswith(".")}


def validate_protocol(value):
    """Check the frozen fitting contract, without consulting evaluation results."""
    expected = {
        ("protocol", "name"): "comesyso2026_probability_fusion",
        ("protocol", "phase"): "COM-01B",
        ("protocol", "status"): "frozen",
        ("models", "architecture"): "swin_t",
        ("models", "seeds"): list(SEEDS),
        ("models", "parameters_frozen"): True,
        ("models", "checkpoint_filename"): "best_checkpoint.pt",
        ("models", "checkpoint_root"): "results/extensions/paul2026_swin/runs",
        ("datasets", "isic", "manifest"): MANIFEST.as_posix(),
        ("datasets", "isic", "same_partition_for_all_model_seeds"): True,
        ("systems", "fusion", "fit_independently_per_seed"): True,
        ("systems", "fusion", "fit_partition"): fusion.FIT_PARTITION,
        ("systems", "fusion", "target"): "true_final_class",
        ("systems", "fusion", "feature_order"): list(fusion.FEATURE_ORDER),
        ("systems", "fusion", "classifier"): fusion.CLASSIFIER_SPEC,
        ("classes", "final"): dict(enumerate(fusion.CLASS_NAMES)),
        ("classes", "task1"): {0: "non_malignant", 1: "malignant"},
        ("classes", "task2"): {0: "melanoma", 1: "bcc", 2: "scc"},
        ("classes", "fixed_class_labels"): [0, 1, 2, 3],
        ("preprocessing", "color"): "RGB",
        ("preprocessing", "resize_short_side"): 256,
        ("preprocessing", "center_crop"): [224, 224],
        ("preprocessing", "normalization"): "imagenet",
        ("preprocessing", "interpolation"): "bilinear_antialiased",
        ("preprocessing", "augmentation"): False,
        ("probability_contract", "calibration"): "none",
        ("probability_contract", "feature_scaling"): "none",
        ("probability_contract", "task2_required_for"):
            "every_sample_including_task1_predicted_non_malignant",
    }
    for key in ("hyperparameter_search", "threshold_search", "feature_selection",
                "alternative_meta_classifier_selection"):
        expected[("systems", "fusion", key)] = False
    for key in ("retraining_allowed", "fine_tuning_allowed", "checkpoint_reselection_allowed",
                "architecture_modification_allowed", "seed_replacement_allowed", "seed_dropping_allowed"):
        expected[("models", key)] = False
    for keys, wanted in expected.items():
        actual = value
        for key in keys:
            if not isinstance(actual, dict) or key not in actual:
                raise ValueError(f"Missing protocol field: {'.'.join(keys)}")
            actual = actual[key]
        _require_equal(actual, wanted, f"Protocol {'.'.join(keys)}")
    selections = value["models"].get("selections", [])
    if any(type(seed) is not int for seed in value["models"]["seeds"]):
        raise ValueError("Frozen seeds must be integers")
    _require_equal([r.get("seed") for r in selections], list(SEEDS), "Selection seeds")
    for record, spec in zip(selections, SPECS):
        _require_equal(record.get("shared_hard_run"), spec.run_name, "Selected run")
        _require_equal(record.get("shared_hard_selected_epoch"), spec.epoch, "Selected epoch")


def _inspect_checkpoint(root, spec):
    """Use historical strict state loading, plus mandatory Shared identity fields."""
    path = spec.path(root)
    summary_path = path.parent / "run_summary.json"
    config_path = Path(f"configs/extensions/paul2026_swin/shared_hard_seed{spec.seed}.yaml")
    before = sha256_file(path)  # Missing best checkpoint fails; no substitution.
    summary_hash = sha256_file(summary_path)
    config_hash = sha256_file(root / config_path)
    config = yaml.safe_load((root / config_path).read_text(encoding="utf-8"))
    summary = _read_json(summary_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_checkpoint_and_summary(payload, summary, spec)
    _require_equal(payload.get("seed"), spec.seed, "Required checkpoint seed")
    metadata = payload.get("config_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Missing Shared-Hard checkpoint config_metadata")
    _require_equal(metadata.get("experiment"), config.get("experiment"), "Checkpoint config identity")
    _require_equal(config["experiment"]["seed"], spec.seed, "Config seed")
    _require_equal(config["experiment"]["run_name"], spec.run_name, "Config run")
    _require_equal(config["model"]["architecture"], "swin_t", "Config architecture")
    _require_equal(config["data"]["isic2019_manifest"], MANIFEST.as_posix(), "Config manifest")
    _require_equal(payload.get("model_metadata"), config["model"], "Checkpoint model config")
    _require_equal(payload.get("task_loss_configuration"), config["task_losses"], "Checkpoint losses")
    _require_equal(payload.get("task_mask_policy"), config["data"]["task_mask_policy"], "Checkpoint masks")
    _require_equal(payload.get("class_mappings"), config["data"]["class_mappings"], "Checkpoint classes")
    # Historical payloads may store a subset of the config. Check every included section.
    for key, section in metadata.items():
        if key in config:
            _require_equal(section, config[key], f"Checkpoint config {key}")
    del payload
    model = load_selected_model(root, spec, expected_sha256=before)
    del model
    for p, digest in ((path, before), (summary_path, summary_hash), (root / config_path, config_hash)):
        _require_equal(sha256_file(p), digest, "Input changed during inspection")
    return {
        "seed": spec.seed, "historical_run_name": spec.run_name,
        "checkpoint_selected_epoch": spec.epoch,
        "checkpoint_path": path.relative_to(root).as_posix(), "checkpoint_sha256": before,
        "run_summary_sha256": summary_hash, "config_path": config_path.as_posix(),
        "config_sha256": config_hash,
    }


def inspect_inputs(project_root):
    """CPU inspection only: no dataset, image loader, collector, or fit call."""
    root = Path(project_root).resolve()
    protocol_path = root / fusion.PROTOCOL_PATH
    protocol_hash = sha256_file(protocol_path)
    validate_protocol(yaml.safe_load(protocol_path.read_text(encoding="utf-8")))
    manifest_hash = sha256_file(root / MANIFEST)  # Byte identity, no held-out label inspection.
    sources = {p: sha256_file(root / p) for p in SOURCE_PATHS}
    checkpoints = [_inspect_checkpoint(root, spec) for spec in SPECS]
    # Inspect only an existing checkpoint lock, never historical prediction/metric artifacts.
    old_lock = root / "results/extensions/paul2026_swin/evaluation/preflight_checkpoint_lock.json"
    historical_lock_hash = None
    if old_lock.exists():
        historical_lock_hash = sha256_file(old_lock)
        historical = _read_json(old_lock)
        for record in checkpoints:
            matches = [r for r in historical.get("checkpoints", [])
                       if r.get("system") == "shared_hard" and r.get("seed") == record["seed"]]
            if len(matches) != 1:
                raise ValueError("Historical checkpoint lock missing unique Shared-Hard selection")
            for key, current in (("sha256", "checkpoint_sha256"),
                                 ("selected_epoch", "checkpoint_selected_epoch"),
                                 ("run_name", "historical_run_name"),
                                 ("run_summary_sha256", "run_summary_sha256")):
                _require_equal(matches[0].get(key), record[current], "Historical checkpoint lock")
        _require_equal(sha256_file(old_lock), historical_lock_hash, "Historical lock changed")
    _require_equal(sha256_file(protocol_path), protocol_hash, "Protocol changed during preflight")
    _require_equal(sha256_file(root / MANIFEST), manifest_hash, "Manifest changed during preflight")
    _require_equal({p: sha256_file(root / p) for p in SOURCE_PATHS}, sources, "Source changed")
    return {
        "schema_version": 1, "phase": "COM-03", "status": "PASS",
        "created_before_validation_inference": True, "seeds": list(SEEDS),
        "dataset": "isic2019", "split": "validation", "stage": "flat_four_class",
        "protocol_path": fusion.PROTOCOL_PATH, "protocol_sha256": protocol_hash,
        "validation_manifest_path": MANIFEST.as_posix(), "validation_manifest_sha256": manifest_hash,
        "source_sha256": sources, "checkpoints": checkpoints,
        "historical_checkpoint_lock_sha256": historical_lock_hash,
        "sklearn_version": sklearn.__version__, "torch_version": str(torch.__version__),
        "feature_order": list(fusion.FEATURE_ORDER), "classifier_specification": fusion.CLASSIFIER_SPEC,
    }


def run_validation_fit_preflight(project_root, *, persist_callback=None):
    output = output_directory(project_root)
    visible = _visible(output)
    if LOCK_NAME not in visible and visible:
        raise ValueError("Cannot create a before-inference lock after visible artifacts exist")
    lock = inspect_inputs(project_root)
    _publish_json(output / LOCK_NAME, lock)
    if persist_callback is not None:
        persist_callback()
    return lock


def verify_preflight_lock(project_root):
    path = output_directory(project_root) / LOCK_NAME
    if not path.is_file():
        raise FileNotFoundError("COM-03 preflight lock required before production")
    if path.resolve() != path.parent.resolve() / path.name:
        raise ValueError("Redirected preflight lock")
    lock = _read_json(path)
    _require_equal(inspect_inputs(project_root), lock, "COM-03 preflight identity drift")
    return lock


def verify_input_bytes(project_root, lock):
    """Recheck locked bytes before publication, without loading any image/model."""
    root = Path(project_root).resolve()
    files = {lock["protocol_path"]: lock["protocol_sha256"],
             lock["validation_manifest_path"]: lock["validation_manifest_sha256"],
             **lock["source_sha256"]}
    for record in lock["checkpoints"]:
        files[record["checkpoint_path"]] = record["checkpoint_sha256"]
        files[record["config_path"]] = record["config_sha256"]
        summary_path = Path(record["checkpoint_path"]).parent / "run_summary.json"
        files[summary_path.as_posix()] = record["run_summary_sha256"]
    for path, digest in files.items():
        _require_equal(sha256_file(root / path), digest, f"Locked input drift: {path}")
    _require_equal(_read_json(output_directory(root) / LOCK_NAME), lock, "Preflight lock changed")


def build_validation_loader(project_root, *, dataset="isic2019", split="validation", stage="flat_four_class"):
    require_validation_scope(dataset=dataset, split=split, stage=stage)
    root = Path(project_root).resolve()
    selected = ISIC2019HierarchicalDataset(
        root / MANIFEST, root, split="validation", stage="flat_four_class",
        transform=build_eval_transform(TransformConfig(image_size=224, eval_resize_size=256)),
    )
    return DataLoader(selected, batch_size=64, num_workers=4, shuffle=False,
                      drop_last=False, generator=torch.Generator().manual_seed(42))


def cohort_identity(sample_ids, targets):
    targets = fusion._labels(targets)
    ids = fusion._ids(sample_ids, len(targets))
    if set(targets.tolist()) != set(fusion.LABELS):
        raise ValueError("Validation must contain all four final classes")
    return {"sample_ids": list(ids), "targets": targets.tolist()}


def validate_collection(shared, *, expected_cohort=None):
    cohort = cohort_identity(shared.sample_ids, shared.flat_targets)
    if expected_cohort is not None:
        _require_equal(cohort, expected_cohort, "Validation count/order/target mismatch")
    p1, p2 = fusion._heads(shared.stage1_probabilities, shared.stage2_probabilities)
    if len(p1) != len(cohort["sample_ids"]):
        raise ValueError("Validation probabilities and targets have different counts")
    _require_equal(np.asarray(shared.stage1_targets).tolist(),
                   (np.asarray(cohort["targets"]) != 0).astype(int).tolist(), "Task-1 targets")
    _require_equal(np.asarray(shared.stage1_predictions).tolist(), p1.argmax(1).tolist(), "Task-1 prediction")
    features = fusion.build_fusion_features(p1, p2, sample_ids=shared.sample_ids,
                                            task2_sample_ids=shared.sample_ids)
    return cohort, features


def _cohort_stats(cohort):
    return {"validation_sample_count": len(cohort["targets"]),
            "validation_class_counts": {str(c): cohort["targets"].count(c) for c in fusion.LABELS},
            "validation_cohort_sha256": _digest(cohort)}


def _provenance(lock, seed):
    fusion._seed(seed)
    record = next(r for r in lock["checkpoints"] if r["seed"] == seed)
    return {
        "seed": seed, "fitting_partition": fusion.FIT_PARTITION,
        "dataset": "isic2019", "split": "validation", "stage": "flat_four_class",
        "preflight_lock_sha256": _digest(lock),
        **{key: lock[key] for key in ("protocol_path", "protocol_sha256",
                                     "validation_manifest_path", "validation_manifest_sha256")},
        **record,
    }


def _metadata(model, cohort, lock, seed):
    return {**_provenance(lock, seed), **_cohort_stats(cohort),
            "feature_order": list(fusion.FEATURE_ORDER), "classes": list(fusion.LABELS),
            "classifier_specification": fusion.CLASSIFIER_SPEC,
            "coefficient_shape": list(model.classifier.coef_.shape),
            "intercept_shape": list(model.classifier.intercept_.shape),
            "converged": True, "n_iter": model.classifier.n_iter_.tolist(),
            "sklearn_version": model.sklearn_version}


def _write_probabilities(path, shared):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(CSV_FIELDS)
        for i, sample_id in enumerate(shared.sample_ids):
            writer.writerow([sample_id, int(shared.flat_targets[i]), int(shared.stage1_targets[i]),
                             int(shared.stage1_predictions[i]), *shared.stage1_probabilities[i],
                             *shared.stage2_probabilities[i]])
        handle.flush()
        os.fsync(handle.fileno())


def _read_probabilities(path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_equal(reader.fieldnames, list(CSV_FIELDS), "Validation CSV fields")
        rows = list(reader)
    if any(set(row) != set(CSV_FIELDS) or any(v is None for v in row.values()) for row in rows):
        raise ValueError("Malformed validation CSV row")
    truth = np.asarray([int(r["true_final_class"]) for r in rows], dtype=np.int64)
    p1 = np.asarray([[float(r[k]) for k in CSV_FIELDS[4:6]] for r in rows])
    p2 = np.asarray([[float(r[k]) for k in CSV_FIELDS[6:]] for r in rows])
    return fusion.SharedPredictions(
        tuple(r["sample_id"] for r in rows), truth,
        np.asarray([int(r["task1_target"]) for r in rows]),
        np.asarray([int(r["task1_prediction"]) for r in rows]), p1,
        np.where(truth == 0, -1, truth - 1), np.zeros(len(rows), dtype=int), p2,
        (None,) * len(rows), 0.0,
    )


def validate_completed_seed(directory, seed, lock, *, expected_cohort=None):
    directory = Path(directory)
    _require_equal(_visible(directory), SEED_FILES | {"seed_complete.json"}, "Completed seed file set")
    marker = _read_json(directory / "seed_complete.json")
    expected_marker = {"status": "PASS", **_provenance(lock, seed),
                       "artifacts": {p: sha256_file(directory / p) for p in sorted(SEED_FILES)}}
    _require_equal(marker, expected_marker, "Completed seed provenance/hash drift")
    shared = _read_probabilities(directory / "validation_probabilities.csv")
    cohort, _ = validate_collection(shared, expected_cohort=expected_cohort)
    # The COM-02 file remains byte/schema compatible. Provenance lives in the
    # mandatory companion metadata, bound by the seed marker's hashes.
    model = fusion.load_fusion(directory / "fusion_model.json", expected_seed=seed)
    _require_equal(model.fitting_partition, fusion.FIT_PARTITION, "Model fitting partition")
    _require_equal(model.fitting_sample_count, len(cohort["targets"]), "Model sample count")
    _require_equal(model.sklearn_version, lock["sklearn_version"], "Model sklearn identity")
    metadata = _metadata(model, cohort, lock, seed)
    _require_equal(_read_json(directory / "fit_metadata.json"), metadata, "Fitting metadata drift")
    return model, metadata, cohort


def load_frozen_fusion(project_root, *, seed):
    """Production reload verifies the lock and complete artifact bundle, never refits."""
    fusion._seed(seed)
    lock = verify_preflight_lock(project_root)
    model, _, _ = validate_completed_seed(output_directory(project_root) / f"seed{seed}", seed, lock)
    return model


def _publish_seed(output, seed, shared, model, lock, cohort):
    destination = output / f"seed{seed}"
    if destination.exists():
        raise FileExistsError(f"Refusing to replace seed artifacts: {destination}")
    staging = Path(tempfile.mkdtemp(prefix=f".seed{seed}.", dir=output))
    _write_probabilities(staging / "validation_probabilities.csv", shared)
    fusion.save_fusion(model, staging / "fusion_model.json")
    _publish_json(staging / "fit_metadata.json", _metadata(model, cohort, lock, seed))
    _publish_json(staging / "seed_complete.json", {
        "status": "PASS", **_provenance(lock, seed),
        "artifacts": {p: sha256_file(staging / p) for p in sorted(SEED_FILES)},
    })
    validate_completed_seed(staging, seed, lock, expected_cohort=cohort)
    if destination.exists():
        raise FileExistsError(f"Seed appeared during publication: {destination}")
    staging.rename(destination)


def three_seed_summary(results):
    if set(results) != set(SEEDS):
        raise ValueError("Exactly seeds 42, 123, 2026 required")
    baseline = results[42]
    for seed in SEEDS:
        _require_equal(results[seed]["seed"], seed, "Summary seed identity")
        for key in ("validation_sample_count", "validation_class_counts", "validation_cohort_sha256",
                    "feature_order", "classifier_specification", "preflight_lock_sha256"):
            _require_equal(results[seed][key], baseline[key], "Cross-seed fitting identity")
    return {"phase": "COM-03", "seeds_completed": list(SEEDS),
            "fits": {str(seed): results[seed] for seed in SEEDS}}


def _final_marker(output, lock):
    return {"status": "PASS", "seeds": list(SEEDS), "preflight_lock_sha256": _digest(lock),
            "summary_sha256": sha256_file(output / SUMMARY_NAME),
            "seed_completion_sha256": {str(s): sha256_file(output / f"seed{s}" / "seed_complete.json")
                                       for s in SEEDS}}


def run_validation_fit_production(project_root, *, device="cuda", persist_callback=None):
    output = output_directory(project_root)
    root = Path(project_root).absolute()
    lock = verify_preflight_lock(root)  # Never create a lock here.
    allowed = {LOCK_NAME, SUMMARY_NAME, COMPLETE_NAME} | {f"seed{s}" for s in SEEDS}
    visible = _visible(output)
    if visible - allowed:
        raise ValueError(f"Unexpected visible artifacts: {sorted(visible - allowed)}")
    results, cohort = {}, None
    # Validate every durable seed before any missing seed is inferred or fit.
    for seed in SEEDS:
        directory = output / f"seed{seed}"
        if directory.exists():
            _, metadata, current = validate_completed_seed(directory, seed, lock, expected_cohort=cohort)
            results[seed], cohort = metadata, current
    if SUMMARY_NAME in visible or COMPLETE_NAME in visible:
        if len(results) != 3 or SUMMARY_NAME not in visible:
            raise ValueError("Partial or incompatible final publication")
        _require_equal(_read_json(output / SUMMARY_NAME), three_seed_summary(results), "Summary drift")
        if COMPLETE_NAME in visible:
            _require_equal(_read_json(output / COMPLETE_NAME), _final_marker(output, lock), "Completion drift")
            return three_seed_summary(results)
        # A visible summary without the completion marker is partial publication.
        raise ValueError("Partial final publication: completion marker missing")
    resolved_device = torch.device(device)
    if resolved_device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU/CUDA inference supported")
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.backends.cudnn.benchmark = False
    for spec in SPECS:
        if spec.seed in results:
            continue
        loader = build_validation_loader(root)
        expected = cohort_identity(loader.dataset.selected_frame["image_id"].tolist(),
                                   np.asarray(loader.dataset.targets, dtype=np.int64))
        if cohort is not None:
            _require_equal(expected, cohort, "Cross-seed validation cohort")
        record = next(r for r in lock["checkpoints"] if r["seed"] == spec.seed)
        model = load_selected_model(root, spec, expected_sha256=record["checkpoint_sha256"])
        shared = fusion.collect_shared_predictions(model, loader, device=resolved_device)
        model.cpu()
        del model, loader
        cohort, features = validate_collection(shared, expected_cohort=expected)
        fitted = fusion.fit_fusion(features, shared.flat_targets, seed=spec.seed,
                                   fitting_partition=fusion.FIT_PARTITION,
                                   sample_ids=shared.sample_ids, target_sample_ids=shared.sample_ids)
        verify_input_bytes(root, lock)
        _publish_seed(output, spec.seed, shared, fitted, lock, cohort)
        if persist_callback is not None:
            persist_callback()
        results[spec.seed] = _metadata(fitted, cohort, lock, spec.seed)
    summary = three_seed_summary(results)
    _publish_json(output / SUMMARY_NAME, summary)
    _publish_json(output / COMPLETE_NAME, _final_marker(output, lock))
    if persist_callback is not None:
        persist_callback()
    return summary
