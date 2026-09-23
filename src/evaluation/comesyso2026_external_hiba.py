"""COM-05 zero-shot HIBA: CPU preflight, inference-only T4, CPU statistics.

No fitting or training entry point. COM-03/COM-04 and historical evidence are
read-only inputs. Serialize launchers; cross-function busy checks are advisory.
"""
from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

import numpy as np
import yaml

from src.evaluation import comesyso2026_probability_fusion as fusion
from src.evaluation import comesyso2026_validation_fit as fit
from src.evaluation import comesyso2026_internal_isic as com04
from src.evaluation import paul2026_swin_hiba_evaluation as hiba
from src.evaluation.paul2026_swin_evaluation import (
    _publish_json, _read_json, _require_equal as eq, sha256_file,
)

SEEDS = (42, 123, 2026)
COUNTS = (696, 196, 229, 111)
PATIENT_COUNT = 568
MANIFEST = Path("data/external/hiba/manifests/hiba_external_dermoscopic_4class_final.csv")
MANIFEST_CANONICAL_SHA256 = "a2f30f14a249d8acb2bd9f03884e5e41c773ddbee19910d5b739407521e3706a"
OUTPUT = Path("results/extensions/comesyso2026_probability_fusion/external_hiba")
LOCK_NAME = "preflight_lock.json"
INFERENCE_COMPLETE = "inference_complete.json"
SUMMARY_NAME = "three_seed_summary.json"
COMPLETE_NAME = "external_hiba_complete.json"
INFERENCE_FILES = {"paired_predictions.csv", "inference_metadata.json"}
ANALYSIS_FILES = {"metrics_and_statistics.json", "rescue_analysis.json"}
SOURCE_PATHS = tuple(dict.fromkeys((*com04.SOURCE_PATHS, *hiba.SOURCE_FILES,
    "src/evaluation/comesyso2026_external_hiba.py",
    "scripts/comesyso2026/modal_comesyso2026_external_hiba.py",
    "scripts/comesyso2026/launch_comesyso2026_external_hiba.py")))
SYSTEMS = com04.SYSTEMS
METRICS = com04.METRICS


def prepare_hiba_volume_paths(project_root):
    """Expose the historical HIBA path from one Modal data-volume mount.

    The verified live paul2026-swin-data layout stores HIBA JPEGs beneath
    emb/images/isic/. Only the container-local alias is created; no volume
    data or manifest is changed.
    Call only in the Modal preflight/inference wrappers, not CPU analysis.
    """
    root = Path(project_root).absolute()
    source = root / "data/raw/emb/images/isic"
    alias = root / "data/external/hiba/extracted"
    if not source.is_dir():
        raise FileNotFoundError(f"Required live HIBA image directory is missing: {source}")
    if alias.is_symlink():
        if alias.resolve() != source.resolve():
            raise ValueError(f"Incompatible HIBA extracted alias: {alias}")
        return
    if alias.exists():
        raise ValueError(f"Refusing to replace existing HIBA extracted path: {alias}")
    alias.parent.mkdir(parents=True, exist_ok=True)
    alias.symlink_to(source, target_is_directory=True)


def output_directory(project_root):
    root = Path(project_root)
    frozen = Path("results/extensions/comesyso2026_probability_fusion/external_hiba")
    if ".." in root.parts or OUTPUT != frozen:
        raise ValueError("COM-05 output must use the frozen relative path without traversal")
    path = root.absolute() / OUTPUT
    if path.resolve() != path.parent.resolve() / path.name:
        raise ValueError("Redirected COM-05 output")
    return path


def cohort_identity(sample_ids, patient_ids, targets):
    truth = fusion._labels(targets)
    ids = fusion._ids(sample_ids, len(truth))
    patients = fusion._ids(patient_ids, len(truth), unique=False)
    eq(np.bincount(truth, minlength=4).tolist(), list(COUNTS), "HIBA class counts")
    eq(len(truth), sum(COUNTS), "HIBA sample count")
    eq(len(set(patients)), PATIENT_COUNT, "HIBA patient count")
    return {"sample_ids": list(ids), "patient_ids": list(patients), "targets": truth.tolist()}


def _cohort(rows):
    return cohort_identity([r["isic_id"] for r in rows], [r["patient_id"] for r in rows],
                           np.asarray([int(r["target_index"]) for r in rows], dtype=np.int64))


def _safe_input(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Input path must be project-relative without traversal")
    return Path(root) / path


def _hashes(root, paths):
    return {str(p): sha256_file(_safe_input(root, p)) for p in paths}


def validate_protocol(protocol):
    fit.validate_protocol(protocol)
    spec = protocol["datasets"]["hiba"]
    for key, value in (("manifest", MANIFEST.as_posix()),
                       ("canonical_crlf_manifest_sha256", MANIFEST_CANONICAL_SHA256),
                       ("expected_images", 1232), ("expected_unique_patients", 568),
                       ("patient_field", "patient_id"), ("source_sample_id_field", "isic_id"),
                       ("zero_shot", True), ("unchanged_fusion_transfer", True)):
        eq(spec.get(key), value, f"HIBA protocol {key}")
    eq(spec["expected_class_counts"], dict(zip(fusion.CLASS_NAMES, (696, 196, 229, 111))), "Class counts")
    bootstrap = protocol["statistics"]["hiba"]["bootstrap"]
    for key, value in (("unit", "patient_cluster"), ("replicates", 5000), ("seed", 42),
                       ("paired", True), ("replacement", True), ("confidence_level", .95),
                       ("interval", "percentile"), ("quantile_method", "linear"),
                       ("patient_order", "sorted_unique_patient_ids"),
                       ("patients_per_replicate", 568),
                       ("preserve_all_images_for_each_sampled_patient", True),
                       ("preserve_repeated_cluster_multiplicity", True)):
        eq(bootstrap.get(key), value, f"HIBA bootstrap {key}")
    eq(protocol["statistics"]["hiba"]["image_level_mcnemar_primary"], False, "HIBA McNemar policy")
    comparisons = [protocol["endpoints"]["primary_comparison"], *protocol["endpoints"]["secondary_comparisons"]]
    eq({r["id"]: (r["minuend"], r["subtrahend"]) for r in comparisons}, fusion.CONTRASTS, "Contrasts")


def validate_checkpoint_records(records, root):
    expected = [(s.system, s.seed, s.epoch, s.path(root).relative_to(root).as_posix())
                for s in (*com04.FLAT_SPECS, *com04.SHARED_SPECS)]
    eq([(r["system"], r["seed"], r["epoch"], r["path"]) for r in records], expected,
       "Frozen checkpoint identities")
    for record in records:
        digest = record["sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Malformed checkpoint SHA256")


def inspect_inputs(project_root):
    """Stage A only: CPU checkpoint validation and byte hashing, no dataset/forward."""
    root = Path(project_root).absolute()
    initial_hashes = _hashes(root, (*SOURCE_PATHS, hiba.CONFIG.as_posix(), MANIFEST.as_posix(),
                                    fusion.PROTOCOL_PATH))
    previous = com04.verify_preflight_lock(root)
    previous_output = com04.output_directory(root)
    if not (previous_output / com04.COMPLETE_NAME).is_file():
        raise ValueError("Completed COM-04 is required")
    com04.validate_outputs(previous_output, previous)
    _, models = com04._completed_fits(root)
    protocol = yaml.safe_load((root / fusion.PROTOCOL_PATH).read_text(encoding="utf-8"))
    validate_protocol(protocol)
    cfg = hiba._config(root)
    eq(cfg["hiba"]["manifest_path"], MANIFEST.as_posix(), "Historical HIBA manifest")
    # Historical _cohort validates the canonical manifest, patient mapping,
    # labels, all image hashes and path containment; it never constructs a dataset.
    rows = hiba._cohort(root, cfg)
    cohort = _cohort(rows)
    checkpoints = previous["checkpoints"]
    validate_checkpoint_records(checkpoints, root)
    for seed in SEEDS:
        eq(models[seed].seed, seed, "Fusion seed")
        eq(models[seed].fitting_partition, fusion.FIT_PARTITION, "Validation-only fusion")
    # Stage C needs source/provenance and persisted predictions, never neural
    # weights or raw images. All upstream COM-03/04 artifacts remain hash-bound.
    common = {p: digest for p, digest in previous["input_sha256"].items()
              if not p.startswith("results/extensions/paul2026_swin/")}
    upstream_paths = [previous_output / n for n in (com04.LOCK_NAME, com04.SUMMARY_NAME, com04.COMPLETE_NAME)]
    upstream_paths += [previous_output / f"seed{s}" / n for s in SEEDS
                       for n in sorted(com04.SEED_FILES | {"seed_complete.json"})]
    common.update(_hashes(root, [p.relative_to(root).as_posix() for p in upstream_paths]))
    common.update(initial_hashes)
    inference_inputs = {p: digest for p, digest in previous["input_sha256"].items()
                        if p.startswith("results/extensions/paul2026_swin/")}
    for row in rows:
        path = _safe_input(root, row["image_path"])
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("HIBA image path escapes project root")
        inference_inputs[row["image_path"]] = row["image_sha256"].lower()
    fusion_paths = {str(s): (fit.OUTPUT / f"seed{s}/fusion_model.json").as_posix() for s in SEEDS}
    lock = {"schema_version": 1, "phase": "COM-05", "status": "PASS", "zero_shot": True,
            "inference_performed": False, "dataset_constructed": False,
            "protocol_identity": fusion.PROTOCOL_ID, "seeds": list(SEEDS),
            "cohort": cohort, "cohort_sha256": fit._digest(cohort), "checkpoints": checkpoints,
            "feature_order": list(fusion.FEATURE_ORDER), "class_order": list(fusion.CLASS_NAMES),
            "manifest_canonical_sha256": MANIFEST_CANONICAL_SHA256,
            "common_input_sha256": common, "inference_input_sha256": inference_inputs,
            "fusion_paths": fusion_paths, "com04_lock_sha256": fit._digest(previous)}
    verify_input_bytes(root, lock, inference=True, require_lock=False)
    com04.verify_input_bytes(root, previous)
    com04.validate_outputs(previous_output, previous)
    verify_input_bytes(root, lock, inference=True, require_lock=False)
    return lock


def verify_input_bytes(root, lock, *, inference=False, require_lock=True):
    files = dict(lock["common_input_sha256"])
    if inference:
        files.update(lock["inference_input_sha256"])
    for path, digest in files.items():
        eq(sha256_file(_safe_input(root, path)), digest, f"Locked input drift: {path}")
    if require_lock:
        eq(_read_json(output_directory(root) / LOCK_NAME), lock, "Preflight lock drift")


def verify_preflight_lock(root, *, inference=False):
    output = output_directory(root)
    fit._visible(output)
    if not (output / LOCK_NAME).is_file():
        raise FileNotFoundError("CPU COM-05 preflight required before infer/analyze")
    lock = _read_json(output / LOCK_NAME)
    for key, value in (("schema_version", 1), ("phase", "COM-05"), ("status", "PASS"), ("zero_shot", True),
                       ("protocol_identity", fusion.PROTOCOL_ID), ("inference_performed", False),
                       ("dataset_constructed", False), ("manifest_canonical_sha256", MANIFEST_CANONICAL_SHA256),
                       ("seeds", list(SEEDS)), ("feature_order", list(fusion.FEATURE_ORDER)),
                       ("class_order", list(fusion.CLASS_NAMES))):
        eq(lock.get(key), value, f"COM-05 lock {key}")
    validate_checkpoint_records(lock["checkpoints"], Path(root))
    for record in lock["checkpoints"]:
        eq(lock["inference_input_sha256"].get(record["path"]), record["sha256"], "Checkpoint hash anchor")
    cohort = lock["cohort"]
    eq(cohort_identity(cohort["sample_ids"], cohort["patient_ids"], np.asarray(cohort["targets"], dtype=np.int64)),
       cohort, "Locked HIBA cohort")
    eq(fit._digest(cohort), lock["cohort_sha256"], "Cohort hash")
    verify_input_bytes(root, lock, inference=inference)
    return lock


def load_frozen_fusion(root, seed, lock):
    fusion._seed(seed)
    expected_path = (fit.OUTPUT / f"seed{seed}/fusion_model.json").as_posix()
    eq(lock["fusion_paths"][str(seed)], expected_path, "Frozen fusion path")
    path = _safe_input(root, expected_path)
    eq(sha256_file(path), lock["common_input_sha256"][expected_path], "Frozen fusion hash")
    model = fusion.load_fusion(path, expected_seed=seed)
    eq(model.fitting_partition, fusion.FIT_PARTITION, "No HIBA fitting")
    return model


def prediction_rows(shared, flat_probabilities, fusion_probabilities, *, seed, expected_cohort):
    """Cheap deterministic rules and row construction only; no statistical loops."""
    fusion._seed(seed)
    truth = fusion._labels(shared.flat_targets)
    n = len(truth)
    eq(cohort_identity(shared.sample_ids, shared.patient_ids, truth), expected_cohort, "Frozen HIBA pairing")
    p1, p2 = fusion._heads(shared.stage1_probabilities, shared.stage2_probabilities)
    if len(p1) != n:
        raise ValueError("Head/target length mismatch")
    pf = fusion._probabilities(flat_probabilities, 4, count=n)
    pm = fusion._probabilities(fusion_probabilities, 4, count=n)
    t1 = (truth != 0).astype(int)
    eq(np.asarray(shared.stage1_targets).tolist(), t1.tolist(), "Task1 targets")
    eq(np.asarray(shared.stage1_predictions).tolist(), p1.argmax(1).tolist(), "Task1 predictions")
    eq(np.asarray(shared.stage2_predictions).tolist(), p2.argmax(1).tolist(), "Unmasked Task2 predictions")
    eq(np.asarray(shared.stage2_targets).tolist(), np.where(truth == 0, -1, truth - 1).tolist(), "Task2 targets")
    predictions = {"flat": pf.argmax(1), "hard": fusion.hard_routing(p1, p2),
                   "path_soft": fusion.path_soft(p1, p2)[1], "fusion": pm.argmax(1),
                   "oracle": fusion.oracle_routing(t1, p2)}
    rows = []
    for i, sample_id in enumerate(shared.sample_ids):
        row = {"dataset": "hiba", "seed": seed, "sample_id": sample_id,
               "patient_id": shared.patient_ids[i], "true_final_class": int(truth[i]),
               "task1_target": int(t1[i]), "task1_prediction": int(p1[i].argmax()),
               "p_task1_non_malignant": float(p1[i, 0]), "p_task1_malignant": float(p1[i, 1])}
        row.update({f"p_task2_{c}": float(p2[i, j]) for j, c in enumerate(fusion.CLASS_NAMES[1:])})
        for name, probabilities in (("flat", pf), ("fusion", pm)):
            row.update({f"p_{name}_{c}": float(probabilities[i, j]) for j, c in enumerate(fusion.CLASS_NAMES)})
        for name, pred in predictions.items():
            row[f"{name}_prediction"] = int(pred[i])
            row[f"{name}_correct"] = bool(pred[i] == truth[i])
        row["rescue_category"] = com04._partition(row["hard_correct"], row["fusion_correct"],
                                                  row["hard_prediction"] == row["fusion_prediction"])
        row["malignant_routing_failure"] = bool(t1[i] == 1 and p1[i].argmax() == 0)
        row["nm_routing_failure"] = bool(t1[i] == 0 and p1[i].argmax() == 1)
        rows.append(row)
    return rows


def _read_predictions(path, seed, lock):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    if not rows or any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError("Malformed paired prediction CSV")
    truth = np.asarray([int(r["true_final_class"]) for r in rows], dtype=np.int64)
    p1 = np.asarray([[float(r["p_task1_non_malignant"]), float(r["p_task1_malignant"])] for r in rows])
    p2 = np.asarray([[float(r[f"p_task2_{c}"]) for c in fusion.CLASS_NAMES[1:]] for r in rows])
    shared = fusion.SharedPredictions(tuple(r["sample_id"] for r in rows), truth,
        np.asarray([int(r["task1_target"]) for r in rows]),
        np.asarray([int(r["task1_prediction"]) for r in rows]), p1,
        np.where(truth == 0, -1, truth - 1), p2.argmax(1), p2,
        tuple(r["patient_id"] for r in rows), 0.)
    probabilities = {name: np.asarray([[float(r[f"p_{name}_{c}"]) for c in fusion.CLASS_NAMES]
                                       for r in rows]) for name in ("flat", "fusion")}
    canonical = prediction_rows(shared, probabilities["flat"], probabilities["fusion"],
                                seed=seed, expected_cohort=lock["cohort"])
    eq(fields, list(canonical[0]), "Prediction CSV columns")
    # Validate every exported derived field, boolean, seed and probability.
    for row, expected in zip(rows, canonical):
        for key, value in expected.items():
            actual = row[key]
            if isinstance(value, bool):
                eq(actual, str(value), f"Prediction {key}")
            elif isinstance(value, (int, float)):
                eq(type(value)(actual), value, f"Prediction {key}")
            else:
                eq(actual, value, f"Prediction {key}")
    return canonical


def _identity(seed, lock):
    return {"phase": "COM-05", "seed": seed, "preflight_lock_sha256": fit._digest(lock),
            "cohort_sha256": lock["cohort_sha256"]}


def _seed_marker(directory, seed, lock, stage, *, inference_hash=None):
    files = INFERENCE_FILES if stage == "inference" else ANALYSIS_FILES
    result = {**_identity(seed, lock), "stage": stage, "status": "PASS",
              "artifacts": {name: sha256_file(directory / name) for name in sorted(files)}}
    if stage == "analysis":
        result["inference_completion_sha256"] = inference_hash
    return result


def _inference_metadata(seed, lock):
    return {**_identity(seed, lock), "zero_shot": True, "sample_count": sum(COUNTS),
            "patient_count": PATIENT_COUNT, "class_order": list(fusion.CLASS_NAMES),
            "feature_order": list(fusion.FEATURE_ORDER),
            "checkpoint_provenance": [r for r in lock["checkpoints"] if r["seed"] == seed],
            "fusion_sha256": lock["common_input_sha256"][lock["fusion_paths"][str(seed)]],
            "statistics_performed": False}


def validate_inference_seed(directory, seed, lock):
    eq(fit._visible(directory), INFERENCE_FILES | {"seed_inference_complete.json"}, "Inference seed file set")
    eq(_read_json(directory / "seed_inference_complete.json"), _seed_marker(directory, seed, lock, "inference"),
       "Inference provenance/hash drift")
    eq(_read_json(directory / "inference_metadata.json"), _inference_metadata(seed, lock), "Inference metadata")
    return _read_predictions(directory / "paired_predictions.csv", seed, lock)


def validate_analysis_seed(directory, seed, lock, inference_directory):
    eq(fit._visible(directory), ANALYSIS_FILES | {"seed_analysis_complete.json"}, "Analysis seed file set")
    source_hash = sha256_file(inference_directory / "seed_inference_complete.json")
    eq(_read_json(directory / "seed_analysis_complete.json"),
       _seed_marker(directory, seed, lock, "analysis", inference_hash=source_hash), "Analysis provenance/hash drift")
    result = _read_json(directory / "metrics_and_statistics.json")
    for key, value in _identity(seed, lock).items():
        eq(result.get(key), value, f"Analysis {key}")
    return result


def _inference_complete(output, lock):
    return {"status": "PASS", "preflight_lock_sha256": fit._digest(lock), "seeds": list(SEEDS),
            "seed_completion_sha256": {str(s): sha256_file(output / "inference" / f"seed{s}" /
                                                          "seed_inference_complete.json") for s in SEEDS}}


def _analysis_complete(output, lock):
    return {"status": "PASS", "preflight_lock_sha256": fit._digest(lock), "seeds": list(SEEDS),
            "inference_completion_sha256": sha256_file(output / "inference" / INFERENCE_COMPLETE),
            "summary_sha256": sha256_file(output / "analysis" / SUMMARY_NAME),
            "seed_completion_sha256": {str(s): sha256_file(output / "analysis" / f"seed{s}" /
                                                          "seed_analysis_complete.json") for s in SEEDS}}


def validate_outputs(output, lock):
    """Validate all durable stages before work; no metrics/bootstrap/model calls."""
    visible = fit._visible(output)
    if visible - {LOCK_NAME, "inference", "analysis"}:
        raise ValueError("Unexpected visible COM-05 output")
    inf, ana = output / "inference", output / "analysis"
    iv, av = fit._visible(inf), fit._visible(ana)
    seed_names = {f"seed{s}" for s in SEEDS}
    if iv - (seed_names | {INFERENCE_COMPLETE}) or av - (seed_names | {SUMMARY_NAME, COMPLETE_NAME}):
        raise ValueError("Unexpected visible stage artifact")
    predictions = {s: validate_inference_seed(inf / f"seed{s}", s, lock)
                   for s in SEEDS if f"seed{s}" in iv}
    if INFERENCE_COMPLETE in iv:
        if len(predictions) != 3:
            raise ValueError("Partial inference completion")
        eq(_read_json(inf / INFERENCE_COMPLETE), _inference_complete(output, lock), "Inference completion drift")
    if av and INFERENCE_COMPLETE not in iv:
        raise ValueError("Analysis requires completed inference")
    analyses = {s: validate_analysis_seed(ana / f"seed{s}", s, lock, inf / f"seed{s}")
                for s in SEEDS if f"seed{s}" in av}
    if SUMMARY_NAME in av or COMPLETE_NAME in av:
        if len(analyses) != 3 or not {SUMMARY_NAME, COMPLETE_NAME} <= av:
            raise ValueError("Partial final analysis publication")
        eq(_read_json(ana / COMPLETE_NAME), _analysis_complete(output, lock), "Analysis completion drift")
    return predictions, analyses


def run_external_hiba_preflight(project_root, *, persist_callback=None):
    output = output_directory(project_root)
    visible = fit._visible(output)
    if visible and LOCK_NAME not in visible:
        raise ValueError("Cannot create before-inference lock after visible output")
    lock = inspect_inputs(project_root)
    if LOCK_NAME in visible:
        eq(_read_json(output / LOCK_NAME), lock, "Preflight identity drift")
    validate_outputs(output, lock)
    _publish_json(output / LOCK_NAME, lock)
    if persist_callback is not None:
        persist_callback()
    return lock


def _publish_seed(output, seed, lock, stage, *, rows=None, result=None, rescue=None):
    if stage not in ("inference", "analysis"):
        raise ValueError("Unknown publication stage")
    parent = output / stage
    if parent.resolve() != output.resolve() / stage:
        raise ValueError("Redirected stage output")
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / f"seed{seed}"
    if destination.exists():
        raise FileExistsError("Refusing to overwrite completed scientific output")
    staging = Path(tempfile.mkdtemp(prefix=f".seed{seed}.", dir=parent))
    if stage == "inference":
        with (staging / "paired_predictions.csv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_json(staging / "inference_metadata.json", _inference_metadata(seed, lock))
        marker = _seed_marker(staging, seed, lock, stage)
        _publish_json(staging / "seed_inference_complete.json", marker)
        validate_inference_seed(staging, seed, lock)
    elif stage == "analysis":
        inf = output / "inference" / f"seed{seed}"
        _publish_json(staging / "metrics_and_statistics.json", result)
        _publish_json(staging / "rescue_analysis.json", rescue)
        marker = _seed_marker(staging, seed, lock, stage,
                              inference_hash=sha256_file(inf / "seed_inference_complete.json"))
        _publish_json(staging / "seed_analysis_complete.json", marker)
        validate_analysis_seed(staging, seed, lock, inf)
    else:
        raise ValueError("Unknown publication stage")
    if destination.exists():
        raise FileExistsError("Seed appeared during publication")
    staging.rename(destination)


def infer_one_seed(root, seed, lock, *, device):
    frozen = load_frozen_fusion(root, seed, lock)
    loader = hiba.build_hiba_loader(Path(root))
    eq(_cohort(loader.dataset.rows), lock["cohort"], "HIBA loader pairing before forward")
    collections = {}
    for spec in (next(s for s in com04.FLAT_SPECS if s.seed == seed),
                 next(s for s in com04.SHARED_SPECS if s.seed == seed)):
        record = next(r for r in lock["checkpoints"] if r["seed"] == seed and r["system"] == spec.system)
        model = com04.historical.load_selected_model(root, spec, expected_sha256=record["sha256"])
        try:
            if spec.system == "flat":
                collections["flat"] = com04.historical.collect_single_task_predictions(
                    model, loader, class_names=fusion.CLASS_NAMES, device=device)
            else:
                collections["shared"] = fusion.collect_shared_predictions(model, loader, device=device)
        finally:
            model.cpu()
            del model
    flat, shared = collections["flat"], collections["shared"]
    fusion._aligned_ids(flat.sample_ids, shared.sample_ids, len(shared.flat_targets))
    eq(np.asarray(flat.targets).tolist(), shared.flat_targets.tolist(), "Flat/Shared targets")
    eq(np.asarray(flat.predictions).tolist(), np.asarray(flat.probabilities).argmax(1).tolist(), "Flat argmax")
    features = fusion.build_fusion_features(shared.stage1_probabilities, shared.stage2_probabilities,
                                            sample_ids=shared.sample_ids, task2_sample_ids=shared.sample_ids)
    probabilities, _ = fusion.predict_fusion(frozen, features)
    return prediction_rows(shared, flat.probabilities, probabilities, seed=seed, expected_cohort=lock["cohort"])


def run_external_hiba_inference(project_root, *, device="cuda", persist_callback=None):
    root = Path(project_root).absolute()
    output = output_directory(root)
    lock = verify_preflight_lock(root, inference=True)
    predictions, _ = validate_outputs(output, lock)
    if (output / "inference" / INFERENCE_COMPLETE).exists():
        return _inference_complete(output, lock)
    com04.historical.torch.backends.cudnn.benchmark = False
    for seed in SEEDS:
        if seed in predictions:
            continue
        rows = infer_one_seed(root, seed, lock, device=device)
        verify_input_bytes(root, lock, inference=True)
        _publish_seed(output, seed, lock, "inference", rows=rows)
        if persist_callback is not None:
            persist_callback()
    marker = _inference_complete(output, lock)
    _publish_json(output / "inference" / INFERENCE_COMPLETE, marker)
    if persist_callback is not None:
        persist_callback()
    return marker


def analyze_rows(rows, seed, lock):
    """Stage C only; arrays from persisted inference, zero neural forwards."""
    truth = np.asarray([r["true_final_class"] for r in rows], dtype=np.int64)
    patients = [r["patient_id"] for r in rows]
    eq(cohort_identity([r["sample_id"] for r in rows], patients, truth), lock["cohort"], "Analysis cohort")
    predictions = {s: np.asarray([r[f"{s}_prediction"] for r in rows], dtype=np.int64) for s in SYSTEMS}
    metrics = {s: fusion.four_class_metrics(truth, predictions[s]) for s in SYSTEMS}
    rescue = fusion.routing_rescue(truth, np.asarray([r["task1_prediction"] for r in rows], dtype=np.int64),
                                   predictions["hard"], predictions["fusion"])
    rescue.update(dataset="hiba", seed=seed)
    def counts(items):
        return {"count": len(items), "hard_correct": sum(r["hard_correct"] for r in items),
                "fusion_correct": sum(r["fusion_correct"] for r in items),
                "rescued": sum(r["rescue_category"] == com04.PARTITIONS[0] for r in items),
                "harmed": sum(r["rescue_category"] == com04.PARTITIONS[1] for r in items)}
    for name, flag in (("malignant_routed_nm", "malignant_routing_failure"),
                       ("nm_routed_malignant", "nm_routing_failure")):
        selected = [r for r in rows if r[flag]]
        rescue[name].update(counts(selected))
        if name == "malignant_routed_nm":
            rescue[name]["by_class"] = {fusion.CLASS_NAMES[c]: counts([
                r for r in selected if r["true_final_class"] == c]) for c in (1, 2, 3)}
        else:
            rescue[name].update(corrections=rescue[name]["rescued"], new_errors=rescue[name]["harmed"])
    eq(sum(rescue["pairwise_partition"].values()), sum(COUNTS), "Rescue partition coverage")
    result = {**_identity(seed, lock), "sample_count": len(rows), "patient_count": len(set(patients)),
              "primary_endpoint": "four_class_macro_f1", "primary_contrast": "fusion_minus_hard",
              "oracle_diagnostic_only": True, "systems": metrics,
              "contrasts": {name: {m: metrics[a][m] - metrics[b][m] for m in METRICS}
                            for name, (a, b) in fusion.CONTRASTS.items()},
              "statistics": {"bootstrap": fusion.patient_cluster_bootstrap(
                  truth, predictions, patients, replicates=5000, seed=42)},
              "additional_paired_tests": "none_declared_for_hiba_in_COM-01B"}
    return result, rescue


def three_seed_summary(results):
    # Generic scalar aggregation only, preserving the completed COM-04 source.
    summary = com04.three_seed_summary(results)
    summary.update(phase="COM-05", dataset="hiba", zero_shot=True)
    return summary


def run_external_hiba_analysis(project_root, *, persist_callback=None):
    root = Path(project_root).absolute()
    output = output_directory(root)
    lock = verify_preflight_lock(root)  # No neural weights or image access.
    predictions, results = validate_outputs(output, lock)
    if not (output / "inference" / INFERENCE_COMPLETE).is_file():
        raise ValueError("CPU analysis requires all completed inference seeds")
    if (output / "analysis" / COMPLETE_NAME).exists():
        summary = three_seed_summary(results)
        eq(_read_json(output / "analysis" / SUMMARY_NAME), summary, "Summary drift")
        return summary
    for seed in SEEDS:
        if seed in results:
            continue
        result, rescue = analyze_rows(predictions[seed], seed, lock)
        verify_input_bytes(root, lock)
        # Recheck all inference bytes after long CPU statistics, before publication.
        validate_outputs(output, lock)
        _publish_seed(output, seed, lock, "analysis", result=result, rescue=rescue)
        if persist_callback is not None:
            persist_callback()
        results[seed] = result
    summary = three_seed_summary(results)
    verify_input_bytes(root, lock)
    _publish_json(output / "analysis" / SUMMARY_NAME, summary)
    _publish_json(output / "analysis" / COMPLETE_NAME, _analysis_complete(output, lock))
    if persist_callback is not None:
        persist_callback()
    return summary
