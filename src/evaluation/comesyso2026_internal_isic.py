"""COM-04 locked ISIC internal test. Importing performs no study execution.

Only the dedicated CoMeSySo output tree is writable. Serialize all launchers;
Modal's busy check is advisory, as in COM-03. No fitting entry point exists.
"""
from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

import numpy as np
import yaml

from src.data.isic2019_dataset import FLAT_FOUR_CLASS_TO_INDEX, map_flat_diagnosis
from src.evaluation import comesyso2026_probability_fusion as fusion
from src.evaluation import comesyso2026_validation_fit as fit
from src.evaluation import paul2026_swin_evaluation as historical
from src.evaluation.paul2026_swin_evaluation import (
    _publish_json, _read_json, _require_equal, sha256_file,
)

SEEDS = (42, 123, 2026)
FLAT_SPECS = tuple(historical.CheckpointSpec("flat", s, e)
                   for s, e in zip(SEEDS, (9, 10, 17)))
SHARED_SPECS = tuple(historical.CheckpointSpec("shared_hard", s, e)
                     for s, e in zip(SEEDS, (10, 1, 26)))
SYSTEMS = ("flat", "hard", "path_soft", "fusion", "oracle")
METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
COUNTS = (2398, 678, 498, 94)
OUTPUT = Path("results/extensions/comesyso2026_probability_fusion/internal_isic")
LOCK_NAME = "preflight_lock.json"
SUMMARY_NAME = "three_seed_summary.json"
COMPLETE_NAME = "internal_isic_complete.json"
SEED_FILES = {"paired_predictions.csv", "metrics_and_statistics.json", "rescue_analysis.json"}
HISTORICAL_LOCK = historical.EVALUATION / historical.LOCK_NAME
SOURCE_PATHS = tuple(dict.fromkeys((*fit.SOURCE_PATHS,
    "src/evaluation/comesyso2026_internal_isic.py",
    "scripts/comesyso2026/modal_comesyso2026_internal_isic.py",
    "scripts/comesyso2026/launch_comesyso2026_internal_isic.py",
    "src/evaluation/phase04_comparative_harness.py",
    "src/evaluation/classification_metrics.py",
    "src/analysis/stored_prediction_statistics.py")))
PARTITIONS = ("hard_wrong_fusion_correct", "hard_correct_fusion_wrong", "both_correct",
              "both_wrong_same_prediction", "both_wrong_different_predictions")


def output_directory(project_root):
    root = Path(project_root)
    frozen = Path("results/extensions/comesyso2026_probability_fusion/internal_isic")
    if ".." in root.parts or OUTPUT != frozen:
        raise ValueError("Output must use the frozen relative path without traversal")
    output = root.absolute() / OUTPUT
    # Allow the CoMeSySo volume mount, but never redirect its internal subtree.
    if output.resolve() != output.parent.resolve() / output.name:
        raise ValueError("Redirected COM-04 output")
    return output


def require_scope(dataset="isic2019", split="internal_test", stage="flat_four_class"):
    if (dataset, split, stage) != ("isic2019", "internal_test", "flat_four_class"):
        raise ValueError("COM-04 accepts only ISIC2019 internal_test / flat_four_class")


def cohort_identity(ids, targets):
    truth = fusion._labels(targets)
    ids = fusion._ids(ids, len(truth))
    _require_equal(np.bincount(truth, minlength=4).tolist(), list(COUNTS), "Test class counts")
    return {"sample_ids": list(ids), "targets": truth.tolist()}


def manifest_cohort(root):
    """Metadata only; no dataset construction, image access, or inference."""
    with (Path(root) / fit.MANIFEST).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    _require_equal(sorted({r["dataset"].strip() for r in rows} - {""}),
                   ["isic2019"], "Manifest dataset")
    selected = [r for r in rows if r["split_included"] == "1"
                and r["split"] in ("internal_test", "test") and r["include_stage_1"] == "1"]
    # Reject external/redirected image paths before any loader can open them.
    raw = (Path(root) / "data/raw").resolve()
    for row in selected:
        path = Path(row["image_path"])
        if path.is_absolute() or ".." in path.parts or "hiba" in path.as_posix().lower():
            raise ValueError("Non-ISIC image path")
        if not (Path(root) / path).resolve().is_relative_to(raw):
            raise ValueError("Image path escapes ISIC raw-data mount")
    return cohort_identity([r["image_id"] for r in selected], np.asarray([
        FLAT_FOUR_CLASS_TO_INDEX[map_flat_diagnosis(r["diagnosis_canonical"])]
        for r in selected], dtype=np.int64))


def _completed_fits(root):
    """Verify frozen COM-03 bundles, including final completion; never fit."""
    lock = fit.verify_preflight_lock(root)
    output = fit.output_directory(root)
    _require_equal(fit._visible(output), {fit.LOCK_NAME, fit.SUMMARY_NAME, fit.COMPLETE_NAME}
                   | {f"seed{s}" for s in SEEDS}, "COM-03 completed file set")
    results, cohort, models = {}, None, {}
    for seed in SEEDS:
        model, metadata, cohort = fit.validate_completed_seed(
            output / f"seed{seed}", seed, lock, expected_cohort=cohort)
        models[seed], results[seed] = model, metadata
    _require_equal(_read_json(output / fit.SUMMARY_NAME), fit.three_seed_summary(results),
                   "COM-03 summary")
    _require_equal(_read_json(output / fit.COMPLETE_NAME), fit._final_marker(output, lock),
                   "COM-03 completion marker")
    return lock, models


def inspect_inputs(project_root):
    """CPU-only provenance check. Does not construct an internal-test dataset."""
    root = Path(project_root).absolute()
    com03, _ = _completed_fits(root)
    protocol = yaml.safe_load((root / fusion.PROTOCOL_PATH).read_text(encoding="utf-8"))
    fit.validate_protocol(protocol)
    for selection, spec in zip(protocol["models"]["selections"], FLAT_SPECS):
        _require_equal(selection.get("flat_run"), spec.run_name, "Flat selection")
        _require_equal(selection.get("flat_selected_epoch"), spec.epoch, "Flat epoch")
    _require_equal(protocol["endpoints"]["primary_metric"], "four_class_macro_f1", "Endpoint")
    comparisons = [protocol["endpoints"]["primary_comparison"],
                   *protocol["endpoints"]["secondary_comparisons"]]
    _require_equal({r["id"]: (r["minuend"], r["subtrahend"]) for r in comparisons},
                   fusion.CONTRASTS, "Contrasts")
    # A mandatory historical lock anchors Flat bytes; COM-03 anchors Shared bytes.
    old = _read_json(root / HISTORICAL_LOCK)
    _require_equal(old.get("status"), "PASS", "Historical lock status")
    _require_equal(old.get("manifest_sha256"), com03["validation_manifest_sha256"], "Manifest anchor")
    files = {fusion.PROTOCOL_PATH: com03["protocol_sha256"],
             fit.MANIFEST.as_posix(): com03["validation_manifest_sha256"],
             HISTORICAL_LOCK.as_posix(): sha256_file(root / HISTORICAL_LOCK)}
    for record in com03["checkpoints"]:
        files[record["config_path"]] = record["config_sha256"]
    records = []
    for spec in (*FLAT_SPECS, *SHARED_SPECS):
        matches = [r for r in old.get("checkpoints", [])
                   if r.get("system") == spec.system and r.get("seed") == spec.seed]
        if len(matches) != 1:
            raise ValueError("Historical checkpoint lock requires a unique selection")
        anchor = matches[0]
        _require_equal(anchor.get("run_name"), spec.run_name, "Checkpoint run")
        _require_equal(anchor.get("selected_epoch"), spec.epoch, "Checkpoint epoch")
        path = spec.path(root)
        files[path.relative_to(root).as_posix()] = anchor["sha256"]
        files[(path.parent / "run_summary.json").relative_to(root).as_posix()] = anchor["run_summary_sha256"]
        if spec.system == "shared_hard":
            record = next(r for r in com03["checkpoints"] if r["seed"] == spec.seed)
            _require_equal(anchor["sha256"], record["checkpoint_sha256"], "COM-03 checkpoint hash")
        model = historical.load_selected_model(root, spec, expected_sha256=anchor["sha256"])
        del model
        records.append({"system": spec.system, "seed": spec.seed, "epoch": spec.epoch,
                        "path": path.relative_to(root).as_posix(), "sha256": anchor["sha256"]})
    fit_output = fit.output_directory(root)
    paths = [fit_output / name for name in (fit.LOCK_NAME, fit.SUMMARY_NAME, fit.COMPLETE_NAME)]
    paths += [fit_output / f"seed{s}" / name for s in SEEDS
              for name in sorted(fit.SEED_FILES | {"seed_complete.json"})]
    files.update({p.relative_to(root).as_posix(): sha256_file(p) for p in paths})
    files.update({p: sha256_file(root / p) for p in SOURCE_PATHS})
    cohort = manifest_cohort(root)
    lock = {"schema_version": 1, "phase": "COM-04", "status": "PASS",
            "protocol_identity": fusion.PROTOCOL_ID, "seeds": list(SEEDS),
            "dataset": "isic2019", "split": "internal_test", "stage": "flat_four_class",
            "created_before_test_inference": True, "internal_test_dataset_constructed": False,
            "hiba_access": False, "input_sha256": files, "checkpoints": records,
            "cohort": cohort, "feature_order": list(fusion.FEATURE_ORDER),
            "class_order": list(fusion.CLASS_NAMES), "com03_lock_sha256": fit._digest(com03)}
    verify_input_bytes(root, lock, require_lock=False)
    # Revalidate completion after capturing hashes to reject concurrent COM-03 changes.
    confirmed, _ = _completed_fits(root)
    _require_equal(confirmed, com03, "COM-03 changed during preflight")
    verify_input_bytes(root, lock, require_lock=False)
    return lock


def verify_input_bytes(root, lock, *, require_lock=True):
    for path, digest in lock["input_sha256"].items():
        _require_equal(sha256_file(Path(root) / path), digest, f"Locked input drift: {path}")
    if require_lock:
        _require_equal(_read_json(output_directory(root) / LOCK_NAME), lock, "COM-04 lock drift")


def run_internal_isic_preflight(project_root, *, persist_callback=None):
    output = output_directory(project_root)
    visible = fit._visible(output)
    if LOCK_NAME not in visible and visible:
        raise ValueError("Cannot create before-inference lock after visible output")
    lock = inspect_inputs(project_root)
    if LOCK_NAME in visible:
        _require_equal(_read_json(output / LOCK_NAME), lock, "COM-04 preflight drift")
    validate_outputs(output, lock)
    _publish_json(output / LOCK_NAME, lock)
    if persist_callback is not None:
        persist_callback()
    return lock


def verify_preflight_lock(root):
    output = output_directory(root)
    fit._visible(output)
    if not (output / LOCK_NAME).is_file():
        raise FileNotFoundError("CPU COM-04 preflight lock required before inference")
    lock = _read_json(output / LOCK_NAME)
    _require_equal(inspect_inputs(root), lock, "COM-04 preflight identity drift")
    return lock


def _partition(hard_correct, fusion_correct, same):
    if not hard_correct and fusion_correct:
        return PARTITIONS[0]
    if hard_correct and not fusion_correct:
        return PARTITIONS[1]
    if hard_correct:
        return PARTITIONS[2]
    return PARTITIONS[3] if same else PARTITIONS[4]


def evaluate_collections(flat, shared, model, *, seed, expected_cohort):
    fusion._seed(seed)
    _require_equal(model.seed, seed, "Fusion seed")
    _require_equal(model.fitting_partition, fusion.FIT_PARTITION, "Fusion fit partition")
    cohort = cohort_identity(shared.sample_ids, shared.flat_targets)
    _require_equal(cohort, expected_cohort, "Test count/order/targets")
    fusion._aligned_ids(shared.sample_ids, flat.sample_ids, len(shared.flat_targets))
    _require_equal(np.asarray(flat.targets).tolist(), cohort["targets"], "Flat targets")
    flat_p = fusion._probabilities(flat.probabilities, 4, count=len(shared.flat_targets))
    _require_equal(np.asarray(flat.predictions).tolist(), flat_p.argmax(1).tolist(), "Flat argmax")
    features = fusion.build_fusion_features(shared.stage1_probabilities, shared.stage2_probabilities,
                                            sample_ids=shared.sample_ids, task2_sample_ids=shared.sample_ids)
    fp, pred = fusion.predict_fusion(model, features)
    rows, rescue = fusion.build_routing_rescue_rows(
        shared, dataset="isic", seed=seed, flat_predictions=flat.predictions,
        flat_sample_ids=flat.sample_ids, fusion_predictions=pred, fusion_sample_ids=shared.sample_ids)
    for i, row in enumerate(rows):
        row["flat_correct"] = bool(flat.predictions[i] == shared.flat_targets[i])
        row["rescue_category"] = _partition(row["hard_correct"], row["fusion_correct"],
                                           row["hard_prediction"] == row["fusion_prediction"])
        row["malignant_routing_failure"] = row["task1_target"] == 1 and row["task1_prediction"] == 0
        row["nm_routing_failure"] = row["task1_target"] == 0 and row["task1_prediction"] == 1
        for name, probabilities in (("flat", flat_p), ("fusion", fp)):
            row.update({f"p_{name}_{c}": float(probabilities[i, j])
                        for j, c in enumerate(fusion.CLASS_NAMES)})
    for name, flag in (("malignant_routed_nm", "malignant_routing_failure"),
                       ("nm_routed_malignant", "nm_routing_failure")):
        selected = [r for r in rows if r[flag]]
        def counts(items):
            return {"count": len(items), "hard_correct": sum(r["hard_correct"] for r in items),
                    "fusion_correct": sum(r["fusion_correct"] for r in items),
                    "rescued": sum(r["rescue_category"] == PARTITIONS[0] for r in items),
                    "harmed": sum(r["rescue_category"] == PARTITIONS[1] for r in items)}
        rescue[name].update(counts(selected))
        if name == "malignant_routed_nm":
            rescue[name]["by_class"] = {fusion.CLASS_NAMES[c]: counts([
                r for r in selected if r["true_final_class"] == c]) for c in (1, 2, 3)}
        else:
            rescue[name]["corrections"] = rescue[name]["rescued"]
            rescue[name]["new_errors"] = rescue[name]["harmed"]
    _require_equal(sum(rescue["pairwise_partition"].values()), sum(COUNTS), "Rescue partition")
    predictions = {s: np.asarray([r[f"{s}_prediction"] for r in rows], dtype=np.int64) for s in SYSTEMS}
    metrics = {s: fusion.four_class_metrics(shared.flat_targets, predictions[s]) for s in SYSTEMS}
    result = {"seed": seed, "primary_endpoint": "four_class_macro_f1",
              "primary_contrast": "fusion_minus_hard", "oracle_diagnostic_only": True,
              "cohort_sha256": fit._digest(cohort), "systems": metrics,
              "contrasts": {name: {m: metrics[a][m] - metrics[b][m] for m in METRICS}
                            for name, (a, b) in fusion.CONTRASTS.items()},
              "statistics": {"bootstrap": fusion.paired_image_bootstrap(shared.flat_targets, predictions,
                                                                          replicates=10000, seed=42),
                             "mcnemar": fusion.mcnemar_comparisons(shared.flat_targets, predictions)}}
    return result, rescue, rows


def three_seed_summary(results):
    if set(results) != set(SEEDS):
        raise ValueError("Exactly seeds 42, 123, 2026 required")
    for s in SEEDS:
        _require_equal(results[s]["seed"], s, "Summary seed")
        _require_equal(results[s]["cohort_sha256"], results[42]["cohort_sha256"], "Cross-seed cohort")
    def summarize(get):
        return fusion.summarize_three_seeds({s: get(results[s]) for s in SEEDS})
    return {"phase": "COM-04", "seeds": list(SEEDS), "std_ddof": 1, "pooled": False,
            "primary_endpoint": "four_class_macro_f1", "primary_contrast": "fusion_minus_hard",
            "systems": {system: {metric: summarize(lambda r: r["systems"][system][metric])
                                  for metric in METRICS} for system in SYSTEMS},
            "per_class": {system: {c: {metric: summarize(lambda r: r["systems"][system]["per_class"][c][metric])
                                       for metric in ("precision", "recall", "f1")}
                                   for c in fusion.CLASS_NAMES} for system in SYSTEMS},
            "contrasts": {name: {metric: summarize(lambda r: r["contrasts"][name][metric])
                                 for metric in METRICS} for name in fusion.CONTRASTS}}


def _seed_marker(directory, seed, lock):
    return {"status": "PASS", "seed": seed, "preflight_lock_sha256": fit._digest(lock),
            "artifacts": {p: sha256_file(directory / p) for p in sorted(SEED_FILES)}}


def validate_completed_seed(directory, seed, lock):
    _require_equal(fit._visible(directory), SEED_FILES | {"seed_complete.json"}, "Completed seed file set")
    _require_equal(_read_json(directory / "seed_complete.json"), _seed_marker(directory, seed, lock),
                   "Completed seed provenance/hash drift")
    result = _read_json(directory / "metrics_and_statistics.json")
    _require_equal(result["seed"], seed, "Completed seed")
    _require_equal(result["cohort_sha256"], fit._digest(lock["cohort"]), "Completed cohort")
    with (directory / "paired_predictions.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    cohort = cohort_identity([r["sample_id"] for r in rows],
                             np.asarray([int(r["true_final_class"]) for r in rows], dtype=np.int64))
    _require_equal(cohort, lock["cohort"], "Durable prediction cohort")
    _require_equal({int(r["seed"]) for r in rows}, {seed}, "Durable prediction seed")
    rescue = _read_json(directory / "rescue_analysis.json")
    partition = {key: sum(r["rescue_category"] == key for r in rows) for key in PARTITIONS}
    _require_equal(sum(partition.values()), sum(COUNTS), "Durable partition coverage")
    _require_equal(rescue["pairwise_partition"], partition, "Durable rescue partition")
    for system in SYSTEMS:
        metrics = fusion.four_class_metrics(np.asarray(cohort["targets"], dtype=np.int64),
                    np.asarray([int(r[f"{system}_prediction"]) for r in rows], dtype=np.int64))
        _require_equal(result["systems"][system], metrics, "Durable prediction metrics")
    return result


def _final_marker(output, lock):
    return {"status": "PASS", "preflight_lock_sha256": fit._digest(lock), "seeds": list(SEEDS),
            "summary_sha256": sha256_file(output / SUMMARY_NAME),
            "seed_completion_sha256": {str(s): sha256_file(output / f"seed{s}/seed_complete.json") for s in SEEDS}}


def validate_outputs(output, lock):
    visible = fit._visible(output)
    allowed = {LOCK_NAME, SUMMARY_NAME, COMPLETE_NAME} | {f"seed{s}" for s in SEEDS}
    if visible - allowed:
        raise ValueError("Unexpected visible COM-04 output")
    results = {s: validate_completed_seed(output / f"seed{s}", s, lock)
               for s in SEEDS if f"seed{s}" in visible}
    if SUMMARY_NAME in visible or COMPLETE_NAME in visible:
        if len(results) != 3 or not {SUMMARY_NAME, COMPLETE_NAME} <= visible:
            raise ValueError("Partial final publication")
        _require_equal(_read_json(output / SUMMARY_NAME), three_seed_summary(results), "Summary drift")
        _require_equal(_read_json(output / COMPLETE_NAME), _final_marker(output, lock), "Completion drift")
    return results


def _publish_seed(output, seed, lock, result, rescue, rows):
    destination = output / f"seed{seed}"
    if destination.exists():
        raise FileExistsError("Refusing to overwrite scientific output")
    staging = Path(tempfile.mkdtemp(prefix=f".seed{seed}.", dir=output))
    with (staging / "paired_predictions.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    _publish_json(staging / "metrics_and_statistics.json", result)
    _publish_json(staging / "rescue_analysis.json", rescue)
    _publish_json(staging / "seed_complete.json", _seed_marker(staging, seed, lock))
    validate_completed_seed(staging, seed, lock)
    if destination.exists():
        raise FileExistsError("Seed appeared during atomic publication")
    staging.rename(destination)


def evaluate_one_seed(root, seed, lock, *, device):
    models = _completed_fits(root)[1]
    loader = historical.build_locked_isic_loader(root)
    expected = cohort_identity(loader.dataset.selected_frame["image_id"].tolist(),
                               np.asarray(loader.dataset.targets, dtype=np.int64))
    _require_equal(expected, lock["cohort"], "Loader manifest order/targets")
    collections = {}
    for spec in (next(s for s in FLAT_SPECS if s.seed == seed),
                 next(s for s in SHARED_SPECS if s.seed == seed)):
        record = next(r for r in lock["checkpoints"] if r["seed"] == seed and r["system"] == spec.system)
        model = historical.load_selected_model(root, spec, expected_sha256=record["sha256"])
        if spec.system == "flat":
            collections["flat"] = historical.collect_single_task_predictions(
                model, loader, class_names=fusion.CLASS_NAMES, device=device)
        else:
            collections["shared"] = fusion.collect_shared_predictions(model, loader, device=device)
        model.cpu()
        del model
    return evaluate_collections(collections["flat"], collections["shared"], models[seed],
                                seed=seed, expected_cohort=expected)


def run_internal_isic_production(project_root, *, device="cuda", persist_callback=None):
    root = Path(project_root).absolute()
    output = output_directory(root)
    lock = verify_preflight_lock(root)
    # All durable seeds are checked before constructing even one missing loader.
    results = validate_outputs(output, lock)
    if (output / COMPLETE_NAME).exists():
        return three_seed_summary(results)
    historical.torch.backends.cudnn.benchmark = False
    for seed in SEEDS:
        if seed in results:
            continue
        verify_input_bytes(root, lock)
        result, rescue, rows = evaluate_one_seed(root, seed, lock, device=device)
        verify_input_bytes(root, lock)
        _publish_seed(output, seed, lock, result, rescue, rows)
        if persist_callback is not None:
            persist_callback()
        results[seed] = result
    summary = three_seed_summary(results)
    verify_input_bytes(root, lock)
    _publish_json(output / SUMMARY_NAME, summary)
    _publish_json(output / COMPLETE_NAME, _final_marker(output, lock))
    if persist_callback is not None:
        persist_callback()
    return summary
