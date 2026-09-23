"""COM-06A read-only evidence inventory for the CoMeSySo probability-fusion study.

This script does not train, infer, bootstrap, or recompute scientific metrics.
It validates completion-marker hashes from the frozen COM-03/04/05 artifact tree
and emits a compact machine-readable/Markdown inventory for manuscript synthesis.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SEEDS = (42, 123, 2026)
SYSTEMS = ("flat", "hard", "path_soft", "fusion", "oracle")
CLASSES = ("non_malignant", "melanoma", "bcc", "scc")
RESULT_ROOT = Path("results/extensions/comesyso2026_probability_fusion")
REPORT_ROOT = Path("reports/extensions/comesyso2026_probability_fusion")


def _read(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _require(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def _verify_seed_completion_hashes(base: Path, marker: dict, filename: str):
    expected = marker.get("seed_completion_sha256")
    _require(isinstance(expected, dict), f"Missing seed_completion_sha256 in {base}")
    _require(set(expected) == {str(s) for s in SEEDS}, f"Unexpected seed set in {base}")
    for seed in SEEDS:
        path = base / f"seed{seed}" / filename
        _require(path.is_file(), f"Missing completion file: {path}")
        _require(_sha256(path) == expected[str(seed)], f"Completion hash drift: {path}")


def verify_completed_artifacts(root: Path):
    validation = root / "validation_fit"
    internal = root / "internal_isic"
    external = root / "external_hiba"

    vf_complete = _read(validation / "validation_fit_complete.json")
    _require(vf_complete.get("status") == "PASS", "COM-03 completion status is not PASS")
    _require(vf_complete.get("seeds") == list(SEEDS), "COM-03 completion seed drift")
    _require(_sha256(validation / "three_seed_fit_summary.json") == vf_complete["summary_sha256"],
             "COM-03 summary hash drift")
    _verify_seed_completion_hashes(validation, vf_complete, "seed_complete.json")

    int_complete = _read(internal / "internal_isic_complete.json")
    _require(int_complete.get("status") == "PASS", "COM-04 completion status is not PASS")
    _require(int_complete.get("seeds") == list(SEEDS), "COM-04 completion seed drift")
    _require(_sha256(internal / "three_seed_summary.json") == int_complete["summary_sha256"],
             "COM-04 summary hash drift")
    _verify_seed_completion_hashes(internal, int_complete, "seed_complete.json")

    ext_inf = external / "inference"
    ext_ana = external / "analysis"
    ext_inf_complete = _read(ext_inf / "inference_complete.json")
    _require(ext_inf_complete.get("status") == "PASS", "COM-05 inference completion status is not PASS")
    _require(ext_inf_complete.get("seeds") == list(SEEDS), "COM-05 inference seed drift")
    _verify_seed_completion_hashes(ext_inf, ext_inf_complete, "seed_inference_complete.json")

    ext_complete = _read(ext_ana / "external_hiba_complete.json")
    _require(ext_complete.get("status") == "PASS", "COM-05 analysis completion status is not PASS")
    _require(ext_complete.get("seeds") == list(SEEDS), "COM-05 analysis seed drift")
    _require(_sha256(ext_inf / "inference_complete.json") == ext_complete["inference_completion_sha256"],
             "COM-05 inference completion anchor drift")
    _require(_sha256(ext_ana / "three_seed_summary.json") == ext_complete["summary_sha256"],
             "COM-05 summary hash drift")
    _verify_seed_completion_hashes(ext_ana, ext_complete, "seed_analysis_complete.json")

    return {
        "validation_fit_complete_sha256": _sha256(validation / "validation_fit_complete.json"),
        "internal_isic_complete_sha256": _sha256(internal / "internal_isic_complete.json"),
        "external_hiba_inference_complete_sha256": _sha256(ext_inf / "inference_complete.json"),
        "external_hiba_complete_sha256": _sha256(ext_ana / "external_hiba_complete.json"),
    }


def _metric_summary(summary: dict, system: str, metric: str):
    item = summary["systems"][system][metric]
    return {
        "individual_seed_values": item["individual_seed_values"],
        "mean": item["mean"],
        "sample_standard_deviation": item["sample_standard_deviation"],
    }


def _contrast_summary(summary: dict, name: str, metric: str):
    item = summary["contrasts"][name][metric]
    return {
        "individual_seed_values": item["individual_seed_values"],
        "mean": item["mean"],
        "sample_standard_deviation": item["sample_standard_deviation"],
    }


def _primary_statistics(seed_result: dict):
    stats = seed_result["statistics"]
    bootstrap = stats["bootstrap"]["contrasts"]["fusion_minus_hard"]
    out = {
        "bootstrap_unit": stats["bootstrap"]["unit"],
        "bootstrap_replicates": stats["bootstrap"]["replicates"],
        "bootstrap_seed": stats["bootstrap"]["seed"],
        "point_difference": bootstrap["point_difference"],
        "confidence_interval_95": bootstrap["confidence_interval"],
    }
    if "mcnemar" in stats:
        mc = stats["mcnemar"]["fusion_minus_hard"]
        out["mcnemar"] = {
            "method": mc["method"],
            "unit": mc["unit"],
            "interpretation": mc["interpretation"],
            "minuend_correct_subtrahend_wrong": mc["minuend_correct_subtrahend_wrong"],
            "minuend_wrong_subtrahend_correct": mc["minuend_wrong_subtrahend_correct"],
            "p_value": mc["p_value"],
        }
    return out


def _dataset_inventory(root: Path, *, external: bool):
    if external:
        base = root / "external_hiba" / "analysis"
        summary_path = base / "three_seed_summary.json"
        seed_base = base
        expected_phase = "COM-05"
    else:
        base = root / "internal_isic"
        summary_path = base / "three_seed_summary.json"
        seed_base = base
        expected_phase = "COM-04"

    summary = _read(summary_path)
    _require(summary.get("phase") == expected_phase, f"{expected_phase} summary phase drift")
    _require(summary.get("seeds") == list(SEEDS), f"{expected_phase} seed drift")
    _require(summary.get("std_ddof") == 1, f"{expected_phase} must use sample SD ddof=1")
    _require(summary.get("pooled") is False, f"{expected_phase} must not pool seeds")
    _require(summary.get("primary_endpoint") == "four_class_macro_f1", f"{expected_phase} endpoint drift")
    _require(summary.get("primary_contrast") == "fusion_minus_hard", f"{expected_phase} contrast drift")
    if external:
        _require(summary.get("dataset") == "hiba", "COM-05 dataset drift")
        _require(summary.get("zero_shot") is True, "COM-05 must remain zero-shot")

    seeds = {}
    for seed in SEEDS:
        metrics = _read(seed_base / f"seed{seed}" / "metrics_and_statistics.json")
        rescue = _read(seed_base / f"seed{seed}" / "rescue_analysis.json")
        _require(metrics.get("seed") == seed, f"Seed result drift: {seed}")
        _require(rescue.get("seed") == seed, f"Rescue seed drift: {seed}")
        seeds[str(seed)] = {
            "primary_statistics": _primary_statistics(metrics),
            "rescue": rescue,
        }

    return {
        "summary_sha256": _sha256(summary_path),
        "systems_macro_f1": {system: _metric_summary(summary, system, "macro_f1") for system in SYSTEMS},
        "systems_accuracy": {system: _metric_summary(summary, system, "accuracy") for system in SYSTEMS},
        "systems_balanced_accuracy": {
            system: _metric_summary(summary, system, "balanced_accuracy") for system in SYSTEMS
        },
        "primary_contrast_macro_f1": _contrast_summary(summary, "fusion_minus_hard", "macro_f1"),
        "path_soft_minus_hard_macro_f1": _contrast_summary(summary, "path_soft_minus_hard", "macro_f1"),
        "fusion_minus_flat_macro_f1": _contrast_summary(summary, "fusion_minus_flat", "macro_f1"),
        "oracle_minus_fusion_macro_f1": _contrast_summary(summary, "oracle_minus_fusion", "macro_f1"),
        "fusion_per_class_f1": {
            cls: {
                "individual_seed_values": summary["per_class"]["fusion"][cls]["f1"]["individual_seed_values"],
                "mean": summary["per_class"]["fusion"][cls]["f1"]["mean"],
                "sample_standard_deviation": summary["per_class"]["fusion"][cls]["f1"]["sample_standard_deviation"],
            }
            for cls in CLASSES
        },
        "hard_per_class_f1": {
            cls: {
                "individual_seed_values": summary["per_class"]["hard"][cls]["f1"]["individual_seed_values"],
                "mean": summary["per_class"]["hard"][cls]["f1"]["mean"],
                "sample_standard_deviation": summary["per_class"]["hard"][cls]["f1"]["sample_standard_deviation"],
            }
            for cls in CLASSES
        },
        "seeds": seeds,
    }


def build_inventory(root: Path):
    completion_hashes = verify_completed_artifacts(root)
    validation_summary = _read(root / "validation_fit" / "three_seed_fit_summary.json")
    return {
        "phase": "COM-06A",
        "status": "PASS",
        "purpose": "read_only_evidence_inventory",
        "scientific_recomputation": False,
        "seeds": list(SEEDS),
        "completion_hashes": completion_hashes,
        "validation_fit": {
            "summary_sha256": _sha256(root / "validation_fit" / "three_seed_fit_summary.json"),
            "summary": validation_summary,
        },
        "internal_isic": _dataset_inventory(root, external=False),
        "external_hiba": _dataset_inventory(root, external=True),
    }


def _fmt(item: dict) -> str:
    return f"{item['mean']:.4f} ± {item['sample_standard_deviation']:.4f}"


def render_markdown(inv: dict) -> str:
    internal = inv["internal_isic"]
    external = inv["external_hiba"]
    lines = [
        "# COM-06A Evidence Inventory",
        "",
        "Status: **PASS**",
        "",
        "This inventory is read-only. It validates the frozen COM-03/04/05 completion",
        "markers and summarizes already-published metrics; it does not rerun inference,",
        "bootstrap, fitting, training, or metric computation.",
        "",
        "## Primary system results",
        "",
        "| Dataset | Flat | Hard | Path-Soft | Fusion | Oracle |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, data in (("ISIC internal test", internal), ("HIBA zero-shot", external)):
        m = data["systems_macro_f1"]
        lines.append("| " + label + " | " + " | ".join(_fmt(m[s]) for s in SYSTEMS) + " |")
    lines += [
        "",
        "Values are four-class macro-F1, mean ± sample SD across seeds 42, 123, 2026; seeds are not pooled.",
        "",
        "## Primary contrast: Fusion - Hard",
        "",
        "| Dataset | Seed 42 | Seed 123 | Seed 2026 | Mean ± SD |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, data in (("ISIC internal test", internal), ("HIBA zero-shot", external)):
        c = data["primary_contrast_macro_f1"]
        v = c["individual_seed_values"]
        lines.append(
            f"| {label} | {v['42']:.4f} | {v['123']:.4f} | {v['2026']:.4f} | {_fmt(c)} |"
        )
    lines += [
        "",
        "## Secondary macro-F1 contrasts",
        "",
        "| Dataset | Path-Soft - Hard | Fusion - Flat | Oracle - Fusion |",
        "|---|---:|---:|---:|",
    ]
    for label, data in (("ISIC internal test", internal), ("HIBA zero-shot", external)):
        lines.append(
            f"| {label} | {_fmt(data['path_soft_minus_hard_macro_f1'])} | "
            f"{_fmt(data['fusion_minus_flat_macro_f1'])} | "
            f"{_fmt(data['oracle_minus_fusion_macro_f1'])} |"
        )
    lines += [
        "",
        "## Fusion per-class F1",
        "",
        "| Dataset | Non-malignant | Melanoma | BCC | SCC |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, data in (("ISIC internal test", internal), ("HIBA zero-shot", external)):
        p = data["fusion_per_class_f1"]
        lines.append("| " + label + " | " + " | ".join(_fmt(p[c]) for c in CLASSES) + " |")
    lines += [
        "",
        "## Seed-level primary uncertainty",
        "",
        "The JSON inventory preserves, for every seed and dataset, the frozen paired-bootstrap",
        "95% confidence interval for Fusion - Hard macro-F1. ISIC also preserves the declared",
        "image-level exact McNemar correctness test; HIBA intentionally has no additional",
        "image-level McNemar test under the frozen protocol.",
        "",
        "## Routing rescue evidence",
        "",
        "The JSON inventory preserves each seed's complete frozen rescue-analysis object,",
        "including the five-way Hard/Fusion correctness partition, malignant→NM routing",
        "failures, NM→malignant routing failures, rescues, harms, and subtype counts.",
        "",
        "## Provenance",
        "",
    ]
    for key, value in inv["completion_hashes"].items():
        lines.append(f"- \`{key}\`: \`{value}\`")
    lines += ["", "COM-06A is ready for publication-table and narrative synthesis.", ""]
    return "\\n".join(lines)


def main():
    root = RESULT_ROOT
    _require(root.is_dir(), f"Frozen result directory is missing: {root}")
    inv = build_inventory(root)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_ROOT / "com06a_evidence_inventory.json"
    md_path = REPORT_ROOT / "COM06A_EVIDENCE_INVENTORY.md"
    json_path.write_text(json.dumps(inv, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    md_path.write_text(render_markdown(inv), encoding="utf-8")
    print("COM-06A: PASS")
    print(f"JSON: {json_path}")
    print(f"MARKDOWN: {md_path}")
    print(f"ISIC Fusion-Hard macro-F1: {_fmt(inv['internal_isic']['primary_contrast_macro_f1'])}")
    print(f"HIBA Fusion-Hard macro-F1: {_fmt(inv['external_hiba']['primary_contrast_macro_f1'])}")


if __name__ == "__main__":
    main()
