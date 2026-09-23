"""COM-06B publication synthesis from the validated COM-06A inventory.

Read-only with respect to scientific results. This script does not train, infer,
fit, bootstrap, or recompute classification metrics. It converts the validated
COM-06A evidence inventory into manuscript-ready tables and a concise scientific
synthesis while preserving the frozen seed-level evidence.
"""
from __future__ import annotations

import json
from pathlib import Path

REPORT_ROOT = Path("reports/extensions/comesyso2026_probability_fusion")
INPUT = REPORT_ROOT / "com06a_evidence_inventory.json"
OUTPUT = REPORT_ROOT / "COM06B_PUBLICATION_SYNTHESIS.md"

SEEDS = ("42", "123", "2026")
SYSTEMS = ("flat", "hard", "path_soft", "fusion", "oracle")
CLASSES = ("non_malignant", "melanoma", "bcc", "scc")


def _read(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _require(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def _fmt(item: dict) -> str:
    return f"{item['mean']:.4f} ± {item['sample_standard_deviation']:.4f}"


def _fmt_ci(ci) -> str:
    _require(isinstance(ci, list) and len(ci) == 2, "Expected two-sided confidence interval")
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def _dataset_name(key: str) -> str:
    return "ISIC internal test" if key == "internal_isic" else "HIBA zero-shot"


def _validate(inv: dict):
    _require(inv.get("phase") == "COM-06A", "Expected COM-06A inventory")
    _require(inv.get("status") == "PASS", "COM-06A inventory must be PASS")
    _require(inv.get("scientific_recomputation") is False, "Inventory must remain read-only")
    _require(inv.get("seeds") == [42, 123, 2026], "Unexpected seed set")
    for dataset in ("internal_isic", "external_hiba"):
        block = inv.get(dataset)
        _require(isinstance(block, dict), f"Missing dataset block: {dataset}")
        _require(set(block["systems_macro_f1"]) == set(SYSTEMS), f"System set drift: {dataset}")
        _require(set(block["seeds"]) == set(SEEDS), f"Seed set drift: {dataset}")


def _system_table(inv: dict) -> list[str]:
    lines = [
        "## Table 1. Four-class macro-F1 across systems",
        "",
        "| Dataset | Flat | Hard | Path-Soft | Probability Fusion | Oracle |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("internal_isic", "external_hiba"):
        block = inv[dataset]["systems_macro_f1"]
        lines.append(
            "| " + _dataset_name(dataset) + " | "
            + " | ".join(_fmt(block[s]) for s in SYSTEMS) + " |"
        )
    lines += [
        "",
        "Values are mean ± sample SD across seeds 42, 123, and 2026; seeds are not pooled.",
        "Oracle is a diagnostic condition rather than a deployable system.",
        "",
    ]
    return lines


def _contrast_table(inv: dict) -> list[str]:
    lines = [
        "## Table 2. Primary comparison: Probability Fusion − Hard",
        "",
        "| Dataset | Seed 42 | Seed 123 | Seed 2026 | Mean ± SD |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in ("internal_isic", "external_hiba"):
        item = inv[dataset]["primary_contrast_macro_f1"]
        vals = item["individual_seed_values"]
        lines.append(
            f"| {_dataset_name(dataset)} | {vals['42']:.4f} | {vals['123']:.4f} | "
            f"{vals['2026']:.4f} | {_fmt(item)} |"
        )
    lines += [
        "",
        "Negative values indicate lower four-class macro-F1 for Probability Fusion than Hard routing.",
        "",
    ]
    return lines


def _secondary_table(inv: dict) -> list[str]:
    lines = [
        "## Table 3. Secondary macro-F1 contrasts",
        "",
        "| Dataset | Path-Soft − Hard | Fusion − Flat | Oracle − Fusion |",
        "|---|---:|---:|---:|",
    ]
    for dataset in ("internal_isic", "external_hiba"):
        block = inv[dataset]
        lines.append(
            f"| {_dataset_name(dataset)} | "
            f"{_fmt(block['path_soft_minus_hard_macro_f1'])} | "
            f"{_fmt(block['fusion_minus_flat_macro_f1'])} | "
            f"{_fmt(block['oracle_minus_fusion_macro_f1'])} |"
        )
    lines.append("")
    return lines


def _per_class_table(inv: dict) -> list[str]:
    lines = [
        "## Table 4. Probability Fusion per-class F1",
        "",
        "| Dataset | Non-malignant | Melanoma | BCC | SCC |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in ("internal_isic", "external_hiba"):
        block = inv[dataset]["fusion_per_class_f1"]
        lines.append(
            "| " + _dataset_name(dataset) + " | "
            + " | ".join(_fmt(block[c]) for c in CLASSES) + " |"
        )
    lines.append("")
    return lines


def _bootstrap_table(inv: dict) -> list[str]:
    lines = [
        "## Table 5. Seed-level paired-bootstrap uncertainty for Fusion − Hard macro-F1",
        "",
        "| Dataset | Seed | Point difference | 95% CI | Resampling unit | Replicates |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for dataset in ("internal_isic", "external_hiba"):
        for seed in SEEDS:
            stat = inv[dataset]["seeds"][seed]["primary_statistics"]
            lines.append(
                f"| {_dataset_name(dataset)} | {seed} | {stat['point_difference']:.4f} | "
                f"{_fmt_ci(stat['confidence_interval_95'])} | {stat['bootstrap_unit']} | "
                f"{stat['bootstrap_replicates']} |"
            )
    lines += [
        "",
        "ISIC uses the frozen paired image bootstrap. HIBA uses the frozen patient-cluster bootstrap.",
        "",
    ]
    return lines


def _rescue_table(inv: dict) -> list[str]:
    lines = [
        "## Table 6. Routing-rescue accounting for Probability Fusion versus Hard",
        "",
        "| Dataset | Seed | Malignant→NM failures | Rescued | Harmed | NM→malignant failures | Corrections | New errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("internal_isic", "external_hiba"):
        for seed in SEEDS:
            rescue = inv[dataset]["seeds"][seed]["rescue"]
            mal = rescue["malignant_routed_nm"]
            nm = rescue["nm_routed_malignant"]
            mal_total = mal.get("count", mal.get("total"))
            mal_rescued = mal.get("rescued", mal.get("fusion_correct"))
            mal_harmed = mal.get("harmed", 0)
            nm_total = nm.get("count", nm.get("total"))
            nm_corr = nm.get("corrections", nm.get("rescued", nm.get("fusion_correct")))
            nm_new = nm.get("new_errors", nm.get("harmed", nm.get("made_worse_correct_to_wrong_count", 0)))
            lines.append(
                f"| {_dataset_name(dataset)} | {seed} | {mal_total} | {mal_rescued} | {mal_harmed} | "
                f"{nm_total} | {nm_corr} | {nm_new} |"
            )
    lines += [
        "",
        "Counts are copied from the frozen rescue-analysis artifacts. They are not re-derived here.",
        "",
    ]
    return lines


def _narrative(inv: dict) -> list[str]:
    ii = inv["internal_isic"]
    eh = inv["external_hiba"]
    ii_primary = ii["primary_contrast_macro_f1"]
    eh_primary = eh["primary_contrast_macro_f1"]
    ii_oracle = ii["oracle_minus_fusion_macro_f1"]
    eh_oracle = eh["oracle_minus_fusion_macro_f1"]
    hiba_scc = eh["fusion_per_class_f1"]["scc"]

    _require(all(ii_primary["individual_seed_values"][s] < 0 for s in SEEDS),
             "Internal primary contrast is not negative for every frozen seed")
    _require(all(eh_primary["individual_seed_values"][s] < 0 for s in SEEDS),
             "External primary contrast is not negative for every frozen seed")

    return [
        "## Manuscript-ready scientific synthesis",
        "",
        "Across the three frozen seeds, Probability Fusion did not improve the primary endpoint over Hard routing on either evaluation cohort. "
        f"On the ISIC internal test, Fusion − Hard macro-F1 was {_fmt(ii_primary)}, with a negative difference for every seed. "
        f"On the zero-shot HIBA cohort, the corresponding difference was {_fmt(eh_primary)}, again negative for every seed.",
        "",
        "The same direction was observed for the fixed Path-Soft construction. "
        f"Path-Soft − Hard macro-F1 was {_fmt(ii['path_soft_minus_hard_macro_f1'])} on ISIC and "
        f"{_fmt(eh['path_soft_minus_hard_macro_f1'])} on HIBA. "
        "Thus, neither propagating conditional probabilities through the hierarchy nor fitting the prespecified validation-only multinomial fusion layer removed the observed end-to-end degradation relative to Hard routing.",
        "",
        "The Oracle diagnostic remained substantially above Probability Fusion. "
        f"Oracle − Fusion macro-F1 was {_fmt(ii_oracle)} on ISIC and {_fmt(eh_oracle)} on HIBA. "
        "Because the downstream shared-model outputs are unchanged in the Oracle condition, this gap is consistent with routing decisions remaining an important source of lost end-to-end performance.",
        "",
        "External transfer exposed a severe minority-class weakness in the fixed fusion formulation. "
        f"On HIBA, Probability Fusion SCC F1 was {_fmt(hiba_scc)} across the three seeds. "
        "The seed-level rescue tables should therefore be reported alongside aggregate accuracy so that majority-class behavior is not mistaken for balanced four-class improvement.",
        "",
        "The HIBA evaluation is zero-shot: the fusion models were fitted on the frozen ISIC validation partition and transferred unchanged to HIBA. "
        "HIBA uncertainty is based on the prespecified patient-cluster bootstrap, while the ISIC internal analysis uses the frozen paired image bootstrap.",
        "",
        "These results support a narrow conclusion: under this shared-backbone, hard-routing experimental setup, the two prespecified probability-propagation alternatives did not recover the primary macro-F1 loss associated with the hierarchy. "
        "They do not establish that hierarchical classification is inherently inferior, nor do they evaluate tuned soft-routing variants, dedicated branch models, or alternative calibration cohorts.",
        "",
    ]


def main():
    _require(INPUT.is_file(), f"Missing validated COM-06A inventory: {INPUT}")
    inv = _read(INPUT)
    _validate(inv)

    lines = [
        "# COM-06B Publication Synthesis",
        "",
        "Status: **PASS**",
        "",
        "Source: validated com06a_evidence_inventory.json only.",
        "No scientific metric, confidence interval, prediction, or rescue count is recomputed by this script.",
        "",
    ]
    lines += _system_table(inv)
    lines += _contrast_table(inv)
    lines += _secondary_table(inv)
    lines += _per_class_table(inv)
    lines += _bootstrap_table(inv)
    lines += _rescue_table(inv)
    lines += _narrative(inv)

    OUTPUT.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("COM-06B: PASS")
    print(f"OUTPUT: {OUTPUT}")
    print(f"ISIC primary contrast: {_fmt(inv['internal_isic']['primary_contrast_macro_f1'])}")
    print(f"HIBA primary contrast: {_fmt(inv['external_hiba']['primary_contrast_macro_f1'])}")


if __name__ == "__main__":
    main()
