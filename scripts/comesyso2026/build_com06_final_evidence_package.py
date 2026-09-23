"""COM-06C final publication evidence package.

Consumes only the validated COM-06A inventory and the COM-06B synthesis.
No training, inference, fitting, bootstrapping, or metric recomputation occurs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPORT_ROOT = Path("reports/extensions/comesyso2026_probability_fusion")
INVENTORY = REPORT_ROOT / "com06a_evidence_inventory.json"
SYNTHESIS = REPORT_ROOT / "COM06B_PUBLICATION_SYNTHESIS.md"
OUTPUT = REPORT_ROOT / "COM06C_FINAL_EVIDENCE_PACKAGE.md"

SEEDS = ("42", "123", "2026")


def _read_json(path: Path):
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


def _fmt(item: dict) -> str:
    return f"{item['mean']:.4f} ± {item['sample_standard_deviation']:.4f}"


def _seed_values(item: dict) -> str:
    v = item["individual_seed_values"]
    return f"42={v['42']:.4f}, 123={v['123']:.4f}, 2026={v['2026']:.4f}"


def main():
    _require(INVENTORY.is_file(), f"Missing COM-06A inventory: {INVENTORY}")
    _require(SYNTHESIS.is_file(), f"Missing COM-06B synthesis: {SYNTHESIS}")

    inv = _read_json(INVENTORY)
    _require(inv.get("phase") == "COM-06A", "Unexpected inventory phase")
    _require(inv.get("status") == "PASS", "COM-06A must be PASS")
    _require(inv.get("scientific_recomputation") is False, "COM-06A must remain read-only")

    internal = inv["internal_isic"]
    external = inv["external_hiba"]

    i_primary = internal["primary_contrast_macro_f1"]
    h_primary = external["primary_contrast_macro_f1"]
    i_path = internal["path_soft_minus_hard_macro_f1"]
    h_path = external["path_soft_minus_hard_macro_f1"]
    i_flat = internal["fusion_minus_flat_macro_f1"]
    h_flat = external["fusion_minus_flat_macro_f1"]
    i_oracle = internal["oracle_minus_fusion_macro_f1"]
    h_oracle = external["oracle_minus_fusion_macro_f1"]

    _require(all(i_primary["individual_seed_values"][s] < 0 for s in SEEDS),
             "Internal Fusion-Hard direction drift")
    _require(all(h_primary["individual_seed_values"][s] < 0 for s in SEEDS),
             "External Fusion-Hard direction drift")

    lines = [
        "# COM-06C Final Publication Evidence Package",
        "",
        "Status: **PASS**",
        "",
        "This package is a publication-facing synthesis of frozen COM-03/04/05 evidence.",
        "It performs no scientific recomputation.",
        "",
        "## 1. Frozen study identity",
        "",
        "- Seeds: 42, 123, 2026.",
        "- Primary endpoint: four-class macro-F1.",
        "- Primary comparison: Probability Fusion − Hard routing.",
        "- Systems: Flat, Hard, Path-Soft, Probability Fusion, Oracle.",
        "- Internal evaluation: frozen ISIC 2019 internal test.",
        "- External evaluation: frozen HIBA cohort, zero-shot transfer.",
        "- HIBA uncertainty: patient-cluster bootstrap.",
        "- Oracle is diagnostic only and is not a deployable method.",
        "",
        "## 2. Core publication numbers",
        "",
        "| Evidence | ISIC internal test | HIBA zero-shot |",
        "|---|---:|---:|",
        f"| Flat macro-F1 | {_fmt(internal['systems_macro_f1']['flat'])} | {_fmt(external['systems_macro_f1']['flat'])} |",
        f"| Hard macro-F1 | {_fmt(internal['systems_macro_f1']['hard'])} | {_fmt(external['systems_macro_f1']['hard'])} |",
        f"| Path-Soft macro-F1 | {_fmt(internal['systems_macro_f1']['path_soft'])} | {_fmt(external['systems_macro_f1']['path_soft'])} |",
        f"| Probability Fusion macro-F1 | {_fmt(internal['systems_macro_f1']['fusion'])} | {_fmt(external['systems_macro_f1']['fusion'])} |",
        f"| Oracle macro-F1 | {_fmt(internal['systems_macro_f1']['oracle'])} | {_fmt(external['systems_macro_f1']['oracle'])} |",
        f"| Fusion − Hard | {_fmt(i_primary)} | {_fmt(h_primary)} |",
        f"| Path-Soft − Hard | {_fmt(i_path)} | {_fmt(h_path)} |",
        f"| Fusion − Flat | {_fmt(i_flat)} | {_fmt(h_flat)} |",
        f"| Oracle − Fusion | {_fmt(i_oracle)} | {_fmt(h_oracle)} |",
        "",
        "Values are mean ± sample SD across the three frozen seeds; seed-image pairs are not pooled.",
        "",
        "## 3. Seed-level primary result",
        "",
        f"- ISIC Fusion − Hard: {_seed_values(i_primary)}.",
        f"- HIBA Fusion − Hard: {_seed_values(h_primary)}.",
        "- The primary macro-F1 contrast is negative for every frozen seed on both cohorts.",
        "",
        "## 4. Minority-class evidence",
        "",
        "| Dataset | Fusion Non-malignant F1 | Fusion Melanoma F1 | Fusion BCC F1 | Fusion SCC F1 |",
        "|---|---:|---:|---:|---:|",
        f"| ISIC internal test | {_fmt(internal['fusion_per_class_f1']['non_malignant'])} | "
        f"{_fmt(internal['fusion_per_class_f1']['melanoma'])} | "
        f"{_fmt(internal['fusion_per_class_f1']['bcc'])} | "
        f"{_fmt(internal['fusion_per_class_f1']['scc'])} |",
        f"| HIBA zero-shot | {_fmt(external['fusion_per_class_f1']['non_malignant'])} | "
        f"{_fmt(external['fusion_per_class_f1']['melanoma'])} | "
        f"{_fmt(external['fusion_per_class_f1']['bcc'])} | "
        f"{_fmt(external['fusion_per_class_f1']['scc'])} |",
        "",
        "## 5. Claims supported by the frozen evidence",
        "",
        "1. Under the frozen shared-backbone hierarchy, the prespecified Probability Fusion model did not improve four-class macro-F1 over Hard routing on the internal ISIC test or the zero-shot HIBA cohort.",
        "2. The fixed Path-Soft probability product also did not improve macro-F1 over Hard routing on either cohort.",
        "3. The Oracle diagnostic remained materially above Probability Fusion, supporting routing decisions as an important source of lost end-to-end performance in this setup.",
        "4. External HIBA transfer exposed substantial minority-class weakness in Probability Fusion; aggregate accuracy must therefore not be used as a substitute for balanced four-class evaluation.",
        "5. The HIBA result is a true zero-shot transfer of the ISIC-validation-fitted fusion models; no HIBA fitting or tuning was performed.",
        "",
        "## 6. Claims NOT supported",
        "",
        "- Do not claim that hierarchical classification is inherently inferior to flat classification.",
        "- Do not claim that all soft-routing or probability-fusion methods fail.",
        "- Do not claim external calibration, because the fusion model was fitted on the ISIC validation partition rather than an independent calibration cohort.",
        "- Do not present Oracle as a practical deployment method.",
        "- Do not present accuracy gains, if any, as evidence of balanced four-class improvement when macro-F1 and minority-class F1 decline.",
        "- Do not pool the three seeds into one enlarged sample.",
        "",
        "## 7. Recommended manuscript evidence layout",
        "",
        "**Main Table 1 — System comparison.** Report Flat, Hard, Path-Soft, Probability Fusion, and Oracle macro-F1 for ISIC and HIBA as mean ± sample SD.",
        "",
        "**Main Table 2 — Primary contrast and uncertainty.** Report Fusion − Hard by seed together with the frozen 95% bootstrap intervals. State explicitly that ISIC uses paired image bootstrap and HIBA uses patient-cluster bootstrap.",
        "",
        "**Main Table 3 — Per-class F1.** At minimum show Probability Fusion and Hard per-class F1 so the minority-class behavior is visible.",
        "",
        "**Figure 1 — Method diagram.** Show shared model outputs P(Task1 malignant) and P(Task2 MEL/BCC/SCC), the fixed Path-Soft product path, and the validation-only multinomial logistic Probability Fusion path. Mark HIBA as zero-shot and unchanged.",
        "",
        "**Figure 2 — Primary macro-F1 comparison.** Plot the five systems for ISIC and HIBA, using the frozen three-seed summaries. Oracle should be visually labeled diagnostic.",
        "",
        "**Figure 3 — Routing-rescue diagnostic.** Visualize, by seed, Hard-wrong/Fusion-correct versus Hard-correct/Fusion-wrong counts and malignant→NM routing failures.",
        "",
        "## 8. Manuscript-ready interpretation",
        "",
        f"Probability Fusion reduced macro-F1 relative to Hard routing by {_fmt(i_primary)} on the frozen ISIC internal test and {_fmt(h_primary)} on the zero-shot HIBA cohort. "
        "The direction was consistent across all three seeds on both datasets. "
        f"Path-Soft likewise remained below Hard by {_fmt(i_path)} on ISIC and {_fmt(h_path)} on HIBA. "
        f"At the same time, Oracle exceeded Probability Fusion by {_fmt(i_oracle)} on ISIC and {_fmt(h_oracle)} on HIBA, indicating that the available downstream predictions retained information that was not recovered by the tested routing alternatives. "
        "The result therefore narrows the conclusion: within this frozen shared-backbone hierarchy, neither the deterministic probability-product formulation nor the prespecified validation-trained multinomial fusion layer recovered the macro-F1 degradation associated with routing.",
        "",
        "## 9. Limitations to retain",
        "",
        "- Three frozen seeds only.",
        "- Shared-backbone hierarchy; dedicated branch models were not tested here.",
        "- Hard, fixed Path-Soft, and one prespecified multinomial fusion formulation only.",
        "- ISIC validation was used for checkpoint selection and fusion fitting; it is not an independent calibration cohort.",
        "- No HIBA adaptation or recalibration.",
        "- External cohort is the frozen historical 1,232-image HIBA cohort with patient-cluster bootstrap.",
        "- The study diagnoses routing-associated degradation; it does not exhaust the broader design space of hierarchical classifiers.",
        "",
        "## 10. Provenance",
        "",
        f"- COM-06A inventory SHA256: {_sha256(INVENTORY)}",
        f"- COM-06B synthesis SHA256: {_sha256(SYNTHESIS)}",
    ]
    for key, value in inv["completion_hashes"].items():
        lines.append(f"- {key}: {value}")

    lines += [
        "",
        "COM-06 scientific synthesis is complete and ready for manuscript drafting.",
        "",
    ]

    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print("COM-06C: PASS")
    print(f"OUTPUT: {OUTPUT}")
    print(f"COM-06A SHA256: {_sha256(INVENTORY)}")
    print(f"COM-06B SHA256: {_sha256(SYNTHESIS)}")
    print("COM-06 SCIENTIFIC SYNTHESIS: COMPLETE")


if __name__ == "__main__":
    main()
