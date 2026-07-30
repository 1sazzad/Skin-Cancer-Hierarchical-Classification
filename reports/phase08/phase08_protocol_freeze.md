# Phase 08 — Protocol Freeze

## Status and scope

This document freezes only decisions supportable before new data or results
exist. `Unresolved` means that a later preregistered decision is required; it
does not authorize a guess.

## Frozen now

1. Phase 05, Phase 06C, and Phase 07 evidence is locked and remains unchanged.
   The authoritative artifacts include
   `reports/phase05/conditional_hierarchical_internal_evaluation.md`,
   `reports/phase06/phase06c_selected_flat_internal_test_result.md`,
   `reports/phase07/generated/statistical_analysis_results.json`, and
   `reports/phase07/generated/claims_lock.json`.
2. Phase 05 hierarchical internal-test inference must not be rerun.
3. Phase 06C flat internal-test inference must not be rerun.
4. The rejected Phase 06B focal candidate must not receive internal-test
   inference.
5. The original target remains a three-stage research framework. The locked
   Phase 05–07 system remains a two-stage comparator.
6. Completed evidence and future evidence must be presented separately in
   every report and manuscript section.
7. Stage 3 label feasibility and a standalone validation-selected baseline
   must pass their gates before shared-model training begins.
8. External datasets and compatibility rules must be selected before model
   performance on those datasets is seen. External data is evaluation-only
   unless a separately approved protocol explicitly assigns another role.
9. XAI cases must not be chosen because their maps look convincing.
   Representative-case rules—including performance strata, correct/error
   categories, class coverage, routing outcomes, and deterministic sampling—must
   be preregistered before map generation.
10. Patient-level separation is mandatory when patient identifiers permit it;
    otherwise lesion-level separation is mandatory when lesion identifiers
    permit it. Weaker grouping must be documented as a limitation.
11. Negative, null, non-significant, and gate-failing results must be reported
    honestly. A failed Stage 3 feasibility gate is a valid finding.
12. Grad-CAM or other saliency output is descriptive. It cannot prove clinical
    correctness, causal reasoning, or safety.
13. External evaluation can support bounded evidence on named datasets; it
    cannot establish broad clinical generalisation.

## Unresolved and requiring prospective decisions

- EMB license and permitted redistribution/storage;
- authoritative EMB source and integrity expectations;
- whether labels support T-category directly;
- whether Breslow grouping is an acceptable fallback and its boundaries;
- whether the justified Stage 3 formulation is categorical, ordinal, or another
  formulation; T-category and Breslow groups are not automatically
  interchangeable, and no class boundary is defined by this freeze;
- missingness, censoring, ulceration, and ambiguous-label handling;
- patient/lesion identifiers and split unit;
- minimum per-class support and exact Stage 3 feasibility threshold;
- the final three-task sharing architecture;
- loss weighting, task sampling, and missing-label masks;
- number of training seeds and uncertainty procedure;
- final external class mapping and evaluable population;
- external-candidate compatibility, licence, modality, labels, identifiers,
  support, and overlap controls; HIBA has no approved role until this audit;
- XAI method parameters and independent review rubric;
- matched batch, warm-up, repetition, synchronization, and memory protocol for
  Tesla T4 profiling;
- ICCIT page limit, template version, submission date, and authorship approval.

No new experiment may begin until its unresolved items are recorded in a
phase-specific protocol and approved. This freeze does not authorize dataset
download, Azure use, training, inference, or evaluation.
