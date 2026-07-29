# Phase 08 — Scope-Deviation and Evidence-Completeness Audit

## Audit basis

This audit uses the repository at synchronized commit
`cb23344258e600c594029116426e904d92738fcf`. It does not run training,
inference, evaluation, download data, access Azure, or alter locked Phase
05–07 evidence.

## Original and implemented objectives

The original objective in `docs/00_project_charter.md` and
`docs/01_scope_lock.md` is a lightweight conditional **three-stage** framework:

1. malignant versus non-malignant;
2. melanoma versus BCC versus SCC for routed malignant lesions;
3. melanoma T-category, or a defensible Breslow-thickness grouping if
   T-category is unavailable.

It also requires partially labelled multi-dataset learning, flat and
task-specific comparisons, imbalance handling, end-to-end routing analysis,
external evaluation, XAI, and complete efficiency evidence.

Through Phase 07, the implemented system is a locked **two-stage** conditional
pipeline using separate EfficientNet-B0 Stage 1 and Stage 2 checkpoints.
Phase 07 compares its four-class output with one flat EfficientNet-B0 model on
the same ISIC 2019 internal-test population. It is not the proposed final
three-stage system.

## Objective-by-objective audit

The machine-readable record is
`reports/phase08/generated/phase08_objective_evidence_matrix.csv`.

| Objective | Status | Exact evidence | Audit finding |
|---|---|---|---|
| Stage 1 malignancy | Complete | `reports/phase03/clean_baseline_internal_evaluation.md`; `reports/phase05/conditional_hierarchical_internal_evaluation.md` | Locked single-seed internal evidence exists. |
| Stage 2 malignant subtype | Complete | `reports/phase04/stage02_imbalance_aware_final_internal_evaluation.md`; `reports/phase05/conditional_hierarchical_internal_evaluation.md` | Locked internal evidence exists; SCC remains weak. |
| Stage 3 melanoma severity | Blocked | `configs/dataset_registry.yaml`; `docs/01_scope_lock.md` | EMB is not acquired; target semantics and feasibility are unresolved. |
| Shared/parameter-efficient model | Missing | `src/models/efficientnet_baseline.py`; `reports/phase07/generated/model_artifact_inventory.csv` | No shared three-task model has been implemented or evaluated. |
| Partially labelled learning | Missing | `configs/dataset_registry.yaml`; repository code search | Dataset roles exist, but no task-mask or masked-loss training implementation exists. |
| Flat comparison | Partial | `reports/phase07/generated/statistical_analysis_results.json` | The paired flat-versus-two-stage internal comparison is locked; the proposed three-stage comparison does not exist. |
| Separate task-specific comparison | Partial | `reports/phase03/clean_baseline_internal_evaluation.md`; `reports/phase04/stage02_imbalance_aware_final_internal_evaluation.md` | Stage 1/2 models exist, but no Stage 3 standalone baseline or matched three-task comparison exists. |
| Imbalance handling | Partial | `reports/phase04/stage02_imbalance_aware_final_internal_evaluation.md` | A locked Stage 2 sub-result exists, but Stage 3 and framework-wide evidence do not. |
| Error propagation | Partial | `reports/phase05/conditional_hierarchical_internal_evaluation.md`; `reports/phase07/generated/hierarchical_routing_decomposition.csv` | The locked analysis covers only the two-stage comparator, not Stage 3 routing. |
| External evaluation | Missing | `configs/dataset_registry.yaml` | HIBA is only a candidate until compatibility, licence, modality, labels, identifiers, and support are audited; no external evaluation exists. |
| XAI/Grad-CAM | Missing | `docs/01_scope_lock.md`; repository code search | No implementation, selection protocol, or result exists. |
| Efficiency | Partial | `reports/phase07/phase07_gate05a_efficiency_evidence_audit.md`; `reports/phase07/generated/efficiency_evidence_inventory.csv` | Parameters, size, recorded timing, and conditional-work proxies exist. FLOPs/MACs, peak memory, energy, and matched final-system profiling are unavailable. |

No audited objective is obsolete. Stage 3 is **blocked**, rather than merely
missing, because its scientific target cannot be chosen until the dataset and
labels pass the feasibility gate.

## Locked statistical conclusion

The locked Phase 07 result must be preserved exactly:

- flat macro-F1: `0.6192224685`;
- hierarchical macro-F1: `0.6053674006`;
- flat minus hierarchical: `+0.0138550680`;
- paired 95% CI: `[-0.0142546488, 0.0419633760]`;
- exact McNemar p: `0.8207415883`.

Therefore, no statistically distinguishable macro-F1 difference was
established on the locked internal split. The interval is not evidence of
equivalence or non-inferiority.

## Missing scientific work and claim risks

The remaining scientific gaps are an auditable EMB dataset and Stage 3 target;
a standalone Stage 3 feasibility baseline; a defensible three-task
parameter-efficient design; partially labelled training; matched separate-model
comparison; frozen HIBA evaluation; preregistered XAI; and complete routing,
statistical, FLOP, latency, model-size, and memory comparisons.

Claims are currently forbidden if they state or imply:

- completion of the proposed three-stage study;
- superiority of a shared model or hierarchy;
- successful Stage 3 severity estimation;
- external or broad clinical generalisation;
- clinical correctness demonstrated by Grad-CAM;
- lower FLOPs, memory, or latency from the current workload proxy;
- statistical significance, equivalence, non-inferiority, clinical readiness,
  fairness, or state-of-the-art performance.

Mislabeling the two-stage comparator as the proposed final system is the
highest scope-deviation risk. Selecting EMB/HIBA mappings, thresholds, XAI
cases, or final architectures after viewing favourable results would add
selection bias.

## ICCIT relevance and priority

| Planned evidence | ICCIT relevance | Priority |
|---|---|---|
| EMB integrity and severity-label audit | Establishes whether Stage 3 is scientifically valid | Must complete first |
| Standalone Stage 3 baseline | Tests feasibility before coupling tasks | Must complete |
| Three-task parameter-efficient framework | Tests the central proposed contribution | Must complete if Stage 3 passes |
| Separate-model and locked flat/two-stage comparison | Tests whether sharing/conditioning adds value | Must complete |
| Approved frozen external evaluation | Tests domain shift and bounds generalisation claims | Must complete; HIBA remains a candidate pending audit |
| Preregistered Grad-CAM/XAI | Adds descriptive failure analysis, not clinical proof | Required but secondary to predictive validity |
| Three-stage statistics/routing/efficiency | Quantifies uncertainty, propagation, and compute trade-offs | Must complete |
| MRA-MIDAS second external dataset | Adds breadth after mandatory evidence | Secondary |
| Calibration/subgroup extensions | Useful if labels/support permit and protocol is frozen | Secondary |

## Azure T4 requirement

Additional Azure Tesla T4 work is required **after** local protocol gates:
Stage 3 baseline training/evaluation; final three-task and task-specific
training; frozen external inference; XAI generation; and matched profiling.
Phase 09 acquisition/legal/label audit and protocol work begins locally and
must not be treated as authorization to download data. No Azure work is
authorized in Phase 08.

## Final audit decision

**Decision: material scope incompleteness; continue under Option A/full original
scope.** Phases 05–07 are valid, locked evidence for a two-stage comparator,
not completion of the original proposal. The ICCIT manuscript must not be
assembled as a completed three-stage contribution until the Phase 09
feasibility gate and the required later evidence are resolved. The next action
is to review and approve the Phase 09 acquisition, licensing, label-semantics,
leakage, and split protocol before any dataset acquisition or experiment.
