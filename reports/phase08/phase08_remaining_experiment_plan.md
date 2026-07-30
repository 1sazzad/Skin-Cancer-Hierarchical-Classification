# Phase 08 — Remaining Experiment Plan

## Governing sequence

The sequence is gate-driven. Phase 09 may stop Stage 3 model work. Phase 10
must establish standalone feasibility before Phase 11. Phase 11 must freeze a
validation-selected final design before external evaluation. Locked Phase
05–07 inference is never rerun.

## Phase 09 — Early-Stage Melanoma Benchmark audit

### E09-01: acquisition, licensing, integrity, and label-semantics audit

- **Research question:** Can the authoritative EMB source be legally acquired
  and can its images/metadata support a reproducible severity task?
- **Hypothesis:** At least one direct, documented severity target will have
  adequate integrity and support without inferred labels.
- **Dataset / labels / split unit:** EMB; raw T-category, Breslow thickness and
  relevant provenance fields; patient first, lesion fallback, image only if
  neither stronger identifier exists.
- **Model / baseline / loss:** None / dataset-card expectations / none.
- **Primary metric:** Auditable count of eligible uniquely grouped melanoma
  cases per candidate class.
- **Secondary metrics:** Missingness, ambiguity, duplicates, cross-group
  conflicts, patient/lesion overlap risk, license status, checksum coverage.
- **Statistical analysis:** Counts, proportions, distributions, and
  deterministic integrity checks; no model hypothesis test.
- **Acceptance gate:** Direct label semantics, lawful use, reproducible mapping,
  leakage-safe grouping, and prospectively approved minimum class support.
- **Expected artifact:** Dataset card, license record, checksum/manifest,
  mapping decision, exclusions, group-safe split protocol and audit report.
- **GPU:** No.
- **Manuscript claim:** A named severity target and cohort were defined
  reproducibly; or feasibility failed for documented reasons.
- **Stopping rule:** Stop before download if license/access is unacceptable;
  stop before modelling if neither T-category nor Breslow grouping passes.

T-category and Breslow-thickness groups are not automatically interchangeable.
No Stage 3 class boundary is frozen or invented in Phase 08. The audited target
may be categorical, ordinal, or another scientifically justified formulation.
Boundaries, missingness, censoring, ulceration, exclusions, minimum support,
and split ratios remain unresolved until authoritative documentation and the
metadata audit can justify them.

## Phase 10 — Standalone Stage 3 baseline and feasibility gate

### E10-01: standalone melanoma-severity baseline

- **Research question:** Does dermoscopic imagery contain sufficient signal for
  the audited severity target under leakage-safe evaluation?
- **Hypothesis:** A validation-selected lightweight baseline will exceed the
  preregistered naive baseline by a practically meaningful macro-F1 margin
  without collapsing a class.
- **Dataset / labels / split unit:** Eligible EMB subset and Phase 09 target;
  frozen patient/lesion grouping.
- **Model / baseline / loss:** EfficientNet-B0 standalone head / majority and
  class-prior baselines / cross-entropy, with one preregistered imbalance-aware
  variant only if training counts justify it.
- **Primary metric:** Macro-F1.
- **Secondary metrics:** Balanced accuracy, per-class precision/recall/F1,
  confusion matrix, PR-AUC where defined, calibration, class support.
- **Statistical analysis:** Frozen-test confidence intervals and multi-seed
  summary; exact method and seed count frozen before training.
- **Acceptance gate:** Beats the preregistered naive baseline and minimum
  class-recall/support criteria on validation, then yields interpretable frozen
  test evidence. Exact thresholds remain protocol-pending.
- **Expected artifact:** Configs, split manifest, selected checkpoint reference,
  predictions, metrics, uncertainty output, feasibility decision.
- **GPU:** Azure Tesla T4.
- **Manuscript claim:** Stage 3 is feasible on the audited cohort, or it failed
  the prospective feasibility criterion.
- **Stopping rule:** Stop if Phase 09 fails; stop architecture escalation if the
  baseline fails the frozen gate.

## Phase 11 — Three-task lightweight framework

### Design decision

One candidate is **one shared lightweight encoder with three
task-specific heads and explicit missing-label masks**, with conditional
activation of Stage 2 and Stage 3 heads at inference. Stage 1 uses ISIC labels;
Stage 2 uses eligible malignant ISIC labels; Stage 3 uses eligible EMB melanoma
labels. Dataset/task masks prevent absent labels from contributing to loss.

Full sharing is not guaranteed before feasibility evidence. Full and partial
sharing are candidate designs that test the parameter-efficiency claim and
negative-transfer risk. The candidate set and validation-only selection rule
must be frozen after Stage 3 feasibility and before training; the final choice
must not be chosen to match proposal wording or test results.

### E11-01: full-sharing versus partial-sharing ablation

- **Research question:** Which parameter-efficient sharing pattern preserves
  task performance under partial labels?
- **Hypothesis:** Full sharing reduces stored parameters, while partial sharing
  may reduce negative transfer.
- **Dataset / labels / split unit:** Frozen ISIC 2019 and EMB development
  splits; task labels defined above; original grouping preserved within each
  dataset and overlap audited across datasets where possible.
- **Model / baseline / loss:** Shared EfficientNet-B0 encoder with three heads
  versus a prospectively specified partial-sharing variant / standalone
  validation references / sum of masked per-task losses with preregistered
  weights and sampling.
- **Primary metric:** Prespecified normalized aggregate of validation macro-F1
  across the three tasks; exact aggregation frozen before training.
- **Secondary metrics:** Per-task macro-F1, balanced accuracy, recall,
  calibration, gradient/task imbalance, active parameters and convergence.
- **Statistical analysis:** Multi-seed paired validation summaries; final
  frozen-test uncertainty deferred to Phase 14.
- **Acceptance gate:** No task exceeds the prospectively defined unacceptable
  degradation and efficiency improves on a declared parameter/size measure.
- **Expected artifact:** Architecture specification, masks, configs, logs,
  validation predictions, ablation table, selection decision.
- **GPU:** Azure Tesla T4.
- **Manuscript claim:** A particular sharing design was selected under a
  prospective multi-task criterion.
- **Stopping rule:** Stop a candidate for instability, label leakage, or
  predeclared validation futility; do not inspect locked Phase 05/06C test data.

### E11-02: frozen final three-stage evaluation

- **Research question:** What is the stage-level and end-to-end performance of
  the validation-selected three-task framework?
- **Hypothesis:** The selected framework produces usable task outputs with a
  measurable efficiency/performance trade-off; superiority is not assumed.
- **Dataset / labels / split unit:** Frozen ISIC and EMB test splits; the
  relevant frozen patient/lesion unit.
- **Model / baseline / loss:** Frozen selected E11-01 model / no new selection /
  training loss already frozen.
- **Primary metric:** Predeclared task macro-F1 plus a separately defined
  end-to-end routing endpoint; no invalid averaging across incompatible
  cohorts.
- **Secondary metrics:** Balanced accuracy, class metrics, calibration,
  routing reach/block rates, confusion matrices.
- **Statistical analysis:** Frozen multi-seed or paired confidence procedure
  specified before test access.
- **Acceptance gate:** Technical validity and complete reporting; no
  performance-based rerun.
- **Expected artifact:** Locked predictions, metrics, manifests, model
  provenance, and evaluation report.
- **GPU:** Azure Tesla T4.
- **Manuscript claim:** Exact bounded performance of the final framework.
- **Stopping rule:** One valid frozen evaluation per protocol; technical retry
  only if no valid metrics were produced and failure is documented.

## Phase 12 — Matched comparisons

### E12-01: shared framework versus separate task-specific models

- **Research question:** What performance/efficiency trade-off does sharing
  produce relative to separately trained Stage 1, Stage 2, and Stage 3 models?
- **Hypothesis:** Sharing reduces stored parameters/model size with no
  unacceptable task-level degradation.
- **Dataset / labels / split unit:** Same frozen ISIC and EMB splits and labels
  as E11; unchanged grouping.
- **Model / baseline / loss:** Frozen E11 model / locked Stage 1/2 comparators
  plus frozen E10 Stage 3 model / each model’s preregistered selected loss.
- **Primary metric:** Per-task macro-F1 differences; parameter and model-size
  differences are co-primary efficiency endpoints.
- **Secondary metrics:** Balanced accuracy, class recall, calibration,
  FLOPs/MACs, latency, peak memory, conditional head-use rates.
- **Statistical analysis:** Paired prediction intervals where populations and
  labels match; multi-seed summaries; no cross-cohort pooled significance.
- **Acceptance gate:** Complete matched evidence; claim direction follows
  results.
- **Expected artifact:** Comparison table, paired outputs, efficiency ledger.
- **GPU:** Azure Tesla T4 for new models and matched profiling; locked stored
  predictions reused where valid.
- **Manuscript claim:** Measured shared-versus-separate trade-off.
- **Stopping rule:** No retraining in response to test results.

### E12-02: relationship to locked flat and two-stage evidence

- **Research question:** How does the new framework relate to the existing
  internal flat and two-stage comparators without violating their locks?
- **Hypothesis:** Not prespecified; this is a conservative evidence synthesis.
- **Dataset / labels / split unit:** Existing locked ISIC population plus
  separately reported EMB Stage 3 population.
- **Model / baseline / loss:** Frozen new model / stored Phase 05 hierarchy and
  Phase 06C flat outputs / not applicable.
- **Primary metric:** Like-for-like ISIC four-class macro-F1; Stage 3 reported
  separately.
- **Secondary metrics:** Paired correctness, class metrics, routing, efficiency.
- **Statistical analysis:** Reuse locked Phase 07 result unchanged; new paired
  analysis only where a prospectively frozen new output permits it.
- **Acceptance gate:** No rerun or post-test selection; denominators explicit.
- **Expected artifact:** Provenance-linked synthesis table.
- **GPU:** No for locked evidence; yes only for authorized new-model output.
- **Manuscript claim:** Bounded comparison with clear cohort/task separation.
- **Stopping rule:** Stop any comparison that mixes incompatible labels,
  populations, or selection histories.

## Phase 13 — External evaluation and XAI

### E13-01: frozen approved external-dataset evaluation

- **Research question:** How do eligible frozen outputs change on an approved
  independently collected external cohort?
- **Hypothesis:** Performance will decrease under domain shift.
- **Dataset / labels / split unit:** Candidate HIBA only if compatibility,
  licence, modality, labels, identifiers, and support pass a prospective audit;
  map only compatible diagnosis semantics and use patient or lesion grouping
  for uncertainty/overlap controls.
- **Model / baseline / loss:** Frozen final model and eligible locked
  comparators / internal performance / no training loss.
- **Primary metric:** Macro-F1 on the prospectively defined evaluable external
  label set.
- **Secondary metrics:** Balanced accuracy, per-class recall/precision/F1,
  calibration without external refitting, coverage, prevalence and domain-shift
  descriptors.
- **Statistical analysis:** Confidence intervals and paired tests only for
  identical external samples; internal-versus-external changes are descriptive
  unless a valid method is preregistered.
- **Acceptance gate:** License, integrity, semantic mapping, overlap audit, and
  frozen model identity pass before inference.
- **Expected artifact:** External manifest, mapping, predictions, metrics,
  uncertainty, domain-shift and limitations report.
- **GPU:** Azure Tesla T4.
- **Manuscript claim:** Performance on the named approved cohort; not broad clinical
  generalisation.
- **Stopping rule:** Stop incompatible classes rather than forcing mappings;
  never tune using the external evaluation cohort.

### E13-02: preregistered Grad-CAM/XAI analysis

- **Research question:** What image regions are associated with selected model
  outputs across correct, error, class, confidence, and routing strata?
- **Hypothesis:** Saliency patterns will vary across outcome strata; clinical
  correctness is not inferred.
- **Dataset / labels / split unit:** Frozen internal and eligible HIBA
  evaluation samples; no change to split.
- **Model / baseline / loss:** Frozen final and selected comparator models /
  deterministic sanity controls where feasible / none.
- **Primary metric:** Completion of preregistered stratum coverage and
  faithfulness/sanity checks chosen before map generation.
- **Secondary metrics:** Localization descriptors if annotations exist,
  cross-model similarity, qualitative blinded review with a frozen rubric.
- **Statistical analysis:** Descriptive summaries and uncertainty where the
  selected quantitative XAI metric permits it.
- **Acceptance gate:** Case-selection seed/rules, layer choice, preprocessing,
  and review rubric frozen before images are viewed.
- **Expected artifact:** Selection manifest, maps, quantitative summaries,
  representative panel, captions, limitations.
- **GPU:** Azure Tesla T4.
- **Manuscript claim:** Descriptive model-attention patterns under specified
  methods; never proof of clinical reasoning.
- **Stopping rule:** Do not replace unattractive or contradictory cases; report
  failed sanity checks.

MRA-MIDAS is optional only after the approved primary external evaluation and
all mandatory experiments complete.
If activated, it requires its own pre-result compatibility protocol.

## Phase 14 — Three-stage statistics, routing, and efficiency

### E14-01: final integrated analysis

- **Research question:** What uncertainty, routing loss, and resource trade-offs
  characterize the frozen final framework?
- **Hypothesis:** Conditional execution changes workload and error distribution;
  the direction of performance and latency differences is not assumed.
- **Dataset / labels / split unit:** All frozen eligible internal/external
  outputs, analyzed separately by cohort and task.
- **Model / baseline / loss:** Frozen final, separate models, locked flat and
  two-stage comparators / same / none.
- **Primary metric:** Paired macro-F1 differences where valid; complete
  stage-wise routing decomposition; matched latency.
- **Secondary metrics:** Confidence intervals, McNemar where applicable,
  calibration, parameters, serialized size, FLOPs/MACs, throughput, peak GPU
  and CPU memory, conditional-compute rates.
- **Statistical analysis:** Preregistered paired bootstrap/quantiles, multiplicity
  handling for confirmatory contrasts, explicit denominators and seed summaries.
- **Acceptance gate:** Pairing, provenance, environment, warm-up, repetition,
  synchronization, and completeness audits pass.
- **Expected artifact:** Locked statistical JSON/CSV, routing tables, profiler
  outputs, figures, claims lock.
- **GPU:** Azure Tesla T4 for matched profiling; stored-output statistics local.
- **Manuscript claim:** Exact measured differences and uncertainty, including
  null or adverse results.
- **Stopping rule:** No performance rerun; failed comparability yields
  “unavailable,” not an estimate.

## Phase 15 — ICCIT assembly and submission checks

### E15-01: manuscript evidence and compliance audit

- **Research question:** Does every manuscript claim trace to eligible evidence
  and satisfy ICCIT and ethical reporting requirements?
- **Hypothesis:** All retained claims can be traced without scope inflation.
- **Dataset / labels / split unit:** No new data / not applicable / not
  applicable.
- **Model / baseline / loss:** None / none / none.
- **Primary metric:** Zero unsupported or untraceable claims.
- **Secondary metrics:** Table/figure provenance, template/page compliance,
  citation completeness, anonymization, author approvals, artifact availability.
- **Statistical analysis:** Recalculation prohibition; consistency checks only.
- **Acceptance gate:** Claims-lock review, negative-result inclusion,
  reproducibility checklist, and human approval all pass.
- **Expected artifact:** Manuscript, evidence ledger, submission checklist, final
  artifact index.
- **GPU:** No.
- **Manuscript claim:** None beyond already locked evidence.
- **Stopping rule:** Do not submit while any central claim lacks evidence or any
  mandatory compliance item is unresolved.
