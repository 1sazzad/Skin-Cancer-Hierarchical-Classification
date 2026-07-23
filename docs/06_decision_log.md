# Research Decision Log

## Purpose

This document records all material research, architecture, data, experiment, and scope decisions made during the project.

The purpose is to preserve the reasoning behind important choices so that future work remains consistent, reproducible, and understandable.

A decision must be recorded when it changes or affects:

* project scope;
* classification stages;
* dataset roles;
* label definitions;
* model architecture;
* preprocessing;
* data splitting;
* evaluation protocol;
* experiment naming;
* external evaluation;
* reproducibility;
* compute usage;
* final research claims.

---

# 1. Decision Status Values

Use the following status values:

```text
proposed
accepted
rejected
superseded
deferred
under_review
```

---

# 2. Decision Identifier Format

Each decision must use:

```text
D-XXX
```

Examples:

```text
D-001
D-002
D-003
```

Decision identifiers must never be reused.

---

# 3. Decision Record Template

Copy this template for every new material decision.

```markdown
## D-XXX â€” Decision title

**Date:** YYYY-MM-DD
**Status:** Proposed
**Phase:** Phase XX
**Owner:** Research lead

### Context

Describe the problem, uncertainty, or choice that required a decision.

### Options Considered

1. Option A
2. Option B
3. Option C

### Decision

State the selected decision clearly.

### Rationale

Explain why this option was selected.

### Supporting Evidence

List the evidence, literature, dataset findings, experiment results, or constraints supporting the decision.

### Expected Benefits

- Benefit 1
- Benefit 2

### Risks and Trade-offs

- Risk 1
- Trade-off 1

### Impacted Files or Components

- Path or component
- Path or component

### Impact on Existing Experiments

State whether existing experiments remain comparable, must be repeated, or become invalid.

### Additional Time or Compute

Describe any additional time, storage, or GPU requirements.

### Review Trigger

State the condition that would justify reviewing this decision.

### Final Outcome

Record the implemented outcome after the decision is executed.
```

---

# 4. Initial Project Decisions

## D-001 â€” Use a Dermoscopic Image-Only Research Scope

**Date:** 2026-07-22
**Status:** Accepted
**Phase:** Phase 00
**Owner:** Research lead

### Context

The initial research discussion considered combining dermoscopic images with multi-omic data.

The available project time, dataset complexity, compute constraints, and need for a coherent final-year research contribution made a combined image and multi-omic pipeline too broad.

### Options Considered

1. Dermoscopic-image classification only
2. Multi-omic classification only
3. Dermoscopic image and multi-omic fusion
4. Separate image and multi-omic experiments within the same project

### Decision

The project will use dermoscopic images as the primary data modality.

Multi-omic fusion is excluded from the current main scope.

### Rationale

A focused image-based project allows:

* stronger dataset auditing;
* more rigorous baseline comparison;
* hierarchical modelling;
* external evaluation;
* calibration;
* explainability;
* efficiency analysis;
* better reproducibility within the available time.

### Expected Benefits

* reduced scope complexity;
* clearer research contribution;
* lower implementation risk;
* more time for external evaluation;
* better experiment quality.

### Risks and Trade-offs

* the project will not contribute to multimodal or multi-omic fusion;
* future work may be required to extend the framework to additional modalities.

### Impacted Files or Components

* `docs/00_project_charter.md`
* `docs/01_scope_lock.md`
* `configs/project.yaml`
* future methodology and experiment files

### Impact on Existing Experiments

No existing experiments are affected because modelling has not started.

### Additional Time or Compute

No additional compute is required.

### Review Trigger

Review only if a reliable multi-omic dataset, sufficient time, and a clearly justified integration method become available after all mandatory image-based experiments are complete.

### Final Outcome

The dermoscopic-image-only scope is locked.

---

## D-002 â€” Use a Conditional Hierarchical Classification Framework

**Date:** 2026-07-22
**Status:** Accepted
**Phase:** Phase 00
**Owner:** Research lead

### Context

A flat multiclass classifier may hide clinically meaningful error patterns and class imbalance.

A hierarchical framework allows malignancy screening, malignant-type classification, and conditional severity analysis.

### Options Considered

1. One flat multiclass classifier
2. Independent binary classifiers
3. Conditional hierarchical classification
4. Multitask classification with shared features

### Decision

The main proposed method will be a conditional hierarchical framework.

### Hierarchy

#### Stage 1

```text
malignant
non_malignant
```

#### Stage 2

```text
melanoma
bcc
scc
```

#### Stage 3

```text
melanoma severity grouping
```

Stage 3 remains dependent on EMB label feasibility.

### Rationale

The hierarchy supports:

* clinically meaningful task decomposition;
* class-specific evaluation;
* explicit routing analysis;
* error-propagation measurement;
* comparison with a flat baseline;
* potential computational savings through conditional execution.

### Risks and Trade-offs

* Stage 1 false negatives may block malignant cases;
* later stages depend on earlier routing;
* end-to-end performance may be lower than independent stage performance;
* additional evaluation complexity is required.

### Impacted Files or Components

* `docs/00_project_charter.md`
* `docs/01_scope_lock.md`
* `docs/02_research_questions.md`
* future stage-specific configurations and models

### Impact on Existing Experiments

No experiments are affected.

### Additional Time or Compute

Additional training and evaluation runs are required for:

* flat baseline;
* Stage 1;
* Stage 2;
* oracle-gate evaluation;
* predicted-gate evaluation;
* end-to-end comparison.

### Review Trigger

Review if the hierarchy cannot be evaluated fairly or if Stage 1 routing causes unacceptable loss without providing useful analytical value.

### Final Outcome

The conditional hierarchy is accepted as the primary proposed framework.

---

## D-003 â€” Assign Fixed Dataset Responsibilities

**Date:** 2026-07-22
**Status:** Accepted
**Phase:** Phase 00
**Owner:** Research lead

### Context

Using datasets for multiple conflicting purposes may contaminate evaluation and weaken external-validity claims.

### Options Considered

1. Merge all datasets for training
2. Randomly distribute all datasets across train and test
3. Assign each dataset a fixed research responsibility
4. Use external datasets for iterative model tuning

### Decision

Dataset roles are fixed as follows:

| Dataset   | Role                                        |
| --------- | ------------------------------------------- |
| ISIC 2019 | Primary development and internal evaluation |
| EMB       | Stage 3 feasibility and severity modelling  |
| HIBA      | Mandatory independent external evaluation   |
| MRA-MIDAS | Optional second external evaluation         |

### Rationale

Fixed roles reduce:

* external-evaluation contamination;
* label-mapping confusion;
* uncontrolled dataset mixing;
* unclear claims about generalisation.

### Risks and Trade-offs

* external sample sizes may be limited after class mapping;
* some classes may not be compatible across datasets;
* negative external results may occur.

### Impacted Files or Components

* `docs/01_scope_lock.md`
* `docs/03_dataset_roles.md`
* `configs/dataset_registry.yaml`
* future manifests and label mappings

### Impact on Existing Experiments

No experiments are affected.

### Additional Time or Compute

Dataset-specific auditing and mapping scripts will be required.

### Review Trigger

Review if official documentation shows that one dataset cannot legally or scientifically fulfil its assigned role.

### Final Outcome

Dataset responsibilities are locked.

---

## D-004 â€” Keep HIBA as an Untouched External Evaluation Dataset

**Date:** 2026-07-22
**Status:** Accepted
**Phase:** Phase 00
**Owner:** Research lead

### Context

External evaluation is valid only when the external dataset does not influence model-development decisions.

### Options Considered

1. Train on HIBA
2. Tune thresholds using HIBA
3. Fine-tune on HIBA before reporting external results
4. Perform frozen zero-shot evaluation first

### Decision

The primary HIBA evaluation will be zero-shot using a fully frozen model and inference pipeline.

### Frozen Before HIBA Evaluation

* model architecture;
* checkpoint;
* preprocessing;
* image resolution;
* label mapping;
* threshold;
* calibration method;
* inference configuration.

### Rationale

This preserves the independence of the external evaluation.

### Risks and Trade-offs

* external performance may be substantially lower;
* the pipeline cannot be changed after seeing HIBA results without creating a separate adapted experiment.

### Impacted Files or Components

* `docs/01_scope_lock.md`
* `docs/03_dataset_roles.md`
* future external-evaluation configurations

### Impact on Existing Experiments

No experiments are affected.

### Additional Time or Compute

A separate post-external adaptation study may require additional compute if performed later.

### Review Trigger

This decision must not be reversed for the primary evaluation.

### Final Outcome

HIBA is reserved for frozen zero-shot external evaluation.

---

## D-005 â€” Make Stage 3 Feasibility-Dependent

**Date:** 2026-07-22
**Status:** Accepted
**Phase:** Phase 00
**Owner:** Research lead

### Context

Melanoma stage or severity classification requires clinically meaningful and sufficiently complete labels.

The EMB dataset may contain T-category or Breslow-thickness information, but its exact suitability has not yet been audited.

### Options Considered

1. Guarantee a Stage 3 classifier before inspecting labels
2. Create severity labels through assumptions
3. Perform a feasibility audit first
4. Remove Stage 3 immediately

### Decision

Stage 3 will proceed only after a formal EMB label-feasibility audit.

### Preferred Target

```text
t_category
```

### Fallback Target

```text
breslow_group
```

### Stop Condition

Stage 3 will not proceed if:

* labels are unclear;
* missingness is excessive;
* class counts are insufficient;
* leakage-safe splitting is impossible;
* target groups require unsupported inference.

### Rationale

A valid feasibility limitation is scientifically stronger than forcing an unreliable classification task.

### Risks and Trade-offs

* the final hierarchy may contain only Stage 1 and Stage 2;
* the project title and claims may require adjustment if Stage 3 is not feasible.

### Impacted Files or Components

* `docs/00_project_charter.md`
* `docs/01_scope_lock.md`
* `docs/02_research_questions.md`
* `docs/03_dataset_roles.md`
* future EMB audit files

### Impact on Existing Experiments

No experiments are affected.

### Additional Time or Compute

The initial EMB audit requires limited compute but careful metadata analysis.

### Review Trigger

Review after the EMB feasibility report is completed.

### Final Outcome

Stage 3 remains conditionally included.

---

## D-006 â€” Use Local Storage as the Permanent Source of Truth

**Date:** 2026-07-22
**Status:** Accepted
**Phase:** Phase 00
**Owner:** Research lead

### Context

Azure GPU resources may be temporary, costly, or deleted after use.

Important research artifacts must remain accessible and organised throughout the project.

### Options Considered

1. Keep all work on Azure
2. Keep all work locally
3. Use Azure only for computation and local storage as the master copy

### Decision

The permanent source of truth is:

```text
F:\Research\Final Year\Skin-Cancer-Hierarchical-Classification
```

Azure will be used only when GPU computation is needed.

### Mandatory Azure Return Artifacts

* code changes;
* configurations;
* checkpoints;
* metrics;
* predictions;
* logs;
* environment records;
* tables;
* figures;
* run notes.

### Rationale

This reduces the risk of losing experiments and keeps version-controlled work in one stable location.

### Risks and Trade-offs

* artifact transfer must be performed consistently;
* large checkpoints require local storage and backup capacity.

### Impacted Files or Components

* `docs/00_project_charter.md`
* `docs/04_reproducibility_protocol.md`
* future transfer and experiment-management scripts

### Impact on Existing Experiments

No experiments are affected.

### Additional Time or Compute

Some time is required for artifact transfer after GPU runs.

### Review Trigger

Review only if the permanent local storage system changes.

### Final Outcome

The local project directory is the permanent master copy.

---

## D-007 â€” Enforce Meaningful and Reusable Naming

**Date:** 2026-07-22
**Status:** Accepted
**Phase:** Phase 00
**Owner:** Entire project

### Context

The project will produce many configurations, checkpoints, predictions, figures, and reports that may be reused across multiple stages.

Ambiguous naming would make experiment comparison and reuse difficult.

### Options Considered

1. Short sequential names
2. Manual informal naming
3. Descriptive structured naming

### Decision

All important files and directories must use descriptive `snake_case` names.

Names should identify relevant components such as:

* phase or stage;
* dataset;
* model;
* experiment variant;
* seed;
* split;
* metric;
* artifact purpose.

### Accepted Examples

```text
stage01_isic2019_efficientnet_b0_cross_entropy.yaml
stage02_isic2019_convnext_tiny_weighted_loss_seed42_test_metrics.json
external_hiba_hierarchical_zero_shot_predictions.csv
```

### Rejected Examples

```text
test.py
new_model.pth
final2.csv
best_result.json
experiment_new
```

### Rationale

Meaningful names improve:

* reuse;
* navigation;
* debugging;
* reporting;
* automated processing;
* collaboration;
* long-term maintenance.

### Impacted Files or Components

All future project files and directories.

### Impact on Existing Experiments

No experiments are affected.

### Additional Time or Compute

No additional compute is required.

### Review Trigger

Review naming rules only if automation requires a stricter schema.

### Final Outcome

Meaningful reusable naming is mandatory.

---

## D-008 â€” Maintain a Clean Modular Project Architecture

**Date:** 2026-07-22
**Status:** Accepted
**Phase:** Phase 00
**Owner:** Entire project

### Context

Datasets, scripts, notebooks, logs, models, and reports can quickly make a research directory disorganised.

### Options Considered

1. Store files wherever convenient
2. Organise only before final submission
3. Enforce a clean architecture from the beginning

### Decision

The project will maintain clear separation between:

* configurations;
* datasets;
* documentation;
* experiments;
* checkpoints;
* notebooks;
* reports;
* reusable source code;
* utility scripts;
* tests.

### Root Directory Rule

The project root must remain minimal.

Datasets, checkpoints, downloaded archives, logs, screenshots, generated figures, and temporary files must not be placed directly in the root.

### Rationale

A clean architecture reduces:

* duplicate work;
* path confusion;
* accidental Git commits;
* broken imports;
* difficulty reproducing experiments.

### Impacted Files or Components

The entire repository structure.

### Impact on Existing Experiments

No experiments are affected.

### Additional Time or Compute

No additional compute is required.

### Review Trigger

Review when a new component cannot reasonably fit the current architecture.

### Final Outcome

Clean architecture and working-directory hygiene are permanent requirements.

---

## D-009 â€” Use Validation Data Only for Model Selection

**Date:** 2026-07-22
**Status:** Accepted
**Phase:** Phase 00
**Owner:** Research lead

### Context

Using internal test or external evaluation results for development decisions would bias the final performance estimate.

### Options Considered

1. Select models using test accuracy
2. Select models using external performance
3. Select models using internal validation data only

### Decision

All development decisions must use internal training and validation data only.

### Validation May Be Used For

* early stopping;
* checkpoint selection;
* hyperparameter comparison;
* threshold selection;
* calibration fitting;
* architecture comparison.

### Test and External Data Must Not Be Used For

* model selection;
* threshold selection;
* checkpoint selection;
* loss selection;
* augmentation selection;
* preprocessing selection.

### Rationale

This preserves the integrity of internal and external evaluation.

### Impacted Files or Components

* future training configurations;
* model-selection scripts;
* evaluation scripts;
* experiment registry.

### Impact on Existing Experiments

No experiments are affected.

### Additional Time or Compute

No additional compute is required.

### Review Trigger

This decision should not be reversed.

### Final Outcome

Validation-only model selection is mandatory.

---

## D-010 â€” Use Multiple Metrics Instead of Overall Accuracy Alone

**Date:** 2026-07-22
**Status:** Accepted
**Phase:** Phase 00
**Owner:** Research lead

### Context

Class imbalance can produce high overall accuracy while rare malignant classes perform poorly.

### Options Considered

1. Use accuracy as the primary metric
2. Use weighted F1-score only
3. Use imbalance-sensitive classwise metrics

### Decision

Primary evaluation will include:

* macro F1-score;
* balanced accuracy;
* per-class recall;
* per-class precision;
* confusion matrix.

Accuracy remains a secondary metric.

### Rationale

These metrics better reveal minority-class performance and clinically important errors.

### Impacted Files or Components

* `docs/02_research_questions.md`
* future evaluation configurations;
* metric-export scripts;
* result tables.

### Impact on Existing Experiments

No experiments are affected.

### Additional Time or Compute

Minimal additional evaluation time.

### Review Trigger

Review only if the task definition changes substantially.

### Final Outcome

Overall accuracy will not be used as the sole measure of success.

---

# 5. Future Decision Categories

Future decisions may include:

* final malignancy label mapping;
* final Stage 2 class mapping;
* EMB Stage 3 feasibility outcome;
* patient-level split strategy;
* selected baseline architecture;
* selected lightweight architecture;
* input resolution;
* augmentation policy;
* imbalance-aware loss;
* checkpoint-selection metric;
* calibration method;
* external label mapping;
* explainability method;
* deployment or compression experiment;
* optional MRA-MIDAS activation.

Each must receive a new decision identifier.

---

# 6. Decision Log Rules

1. Never delete an old decision.
2. Mark replaced decisions as `superseded`.
3. Link the new decision to the superseded decision.
4. Record rejected options when they were seriously considered.
5. Do not silently change an accepted experimental rule.
6. Record the effect on previous experiment comparability.
7. Use factual reasoning rather than rewriting history after seeing results.
8. Commit every material decision to Git.
9. Keep decision titles specific and meaningful.
10. Update the log before implementing a major scope change.

---

# 7. Current Decision Log Status

**Current phase:** Phase 00
**Last updated:** 2026-07-22
**Next decision identifier:** `D-011`

The initial project scope, architecture, data responsibilities, evaluation boundaries, reproducibility rules, and naming standards are now formally recorded.

---

## D-011 — Primary ISIC 2019 hierarchical label mapping

**Date:** 2026-07-23
**Status:** Accepted
**Phase:** Phase 01

### Decision

- MEL, BCC, and SCC are mapped to the Stage 1 malignant class.
- NV, BKL, DF, and VASC are mapped to the Stage 1 non_malignant class.
- MEL, BCC, and SCC form the Stage 2 melanoma, bcc, and scc classes.
- AK is excluded from the primary Stage 1 and Stage 2 tasks.
- UNK is excluded because the diagnosis is unknown.

### Rationale

AK is premalignant and does not fit cleanly into the locked malignant versus non_malignant primary task. Assigning it to either class would introduce an avoidable clinical and methodological ambiguity.

### Consequences

- Primary Stage 1 experiments use only rows where include_stage_1 equals 1.
- Primary Stage 2 experiments use only rows where include_stage_2 equals 1.
- AK may be evaluated later only through an explicitly declared sensitivity analysis.

---

## D-012 — Leakage-aware ISIC 2019 split policy

**Date:** 2026-07-23
**Status:** Accepted
**Phase:** Phase 01

### Decision

- Train, validation, and internal test partitions use a deterministic 70/15/15 split with seed 42.
- Images sharing a non-empty lesion_id must remain in the same partition.
- Images sharing an exact file SHA-256 must remain in the same partition.
- The transitive connected component of lesion-ID and exact-hash relations is treated as one indivisible split group.
- Components containing conflicting diagnoses or hierarchy labels are excluded from primary development.

### Identified exclusion

One four-image component joining BCN_0000237 and BCN_0003560 contains byte-identical images labelled both MEL and NV. All four images are excluded using the reason cross_diagnosis_exact_duplicate_component.

### Validation

- Split-group overlap count: 0
- Lesion-ID overlap count: 0
- Exact-hash overlap count: 0
- Leakage validation: passed

### Limitation

ISIC 2019 metadata does not provide patient_id. The split is lesion-aware and exact-duplicate-aware, but patient-independent separation cannot be guaranteed.
