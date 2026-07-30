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

**Current phase:** Phase 05
**Last updated:** 2026-07-27
**Next decision identifier:** `D-017`

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

---

## D-013 â€” Lock Phase 02 preprocessing and manifest-driven loading policy

**Date:** 2026-07-24
**Status:** Accepted
**Phase:** Phase 02
**Owner:** Research lead

### Context

The project requires one reproducible preprocessing and loading policy shared by
Stage 1, Stage 2, and fair baseline comparisons. The frozen leakage-aware split
manifest already contains the image paths, partition assignments, hierarchy
labels, inclusion flags, split-group identifiers, and exact-image hashes.

### Decision

- Images are read directly from the untouched raw dataset at runtime.
- No duplicate resized image collection is created under `data/processed/`.
- The frozen split manifest
  `data/manifests/isic2019_train_val_test_split_seed42.csv` is the single
  loader source.
- Every selected row must satisfy `split_included == 1`, the requested split,
  and the relevant stage-inclusion flag.
- Stage 1 class indices are `non_malignant: 0` and `malignant: 1`.
- Stage 2 class indices are `melanoma: 0`, `bcc: 1`, and `scc: 2`.
- Model input size is 224 x 224 with ImageNet mean and standard-deviation
  normalization.
- Training uses moderate random resized cropping, horizontal and vertical
  flipping, limited rotation, and mild colour jitter.
- Validation and internal-test preprocessing use deterministic resize,
  centre-crop, tensor conversion, and normalization.
- DataLoader sampling and workers use explicit seeded generators and worker
  initialization.
- Full training remains blocked until the automated tests, real-data smoke
  test, and class-count reconciliation pass.

### Rationale

This policy preserves the frozen split, avoids undocumented preprocessing,
keeps raw data unchanged, minimizes storage duplication, and ensures that flat
and hierarchical experiments can use equivalent inputs.

### Risks and Trade-offs

- Runtime transformations use additional CPU time.
- ImageNet normalization may not be domain-optimal, but it provides a standard
  starting point for pretrained ImageNet backbones.
- Stochastic training augmentation prevents bitwise-identical samples while
  remaining procedurally reproducible under the recorded seed policy.

### Impacted Files or Components

- `src/data/`
- `src/utils/reproducibility.py`
- `tests/`
- baseline experiment configurations
- Phase 02 class-statistics reports

### Impact on Existing Experiments

No experiment is invalidated because model training has not started.

### Review Trigger

Review only through a new decision if validation-only evidence later supports a
different resolution, normalization, or augmentation policy.

### Final Outcome

The Phase 02 preprocessing, filtering, and dataloader policy is locked.

---

## D-014 â€” Prepare EfficientNet-B0 as the initial clean baseline backbone

**Date:** 2026-07-24
**Status:** Accepted
**Phase:** Phase 02
**Owner:** Research lead

### Context

A standard pretrained image-classification backbone is needed to verify the
training pipeline and provide a clean reference before imbalance-aware or
hierarchical innovations are introduced.

### Decision

- EfficientNet-B0 with ImageNet pretrained weights is prepared as the initial
  Stage 1 and Stage 2 baseline backbone.
- Stage 1 uses a two-output classification head.
- Stage 2 uses a three-output classification head.
- The clean baseline uses ordinary cross-entropy.
- Weighted sampling, class-weighted loss, and focal loss are disabled.
- This is a reference baseline, not the final proposed architecture.
- Full training is not permitted until Phase 02 tests and the real-data smoke
  test pass.

### Rationale

EfficientNet-B0 is compact enough for the available compute environment while
providing a well-established pretrained baseline for later fair comparisons.

### Risks and Trade-offs

- The architecture may not become the final lightweight model.
- Minority-class performance may be weak without imbalance treatment; that
  weakness is intentionally measured by the clean baseline.

### Impacted Files or Components

- `configs/experiments/stage01_isic2019_efficientnet_b0_cross_entropy.yaml`
- `configs/experiments/stage02_isic2019_efficientnet_b0_cross_entropy.yaml`
- future model and training modules

### Impact on Existing Experiments

No experiment is affected because training has not started.

### Review Trigger

Review after the clean baseline is evaluated on validation data or if hardware
constraints prevent practical training.

### Final Outcome

EfficientNet-B0 is prepared as the initial standard baseline.

---

## D-015 - Select Class-Balanced Focal Loss as the Frozen Stage 2 Model

**Date:** 2026-07-26
**Status:** Accepted
**Phase:** Phase 04
**Owner:** Research lead

### Context

The clean Stage 2 baseline showed weak SCC performance. Weighted
cross-entropy produced only marginal validation improvement. A
class-balanced focal-loss variant was therefore evaluated using the same
frozen split, architecture, preprocessing, optimizer, scheduler, and seed.

### Decision

Select EfficientNet-B0 with class-balanced focal loss as the frozen Stage 2
model.

- Effective-number beta: `0.9999`
- Focal gamma: `2.0`
- Seed: `42`
- Frozen checkpoint epoch: `8`

### Rationale

The selected variant achieved the highest validation macro-F1, balanced
accuracy, SCC recall, and SCC F1. The decision was made before evaluating
the checkpoint on the internal-test partition.

### Supporting Evidence

- Validation macro-F1: `0.776307`
- Validation balanced accuracy: `0.776287`
- Validation SCC recall: `0.617021`
- Validation SCC F1: `0.604167`
- Internal-test macro-F1: `0.724875`
- Internal-test balanced accuracy: `0.722716`
- Internal-test SCC recall: `0.457447`
- Internal-test SCC F1: `0.459893`
- Checkpoint SHA-256:
  `10986d41b64a685fcd8fe166623c5b1c7fd2f21bdad7cf4d55dedc3967a397fd`

### Risks and Trade-offs

- Overall accuracy was not the highest validation result.
- SCC precision decreased relative to the clean internal-test baseline.
- The validation SCC improvement did not fully generalize.
- Current evidence represents one random seed.

### Impact on Existing Experiments

The clean CE and weighted CE experiments remain valid controlled
comparators. Weighted CE was not internally tested because it was not
selected using validation evidence.

### Review Trigger

The frozen checkpoint must remain unchanged for primary hierarchical and
external evaluations. Any modified model must be recorded as a separate
experiment.

### Final Outcome

The epoch-8 class-balanced focal-loss checkpoint is frozen as the selected
Stage 2 model. Its one-time internal-test result is locked, and Phase 04 is
complete.

---

## D-016 - Lock the Phase 05 Hierarchical Result and Routing Interpretation

**Date:** 2026-07-27
**Status:** Accepted
**Phase:** Phase 05
**Owner:** Research lead

### Context

The frozen Stage 1 and selected class-balanced focal Stage 2 checkpoints were
combined in the locked conditional hierarchy. A one-time internal-test
evaluation was required to measure actual predicted-gate performance, oracle
routing performance, conditional execution, and error propagation.

### Decision

Accept and lock the successful Phase 05 predicted-gate result as the primary
hierarchical internal-test result.

- Predicted-gate four-class macro-F1: `0.605367`
- Predicted-gate accuracy: `0.740185`
- Predicted-gate balanced accuracy: `0.631199`
- Oracle-gate four-class macro-F1: `0.793656`
- Absolute macro-F1 loss from Stage 1 propagation: `0.188289`
- Production-style Stage 2 execution count: `1544 / 3668`
- Production-style Stage 2 execution rate: approximately `42.09%`
- Rerun permitted: `false`

### Rationale

The evaluation used frozen checkpoints, the predeclared argmax gate, the frozen
seed-42 leakage-aware split, and a locked output policy. Oracle-versus-predicted
analysis shows that Stage 1 routing is the dominant end-to-end bottleneck, while
SCC discrimination remains a secondary Stage 2 limitation.

### Supporting Evidence

- Malignant lesions blocked by Stage 1: `255 / 1270`
- Non-malignant lesions incorrectly routed: `529 / 2398`
- Subtype errors after correct malignant routing: `169 / 1015`
- Oracle-gated Stage 2 macro-F1: `0.724875`
- Oracle-gated SCC F1: `0.459893`
- Locked result artifact count: `16`
- Checksum entries verified locally: `18 / 18`
- Local archive SHA-256:
  `48455c488ecc74f5d859f796a343399ff9653eaf8b439de38d478bfc4362475a`

### Numerical Consistency Note

Five borderline non-malignant Stage 1 predictions differed from the earlier
standalone Phase 03 evaluation despite use of the same frozen checkpoint. All
five probabilities were close to the `0.5` decision boundary.

The difference is consistent with small floating-point changes under different
inference execution configurations. No malignant predictions changed, and no
checkpoint, label mapping, architecture, or gate policy changed.

The Phase 03 result remains the authoritative standalone baseline result. The
Phase 05 result remains the authoritative end-to-end hierarchical result.

### Failed Attempt and Recovery

Attempt 1 failed before metrics and before creation of the locked result
directory because half-precision probabilities could not be assigned to a
float32 collection tensor.

The failed attempt was preserved. The implementation was corrected by promoting
logits to float32 before probability collection. Regression tests, synthetic
CUDA validation, and checkpoint preflight passed before the successful recovery
attempt.

No model weight, checkpoint, hyperparameter, label, or routing decision changed.

### Risks and Trade-offs

- End-to-end macro-F1 is substantially below the oracle-gate diagnostic ceiling.
- Stage 1 blocks approximately one fifth of malignant samples.
- SCC remains weak even with oracle routing.
- Results represent one seed and one internal dataset.
- A direct flat four-class comparator is still required.
- No clinical or external-generalisation claim is supported yet.

### Impacted Files or Components

- `configs/evaluation/phase05_hierarchical_internal_test.yaml`
- `src/evaluation/hierarchical_inference_engine.py`
- `runs/phase05_hierarchical_internal_test/`
- `reports/phase05/conditional_hierarchical_internal_evaluation.md`
- `reports/phase05/stage01_numerical_consistency_audit.csv`
- `experiments/experiment_registry.csv`

### Impact on Existing Experiments

Phase 03 and Phase 04 results remain valid and locked. Their internal-test
results must not be rerun or used for retuning. Phase 05 does not replace the
standalone reports; it adds the end-to-end hierarchical evaluation view.

### Review Trigger

The locked result must not be reviewed or replaced through another internal-test
run. Any alternative gate, threshold, model, or routing method must be declared
as a separate experiment and must not use the locked internal-test result for
selection.

### Final Outcome

Phase 05 is complete. The reportable hierarchical result is frozen. Phase 06
will train and evaluate a fair direct flat four-class comparator.

---

## D-017 - Define the Phase 06 Fair Flat Four-Class Comparison

**Date:** 2026-07-27
**Status:** Accepted
**Phase:** Phase 06
**Owner:** Research lead

### Context

Phase 05 established the locked predicted-gate hierarchical result. A direct
four-class comparator is required to answer whether task decomposition helps
under a fair backbone, preprocessing, split, seed, and selection policy.

### Decision

Prepare Experiment A as a clean cross-entropy EfficientNet-B0 classifier with
class order `[non_malignant, melanoma, bcc, scc]`. Derive its target from
`diagnosis_canonical` on rows satisfying the frozen
`split_included=1 and include_stage_1=1` policy:

- `melanocytic_nevus`, `benign_keratosis_like_lesion`, `dermatofibroma`, and
  `vascular_lesion` -> `non_malignant`
- `melanoma` -> `melanoma`
- `basal_cell_carcinoma` -> `bcc`
- `squamous_cell_carcinoma` -> `scc`

Use validation macro-F1 for checkpoint selection. Keep the internal test locked
until the selected Experiment A checkpoint and evaluation protocol are frozen,
then evaluate it once. The primary comparison is Experiment A versus the Phase
05 predicted-gate hierarchy; the oracle gate is diagnostic only.

An imbalance-aware Experiment B is allowed only after clean-CE validation
analysis and may not use internal-test evidence.

### Rationale

Reusing the locked Stage 1 cohort excludes the same 867 actinic-keratosis rows
that are outside the project hierarchy and preserves the exact 24,460-row
comparison population. Reusing the existing trainer avoids implementation
differences unrelated to the research question.

### Supporting Evidence

- Label audit: `reports/phase06/flat_four_class_label_audit.json`
- Protocol: `reports/phase06/fair_flat_four_class_protocol.md`
- Mapped rows: `24,460`
- Train / validation / internal-test rows: `17,124 / 3,668 / 3,668`
- Split-group and exact-hash cross-split overlap: `0 / 0`

### Risks and Trade-offs

- SCC is rare, but Experiment A must remain clean CE for the first comparison.
- The result will represent one seed and one internal dataset.
- Internal-test access before freeze would invalidate the protocol.
- Latency and throughput are hardware-specific and must be measured on T4.

### Impacted Files or Components

- Phase 06 task adapter, audit, config, tests, protocol, and VM commands
- Experiment registry planned entry
- Existing Stage 1 and Stage 2 behavior remains unchanged

### Impact on Existing Experiments

None. Phase 03 through Phase 05 checkpoints, metrics, reports, and artifacts
remain locked and unchanged. Phase 05 references do not influence Phase 06
training or checkpoint selection.

### Review Trigger

Review after clean-CE validation results exist, before proposing Experiment B,
and again before authorizing the one-time internal-test evaluation.

### Final Outcome

The fair-comparison protocol and Experiment A preparation are accepted. No
Phase 06 model result exists yet.

---

## D-018 - Prepare One Phase 06B Class-Balanced Focal Candidate

**Date:** 2026-07-27
**Status:** Accepted
**Phase:** Phase 06B
**Owner:** Research lead

### Context

Phase 06A supplies the completed clean-CE flat baseline. One imbalance-aware
flat candidate is needed without changing any non-loss experimental setting.

### Decision

Evaluate EfficientNet-B0 with the repository's established class-balanced
focal loss, effective-number beta `0.9999`, and focal gamma `2.0`. Derive
counts only from the locked Phase 06 seed-42 training split in exact order
`[non_malignant, melanoma, bcc, scc]`: `[11193, 3164, 2327, 440]`. Compute and
persist the weights with the established implementation; do not invent manual
weights.

Keep every non-loss setting equivalent to Phase 06A. Select using validation
macro-F1 only, then validation balanced accuracy as tie-breaker. If both are
exactly tied, retain clean CE. SCC precision, recall, and F1 are secondary
interpretation metrics and do not override this rule.

### Rationale

This is a predeclared loss-only comparison that tests imbalance handling while
preserving the fairness controls of Phase 06A.

### Risks and Trade-offs

- Phase 06B has not yet been trained or evaluated.
- Results will represent one seed and one internal dataset.
- The internal test must remain hidden until a validation winner is frozen.
- No Phase 06B model construction, training, inference, or evaluation is
  permitted locally.

### Impacted Files or Components

- Phase 06B experiment config, focal config validation, local-safe tests
- Phase 06 documentation, VM commands, and experiment registry

### Impact on Existing Experiments

Phase 03, Phase 04, Phase 05, and Phase 06A behavior and artifacts remain
unchanged. The Phase 04 three-class focal numerical policy is reused.

### Review Trigger

Review after Azure T4 full tests and successful Phase 06B training, before any
internal-test access.

### Final Outcome

Phase 06B is accepted as a planned candidate. It is not experimentally
complete, and the internal test remains untouched.

---

## D-019 - Freeze the Validation-Selected Phase 06 Flat Model

**Date:** 2026-07-27
**Status:** Accepted
**Phase:** Phase 06B / Phase 06C
**Owner:** Research lead

### Context

Phase 06A clean cross-entropy and Phase 06B class-balanced focal loss completed
the predeclared validation-only model selection. The internal test remained
untouched.

### Options Considered

1. Phase 06A clean cross-entropy
2. Phase 06B class-balanced focal loss

### Decision

Freeze Phase 06A clean CE as the validation-selected flat model. Its only
eligible Phase 06C checkpoint is
`runs/phase06_full/full__phase06_flat_four_class_isic2019_efficientnet_b0_cross_entropy_seed42__20260726T232308Z/best_checkpoint.pt`,
SHA-256
`f3d8b8b0e5ef42e3c287a2377b5570411d442246acd16cb874ccf903facdc7a7`.

Validation macro-F1 is primary, validation balanced accuracy is the
tie-breaker, and the simpler clean CE is preferred on an exact tie. Phase 06B
is rejected under that policy. Its checkpoint is
`runs/phase06b/full/full__phase06b_flat_four_class_isic2019_efficientnet_b0_class_balanced_focal_loss_seed42__20260727T120615Z/best_checkpoint.pt`,
SHA-256
`07586d515cd9378e05831ca542f391e32b3b7a6c669c7dd83ce1df219b2af015`.

Exactly one future flat-model internal-test evaluation is allowed, using only
the selected checkpoint. No comparison of flat candidates on the internal
test, post-test tuning, or candidate switching is permitted.

### Rationale

Clean CE achieved validation macro-F1 `0.6535716654`, above focal loss at
`0.6490067298`. Balanced accuracy was nearly identical, but no tie-break was
needed. Focal loss improved SCC F1 from `0.3870967742` to `0.4302325581`, an
absolute secondary SCC F1 improvement of `0.0431357839`, but it did not win the
predeclared primary metric. The internal test cannot become another selection
stage.

### Supporting Evidence

- `reports/phase06/phase06b_class_balanced_focal_amendment.md`
- `configs/evaluation/phase06c_selected_flat_internal_test.yaml`
- Verified Phase 06A and Phase 06B validation artifacts and checkpoint hashes

### Risks and Trade-offs

- The decision prioritizes aggregate macro-F1 over the focal candidate's SCC
  improvement.
- The result represents one seed and one internal dataset.
- A valid internal-test run consumes the one-time protocol.

### Impacted Files or Components

- Phase 06 reports and experiment registry
- Phase 06C protocol and safe enforcement tests

### Impact on Existing Experiments

Earlier locked experiments are unchanged. Phase 06A becomes the sole Phase 06C
candidate; Phase 06B remains a documented, reproducible rejected candidate.

### Additional Time or Compute

One later Azure Tesla T4 internal-test evaluation is authorized. None was run
for this decision.

### Review Trigger

No performance-based review is allowed after internal-test access. A technical
retry is allowed only if the preceding attempt failed before producing valid
metrics and its failure reason is documented.

### Final Outcome

The selected checkpoint is frozen before internal-test access. The Phase 06C
one-time internal-test evaluation is prepared but not executed; the internal
test remained untouched.

---

## D-020 - Consume and Lock the Phase 06C Flat Internal-Test Protocol

**Date:** 2026-07-27
**Status:** Accepted
**Phase:** Phase 06C
**Owner:** Research lead

### Context

Phase 06A clean cross-entropy was frozen through validation-only selection over
the rejected Phase 06B focal candidate. Phase 06C authorized exactly one
internal-test evaluation using only that selected checkpoint.

### Decision

Accept the completed Phase 06C evaluation as the sole reportable flat-model
internal-test result. Mark the protocol `consumed_locked`,
`internal_test_accessed=true`, and
`valid_internal_test_run_completed=true`.

No additional run, candidate switch, focal-checkpoint evaluation, threshold
change, post-test tuning, or performance retry is permitted.

### Supporting Evidence

- Evaluation commit: `550e7cdb1144f059c940d4240fe4579e0280a803`
- Selected checkpoint SHA-256:
  `f3d8b8b0e5ef42e3c287a2377b5570411d442246acd16cb874ccf903facdc7a7`
- Final status: `0`
- Internal-test samples: `3668`
- Accuracy: `0.7420937841`
- Balanced accuracy: `0.6503125394`
- Macro-F1: `0.6192224685`
- Weighted F1: `0.7525567214`
- Mean loss: `0.6232672186`
- Verified local archive:
  `runs/backups/phase06c/phase06c_selected_flat_internal_test_550e7cdb1144.tar.gz`
- Archive SHA-256:
  `b76762b53a35a8d9b0aa96621d78ea0e4421aa6e8052d068ffc10648a4e63e91`
- Embedded artifact hashes verified: `12` entries

### Interpretation

The locked Phase 05 predicted-gate hierarchical macro-F1 was
`0.6053674006`. The selected flat model was higher by
`0.0138550680` on the same locked internal comparison
population.

This is a descriptive single-seed internal-dataset result. It does not establish
statistical significance, clinical superiority, fairness, external
generalisation, or state-of-the-art performance.

### Risks and Trade-offs

- SCC F1 remains low at `0.3571428571`.
- Performance represents one seed and one internal dataset.
- The internal test can no longer be used for model-development decisions.
- External evaluation remains necessary before any generalisation claim.

### Impacted Files or Components

- `configs/evaluation/phase06c_selected_flat_internal_test.yaml`
- `reports/phase06/phase06c_selected_flat_internal_test_protocol.md`
- `reports/phase06/phase06c_selected_flat_internal_test_result.md`
- `experiments/experiment_registry.csv`
- `tests/test_phase06c_selected_flat_internal_test_protocol.py`
- verified local Phase 06C archive

### Impact on Existing Experiments

Phase 05, Phase 06A, and Phase 06B artifacts remain unchanged. Phase 06A is the
only flat checkpoint that accessed the internal test. Phase 06B remains rejected
and prohibited from internal-test evaluation.

### Review Trigger

The result must not be replaced through another internal-test run. Future
review may use external-dataset evidence or a separately predeclared study, but
not the consumed ISIC 2019 internal test for further selection.

### Final Outcome

Phase 06C is complete. The selected flat internal-test result and its verified
local archive are locked. The fair locked comparison records flat macro-F1
`0.6192224685` versus hierarchical macro-F1
`0.6053674006`.

---

## D-021 - Close Phase 07 ICCIT Evidence Package

**Date:** 2026-07-28
**Status:** Accepted as local closure candidate
**Phase:** Phase 07
**Owner:** Research lead

### Decision

Accept the paired stored-prediction analysis, independent evidence review,
claims lock, efficiency audit, paper tables, and three required figures as the
Phase 07 ICCIT closure candidate. The flat-minus-hierarchical macro-F1
difference was `0.0138550680` with paired 95% CI
`[-0.0142546488, 0.0419633760]`; the interval included zero. Exact McNemar
testing did not detect a paired-correctness difference (`p=0.8207415883`).

The permitted conclusion is that the analysis did not establish a
statistically distinguishable macro-F1 difference on the locked split.
Equivalence, non-inferiority, clinical superiority, deployment, causal, and
external-generalization claims remain prohibited.

### Evidence and controls

- Paired samples: `3668`; SCC support: `94`.
- Bootstrap: paired, class-stratified, `10000` replicates, seed `42`.
- Quantile: explicit NumPy `method="linear"`, frozen before execution.
- Claims and routing denominators independently reviewed.
- Efficiency timing classified comparable with limitations; no speed ratio.
- Required figures: architecture, normalized confusion matrices, exploratory
  per-class F1 intervals.
- Gate 3, Gate 4, Gate 5A, locked predictions, and bootstrap replicates
  unchanged.
- Phase 05 and Phase 06C internal-test protocols remain consumed; no rerun,
  candidate switching, threshold tuning, or post-test model development is
  permitted.

### Final outcome

Phase 07 is complete as a local closure candidate. Push and merge require
explicit independent human review and approval.

---

## D-022 - Continue Under the Full Original Three-Stage Scope

**Date:** 2026-07-29
**Status:** Accepted
**Phase:** Phase 08
**Owner:** Research lead

### Context

The original charter and scope lock specify three stages, partially labelled
multi-dataset learning, a lightweight shared or parameter-efficient framework,
separate-model and flat comparisons, external evaluation, XAI, and complete
efficiency analysis. Phases 05–07 provide locked evidence for a two-stage
conditional comparator and its flat comparison, but do not complete those
remaining objectives.

### Options Considered

1. Option A — continue toward the full original three-stage scope.
2. Reduce the research claim to the completed two-stage internal study.

### Decision

Select Option A. The study will continue toward a three-stage system:
malignancy screening, malignant-subtype classification, and a scientifically
defensible melanoma T-category or Breslow-thickness-group task.

The current Phase 05–07 two-stage system remains a locked comparator. Stage 3,
an approved external evaluation, preregistered XAI, and comparison against
separate task-specific models remain required. No new experiment may start
before its applicable protocol is frozen. Stage 3 shared-model training may
not begin until the EMB audit and standalone feasibility gate pass.

Stage 3 semantics remain unresolved pending authoritative EMB documentation
and metadata audit. T-category and Breslow-thickness groups are not
automatically interchangeable; no class boundary is invented here. The final
target may be categorical, ordinal, or another justified formulation. A fully
shared encoder is not guaranteed before feasibility evidence. HIBA is only a
candidate external dataset until compatibility, licence, modality, labels,
identifiers, and support are audited. XAI cannot prove clinical correctness,
and its examples require predefined selection rules. External evaluation
cannot prove universal clinical generalisation. Negative, null, and
non-significant results must be reported.

### Supporting Evidence

- `docs/00_project_charter.md`
- `docs/01_scope_lock.md`
- `reports/phase08/phase08_scope_deviation_and_evidence_audit.md`
- `reports/phase08/phase08_remaining_experiment_plan.md`
- `reports/phase08/phase08_protocol_freeze.md`
- `reports/phase08/generated/phase08_objective_evidence_matrix.csv`

### Risks and Trade-offs

- Additional local governance work and later Azure Tesla T4 computation are
  required.
- EMB may fail its licensing, label-semantics, support, or leakage-control gate.
- A shared model may underperform separate models; no superiority is assumed.
- External performance or XAI may be unfavourable and must still be reported.

### Impact on Existing Experiments

Phase 05 hierarchical inference, Phase 06C flat inference, the rejected Phase
06B focal candidate, and all Phase 07 analysis remain locked and unchanged.
They must not be rerun or repurposed for model selection.

### Additional Time or Compute

Phase 08 requires no GPU. Later accepted protocols will require Azure Tesla T4
training, frozen inference, XAI generation, and matched efficiency profiling.
No Azure execution is authorized by this decision alone.

### Review Trigger

Review at the Phase 09 Stage 3 feasibility gate and before every later first
training or inference access.

### Final Outcome

Full original scope is retained. Completed and future evidence will remain
explicitly separated, and manuscript claims will follow the evidence rather
than the proposal wording.

---

## D-023 - Accept the ISIC-Derived Stage-3 Feasibility Result and Select Weighted Cross-Entropy

**Date:** 2026-07-30
**Status:** Accepted
**Phase:** Phase 09
**Owner:** Research lead

### Context

The EMB repository at commit
`3ec674f43e73cb08682b99b7fb996aca5f8040d8` had no identifiable licence, so
no EMB or Atlas images could be used. Phase 09 instead used the EMB CSV only as
an index of candidate public ISIC identifiers and independently established
the licensed, attributed **ISIC-derived melanoma T-category subset** from
official ISIC Archive API v2 metadata.

The leakage-safe seed-42 split contained 848 images. The ordinary
cross-entropy baseline selected epoch 2 with validation macro-F1
`0.36561465460163317` and showed strong Tis majority collapse on its locked
test. One predeclared train-only inverse-frequency weighted-cross-entropy
candidate selected epoch 12 with validation macro-F1 `0.43657311157311157`.

### Options Considered

1. Retain ordinary cross-entropy as the standalone Stage-3 result.
2. Select the single weighted-cross-entropy candidate because it exceeded the
   predeclared validation threshold.
3. Continue tuning Stage 3 after viewing internal-test results.
4. Abandon the audited standalone feasibility result.

### Decision

Accept the ISIC-derived standalone Stage-3 feasibility result and select the
inverse-frequency weighted-cross-entropy candidate. Selection was made from
validation macro-F1 before weighted-candidate test access because
`0.43657311157311157` was strictly greater than `0.365615`.

Both baseline and weighted internal-test evaluations are now consumed and
locked. `rerun_allowed=false`; no further Stage-3 tuning or internal-test rerun
is permitted.

### Rationale

The weighted candidate improved locked-test macro-F1 from
`0.16283767911674887` to `0.2756106656721984` and balanced accuracy from
`0.20240259740259742` to `0.38603896103896107`. Accuracy decreased from
`0.6062992125984252` to `0.5433070866141733`. T2 and T4 recall remained zero,
and the apparently perfect T3 recall was based on only two images. The result
therefore supports feasibility and a descriptive model choice, not statistical
superiority or clinical adequacy.

### Supporting Evidence

- `data/manifests/emb_stage03_dermoscopic_split_seed42.csv`
- `data/manifests/emb_stage03_dermoscopic_split_seed42.audit.json`
- `experiments/evaluations/stage03_isic_derived_internal_test_seed42__best_epoch02/`
- `experiments/evaluations/stage03_isic_derived_wce_internal_test_seed42__best_epoch12/`
- `reports/phase09/isic_stage03_fasttrack_result.md`
- `reports/phase09/emb_stage03_fasttrack_protocol.md`

### Risks and Trade-offs

- T3 and T4 support is extremely small.
- T2 and T4 had zero recall under the selected model.
- Patient and lesion metadata is incomplete, although all known relations are
  preserved without cross-split leakage.
- This is a single-seed internal result and cannot establish statistical
  superiority or external generalisation.
- Lower weighted-model accuracy reflects the trade-off made for less collapsed
  classwise performance.

### Impacted Files or Components

- Phase 09 report and protocol
- Dataset roles and registry
- Experiment registry
- Locked Stage-3 manifest, audit, predictions, and metric artifacts

### Impact on Existing Experiments

Earlier Phase 05-08 evidence remains unchanged and locked. The generic blocked
Stage-3 row is superseded. This result is not an integrated three-stage system
and does not authorize later shared-model, external-evaluation, or clinical
claims.

### Additional Time or Compute

None. Phase 09 is closed without further training, evaluation, or inference.

### Review Trigger

Review only for evidence-integrity correction or a separately preregistered
future study with genuinely independent data. The locked internal test cannot
be reopened for tuning.

### Final Outcome

The weighted candidate is `completed_locked_selected`; the clean CE baseline
is `completed_locked_baseline`. Both internal-test evaluations are consumed,
all evidence hashes are locked, and no further Stage-3 tuning or test rerun is
allowed.

---

## D-024 - Register HIBA as a Pending Frozen External-Evaluation Candidate

**Date:** 2026-07-30
**Status:** Accepted
**Phase:** Phase 10A
**Owner:** Research lead

### Context

HIBA is mandatory for the planned external evaluation, but its local official
metadata and files have not yet passed licence, modality, label, identifier,
support, integrity, and ISIC-overlap review. Its official release identity is
ISIC collection 251, DOI `10.34970/587329`.

### Decision

Register HIBA with status
`candidate_pending_official_acquisition_audit`. Only dermoscopic rows may enter
the eventual primary four-class cohort. Original diagnoses are preserved and
only explicit exact mappings are allowed. Actinic keratosis, clinical images,
unknown or ambiguous diagnoses, and unsupported labels are excluded.

The audit protocol creates one manifest without train/validation/test splits,
streams file SHA-256 values, rejects duplicate image IDs, identifies duplicate
content and label conflicts, and compares image IDs and hashes with the locked
ISIC 2019 manifest. Any overlap or unresolved mapping blocks approval.

### Rationale

This separates dataset registration and compatibility auditing from inference
and prevents HIBA from influencing model development. Automated passage is
necessary but does not replace human verification of official release
metadata, licence, attribution, and overlap disposition.

### Supporting Evidence

- `reports/phase10/hiba_external_dataset_audit_protocol.md`
- `configs/datasets/hiba_external_label_mapping.yaml`
- `configs/evaluation/phase10_hiba_frozen_zero_shot.yaml`
- `scripts/audit_hiba_external_dataset.py`

### Impact on Existing Experiments

None. Phase 05, Phase 06C, Phase 07, and Phase 09 evidence remains untouched.
No HIBA data was downloaded, no checkpoint was loaded, and no inference or
performance inspection was authorized.

### Review Trigger

After official HIBA metadata and files are acquired and the complete audit is
reviewed, but before any one-time external inference.

### Final Outcome

HIBA remains disabled and unapproved for evaluation. The audit framework is
registered; inference remains prohibited.

