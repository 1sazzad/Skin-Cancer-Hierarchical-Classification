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

