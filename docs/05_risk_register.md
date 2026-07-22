# Project Risk Register and Mitigation Plan

## Purpose

This document records the major scientific, technical, data, operational, and project-management risks associated with the research.

The objective is not to eliminate all uncertainty. The objective is to identify important risks early, reduce avoidable failure, and prevent weak experimental decisions from being introduced silently.

---

# 1. Risk Rating Scale

## Likelihood

| Rating | Meaning                                      |
| ------ | -------------------------------------------- |
| Low    | Unlikely under the current plan              |
| Medium | Plausible and requires monitoring            |
| High   | Expected to occur unless actively controlled |

## Impact

| Rating   | Meaning                                       |
| -------- | --------------------------------------------- |
| Low      | Limited effect on results or schedule         |
| Medium   | Requires rework or reduces experiment quality |
| High     | Threatens a major research objective          |
| Critical | Can invalidate the study or its main claims   |

## Status Values

Use the following status values:

```text
open
monitoring
mitigated
accepted
triggered
closed
```

---

# 2. Risk Summary

| ID  | Risk                                                               | Likelihood | Impact   | Status     |
| --- | ------------------------------------------------------------------ | ---------- | -------- | ---------- |
| R01 | Stage 3 labels are insufficient or clinically inconsistent         | High       | High     | Open       |
| R02 | SCC or another malignant class has critically low sample count     | High       | High     | Open       |
| R03 | Patient-level or lesion-level data leakage                         | Medium     | Critical | Open       |
| R04 | External dataset labels do not map cleanly                         | Medium     | High     | Open       |
| R05 | Hierarchical routing causes severe error propagation               | High       | High     | Open       |
| R06 | Overall accuracy hides minority-class failure                      | High       | High     | Open       |
| R07 | Model explanations are visually attractive but scientifically weak | Medium     | High     | Open       |
| R08 | Dataset duplication or near-duplication inflates performance       | Medium     | Critical | Open       |
| R09 | External evaluation contaminates model-development decisions       | Medium     | Critical | Open       |
| R10 | Scope creep delays the mandatory research pipeline                 | High       | High     | Monitoring |
| R11 | GPU quota, cost, or availability interrupts training               | Medium     | High     | Open       |
| R12 | Experiment artifacts remain only on Azure                          | Medium     | High     | Open       |
| R13 | Inconsistent naming makes experiments difficult to reuse           | Medium     | Medium   | Mitigated  |
| R14 | Working directory becomes cluttered with temporary outputs         | Medium     | Medium   | Mitigated  |
| R15 | A single favourable seed is treated as strong evidence             | Medium     | High     | Open       |
| R16 | Preprocessing differences make model comparisons unfair            | Medium     | High     | Open       |
| R17 | Test-set results influence later model selection                   | Medium     | Critical | Open       |
| R18 | External domain shift causes major performance degradation         | High       | High     | Open       |
| R19 | Probability estimates are severely overconfident                   | High       | Medium   | Open       |
| R20 | Project time is insufficient for all optional experiments          | High       | Medium   | Accepted   |
| R21 | Dataset licence or access conditions restrict usage                | Medium     | High     | Open       |
| R22 | Clinical terminology or label mapping is misunderstood             | Medium     | Critical | Open       |
| R23 | Corrupted, missing, or mismatched image files reduce usable data   | Medium     | Medium   | Open       |
| R24 | Model efficiency claims are based on unfair hardware comparisons   | Medium     | Medium   | Open       |
| R25 | Final conclusions overstate clinical readiness                     | Medium     | Critical | Open       |

---

# 3. Detailed Risk Register

## R01 — Stage 3 Label Feasibility

### Risk

The EMB dataset may not provide enough reliable T-category or Breslow-thickness information to support a valid Stage 3 classification task.

### Likelihood

High

### Impact

High

### Warning Signs

* unclear severity-field definitions;
* high label missingness;
* very few samples in one or more groups;
* severity groups require unsupported assumptions;
* multiple images cannot be connected safely to patients or lesions;
* inconsistent clinical units;
* labels conflict with dataset documentation.

### Mitigation

* perform the EMB feasibility audit before Stage 3 model development;
* preserve all original labels;
* verify clinical definitions using authoritative documentation;
* calculate sample counts before defining severity groups;
* define grouping thresholds in a version-controlled configuration;
* require leakage-safe splitting;
* document every exclusion.

### Contingency

Use Breslow-thickness grouping only when it is scientifically defensible.

When neither target is valid, report Stage 3 as a feasibility limitation rather than forcing an unreliable experiment.

### Owner

Research lead

### Status

Open

---

## R02 — Severe Class Imbalance

### Risk

SCC or another malignant class may contain too few samples for stable training and evaluation.

### Likelihood

High

### Impact

High

### Warning Signs

* extremely low class frequency;
* unstable validation metrics;
* high variance across seeds;
* zero or near-zero recall;
* class absent from one split;
* strong majority-class bias.

### Mitigation

* audit class counts before splitting;
* use grouped stratification where feasible;
* ensure each split contains adequate class representation;
* compare standard and imbalance-aware losses;
* report classwise metrics;
* use PR-AUC and macro F1-score;
* consider bootstrap confidence intervals;
* avoid excessive augmentation that creates misleading confidence.

### Contingency

Reduce experimental claims and treat the rare class as an uncertainty-sensitive evaluation target.

Do not remove a clinically important class solely to improve headline accuracy without a documented decision.

### Owner

Data and modelling pipeline

### Status

Open

---

## R03 — Patient or Lesion Leakage

### Risk

Images from the same patient or lesion may appear across training, validation, and test splits.

### Likelihood

Medium

### Impact

Critical

### Warning Signs

* repeated patient identifiers across splits;
* repeated lesion identifiers across splits;
* identical or near-identical images in different splits;
* unexpectedly high validation or test performance;
* multiple crops or views of the same lesion split separately.

### Mitigation

* split by patient whenever possible;
* otherwise split by lesion;
* audit identifiers after split generation;
* compute duplicate hashes;
* perform near-duplicate checks where feasible;
* save and hash split manifests;
* freeze the split before modelling.

### Contingency

Invalidate affected experiments, correct the split, and rerun them.

Do not retain contaminated results for final comparison.

### Owner

Data pipeline

### Status

Open

---

## R04 — External Label Incompatibility

### Risk

HIBA or MRA-MIDAS labels may not correspond directly to the classes used in the development dataset.

### Likelihood

Medium

### Impact

High

### Warning Signs

* ambiguous diagnostic categories;
* different terminology;
* mixed or combined diagnoses;
* unsupported malignant subtypes;
* unclear benign categories;
* inconsistent class prevalence.

### Mitigation

* audit external labels before inference;
* create explicit mapping configurations;
* preserve original external labels;
* document included and excluded classes;
* avoid forcing ambiguous classes into target categories;
* report external sample counts after mapping.

### Contingency

Restrict external evaluation to compatible tasks and classes.

Clearly state that the resulting evaluation covers a subset of the full hierarchy when necessary.

### Owner

Research lead and data pipeline

### Status

Open

---

## R05 — Hierarchical Error Propagation

### Risk

Stage 1 false negatives may prevent malignant lesions from reaching Stage 2, reducing end-to-end performance.

### Likelihood

High

### Impact

High

### Warning Signs

* strong Stage 2 performance but weak end-to-end performance;
* high Stage 1 malignant false-negative rate;
* melanoma, BCC, or SCC cases blocked by the first gate;
* large oracle-gate versus predicted-gate performance gap.

### Mitigation

* optimise Stage 1 using malignant recall and balanced metrics;
* report stage-level and end-to-end results;
* compare oracle-gate and predicted-gate evaluation;
* evaluate threshold trade-offs using validation data;
* analyse blocked malignant samples;
* consider calibrated or uncertainty-aware routing as a later ablation.

### Contingency

Report the hierarchy as a trade-off rather than claiming automatic superiority.

A flat classifier may remain competitive or preferable if routing damage is excessive.

### Owner

Modelling and evaluation pipeline

### Status

Open

---

## R06 — Misleading Overall Accuracy

### Risk

High overall accuracy may hide poor performance on rare malignant classes.

### Likelihood

High

### Impact

High

### Warning Signs

* high accuracy with low macro F1-score;
* high weighted F1-score with poor SCC recall;
* strong majority-class prediction bias;
* clinically important errors hidden in aggregate metrics.

### Mitigation

Use the following as primary evidence:

* macro F1-score;
* balanced accuracy;
* per-class recall;
* per-class precision;
* confusion matrix;
* PR-AUC where applicable.

### Contingency

Reject interpretations based only on overall accuracy.

Explicitly discuss majority-class dominance.

### Owner

Evaluation pipeline

### Status

Open

---

## R07 — Weak Explainability Evidence

### Risk

Explanation maps may look convincing without demonstrating that the model uses clinically meaningful evidence.

### Likelihood

Medium

### Impact

High

### Warning Signs

* explanations focus on borders, rulers, hair, colour charts, or backgrounds;
* only favourable examples are shown;
* different methods produce contradictory maps;
* explanations are interpreted as proof of clinical reasoning;
* no external examples are evaluated.

### Mitigation

* include correct and incorrect predictions;
* include high-confidence errors;
* include minority-class examples;
* compare internal and external samples;
* assess explanation stability under controlled transformations;
* quantify lesion focus when reliable masks are available;
* state the limitations of post-hoc XAI.

### Contingency

Present explainability as exploratory error analysis rather than evidence of clinical validity.

### Owner

Explainability pipeline

### Status

Open

---

## R08 — Duplicate and Near-Duplicate Images

### Risk

Duplicate or visually near-identical images may inflate internal or external performance.

### Likelihood

Medium

### Impact

Critical

### Warning Signs

* identical file hashes;
* similar filenames or metadata;
* multiple resized versions of the same image;
* visually identical lesions in different datasets;
* unexpectedly confident external predictions.

### Mitigation

* compute SHA-256 hashes;
* inspect metadata overlap;
* apply perceptual-hash or embedding-based similarity checks where feasible;
* record duplicate decisions in manifests;
* keep related images in the same split.

### Contingency

Remove or regroup contaminated records and rerun affected experiments.

### Owner

Data integrity pipeline

### Status

Open

---

## R09 — External Evaluation Contamination

### Risk

Repeated inspection of HIBA results may influence model selection, making the external evaluation no longer independent.

### Likelihood

Medium

### Impact

Critical

### Warning Signs

* preprocessing changed after checking HIBA;
* thresholds adjusted using HIBA;
* architecture changed specifically to improve HIBA;
* multiple attempts are reported as a single zero-shot result.

### Mitigation

* freeze model and inference configuration before external evaluation;
* record the freeze point;
* preserve the first zero-shot result;
* treat all later adaptation as separate experiments;
* document decisions in the decision log.

### Contingency

Relabel the affected analysis as post-external adaptation and avoid presenting it as untouched validation.

### Owner

Research lead

### Status

Open

---

## R10 — Scope Creep

### Risk

Additional datasets, model families, modalities, or deployment tasks may delay the mandatory research objectives.

### Likelihood

High

### Impact

High

### Warning Signs

* new ideas added before baseline completion;
* repeated architecture switching;
* optional datasets used before HIBA;
* mobile deployment begins before evaluation is complete;
* multi-omic work re-enters the main scope.

### Mitigation

* enforce `docs/01_scope_lock.md`;
* use the decision log;
* maintain mandatory and optional task separation;
* complete one phase before activating the next;
* reject tasks that do not answer a research question.

### Contingency

Move nonessential work to future work or an optional backlog.

### Owner

Research lead

### Status

Monitoring

---

## R11 — GPU Availability, Quota, or Cost

### Risk

Azure GPU availability, regional quota, or cost may interrupt training.

### Likelihood

Medium

### Impact

High

### Warning Signs

* VM allocation failure;
* insufficient regional quota;
* unexpected spending;
* long-running experiments;
* repeated failed runs consuming compute.

### Mitigation

* debug locally using small subsets;
* use lightweight models first;
* verify scripts before GPU launch;
* use mixed precision when valid;
* use early stopping;
* define maximum epochs;
* shut down the VM after use;
* retain resumable checkpoints;
* maintain a compute budget.

### Contingency

Reduce seed count for secondary experiments, prioritise mandatory comparisons, and defer optional models or datasets.

### Owner

Research lead and infrastructure

### Status

Open

---

## R12 — Azure-Only Artifacts

### Risk

Checkpoints, logs, predictions, or metrics may remain only on the Azure VM and be lost.

### Likelihood

Medium

### Impact

High

### Warning Signs

* local run directory missing;
* no copied checkpoint;
* no environment record;
* VM deletion or storage reset;
* manually remembered metrics.

### Mitigation

* copy required artifacts after every meaningful run;
* use the standard run-directory structure;
* verify file integrity after transfer;
* update the experiment registry;
* maintain at least one backup of final artifacts.

### Contingency

Repeat the experiment only when the missing outputs are essential and cannot be reconstructed.

### Owner

Infrastructure and experiment management

### Status

Open

---

## R13 — Ambiguous Naming

### Risk

Poor names may make configurations, checkpoints, figures, and results difficult to identify or reuse.

### Likelihood

Medium

### Impact

Medium

### Mitigation

Use descriptive names containing:

* stage;
* dataset;
* model;
* variant;
* seed;
* metric or output purpose where relevant.

Reject names such as:

```text
final
new
best
test
result
model1
experiment2
```

### Contingency

Rename lightweight artifacts before broader use and document changes when paths are referenced elsewhere.

### Owner

Entire project

### Status

Mitigated

---

## R14 — Working Directory Clutter

### Risk

Temporary files, downloaded archives, screenshots, logs, and copied outputs may make the project difficult to maintain.

### Likelihood

Medium

### Impact

Medium

### Mitigation

* keep the project root minimal;
* use dedicated directories;
* maintain `.gitignore`;
* remove obsolete temporary files;
* archive rather than duplicate;
* review `git status` before every commit;
* do not place datasets or checkpoints in the root.

### Contingency

Perform a controlled cleanup, checking references before moving or deleting files.

### Owner

Entire project

### Status

Mitigated

---

## R15 — Single-Seed Conclusions

### Risk

A single favourable random seed may be mistaken for reliable model superiority.

### Likelihood

Medium

### Impact

High

### Mitigation

* use seed `42` for pipeline debugging;
* use `42`, `123`, and `2026` for major final comparisons where feasible;
* report mean and standard deviation;
* preserve every registered run;
* avoid selecting the best seed only.

### Contingency

Reduce the strength of conclusions when compute permits only one seed.

### Owner

Modelling and evaluation pipeline

### Status

Open

---

## R16 — Unfair Model Comparison

### Risk

Different image sizes, augmentations, splits, training budgets, or selection metrics may make flat and hierarchical comparisons unfair.

### Likelihood

Medium

### Impact

High

### Mitigation

Use equivalent:

* dataset manifests;
* split manifests;
* preprocessing;
* augmentation;
* training budget;
* pretrained-weight policy;
* checkpoint-selection method;
* metrics;
* seed policy.

Document unavoidable differences.

### Contingency

Label the comparison as non-equivalent and avoid strong superiority claims.

### Owner

Modelling pipeline

### Status

Open

---

## R17 — Test-Set Feedback Leakage

### Risk

Internal test results may influence later model-development decisions.

### Likelihood

Medium

### Impact

Critical

### Warning Signs

* frequent test evaluation during training;
* changing hyperparameters after test inspection;
* selecting checkpoints by test score;
* reporting the best of many test attempts.

### Mitigation

* use validation data for all development decisions;
* freeze the selected model before test evaluation;
* limit final test execution;
* record the model-selection decision;
* preserve the first final test result.

### Contingency

If contamination occurs, create a fresh untouched test split only when scientifically justified and documented, or reduce claims accordingly.

### Owner

Research lead

### Status

Open

---

## R18 — External Domain Shift

### Risk

HIBA performance may be substantially lower than internal ISIC 2019 performance.

### Likelihood

High

### Impact

High

### Warning Signs

* large accuracy and macro-F1 drop;
* increased calibration error;
* class-prevalence mismatch;
* image-style differences;
* explanations focus on acquisition artifacts;
* reduced malignant recall.

### Mitigation

* treat external evaluation as a primary research question;
* quantify the internal-external gap;
* analyse classwise failure;
* examine image and metadata distributions;
* report calibration changes;
* avoid hiding negative external results.

### Contingency

Present the domain-shift result as an important contribution and evaluate adaptation only as a separate secondary experiment.

### Owner

Evaluation pipeline

### Status

Open

---

## R19 — Model Overconfidence

### Risk

The model may assign high confidence to incorrect predictions, especially on external data.

### Likelihood

High

### Impact

Medium

### Mitigation

* evaluate expected calibration error;
* calculate Brier score;
* produce reliability diagrams;
* compare correct and incorrect confidence distributions;
* fit calibration using validation data only;
* report selective threshold trade-offs.

### Contingency

Avoid presenting raw probabilities as reliable clinical confidence.

### Owner

Calibration and evaluation pipeline

### Status

Open

---

## R20 — Insufficient Time for Optional Work

### Risk

The project schedule may not allow MRA-MIDAS, advanced explainability, extensive architecture search, or deployment.

### Likelihood

High

### Impact

Medium

### Mitigation

Prioritise:

1. data audit;
2. leakage-safe splits;
3. baseline;
4. hierarchical framework;
5. imbalance experiment;
6. HIBA external evaluation;
7. calibration;
8. core explainability;
9. efficiency analysis.

Optional tasks begin only after mandatory outputs are complete.

### Contingency

Move incomplete optional work to future work.

### Owner

Research lead

### Status

Accepted

---

## R21 — Dataset Licence or Access Restrictions

### Risk

A dataset may have licence, citation, access, redistribution, or usage restrictions.

### Likelihood

Medium

### Impact

High

### Mitigation

* record official source pages;
* record access dates;
* preserve licence information;
* follow citation requirements;
* avoid committing restricted data;
* avoid redistributing raw datasets;
* document any access approval.

### Contingency

Remove the dataset from active scope through the formal decision process when it cannot be used legally or ethically.

### Owner

Research lead

### Status

Open

---

## R22 — Incorrect Clinical Label Interpretation

### Risk

Clinical labels such as malignancy, T-category, Breslow thickness, or diagnostic subtype may be mapped incorrectly.

### Likelihood

Medium

### Impact

Critical

### Mitigation

* preserve original labels;
* consult official dataset documentation;
* validate mappings using authoritative clinical references;
* separate uncertain labels;
* store mappings in configuration files;
* review definitions before modelling.

### Contingency

Invalidate affected mappings and rerun all dependent experiments.

### Owner

Research lead

### Status

Open

---

## R23 — Missing or Corrupted Files

### Risk

Metadata may reference missing, corrupted, unreadable, or mismatched images.

### Likelihood

Medium

### Impact

Medium

### Mitigation

* build a dataset manifest;
* validate every image path;
* test image decoding;
* record dimensions and formats;
* compute file hashes;
* document exclusions.

### Contingency

Exclude affected records through the manifest using a clear reason.

### Owner

Data pipeline

### Status

Open

---

## R24 — Unfair Efficiency Claims

### Risk

Latency, memory, and training-time comparisons may use different hardware or measurement conditions.

### Likelihood

Medium

### Impact

Medium

### Mitigation

Use the same:

* hardware;
* batch size;
* precision mode;
* input resolution;
* warm-up procedure;
* repetition count;
* inference implementation.

Record all measurement conditions.

### Contingency

Report results separately by hardware and avoid direct claims when conditions differ.

### Owner

Efficiency evaluation pipeline

### Status

Open

---

## R25 — Overstated Clinical Claims

### Risk

The final paper may imply clinical diagnostic readiness without appropriate validation.

### Likelihood

Medium

### Impact

Critical

### Mitigation

Use careful language:

* classification framework;
* research evaluation;
* decision-support potential;
* retrospective dataset study;
* not a medical device;
* not a replacement for dermatologist assessment.

Explicitly report limitations involving:

* dataset bias;
* external performance;
* calibration;
* subgroup coverage;
* retrospective evaluation;
* absence of prospective clinical validation.

### Contingency

Revise claims, title, abstract, conclusion, and figures before submission.

### Owner

Research lead and manuscript review

### Status

Open

---

# 4. Risk Review Procedure

The risk register must be reviewed:

* at the end of every major phase;
* before dataset splitting;
* before final model training;
* before external evaluation;
* before final manuscript submission;
* whenever a major problem occurs.

When a risk is triggered, update:

* status;
* date;
* observed evidence;
* action taken;
* affected files or experiments;
* final resolution.

---

# 5. Triggered-Risk Record Template

Use this format below the corresponding risk or in a future dedicated risk log.

```text
Risk ID:
Date triggered:
Observed event:
Affected phase:
Affected dataset or experiment:
Immediate action:
Root cause:
Corrective action:
Experiments invalidated:
Files modified:
Current status:
Resolution date:
```

---

# 6. Phase 00 Risk Position

At the end of Phase 00:

* scope creep is actively controlled;
* naming and directory-clutter risks are mitigated through architecture rules;
* Stage 3 remains a high-risk feasibility-dependent component;
* leakage prevention remains a critical requirement;
* external evaluation contamination remains prohibited;
* GPU and project-time constraints must influence experiment prioritisation;
* no risk should be hidden to protect a positive result.

A negative finding, incomplete optional component, or reduced external performance is scientifically acceptable when it is documented honestly and evaluated rigorously.
