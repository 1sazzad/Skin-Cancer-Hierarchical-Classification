# Research Scope Lock

## Purpose

This document freezes the primary scope of the Final Year Project.

Its purpose is to prevent uncontrolled changes in datasets, classes, models, objectives, and evaluation procedures after experiments begin.

Any material change must be documented in:

`docs/06_decision_log.md`

---

# 1. Locked Research Objective

The project will design and evaluate a lightweight conditional hierarchical deep-learning framework for dermoscopic skin-lesion classification.

The framework will be compared against conventional flat classification under equivalent experimental conditions.

The primary research focus is:

* hierarchical decision-making;
* malignant-lesion recognition;
* minority-class performance;
* error propagation;
* external generalisation;
* model calibration;
* computational efficiency;
* explainability.

---

# 2. Locked Classification Hierarchy

## Stage 1 — Malignancy Screening

### Task

Binary classification.

### Classes

* Malignant
* Non-malignant

### Purpose

Stage 1 determines whether a lesion should be routed to the malignant-lesion classifier.

### Routing Rule

A sample predicted as malignant proceeds to Stage 2.

A sample predicted as non-malignant does not proceed to Stage 2 during predicted-gate evaluation.

---

## Stage 2 — Malignant Lesion Classification

### Task

Three-class classification.

### Classes

* Melanoma
* Basal cell carcinoma
* Squamous cell carcinoma

### Abbreviations

* Melanoma: `melanoma`
* Basal cell carcinoma: `bcc`
* Squamous cell carcinoma: `scc`

### Purpose

Stage 2 differentiates the major malignant lesion categories included in the project.

### Evaluation Modes

Stage 2 must be evaluated in two modes:

1. **Oracle-gate evaluation**

   Samples are routed using the true Stage 1 label.

2. **Predicted-gate evaluation**

   Samples are routed using the Stage 1 prediction.

The difference between these modes will quantify hierarchical error propagation.

---

## Stage 3 — Melanoma Severity Classification

### Task

Classification of melanoma severity using clinically meaningful groups.

### Preferred Target

T-category grouping.

### Fallback Target

Breslow-thickness grouping.

### Dataset

EMB.

### Feasibility Condition

Stage 3 is not automatically guaranteed.

It will proceed only after a formal audit confirms:

* sufficient sample count;
* clear clinical label definitions;
* acceptable missingness;
* usable patient identifiers;
* valid class-group boundaries;
* adequate samples per class;
* no unsupported label inference.

### Fallback Rule

If neither T-category nor Breslow-thickness grouping is scientifically defensible, Stage 3 will be reported as a feasibility limitation.

The project must not force a Stage 3 experiment using unreliable labels.

---

# 3. Locked Dataset Roles

## ISIC 2019

### Role

Primary development dataset.

### Permitted Uses

* training;
* internal validation;
* internal testing;
* Stage 1 experiments;
* Stage 2 experiments;
* baseline comparison;
* flat-versus-hierarchical comparison;
* imbalance-aware experiments;
* calibration experiments;
* efficiency experiments;
* internal explainability analysis.

### Restrictions

* patient or lesion leakage must be audited;
* test data must not be used for model selection;
* preprocessing decisions must be based on training and validation data only.

---

## EMB

### Role

Stage 3 label-feasibility analysis and melanoma-severity modelling.

### Permitted Uses

* metadata and label audit;
* severity-target definition;
* Stage 3 training;
* Stage 3 validation;
* Stage 3 testing.

### Restrictions

EMB must not be merged into the main ISIC 2019 development pipeline without a documented scientific justification.

---

## HIBA

### Role

Mandatory independent external evaluation dataset.

### Permitted Uses

* zero-shot external evaluation;
* domain-shift analysis;
* external calibration analysis;
* external explainability analysis;
* classwise performance comparison.

### Prohibited Uses

HIBA must not be used for:

* model training;
* model selection;
* hyperparameter tuning;
* threshold selection;
* early stopping;
* architecture selection;
* preprocessing decisions based on HIBA performance.

The primary HIBA result must represent evaluation of a frozen model.

---

## MRA-MIDAS

### Role

Optional second external evaluation dataset.

### Priority

Deferred until all mandatory experiments are complete.

### Activation Conditions

MRA-MIDAS may be used only when:

* ISIC 2019 experiments are complete;
* HIBA external evaluation is complete;
* mandatory tables and figures are generated;
* sufficient project time remains;
* label compatibility is confirmed.

---

# 4. Locked Data Modality

The primary project uses dermoscopic images.

The following are outside the current scope:

* genomic data;
* transcriptomic data;
* proteomic data;
* metabolomic data;
* multi-omic fusion;
* histopathology-image fusion;
* electronic health record fusion;
* mobile-camera lesion diagnosis;
* clinical-text fusion.

Metadata may be used for auditing, grouping, leakage control, and subgroup evaluation.

Metadata will not automatically be used as model input unless separately approved through the decision log.

---

# 5. Locked Baseline Structure

The project must include at least one flat-classification baseline.

## Mandatory Baseline

A flat multiclass classifier trained under the same data-split and preprocessing conditions as the hierarchical framework.

## Mandatory Hierarchical Evaluation

The proposed hierarchy must include:

* independent Stage 1 evaluation;
* independent Stage 2 evaluation;
* oracle-gate evaluation;
* predicted-gate evaluation;
* end-to-end hierarchical evaluation.

## Fair-Comparison Rule

Flat and hierarchical models must use equivalent:

* image preprocessing;
* data splits;
* augmentation policy;
* model-selection procedure;
* evaluation metrics;
* random-seed policy;
* compute reporting.

Any unavoidable difference must be documented.

---

# 6. Locked Evaluation Priorities

## Primary Metrics

* Macro F1-score
* Balanced accuracy
* Per-class recall
* Per-class precision
* Confusion matrix

## Secondary Metrics

* Weighted F1-score
* Overall accuracy
* ROC-AUC
* PR-AUC
* Specificity
* Expected calibration error
* Brier score
* Inference latency
* Parameter count
* Model size
* Peak memory usage

Overall accuracy must not be used as the only measure of success.

---

# 7. Locked Model-Selection Rules

Models must be selected using internal validation data only.

The following must not influence model selection:

* internal test-set performance;
* HIBA performance;
* MRA-MIDAS performance;
* manually selected favourable test examples;
* external explainability results.

The selected checkpoint and decision threshold must be frozen before final test and external evaluation.

---

# 8. Locked Leakage-Control Rules

The project must audit:

* patient-level overlap;
* lesion-level overlap;
* duplicate images;
* near-duplicate images;
* metadata duplication;
* train-validation overlap;
* train-test overlap;
* internal-external overlap where identifiers or hashes permit.

When patient identifiers are available, splitting must be performed at patient level.

When only lesion identifiers are available, splitting must be performed at lesion level.

Image-level random splitting is allowed only when stronger identifiers are unavailable, and this limitation must be documented.

---

# 9. Locked Preprocessing Rules

Raw dataset files must never be modified directly.

The directory roles are:

```text
data/raw/        Original untouched datasets
data/interim/    Audited or temporarily transformed data
data/processed/  Final reproducible model-ready data
data/external/   External evaluation datasets
data/manifests/  Dataset manifests and split definitions
data/checksums/  Dataset integrity records
```

Any preprocessing transformation must be reproducible through scripts stored in:

```text
src/data/
```

or:

```text
scripts/
```

Manual undocumented preprocessing is prohibited.

---

# 10. Locked Reproducibility Rules

Every experiment must record:

* run identifier;
* research stage;
* dataset;
* model;
* experiment variant;
* random seed;
* configuration file;
* Git commit hash;
* split-manifest hash;
* environment information;
* training logs;
* evaluation metrics;
* predictions;
* checkpoint path;
* notes and failure status.

No experiment output may be saved using ambiguous names such as:

```text
test
new
final
final2
best_model
result
experiment1
```

---

# 11. Locked Naming Convention

Files and folders must use meaningful `snake_case` names.

## Experiment Configuration

```text
stage01_isic2019_efficientnet_b0_cross_entropy.yaml
```

## Run Directory

```text
20260722_stage01_isic2019_efficientnet_b0_cross_entropy_seed42
```

## Checkpoint

```text
stage01_isic2019_efficientnet_b0_cross_entropy_seed42_best_macro_f1.pth
```

## Prediction File

```text
stage01_isic2019_efficientnet_b0_cross_entropy_seed42_test_predictions.csv
```

## Metrics File

```text
stage01_isic2019_efficientnet_b0_cross_entropy_seed42_test_metrics.json
```

## Figure

```text
stage01_isic2019_efficientnet_b0_test_confusion_matrix.png
```

Names must identify the purpose without requiring the file to be opened.

---

# 12. Locked Non-Goals

The following are not part of the primary project contribution:

* developing a medical device;
* making clinical diagnostic claims;
* replacing dermatologist assessment;
* deploying a production application;
* building a mobile application;
* collecting new patient data;
* using private clinical data without approval;
* multi-omic modelling;
* testing many models without research justification;
* reporting only the highest-performing run;
* tuning models using external datasets.

---

# 13. Scope-Change Procedure

A material scope change includes:

* adding or removing a dataset;
* changing Stage 1, Stage 2, or Stage 3 classes;
* changing the main research objective;
* adding another data modality;
* changing the primary evaluation protocol;
* using external data for training;
* adding a new major model family;
* changing the primary metrics;
* altering the data split after test results are known.

Each proposed change must be recorded in:

`docs/06_decision_log.md`

The record must include:

1. change identifier;
2. date;
3. proposed change;
4. reason;
5. supporting evidence;
6. expected benefit;
7. expected time and compute cost;
8. risks;
9. effect on existing experiments;
10. final decision.

Experiments affected by an accepted scope change must not be silently compared with earlier incompatible experiments.

---

# 14. Scope Status

**Status:** Locked

**Scope-lock date:** 2026-07-22

**Current project phase:** Phase 00

The next phase may begin only after the remaining Phase 00 research-governance and reproducibility documents are completed.
