# Project Charter

## Working Title

**A Lightweight Conditional Hierarchical Framework for Dermoscopic Skin Cancer Classification with External Evaluation and Explainability**

## Research Problem

Dermoscopic skin-lesion classification is commonly treated as a flat multiclass problem. However, flat classifiers may hide clinically important failure patterns, particularly when malignant classes are imbalanced, visually similar, or underrepresented.

A model may achieve high overall accuracy while performing poorly on rare but important malignant classes. Performance measured only on an internal test set may also fail to represent generalisation to independently collected clinical datasets.

## Proposed Research Direction

This project will develop and evaluate a conditional hierarchical image-classification framework.

### Stage 1

Binary classification:

* Malignant
* Non-malignant

### Stage 2

Classification of major malignant lesion categories:

* Melanoma
* Basal cell carcinoma
* Squamous cell carcinoma

Stage 2 will run only when Stage 1 predicts that a lesion is malignant.

### Stage 3

Melanoma severity classification using one of the following:

* T-category grouping
* Breslow-thickness grouping

The final Stage 3 target will be selected only after auditing the available EMB dataset labels.

## Main Research Contribution

The intended contribution is not simply the use of a new deep-learning backbone.

The main contribution is the design and rigorous evaluation of a lightweight conditional hierarchical framework that includes:

1. comparison between flat and hierarchical classification;
2. measurement of error propagation between hierarchy stages;
3. imbalance-aware learning;
4. independent external evaluation;
5. calibration and confidence analysis;
6. explainability analysis;
7. computational-efficiency evaluation.

## Dataset Roles

| Dataset   | Role                                                  |
| --------- | ----------------------------------------------------- |
| ISIC 2019 | Primary development, validation, and internal testing |
| EMB       | Stage 3 feasibility analysis and severity modelling   |
| HIBA      | Mandatory independent external evaluation             |
| MRA-MIDAS | Optional additional external evaluation               |

## Research Boundaries

This project does not aim to:

* develop a clinically deployable diagnostic system;
* replace dermatologist judgement;
* perform multi-omic data fusion;
* develop a mobile application during the main research stages;
* combine unrelated datasets without documented label mapping;
* use external datasets for model selection.

## Source-of-Truth Rule

The permanent source of truth is the local project directory:

`F:\Research\Final Year\Skin-Cancer-Hierarchical-Classification`

Azure GPU infrastructure will be used only when GPU computation is required.

All code, configurations, manifests, checkpoints, logs, predictions, metrics, figures, and reports produced on Azure must be copied back to the local project directory.

## Scope-Control Rule

Any material change to the research scope must be documented in:

`docs/06_decision_log.md`

Each proposed change must include:

* reason for the change;
* supporting evidence;
* expected benefit;
* additional time and compute requirements;
* effect on experiment comparability;
* final acceptance or rejection decision.

## Current Status

Current phase:

**Phase 00 — Foundation, scope definition, and reproducibility preparation**
