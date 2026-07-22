# Dataset Roles and Governance

## Purpose

This document defines the fixed responsibility of each dataset used in the project.

The same dataset must not be used simultaneously for development and independent external evaluation unless a formally documented experimental protocol permits it.

Dataset access, preprocessing, label mapping, exclusions, splitting, and integrity checks must remain reproducible.

---

# 1. Dataset Assignment Summary

| Dataset   | Project role                               |       Development use | Final evaluation use |
| --------- | ------------------------------------------ | --------------------: | -------------------: |
| ISIC 2019 | Primary development dataset                |                   Yes |        Internal test |
| EMB       | Stage 3 feasibility and severity modelling | Yes, for Stage 3 only |         Stage 3 test |
| HIBA      | Mandatory independent external evaluation  |                    No |                  Yes |
| MRA-MIDAS | Optional second external evaluation        |                    No |             Optional |

These roles are fixed unless a change is approved through:

`docs/06_decision_log.md`

---

# 2. ISIC 2019

## Fixed Role

ISIC 2019 is the primary development dataset for Stage 1 and Stage 2.

It will be used for:

* dataset auditing;
* class-definition development;
* training;
* internal validation;
* internal testing;
* flat-classification baselines;
* hierarchical-classification experiments;
* imbalance-aware experiments;
* calibration experiments;
* computational-efficiency analysis;
* internal explainability analysis.

## Stage 1 Use

ISIC 2019 labels will be mapped into:

* `malignant`
* `non_malignant`

The exact mapping from original diagnoses to these two classes must be documented in a version-controlled mapping file.

Recommended future path:

```text
configs/datasets/isic2019_stage01_label_mapping.yaml
```

## Stage 2 Use

Only samples mapped to the following malignant categories will be considered for Stage 2:

* `melanoma`
* `bcc`
* `scc`

The original dataset labels must never be silently renamed or grouped.

All included and excluded diagnoses must be documented.

Recommended future path:

```text
configs/datasets/isic2019_stage02_label_mapping.yaml
```

## Required Audits

Before training, the ISIC 2019 audit must examine:

* total number of images;
* original diagnosis distribution;
* mapped class distribution;
* missing labels;
* missing image files;
* duplicate filenames;
* duplicate image hashes;
* near-duplicate images where feasible;
* patient identifiers;
* lesion identifiers;
* patient-level overlap;
* lesion-level overlap;
* image-dimension distribution;
* image-format distribution;
* corrupted or unreadable images;
* metadata consistency;
* class imbalance;
* excluded samples and reasons.

## Split Rule

The preferred split hierarchy is:

1. patient-level split;
2. lesion-level split when patient identifiers are unavailable;
3. image-level split only when stronger grouping identifiers are unavailable.

The split method and its limitations must be explicitly documented.

## Model-Selection Boundary

The internal training split may be used for model fitting.

The internal validation split may be used for:

* early stopping;
* checkpoint selection;
* hyperparameter comparison;
* threshold selection;
* calibration fitting;
* model selection.

The internal test split must not be used for these purposes.

---

# 3. EMB Dataset

## Fixed Role

EMB is reserved for Stage 3.

Its first purpose is not model training. Its first purpose is to determine whether melanoma severity classification is scientifically feasible.

## Required Feasibility Audit

The EMB audit must determine:

* the exact meaning of every severity-related field;
* whether T-category labels are directly available;
* whether Breslow thickness is directly available;
* the measurement units;
* missing-value frequency;
* duplicate records;
* image-label consistency;
* patient identifiers;
* lesion identifiers;
* multiple images per lesion;
* number of samples per candidate severity group;
* class imbalance;
* whether a patient-level split is feasible;
* whether target groups can be defined without unsupported assumptions.

## Preferred Target

The preferred Stage 3 target is a clinically defensible T-category grouping.

## Fallback Target

When direct T-category modelling is not valid but Breslow thickness is sufficiently available, a documented Breslow-thickness grouping may be used.

The thresholds must be defined in a configuration file rather than hardcoded inside training code.

Recommended future path:

```text
configs/datasets/emb_stage03_severity_mapping.yaml
```

## Stage 3 Stop Rule

Stage 3 must not proceed when:

* labels are ambiguous;
* severity groups require unsupported inference;
* sample counts are critically insufficient;
* leakage-free splitting is impossible;
* missingness makes the target unreliable;
* labels and images cannot be matched confidently.

In that case, the project will report Stage 3 as a feasibility limitation.

A scientifically valid limitation is preferable to an unreliable experiment.

---

# 4. HIBA Dataset

## Fixed Role

HIBA is the mandatory independent external evaluation dataset.

Its purpose is to measure whether a model developed using ISIC 2019 generalises to independently collected data.

## Permitted Uses

HIBA may be used for:

* frozen-model inference;
* external classwise evaluation;
* domain-shift analysis;
* external calibration analysis;
* external error analysis;
* external explainability analysis;
* comparison with internal test performance.

## Prohibited Uses

HIBA must not be used for:

* model training;
* fine-tuning before the primary external evaluation;
* architecture selection;
* augmentation selection;
* preprocessing selection based on performance;
* checkpoint selection;
* hyperparameter tuning;
* decision-threshold selection;
* calibration fitting before the primary external result;
* repeated trial-and-error optimisation.

## Zero-Shot Evaluation Rule

The primary HIBA result must be produced using a frozen pipeline selected entirely from internal development data.

The following must be frozen before HIBA evaluation:

* model architecture;
* model checkpoint;
* preprocessing pipeline;
* image size;
* label mapping;
* decision threshold;
* calibration method;
* inference configuration.

## External Label Compatibility

HIBA labels must be audited before evaluation.

The audit must identify:

* directly compatible classes;
* incompatible classes;
* ambiguous labels;
* excluded samples;
* mapping assumptions;
* class prevalence;
* missing metadata;
* image modality differences.

No incompatible label may be forced into an existing class merely to increase sample size.

## Adaptation Rule

Any later HIBA-based fine-tuning or domain adaptation must be treated as a separate secondary experiment.

It must not replace or overwrite the original zero-shot external result.

---

# 5. MRA-MIDAS Dataset

## Fixed Role

MRA-MIDAS is an optional second external evaluation dataset.

It is not part of the minimum required experimental pipeline.

## Activation Conditions

MRA-MIDAS may be activated only after:

* ISIC 2019 data auditing is complete;
* Stage 1 and Stage 2 baselines are complete;
* hierarchical evaluation is complete;
* HIBA zero-shot evaluation is complete;
* mandatory ablations are complete;
* required tables and figures are generated;
* sufficient time and compute remain.

## Evaluation Boundary

When activated, MRA-MIDAS must follow the same external-evaluation restrictions used for HIBA.

It must not be used to select or improve the primary model before its initial external evaluation.

---

# 6. Raw and Derived Data Separation

Raw dataset files must remain unchanged.

The expected directory responsibilities are:

```text
data/raw/
    Original development datasets

data/external/
    Original external evaluation datasets

data/interim/
    Audited, extracted, normalised, or temporarily transformed data

data/processed/
    Final reproducible model-ready data

data/manifests/
    Dataset inventories, mappings, exclusions, and split assignments

data/checksums/
    Dataset and file-integrity records
```

Recommended dataset paths:

```text
data/raw/isic2019/
data/raw/emb/

data/external/hiba/
data/external/mra_midas/
```

These directories remain excluded from Git.

Only lightweight manifests, schemas, mappings, and documentation should be committed.

---

# 7. Dataset Manifest Requirements

Every dataset must have a machine-readable manifest.

Recommended filenames:

```text
data/manifests/isic2019_dataset_manifest.csv
data/manifests/emb_dataset_manifest.csv
data/manifests/hiba_dataset_manifest.csv
data/manifests/mra_midas_dataset_manifest.csv
```

Each manifest should contain fields such as:

```text
dataset_name
image_id
image_relative_path
patient_id
lesion_id
original_label
mapped_stage01_label
mapped_stage02_label
mapped_stage03_label
split
include
exclusion_reason
image_sha256
source_reference
```

Fields that are unavailable should remain empty rather than being invented.

---

# 8. Dataset Registry Requirements

Dataset-level information must be recorded in:

```text
configs/dataset_registry.yaml
```

For each dataset, record:

* official dataset name;
* project identifier;
* version or release;
* source page;
* access date;
* licence or usage terms;
* citation;
* assigned project role;
* local raw-data path;
* manifest path;
* checksum record;
* current audit status;
* notes and restrictions.

Recommended project identifiers:

```text
isic2019
emb
hiba
mra_midas
```

These identifiers must be used consistently across configuration files, scripts, experiments, reports, and output filenames.

---

# 9. Label-Mapping Governance

Original labels must always be preserved.

Mapped labels must be stored in separate fields.

Every mapping must be:

* explicit;
* version controlled;
* reproducible;
* supported by dataset documentation;
* reviewed before training;
* consistent across experiments.

Label mapping must not be hidden inside notebooks or model-training scripts.

Recommended location:

```text
configs/datasets/
```

Recommended files:

```text
isic2019_stage01_label_mapping.yaml
isic2019_stage02_label_mapping.yaml
emb_stage03_severity_mapping.yaml
hiba_external_label_mapping.yaml
mra_midas_external_label_mapping.yaml
```

---

# 10. Dataset Exclusion Governance

Every excluded record must have a documented reason.

Valid exclusion reasons may include:

* missing image;
* corrupted image;
* missing required label;
* unsupported diagnosis;
* ambiguous mapping;
* duplicate record;
* duplicate image;
* confirmed cross-split leakage;
* incompatible image modality;
* invalid metadata;
* failed integrity check.

Exclusions must never be performed manually without updating the corresponding manifest.

---

# 11. Dataset Integrity and Checksums

Dataset integrity should be recorded using SHA-256 checksums.

Recommended checksum files:

```text
data/checksums/isic2019_sha256.txt
data/checksums/emb_sha256.txt
data/checksums/hiba_sha256.txt
data/checksums/mra_midas_sha256.txt
```

Checksums help confirm that:

* files were copied correctly;
* Azure and local copies are identical;
* raw data did not change between experiments;
* manifests correspond to the intended dataset version.

---

# 12. Data Leakage Prevention

Before any final experiment, the project must investigate:

* identical files across splits;
* identical files across datasets;
* near-duplicate images;
* repeated lesions;
* repeated patients;
* multiple crops from the same source image;
* filenames that expose class labels;
* metadata fields that directly reveal targets;
* preprocessing performed using test-set statistics.

When overlap is detected, the resolution and affected records must be documented.

---

# 13. External Evaluation Contamination Rule

Results from HIBA or MRA-MIDAS may reveal weaknesses, but they must not be used to repeatedly modify the primary model and re-report the result as if it were untouched external validation.

When external results motivate a change:

1. preserve the original external result;
2. document the observation;
3. record the proposed change in the decision log;
4. label the modified experiment as post-external adaptation;
5. avoid presenting it as the primary zero-shot result.

---

# 14. Dataset Naming Rules

Dataset-related files must use descriptive names.

Accepted examples:

```text
isic2019_dataset_manifest.csv
isic2019_patient_level_split_seed42.csv
isic2019_stage02_class_distribution.csv
emb_stage03_label_feasibility_report.md
hiba_external_label_mapping.yaml
hiba_zero_shot_evaluation_predictions.csv
```

Avoid names such as:

```text
data.csv
new_data.csv
final_dataset.csv
labels2.csv
split_new.csv
external_test.csv
```

Names must communicate the dataset, purpose, stage, and relevant version or seed.

---

# 15. Dataset Status at Phase 00

At the end of Phase 00:

* dataset responsibilities are defined;
* evaluation boundaries are frozen;
* raw data has not yet been used for uncontrolled experimentation;
* label mappings remain pending formal audits;
* class counts remain unconfirmed until manifests are generated;
* Stage 3 remains feasibility-dependent;
* HIBA remains isolated from development decisions;
* MRA-MIDAS remains optional.

The next data-related phase must begin with dataset registration and audit, not model training.
