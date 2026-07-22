# Reproducibility and Experiment Management Protocol

## Purpose

This document defines how experiments, configurations, datasets, outputs, environments, and decisions must be recorded.

The objective is to ensure that every important result can be reproduced, inspected, compared, and reused without depending on memory, manually copied values, or undocumented steps.

---

# 1. Permanent Source of Truth

The permanent project directory is:

`F:\Research\Final Year\Skin-Cancer-Hierarchical-Classification`

This local directory is the authoritative source for:

* source code;
* configuration files;
* dataset manifests;
* split definitions;
* documentation;
* experiment registries;
* final checkpoints;
* prediction files;
* metrics;
* tables;
* figures;
* reports;
* environment records.

Azure GPU infrastructure is temporary computational infrastructure only.

No important project artifact may exist only on the Azure virtual machine.

---

# 2. Local and Azure Responsibilities

## Local Environment

The local project directory is responsible for:

* code development;
* configuration management;
* documentation;
* Git version control;
* dataset manifests;
* experiment planning;
* final result storage;
* report writing;
* long-term backups.

## Azure Environment

The Azure NCasT4_v3 virtual machine may be used for:

* GPU training;
* GPU inference;
* computationally expensive explainability;
* large-scale feature extraction;
* performance benchmarking requiring GPU resources.

## Mandatory Transfer Rule

After every meaningful Azure run, the following artifacts must be copied back to the local project directory:

* experiment configuration;
* environment export;
* Git commit hash;
* dataset-manifest hash;
* split-manifest hash;
* training log;
* evaluation log;
* selected checkpoint;
* final metrics;
* prediction file;
* generated figures;
* generated tables;
* run notes;
* failure information when the run fails.

An experiment is not considered complete until its required artifacts are stored locally.

---

# 3. Reproducible Workflow

Every experiment must follow this order:

1. define the research purpose;
2. create a version-controlled configuration;
3. confirm the dataset manifest;
4. confirm the split manifest;
5. record the Git commit;
6. launch the experiment;
7. save all outputs in a unique run directory;
8. evaluate using a reproducible script;
9. register the result;
10. copy Azure artifacts back to the local project;
11. commit relevant source, configuration, and documentation changes.

Manual experimental changes made only through terminal arguments should be avoided.

Important parameters must be stored in configuration files.

---

# 4. Experiment Naming Convention

Every experiment run must use a meaningful unique identifier.

## Required Format

```text
YYYYMMDD_HHMM_stageXX_dataset_model_variant_seedNN
```

## Example

```text
20260722_1530_stage01_isic2019_efficientnet_b0_cross_entropy_seed42
```

## Components

* `YYYYMMDD`: experiment date;
* `HHMM`: experiment start time using 24-hour format;
* `stageXX`: research stage;
* `dataset`: main dataset identifier;
* `model`: architecture identifier;
* `variant`: experimental method;
* `seedNN`: random seed.

## Approved Stage Identifiers

```text
stage00
stage01
stage02
stage03
end_to_end
external
explainability
efficiency
```

## Approved Dataset Identifiers

```text
isic2019
emb
hiba
mra_midas
```

## Naming Rule

Run identifiers must not contain spaces.

Use lowercase `snake_case`.

Avoid ambiguous names such as:

```text
test_run
new_model
final_run
best_experiment
experiment_01
model_new
```

---

# 5. Experiment Configuration Naming

Each experiment configuration must have a descriptive filename.

## Recommended Format

```text
stageXX_dataset_model_variant.yaml
```

## Examples

```text
stage01_isic2019_efficientnet_b0_cross_entropy.yaml
stage01_isic2019_efficientnet_b0_weighted_cross_entropy.yaml
stage02_isic2019_convnext_tiny_class_balanced_focal.yaml
stage03_emb_efficientnet_b0_breslow_grouping.yaml
external_hiba_hierarchical_zero_shot.yaml
```

Configuration files must be stored in:

```text
configs/experiments/
```

A configuration must never be overwritten after it has produced a reported experiment.

When a meaningful change is required, create a new descriptive configuration.

---

# 6. Run Directory Structure

Each experiment must create one unique run directory under:

```text
experiments/runs/
```

Recommended structure:

```text
experiments/runs/
└── 20260722_1530_stage01_isic2019_efficientnet_b0_cross_entropy_seed42/
    ├── config/
    │   └── resolved_config.yaml
    ├── environment/
    │   ├── python_version.txt
    │   ├── package_versions.txt
    │   ├── hardware_information.txt
    │   └── git_information.txt
    ├── logs/
    │   ├── training.log
    │   └── evaluation.log
    ├── checkpoints/
    │   ├── best_macro_f1.pth
    │   └── last_epoch.pth
    ├── predictions/
    │   ├── validation_predictions.csv
    │   └── test_predictions.csv
    ├── metrics/
    │   ├── validation_metrics.json
    │   ├── test_metrics.json
    │   └── training_history.csv
    ├── figures/
    │   ├── training_curves.png
    │   └── confusion_matrix.png
    ├── tables/
    │   └── classwise_metrics.csv
    ├── metadata/
    │   ├── dataset_manifest_hash.txt
    │   ├── split_manifest_hash.txt
    │   └── run_metadata.json
    └── notes/
        └── run_notes.md
```

Temporary files should not remain in completed run directories.

---

# 7. Required Run Metadata

Every run must record:

* run identifier;
* start time;
* completion time;
* project phase;
* research stage;
* dataset;
* dataset version;
* dataset-manifest path;
* dataset-manifest hash;
* split-manifest path;
* split-manifest hash;
* model architecture;
* pretrained-weight source;
* image resolution;
* batch size;
* optimiser;
* learning rate;
* scheduler;
* loss function;
* augmentation policy;
* random seed;
* number of epochs;
* early-stopping configuration;
* checkpoint-selection metric;
* threshold-selection method;
* calibration method;
* hardware;
* Python version;
* package versions;
* Git branch;
* Git commit hash;
* run status;
* failure reason when applicable.

This information should be stored in:

```text
metadata/run_metadata.json
```

---

# 8. Random Seed Policy

Random seeds must be explicitly configured.

The initial standard seed set is:

```text
42
123
2026
```

During early development, seed `42` may be used for debugging and pipeline verification.

Final major comparisons should preferably use all three seeds, subject to compute availability.

The seed must control, where supported:

* Python randomness;
* NumPy randomness;
* deep-learning framework randomness;
* data-split generation;
* data-loader shuffling;
* augmentation randomness;
* weight initialisation.

The exact seed must appear in:

* the configuration;
* the run identifier;
* the experiment registry;
* the run metadata;
* prediction and metric filenames when they are exported outside the run directory.

---

# 9. Determinism Policy

Deterministic execution should be enabled where reasonably possible.

The project must document:

* deterministic framework settings;
* operations that remain nondeterministic;
* performance costs caused by deterministic execution;
* differences between local and Azure hardware;
* library limitations.

Perfect numerical identity across different hardware is not guaranteed.

The goal is procedural reproducibility and comparable results, not an unsupported claim of bitwise-identical execution.

---

# 10. Environment Recording

Every important experiment must record its software and hardware environment.

## Required Software Information

* operating system;
* Python version;
* CUDA version;
* cuDNN version when available;
* PyTorch or TensorFlow version;
* torchvision or equivalent version;
* NumPy version;
* pandas version;
* scikit-learn version;
* image-processing library versions;
* explainability-library versions.

## Required Hardware Information

* CPU model;
* system RAM;
* GPU model;
* GPU memory;
* storage environment;
* Azure VM type when applicable.

## Recommended Files

```text
environment/python_version.txt
environment/package_versions.txt
environment/hardware_information.txt
environment/git_information.txt
```

For Python environments, save one of the following:

```text
requirements.txt
environment.yml
pyproject.toml
```

The project should later standardise on one primary dependency-management method.

---

# 11. Git Reproducibility Rules

Before launching an important experiment:

```powershell
git status
```

The preferred state is:

```text
nothing to commit, working tree clean
```

Every run must record:

```powershell
git rev-parse HEAD
git branch --show-current
```

Experiments may be launched with uncommitted changes only during temporary debugging.

Such runs must be marked:

```text
development_only
```

They must not be used as final reported evidence.

Configurations and source code used for final experiments must be committed.

---

# 12. Dataset and Split Integrity

Every experiment must reference a fixed dataset manifest and split manifest.

## Dataset Manifest Example

```text
data/manifests/isic2019_dataset_manifest.csv
```

## Split Manifest Example

```text
data/manifests/isic2019_patient_level_split_seed42.csv
```

The manifest hash must be recorded before training.

Recommended algorithm:

```text
SHA-256
```

If the manifest changes, the new experiments must use a new manifest version or clearly updated filename.

Experiments produced from incompatible manifests must not be compared silently.

---

# 13. Checkpoint Naming Convention

Checkpoint names must explain what they contain.

## Recommended Names

```text
best_macro_f1.pth
best_balanced_accuracy.pth
best_validation_loss.pth
last_epoch.pth
```

When copied outside the run directory, use the full context:

```text
stage01_isic2019_efficientnet_b0_cross_entropy_seed42_best_macro_f1.pth
```

Avoid:

```text
model.pth
best.pth
final.pth
new_model.pth
```

The selected checkpoint metric must be specified before final evaluation.

---

# 14. Prediction File Requirements

Predictions must be saved for final experiments.

Recommended fields:

```text
dataset_name
split
image_id
patient_id
lesion_id
true_label
predicted_label
predicted_probability
class_probability_01
class_probability_02
class_probability_03
run_id
checkpoint_name
```

Prediction files allow:

* recalculation of metrics;
* confidence analysis;
* calibration analysis;
* error analysis;
* subgroup analysis;
* explainability-case selection;
* statistical comparison.

Metrics should not exist only as terminal output.

---

# 15. Metric Storage Requirements

Metrics must be stored in machine-readable files.

Recommended formats:

```text
JSON
CSV
```

## Required Metric Outputs

```text
validation_metrics.json
test_metrics.json
classwise_metrics.csv
training_history.csv
```

The final report should not depend on manually copied metric values.

Tables should be generated from stored result files whenever practical.

---

# 16. Experiment Registry

All meaningful experiments must be registered in:

```text
experiments/experiment_registry.csv
```

Recommended columns:

```text
run_id
phase
research_stage
dataset
model
variant
seed
status
git_commit
config_path
dataset_manifest
dataset_manifest_hash
split_manifest
split_manifest_hash
checkpoint_metric
checkpoint_path
primary_metric
primary_value
run_directory
notes
```

## Run Status Values

Use consistent statuses:

```text
planned
running
completed
failed
stopped
invalid
archived
```

A failed or invalid experiment should remain recorded rather than being silently deleted.

---

# 17. Failure Documentation

Failed experiments can provide useful information.

When a run fails, record:

* failure time;
* error message;
* affected epoch or processing step;
* suspected cause;
* confirmed cause when known;
* whether partial outputs are valid;
* corrective action;
* whether the run should be repeated.

Recommended location:

```text
notes/run_notes.md
```

Do not reuse the same run directory for the corrected experiment.

Create a new run identifier.

---

# 18. Model Selection Protocol

The model-selection metric must be defined before final testing.

Primary checkpoint-selection candidates are:

* validation macro F1-score;
* validation balanced accuracy.

The internal test set and external datasets must not be used to select:

* checkpoints;
* hyperparameters;
* model architecture;
* thresholds;
* augmentation policies;
* loss functions.

The final selected model must be frozen before test and external evaluation.

---

# 19. Threshold and Calibration Protocol

Decision thresholds must be selected using internal validation data only.

Any calibration method must record:

* calibration dataset;
* calibration technique;
* fitted parameters;
* pre-calibration metrics;
* post-calibration metrics.

External datasets must not be used to fit the primary calibration method.

External calibration fitting may be performed only as a separately labelled secondary adaptation experiment.

---

# 20. Statistical Comparison Protocol

Final model comparisons should use stored prediction files.

Where feasible, report:

* multiple random seeds;
* mean performance;
* standard deviation;
* bootstrap confidence intervals;
* paired comparison using the same test samples;
* patient-level resampling when patient identifiers are available.

A single favourable run must not be presented as proof of general superiority.

---

# 21. Figure and Table Naming

Generated figures and tables must use descriptive names.

## Figure Examples

```text
stage01_isic2019_efficientnet_b0_training_curves.png
stage02_isic2019_hierarchical_test_confusion_matrix.png
external_hiba_zero_shot_reliability_diagram.png
```

## Table Examples

```text
stage01_isic2019_classwise_metrics.csv
stage02_flat_vs_hierarchical_comparison.csv
external_hiba_internal_external_performance_gap.csv
```

Avoid:

```text
figure1.png
graph_new.png
final_table.csv
result2.csv
```

---

# 22. Notebook Governance

Notebooks may be used for:

* exploratory data analysis;
* visual inspection;
* temporary investigation;
* result presentation.

Core reproducible processing must not exist only inside notebooks.

Reusable logic must be moved into:

```text
src/
```

or:

```text
scripts/
```

Notebook filenames must be descriptive.

Examples:

```text
isic2019_class_distribution_exploration.ipynb
emb_stage03_label_feasibility_exploration.ipynb
hiba_domain_shift_visual_analysis.ipynb
```

Avoid:

```text
test.ipynb
new.ipynb
final.ipynb
untitled.ipynb
```

---

# 23. Working Directory Cleanliness

The project root must remain clean.

Do not place the following directly in the root:

* datasets;
* checkpoints;
* screenshots;
* temporary scripts;
* downloaded archives;
* generated figures;
* experiment logs;
* copied result files;
* unnamed notebooks;
* backup files.

Temporary work must be placed in an appropriate ignored directory and removed when no longer needed.

Before each major commit, review:

```powershell
git status
Get-ChildItem
```

Every file must have a clear purpose and correct location.

---

# 24. Session Handover Records

At the end of each major working session, record:

* completed tasks;
* modified files;
* decisions;
* experiment results;
* problems;
* exact next task;
* required commands.

Use:

```text
reports/session_handover_template.md
```

Recommended session filename:

```text
reports/handovers/20260722_phase00_reproducibility_setup.md
```

This prevents loss of context between work sessions.

---

# 25. Backup Policy

Important source code, configurations, documentation, and lightweight result summaries must be backed up through Git.

Large artifacts that cannot be committed should have at least one additional controlled copy.

Important checkpoints and final experiment outputs should exist in:

1. the local master project directory;
2. a separate backup location or approved cloud storage.

Raw datasets should follow their licence and access restrictions.

---

# 26. Minimum Completion Requirements for an Experiment

An experiment is considered reproducible only when it has:

* a unique run identifier;
* a committed configuration;
* a recorded Git commit;
* fixed dataset and split manifests;
* recorded random seed;
* environment information;
* logs;
* checkpoint;
* machine-readable metrics;
* prediction file;
* experiment-registry entry;
* run notes;
* locally stored final artifacts.

Runs missing these requirements may be useful for debugging but must not be treated as final experiments.

---

# 27. Protocol Status

**Status:** Active

**Effective date:** 2026-07-22

**Current phase:** Phase 00

This protocol applies to all future data processing, model training, evaluation, explainability, efficiency, and external-validation experiments.
