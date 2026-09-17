# Skin Cancer Hierarchical Classification

Research repository for the paper **Routing Errors Limit Hierarchical Skin Lesion Classification**.

## Main experiment

The paper compares seven CNN backbones under two systems:

- Flat four-class classification.
- Shared-Hard hierarchical classification with predicted-gate and oracle-gate diagnostics.

The backbones are DenseNet121, DenseNet169, ResNet50, MobileNetV3-Large,
EfficientNet-B0, EfficientNet-B2, and EfficientNet-B3. The primary dataset is
ISIC 2019 with four labels: `non_malignant`, `melanoma`, `bcc`, and `scc`.
HIBA is used only for frozen zero-shot external evaluation.

## Repository layout

- `src/`: data, models, training, evaluation, and utility code.
- `configs/paper/`: final-paper Flat, Shared-Hard, and evaluation configs.
- `configs/extensions/`: actual post-paper research configurations.
- `configs/protocols/`: reusable frozen protocols.
- `data/manifests/`: dataset manifests and audit metadata; raw data is local-only.
- `experiments/`: `paper_registry.csv` and `extension_registry.csv`.
- `results/paper/`: canonical ISIC, HIBA, routing, and statistical evidence.
- `results/extensions/`: non-paper extension evidence.
- `paper/routing_errors_limit_hierarchical_skin_lesion_classification/`: source, PDF, figures, and publication files.
- `tests/`: automated tests.
- `docs/paper/` and `docs/extensions/`: paper documentation, provenance, and Stage-3 records.

## Repository validation

Run `python -m pytest` for the full suite. Pytest collects only `tests/`, so
evaluation-script helper functions are not mistaken for tests.

For repository cleanup without reprocessing stored scientific evidence, run
`python -m pytest -m "not scientific_recomputation"`. The marked integration
tests remain available for a separately authorized evidence-validation run.
The remaining tests use synthetic inputs or read existing configs and outputs;
they do not launch research training or internal/HIBA evaluation jobs.

## Reproducibility

The canonical paper evidence is stored as lightweight metrics, predictions,
tables, configs, and provenance records. Raw datasets, most training logs, and
model checkpoints are not committed; checkpoints remain in ignored local run
directories when available.

The repository contains both the current ISIC manifest and historical evidence
for the paper's 3,668-image evaluation population. Their relationship is not
yet resolved. Do not substitute the current manifest for the paper split until
the recorded Gate 06D and Phase 07 hashes are reconciled.

## Current research

New work belongs under `configs/extensions/`, `results/extensions/`, and
extension registry entries. Shared-Soft, Dedicated-Hard, Dedicated-Soft,
probability fusion, multi-seed, transformer, and calibration experiments are
extensions and must not be presented as part of the original paper baseline
unless separately completed and documented.

## Status

The paper baseline contains seven Flat and seven Shared-Hard models, locked
ISIC internal comparisons, routing-loss and paired statistical analysis, and
seven-backbone HIBA evaluation evidence. No commit or push is implied by local
cleanup changes.
