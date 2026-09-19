# PAUL 2026 Swin-T multi-seed protocol

## Scope and research question

Does the routing-associated performance pattern observed across the seven CNN
backbones persist under a Swin-T representation, and are the Swin-T results
stable across three predeclared training seeds?

This is a new extension. The original seven-CNN paper evidence remains immutable
and is not rerun or overwritten. Historical architecture registries are unchanged.
This protocol declares six planned training runs; it does not execute training,
dataset processing, evaluation, or GPU work. Training integration is a later task;
these configs do not add Swin-T to historical training entry points.

## Systems and fixed partitions

Use torchvision Swin-T with ImageNet pretrained weights, 224x224 input, dropout
0.2, ImageNet normalization, and the locked moderate training augmentation.
Validation preprocessing is deterministic resize to 256 and center crop to 224.

The systems are Flat Swin-T and Shared-Hard Swin-T, each trained with seeds
**42, 123, and 2026**. Oracle routing is a diagnostic from the same trained
Shared-Hard checkpoint for each seed, requiring no additional training.

**TRAINING SEED != DATASET SPLIT SEED.** All six runs use the same frozen data
partitions. These manifest files stay identical for every training seed:

- ISIC: `data/manifests/isic2019_train_val_test_split_seed42.csv`
- Stage-3: `data/manifests/emb_stage03_dermoscopic_split_seed42.csv`

Flat uses the ISIC manifest; Shared-Hard uses both manifests. There is no
resplitting or manifest regeneration. `experiment.seed` changes only training
randomness such as initialization, augmentation, shuffling, and RNG-controlled
operations. The seed42 suffix denotes the frozen dataset split, even when the
training seed is 123 or 2026.

Flat class indices are non_malignant=0, melanoma=1, bcc=2, scc=3.
Shared task indices are:

- Task 1: non_malignant=0, malignant=1.
- Task 2: melanoma=0, bcc=1, scc=2.
- Task 3: Tis=0, T1=1, T2=2, T3=3, T4=4.

## Locked training semantics

Flat uses `configs/paper/flat/efficientnet_b0.yaml` as its scientific template:
cross entropy, no weighted sampler, no class weights or focal loss, batch size
64, four loader workers, AMP enabled, AdamW with learning rate 0.0003 and weight
decay 0.0001, cosine annealing with minimum learning rate 0.000001, at most 30
epochs, and early stopping patience 7. Select the checkpoint by the highest
validation macro-F1. Internal-test access, loader construction, and influence on
selection are forbidden during training.

`configs/paper/shared_hard/shared_three_task.yaml` is the exact scientific source
of truth for Shared-Hard. Only the architecture, training seed, and extension
run/output metadata change. Preserve all task mappings, masks, missing-target
semantics, and source counts. Use natural concatenation with shuffling each
epoch, no weighted sampler, no Stage-3 oversampling, and no forced source mixing.
One shared encoder pass produces features for all three heads.

The locked shared losses are:

- Task 1: cross entropy.
- Task 2: class-balanced focal loss, beta=0.9999, gamma=2.0, with class weights
  melanoma=0.3485376280807543, bcc=0.4553489231324597,
  scc=2.196113448786786. Retain the historical training-only effective-number
  weight provenance and normalization.
- Task 3: weighted cross entropy, with class weights Tis=0.063475735584673,
  T1=0.12246677245955931, T2=0.682845034319967,
  T3=2.253388613255891, T4=1.8778238443799093. Retain the historical
  training-only inverse-frequency weight provenance and normalization.
- lambda_task1=lambda_task2=lambda_task3=1. Preserve weighted active-task mean
  normalization, selecting active samples before loss, and skipping zero-active
  tasks without denominator contribution.

Shared training retains batch size 64, four workers, AMP, AdamW learning rate
0.0003 and weight decay 0.0001, cosine annealing with T_max=30 and minimum
learning rate 0.000001, at most 30 epochs, and patience 7. Checkpoint selection
uses the highest arithmetic mean of Task1/Task2/Task3 validation macro-F1, with
the historical per-task validation cohorts. No internal-test loader is
constructed during training, and internal-test data cannot influence selection.

## Diagnostics and reporting

Oracle routing is diagnostic only. It uses the same selected Shared-Hard
checkpoint as hard routing for that seed; it cannot influence checkpoint
selection or tuning and does not create an additional training run.

HIBA evaluation is strict zero-shot only: no tuning, no calibration fitting,
and no model selection from HIBA. Internal-test and HIBA evaluation occur only
after checkpoint selection, in a later evaluation task.

For Flat, Hard, and Oracle, report each of seed 42, seed 123, and seed 2026,
the mean macro-F1, and the sample standard deviation across the three seeds
(ddof=1), separately for each evaluation cohort. Do not omit unfavorable seeds.
Disclose failed or incomplete runs; do not silently replace seeds or report an
incomplete set as the planned three-seed result. Do not claim statistical
superiority solely from three training seeds.

## Planned runs

All configs are under `configs/extensions/paul2026_swin/`; all registry entries
are `PLANNED`. Extension outputs belong under `runs/extensions/paul2026_swin/`
and must use the distinct run identity. No paper outputs may be overwritten.

| Run name | Config |
| --- | --- |
| paul2026_flat_swin_t_seed42 | flat_seed42.yaml |
| paul2026_flat_swin_t_seed123 | flat_seed123.yaml |
| paul2026_flat_swin_t_seed2026 | flat_seed2026.yaml |
| paul2026_shared_hard_swin_t_seed42 | shared_hard_seed42.yaml |
| paul2026_shared_hard_swin_t_seed123 | shared_hard_seed123.yaml |
| paul2026_shared_hard_swin_t_seed2026 | shared_hard_seed2026.yaml |
