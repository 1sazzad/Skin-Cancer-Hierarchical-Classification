# Gate 03B — Shared Three-Task Hierarchical Baseline Design and Implementation

## Verdict and scope

Gate 03B establishes the local implementation and frozen configuration for the
new shared three-task baseline. No training, sanity run, validation inference,
internal-test access, GPU execution, efficiency measurement, commit, or push
was performed.

This campaign uses only unambiguous new paths:

- `configs/experiments/phase03_shared_three_task_*.yaml`
- `reports/phase03_shared_three_task/`
- later execution output under `runs/phase03_shared_three_task/`

Historical `phase03` reports and configs remain unchanged.

## Frozen architecture

The model uses an ImageNet-pretrained EfficientNet-B0 encoder to produce one
shared pooled feature vector from one encoder forward. Three freshly
initialized heads consume that same representation:

| Task | Head | Logits | Class order |
|---|---|---:|---|
| Task 1 | Dropout(0.2) → Linear | 2 | non_malignant, malignant |
| Task 2 | Dropout(0.2) → Linear | 3 | melanoma, bcc, scc |
| Task 3 | Dropout(0.2) → Linear | 5 | Tis, T1, T2, T3, T4 |

Historical Stage-1, Stage-2, and Stage-3 checkpoints are not loaded. They
remain comparison evidence only.

## Dataset and mask behavior

The training-safe data path constructs only train and validation datasets. It
does not reuse the Phase-05 hierarchical internal-test inference loader and
does not construct an internal-test loader.

| Source sample | Targets | Task mask |
|---|---|---|
| ISIC non-malignant | `[0, -100, -100]` | `[1, 0, 0]` |
| ISIC melanoma | `[1, 0, -100]` | `[1, 1, 0]` |
| ISIC BCC | `[1, 1, -100]` | `[1, 1, 0]` |
| ISIC SCC | `[1, 2, -100]` | `[1, 1, 0]` |
| ISIC-derived Stage-3 | `[-100, -100, T]` | `[0, 0, 1]` |

`-100` is an explicit unavailable-target sentinel, never a valid class.
Losses select active rows before targets reach a criterion.

The natural training pool concatenates 17,124 ISIC training samples and 594
ISIC-derived melanoma T-category training samples, for 17,718 samples. It is
shuffled each epoch with seed 42. There is no oversampling, weighted sampler,
or forced source-balanced batch mixing.

Validation cohorts are:

- Task 1: all ISIC validation samples;
- Task 2: malignant ISIC validation samples only;
- Task 3: the ISIC-derived Stage-3 validation split.

Frozen transforms reuse the existing 224×224 moderate train augmentation and
deterministic resize/center-crop validation preprocessing with ImageNet
normalization.

## Masked losses

| Task | Loss | Weight source |
|---|---|---|
| Task 1 | Cross-Entropy | Unweighted |
| Task 2 | Class-Balanced Focal Loss, beta 0.9999, gamma 2.0 | ISIC train counts only |
| Task 3 | Weighted Cross-Entropy | Stage-3 train counts only |

The existing `ClassBalancedFocalLoss` is reused. Task-2 effective-number
weights use training counts `3164/2327/440`. Task-3 inverse-frequency weights
use training counts `355/184/33/10/12`; validation and test labels do not
contribute to weights.

Each task criterion receives only rows whose boolean task mask is active. A
task with zero active samples is skipped and supplies neither a fake loss nor a
denominator term. With frozen lambdas `1/1/1`, total batch loss is the mean of
the active task mean losses.

## Training and validation control

- AdamW, learning rate `3e-4`, weight decay `1e-4`
- CosineAnnealingLR, `T_max=30`, `eta_min=1e-6`
- Maximum 30 epochs
- Early-stopping patience 7
- CUDA AMP supported for later GPU execution
- Primary metric per task: validation Macro-F1
- Shared score: arithmetic mean of the three task validation Macro-F1 values
- Checkpoint selection: maximize shared validation score

The implementation provides a natural-pool training epoch, task-specific
validation filtering, maximize-mode early stopping, optimizer/scheduler
construction, and checkpoint/history payload utilities.

The primary checkpoint filename is `best_checkpoint.pt`. Its payload supports
model, optimizer, and scheduler states; epoch; shared score; three task
Macro-F1 values; seed; model/config metadata; class mappings; task losses; and
task-mask policy.

History support includes:

- `train_total_loss`
- `train_task1_loss`
- `train_task2_loss`
- `train_task3_loss`
- `val_task1_macro_f1`
- `val_task2_macro_f1`
- `val_task3_macro_f1`
- `shared_validation_score`
- `learning_rate`
- machine-readable `run_summary.json`

## Later efficiency protocol

No efficiency benchmark was run. The config freezes later measurement on the
same Tesla T4 and 224×224 input, with fixed batch size, evaluation mode,
`no_grad`, warm-up, and repeated timing. Planned measures are parameter count,
checkpoint/model size, reliable FLOPs if available, inference latency, and peak
GPU memory.

## Gate 03B test coverage

Focused tests were added for:

1. Task output shapes and all heads receiving one encoder forward.
2. ISIC non-malignant, melanoma, BCC, SCC, and Stage-3 masks.
3. Explicit missing-label sentinels.
4. Task-specific active-row loss selection for all three tasks.
5. Safe zero-active-task skipping.
6. Active-task-only denominator normalization.
7. Shared-encoder and active-head gradients.
8. Inactive-head invalid-target safety.
9. Arithmetic shared validation score.
10. Patience-seven maximize-mode early stopping.
11. Frozen config, seed, optimizer, and scheduler behavior.

The full Gate 03C test campaign was intentionally not run.

## Static verification

AST parsing passed for the four new Python implementation/test files. YAML
parsing passed for the frozen experiment config. Import checks passed for the
new model, data, and training modules using the repository's existing package
set.

## Open execution gates

No known design blocker remains in Gate 03B. Dataset-backed loader audits,
focused unit execution, CPU smoke coverage, and the full test campaign remain
for the separately authorized Gate 03C. GPU sanity and full training remain
future gates.

Internal test remains closed.
