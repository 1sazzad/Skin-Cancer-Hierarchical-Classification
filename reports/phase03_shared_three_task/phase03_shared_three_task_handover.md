# Phase 03 Shared Three-Task Baseline — Closure Handover

## Phase status

**Phase 03: COMPLETE / PASS / FROZEN.**

- Branch: `phase03-shared-three-task-hierarchical-baseline`
- Implementation and training commit:
  `0b466ff7cde630867192fe029f6164afbca8c4e6`
- Internal-test status: **CLOSED / NOT EVALUATED**

## Gate status

| Gate | Status |
|---|---|
| Gate 03A | PASS |
| Gate 03B | PASS |
| Gate 03C | PASS |
| Gate 03D | PASS |
| Gate 03E | PASS |
| Gate 03F | PASS |

## Architecture

The frozen model is one ImageNet-pretrained EfficientNet-B0 shared encoder with
one feature representation and three fresh `Dropout(0.2) → Linear` heads:

- Task 1: 2 logits — non_malignant / malignant
- Task 2: 3 logits — melanoma / bcc / scc
- Task 3: 5 logits — Tis / T1 / T2 / T3 / T4

Historical task-specific checkpoints were not used for initialization.

## Datasets and task masks

Training used the natural concatenation of:

- ISIC2019 train: 17,124 samples
- ISIC-derived melanoma T-category train: 594 samples
- Combined natural pool: 17,718 samples

No oversampling, weighted sampling, or forced source-balanced batching was
used.

| Source | Task mask |
|---|---|
| ISIC non-malignant | `[1, 0, 0]` |
| ISIC melanoma/BCC/SCC | `[1, 1, 0]` |
| ISIC-derived Stage-3 | `[0, 0, 1]` |

Unavailable targets use `-100` and are filtered before loss evaluation.

## Frozen loss and training protocol

- Task 1: ordinary cross-entropy
- Task 2: existing class-balanced focal loss, beta 0.9999, gamma 2.0
- Task 3: train-only inverse-frequency weighted cross-entropy
- Task weights: 1 / 1 / 1
- Total loss: mean of active task mean losses
- Seed: 42
- Batch size: 64
- Optimizer: AdamW
- Learning rate: `3e-4`
- Weight decay: `1e-4`
- Scheduler: CosineAnnealingLR
- Minimum learning rate: `1e-6`
- Maximum epochs: 30
- Early-stopping patience: 7
- Selection criterion: maximize the arithmetic mean of the three validation
  Macro-F1 values

## Final validation result

Training completed by early stopping after 13 epochs. Epoch 6 was selected:

| Measure | Exact value |
|---|---:|
| Task 1 validation Macro-F1 | 0.7624178951977814 |
| Task 2 validation Macro-F1 | 0.6987503899226807 |
| Task 3 validation Macro-F1 | 0.4366844746640764 |
| Shared validation score | 0.6326175865948461 |

## Frozen checkpoint and VM artifacts

- VM run directory: `runs/phase03_shared_three_task/seed_42/`
- Checkpoint:
  `runs/phase03_shared_three_task/seed_42/best_checkpoint.pt`
- Checkpoint SHA-256:
  `2f1c2393c5c9de15dfa4a1a132a31b9a5b8ede07d7ed6e07ab90918fc2aaa9eb`
- Checkpoint size: 48,737,992 bytes
- Training environment: NVIDIA Tesla T4
- Training time: 1436.8322077899938 seconds
- Training execution commit:
  `0b466ff7cde630867192fe029f6164afbca8c4e6`

The repository branch contained the authoritative implementation and launcher
before training. The Phase 03 closure commit is the later commit containing
this handover and should be obtained from Git history after commit.

## Unresolved limitations

- Single seed only
- Single internal dataset family
- Task-3 validation N=127 with severe imbalance
- Very small T3/T4 validation support
- No internal-test evaluation
- No statistical-significance claim for validation comparisons
- No final-protocol efficiency benchmark
- No external validation
- No fairness evaluation
- No clinical-readiness or state-of-the-art claim

## Protocol lock

Phase 03 must not be reopened, retuned, or switched based on later test
results. The frozen epoch-6 checkpoint and validation decision are final.
Internal-test access, if separately authorized later, may characterize the
already frozen model but cannot alter this phase.

## Repository and next-session boundary

At handover creation, the current branch is
`phase03-shared-three-task-hierarchical-baseline`, based on implementation and
training commit `0b466ff7cde630867192fe029f6164afbca8c4e6`. The closure reports
are the only intended new changes before the closure commit.

**Next phase: Phase 04 — to be defined/frozen in the next chat session.**

Phase 04 must begin in a **new chat session only**. No Phase 04 scientific
content is defined or authorized by this handover.
