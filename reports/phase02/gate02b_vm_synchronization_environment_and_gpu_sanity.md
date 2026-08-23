# Phase 02 / Gate 02B — VM Synchronization, Environment Verification and GPU Sanity

## Final verdict

**Gate 02B: PASS.** SSH access was recovered, the authoritative Phase 02 commit
was verified in a separate clean VM worktree, the isolated runtime and dataset
were verified, all VM tests passed, all five batch-size-64 CUDA forward checks
passed, and the bounded DenseNet169 real-data sanity run completed successfully.

No internal-test inference was performed. No benchmark result or architecture
ranking may be inferred from the bounded sanity run.

## Initial blocker and recovery history

The first connection attempt to the configured Azure target `research-gpu`
(`4.194.30.39`) timed out on TCP port 22 before authentication. Gate 02B was
therefore initially recorded as FAIL/BLOCKED, and no VM state was changed during
that attempt.

During recovery, TCP port 22 became reachable and SSH key authentication
succeeded. The original VM repository contained untracked Phase 11 evidence.
That repository was not modified: its historical artifacts were preserved, its
backup archive SHA-256 was verified, and Phase 02 work continued in a separate
clean Git worktree.

## Local branch and push verification

- Branch: `phase02-controlled-seven-backbone-benchmark`
- Authoritative local commit: `70ffcb7ecd99bfaf9ea3c792b8a4a87a3c4f0174`
- Pre-push worktree: clean
- Push: succeeded; only the Phase 02 branch was pushed
- Remote branch resolved to the authoritative commit
- Merge to `main`: not performed

`git diff --check 7c4460fd..70ffcb7` passed. The reviewed change set was
confined to the five Phase 02 configs, backbone/model and training support,
associated tests, and the Gate 02A report.

## VM repository synchronization

- Isolated worktree:
  `~/projects/Skin-Cancer-Hierarchical-Classification-phase02`
- VM-local branch: `phase02-controlled-seven-backbone-benchmark-vm`
- VM HEAD: `70ffcb7ecd99bfaf9ea3c792b8a4a87a3c4f0174`
- VM Phase 02 worktree: clean
- Original Phase 11 VM repository: not modified

The VM executed the exact authoritative Phase 02 code commit.

## Runtime and GPU audit

- Python: `3.11.9`
- Python environment: `~/venvs/skin-cancer-phase02-py311`
- torch: `2.13.0+cu130`
- torchvision: `0.28.0+cu130`
- scikit-learn: `1.9.0`
- NumPy: `2.4.6`
- pandas: `3.0.5`
- PyYAML: `6.0.3`
- Pillow: `12.3.0`
- CUDA available: `true`
- GPU: Tesla T4
- Compute capability: `7.5`
- Usable reported GPU memory: approximately `15.57 GiB`

Dependencies were installed from `requirements-dev.txt`. The isolated Phase 02
environment did not reuse or modify the AQUA20 environment.

## VM test gate

- Targeted Phase 02 tests: **42 passed**
- First full-suite attempt: **324 passed, 1 failed, 1 skipped**
- Cause of the first failure: required historical checkpoint paths were absent
  from the isolated worktree; this was not a Phase 02 regression
- Final full suite after attaching the three hash-verified checkpoints:
  **326 passed in 28.28 seconds**

The historical checkpoints were attached with narrow ignored symlinks:

| Checkpoint | SHA-256 |
|---|---|
| Stage 1 | `95e02c26b1ea4a0dba17016313c81f97c9c2635270a37b4debbee0f84e07ba3b` |
| Stage 2 | `10986d41b64a685fcd8fe166623c5b1c7fd2f21bdad7cf4d55dedc3967a397fd` |
| Flat | `f3d8b8b0e5ef42e3c287a2377b5570411d442246acd16cb874ccf903facdc7a7` |

## Dataset verification

- Authoritative target: `/home/m-sazzad-h/skin-cancer-work/data/raw/isic2019`
- Phase 02 attachment: ignored symlink at `data/raw/isic2019`
- JPG images verified: `25,331`
- Frozen manifest: present
- Split regeneration: not performed

The frozen manifest count audit passed:

| Cohort | Total | non_malignant | melanoma | bcc | scc |
|---|---:|---:|---:|---:|---:|
| Train | 17,124 | 11,193 | 3,164 | 2,327 | 440 |
| Validation | 3,668 | 2,398 | 678 | 498 | 94 |
| Test | 3,668 | — | — | — | — |
| All | 24,460 | — | — | — | — |

The test cohort was used only for manifest integrity counting. No test image
was loaded for model evaluation.

## Frozen configuration and internal-test quarantine

All five configs preserve the frozen fields: 224×224 input, seed 42, dropout
0.2, ordinary cross entropy, AdamW at `3e-4` with weight decay `1e-4`,
CosineAnnealingLR with `eta_min=1e-6`, 30 epochs, patience 7, batch size 64,
four workers, CUDA AMP enabled, and validation macro-F1 selection.

The verified models are DenseNet169, ResNet50, MobileNetV3-Large,
EfficientNet-B2, and EfficientNet-B3. Every config explicitly sets
`evaluate_internal_test_after_training: false`. Internal-test quarantine is
**PASS**.

## CUDA batch-size-64 memory precheck

| Architecture | Forward result | Peak allocated | Peak reserved |
|---|---|---:|---:|
| DenseNet169 | PASS | 4.896 GiB | 5.096 GiB |
| ResNet50 | PASS | 2.803 GiB | 2.854 GiB |
| MobileNetV3-Large | PASS | 1.400 GiB | 1.479 GiB |
| EfficientNet-B2 | PASS | 3.991 GiB | 4.121 GiB |
| EfficientNet-B3 | PASS | 5.247 GiB | 5.318 GiB |

All models completed the synthetic batch-size-64 CUDA forward pass. No
batch-size change was required.

## DenseNet169 bounded sanity run

- tmux session: `phase02-gate02b-sanity`
- Backbone: DenseNet169 with ImageNet pretrained weights
- Device: CUDA
- Exit code: `0`
- `epoch_limit`: `1`
- `max_train_batches`: `1`
- `max_validation_batches`: `1`
- Batch size: frozen at `64`
- Seed: frozen at `42`
- AMP: PASS
- Cross-Entropy loss: PASS
- Backward pass: PASS
- AdamW optimizer step: PASS
- Validation evaluator: PASS
- Checkpoint and artifact writing: PASS

The closure evidence records the tmux session and effective bounded-run
arguments but does not include the full shell command string; no command is
reconstructed here.

| Sanity measure | Value |
|---|---:|
| Train loss | 1.395470 |
| Validation loss | 1.163116 |
| Validation macro-F1 | 0.29022988505747127 |
| Best epoch | 1 |

These values are **non-reportable sanity evidence only**. The metadata confirms
`sanity_run = true` and `reportable_as_full_result = false`.

A scikit-learn warning occurred because the single bounded validation batch did
not contain every class. This is expected and non-blocking for a one-batch,
non-reportable sanity run.

## Sanity artifacts

The sanity run produced:

- `best_checkpoint.pt`
- `last_checkpoint.pt`
- `best_validation_metrics.json`
- `validation_metrics_epoch_001.json`
- `history.csv`
- `history.json`
- `resolved_config.yaml`
- `environment.json`
- `run_summary.json`

The closure evidence supplies these artifact names but not the enclosing run
directory, so no path is inferred. No internal-test metrics, predictions, or
other test artifacts were generated.

## Deviations, blockers and next action

The recovered execution used isolation to preserve the dirty historical Phase
11 repository. The initial full-suite checkpoint-path failure was resolved with
narrow, ignored, hash-verified symlinks. Neither issue represents a scientific
or protocol blocker.

Scientific/protocol blockers: **none**.

**Recommended next action:** begin Gate 02C controlled full validation-only
training, starting with DenseNet169 and using the frozen Phase 02 config
unchanged. Do not evaluate the internal test, rank architectures prematurely,
start Phase 03, or merge to `main` as part of this closure.
