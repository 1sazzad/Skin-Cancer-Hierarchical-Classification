# Final Year Project — Phase 02 Closure Handover

- Date: 2026-08-23
- Phase: Phase 02 — Controlled Seven-Backbone Benchmark
- Status: **COMPLETE / PASS / LOCKED**

## Goal completed

Phase 02 completed the controlled validation-only comparison of seven backbones
under the frozen flat four-class ISIC 2019 protocol. DenseNet169 was selected
before new internal-test access, and its single authorized internal-test
evaluation has been consumed and locked.

## Branch and important commits

- Branch: `phase02-controlled-seven-backbone-benchmark`
- Phase 01 starting closure:
  `7c4460fd339ca13bf3d4b47122181648fa7fefdb`
- Gate 02A implementation:
  `70ffcb7ecd99bfaf9ea3c792b8a4a87a3c4f0174`
- Gate 02B closure:
  `fd7fa7e2b3ac8eafb2a3b3af8db06c1431dbeec9`
- Gate 02C validation-selection freeze:
  `f9cc15494aceaccf5320e61468e84e39ba475f9f`

The final Phase 02 closure commit is the commit containing this handover. Obtain
its SHA from Git history after the commit; a Git commit cannot reliably contain
its own future hash.

## Frozen dataset and task

- Dataset: ISIC 2019
- Task: flat four-class classification
- Class order: `[non_malignant, melanoma, bcc, scc]`
- Total included: 24,460
- Train: 17,124
- Validation: 3,668
- Internal test: 3,668
- Manifest:
  `data/manifests/isic2019_train_val_test_split_seed42.csv`
- Manifest SHA-256:
  `818baee0aee1353867553db7ada79492e074b9799583a769207423547157fafa`

| Cohort | non_malignant | melanoma | bcc | scc |
|---|---:|---:|---:|---:|
| Train | 11,193 | 3,164 | 2,327 | 440 |
| Validation | 2,398 | 678 | 498 | 94 |
| Internal test | 2,398 | 678 | 498 | 94 |

## Frozen controlled protocol

- Seed: 42
- Resolution: 224×224
- Initialization: ImageNet pretrained
- Dropout: 0.2
- Loss: ordinary cross-entropy
- Optimizer: AdamW
- Learning rate: `3e-4`
- Weight decay: `1e-4`
- Scheduler: cosine annealing with `eta_min=1e-6`
- Maximum epochs: 30
- Early-stopping patience: 7
- Batch size: 64
- Mixed precision: AMP on CUDA
- Architecture-selection metric: validation Macro-F1 only

## Seven-backbone validation ranking

| Rank | Architecture | Validation Macro-F1 | Best epoch |
|---:|---|---:|---:|
| 1 | DenseNet169 | 0.6659647727205366 | 12 |
| 2 | EfficientNet-B3 | 0.6583008001230898 | 5 |
| 3 | MobileNetV3-Large | 0.658240722895119 | 8 |
| 4 | EfficientNet-B2 | 0.6544148913018579 | 2 |
| 5 | EfficientNet-B0 | 0.6535716654 | 2 |
| 6 | ResNet50 | 0.6533194436076135 | 6 |
| 7 | DenseNet121 | 0.6449820791 | 4 |

DenseNet169 was frozen as the architecture winner **before** new internal-test
access.

## Selected checkpoint

- Path:
  `runs/phase02_full/densenet169/full__phase02_flat_four_class_isic2019_densenet169_cross_entropy_seed42__20260823T071733Z/best_checkpoint.pt`
- SHA-256:
  `73c9fd236ab0a630d4bf92cd459ec72abc5d1a45c4bd09e97fbd09d8481d8896`

## Locked internal-test result

| Metric | Value |
|---|---:|
| Accuracy | 0.7696292257360959 |
| Balanced accuracy | 0.6532892120526906 |
| Macro-F1 | 0.6493306152858874 |
| Weighted F1 | 0.7744815161302145 |
| Mean loss | 0.7232241063757577 |

SCC results:

| Precision | Recall | F1 | Support |
|---:|---:|---:|---:|
| 0.515625 | 0.35106382978723405 | 0.4177215189873418 | 94 |

## Historical comparison context

| Historical system | Locked test Macro-F1 |
|---|---:|
| DenseNet121 | 0.6351074531606824 |
| EfficientNet-B0 | 0.6192224685168973 |
| Predicted hierarchy | 0.6053674005561019 |

All comparisons are descriptive unless separately supported by a predeclared
statistical analysis. Historical test evidence did not influence the Phase 02
validation-only architecture selection.

## Execution anomalies and recovery

- The initial Gate 02B SSH connectivity blocker was recovered.
- An isolated VM worktree was used to preserve historical repository evidence.
- The first DenseNet169 full-run attempt created only `environment.json` and
  `resolved_config.yaml`, then terminated before completing an epoch.
- That attempt produced no history, checkpoint, run summary, or reportable
  result.
- The later successful DenseNet169 run is authoritative.
- No test data was used to recover execution or choose the architecture.

## Execution environment

- VM worktree:
  `~/projects/Skin-Cancer-Hierarchical-Classification-phase02`
- Python environment: `~/venvs/skin-cancer-phase02-py311`
- Python: 3.11.9
- torch: 2.13.0+cu130
- GPU: Tesla T4
- Execution order: Phase 02 full runs executed sequentially

## Evidence backup

- Local authoritative backup:
  `runs/backups/phase02/phase02_locked_evidence_f9cc15494ace.tar.gz`
- SHA-256:
  `892faafe95be092c0d3ecd67fdbfd0ffbc863de2e95593498fa95a8c982dba96`
- VM and local hashes: verified exact match

## Protocol prohibitions after closure

Do not:

- rerun the DenseNet169 internal test;
- evaluate losing candidates on the internal test;
- switch architecture based on test results;
- tune hyperparameters using the test set;
- alter the frozen Phase 02 validation ranking;
- reinterpret sanity runs as scientific results.

## Scientific limitations

- Single random seed
- Single internal dataset
- No new significance test for the seven-backbone ranking
- Strong class imbalance
- Low SCC recall
- No external validation in Phase 02
- No clinical-readiness claim
- No state-of-the-art claim

## Final verdict and next-session boundary

Phase 02 is **PASS / COMPLETE / LOCKED**. It is closed, and no further Phase 02
model execution is authorized.

The next session must begin from this handover and follow the already approved
experimental campaign and one-phase-per-chat workflow. Do not execute the next
phase or task in this closure session.
