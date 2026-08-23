# Phase 02 Gate 02D — Locked DenseNet169 Internal-Test Result

## Final verdict

**PASS — CONSUMED AND LOCKED.**

## Purpose and selection freeze

Gate 02D records exactly one internal-test evaluation of the
validation-selected DenseNet169 checkpoint. Model and checkpoint selection
occurred before internal-test access, and this result cannot change or reopen
the Phase 02 architecture-selection decision.

- Gate 02C selection-freeze commit:
  `f9cc15494aceaccf5320e61468e84e39ba475f9f`
- Selected validation Macro-F1: `0.6659647727205366`
- Selected checkpoint epoch: `12`
- Selected checkpoint SHA-256:
  `73c9fd236ab0a630d4bf92cd459ec72abc5d1a45c4bd09e97fbd09d8481d8896`
- Gate 02C report SHA-256:
  `6bdc5cd253bdfdf566e204a859ffc0df9175709e7b6b548f81412fa56904d354`

## Evaluation environment and execution

| Field | Value |
|---|---|
| Hardware | Azure Tesla T4 |
| Python | 3.11.9 |
| torch | 2.13.0+cu130 |
| CUDA build | 13.0 |
| Batch size | 64 |
| `num_workers` | 4 |
| Started | 2026-08-23T12:08:34Z |
| Completed | 2026-08-23T12:09:09Z |
| Exit status | 0 |
| Evaluation time | 26.003861930999847 seconds |
| Throughput | 141.05597121430978 samples/s |
| Sample count | 3,668 |

## Aggregate internal-test metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.7696292257360959 |
| Balanced accuracy | 0.6532892120526906 |
| Macro-F1 | 0.6493306152858874 |
| Weighted F1 | 0.7744815161302145 |
| Macro precision | 0.6591216259106703 |
| Macro recall | 0.6532892120526906 |
| Mean loss | 0.7232241063757577 |

## Per-class metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| non_malignant | 0.8839326354119253 | 0.8098415346121768 | 0.8452665941240479 | 2,398 |
| melanoma | 0.5471478463329453 | 0.6932153392330384 | 0.6115810019518543 | 678 |
| bcc | 0.6897810218978102 | 0.7590361445783133 | 0.722753346080306 | 498 |
| scc | 0.515625 | 0.35106382978723405 | 0.4177215189873418 | 94 |

SCC remains the weakest class by recall and is an important limitation.

## Confusion matrix

Rows are true labels and columns are predicted labels in this order:
`[non_malignant, melanoma, bcc, scc]`.

| True \ Predicted | non_malignant | melanoma | bcc | scc |
|---|---:|---:|---:|---:|
| non_malignant | 1,942 | 322 | 114 | 20 |
| melanoma | 186 | 470 | 22 | 0 |
| bcc | 54 | 55 | 378 | 11 |
| scc | 15 | 12 | 34 | 33 |

## Validation-to-test comparison

| Metric | Validation | Internal test |
|---|---:|---:|
| Accuracy | 0.7955288985823337 | 0.7696292257360959 |
| Balanced accuracy | 0.6711377638863596 | 0.6532892120526906 |
| Macro-F1 | 0.6659647727205366 | 0.6493306152858874 |
| Weighted F1 | 0.8004971172479687 | 0.7744815161302145 |

Internal-test Macro-F1 is approximately `0.01663` lower than validation
Macro-F1 in absolute terms. This descriptive difference does not establish
statistical significance.

## Historical descriptive context

Locked historical internal-test Macro-F1 point estimates are:

| System | Macro-F1 |
|---|---:|
| DenseNet169 | 0.6493306152858874 |
| DenseNet121 | 0.6351074531606824 |
| EfficientNet-B0 | 0.6192224685168973 |
| Predicted-gate hierarchy | 0.6053674005561019 |

DenseNet169 has the highest point-estimate Macro-F1 among these locked
historical systems. This comparison is descriptive only: test performance did
not influence Phase 02 architecture selection, and no statistical-significance
claim is made from these point estimates.

## Artifact integrity

| Artifact | SHA-256 |
|---|---|
| `internal_test_metrics.json` | `acc400b81dd92a3a7564da00fb072e27eb281a41fe8e64d5928620f32fec762e` |
| `internal_test_predictions.csv` | `64e3a56db5eb4068a020d20041f477a3fe0367ff642363a04c2007ba119ca97e` |
| `confusion_matrix.csv` | `6c08be52e55d0a7709e139a752f4ca8d6d98901c786ba06534adda8190cb98d0` |
| `per_class_metrics.csv` | `c7ce1d0921c89e96c593f3e47e5f0f3ce55098f24ea70d9f0c80ba0947ec38ff` |
| `evaluation_summary.json` | `b28dc07386f126be521ae86c5f09990b2b3f46d3987ea9f156eb273d63fc2925` |

## Backup integrity

- Archive:
  `runs/backups/phase02/phase02_locked_evidence_f9cc15494ace.tar.gz`
- Archive SHA-256:
  `892faafe95be092c0d3ecd67fdbfd0ffbc863de2e95593498fa95a8c982dba96`
- VM and local archive hashes matched exactly
- Archive audit: PASS
- Five best checkpoints: present
- Locked internal-test prediction files: exactly one
- Required locked internal-test artifacts: present
- `last_checkpoint.pt` count: 0
- Last checkpoints: intentionally excluded from the archive
- Git repository after execution and backup: clean

## Protocol lock

The Phase 02 internal-test protocol is consumed. The following are forbidden:

- rerunning the DenseNet169 internal test;
- evaluating any losing candidate on the internal test;
- switching candidates after observing test results;
- post-test hyperparameter or threshold tuning;
- reopening Phase 02 architecture selection based on the test result.

## Claims boundary

This result does not establish statistical significance, clinical readiness,
external generalization, fairness, or state-of-the-art performance. It is one
locked evaluation on the frozen internal split. Strong class imbalance and the
low SCC recall remain material scientific limitations.
