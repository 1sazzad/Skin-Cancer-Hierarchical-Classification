# Phase 02 / Gate 02C — Controlled Seven-Backbone Validation Benchmark and Architecture Selection

## Gate identity and verdict

**Gate 02C: PASS.**

This gate closes the Phase 02 controlled seven-backbone validation benchmark
and formally freezes the architecture-selection decision before any new
internal-test access.

- Phase: 02
- Gate: 02C
- Benchmark: controlled seven-backbone validation benchmark
- Execution commit for all five new full runs:
  `fd7fa7e2b3ac8eafb2a3b3af8db06c1431dbeec9`
- Execution hardware: Azure Tesla T4
- Python: `3.11.9`
- torch: `2.13.0+cu130`
- Seed: `42`
- Final verdict: **PASS**

## Frozen task and protocol

The benchmark used ISIC 2019 flat four-class classification with the fixed
class order:

`[non_malignant, melanoma, bcc, scc]`

All candidates used the same frozen leakage-safe split and protocol:

| Field | Frozen value |
|---|---|
| Architecture-selection metric | Validation Macro-F1 only |
| Maximum epochs | 30 |
| Early-stopping patience | 7 |
| Batch size | 64 |
| Input resolution | 224×224 |
| Initialization | ImageNet pretrained |
| Loss | Ordinary cross-entropy |
| Optimizer | AdamW |
| Learning rate | `3e-4` |
| Weight decay | `1e-4` |
| Scheduler | Cosine annealing |
| Mixed precision | AMP on CUDA |
| Random seed | 42 |

Internal-test evidence was prohibited from training, early stopping, checkpoint
selection, ranking, and architecture selection.

EfficientNet-B0 and DenseNet121 were reused from previous
protocol-compatible runs. They were **not retrained** for Gate 02C. The five new
full runs were DenseNet169, ResNet50, MobileNetV3-Large, EfficientNet-B2, and
EfficientNet-B3.

## Frozen seven-backbone validation ranking

The following ranking is final and is based exclusively on the frozen
validation Macro-F1:

| Rank | Architecture | Best epoch | Validation Macro-F1 | Run source |
|---:|---|---:|---:|---|
| 1 | DenseNet169 | 12 | 0.6659647727205366 | New Gate 02C full run |
| 2 | EfficientNet-B3 | 5 | 0.6583008001230898 | New Gate 02C full run |
| 3 | MobileNetV3-Large | 8 | 0.658240722895119 | New Gate 02C full run |
| 4 | EfficientNet-B2 | 2 | 0.6544148913018579 | New Gate 02C full run |
| 5 | EfficientNet-B0 | 2 | 0.6535716654 | Reused protocol-compatible run |
| 6 | ResNet50 | 6 | 0.6533194436076135 | New Gate 02C full run |
| 7 | DenseNet121 | 4 | 0.6449820791 | Reused protocol-compatible run |

## New full-run records

| Architecture | Completed epochs | Early stopped | Best epoch | Best validation Macro-F1 | Best-checkpoint SHA-256 |
|---|---:|---|---:|---:|---|
| DenseNet169 | 19 | true | 12 | 0.6659647727205366 | `73c9fd236ab0a630d4bf92cd459ec72abc5d1a45c4bd09e97fbd09d8481d8896` |
| ResNet50 | 13 | true | 6 | 0.6533194436076135 | `d7e8a29975b73bf459c653a6f7803493c19e1042ae41ae7a35ca8e68dad49715` |
| MobileNetV3-Large | 15 | true | 8 | 0.658240722895119 | `9024c37010758a148697f8714525216f573648087eb37285474c7d6c1e583d8a` |
| EfficientNet-B2 | 9 | true | 2 | 0.6544148913018579 | `298d314e20be2d8a23bffddeaa033dbb3e180510d5e8bda5b2e5fe3269f9bc14` |
| EfficientNet-B3 | 12 | true | 5 | 0.6583008001230898 | `67c8cfc64962fafa72f91a4623df85e2d61240ec57cefe021b0dd5244a2eb9e8` |

All five new full runs used execution commit
`fd7fa7e2b3ac8eafb2a3b3af8db06c1431dbeec9`.

## Dataset manifest identity

The frozen split manifest used by the controlled benchmark is identified by:

`SHA-256: 818baee0aee1353867553db7ada79492e074b9799583a769207423547157fafa`

The split was not regenerated or changed during Gate 02C.

## Internal-test quarantine

Internal-test quarantine is **PASS**:

- No new internal-test evaluation was performed for any of the five new
  candidates during Gate 02C.
- No internal-test metrics, predictions, or other internal-test artifacts were
  present in their full-run directories.
- No internal-test metric influenced training, early stopping, checkpoint
  selection, validation ranking, or architecture selection.
- Historical EfficientNet-B0 and DenseNet121 test evidence remained
  quarantined from this selection decision.

This gate freezes the selection before the predeclared one-time internal-test
evaluation. Losing candidates remain prohibited from new internal-test
evaluation.

## Execution-integrity note

The first attempted DenseNet169 full launch created only:

- `environment.json`
- `resolved_config.yaml`

It terminated before completing any epoch and produced no history, checkpoint,
run summary, or reportable scientific result. It is an aborted execution
attempt and must not be mixed with the successful run.

The authoritative successful DenseNet169 full run is:

`full__phase02_flat_four_class_isic2019_densenet169_cross_entropy_seed42__20260823T071733Z`

Only this successful run supplies the DenseNet169 values and checkpoint used in
the frozen ranking and selection decision.

## Frozen architecture-selection decision

**DenseNet169 is selected as the Phase 02 architecture winner solely because it
achieved the highest frozen validation Macro-F1:
`0.6659647727205366`.**

The selected best checkpoint is frozen by:

`SHA-256: 73c9fd236ab0a630d4bf92cd459ec72abc5d1a45c4bd09e97fbd09d8481d8896`

No candidate switching is permitted based on any subsequent internal-test
result. The later internal-test evaluation may characterize the already
selected DenseNet169 checkpoint, but it cannot reopen or alter architecture
selection.

## Scientific interpretation

DenseNet169 currently leads this controlled architecture benchmark on the
single frozen validation split. EfficientNet-B3 and MobileNetV3-Large are
extremely close to one another. EfficientNet-B2, EfficientNet-B0, and ResNet50
form another close cluster.

This is a single-seed validation comparison. The observed differences do not
establish statistical significance. This gate makes no state-of-the-art or
clinical-readiness claim.

## Gate closure and next action

Gate 02C is **PASS**. No scientific or protocol blocker remains before the
predeclared one-time internal-test evaluation of the validation-selected
DenseNet169 checkpoint.

After this report is reviewed, committed, pushed, and synchronized to the VM,
perform exactly one locked internal-test evaluation of the selected DenseNet169
best checkpoint.

Do not evaluate any losing new candidate on the internal test. Do not use the
DenseNet169 internal-test result to reconsider architecture selection. Do not
merge to `main` as part of this gate closure.
