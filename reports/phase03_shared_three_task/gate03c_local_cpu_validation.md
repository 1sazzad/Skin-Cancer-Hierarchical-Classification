# Gate 03C — Local CPU Validation

## Verdict

**Gate 03C: PASS WITH ENVIRONMENT LIMITATION.**

All locally executable CPU unit, regression, configuration, task-mask,
manifest, and loss-weight audits passed. The only unverified item is actual
image-backed batch loading because the local machine does not contain the ISIC
image payload.

## Focused Phase 03 tests

| Collected | Passed | Failed | Skipped |
|---:|---:|---:|---:|
| 21 | 21 | 0 | 0 |

## Relevant regression tests

| Collected | Passed | Failed | Skipped |
|---:|---:|---:|---:|
| 22 | 22 | 0 | 0 |

The regression set covered reusable EfficientNet/classification construction,
ClassBalancedFocalLoss and existing loss utilities, classification metrics,
safe ISIC train/validation dataset behavior, transforms, and reproducibility.
Internal-test evaluation, Phase 05 locked inference, GPU, full-training, and
sanity-training tests were excluded.

## Dataset and cohort audit

| Cohort | Samples |
|---|---:|
| ISIC2019 train | 17,124 |
| Stage-3 train | 594 |
| Combined train | 17,718 |
| Task 1 validation | 3,668 |
| Task 2 malignant validation | 1,270 |
| Task 3 validation | 127 |

## Training task-mask activity

| Task | Active training samples |
|---|---:|
| Task 1 | 17,124 |
| Task 2 | 5,931 |
| Task 3 | 594 |

## Target and mask semantics

| Source label | Targets | Mask |
|---|---|---|
| Non-malignant | `[0, -100, -100]` | `[1, 0, 0]` |
| Melanoma | `[1, 0, -100]` | `[1, 1, 0]` |
| BCC | `[1, 1, -100]` | `[1, 1, 0]` |
| SCC | `[1, 2, -100]` | `[1, 1, 0]` |
| Stage-3 | `[-100, -100, target]` | `[0, 0, 1]` |

The missing-target sentinel is `-100`. It is never interpreted as valid class
zero.

## Stage-3 training distribution

| Class | Training count |
|---|---:|
| Tis | 355 |
| T1 | 184 |
| T2 | 33 |
| T3 | 10 |
| T4 | 12 |

## Train-only Task-3 inverse-frequency weights

| Class | Computed weight |
|---|---:|
| Tis | 0.063475735584673 |
| T1 | 0.1224667724595593 |
| T2 | 0.682845034319967 |
| T3 | 2.253388613255891 |
| T4 | 1.8778238443799093 |

These weights were derived only from the Stage-3 training counts above. The
frozen weighting formula was not changed.

## Loader configuration audit

- Batch size: 64
- Workers: 4
- Training sampler: `RandomSampler`
- Training shuffle: enabled
- `drop_last`: false
- Weighted sampler: none
- Stage-3 oversampling: none
- Forced source-balanced batching: none
- Validation sampling: sequential

## Local execution limitation

Actual image-backed tensor and collate loading could not be completed because
the local machine does not contain the ISIC image payload. One representative
missing path was:

`data/raw/isic2019/images/ISIC_2019_Training_Input/ISIC_0000000.jpg`

**This is an ENVIRONMENT LIMITATION, not a validation failure.** Actual
image-backed loader and tensor verification is deferred to Gate 03D on the
authorized VM where the dataset is available.

## Execution boundary

- No training was run.
- No model forward/backward training step was run.
- No GPU was used.
- The internal test remained untouched.
- No commit was created.
- Nothing was pushed.
