# Phase 06 Flat Four-Class Label Audit

- Manifest rows: 25331
- Mapped rows: 24460
- Excluded rows: 871
- Reconciled: True
- Class order: `['non_malignant', 'melanoma', 'bcc', 'scc']`

## Counts

| Split | non_malignant | melanoma | bcc | scc | Total |
|---|---:|---:|---:|---:|---:|
| full_dataset | 15989 (65.3679%) | 4520 (18.4791%) | 3323 (13.5854%) | 628 (2.5675%) | 24460 |
| train | 11193 (65.3644%) | 3164 (18.4770%) | 2327 (13.5891%) | 440 (2.5695%) | 17124 |
| validation | 2398 (65.3762%) | 678 (18.4842%) | 498 (13.5769%) | 94 (2.5627%) | 3668 |
| test | 2398 (65.3762%) | 678 (18.4842%) | 498 (13.5769%) | 94 (2.5627%) | 3668 |

## Mapping

- `melanocytic_nevus` -> `non_malignant`
- `benign_keratosis_like_lesion` -> `non_malignant`
- `dermatofibroma` -> `non_malignant`
- `vascular_lesion` -> `non_malignant`
- `melanoma` -> `melanoma`
- `basal_cell_carcinoma` -> `bcc`
- `squamous_cell_carcinoma` -> `scc`

## Explicit exclusions

- `melanocytic_nevus`: 2 rows; `frozen_split_exclusion`.
- `melanoma`: 2 rows; `frozen_split_exclusion`.
- `actinic_keratosis`: 867 rows; `outside_locked_stage_1_task_scope`.

## Leakage checks

- Split-group cross-split count: 0
- Exact-hash cross-split count: 0
- Passed: True

This audit reads labels only. It performs no model loading, inference, or metrics.
