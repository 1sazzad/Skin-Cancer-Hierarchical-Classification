# Phase 04 - Stage 2 Imbalance-Aware Validation Comparison

Generated: 2026-07-26

## Evaluation policy

- Dataset: ISIC 2019 frozen leakage-aware split, seed 42.
- Task: Stage 2 malignant subtype classification.
- Classes: melanoma, BCC, and SCC.
- Model: EfficientNet-B0 with ImageNet initialization.
- Checkpoint-selection metric: validation macro-F1.
- Internal-test results were not used for model selection.
- All comparisons below use the frozen validation partition only.

## Experiment variants

1. Clean cross-entropy baseline.
2. Weighted cross-entropy using inverse-frequency training weights.
3. Class-balanced focal loss using effective-number weights with beta=0.9999 and gamma=2.0.

## Overall validation results

| Metric | Clean CE | Weighted CE | CB Focal | CB Focal vs Clean |
|---|---:|---:|---:|---:|
| Accuracy | 0.861417 | 0.864567 | 0.851969 | -0.009449 |
| Balanced accuracy | 0.749586 | 0.745833 | 0.776287 | +0.026700 |
| Macro-F1 | 0.763902 | 0.764999 | 0.776307 | +0.012404 |
| Weighted-F1 | 0.858495 | 0.859833 | 0.849904 | -0.008591 |

## Per-class validation results

| Class | Metric | Clean CE | Weighted CE | CB Focal | CB Focal vs Clean |
|---|---|---:|---:|---:|---:|
| melanoma | precision | 0.907914 | 0.885714 | 0.869448 | -0.038465 |
| melanoma | recall | 0.930678 | 0.960177 | 0.952802 | +0.022124 |
| melanoma | f1 | 0.919155 | 0.921444 | 0.909219 | -0.009936 |
| bcc | precision | 0.834331 | 0.866667 | 0.881119 | +0.046788 |
| bcc | recall | 0.839357 | 0.809237 | 0.759036 | -0.080321 |
| bcc | f1 | 0.836837 | 0.836968 | 0.815534 | -0.021303 |
| scc | precision | 0.608108 | 0.628571 | 0.591837 | -0.016271 |
| scc | recall | 0.478723 | 0.468085 | 0.617021 | +0.138298 |
| scc | f1 | 0.535714 | 0.536585 | 0.604167 | +0.068452 |

## Training and checkpoint summary

| Variant | Best epoch | Completed epochs | Training time (seconds) | Validation macro-F1 |
|---|---:|---:|---:|---:|
| Clean CE | 7 | 14 | 512.041 | 0.763902 |
| Weighted CE | 13 | 20 | 741.932 | 0.764999 |
| CB Focal | 8 | 15 | 561.167 | 0.776307 |

## Confusion matrices

Rows are true classes and columns are predicted classes in the order `melanoma`, `bcc`, `scc`.

### Clean CE

```text
[631, 46, 1]
[52, 418, 28]
[12, 37, 45]
```

### Weighted CE

```text
[651, 23, 4]
[73, 403, 22]
[11, 39, 44]
```

### CB Focal

```text
[646, 21, 11]
[91, 378, 29]
[6, 30, 58]
```

## Validation-based interpretation

Weighted cross-entropy produced only a marginal macro-F1 improvement over clean cross-entropy and did not improve SCC recall.

Class-balanced focal loss achieved the highest validation macro-F1 and balanced accuracy. Relative to clean cross-entropy, it substantially improved SCC recall and SCC F1, although overall accuracy and BCC recall were lower.

The reduction in overall accuracy is considered acceptable for the imbalance-aware objective because the improvement is concentrated in the rare SCC class and is reflected in both balanced accuracy and macro-F1.

## Model-selection decision

**Preferred Stage 2 candidate: Class-Balanced Focal Loss.**

Selection reasons:

- Highest validation macro-F1.
- Highest validation balanced accuracy.
- Highest SCC recall.
- Highest SCC F1.
- Selection was made without consulting the internal-test partition.

The checkpoint from epoch 8 is frozen as the preferred Stage 2 candidate for the next internal-test evaluation.

## Frozen checkpoint

`runs/phase04_cb_focal_full/full__stage02_isic2019_efficientnet_b0_class_balanced_focal_loss_seed42__20260726T064808Z/best_checkpoint.pt`

## Important limitation

This comparison currently represents one random seed. Multi-seed stability analysis should be completed before final ICCIT reporting when computational time permits.


