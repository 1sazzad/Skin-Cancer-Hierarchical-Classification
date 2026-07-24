# Phase 03 — Clean Baseline Internal Evaluation

## Experimental status

- Dataset: ISIC 2019
- Split: frozen leakage-aware seed-42 split
- Architecture: EfficientNet-B0, ImageNet pretrained
- Loss: ordinary cross-entropy
- Checkpoint selection: validation macro-F1
- Internal test used only after checkpoint freezing
- Test results must not be used to retune these baseline models

## Stage 1 — Malignant vs Non-malignant

- Best validation epoch: 5
- Validation macro-F1: 0.808693
- Internal-test samples: 3668
- Accuracy: 0.787077
- Balanced accuracy: 0.789932
- Macro-F1: 0.774783
- Weighted-F1: 0.790965

### Per-class results

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Non-malignant | 0.880113 | 0.780651 | 0.827403 | 2398 |
| Malignant | 0.658663 | 0.799213 | 0.722163 | 1270 |

### Confusion matrix

| Actual / Predicted | Non-malignant | Malignant |
|---|---:|---:|
| Non-malignant | 1872 | 526 |
| Malignant | 255 | 1015 |

## Stage 2 — Melanoma, BCC, SCC

- Best validation epoch: 7
- Validation macro-F1: 0.763902
- Internal-test samples: 1270
- Accuracy: 0.826772
- Balanced accuracy: 0.696166
- Macro-F1: 0.712918
- Weighted-F1: 0.821983

### Per-class results

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Melanoma | 0.875362 | 0.890855 | 0.883041 | 678 |
| BCC | 0.794971 | 0.825301 | 0.809852 | 498 |
| SCC | 0.555556 | 0.372340 | 0.445860 | 94 |

### Confusion matrix

| Actual / Predicted | Melanoma | BCC | SCC |
|---|---:|---:|---:|
| Melanoma | 604 | 64 | 10 |
| BCC | 69 | 411 | 18 |
| SCC | 17 | 42 | 35 |

## Interpretation

Stage 1 provides a viable clean baseline but retains substantial false-positive
and false-negative errors. Stage 2 performs strongly for melanoma and BCC but
poorly for the minority SCC class. The gap between accuracy and balanced
accuracy confirms that overall accuracy hides minority-class weakness.

The next phase will develop imbalance-aware Stage 2 variants using only the
training and validation partitions. Internal-test results from this phase will
not be used for model or hyperparameter selection.
