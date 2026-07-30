# Phase 11 Final DenseNet-121 Baseline Result

**Status:** Completed, internally evaluated once, and evidence locked
**Execution date:** 2026-07-30
**Implementation commit:** `133442a185060d78f23019d6997b12526e6cef3c`
**Dataset:** ISIC 2019
**Seed:** 42
**Locked internal-test cohort:** 3,668 images

## 1. Purpose

Phase 11 executed the single final architecture comparator approved after the
ICCIT scientific audit: a flat four-class DenseNet-121 baseline.

The Phase 06 clean cross-entropy protocol was preserved. The only modelling
change was the backbone architecture from EfficientNet-B0 to DenseNet-121.

No additional backbone, hyperparameter search, multi-seed run, threshold
tuning, test-time augmentation, ensemble, new dataset, or internal-test rerun
was performed.

## 2. Locked protocol

The following elements matched the original flat EfficientNet-B0 protocol:

- seed-42 leakage-aware split;
- four-class label mapping;
- class order: non-malignant, melanoma, BCC, SCC;
- ImageNet normalization;
- 224-by-224 model input;
- moderate training augmentation;
- cross-entropy loss;
- AdamW optimizer;
- learning rate `3e-4`;
- weight decay `1e-4`;
- cosine annealing;
- batch size 64;
- mixed-precision CUDA training;
- validation macro-F1 checkpoint selection;
- maximum 30 epochs;
- early-stopping patience 7;
- one-time frozen internal-test evaluation.

## 3. Training outcome

| Quantity | Result |
|---|---:|
| Parameter count | 6,957,956 |
| Configured epochs | 30 |
| Completed epochs | 11 |
| Early stopping | Yes |
| Selected epoch | 4 |
| Best validation macro-F1 | 0.6449820791 |
| Training time | 1,153.2687 seconds |
| Best checkpoint size | 84,427,588 bytes |

The checkpoint was selected only from validation macro-F1. The internal test
was not used for training, checkpoint selection, or protocol modification.

## 4. Locked internal-test performance

| Metric | Value |
|---|---:|
| Accuracy | 0.7914394766 |
| Balanced accuracy | 0.6168282266 |
| Macro-F1 | 0.6351074532 |
| Weighted-F1 | 0.7861623640 |
| Mean loss | 0.5646466127 |

### Per-class results

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Non-malignant | 0.845667 | 0.891159 | 0.867817 | 2,398 |
| Melanoma | 0.647458 | 0.563422 | 0.602524 | 678 |
| BCC | 0.738589 | 0.714859 | 0.726531 | 498 |
| SCC | 0.405797 | 0.297872 | 0.343558 | 94 |

### Confusion matrix

Rows are true classes and columns are predicted classes in the order
`[non_malignant, melanoma, bcc, scc]`.

| True class | Non-malignant | Melanoma | BCC | SCC |
|---|---:|---:|---:|---:|
| Non-malignant | 2,137 | 173 | 70 | 18 |
| Melanoma | 262 | 382 | 31 | 3 |
| BCC | 97 | 25 | 356 | 20 |
| SCC | 31 | 10 | 25 | 28 |

## 5. Final three-system comparison

| System | Accuracy | Balanced accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|
| Flat DenseNet-121 | **0.791439** | 0.616828 | **0.635107** | **0.786162** |
| Flat EfficientNet-B0 | 0.742094 | **0.650313** | 0.619222 | 0.752557 |
| Predicted-gate hierarchy | 0.740185 | 0.631199 | 0.605367 | 0.750332 |

DenseNet minus EfficientNet:

- accuracy: `+0.049346`;
- balanced accuracy: `-0.033484`;
- macro-F1: `+0.015885`;
- weighted-F1: `+0.033606`.

DenseNet minus predicted-gate hierarchy:

- accuracy: `+0.051254`;
- balanced accuracy: `-0.014371`;
- macro-F1: `+0.029740`;
- weighted-F1: `+0.035831`.

## 6. Scientific interpretation

DenseNet-121 produced the highest accuracy, macro-F1, and weighted-F1 point
estimates. EfficientNet-B0 retained the highest balanced accuracy.

The DenseNet aggregate gain did not remove minority-class sensitivity
limitations. SCC recall remained `0.297872`, meaning only 28 of 94 SCC images
were classified correctly.

The previously frozen bootstrap confidence interval and exact McNemar test
apply only to the EfficientNet-B0 flat-versus-hierarchy comparison. No new
paired confidence interval or hypothesis test was performed for DenseNet-121.

Therefore, the DenseNet result is a descriptive point-estimate improvement and
must not be reported as statistically significant superiority.

The hierarchy remains scientifically useful as a diagnostic architecture.
Oracle routing produced macro-F1 `0.793656`, compared with `0.605367` under
predicted routing, exposing a routing-associated loss of `0.188289`.

## 7. Evidence integrity

| Artifact | SHA-256 |
|---|---|
| Best checkpoint | `97f50dd5fb6b8d5a65b1c08035f07bab2bb5683e5647e519527c9cb56afbaa01` |
| Run summary | `98be848704f543b09a96fe52d50c53586c17f6dc907ce13d4b5acfce11ebc068` |
| Internal-test metrics | `4766d3255353a875a2f83b4f63b3abaf7970c100dc14db8213eb32605af106e3` |
| Internal-test predictions | `0097a64b0410b99999fadb8e231146604404edb081b8fa6f62a30d5a5ece263f` |
| Evaluation summary | `3cb5fcc9122c9d6f619f7f7ac25ea16af1fa512057bf855ddf2b8e07da8b8324` |
| Local evidence archive | `6879d7374fb748c2ad03aa846b64638a11f1b1d4478972ccda68b3ed50513a05` |

The archive is retained locally under `backups/phase11/` and excluded from
Git. Compact evaluation evidence is retained under
`experiments/evaluations/`.

## 8. Final outcome

Phase 11 is complete and locked.

DenseNet-121 is the strongest aggregate point-estimate flat comparator.
EfficientNet-B0 remains strongest on balanced accuracy.

No additional model training or internal-test rerun is authorized.
