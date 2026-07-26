# Phase 04 - Stage 2 Imbalance-Aware Final Internal Evaluation

## Experimental protocol

- Dataset: ISIC 2019.
- Split: frozen leakage-aware seed-42 split.
- Task: melanoma vs BCC vs SCC.
- Backbone: EfficientNet-B0.
- Model selection used validation macro-F1 only.
- Internal-test results were not used for model selection.
- The selected checkpoint was evaluated once on the internal test.

## Validation-based model selection

| Variant | Macro-F1 | Balanced accuracy | SCC recall | SCC F1 |
|---|---:|---:|---:|---:|
| Clean CE | 0.763902 | 0.749586 | 0.478723 | 0.535714 |
| Weighted CE | 0.764999 | 0.745833 | 0.468085 | 0.536585 |
| Class-balanced focal | 0.776307 | 0.776287 | 0.617021 | 0.604167 |

## Frozen model

- Loss: class-balanced focal loss.
- Effective-number beta: `0.9999`.
- Focal gamma: `2.0`.
- Seed: `42`.
- Frozen epoch: `8`.
- Checkpoint SHA-256: `10986d41b64a685fcd8fe166623c5b1c7fd2f21bdad7cf4d55dedc3967a397fd`.
- Repository commit: `a9fc63b3af1fca83992b09debf4f12c5b9bb0c83`.

## Final internal-test comparison

| Metric | Clean CE | CB Focal | Delta |
|---|---:|---:|---:|
| Accuracy | 0.826772 | 0.833858 | +0.007087 |
| Balanced accuracy | 0.696166 | 0.722716 | +0.026550 |
| Macro-F1 | 0.712918 | 0.724875 | +0.011958 |
| Weighted-F1 | 0.821983 | 0.832915 | +0.010932 |

## Per-class internal-test comparison

| Class | Metric | Clean CE | CB Focal | Delta |
|---|---|---:|---:|---:|
| melanoma | precision | 0.875362 | 0.874116 | -0.001246 |
| melanoma | recall | 0.890855 | 0.911504 | +0.020649 |
| melanoma | f1 | 0.883041 | 0.892419 | +0.009378 |
| bcc | precision | 0.794971 | 0.846809 | +0.051838 |
| bcc | recall | 0.825301 | 0.799197 | -0.026104 |
| bcc | f1 | 0.809852 | 0.822314 | +0.012462 |
| scc | precision | 0.555556 | 0.462366 | -0.093190 |
| scc | recall | 0.372340 | 0.457447 | +0.085106 |
| scc | f1 | 0.445860 | 0.459893 | +0.014033 |

## Selected-model confusion matrix

Rows are actual classes and columns are predicted classes in the order `melanoma`, `bcc`, `scc`.

```text
[618, 36, 24]
[74, 398, 26]
[15, 36, 43]
```

## Validation-to-test generalization gap

| Metric | Validation | Internal test | Delta |
|---|---:|---:|---:|
| Accuracy | 0.851969 | 0.833858 | -0.018110 |
| Balanced accuracy | 0.776287 | 0.722716 | -0.053571 |
| Macro-F1 | 0.776307 | 0.724875 | -0.051431 |
| Weighted-F1 | 0.849904 | 0.832915 | -0.016989 |

### SCC validation-to-test gap

| Metric | Validation | Internal test | Delta |
|---|---:|---:|---:|
| precision | 0.591837 | 0.462366 | -0.129471 |
| recall | 0.617021 | 0.457447 | -0.159574 |
| f1 | 0.604167 | 0.459893 | -0.144274 |

## Interpretation

The validation-selected class-balanced focal model improved internal-test macro-F1 and balanced accuracy relative to the frozen clean baseline.

SCC recall and SCC F1 improved modestly, but SCC precision decreased. The model therefore detected more SCC cases while also producing more false-positive SCC predictions.

The validation SCC gain did not fully generalize to the internal test. This result supports a modest imbalance-aware improvement, not a claim that minority-class instability has been solved.

## ICCIT claim boundaries

- Stage 2 macro-F1 and balanced accuracy improved over clean CE.
- SCC recall and F1 improved modestly.
- SCC precision remains limited.
- Results currently represent one random seed.
- No clinical-readiness claim is supported.

## Internal-test lock

The selected checkpoint was evaluated once on the frozen internal-test partition. The internal-test result must not be used to retune loss parameters, epoch selection, augmentation, or model selection.

## Phase 04 outcome

Phase 04 is complete. The selected Stage 2 checkpoint is frozen for conditional hierarchical end-to-end evaluation.
