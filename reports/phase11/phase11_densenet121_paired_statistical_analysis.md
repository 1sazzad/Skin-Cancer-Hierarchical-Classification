# Phase 11 DenseNet-121 Paired Statistical Analysis

## Evidence and alignment

No training or inference was performed. The analysis used these locked stored predictions:

- `experiments/evaluations/phase11_densenet121_internal_test_seed42__best_epoch04/internal_test_predictions.csv` — SHA-256 `0097a64b0410b99999fadb8e231146604404edb081b8fa6f62a30d5a5ece263f`
- `reports/phase07/generated/paired_prediction_manifest.csv` — SHA-256 `717a2f29a9873bb9fd25d6478858bccd7a03a463e882e33ea25583f236c459cd`
- `runs/phase05_hierarchical_internal_test/locked_primary_evaluation/per_image_hierarchical_predictions.csv` — SHA-256 `391557deb9a1aeb9b9f97edc9d3d38759e597d56b54bfdbab9ea7482451a221a`

All three sources contained 3,668 unique matching sample IDs. Missing IDs, duplicates, target mismatches, unsupported labels, and non-finite DenseNet probabilities were absent. Pairing was by sample ID after stable sorting, never CSV row order.

Ground-truth support was non-malignant 2,398; melanoma 678; BCC 498; SCC 94.

## Methods

Metrics use the fixed endpoint order `non_malignant, melanoma, bcc, scc` and zero-division value 0. Confidence intervals are percentile 95% intervals from 10,000 paired bootstrap replicates (seed 42), resampling with replacement within each true endpoint class and preserving class support. Quantiles use unrounded float64 values and NumPy's linear method. McNemar p-values are exact, two-sided binomial tests over discordant paired correctness outcomes.

## Results

### DenseNet-121 versus Flat EfficientNet-B0

| Metric | DenseNet-121 | Comparator | Difference | Paired 95% CI |
|---|---:|---:|---:|---:|
| Accuracy | 0.791439 | 0.742094 | +0.049346 | [+0.035169, +0.063250] |
| Balanced Accuracy | 0.616828 | 0.650313 | -0.033484 | [-0.063579, -0.003713] |
| Macro F1 | 0.635107 | 0.619222 | +0.015885 | [-0.014833, +0.046321] |
| Weighted F1 | 0.786162 | 0.752557 | +0.033606 | [+0.019848, +0.047081] |

Both correct: 2428; DenseNet-only correct: 475; comparator-only correct: 294; both incorrect: 471. Exact two-sided McNemar p = 7.0079542951773787e-11.

### DenseNet-121 versus Predicted-gate hierarchy

| Metric | DenseNet-121 | Comparator | Difference | Paired 95% CI |
|---|---:|---:|---:|---:|
| Accuracy | 0.791439 | 0.740185 | +0.051254 | [+0.037077, +0.065431] |
| Balanced Accuracy | 0.616828 | 0.631199 | -0.014371 | [-0.044864, +0.016874] |
| Macro F1 | 0.635107 | 0.605367 | +0.029740 | [+0.000335, +0.058815] |
| Weighted F1 | 0.786162 | 0.750332 | +0.035831 | [+0.022129, +0.049474] |

Both correct: 2431; DenseNet-only correct: 472; comparator-only correct: 284; both incorrect: 481. Exact two-sided McNemar p = 8.1535009489804309e-12.

## DenseNet-121 per-class verification

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| non_malignant | 0.845667 | 0.891159 | 0.867817 | 2398 |
| melanoma | 0.647458 | 0.563422 | 0.602524 | 678 |
| bcc | 0.738589 | 0.714859 | 0.726531 | 498 |
| scc | 0.405797 | 0.297872 | 0.343558 | 94 |

SCC recall reproduces the locked value: 28/94 = 0.297872.

## Interpretation

A confidence interval excluding zero supports a difference for that metric under this prespecified resampling analysis; McNemar tests only paired accuracy/correctness. The results do not establish clinical superiority, equivalence, non-inferiority, or external generalization. SCC conclusions remain imprecise because support is only 94.

## Manuscript-ready factual values

DenseNet-121 achieved accuracy 0.791439, balanced accuracy 0.616828, macro-F1 0.635107, and weighted-F1 0.786162 on the locked 3,668-image internal test set. The tables above provide DenseNet-minus-comparator paired differences, percentile 95% confidence intervals, and exact two-sided McNemar results for both comparisons.
