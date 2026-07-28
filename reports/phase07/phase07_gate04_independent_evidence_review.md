# Phase 07 Gate 4 — Independent Evidence Review

## Decision

PASS. All committed Gate 3 numerical evidence independently reconciled.

## Numerical review

- Confusion-matrix metrics, class supports, macro/weighted F1, balanced accuracy, paired differences, interval ordering and original point estimates reconciled.
- Flat correct: 2,722; hierarchical correct: 2,715; all four correctness cells sum to 3,668.
- Accuracy difference: `(354 - 347) / 3668 = 0.0019083969465648854`.
- Prediction agreement and disagreement sum to 3,668.
- The unrounded primary interval includes zero.

## Independent McNemar check

- Recomputed exact two-sided p-value: 0.82074158826914845
- Committed p-value: 0.82074158826914845
- Absolute difference: 0
- Tolerance: 1.0e-12; status: PASS.

## Routing audit

The malignant subset partitions as `255 + 169 + 846 = 1,270`; the non-malignant subset partitions as `529 + 1,869 = 2,398`. Together the five substantive routing rows partition all 3,668 samples. The emitted rows are mutually exclusive, but structural Stage 2 missingness is a data-availability state that coincides with the implicit correct-non-malignant-routing category; it is not an error.

Because Phase 05 stored Stage 2 outputs for the union of true and predicted malignant samples, every true malignant sample invoked Stage 2. Thus structural missingness 1,869 covers all Stage-2-not-invoked samples and, in this execution policy, exactly the true non-malignant samples correctly not routed. It excludes the 255 malignant Stage 1 routing failures, whose Stage 2 outputs were nevertheless stored.

Exact definitions and denominators are locked in `generated/routing_metric_data_dictionary.csv`.

## Primary interpretation

On the locked ISIC 2019 internal-test split, the flat model achieved a higher observed macro-F1 than the hierarchical model, but the paired 95% bootstrap confidence interval for the difference included zero. The analysis therefore did not establish a statistically distinguishable macro-F1 difference.
