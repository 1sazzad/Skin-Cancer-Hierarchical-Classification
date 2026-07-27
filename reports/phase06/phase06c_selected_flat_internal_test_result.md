# Phase 06C Selected Flat Model Internal-Test Result

## Protocol status

Phase 06C executed the validation-selected Phase 06A clean cross-entropy
EfficientNet-B0 checkpoint exactly once on the untouched ISIC 2019 internal-test
split. The evaluation completed successfully and the one-time protocol is now
`consumed_locked`.

No Phase 06B focal checkpoint was evaluated. No threshold, split, preprocessing,
batch-size, worker, seed, checkpoint, or candidate was changed after
internal-test access.

## Frozen identity

- Model: EfficientNet-B0
- Task: flat four-class classification
- Class order: `[non_malignant, melanoma, bcc, scc]`
- Seed: `42`
- Checkpoint:
  `runs/phase06_full/full__phase06_flat_four_class_isic2019_efficientnet_b0_cross_entropy_seed42__20260726T232308Z/best_checkpoint.pt`
- Checkpoint SHA-256:
  `f3d8b8b0e5ef42e3c287a2377b5570411d442246acd16cb874ccf903facdc7a7`
- Checkpoint epoch: `2`
- Evaluation commit: `550e7cdb1144f059c940d4240fe4579e0280a803`
- Environment: Azure Tesla T4
- Started at: `2026-07-27T15:22:59Z`
- Completed at: `2026-07-27T15:34:03Z`
- Final status: `0`

## Internal-test results

The locked internal-test split contained `3668` images.

| Metric | Value |
|---|---:|
| Accuracy | 0.7420937841 |
| Balanced accuracy | 0.6503125394 |
| Macro-F1 | 0.6192224685 |
| Weighted F1 | 0.7525567214 |
| Mean loss | 0.6232672186 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| non_malignant | 0.9128884361 | 0.7472894078 | 0.8218298555 | 2398 |
| melanoma | 0.5115115115 | 0.7536873156 | 0.6094215862 | 678 |
| bcc | 0.6155063291 | 0.7811244980 | 0.6884955752 | 498 |
| scc | 0.4054054054 | 0.3191489362 | 0.3571428571 | 94 |

Evaluation elapsed time was `30.678490` seconds, corresponding to
`119.562599` samples per second for the recorded evaluator path
on this Tesla T4 environment.

## Validation-to-test observation

The validation-selected checkpoint had validation macro-F1
`0.6535716654`. Its locked internal-test macro-F1 was
`0.6192224685`, a change of `-0.0343491969`.
This difference is descriptive and is not used for any further model selection
or tuning.

## Locked hierarchical comparison

The already locked Phase 05 predicted-gate hierarchy achieved internal-test
macro-F1 `0.6053674006`. The selected flat model achieved
`0.6192224685`, an absolute difference of
`0.0138550680` in favour of the flat model on this one internal
split.

This result answers the fair internal-comparison question for the frozen
seed-42 protocols. It does not by itself establish statistical significance,
clinical superiority, external generalisation, fairness, calibration quality,
or state-of-the-art performance.

## Artifacts and backup

- Metrics: `runs/phase06c/selected_flat_internal_test/locked_primary_evaluation/internal_test_metrics.json`
- Predictions: `runs/phase06c/selected_flat_internal_test/locked_primary_evaluation/internal_test_predictions.csv`
- Confusion matrix: `runs/phase06c/selected_flat_internal_test/locked_primary_evaluation/confusion_matrix.csv`
- Per-class metrics: `runs/phase06c/selected_flat_internal_test/locked_primary_evaluation/per_class_metrics.csv`
- Evaluation summary: `runs/phase06c/selected_flat_internal_test/locked_primary_evaluation/evaluation_summary.json`
- Local verified archive:
  `runs/backups/phase06c/phase06c_selected_flat_internal_test_550e7cdb1144.tar.gz`
- Archive SHA-256:
  `b76762b53a35a8d9b0aa96621d78ea0e4421aa6e8052d068ffc10648a4e63e91`
- Verified artifact-manifest entries: `12`

The archive was transferred from the Azure VM, independently hashed locally,
extracted locally, and all embedded artifact hashes were verified.

## Final lock

The Phase 06C protocol has been consumed. The internal test must not be rerun
for this checkpoint or used for threshold tuning, candidate switching,
hyperparameter selection, or recovery of a preferred result. The rejected
Phase 06B focal candidate remains prohibited from internal-test evaluation.
