# Phase 06 Fair Flat Four-Class Protocol

## Question and hypothesis

Using the frozen seed-42 leakage-aware split and the same clean EfficientNet-B0
policy, how does a direct four-class classifier compare with the Phase 05
predicted-gate hierarchy? The hypothesis is intentionally non-directional:
direct classification may reduce routing error, while the hierarchy may retain
useful task decomposition. No superiority claim is permitted before results.

## Cohort and labels

The task uses the 24,460 split-included rows eligible for the locked Stage 1
task. Class order is exactly `[non_malignant, melanoma, bcc, scc]`.
`diagnosis_canonical` is mapped by `phase06_flat_four_class_v1`, as documented
in `flat_four_class_label_audit.md`. Actinic keratosis remains outside the
locked project task, and the four frozen duplicate-conflict rows remain
excluded.

The frozen manifest is
`data/manifests/isic2019_train_val_test_split_seed42.csv`. No regrouping,
resplitting, or test-driven tuning is allowed.

## Fair training policy

- EfficientNet-B0 with ImageNet pretrained weights and a random four-logit head.
- Seed 42; 224 x 224 input; ImageNet normalization.
- The locked moderate train augmentation and deterministic resize-256,
  center-crop-224 validation/internal-test transform.
- Clean cross-entropy first; no sampler or class weights.
- AdamW (`lr=0.0003`, weight decay `0.0001`), cosine annealing to `0.000001`,
  at most 30 epochs, patience 7, and AMP, matching the clean Phase 03 baselines.
- Validation macro-F1 alone selects the checkpoint.
- Sanity runs are non-reportable and may use train and validation only.

There is no unavoidable policy difference from Phase 03 other than the task
adapter, four-class head, and corresponding semantic class order.

## Evaluation lock

The internal test remains sealed until the Experiment A configuration,
training policy, selected validation checkpoint, and checkpoint metadata are
frozen. It will then be evaluated exactly once. Internal-test accuracy,
balanced accuracy, macro-F1, weighted F1, per-class precision/recall/F1/support,
and confusion matrix are final evaluation outputs—not tuning inputs. Melanoma,
BCC, and SCC require explicit per-class discussion.

Experiment B (an imbalance-aware direct classifier) may be proposed only after
Experiment A validation analysis and may use validation evidence only.

## Primary comparison

The primary fair comparison is the direct flat classifier versus the Phase 05
predicted-gate hierarchy. Locked comparison references are:

- Macro-F1: `0.6053674006`
- Balanced accuracy: `0.6311989239`
- Weighted F1: `0.7503315694`
- Accuracy: `0.7401853871`

These references must not influence training or checkpoint selection. The
Phase 05 oracle-gate result is diagnostic and must not be used as the primary
production comparison.

## Tesla T4 performance protocol

Latency and throughput will be measured only on the Azure Tesla T4 using the
frozen checkpoint, fixed 224 x 224 preprocessing, declared batch sizes, warm-up
iterations, synchronized CUDA timing, and recorded software/hardware state.
Report batch size, warm-up and measurement counts, mean/dispersion, images per
second, AMP mode, and inclusion/exclusion of preprocessing. No local timing is
valid.

## Artifacts and failure policy

Expected artifacts are resolved config, environment record, epoch history,
validation metrics, best/last checkpoints, run summary, later locked
internal-test predictions/metrics/confusion matrix, and T4 timing results.
Incomplete sanity/full-run directories must be labelled or removed before a
clean retry. A failed run cannot be promoted. Checkpoint, class names, mapping,
Git commit, manifest, and validation selection provenance must reconcile.

Full training requires passing local and VM tests, the label audit, config
validation, CUDA/GPU verification, a successful non-reportable CUDA sanity run,
and artifact inspection. Clinical, external-generalisation, hierarchy
superiority, or production-efficiency claims are prohibited.
