# Phase 06B Class-Balanced Focal Amendment

## Objective and frozen comparison

Phase 06B is a planned direct flat four-class EfficientNet-B0 candidate. It
changes only the Phase 06A clean cross-entropy loss to the repository's
class-balanced focal loss. All split, mapping, model, preprocessing,
augmentation, optimization, scheduling, loading, stopping, reproducibility,
logging, and artifact settings remain equivalent to Phase 06A.

The class order is exactly
`[non_malignant, melanoma, bcc, scc]`. Training counts are read from the locked
seed-42 manifest and are `11,193 / 3,164 / 2,327 / 440`. Effective-number
weights use beta `0.9999`, are normalized to sum to four, and feed focal loss
with gamma `2.0`. The resolved configuration, checkpoints, and run summary
preserve this provenance.

## Selection and test lock

Only validation results may select between Phase 06A and Phase 06B. Highest
validation macro-F1 wins; validation balanced accuracy breaks a tie. If both
are exactly tied, retain clean CE. SCC precision, recall, and F1 are secondary
interpretation metrics and cannot override the primary rule.

The Phase 06A validation reference is macro-F1 `0.6535716654`, balanced
accuracy `0.6796986150`, accuracy `0.7729007634`, weighted F1 `0.7806549873`,
and SCC F1 `0.3870967742`. No comparison is possible until Phase 06B training
finishes. The internal test remains untouched until a winner is selected and
frozen.

## Execution and completion

No Phase 06B model may be constructed or executed locally. Full tests, CUDA
sanity, full training, inference, evaluation, XAI, and timing require the Azure
Tesla T4. Long jobs must run in `tmux`.

Required run artifacts are resolved config, environment metadata, histories,
per-epoch validation metrics, best validation metrics, best and last
checkpoints, run summary, persistent log, final status, Git commit, manifest
provenance, and SHA-256 hashes. Phase 06B becomes complete only after VM tests,
CUDA sanity, successful training, artifact/hash backup verification,
validation-only comparison, and winner freeze. It is currently a locally
prepared candidate and is not reportable.
