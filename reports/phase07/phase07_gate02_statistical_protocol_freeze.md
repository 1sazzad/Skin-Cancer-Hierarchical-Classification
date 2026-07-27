# Phase 07 Gate 2 — Statistical Protocol Freeze

## Decision

Gate 2 passed: the machine-readable protocol validated and all repository
checks succeeded. This gate freezes methods only; it contains no statistical
result and authorizes no analysis execution.

The configuration source is
`configs/analysis/phase07_paired_model_comparison.yaml`. Its
`statistical_protocol` section is validated into
`reports/phase07/generated/statistical_protocol_lock.json`. Unknown top-level
protocol fields and contradictory frozen values fail validation.

## Frozen design summary

- Unit: one paired internal-test image (`image_id`).
- Primary estimand: flat macro-F1 minus hierarchical macro-F1.
- Point estimates: complete original 3,668-sample data.
- Bootstrap: paired and ground-truth stratified, sampling with replacement
  within class while preserving observed support.
- Replicates and seed: 10,000 and 42.
- Interval: two-sided 95% percentile, quantiles 0.025 and 0.975.
- Metric policy: fixed labels `[0, 1, 2, 3]`, `zero_division=0`, no silently
  dropped or non-finite replicates.
- McNemar: exact two-sided binomial test on discordant pairs, alpha 0.05.
- Multiplicity: one unadjusted primary estimand; model-level comparisons are
  secondary; per-class F1 comparisons are exploratory, with Holm–Bonferroni
  required only if class-wise inferential p-values are later introduced.
- SCC: all findings explicitly uncertain because test support is 94.

The protocol also freezes effect measures, agreement/transition descriptions,
conditional routing-error decomposition, reproducibility metadata, claims
boundaries, and fail-closed conditions. Full definitions are in
`reports/phase07/phase07_statistical_analysis_protocol.md` and the JSON lock.

## Integrity boundary

The protocol records the Gate 1 paired-manifest SHA-256
`d53e8581a95661de0446961b81458bc17295efe9c6a513c0225e442a281bf941`
and both locked prediction hashes. It configures no checkpoint, model, or
inference input. The validator does not read predictions to compute metrics,
resample observations, calculate intervals, run a statistical test, or
produce a p-value.

The paired manifest and all locked artifacts remain unchanged. Gate 3 may
begin only after this protocol freeze is reviewed and committed.
