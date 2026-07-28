# Phase 07 Gate 2A — Percentile-Quantile Protocol Amendment

## Reason and timing

Gate 3 stopped because Gate 2 froze percentile probabilities 0.025 and 0.975
but did not freeze the interpolation algorithm. The stop occurred before a
bootstrap engine was implemented, before statistical execution, and before
any result was viewed. This amendment was therefore completed pre-execution.

## Exact amendment

The amended protocol requires exactly 10,000 finite, equally weighted
IEEE-754 `float64` replicate values and
`numpy.quantile(values, [0.025, 0.975], method="linear")`, with no rounding
before quantile calculation. For sorted values `x[0]` through `x[n-1]`, it
sets `h=(n-1)q`, `j=floor(h)`, and `gamma=h-j`, then returns
`(1-gamma)x[j] + gamma*x[j+1]`, or `x[n-1]` when `j=n-1`.

NaN, positive infinity, and negative infinity fail closed; no replicate may
be dropped. Zero inclusion is decided from unrounded machine-readable bounds.
JSON numbers must be finite and round-trip safe, and deterministic CSV floats
use `.17g`.

The runtime NumPy must support the explicit `method="linear"` argument and
pass a synthetic conformance test against the independent formula. No
deprecated `interpolation` fallback or unspecified default is permitted. The
execution environment records the exact NumPy version; compatible future
versions remain acceptable only when the conformance test passes.

## Phase 06C archive-member clarification

If the direct CSV is absent, execution first verifies archive
`runs/backups/phase06c/phase06c_selected_flat_internal_test_550e7cdb1144.tar.gz`
at SHA-256
`b76762b53a35a8d9b0aa96621d78ea0e4421aa6e8052d068ffc10648a4e63e91`,
then verifies exactly one regular member at
`runs/phase06c/selected_flat_internal_test/locked_primary_evaluation/internal_test_predictions.csv`
with uncompressed SHA-256
`08b3462549210ed7f2330a687c37a6de4e013e00185fadc3167aa980995e497d`.
Absolute paths, traversal, links, and duplicate matches are rejected. The
verified member may be read directly or temporarily extracted outside locked
directories and removed afterward; it must not be restored into a locked run
directory.

## Unchanged protocol and hashes

Every other Gate 2 field remains unchanged, including the paired image unit,
primary flat-minus-hierarchical macro-F1 estimand, paired class-stratified
resampling, seed 42, 10,000 replicates, 95% confidence, probabilities, fixed
labels, zero-division policy, exact McNemar method, multiplicity boundaries,
SCC limitation, routing restrictions, claims restrictions, and fail-closed
conditions.

- Previous protocol-lock SHA-256:
  `9664e8e30cb46b97c29e9d47515d598495fb18c53b88e8ae8d370d5c88034b18`
- Amended protocol-lock SHA-256:
  `efaace517733ae7c91d2284bb4a7ca55fa7f8052790f75ebb146256ba7d8a73f`
