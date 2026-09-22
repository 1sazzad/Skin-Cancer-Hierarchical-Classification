# ITT-04 — Frozen Internal ISIC Test Execution

The full repository regression gate passed before this phase:

- 387 tests passed
- 6 warnings
- no test failures

ITT-04 consumes only the seven stored Flat per-image prediction CSVs from Gate 06D.
It performs no training, checkpoint loading, HIBA access, or threshold optimization.

Run from the ITT branch after pulling:

```powershell
python scripts/itt2026/run_internal_isic.py
```

Expected output directory:

`results/extensions/itt2026_consensus/internal_isic/`

The runner refuses to overwrite an existing non-empty ITT-04 output directory.

Generated files:

- `per_sample_consensus_predictions.csv`
- `full_coverage_metrics.csv`
- `selective_referral_metrics.csv`
- `agreement_strata.csv`
- `itt04_internal_isic_results.json`
- `itt04_execution_manifest.json`

This phase reports point estimates and the complete frozen referral series only.
Inferential bootstrap/McNemar analysis remains reserved for ITT-06.
