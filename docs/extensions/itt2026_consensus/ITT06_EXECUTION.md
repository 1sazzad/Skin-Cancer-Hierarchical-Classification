# ITT-06 — Statistics, Tables, and Figures

Prerequisites:

- ITT-04 internal ISIC output exists locally.
- ITT-05 HIBA output exists locally.
- No ITT-06 output directory exists yet.

Run:

```powershell
git pull
python scripts/itt2026/run_statistics_figures.py
```

The script performs the frozen inferential analysis:

- ISIC: 10,000 paired image-bootstrap replicates, seed 42
- HIBA: 10,000 patient-cluster bootstrap replicates, seed 42
- Internal ISIC exact two-sided McNemar comparisons
- Primary comparison: majority vote vs validation-selected DenseNet169
- Secondary comparisons: weighted vote vs DenseNet169; weighted vs majority

Selective-referral threshold results remain descriptive. No threshold is selected.

Outputs under:

`results/extensions/itt2026_consensus/statistics_figures/`

Publication tables:

- `table_inferential_macro_f1.csv`
- `table_internal_mcnemar.csv`
- `table_full_coverage_metrics.csv`
- `table_selective_referral.csv`
- `table_agreement_strata.csv`

Figures:

- `fig1_full_coverage_macro_f1.png`
- `fig2_risk_coverage.png`
- `fig3_agreement_distribution.png`
- `fig4_macro_f1_coverage.png`

The console prints the primary ISIC and HIBA inferential results needed for immediate interpretation.
