# ITT-05 — Frozen HIBA Zero-Shot Execution

ITT-05 evaluates the frozen seven-model consensus on the committed HIBA external
per-image Flat predictions only.

Run:

```powershell
git pull
python scripts/itt2026/run_hiba.py
```

Expected output:

`results/extensions/itt2026_consensus/external_hiba/`

The runner verifies:

- 1,232 images
- 568 patients
- identical image IDs across all seven models
- identical ground truth
- non-missing patient IDs
- frozen seven-model set

It performs no training, checkpoint loading, threshold tuning, model selection,
or ISIC internal-test recomputation.

Inferential patient-cluster bootstrap remains reserved for ITT-06.
