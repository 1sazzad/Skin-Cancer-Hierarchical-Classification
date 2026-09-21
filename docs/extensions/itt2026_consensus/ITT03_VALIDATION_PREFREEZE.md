# ITT 2026 — Phase ITT-03 Validation / Pre-execution Freeze Check

## Current status

Phase ITT-02 passed its dedicated local validation:

- `13 passed` for `tests/test_itt2026_consensus.py`
- synthetic smoke runner returned `status: PASS`
- majority and weighted predictions matched on the eight-sample smoke fixture
- all four frozen referral thresholds were exercised

No real ISIC internal-test or HIBA consensus computation has been run.

## GitHub evidence audit

The repository does not contain the seven original Flat checkpoint binaries or seven-CNN per-image validation prediction CSVs.

This absence is expected because `.gitignore` excludes:

- `/runs/`
- `models/checkpoints/*`
- `*.pt`
- `*.pth`
- `*.ckpt`
- compressed archives/backups

Therefore GitHub absence is not treated as proof that the checkpoints were lost.

## Recoverable provenance

The locked Gate 06D metrics retain the exact original Flat checkpoint path, selected epoch, and SHA-256 for all seven constituent models. These have been copied into:

`configs/extensions/itt2026_consensus/checkpoint_recovery_manifest.yaml`

This manifest is recovery metadata only. It does not change model selection.

## Local recovery audit

Run:

```powershell
python scripts/itt2026/audit_validation_recovery.py "F:\Research\Final Year"
```

If old VM exports/backups are stored elsewhere, add additional roots:

```powershell
python scripts/itt2026/audit_validation_recovery.py "F:\Research\Final Year" "D:\Backups" "C:\Users\<user>\Downloads"
```

The script:

- hashes local `.pt/.pth/.ckpt` files and compares them with the seven locked hashes;
- lists candidate validation-prediction CSV filenames;
- does not load checkpoints;
- does not run inference;
- does not inspect or recompute ISIC internal-test/HIBA consensus results.

Output:

`results/extensions/itt2026_consensus/itt03_validation_recovery_audit.json`

## Decision rule after recovery audit

### Case A — all seven exact checkpoints recovered

Regenerate per-image predictions on the frozen ISIC **validation** partition only, using the original deterministic evaluation transform. Verify that each regenerated model macro-F1 matches the frozen registry validation metric within numerical tolerance before using the predictions for any additional validation-only analysis.

Do not run ISIC internal-test or HIBA during this regeneration step.

### Case B — original per-image validation prediction files recovered

Verify:

- seven-model sample IDs are identical;
- labels are identical;
- split is validation only;
- provenance is consistent with the frozen models;
- recomputed macro-F1 matches the frozen registry values.

If all checks pass, the recovered predictions may support a secondary validation-selected referral operating point.

### Case C — fewer than seven exact checkpoints and no complete seven-model validation predictions

Formally freeze the documented limitation:

> Per-image validation predictions required for selecting a single referral operating point were unavailable. Therefore no referral threshold was optimized on internal-test or external data. Selective prediction is evaluated using the complete prospectively specified 7/7, >=6/7, >=5/7, and >=4/7 agreement series.

Then proceed without a selected threshold.

## ITT-03 closure

ITT-03 closes only after the local recovery audit establishes Case A, B, or C and the decision is recorded before any ITT internal-test ensemble computation.
