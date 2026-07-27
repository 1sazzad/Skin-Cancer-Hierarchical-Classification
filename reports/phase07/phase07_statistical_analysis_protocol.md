# Phase 07 Statistical Analysis Protocol

## Current gate

Only prediction discovery, integrity verification, schema inspection, and
deterministic identity pairing are authorized at this gate. All computations
use existing stored prediction CSV data locally.

No bootstrap confidence intervals, McNemar test, statistical-significance
test, effect-size inference, or paper claim has been executed.

Stored predictions are the only permitted input to future Phase 07 analysis.
Internal-test inference cannot be rerun, and model checkpoints must not be
loaded to recreate or replace the locked predictions.

## Locked comparison

The future paired comparison will use the Phase 05 predicted-gate end-to-end
four-class prediction and the validation-selected Phase 06C flat four-class
prediction for the same 3,668-image seed-42 internal-test split. Pairing must
first pass exact `image_id` identity and normalized ground-truth agreement.

The fixed class order is `non_malignant`, `melanoma`, `bcc`, `scc`, mapped to
indices 0, 1, 2, and 3 respectively. Label normalization is limited to
trimming, lowercasing, and replacing hyphens or whitespace with underscores.

## Future work (not executed)

Statistical methods remain frozen pending explicit Gate 2 authorization.
Gate 2 must freeze the resampling units, confidence level, random seed,
replicate count, multiplicity handling, effect measures, paired tests, and
reporting boundaries before any statistical execution is authorized.
