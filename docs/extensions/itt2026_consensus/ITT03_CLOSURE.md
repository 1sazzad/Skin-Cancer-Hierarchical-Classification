# ITT 2026 — Phase ITT-03 Closure

**Branch:** `itt2026-consensus-selective-referral`  
**Status:** PASS / CLOSED — CASE C  
**Date:** 2026-09-21

## 1. Local recovery audit result

The recovery audit was executed over:

`F:\Research\Final Year`

Observed:

- scanned checkpoint files: 36
- scanned CSV files: 152
- expected frozen Flat checkpoints: 7
- exact SHA-256 checkpoint matches recovered: 2
- candidate per-image validation prediction CSVs: 0
- hash errors: 0

Exact checkpoint recovery:

- DenseNet121: recovered
- EfficientNet-B0: recovered
- DenseNet169: not recovered
- ResNet50: not recovered
- MobileNetV3-Large: not recovered
- EfficientNet-B2: not recovered
- EfficientNet-B3: not recovered

The Phase-02 sibling path was also checked and does not exist locally.

## 2. ITT-03 decision

This is **Case C** from the prospectively defined ITT-03 decision gate:

> Fewer than seven exact checkpoints were recovered and no complete seven-model per-image validation prediction set is available.

Therefore the ITT study will **not regenerate a partial validation ensemble**, because using only two models would not correspond to the frozen seven-model method.

The study will also **not use the ISIC internal-test split or HIBA to choose a referral operating point**.

## 3. Frozen selective-referral policy

The following complete prespecified series remains the only primary referral analysis:

- 7/7 agreement
- >=6/7 agreement
- >=5/7 agreement
- >=4/7 agreement

No threshold in this series will be labeled optimal, selected, preferred, or tuned on the internal test or external cohort.

The no-referral/full-coverage ensemble result is reported separately as the 100% coverage reference point.

## 4. Frozen ensemble definitions remain unchanged

Primary ensemble:

- seven-model hard majority vote
- validation-weighted tie resolution
- final deterministic fallback from frozen validation ranking

Secondary ensemble:

- validation-macro-F1-weighted hard vote

Primary single-model comparator:

- DenseNet169, selected prospectively from the highest frozen validation macro-F1

No model is dropped because its checkpoint is unavailable. ITT evaluation uses the already stored per-image **Flat predictions** from all seven frozen models.

## 5. Validation limitation to report in the manuscript

Recommended wording:

> Per-image validation predictions for all seven constituent CNNs were not retained, and only two of the seven exact historical checkpoints were recoverable from local storage. Consequently, a single referral operating point was not selected from validation data. To avoid test-driven threshold selection, selective prediction was evaluated using a prospectively specified agreement series (7/7, >=6/7, >=5/7, and >=4/7) on the locked internal and external cohorts.

This is a reproducibility/selection limitation, not a reason to retrain the seven constituent CNNs.

## 6. Leakage boundary

From this point forward:

- no threshold selection from ISIC test;
- no threshold selection from HIBA;
- no model removal or replacement;
- no redefinition of the primary ensemble;
- no redefinition of the primary comparator;
- no use of hierarchical/oracle predictions for ITT consensus.

## 7. Phase status

**ITT-03: PASS / CLOSED — CASE C.**

The next phase is:

**ITT-04 — Frozen Internal ISIC Test Evaluation**

Before ITT-04 execution, the implementation regression gate should pass. The ITT-specific suite and synthetic smoke test already passed; the full repository test suite should be run once before consuming ITT internal-test results.
