# ITT 2026 — Phase ITT-02 Implementation Handover

**Branch:** `itt2026-consensus-selective-referral`  
**Status:** IMPLEMENTED / SYNTHETIC CORE CHECKS PASS / FULL REPOSITORY PYTEST STILL REQUIRED  
**Protocol source:** `configs/extensions/itt2026_consensus/itt2026_protocol.yaml`

## Implemented

Reusable dataset-agnostic ITT utilities were added at:

`src/analysis/itt2026_consensus.py`

The module implements:

- frozen seven-model prediction alignment and integrity checks;
- hard majority vote;
- validation-weighted hard vote;
- frozen validation-based tie resolution;
- unweighted vote agreement/disagreement;
- four fixed selective-referral thresholds: 7/7, >=6/7, >=5/7, >=4/7;
- fixed-four-class full-coverage metrics;
- class-specific and malignant coverage;
- paired image-level macro-F1 bootstrap;
- patient-cluster macro-F1 bootstrap;
- exact paired-correctness McNemar wrapper;
- SHA-256 helper for later result provenance.

No ISIC test or HIBA ensemble computation is performed by this module automatically.

## Synthetic tests

Added:

`tests/test_itt2026_consensus.py`

The synthetic tests cover:

1. correct seven-model alignment;
2. duplicate-ID rejection;
3. cross-model ground-truth mismatch rejection;
4. majority voting;
5. 2/2/2/1 tie resolution using validation-weight sums;
6. exact weighted-tie fallback using frozen validation ranking;
7. weighted-vote/majority-vote divergence;
8. fixed four-class metric behavior with absent classes;
9. referral coverage and class-specific coverage;
10. rejection of non-frozen referral thresholds;
11. deterministic paired image bootstrap;
12. deterministic patient-cluster bootstrap;
13. A-minus-B McNemar direction.

A synthetic-only smoke runner was added at:

`scripts/itt2026/validate_implementation.py`

## Validation performed in the current development session

Direct synthetic assertions were executed outside the repository checkout and passed for:

- standard majority voting;
- vote-count/agreement behavior;
- 2/2/2/1 tie handling;
- validation-ranking tie fallback;
- weighted-vs-majority divergence;
- fixed-four-class macro-F1 behavior;
- 7/7, >=6/7, >=5/7, >=4/7 referral coverage;
- class-specific coverage;
- malignant coverage;
- deterministic paired bootstrap behavior.

The current execution environment could not clone GitHub because outbound DNS/network resolution is disabled. Therefore the committed pytest file has not yet been executed as part of the repository's full test suite from this environment.

## Mandatory gate before ITT-03

Run from the repository root on the ITT branch:

```powershell
git checkout itt2026-consensus-selective-referral
python -m pytest tests/test_itt2026_consensus.py -q
python scripts/itt2026/validate_implementation.py
```

Then run the complete regression suite:

```powershell
python -m pytest -q
```

Phase ITT-03 must not consume validation/ISIC/HIBA prediction files until these commands pass.

## Real-cohort boundary

Do not yet run:

- the seven-model ISIC internal-test consensus;
- HIBA consensus;
- test-set referral curves;
- HIBA referral curves;
- final bootstrap comparisons.

Those belong only after the ITT-02 test gate passes.

## Next phase

After all tests pass:

**Phase ITT-03 — Validation / Pre-execution Freeze Check**

Because per-image seven-CNN validation predictions are not currently committed, ITT-03 should first check whether the original validation prediction artifacts/checkpoints can be recovered. If they cannot, document the gap and proceed with the already frozen full referral-threshold series without selecting an operating point from ISIC test or HIBA.
