# ITT 2026 — Phase ITT-05 Closure

**Status:** PASS / CLOSED  
**Cohort:** Frozen HIBA external zero-shot cohort  
**Images:** 1,232  
**Patients:** 568

## Full-coverage point estimates

- Primary majority-vote macro-F1: **0.6621826511**
- Secondary validation-weighted-vote macro-F1: **0.6621826511**
- Prospectively selected DenseNet169 macro-F1: **0.6333103424**
- Majority minus DenseNet169 macro-F1: **+0.0288723087**
- Weighted minus DenseNet169 macro-F1: **+0.0288723087**

These are point estimates only. Patient-cluster bootstrap inference is deferred to ITT-06.

## Agreement distribution

| Maximum agreeing votes | Count |
|---:|---:|
| 7 | 761 |
| 6 | 148 |
| 5 | 135 |
| 4 | 140 |
| 3 | 45 |
| 2 | 3 |

Majority tie resolution was required for 26 images.

## Frozen selective-referral series

| Acceptance rule | Coverage | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| 7/7 | 0.617695 | 0.925099 | 0.755570 |
| >=6/7 | 0.737825 | 0.886689 | 0.723554 |
| >=5/7 | 0.847403 | 0.848659 | 0.697422 |
| >=4/7 | 0.961039 | 0.804899 | 0.668536 |

The external cohort shows the same descriptive coverage-performance pattern as the internal cohort: stricter agreement retains fewer cases and yields higher retained point-estimate accuracy/macro-F1.

No HIBA-derived model selection, threshold selection, weight change, or tuning occurred.

## Interpretation boundary

The repeated point-estimate pattern across ISIC and HIBA supports proceeding to the prespecified inferential analysis. Statistical distinguishability is not claimed until ITT-06 patient-cluster/bootstrap results are available.

## Phase status

**ITT-05: PASS / CLOSED.**

Next: **ITT-06 — Statistical inference, publication tables, and figures.**
