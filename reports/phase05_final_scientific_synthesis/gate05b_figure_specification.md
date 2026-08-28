# Phase 05 — Gate 05B Figure Specification

Date: 2026-08-28

All figures are derived only from frozen Phase 04 internal-test artifacts. No new model inference is involved.

## Figure 1 — End-to-end four-class macro-F1

Bars:

- Flat four-class: 0.619222
- Shared hierarchy, predicted gate: 0.568591
- Shared hierarchy, oracle gate: 0.776976

Purpose: visually communicate the primary comparison and the latent performance revealed by oracle routing.

Caption guidance:

> End-to-end four-class macro-F1 on the frozen internal test. The deployed shared hierarchy under predicted routing underperformed the flat classifier, whereas oracle routing substantially improved hierarchical performance. Oracle-gate performance is diagnostic and does not represent deployable performance.

## Figure 2 — Routing-loss visualization

Bars:

- Predicted gate: 0.568591
- Oracle gate: 0.776976

Annotated difference:

- Routing loss: 0.208385 macro-F1

Purpose: isolate the impact associated with routing decisions.

Caption guidance:

> Predicted-gate versus oracle-gate hierarchical macro-F1. The 0.2084 macro-F1 gap indicates substantial end-to-end degradation associated with routing errors.

## Figure 3 — Per-class F1, flat vs deployed hierarchy

Values:

| Class | Flat F1 | Hierarchy F1 |
|---|---:|---:|
| Non-malignant | 0.821830 | 0.778597 |
| Melanoma | 0.609422 | 0.545669 |
| BCC | 0.688496 | 0.659189 |
| SCC | 0.357143 | 0.290909 |

Purpose: show that the flat-model advantage is directionally present across all four classes.

Required caution: SCC support is only 94 and its paired confidence interval crosses zero.

## Figure 4 — Paired classwise F1 deltas with 95% CI

Plot hierarchy-minus-flat F1 difference:

| Class | Delta H−F | 95% CI |
|---|---:|---:|
| Non-malignant | -0.043233 | [-0.056035, -0.030510] |
| Melanoma | -0.063753 | [-0.088293, -0.039765] |
| BCC | -0.029307 | [-0.056238, -0.002722] |
| SCC | -0.066234 | [-0.150000, 0.019586] |

Purpose: show uncertainty rather than only point estimates.

Caption guidance:

> Paired classwise F1 differences (hierarchy minus flat) with 95% stratified-bootstrap confidence intervals. Negative values favor the flat model. The SCC interval crosses zero, so the SCC-specific difference is uncertain.

## Recommended primary paper figures

If page space is limited, prioritize:

1. Figure 1 — end-to-end comparison.
2. Figure 2 — routing loss.
3. Figure 4 — paired classwise uncertainty.

Figure 3 is useful but partially redundant with Figure 4 and may move to supplementary material.

## Gate 05B figure verdict

PASS. Figure content and claims are publication-ready and scientifically bounded.
