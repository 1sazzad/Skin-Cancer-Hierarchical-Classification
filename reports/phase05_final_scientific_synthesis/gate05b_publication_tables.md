# Phase 05 — Gate 05B Publication-Ready Tables

Date: 2026-08-28

All values below are copied from the frozen Phase 04 final internal-test evidence. No new model execution, tuning, thresholding, or checkpoint selection was performed.

## Table 1. End-to-end four-class performance

| System | N | Accuracy | Balanced accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| Flat four-class | 3668 | **0.742094** | 0.650313 | **0.619222** | **0.752557** |
| Shared hierarchy — predicted gate | 3668 | 0.687568 | 0.639536 | 0.568591 | 0.706832 |
| Shared hierarchy — oracle gate | 3668 | 0.933751 | **0.779171** | **0.776976** | 0.933991 |

**Paper use:** primary result table.

**Required note:** Oracle-gate performance is a diagnostic upper-bound-style condition using ground-truth routing and is not deployable system performance.

## Table 2. Paired flat-vs-hierarchy comparison

| Comparison | Point estimate | 95% CI | Interpretation |
|---|---:|---:|---|
| Hierarchy − Flat macro-F1 | -0.050632 | [-0.075993, -0.024827] | Favors flat; CI excludes zero |
| Hierarchy − Flat accuracy | -0.054526 | [-0.068702, -0.040349] | Favors flat; CI excludes zero |

McNemar paired correctness:

- Both correct: 2248
- Hierarchy only correct: 274
- Flat only correct: 474
- Both wrong: 672
- Exact two-sided p = 2.49118e-13

**Paper use:** statistical support for the primary comparative conclusion.

## Table 3. Per-class F1 comparison

| Class | Support | Hierarchy F1 | Flat F1 | Delta H−F | 95% CI |
|---|---:|---:|---:|---:|---:|
| Non-malignant | 2398 | 0.778597 | **0.821830** | -0.043233 | [-0.056035, -0.030510] |
| Melanoma | 678 | 0.545669 | **0.609422** | -0.063753 | [-0.088293, -0.039765] |
| BCC | 498 | 0.659189 | **0.688496** | -0.029307 | [-0.056238, -0.002722] |
| SCC | 94 | 0.290909 | 0.357143 | -0.066234 | [-0.150000, 0.019586] |

**Required note:** SCC confidence interval crosses zero; do not claim a reliable class-specific difference for SCC.

## Table 4. Shared versus standalone task performance

| Task | Shared macro-F1 | Standalone macro-F1 | Shared − Standalone |
|---|---:|---:|---:|
| Task 1 | 0.740624 | **0.774009** | -0.033385 |
| Task 2 | 0.702634 | **0.724875** | -0.022241 |
| Task 3 | **0.298710** | 0.275611 | +0.023099 |

**Required note:** Task1/Task2 results are consistent with possible negative transfer but do not establish causation or paired statistical significance. Task3 is highly unstable because T2/T3/T4 supports are 7/2/1.

## Table 5. Routing decomposition

| Routing quantity | Value |
|---|---:|
| True malignant | 1270 |
| Correctly routed malignant | 1100 |
| Malignant blocked by Stage 1 | 170 |
| Malignant block rate | 13.39% |
| True non-malignant | 2398 |
| Non-malignant incorrectly routed to Stage 2 | 761 |
| Non-malignant incorrect-route rate | 31.73% |
| Correctly routed malignant with subtype error | 215 / 1100 |
| Subtype error rate after correct route | 19.55% |
| Predicted-gate macro-F1 | 0.568591 |
| Oracle-gate macro-F1 | 0.776976 |
| Routing loss | **0.208385** |

## Recommended manuscript ordering

1. Table 1 in the main Results section.
2. Table 2 immediately after the primary result paragraph or condensed into prose if page-limited.
3. Table 3 as main or supplementary evidence depending on venue page limit.
4. Table 5 paired with the routing figure in the Discussion/Failure Analysis subsection.
5. Table 4 should appear after the routing analysis as secondary evidence, not as the headline result.

## Gate 05B table verdict

PASS. The canonical publication tables are now frozen for Phase 05 synthesis.
