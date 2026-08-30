# Phase 05 — Gate 05A Evidence Inventory and Consistency Audit

Date: 2026-08-28

## Status

Gate 05A: STARTED / INITIAL AUDIT PASS

This phase is reporting-only. No new training, model selection, checkpoint substitution, preprocessing change, threshold tuning, or test-driven rerun is permitted.

The internal test has already been consumed once and remains frozen.

## Authoritative Phase 04 source

- Repository: `1sazzad/Skin-Cancer-Hierarchical-Classification`
- Phase 04 branch: `phase04-controlled-comparative-evaluation`
- Phase 04 closure commit: `6c1d07bcd2d6bf93ce756340155cdc23f3eef01b`
- Closure message: `Complete Phase 04 final internal-test analysis`
- Authoritative artifact directory: `reports/phase04_controlled_comparative/final_internal_test/`
- Frozen internal-test N: `3668`

## Authoritative evidence files confirmed

1. `final_internal_test_summary.json`
2. `final_results_table.csv`
3. `final_paired_statistics.json`
4. `final_routing_analysis.json`
5. `final_classwise_comparison.csv`
6. `final_statistical_analysis.md`
7. `paired_internal_test_predictions.csv`

## Canonical end-to-end results

| System | Accuracy | Balanced accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|
| Flat four-class | 0.742094 | 0.650313 | **0.619222** | 0.752557 |
| Shared hierarchy — predicted gate | 0.687568 | 0.639536 | **0.568591** | 0.706832 |
| Shared hierarchy — oracle gate | 0.933751 | 0.779171 | **0.776976** | 0.933991 |

Canonical routing loss in macro-F1:

`0.7769758578867025 - 0.5685909456725847 = 0.2083849122141178`

Interpretation lock:

- The flat four-class classifier outperformed the deployed shared hierarchy.
- Oracle routing substantially improved hierarchical performance.
- Routing error is therefore a major end-to-end bottleneck under this frozen evaluation.
- Oracle-gate performance is diagnostic evidence, not a deployable performance claim.

## Paired overall comparison

Hierarchy minus flat macro-F1:

- Delta: `-0.050631522844312604`
- 95% CI: `[-0.07599324143900181, -0.02482655293932654]`

Hierarchy minus flat accuracy:

- Delta: `-0.05452562704471098`
- 95% CI: `[-0.06870229007633588, -0.040348964013086075]`

McNemar paired correctness comparison:

- Hierarchy-only correct: `274`
- Flat-only correct: `474`
- p-value: `2.49117637964427e-13`

Interpretation lock:

- Overall flat-model advantage is statistically supported under the paired internal-test analysis.
- Do not translate statistical significance into clinical significance.

## Four-class per-class F1 audit

| Class | Support | Hierarchy predicted-gate F1 | Flat F1 | Hierarchy - Flat | 95% paired bootstrap CI |
|---|---:|---:|---:|---:|---:|
| non_malignant | 2398 | 0.778597 | 0.821830 | -0.043233 | [-0.056035, -0.030510] |
| melanoma | 678 | 0.545669 | 0.609422 | -0.063753 | [-0.088293, -0.039765] |
| bcc | 498 | 0.659189 | 0.688496 | -0.029307 | [-0.056238, -0.002722] |
| scc | 94 | 0.290909 | 0.357143 | -0.066234 | [-0.150000, 0.019586] |

SCC interpretation lock:

- The point estimate favors the flat model.
- The SCC confidence interval crosses zero.
- Do not claim a reliable SCC-specific superiority.

## Shared versus standalone task audit

| Task | Shared macro-F1 | Standalone macro-F1 | Shared - Standalone |
|---|---:|---:|---:|
| Task 1 | 0.740624 | 0.774009 | -0.033385 |
| Task 2 | 0.702634 | 0.724875 | -0.022241 |
| Task 3 | 0.298710 | 0.275611 | +0.023099 |

Interpretation lock:

- Shared Task1 and Task2 underperformed their standalone counterparts.
- This is consistent with possible negative transfer.
- Causation is not established and must not be claimed.
- Task3 conclusions are unstable because rare T-category support is extremely small.

## Routing audit

Frozen routing evidence from the shared hierarchy:

- True malignant count: `1270`
- Correctly routed malignant: `1100`
- Malignant blocked by Stage 1: `170`
- Malignant block rate: `0.13385826771653545`
- True non-malignant count: `2398`
- Non-malignant incorrectly routed to Stage 2: `761`
- Non-malignant incorrect-route rate: `0.31734778982485407`
- Subtype error after correct malignant routing: `215 / 1100`
- Subtype error rate after correct route: `0.19545454545454546`

Publication interpretation:

Routing error is empirically important because the same downstream hierarchical subtype decision process improves from macro-F1 `0.568591` under predicted routing to `0.776976` under oracle routing. The oracle condition isolates routing as a major source of end-to-end degradation, but does not prove that routing is the only bottleneck.

## Efficiency evidence available for later publication synthesis

Batch-size-1 Tesla T4 benchmark from the frozen final summary:

| Model | Parameters | Latency ms/image | Throughput images/s | Peak CUDA memory bytes |
|---|---:|---:|---:|---:|
| Flat | 4,012,672 | 6.8456 | 146.0793 | 35,857,408 |
| Shared | 4,020,358 | 6.8650 | 145.6669 | 35,889,152 |

Initial interpretation:

The deployed flat and shared systems have nearly identical measured single-image inference efficiency in this benchmark. Efficiency should therefore not be presented as the explanation for the observed accuracy/macro-F1 difference.

## Claim-strength rules for Phase 05

### Supported claims

1. Flat four-class classification outperformed the deployed shared hierarchy on the frozen internal test.
2. The overall macro-F1 and accuracy differences are supported by paired confidence intervals that exclude zero.
3. McNemar analysis shows significantly more samples were correctly classified only by the flat model than only by the hierarchy.
4. Oracle routing substantially improves hierarchical end-to-end performance.
5. Routing error is a major end-to-end bottleneck in the evaluated hierarchical pipeline.
6. Shared Task1/Task2 underperformance is consistent with possible negative transfer.

### Claims that are not supported

1. State-of-the-art performance.
2. Clinical superiority or clinical readiness.
3. External clinical generalization.
4. Causal negative transfer.
5. Reliable hierarchy-versus-flat superiority for SCC.
6. Stable fine-grained Task3 T2/T3/T4 conclusions.
7. Oracle-gate macro-F1 as deployable real-world performance.
8. Routing error as the sole cause of hierarchical underperformance.

## Validation-to-test consistency

Phase 04 closure states that validation and final internal test agreed in direction:

- flat advantage replicated;
- routing-loss pattern replicated;
- standalone Task1/Task2 advantage broadly replicated.

Phase 05 may report this directional agreement as evidence of consistency, but must not imply that the internal test was used for additional model selection or tuning.

## Gate 05A initial verdict

**INITIAL PASS**

The handover numbers for the primary end-to-end comparison match the authoritative Phase 04 artifacts inspected at the start of Phase 05. No contradiction has been identified in the canonical flat, predicted-gate hierarchy, oracle-gate hierarchy, routing-loss, task-level, or classwise results.

## Next Phase 05 action

Continue Gate 05A by converting this inventory into a publication evidence map: each intended manuscript claim must be linked to its exact source artifact, metric, statistical qualifier, and required limitation. After that audit is complete, proceed to Gate 05B for publication-ready tables and figures.
