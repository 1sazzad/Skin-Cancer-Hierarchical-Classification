# Phase 05 — Gate 05D Publication-Ready Evidence Package and Claim Audit

Date: 2026-08-28

## Gate status

**Gate 05D: PASS**

This gate audits the Phase 05 publication package against the frozen Phase 04 evidence. No model training, inference, tuning, threshold optimization, checkpoint substitution, preprocessing change, or reuse of the consumed internal test occurred.

## Publication package inventory

### Primary manuscript evidence

1. `gate05b_publication_tables.md`
   - End-to-end four-class performance
   - Paired flat-vs-hierarchy statistics
   - Per-class F1 comparison
   - Shared-vs-standalone task comparison
   - Routing decomposition

2. `figures/fig1_end_to_end_macro_f1.png`
   - Primary end-to-end macro-F1 comparison

3. `figures/fig2_routing_loss.png`
   - Predicted-gate versus oracle-gate diagnostic

4. `figures/fig3_classwise_f1.png`
   - Per-class F1 comparison

5. `figures/fig4_classwise_delta_ci.png`
   - Hierarchy-minus-flat per-class F1 differences with 95% confidence intervals

6. `gate05c_final_scientific_synthesis.md`
   - Results
   - Discussion
   - Limitations
   - Contribution statements
   - Conclusion

### Reproducibility and provenance support

- `scripts/reporting/phase05_make_publication_figures.py`
- Phase 04 authoritative evidence under `reports/phase04_controlled_comparative/final_internal_test/`
- Phase 05 Gate 05A evidence inventory and publication evidence map

## Canonical headline evidence

| Quantity | Frozen value | Publication role |
|---|---:|---|
| Internal-test N | 3668 | Evaluation population |
| Flat macro-F1 | 0.619222 | Deployable flat result |
| Predicted-gate hierarchy macro-F1 | 0.568591 | Deployable hierarchy result |
| Oracle-gate hierarchy macro-F1 | 0.776976 | Diagnostic routing condition only |
| Hierarchy − Flat macro-F1 | -0.050632 | Primary effect estimate |
| Macro-F1 95% CI | [-0.075993, -0.024827] | Paired uncertainty |
| Hierarchy − Flat accuracy | -0.054526 | Secondary effect estimate |
| Accuracy 95% CI | [-0.068702, -0.040349] | Paired uncertainty |
| Flat-only correct | 474 | Paired correctness evidence |
| Hierarchy-only correct | 274 | Paired correctness evidence |
| McNemar p | 2.49118e-13 | Accuracy discordance only |
| Routing loss | 0.208385 | Predicted-versus-oracle macro-F1 gap |

## Reviewer-style claim audit

### Claim 1 — Flat model outperformed the deployed hierarchy

**Status: SUPPORTED.**

Permitted wording:

> On the frozen internal test, the flat four-class classifier achieved higher macro-F1 and accuracy than the shared predicted-gate hierarchy.

Supporting evidence:

- Macro-F1: 0.619222 versus 0.568591
- Accuracy: 0.742094 versus 0.687568
- Paired confidence intervals for both overall differences exclude zero
- Paired correctness discordance favors flat

Do not extend this to clinical superiority, state-of-the-art performance, or external generalization.

### Claim 2 — The primary difference is statistically supported

**Status: SUPPORTED WITH TEST-SPECIFIC WORDING.**

Permitted wording:

> The paired bootstrap confidence interval for the macro-F1 difference excluded zero, while the paired accuracy difference was also supported by its confidence interval and exact McNemar analysis.

Required distinction:

- Bootstrap CI supports the macro-F1 effect estimate under the specified resampling scheme.
- McNemar tests paired correctness/accuracy discordance; it is not a macro-F1 significance test.

### Claim 3 — Routing is a major bottleneck

**Status: SUPPORTED.**

Permitted wording:

> Oracle routing increased hierarchical macro-F1 from 0.568591 to 0.776976, yielding a routing-associated gap of 0.208385 and identifying routing as a major source of end-to-end degradation.

Required qualifiers:

- Oracle routing uses ground-truth gate information.
- Oracle performance is diagnostic and non-deployable.
- Routing is a major source, not proven to be the sole source, of degradation.

### Claim 4 — The hierarchy has latent conditional predictive capability

**Status: SUPPORTED AS A DIAGNOSTIC INTERPRETATION.**

Permitted wording:

> The oracle-routing and conditional Task 2 results indicate that downstream hierarchical prediction retained substantial conditional capability that was not fully realized end to end.

Do not write that the deployed hierarchy is superior to flat based on oracle results.

### Claim 5 — Shared learning may involve negative transfer

**Status: TENTATIVE / HYPOTHESIS-GENERATING ONLY.**

Permitted wording:

> Lower shared Task 1 and Task 2 scores are consistent with possible negative transfer or insufficient task-specific specialization.

Prohibited wording:

- “Negative transfer caused the performance loss.”
- “The shared backbone harms Tasks 1 and 2.”
- Any causal attribution not established by a controlled causal experiment.

### Claim 6 — SCC favors flat

**Status: POINT ESTIMATE ONLY; RELIABLE DIFFERENCE NOT ESTABLISHED.**

Frozen SCC evidence:

- Flat F1: 0.357143
- Hierarchy F1: 0.290909
- Delta H−F: -0.066234
- 95% CI: [-0.150000, 0.019586]
- Support: 94

Required wording:

> The SCC point estimate favored the flat model, but the confidence interval crossed zero and the support was small; no reliable SCC-specific difference is claimed.

### Claim 7 — Task 3 supports a performance conclusion

**Status: STRONG CONCLUSION NOT SUPPORTED.**

Frozen sparse supports:

- T2: 7
- T3: 2
- T4: 1

Permitted wording:

> Task 3 results are descriptive and highly unstable because of extremely sparse advanced T-category support.

### Claim 8 — Shared architecture is more storage-efficient than three separate task models

**Status: SUPPORTED AS A STORAGE/ARCHITECTURAL COMPARISON.**

Permitted wording:

> The shared model stores one encoder with three heads and requires approximately one third of the total parameters/checkpoint storage of the three independently stored task models.

Required qualifier:

This is not a full conditional deployment-latency comparison.

### Claim 9 — Validation and internal test are directionally consistent

**Status: SUPPORTED WITHIN STUDY.**

Permitted wording:

> The primary flat-versus-hierarchy direction and routing-loss pattern were consistent across the predefined validation and internal-test partitions.

Required qualifier:

This does not establish external robustness or clinical generalization.

## Figure audit and manuscript captions

### Figure 1 — End-to-end four-class performance

Recommended caption:

> **Figure 1. End-to-end macro-F1 on the frozen internal test.** The flat four-class classifier outperformed the deployable shared hierarchy using predicted routing (0.619 versus 0.569 macro-F1). Oracle routing increased hierarchical macro-F1 to 0.777; this oracle condition uses ground-truth routing and is included only as a diagnostic reference, not as deployable performance.

### Figure 2 — Routing-error diagnostic

Recommended caption:

> **Figure 2. Predicted-versus-oracle routing diagnostic for the shared hierarchy.** Replacing predicted routing with ground-truth routing increased macro-F1 from 0.569 to 0.777, a gap of 0.208. The oracle result isolates the effect associated with routing decisions but does not establish that routing is the only source of error.

### Figure 3 — Per-class F1 comparison

Recommended caption:

> **Figure 3. Per-class F1 for the flat classifier and deployed shared hierarchy.** Point estimates favored the flat model for all four output classes. SCC should be interpreted cautiously because only 94 cases were available and its paired confidence interval included zero.

### Figure 4 — Per-class paired differences

Recommended caption:

> **Figure 4. Paired hierarchy-minus-flat classwise F1 differences with 95% bootstrap confidence intervals.** Negative values favor the flat classifier. Intervals excluded zero for non-malignant, melanoma, and BCC, whereas the SCC interval crossed zero.

## Recommended Results-section evidence order

1. Present Table 1 and Figure 1 first.
2. Immediately report hierarchy-minus-flat effect estimates and paired confidence intervals.
3. Report McNemar only in the context of paired correctness/accuracy.
4. Present classwise evidence, explicitly preserving the SCC uncertainty qualifier.
5. Present Figure 2 and routing counts as the principal failure-analysis result.
6. Present shared-versus-standalone task evidence as secondary exploratory interpretation.
7. Present efficiency/storage evidence after predictive results rather than as a headline claim.

## Mandatory limitations to retain in any manuscript compression

Even if page limits require shortening, the following limitations must remain visible:

1. Internal rather than external/prospective/clinical evaluation.
2. One-time consumed internal test; no post-test tuning.
3. Oracle routing is diagnostic and non-deployable.
4. SCC uncertainty due to small support and CI crossing zero.
5. Task 3 instability due to T2/T3/T4 supports of 7/2/1.
6. Shared-versus-standalone comparisons do not establish causal negative transfer.
7. McNemar supports paired accuracy/correctness, not macro-F1.
8. Efficiency benchmarks do not measure full independently routed deployment latency.

## Prohibited publication claims

The Phase 05 evidence does **not** support any of the following:

- State-of-the-art or best-known performance.
- Clinical superiority, safety, readiness, utility, or deployment recommendation.
- External population or multi-centre generalization.
- Causal negative transfer.
- Reliable SCC-specific superiority.
- Stable rare Task 3 T-category conclusions.
- Oracle-gate macro-F1 as operational performance.
- Routing as the sole cause of hierarchical underperformance.
- Full deployment-speed superiority from isolated single-forward-pass benchmarks.

## Final publication message

The paper should be framed as a controlled comparative and failure-analysis study rather than a leaderboard-performance paper. The strongest defensible message is:

> Under a frozen internal evaluation protocol, the flat four-class classifier provided stronger deployable performance than the shared hard-routing hierarchy. However, the large predicted-versus-oracle routing gap showed that the hierarchy retained substantial conditional predictive capability and that routing error was a major end-to-end bottleneck. Secondary task comparisons suggest possible shared-representation trade-offs, while rare-class and external-generalization limitations remain unresolved.

## Gate 05D verdict

**PASS.** The publication tables, figures, synthesis, captions, statistical wording, and claim boundaries are mutually consistent with the frozen Phase 04 evidence. No numerical contradiction or unsupported headline claim was identified. The evidence package is ready for Phase 05 closure and handover.
