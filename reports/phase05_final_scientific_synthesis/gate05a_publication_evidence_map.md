# Phase 05 — Gate 05A Publication Evidence Map

Date: 2026-08-28

## Gate status

**Gate 05A: PASS**

The publication-facing claims below are mapped to frozen Phase 04 artifacts. This map is the claim-control layer for subsequent Phase 05 tables, figures, Results, Discussion, and Conclusion drafting.

No new model training, test-driven tuning, checkpoint changes, preprocessing changes, or threshold changes are authorized.

## Claim map

| ID | Intended publication claim | Authoritative evidence | Required statistical qualifier | Mandatory limitation / wording guardrail |
|---|---|---|---|---|
| C1 | The flat four-class classifier outperformed the deployed shared predicted-gate hierarchy on the frozen internal test. | `final_results_table.csv`; `final_internal_test_summary.json`; `final_statistical_analysis.md` | Report macro-F1 0.619222 vs 0.568591 and accuracy 0.742094 vs 0.687568. | Say "on the frozen internal test"; do not imply external or clinical superiority. |
| C2 | The overall macro-F1 difference favored the flat classifier. | `final_paired_statistics.json`; `final_statistical_analysis.md` | Hierarchy−flat delta = -0.050632; paired 95% CI [-0.075993, -0.024827]. | Prefer effect size + CI; do not rely on significance language alone. |
| C3 | The overall accuracy difference favored the flat classifier. | `final_paired_statistics.json`; `final_statistical_analysis.md` | Hierarchy−flat delta = -0.054526; paired 95% CI [-0.068702, -0.040349]. | Same internal-test limitation. |
| C4 | Paired correctness discordance favored the flat classifier. | `final_paired_statistics.json`; `final_statistical_analysis.md` | Flat-only correct = 474; hierarchy-only correct = 274; exact two-sided McNemar p=2.49118e-13. | McNemar supports accuracy discordance, not macro-F1. |
| C5 | Oracle routing substantially improved hierarchical four-class macro-F1. | `final_internal_test_summary.json`; `final_results_table.csv`; `final_routing_analysis.json` | Predicted gate = 0.568591; oracle gate = 0.776976; difference = 0.208385. | Oracle routing is a diagnostic upper-bound-style condition, not deployable performance. |
| C6 | Routing error is a major end-to-end bottleneck in the evaluated hierarchy. | `final_routing_analysis.json`; `final_statistical_analysis.md` | 170/1270 malignant cases blocked at Task1; 761/2398 non-malignant cases incorrectly routed to Task2; oracle-predicted macro-F1 gap 0.208385. | Use "a major source/bottleneck"; do not say routing is the sole cause. |
| C7 | Conditional malignant subtype classification remained comparatively strong when routing was correct/oracle-controlled. | `final_internal_test_summary.json`; `final_statistical_analysis.md` | Shared Task2 malignant-subset macro-F1 = 0.702634; oracle four-class macro-F1 = 0.776976. | Keep distinction between conditional Task2 performance and deployed end-to-end performance. |
| C8 | Shared Task1 and Task2 underperformed their standalone counterparts. | `final_results_table.csv`; `final_statistical_analysis.md` | Task1 shared−standalone macro-F1 = -0.033385; Task2 = -0.022241. | These task comparisons do not establish causal negative transfer or paired statistical significance. |
| C9 | Shared-representation results are consistent with possible negative transfer for Task1/Task2. | Same as C8 | Descriptive directional evidence only. | Must say "consistent with possible" or "suggests"; never claim causality. |
| C10 | Task3 evidence is too sparse for strong rare-T-category conclusions. | `final_statistical_analysis.md`; Phase 04 summary artifacts | T2/T3/T4 supports = 7/2/1. | Explicitly characterize class-level estimates as unstable. |
| C11 | The flat point estimate for SCC F1 exceeded the predicted-gate hierarchy, but the difference is uncertain. | `final_classwise_comparison.csv`; `final_statistical_analysis.md` | Flat F1 = 0.357143; hierarchy F1 = 0.290909; delta = -0.066234; 95% CI [-0.150000, 0.019586]. | State that the CI crosses zero; no reliable SCC-specific superiority claim. |
| C12 | Flat-vs-hierarchy direction replicated from validation to the final internal test. | `final_statistical_analysis.md` | Validation H−F macro-F1 -0.070285; test -0.050632. Validation accuracy -0.063795; test -0.054526. | Report only directional consistency; the internal test was not used for further model selection/tuning. |
| C13 | Routing-loss direction replicated from validation to test. | `final_statistical_analysis.md` | Validation routing loss = 0.192480; test = 0.208385. | Do not reinterpret this post hoc as a tuning signal. |
| C14 | Flat and shared models had nearly identical single-forward-pass efficiency in the frozen Tesla T4 benchmark. | `final_internal_test_summary.json`; `final_statistical_analysis.md` | Flat latency 6.8456 ms/image vs shared 6.8650; 146.08 vs 145.67 img/s; ~4.01M vs ~4.02M params. | This is not a full conditional hierarchy deployment benchmark; FLOPs/MACs unavailable. |
| C15 | A shared encoder with three heads is materially smaller in stored parameter count than three independently stored task models. | `final_statistical_analysis.md` | Shared = 4,020,358 parameters; independent Tasks1–3 sum = 12,035,454 (~3.0× shared). | Storage comparison only; do not equate directly with end-to-end latency. |

## Canonical paper-facing primary result

The primary paper result must be presented in this order:

1. **Deployed comparison:** flat four-class macro-F1 `0.619222` vs shared predicted-gate hierarchy `0.568591`.
2. **Paired evidence:** hierarchy−flat macro-F1 `-0.050632`, 95% CI `[-0.075993, -0.024827]`; accuracy delta `-0.054526`, 95% CI `[-0.068702, -0.040349]`.
3. **Discordance evidence:** McNemar flat-only correct `474` vs hierarchy-only correct `274`, `p=2.49118e-13`.
4. **Mechanistic diagnostic:** oracle-gate hierarchy macro-F1 `0.776976`, creating routing loss `0.208385` relative to predicted routing.
5. **Interpretation:** the deployed flat classifier is stronger under the frozen protocol, while the oracle diagnostic shows that the hierarchical decomposition retains substantial latent performance when routing errors are removed.

## Results-section wording policy

Use factual language:

- "outperformed on the frozen internal test"
- "the paired confidence interval excluded zero"
- "oracle routing increased macro-F1"
- "routing error was a major source of end-to-end degradation"
- "consistent with possible negative transfer"
- "uncertain for SCC because the confidence interval crossed zero"

Avoid:

- "state of the art"
- "clinically superior"
- "clinically deployable"
- "proves negative transfer"
- "proves hierarchy is better"
- "routing is the only problem"
- "generalizes to clinical populations"

## Figure/table evidence policy

Every Phase 05 figure or table must satisfy all of the following:

1. Derived only from frozen Phase 04 artifacts.
2. No threshold search, model selection, or new inference based on internal-test outcomes.
3. Oracle-gate results visually labeled as diagnostic/oracle, never as deployed performance.
4. Confidence intervals shown where a comparative inferential claim is made.
5. SCC and Task3 limitations retained in caption or accompanying text when those results are displayed.
6. Validation and test numbers visually separated to avoid implying that test informed model selection.

## Gate 05A closure verdict

**PASS.**

The Phase 04 closure commit, authoritative final-test artifacts, primary numerical results, paired statistical evidence, routing analysis, classwise evidence, task-level evidence, validation-to-test directional consistency, and efficiency evidence have been checked and mapped to publication claims. No numerical contradiction was found among the inspected authoritative artifacts.

## Next gate

**Gate 05B — Publication-Ready Tables and Figures.**

Gate 05B may create tables and visualizations from the frozen evidence only. It must not run training or perform any form of post-test model optimization.
