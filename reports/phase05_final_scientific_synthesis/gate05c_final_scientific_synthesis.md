# Phase 05 — Gate 05C Final Scientific Synthesis

Date: 2026-08-28

## Gate status

**Gate 05C: PASS**

This synthesis is derived exclusively from the frozen Phase 04 validation and final internal-test evidence. The internal test was evaluated once under the frozen protocol. No model was trained, tuned, reselected, recalibrated, or rerun during Phase 05.

## Publication-ready Results

### End-to-end four-class classification

The primary comparison evaluated the flat four-class classifier against the shared hierarchy under its deployable predicted-gate routing condition on 3,668 frozen internal-test samples. The flat classifier achieved an accuracy of 0.742094, balanced accuracy of 0.650313, macro-F1 of 0.619222, and weighted F1 of 0.752557. The shared predicted-gate hierarchy achieved an accuracy of 0.687568, balanced accuracy of 0.639536, macro-F1 of 0.568591, and weighted F1 of 0.706832. Thus, the flat classifier achieved higher observed end-to-end macro-F1 and accuracy under the frozen evaluation protocol (Table 1; Figure 1).

The paired hierarchy-minus-flat macro-F1 difference was −0.050632. Its 95% paired, ground-truth-class-stratified bootstrap percentile confidence interval was [−0.075993, −0.024827], based on 10,000 replicates with seed 42. The corresponding accuracy difference was −0.054526, with a 95% confidence interval of [−0.068702, −0.040349]. Both intervals excluded zero and favored the flat classifier. At the sample level, both systems were correct for 2,248 cases and both were wrong for 672 cases; the flat classifier alone was correct for 474 cases, compared with 274 cases for the hierarchy alone. The exact two-sided McNemar p-value was 2.49118 × 10⁻¹³. This test supports an asymmetry in paired correctness and therefore the accuracy comparison; it is not a test of the macro-F1 difference.

### Class-specific performance

The flat classifier had higher observed F1 than the predicted-gate hierarchy for each of the four output classes (Table 3; Figures 3–4). For non-malignant lesions, F1 was 0.821830 for the flat model and 0.778597 for the hierarchy, giving a hierarchy-minus-flat difference of −0.043233 (95% CI [−0.056035, −0.030510]). For melanoma, the corresponding values were 0.609422 and 0.545669, with a difference of −0.063753 (95% CI [−0.088293, −0.039765]). For basal cell carcinoma, F1 was 0.688496 and 0.659189, with a difference of −0.029307 (95% CI [−0.056238, −0.002722]).

For squamous cell carcinoma, the flat and hierarchical F1 estimates were 0.357143 and 0.290909, respectively. Although the point estimate again favored the flat classifier, the hierarchy-minus-flat difference of −0.066234 had a 95% confidence interval of [−0.150000, 0.019586]. The interval crossed zero, and SCC support was only 94; consequently, the evidence does not support a reliable SCC-specific difference between systems.

### Routing-error decomposition

Oracle routing was used as a diagnostic condition to separate subtype classification capability from upstream routing error. Under oracle routing, the shared hierarchy achieved four-class accuracy of 0.933751, balanced accuracy of 0.779171, macro-F1 of 0.776976, and weighted F1 of 0.933991. Relative to the predicted-gate macro-F1 of 0.568591, this produced a routing loss of 0.208385 (Figure 2). Because oracle routing uses the ground-truth gate, it is not deployable system performance and should not be interpreted as a competing operational result.

The routing counts identify two distinct sources of error. Of 1,270 malignant cases, 170 were blocked by Task 1, corresponding to a malignant block rate of 13.39%; 1,100 malignant cases were routed correctly. Of 2,398 non-malignant cases, 761 were incorrectly sent to Task 2, corresponding to an incorrect-route rate of 31.73%. Stage 2 was executed for 2,031 samples, or 55.37% of the internal test. Among correctly routed malignant cases, 215 of 1,100 had an incorrect subtype prediction, giving a conditional subtype error rate of 19.55%. The large predicted-versus-oracle macro-F1 gap, together with the observed routing counts, identifies routing as a major source of end-to-end degradation, but not necessarily its sole source.

Conditional subtype performance was stronger than deployed end-to-end hierarchical performance. On the true malignant subset, the shared Task 2 head achieved accuracy of 0.808661 and macro-F1 of 0.702634. This distinction is important: Task 2 can perform comparatively well when evaluated on the appropriate malignant subset, while the deployed hierarchy remains vulnerable to errors introduced before or during routing.

### Shared and standalone task models

The shared Task 1 head achieved macro-F1 of 0.740624 and accuracy of 0.746183, compared with 0.774009 and 0.786260 for standalone Task 1. The shared-minus-standalone differences were therefore −0.033385 in macro-F1 and −0.040076 in accuracy. For Task 2, the shared head achieved macro-F1 of 0.702634 and accuracy of 0.808661, compared with 0.724875 and 0.833858 for the standalone model; the corresponding differences were −0.022241 and −0.025197.

Task 3 showed a mixed pattern. The shared head achieved macro-F1 of 0.298710 compared with 0.275611 for the standalone model, a difference of +0.023099, while its accuracy was lower by 0.070866 (0.472441 versus 0.543307). These Task 3 results must be interpreted cautiously because the complete Task 3 sample contained only 127 cases and the T2, T3, and T4 supports were 7, 2, and 1, respectively. Such sparse categories make class-level estimates highly unstable.

The Task 1 and Task 2 results are consistent with possible negative transfer or insufficient shared-task specialization. However, these descriptive comparisons were not supported by task-specific paired confidence intervals, and they do not establish that shared learning caused the observed differences.

### Validation-to-test consistency

The direction of the primary comparison replicated from validation to the final internal test. The hierarchy-minus-flat macro-F1 difference was −0.070285 on validation and −0.050632 on the internal test. The corresponding accuracy differences were −0.063795 and −0.054526. Routing loss was also consistent, increasing from 0.192480 on validation to 0.208385 on the internal test. All four observed class-level F1 differences favored the flat classifier on both splits, although the final SCC interval included zero. The standalone advantage for Tasks 1 and 2 also remained directionally consistent, while Task 3 continued to show a mixed and support-limited pattern.

These findings demonstrate directional consistency within the project’s predefined data partitions. They do not constitute evidence of external, prospective, or clinical generalization. Importantly, the internal-test results were not used for additional model selection or tuning.

### Efficiency and storage

In the frozen batch-size-one Tesla T4 benchmark, the flat and shared models had nearly identical single-forward-pass efficiency. The flat model required 6.8456 ms per image and achieved 146.08 images per second, while the shared model required 6.8650 ms per image and achieved 145.67 images per second. Peak CUDA memory was 35,857,408 bytes for the flat model and 35,889,152 bytes for the shared model. Their parameter counts were also similar: 4,012,672 for the flat model and 4,020,358 for the shared model. FLOPs and MACs were unavailable because the measurement backend did not support them.

The shared architecture nevertheless has a storage advantage over an independently stored three-model hierarchy. Its shared encoder and three heads contain 4,020,358 parameters in one checkpoint of 48,737,992 bytes. The three standalone Task 1–3 models contain 12,035,454 parameters and occupy 145,891,211 checkpoint bytes in total, approximately three times the shared storage. This is an architectural and storage comparison, not a direct measurement of full conditional deployment latency. A single standalone forward pass should not be treated as the complete runtime cost of the independently routed hierarchy.

## Discussion

The central result is that hierarchical task structure did not translate into stronger deployed four-class performance under the frozen protocol. The flat classifier exceeded the shared predicted-gate hierarchy in both macro-F1 and accuracy, and the paired confidence intervals indicated that the observed differences were consistently directed toward the flat model across bootstrap resamples. Paired correctness discordance likewise favored the flat classifier. The result was not confined to the majority non-malignant class: point estimates favored the flat model for melanoma, BCC, and SCC as well, although the SCC-specific interval was too wide to support a reliable difference.

The oracle-routing diagnostic provides the most important mechanistic interpretation. When ground-truth routing removed Task 1 gate errors, hierarchical macro-F1 increased from 0.568591 to 0.776976. This result does not make the oracle system deployable, nor does it show that the hierarchy is operationally superior. Instead, it demonstrates that the hierarchical decomposition retained substantial conditional predictive capability that was not realized end to end. The 0.208385 routing loss and the observed malignant-block and non-malignant false-route counts make upstream routing a clear bottleneck.

Routing errors are especially consequential in a hard hierarchy because they constrain all downstream decisions. A malignant case blocked at Task 1 cannot receive a malignant subtype prediction, regardless of Task 2 quality. Conversely, a non-malignant case sent to Task 2 must be assigned a malignant subtype in the final four-class output. The latter failure mode was common in this evaluation: 761 non-malignant cases were incorrectly routed, compared with 170 malignant cases blocked. These asymmetric consequences help explain why comparatively strong conditional Task 2 performance did not yield competitive deployed four-class performance.

The shared-versus-standalone comparisons add a second, more tentative interpretation. Shared heads underperformed standalone models for Tasks 1 and 2, while Task 3 had higher shared macro-F1 but lower shared accuracy. The Task 1 and Task 2 pattern is consistent with possible negative transfer, competition for shared representational capacity, or a mismatch among task-specific optimization requirements. However, multiple explanations remain compatible with the evidence, and the present descriptive comparison cannot isolate a causal mechanism. The Task 3 evidence is still less decisive because several T categories have extremely small support.

The architecture therefore presents a meaningful trade-off rather than a uniformly better or worse design. The shared model reduces stored parameters and checkpoint size by approximately two thirds relative to three independently stored task models and has essentially the same measured single-pass cost as the flat classifier. In exchange, the evaluated hard-routing design incurs error propagation that materially reduces deployed performance. Under the current evidence, compact shared storage does not offset the predictive cost of the frozen routing behavior when macro-F1 is the primary objective.

The consistency between validation and internal-test directions strengthens confidence that the main finding was not unique to one project partition. The effect was somewhat smaller on the internal test than on validation, while routing loss was slightly larger. This supports a robust within-study conclusion: the flat model was the stronger deployed four-class system, and routing remained a substantial limitation of the hierarchy. It does not extend that conclusion beyond the evaluated data sources or justify post-test modification of the frozen models.

## Limitations

First, the final evidence comes from one internal-test partition constructed within the project’s datasets. It is not an external, multi-centre, prospective, or clinical evaluation, and no claim about clinical safety, clinical utility, or population-level generalization is supported.

Second, the internal test was consumed once, as required by the frozen protocol. This protects the evaluation from test-driven optimization, but it also means that alternative thresholds, routing mechanisms, checkpoints, or architectural changes cannot be evaluated on the same test without violating the protocol. The oracle result is therefore diagnostic rather than a basis for post-test tuning.

Third, class imbalance limits minority-class precision. SCC had only 94 internal-test examples, and its paired classwise interval crossed zero. Task 3 was particularly sparse: T2, T3, and T4 contained 7, 2, and 1 examples. Task 3 class-level comparisons should consequently be treated as unstable descriptive estimates.

Fourth, the bootstrap confidence intervals quantify uncertainty under paired, ground-truth-class-stratified resampling of the observed internal-test cases. They do not account for uncertainty from retraining, alternative seeds, dataset construction, acquisition sites, label noise, or distribution shift. McNemar’s test addresses paired correctness and the accuracy comparison; it does not test macro-F1 or establish clinical importance.

Fifth, the shared-versus-standalone task comparisons are descriptive. Without paired task-specific intervals or a controlled causal study of representation sharing, lower shared Task 1 and Task 2 scores cannot prove negative transfer.

Sixth, efficiency measurements represent isolated batch-size-one forward passes on a Tesla T4. They do not directly benchmark a complete conditionally executed independent hierarchy, data-transfer overhead, preprocessing, concurrency, energy consumption, or production latency. FLOPs and MACs were unavailable.

Finally, checkpoint identities were frozen and verified through the evaluation configuration and runner, but the final summary does not duplicate all checkpoint hashes. The provenance chain is internally consistent, although duplicating immutable checkpoint identifiers in future final summaries would make the evidence package more self-contained.

## Contribution statements

This project makes the following bounded contributions:

1. It defines and evaluates a three-task skin-lesion hierarchy that separates malignancy gating, malignant subtype classification, and melanoma T-category prediction while accommodating task-specific label availability.

2. It provides a controlled comparison between a shared multi-head hierarchy, standalone task models, and a flat four-class classifier under frozen checkpoints, preprocessing, class order, routing rules, and model-selection decisions.

3. It evaluates the deployed hierarchy using paired sample-level evidence, including 10,000-replicate paired stratified-bootstrap confidence intervals and an exact two-sided McNemar analysis for correctness discordance.

4. It separates end-to-end performance from conditional subtype capability through a predicted-versus-oracle routing diagnostic and a count-based decomposition of malignant blocks, non-malignant false routes, and subtype errors after correct routing.

5. It quantifies the empirical trade-off between predictive performance and model consolidation: the shared architecture provides approximately threefold lower task-model storage than three independently stored models, while the frozen flat classifier provides stronger deployed four-class performance.

6. It preserves a one-time frozen internal-test protocol and a claim-to-evidence map that constrains publication language to the observed internal evidence and explicitly retains minority-class, Task 3, statistical, efficiency, and generalization limitations.

## Conclusion

On the frozen internal test, the flat four-class classifier was the stronger deployed system, achieving macro-F1 of 0.619222 compared with 0.568591 for the shared predicted-gate hierarchy. The hierarchy-minus-flat difference was −0.050632, with a paired 95% confidence interval of [−0.075993, −0.024827]; accuracy and paired correctness evidence pointed in the same direction. The primary comparative direction replicated from validation to test.

The hierarchy’s lower deployed performance should not be interpreted as evidence that hierarchical decomposition lacks predictive value. Under diagnostic oracle routing, hierarchical macro-F1 reached 0.776976, revealing a routing loss of 0.208385 and identifying routing as a major end-to-end bottleneck. Conditional Task 2 performance further showed that downstream subtype classification remained comparatively strong when evaluated on the appropriate malignant subset.

The shared architecture reduced storage substantially, but shared Tasks 1 and 2 underperformed their standalone counterparts, a pattern consistent with possible negative transfer without proving it. SCC uncertainty and extremely sparse Task 3 categories prevent strong rare-class conclusions. Overall, the study supports the flat classifier as the preferred frozen four-class model within this internal evaluation and identifies robust routing and task-sharing design as the principal unresolved research directions. External and prospective validation would be required before drawing conclusions about clinical deployment or broader population generalization.

## Gate 05C closure verdict

**PASS.** The Results, Discussion, limitations, contribution statements, and conclusion are publication-ready, numerically reconciled with frozen Phase 04 evidence, and bounded by the approved Phase 05 claim policy.
