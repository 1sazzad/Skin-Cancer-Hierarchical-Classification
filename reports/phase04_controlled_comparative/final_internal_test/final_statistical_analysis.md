# Phase 04 Final Internal-Test Statistical Analysis

## 1. Evaluation integrity

**PASS.** The stored evidence contains 3,668 unique paired samples and matches the frozen schema, class support, manifests, Tesla T4 environment, and pre-test commit `3bf8c137f585ba720cf96d882003fadc678ea059`. Independently recomputed metrics and confusion matrices match the JSON within 1e-12. The prediction CSV and summary were not modified.

Checkpoint provenance is indirect: hashes, epochs, kinds, and class orders are frozen in the evaluation config and were enforced by the runner, but are not duplicated in the final summary. This is an evidence-design limitation, not an observed execution inconsistency.

## 2. Final test results

The flat model achieved accuracy 0.742094, balanced accuracy 0.650313, macro-F1 0.619222, macro precision 0.611328, macro recall 0.650313, and weighted F1 0.752557.

The shared hierarchy achieved Task1 macro-F1 0.740624, Task2 malignant-subset macro-F1 0.702634, Task3 macro-F1 0.298710, predicted-gate four-class macro-F1 0.568591, and oracle-gate four-class macro-F1 0.776976.

### Overall model/task metrics

| Model/task | N | Accuracy | Balanced accuracy | Macro-F1 | Macro precision | Macro recall | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| flat_four_class | 3668 | 0.742094 | 0.650313 | 0.619222 | 0.611328 | 0.650313 | 0.752557 |
| shared_task1 | 3668 | 0.746183 | 0.774397 | 0.740624 | 0.748501 | 0.774397 | 0.752302 |
| shared_task2_malignant_subset | 1270 | 0.808661 | 0.705562 | 0.702634 | 0.700049 | 0.705562 | 0.809353 |
| shared_task3 | 127 | 0.472441 | 0.308052 | 0.298710 | 0.311282 | 0.308052 | 0.471679 |
| shared_predicted_gate | 3668 | 0.687568 | 0.639536 | 0.568591 | 0.539799 | 0.639536 | 0.706832 |
| shared_oracle_gate | 3668 | 0.933751 | 0.779171 | 0.776976 | 0.775037 | 0.779171 | 0.933991 |
| standalone_task1 | 3668 | 0.786260 | 0.789306 | 0.774009 | 0.768663 | 0.789306 | 0.790190 |
| standalone_task2 | 1270 | 0.833858 | 0.722716 | 0.724875 | 0.727763 | 0.722716 | 0.832915 |
| standalone_task3 | 127 | 0.543307 | 0.386039 | 0.275611 | 0.241288 | 0.386039 | 0.493247 |

### Flat four-class confusion matrix

Rows are true classes and columns are predicted classes in the order non_malignant, melanoma, bcc, scc.

| True class | non_malignant | melanoma | bcc | scc |
|---|---:|---:|---:|---:|
| non_malignant | 1792 | 408 | 170 | 28 |
| melanoma | 135 | 511 | 30 | 2 |
| bcc | 27 | 68 | 389 | 14 |
| scc | 9 | 12 | 43 | 30 |

### Flat four-class per-class metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| non_malignant | 0.912888 | 0.747289 | 0.821830 | 2398 |
| melanoma | 0.511512 | 0.753687 | 0.609422 | 678 |
| bcc | 0.615506 | 0.781124 | 0.688496 | 498 |
| scc | 0.405405 | 0.319149 | 0.357143 | 94 |

All remaining confusion matrices and per-class metrics are retained in `final_paired_statistics.json`; compact overall metrics are in `final_results_table.csv`.

## 3. Flat vs hierarchy comparison

The flat model achieved higher macro-F1 than the deployed shared hierarchy: delta hierarchy − flat = -0.050632. Accuracy delta was -0.054526.

## 4. Paired statistical evidence

Using 10,000 paired, ground-truth-class-stratified bootstrap replicates (seed 42), the macro-F1 delta 95% percentile CI was [-0.075993, -0.024827] and the accuracy delta CI was [-0.068702, -0.040349]. Both exclude zero and favor flat. Paired outcomes were: both correct 2248, hierarchy only 274, flat only 474, both wrong 672. Exact two-sided McNemar p=2.49118e-13. The direction and intervals, rather than the p-value alone, support the conclusion.

## 5. Class-wise analysis

| Class | Support | Hierarchy F1 | Flat F1 | Delta H−F | 95% CI |
|---|---:|---:|---:|---:|---:|
| non_malignant | 2398 | 0.778597 | 0.821830 | -0.043233 | [-0.056035, -0.030510] |
| melanoma | 678 | 0.545669 | 0.609422 | -0.063753 | [-0.088293, -0.039765] |
| bcc | 498 | 0.659189 | 0.688496 | -0.029307 | [-0.056238, -0.002722] |
| scc | 94 | 0.290909 | 0.357143 | -0.066234 | [-0.150000, 0.019586] |

SCC has only 94 cases. Its interval is reported, but minority-class conclusions remain less stable and should not be overgeneralized.

## 6. Routing-error decomposition

Among 1270 malignant and 2398 non-malignant cases, Task1 blocked 170 malignant cases (13.39%) and incorrectly sent 761 non-malignant cases to Task2 (31.73%). It correctly routed 1100 malignant cases. Stage2 ran 2031 times (55.37%); 215 correctly routed malignant cases had subtype errors (19.55%). Oracle minus predicted-gate macro-F1 was 0.208385, indicating substantial end-to-end performance loss associated with routing.

## 7. Shared vs standalone task comparison

- Task1: shared minus standalone macro-F1 -0.033385; accuracy -0.040076.
- Task2: shared minus standalone macro-F1 -0.022241; accuracy -0.025197.
- Task3: shared minus standalone macro-F1 +0.023099; accuracy -0.070866.

Task1 and Task2 favor the standalone models, which is consistent with possible negative transfer but does not establish causation or paired statistical significance. Shared Task3 has higher macro-F1 but lower accuracy; with T2/T3/T4 supports of 7/2/1, its class-level pattern is extremely unstable.

## 8. Validation-to-test consistency

The direction replicated: hierarchy-minus-flat macro-F1 changed from −0.070285 on validation to -0.050632 on test, and accuracy from −0.063795 to -0.054526. Routing loss increased from 0.192480 to 0.208385. All four observed class F1 deltas favored flat on both splits; the test SCC interval nevertheless included zero. Standalone Tasks 1–2 remained ahead of their shared heads; Task3 retained the mixed pattern of sparse, unstable results. Overall robustness is supported for the main direction, with modest effect-size variation.

## 9. Efficiency

Measured batch-1 Tesla T4 evidence:

| Model | Parameters | Checkpoint bytes | Latency ms/image | Throughput img/s | Peak CUDA bytes | FLOPs/MACs |
|---|---:|---:|---:|---:|---:|---|
| flat | 4012672 | 48640921 | 6.8456 | 146.08 | 35857408 | unavailable |
| shared | 4020358 | 48737992 | 6.8650 | 145.67 | 35889152 | unavailable |
| task1 | 4010110 | 48609369 | 6.8330 | 146.35 | 35847168 | unavailable |
| task2 | 4011391 | 48625369 | 6.8014 | 147.03 | 35852288 | unavailable |
| task3 | 4013953 | 48656473 | 6.8415 | 146.17 | 35862528 | unavailable |

These measurements compare one forward pass per model. Architecturally, the shared system stores one encoder with three heads (4,020,358 parameters; 48,737,992 checkpoint bytes), whereas the three independently stored task models total 12,035,454 parameters and 145,891,211 checkpoint bytes (derived sums, about 3.0× the shared storage). Therefore a single standalone forward-pass measurement is not the total storage or conditional execution cost of the independent hierarchy; full conditional hierarchy latency was not directly benchmarked.

## 10. Scientific interpretation

The flat four-class model performed better than the deployed predicted-gate shared hierarchy on the frozen internal test. The paired bootstrap intervals excluded zero. The oracle-routing diagnostic indicates substantial performance loss attributable to routing decisions, while conditional Task2 performance itself was comparatively strong.

## 11. Limitations

This is one frozen internal test from the project’s data construction, not external or clinical validation. Bootstrap intervals quantify sampling uncertainty under the chosen paired stratified resampling scheme. McNemar addresses accuracy discordance, not macro-F1. Small SCC support and extremely rare Task3 categories limit stable class-level inference. Efficiency excludes FLOPs/MACs because the backend did not support them and does not directly benchmark a full independently routed deployment.

## 12. Final Phase04 verdict

**PASS (analysis and evidence integrity).** The primary comparative conclusion replicated on internal test: the flat classifier achieved higher macro-F1 and accuracy than the deployed shared hierarchy. The evidence is consistent with routing as a major source of hierarchical performance loss and with possible negative transfer for shared Tasks 1–2, while Task3 is too sparse for strong class-level conclusions. No claim of clinical superiority, causation, or external generalization is supported.
