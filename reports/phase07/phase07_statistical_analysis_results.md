# Phase 07 Stored-Prediction Statistical Analysis

## Provenance and scope

- Samples: 3,668; support: non_malignant 2,398, melanoma 678, bcc 498, scc 94.
- Amended protocol SHA-256: `efaace517733ae7c91d2284bb4a7ca55fa7f8052790f75ebb146256ba7d8a73f`.
- Phase 05 predictions SHA-256: `391557deb9a1aeb9b9f97edc9d3d38759e597d56b54bfdbab9ea7482451a221a`.
- Phase 06C archive SHA-256: `b76762b53a35a8d9b0aa96621d78ea0e4421aa6e8052d068ffc10648a4e63e91`.
- Phase 06C member SHA-256: `08b3462549210ed7f2330a687c37a6de4e013e00185fadc3167aa980995e497d`.
- Paired manifest SHA-256: `d53e8581a95661de0446961b81458bc17295efe9c6a513c0225e442a281bf941`.
- Environment: `{"numpy": "2.4.6", "pandas": "3.0.5", "python": "3.11.9", "scikit_learn": "1.9.0", "scipy": "1.17.1"}`.

The Phase 06C member was read only after safe archive validation: exactly one regular canonical member, with links, absolute paths, traversal and duplicates rejected.

## Point estimates and paired intervals

- Flat macro-F1: 0.619222, 95% CI [0.591983, 0.646581].
- Hierarchical macro-F1: 0.605367, 95% CI [0.581034, 0.629275].
- Flat minus hierarchical macro-F1: 0.013855, paired 95% CI [-0.014255, 0.041963].
- The analysis did not establish a statistically distinguishable macro-F1 difference.
- Flat accuracy: 0.742094; hierarchical accuracy: 0.740185; paired difference CI [-0.011996, 0.015812].
- Flat balanced accuracy: 0.650313; hierarchical balanced accuracy: 0.631199; paired difference CI [-0.010864, 0.049639].

Point estimates use the complete original sample. Intervals use all 10,000 paired, ground-truth-stratified replicates, seed 42, and explicit NumPy `method="linear"` on unrounded float64 values.

## Paired correctness and McNemar

- Both correct: 2368; flat only: 354; hierarchy only: 347; both wrong: 599.
- Exact two-sided McNemar p-value: 0.82074158826914845.
- Net paired correctness advantage: 0.0019083969465648854.
- Raw discordant-pair odds ratio: `{"numeric_value": 1.0201729106628241, "status": "finite"}`.

## Per-class, SCC, agreement and routing

Per-class precision, recall, F1, support and descriptive paired intervals are in `per_class_metric_summary.csv` and `bootstrap_confidence_intervals.csv`. No class-wise inferential p-values were calculated.

SCC results are descriptive and uncertainty is high because support is only 94. Complete SCC counts and metrics are in `scc_error_analysis.csv`.

Complete agreement, transition and ground-truth-stratified error counts are provided in the generated CSV outputs; no samples were cherry-picked.

Routing decomposition uses stored columns `final_target_index`, `stage_1_predicted_index`, `stage_2_executed`, `stage_2_predicted_index`, and `predicted_gate_correct`. Structural Stage 2 missingness is kept separate from anomalous missingness.

## Interpretation boundaries and reproducibility

Accuracy, balanced accuracy and McNemar are secondary; per-class effects are exploratory descriptive comparisons. Results concern one locked ISIC 2019 internal-test split and do not establish cross-dataset or population generalization.

Clinical superiority, clinical validation, improved diagnosis, mortality reduction, deployment readiness, equivalence, non-inferiority and causal claims are prohibited.

Exact command: `.\.venv\Scripts\python.exe scripts/run_phase07_statistical_analysis.py --output-directory reports/phase07/generated --control-directory reports/phase07/control --report-path reports/phase07/phase07_statistical_analysis_results.md`

No training, inference, evaluation rerun, checkpoint loading, model initialization, or GPU work occurred.
