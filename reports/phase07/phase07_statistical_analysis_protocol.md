# Phase 07 Frozen Statistical Analysis Protocol

## Status and locked inputs

This protocol is frozen at Phase 07 Gate 2 before statistical execution.
No bootstrap replicate, confidence interval, McNemar test, p-value, or
statistical comparison result has been calculated.

Future analysis may use only the 3,668 rows in
`reports/phase07/generated/paired_prediction_manifest.csv` (SHA-256
`d53e8581a95661de0446961b81458bc17295efe9c6a513c0225e442a281bf941`).
The locked Phase 05 and Phase 06C prediction SHA-256 values are respectively
`391557deb9a1aeb9b9f97edc9d3d38759e597d56b54bfdbab9ea7482451a221a`
and
`08b3462549210ed7f2330a687c37a6de4e013e00185fadc3167aa980995e497d`.
Internal-test inference cannot be rerun and checkpoints cannot be loaded.

The independent unit is one paired internal-test image identified by
`image_id`. No lesion, patient, or duplicate-group structure may be inferred.
All comparisons remain paired at image level. The fixed class mapping is
0 `non_malignant`, 1 `melanoma`, 2 `bcc`, and 3 `scc`, with supports 2,398,
678, 498, and 94.

## Estimands and original-sample estimates

The single primary estimand is flat macro-F1 minus hierarchical macro-F1;
positive values favour the flat model. Its family size is one and it receives
no multiplicity correction.

All point estimates must come from the complete original paired sample, never
the mean or median of bootstrap replicates. The future execution must report
accuracy, balanced accuracy, macro-F1, weighted F1, and per-class precision,
recall, and F1 for each model. It must report flat-minus-hierarchical
differences for accuracy, balanced accuracy, macro-F1, and each class's F1.

## Paired stratified bootstrap

The frozen design uses 10,000 paired replicates with seed 42. Within every
replicate, sampling is with replacement separately inside each ground-truth
class, preserving supports 2,398, 678, 498, and 94. The same sampled
`image_id` indices are applied to both models.

Intervals are two-sided 95% percentile intervals using quantiles 0.025 and
0.975. Metrics must use explicit labels `[0, 1, 2, 3]` and `zero_division=0`.
No replicate may be dropped. A non-finite replicate metric is a fail-closed
error. Stratification conditions uncertainty estimation on observed class
support and prevents replicates from omitting the rare SCC class.

Authorized model-specific intervals are accuracy, balanced accuracy,
macro-F1, and four per-class F1 values for each model. Authorized paired
difference intervals are flat minus hierarchy for accuracy, balanced
accuracy, macro-F1, and four per-class F1 values. Adding an estimand requires a
protocol amendment made before its result is viewed.

If the primary paired interval excludes zero, the difference may be described
as statistically distinguishable under this prespecified paired bootstrap. If
it includes zero, the analysis did not establish a statistically
distinguishable macro-F1 difference. Neither outcome establishes equivalence,
non-inferiority, clinical superiority, or clinical significance.

## Exact McNemar comparison and effect measures

One overall 2-by-2 paired correctness table is frozen: both correct, flat only
correct, hierarchy only correct, and both wrong. The primary McNemar method is
the exact two-sided binomial test on discordant pairs, with null hypothesis of
equally likely discordant directions and alpha 0.05. Report both discordant
counts, their total, the exact p-value, and paired accuracy difference. An
unpaired chi-square test is prohibited. An asymptotic continuity-corrected
result requires later authorization and must be labelled sensitivity-only.

The primary effect is the absolute flat-minus-hierarchy macro-F1 difference.
Secondary effects are absolute differences in accuracy, balanced accuracy,
and per-class F1, plus net paired correctness advantage:
`(flat-only correct - hierarchy-only correct) / 3668`.

The raw discordant-pair odds ratio is flat-only correct divided by
hierarchy-only correct. Raw counts must accompany it. A zero denominator
produces infinity and a zero numerator produces zero. A Haldane–Anscombe
estimate is permitted only as a labelled sensitivity estimate. The odds ratio
must not be called a risk ratio.

## Multiplicity and descriptive analyses

Accuracy difference, balanced-accuracy difference, and exact McNemar are
secondary. Four per-class F1 differences are exploratory secondary
comparisons; raw intervals are descriptive and cannot support independent
unadjusted class-wise significance claims. If class-wise inferential p-values
are later added, Holm–Bonferroni correction across the four classes is
mandatory. SCC findings must be labelled uncertain because support is 94.

Authorized descriptive outputs include paired correctness categories, exact
prediction agreement/disagreement, ground-truth-stratified correctness,
model confusion-pair counts, flat-to-hierarchy prediction transitions, and
error transitions by ground truth. SCC-specific descriptions cover each
system's true-SCC predictions, recall, F1, three confusion destinations, and
samples rescued only by each model. These are not independently powered
confirmatory tests.

## Hierarchical routing boundary

Routing decomposition is allowed only when required stored Phase 05 fields
exist and their semantics are proven by Phase 05 documentation or code.
Permitted categories are malignant samples blocked as non-malignant,
non-malignant samples routed to Stage 2, correctly routed malignant samples
with wrong subtype, correct routing with correct subtype, and structurally
missing Stage 2 fields when Stage 2 was not invoked.

Missing routing states must not be inferred. Predictions must not be
reconstructed, conditional missing values must not be filled, and structural
missingness must be distinguished from data-quality missingness. Unsupported
decompositions must be skipped with a documented reason.

## Claims and reproducibility

Reports may state the higher observed macro-F1, whether the paired interval
included zero, whether exact McNemar detected a paired correctness difference,
that SCC is uncertain at support 94, and that results concern one locked ISIC
2019 internal-test split.

Claims of clinical superiority, clinical validation, population-wide
generalization, statistical equivalence, non-inferiority, cross-dataset
robustness, improved diagnosis, reduced mortality, readiness for deployment,
or other unsupported causal/clinical effects are prohibited.

Execution records must capture Python and NumPy/pandas/SciPy/scikit-learn
versions, paths and hashes, paired-manifest hash, seed, replicate count,
confidence level, CI method, class mapping, zero-division policy, test
implementation, exact command, Git commit, timestamps, and machine- and
human-readable outputs. Deterministic result payloads must be byte-stable;
timestamps are stored separately.

Future execution fails closed on any changed hash, wrong row/support count,
missing or duplicate identifier, identifier/ground-truth mismatch,
unsupported label, non-finite metric, skipped replicate, changed seed or
replicate count, locked-artifact write attempt, checkpoint load, or inference
attempt.
