# Phase 07 Paper-Ready Figure Captions

**Figure 1. Conditional hierarchical and flat comparison architectures.** The
hierarchical system applies Stage 1 to every image, returns a non-malignant
prediction on that route, and conditionally invokes Stage 2 for melanoma, BCC,
or SCC classification on the malignant route. The flat comparison uses one
direct four-class EfficientNet-B0 decision path.

**Figure 2. Row-normalized confusion-matrix comparison on the same locked
3,668-sample internal-test split.** Panels show the flat model and conditional
hierarchy using a common 0–100% scale. Each cell reports its within-true-class
percentage and raw count. Rows are true classes and columns are predicted
classes; SCC support is 94.

**Figure 3. Exploratory per-class F1 comparison.** Error bars are model-specific
95% ground-truth-class-stratified bootstrap intervals from 10,000 replicates
with seed 42. These comparisons are exploratory, with no class-wise
inferential p-values. SCC support is only 94, so its estimate is uncertain.

The optional paired-correctness figure was omitted because its four counts are
already compactly reported in the main comparison evidence and an additional
figure would duplicate information within the six-page limit.
