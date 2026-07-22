# Research Questions and Hypotheses

## Purpose

This document defines the questions that every experiment must answer.

The project should not become a collection of disconnected model-training runs. Each experiment must contribute evidence toward at least one research question listed here.

---

## RQ1 — Hierarchical versus Flat Classification

### Research Question

Does a conditional hierarchical framework provide better clinically relevant classification performance than a conventional flat multiclass classifier?

### Hypothesis

The hierarchical framework will improve:

* macro F1-score;
* balanced accuracy;
* malignant-class recall;
* minority-class recall;
* interpretability of failure patterns.

The hierarchy may not always produce the highest overall accuracy because overall accuracy can be dominated by majority classes.

### Required Comparison

Compare:

1. a flat classifier trained on the selected lesion classes;
2. the complete predicted-gate hierarchical pipeline;
3. an oracle-gate hierarchical pipeline.

The oracle-gate experiment uses the true previous-stage label to route samples. It measures the performance that later stages could achieve without routing errors.

---

## RQ2 — Error Propagation

### Research Question

How much performance is lost because classification errors propagate through the hierarchy?

### Hypothesis

The predicted-gate hierarchy will perform worse than the oracle-gate hierarchy because malignant lesions incorrectly classified as non-malignant in Stage 1 will never reach Stage 2.

### Required Analysis

Measure:

* Stage 1 false-negative rate;
* number of malignant samples blocked by Stage 1;
* conditional Stage 2 performance;
* end-to-end Stage 2 performance;
* oracle-gate versus predicted-gate performance difference;
* class-specific routing failures.

The project must report both stage-level and end-to-end metrics.

---

## RQ3 — Imbalance-Aware Learning

### Research Question

Do imbalance-aware learning strategies improve minority-class performance without causing unacceptable degradation in calibration or majority-class performance?

### Hypothesis

Class weighting, balanced sampling, or imbalance-aware loss functions will improve minority-class recall and macro F1-score.

However, aggressive balancing may:

* increase false positives;
* reduce calibration quality;
* reduce majority-class precision;
* produce unstable validation results.

### Required Comparison

At minimum, compare:

1. standard cross-entropy baseline;
2. class-weighted loss;
3. one additional imbalance-aware method selected from:

   * focal loss;
   * class-balanced focal loss;
   * weighted sampling;
   * logit adjustment.

Only one additional method is mandatory unless evidence justifies further experiments.

---

## RQ4 — External Generalisation

### Research Question

How well does the trained system generalise to independently collected dermoscopic datasets?

### Hypothesis

Performance on HIBA will be lower than performance on the internal ISIC 2019 test set because of domain shift.

The expected sources of domain shift include:

* acquisition-device differences;
* image resolution;
* illumination;
* preprocessing differences;
* population differences;
* lesion distribution;
* annotation practices;
* class prevalence.

### Evaluation Rule

HIBA is an evaluation-only dataset.

It must not be used for:

* model selection;
* hyperparameter tuning;
* threshold selection;
* architecture selection;
* early stopping;
* preprocessing decisions based on performance.

The primary external evaluation must be zero-shot.

---

## RQ5 — Stage 3 Feasibility

### Research Question

Can melanoma severity be modelled reliably using the labels available in the EMB dataset?

### Hypothesis

Stage 3 feasibility will depend on:

* exact label definitions;
* sample count per severity group;
* missing clinical metadata;
* class imbalance;
* patient-level split feasibility;
* image-label consistency.

### Decision Rule

Stage 3 will proceed only if the EMB audit confirms that the target labels can be defined without unsupported assumptions.

Preferred target:

* clinically defensible T-category grouping.

Fallback target:

* documented Breslow-thickness grouping.

If neither target is sufficiently reliable, Stage 3 will be reported as a feasibility limitation rather than forcing an invalid classification experiment.

---

## RQ6 — Model Efficiency

### Research Question

Can a lightweight model achieve competitive performance while reducing parameter count, memory usage, inference latency, and computational cost?

### Hypothesis

A lightweight model can retain most of the performance of a stronger reference model while requiring fewer computational resources.

### Required Measurements

Report:

* number of trainable parameters;
* model file size;
* floating-point operations when available;
* mean inference latency;
* peak GPU memory;
* training duration;
* primary predictive metrics.

Efficiency comparisons must use equivalent preprocessing and evaluation conditions.

---

## RQ7 — Calibration and Confidence

### Research Question

Are the probability estimates produced by the model reliable enough to distinguish confident and uncertain predictions?

### Hypothesis

The baseline model will be overconfident, particularly on minority classes and external data.

Calibration methods may improve probability reliability without changing classification ranking substantially.

### Required Measurements

Report:

* expected calibration error;
* Brier score;
* reliability diagrams;
* confidence distribution for correct predictions;
* confidence distribution for incorrect predictions;
* internal versus external calibration.

Thresholds must be selected using validation data only.

---

## RQ8 — Explainability Consistency

### Research Question

Do explanation maps focus consistently on lesion-relevant regions, and how does explanation behaviour change under domain shift?

### Hypothesis

Correct high-confidence predictions will generally produce more lesion-focused explanations than incorrect or externally shifted predictions.

Explanation stability may decrease on external datasets.

### Required Analysis

The explainability study should include:

* representative correct predictions;
* representative incorrect predictions;
* high-confidence errors;
* minority-class examples;
* internal and external examples;
* explanation consistency under selected transformations.

Explainability results must not be presented as proof of clinical reasoning.

---

# Primary Evaluation Metrics

The primary metrics are:

1. macro F1-score;
2. balanced accuracy;
3. per-class recall;
4. per-class precision;
5. confusion matrix.

Secondary metrics include:

* weighted F1-score;
* ROC-AUC;
* PR-AUC;
* specificity;
* calibration metrics;
* efficiency metrics.

Overall accuracy must not be treated as the sole primary metric.

---

# Statistical Reporting

Where feasible, final comparisons should report uncertainty using:

* bootstrap confidence intervals;
* repeated-seed experiments;
* patient-level resampling when patient identifiers are available.

A single training run should not be used to make strong superiority claims.

The minimum preferred final configuration is three random seeds for the main baseline and proposed method, subject to compute availability.

---

# Research Success Criteria

The project will be considered successful if it produces a rigorous answer to the research questions, even if the hierarchical method does not outperform every baseline.

A valid negative or mixed result is acceptable when:

* evaluation is leakage-free;
* comparisons are fair;
* error propagation is quantified;
* external evaluation is independent;
* limitations are reported honestly;
* experiments are reproducible.

The project should prioritise scientific validity over producing an artificially positive result.
