# ITT 2026 — Phase ITT-07 Manuscript Evidence Draft

## Working title

**Cross-Backbone Consensus and Selective Referral for Robust Dermoscopic Skin Lesion Classification**

## Study question

Can predictions from heterogeneous CNN backbones be combined to improve four-class dermoscopic lesion classification, and can cross-backbone disagreement identify cases that are more suitable for referral?

## Final experimental scope

Seven frozen Flat CNN classifiers:

- DenseNet121
- DenseNet169
- ResNet50
- MobileNetV3-Large
- EfficientNet-B0
- EfficientNet-B2
- EfficientNet-B3

Four classes:

- non-malignant
- melanoma
- basal cell carcinoma
- squamous cell carcinoma

Cohorts:

- ISIC 2019 frozen internal test: 3,668 images
- HIBA external zero-shot cohort: 1,232 images from 568 patients

Primary ensemble:

- hard majority vote across seven backbones

Secondary ensemble:

- validation-macro-F1-weighted hard vote

Primary individual comparator:

- DenseNet169, prospectively selected from validation performance

Disagreement:

- agreement = maximum class vote count / 7
- disagreement = 1 - agreement

Selective referral:

- 7/7
- >=6/7
- >=5/7
- >=4/7 agreement

No threshold was optimized on ISIC test or HIBA.

---

## Draft abstract

### Background

Predictions from independently trained CNN backbones can differ on the same dermoscopic lesion, creating an opportunity to use cross-model consensus both for prediction and for uncertainty-aware referral.

### Methods

We combined seven frozen four-class CNN classifiers—DenseNet121, DenseNet169, ResNet50, MobileNetV3-Large, EfficientNet-B0, EfficientNet-B2, and EfficientNet-B3—using a prespecified hard majority vote and a secondary validation-weighted vote. Models predicted non-malignant, melanoma, basal cell carcinoma, or squamous cell carcinoma. Evaluation used a frozen ISIC 2019 internal test set (N=3,668) and the independent HIBA cohort (1,232 images; 568 patients). Cross-backbone agreement was defined by the maximum vote share, and selective referral was evaluated at 7/7, >=6/7, >=5/7, and >=4/7 agreement without test-driven threshold selection. Macro-F1 was the primary endpoint. Paired image bootstrap was used internally, while HIBA inference used patient-cluster bootstrap.

### Results

The majority ensemble achieved macro-F1 of **0.6848** on ISIC versus **0.6493** for the prospectively selected DenseNet169 comparator, a difference of **+0.0355** (95% CI **0.0130 to 0.0582**; exact McNemar p = **4.0×10^-9**). On HIBA, macro-F1 was **0.6622** versus **0.6333**, a difference of **+0.0289** with patient-cluster 95% CI **0.0003 to 0.0574**. At unanimous 7/7 agreement, coverage was **51.9%** on ISIC and **61.8%** on HIBA, with retained macro-F1 of **0.8257** and **0.7556**, respectively. Less restrictive agreement thresholds increased coverage while reducing retained performance.

### Conclusion

Across two evaluation cohorts, a simple cross-backbone majority vote improved macro-F1 relative to a validation-selected individual comparator, while vote agreement provided an interpretable signal for selective referral. The results support consensus and disagreement as lightweight post-hoc mechanisms for combining heterogeneous dermoscopic classifiers, although the external confidence interval was close to zero and no single referral threshold was selected.

---

## 1. Introduction — evidence points

Use the introduction to establish four ideas:

1. Dermoscopic lesion classification is commonly studied with individual CNN architectures, but different backbones can make different errors.
2. Ensemble methods may improve predictive stability without retraining constituent models.
3. In safety-sensitive settings, a model can abstain or refer uncertain cases instead of forcing a prediction.
4. Cross-backbone disagreement provides a simple, architecture-agnostic uncertainty signal that can be computed directly from hard predictions.

Do not frame disagreement as calibrated uncertainty unless calibration is actually evaluated.

### Suggested final paragraph of Introduction

This study examines whether heterogeneous CNN predictions can be combined through a simple cross-backbone consensus rule and whether disagreement can support selective referral. Seven frozen four-class CNN classifiers are evaluated using prespecified majority and validation-weighted voting on the locked ISIC 2019 internal test set and the independent HIBA cohort. We further quantify agreement using the maximum vote share and evaluate a fixed series of referral rules without selecting an operating point from either evaluation cohort. The primary question is whether consensus improves macro-F1 relative to a validation-selected individual comparator, while secondary analyses assess whether stronger agreement is associated with improved retained performance.

---

## 2. Methods

### 2.1 Constituent classifiers

State that all seven classifiers were pre-existing frozen Flat four-class models. No new CNN training was performed for this study.

### 2.2 Consensus rules

Primary majority vote:

For image x, let n_c(x) be the number of constituent models predicting class c.

The majority prediction is:

argmax_c n_c(x)

When vote counts tie, tied classes are resolved using the frozen validation-macro-F1 weights; any remaining exact weighted tie is resolved using the frozen validation ranking.

Secondary weighted vote:

Score(c|x) = sum_m w_m I(y_hat_m = c)

where w_m is proportional to the frozen validation macro-F1 of model m.

### 2.3 Agreement and disagreement

A(x) = max_c n_c(x) / 7

D(x) = 1 - A(x)

Referral rules classify automatically only when max_c n_c(x) is at least 7, 6, 5, or 4, respectively.

### 2.4 Comparator selection

DenseNet169 is the primary individual comparator because it had the highest frozen validation macro-F1 among the seven constituent models.

This comparator was fixed before evaluating the new consensus results.

### 2.5 Evaluation cohorts

ISIC 2019 internal test:

- N = 3,668
- fixed four-class endpoint

HIBA:

- 1,232 images
- 568 patients
- zero-shot external evaluation

### 2.6 Statistical analysis

Primary endpoint: macro-F1.

ISIC:

- paired image bootstrap
- 10,000 replicates
- seed 42
- percentile 95% CI
- exact two-sided McNemar on paired correctness

HIBA:

- patient-cluster bootstrap
- 10,000 replicates
- seed 42
- percentile 95% CI

Selective-referral results are descriptive because no operating threshold was selected.

---

## 3. Results

### 3.1 Full-coverage consensus

On the ISIC internal test set, the majority ensemble achieved macro-F1 **0.6848**, compared with **0.6493** for DenseNet169, for a difference of **+0.0355**. The paired bootstrap 95% CI was **[0.0130, 0.0582]**. Paired correctness also favored the ensemble: 248 cases were correct only for the ensemble versus 133 correct only for DenseNet169, with exact two-sided McNemar p = **4.0×10^-9**.

On HIBA, the majority ensemble achieved macro-F1 **0.6622**, compared with **0.6333** for DenseNet169, a difference of **+0.0289**. The patient-cluster bootstrap 95% CI was **[0.0003, 0.0574]**. The interval excluded zero but its lower bound was close to zero.

The validation-weighted ensemble produced the same cohort-level macro-F1 point estimates as the majority ensemble on both datasets.

### 3.2 Agreement distribution

ISIC maximum-vote counts:

- 7/7: 1,902 images
- 6/7: 684
- 5/7: 526
- 4/7: 466
- 3/7: 84
- 2/7: 6

HIBA:

- 7/7: 761 images
- 6/7: 148
- 5/7: 135
- 4/7: 140
- 3/7: 45
- 2/7: 3

### 3.3 Selective referral

ISIC:

- 7/7: coverage 0.5185, accuracy 0.9317, macro-F1 0.8257
- >=6/7: coverage 0.7050, accuracy 0.8979, macro-F1 0.7784
- >=5/7: coverage 0.8484, accuracy 0.8538, macro-F1 0.7289
- >=4/7: coverage 0.9755, accuracy 0.8116, macro-F1 0.6893

HIBA:

- 7/7: coverage 0.6177, accuracy 0.9251, macro-F1 0.7556
- >=6/7: coverage 0.7378, accuracy 0.8867, macro-F1 0.7236
- >=5/7: coverage 0.8474, accuracy 0.8487, macro-F1 0.6974
- >=4/7: coverage 0.9610, accuracy 0.8049, macro-F1 0.6685

These results show a descriptive coverage-performance tradeoff on both cohorts.

---

## 4. Discussion — key interpretation

The main finding is that a simple post-hoc consensus rule improved macro-F1 relative to a prospectively selected individual comparator on both the internal and external cohorts. This improvement required no retraining of the constituent CNNs and therefore isolates the value of combining already available predictions.

A second finding is that agreement among heterogeneous backbones was informative. Cases with unanimous or near-unanimous predictions had substantially higher retained performance than the full cohort. This supports the use of cross-backbone disagreement as an interpretable referral signal.

The pattern was visible on both datasets, but the external inferential result should be treated cautiously because the lower bound of the HIBA patient-cluster confidence interval was close to zero.

The weighted voting scheme did not improve cohort-level macro-F1 over ordinary majority voting in these experiments. This suggests that the simple majority rule captured most of the available benefit from the seven constituent models under the tested weighting scheme.

---

## 5. Limitations

Include all of the following:

1. Constituent CNNs were single-seed frozen models rather than newly trained multi-seed ensemble members.
2. Per-image validation predictions for all seven models were unavailable, and only two exact historical checkpoints were recoverable locally.
3. Therefore a single referral operating point could not be selected from validation data.
4. To avoid test-driven tuning, all four referral thresholds were prospectively fixed and reported.
5. The primary comparator was DenseNet169 because it had the highest frozen validation macro-F1; the study does not establish superiority over every possible individual or ensemble configuration.
6. HIBA contained repeated images from patients, addressed through patient-cluster bootstrap, but it remains a single external cohort.
7. The HIBA CI for the primary difference was positive but close to zero at its lower bound.
8. Hard-vote disagreement is an interpretable consensus measure but is not equivalent to calibrated predictive uncertainty.
9. No clinical workflow, prospective trial, or deployment study was performed.
10. SCC and other minority-class conclusions require care because class support is lower than for non-malignant lesions.

---

## 6. Conclusion

A seven-backbone hard majority ensemble improved four-class dermoscopic lesion macro-F1 relative to a validation-selected individual comparator on both the frozen ISIC internal test set and the independent HIBA cohort. Cross-backbone agreement also provided a simple selective-referral signal: stricter agreement rules reduced coverage while increasing retained predictive performance. These findings show that heterogeneous hard predictions can provide useful consensus and disagreement information without retraining or calibration. Further work should evaluate multi-seed ensembles, calibrated uncertainty, additional external cohorts, and prospectively selected referral operating points.

---

## Recommended figures

1. **Full-Coverage Macro-F1 Across Individual CNNs and Consensus**
2. **Selective Referral Risk-Coverage Curve**
3. **Cross-Backbone Agreement Distribution**
4. **Macro-F1 Across the Frozen Referral Series**

Use the ITT-06 generated figures as the evidence source.

## Recommended tables

1. Constituent model validation/internal/HIBA macro-F1
2. Full-coverage ensemble comparison
3. Primary paired inference
4. Selective-referral coverage/performance table

## Manuscript claim boundary

Use phrases such as:

- "higher macro-F1"
- "the paired/bootstrap interval excluded zero"
- "agreement was associated with higher retained performance"
- "external zero-shot evaluation"

Avoid:

- "clinically superior"
- "deployment-ready"
- "guaranteed robust"
- "optimal threshold"
- "calibrated uncertainty"
- "universal generalization"
