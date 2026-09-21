# ITT 2026 — Phase ITT-01 Protocol Freeze

**Project:** Skin Cancer Hierarchical Classification  
**ITT study:** Cross-backbone consensus, disagreement, and selective referral  
**Frozen from main commit:** `edbb506b941ae97a45a21b7274e786eba607d885`  
**Status:** FROZEN BEFORE ITT ENSEMBLE TEST/HIBA COMPUTATION

## 1. Scientific boundary

This ITT extension uses the seven existing frozen **Flat four-class CNN classifiers** only. It does not retrain or reselect the ICCIT models and does not use Shared-Hard predictions as ITT inputs.

Classes are fixed as:

- 0 = non_malignant
- 1 = melanoma
- 2 = bcc
- 3 = scc

Constituent models are fixed to:

1. DenseNet121
2. DenseNet169
3. ResNet50
4. MobileNetV3-Large
5. EfficientNet-B0
6. EfficientNet-B2
7. EfficientNet-B3

No model may be removed, replaced, or added after viewing ITT internal-test or HIBA ensemble results.

## 2. Evidence sources

Internal ISIC per-image Flat predictions must be read from:

`results/paper/internal_isic/gate06d/<backbone>/paired_internal_test_predictions.csv`

using only:

- `sample_id`
- `true_label`
- `flat_prediction`

External HIBA per-image Flat predictions must be read from:

`results/paper/external_hiba/gate06f/<backbone>/paired_hiba_predictions.csv`

using only:

- `isic_id`
- `patient_id`
- `target`
- `flat_prediction`

The ICCIT hierarchical fields must not participate in ITT consensus generation.

## 3. Validation evidence and fixed ranking

The existing Flat validation macro-F1 values from `experiments/paper_registry.csv` are frozen:

| Model | Validation macro-F1 | Normalized weight |
|---|---:|---:|
| DenseNet169 | 0.6659647727 | 0.145128484360 |
| EfficientNet-B3 | 0.6583008001 | 0.143458334867 |
| MobileNetV3-Large | 0.6582407229 | 0.143445242714 |
| EfficientNet-B2 | 0.6544148913 | 0.142611509213 |
| EfficientNet-B0 | 0.6535716654 | 0.142427751600 |
| ResNet50 | 0.6533194436 | 0.142372786880 |
| DenseNet121 | 0.6449820791 | 0.140555890366 |

The fixed validation ranking above is the only model ranking allowed for deterministic tie resolution.

Per-image validation predictions are not present in the repository at protocol freeze. Therefore no referral threshold will be selected by validation unless those original validation predictions are later recovered/regenerated without using internal-test or HIBA outcomes.

## 4. Primary ensemble — seven-model majority vote

For each image, collect the seven Flat hard-label predictions.

For class `c`:

`n_c(x) = number of constituent models predicting c`

The primary ensemble prediction is the class with maximum vote count.

### Majority-vote tie rule

If two or more classes share the maximum vote count:

1. Among tied classes, compute the sum of the frozen validation weights of models voting for each tied class.
2. Choose the tied class with the largest validation-weight sum.
3. If an exact weighted tie remains, inspect the fixed validation ranking from highest to lowest and choose the first model whose prediction is one of the tied classes.

No class-priority rule based on test/HIBA prevalence or performance is allowed.

## 5. Secondary ensemble — validation-weighted vote

For class `c`:

`Score(c|x) = sum_m w_m I(yhat_m = c)`

where `w_m` is the normalized validation macro-F1 weight frozen above.

Prediction:

`argmax_c Score(c|x)`

If scores tie exactly, use the fixed validation ranking: traverse models from highest to lowest validation macro-F1 and choose the first prediction among the tied classes.

The weighted ensemble is secondary. It must not replace the majority ensemble as the primary method because of later test/HIBA performance.

## 6. Prospectively selected single-model comparator

The primary individual-model comparator is **DenseNet169**, selected solely because it has the highest frozen validation macro-F1 (0.6659647727).

All seven individual-model results may be reported descriptively, but test/HIBA performance must not be used to redefine the primary comparator.

## 7. Agreement and disagreement

For seven models:

`A(x) = max_c n_c(x) / 7`

`D(x) = 1 - A(x)`

Store:

- maximum vote count
- agreement fraction
- disagreement fraction
- complete 4-class vote counts
- whether majority tie-breaking was required

Disagreement is defined from unweighted hard votes even when reporting the weighted ensemble.

## 8. Selective referral

The primary selective-prediction system is **majority vote + unweighted agreement**.

Predeclared automatic-classification rules:

- accept only 7/7 agreement
- accept >=6/7 agreement
- accept >=5/7 agreement
- accept >=4/7 agreement

All remaining cases are referred/abstained.

No single threshold will be called "optimal" from internal-test or HIBA results.

The primary selective analysis is the full predeclared coverage-performance/risk-coverage series.

If original per-image validation predictions are later recovered, a validation-selected operating point may be added as a secondary prespecified extension, documented before test/HIBA re-analysis.

## 9. Coverage metrics

Overall coverage:

`accepted cases / all cases`

Also report class-specific coverage:

`accepted true-class cases / all cases of that true class`

for non_malignant, melanoma, bcc, and scc.

Report malignant coverage over true classes melanoma+bcc+scc.

These class-specific coverage metrics are mandatory so apparent retained performance cannot hide referral of difficult minority classes.

## 10. Predictive metrics

At full coverage for each individual model, majority ensemble, and weighted ensemble report:

- accuracy
- macro-F1
- weighted-F1
- balanced accuracy
- per-class precision
- per-class recall
- per-class F1
- confusion matrix
- malignant sensitivity
- SCC sensitivity/F1

At every selective threshold report:

- coverage
- referred count
- retained count
- retained accuracy
- retained macro-F1
- retained weighted-F1
- retained balanced accuracy
- retained error rate
- per-class retained precision/recall/F1
- class-specific coverage
- malignant sensitivity among retained malignant cases
- malignant coverage
- SCC sensitivity/F1 where estimable
- SCC coverage

For metric calculation, keep the fixed class label set `[0,1,2,3]`. Undefined precision/F1 divisions use zero. Class support and accepted support must always be printed beside class-wise metrics.

## 11. Primary endpoint and comparisons

Primary endpoint: **macro-F1**.

Primary inferential comparison:

- majority-vote ensemble vs DenseNet169

Secondary comparisons:

- validation-weighted ensemble vs DenseNet169
- validation-weighted ensemble vs majority vote

All seven individual backbones may be shown descriptively. No alternative "best individual" may be substituted after inspecting test/HIBA values.

## 12. Internal ISIC statistical protocol

Internal population is the frozen 3,668-image test set.

For macro-F1 differences:

- paired bootstrap over images
- 10,000 replicates
- seed = 42
- percentile 95% confidence interval
- resample paired image indices so both systems use the same bootstrap sample

For paired correctness:

- exact two-sided McNemar test
- report both discordant counts
- report paired accuracy difference

Bootstrap and McNemar answer different questions and must not be treated as interchangeable.

## 13. External HIBA statistical protocol

HIBA remains zero-shot and unchanged.

Because multiple images may belong to the same patient, inferential confidence intervals use **patient-cluster bootstrap**:

- sample patients with replacement
- include all images belonging to each sampled patient
- 10,000 replicates
- seed = 42
- percentile 95% confidence interval

Do not use ordinary image-level McNemar p-values as primary inferential evidence on HIBA because image observations are clustered within patients.

## 14. Domain-shift/disagreement analysis

Compare ISIC and HIBA descriptively using:

- distribution of maximum vote count
- distribution of agreement/disagreement
- fraction at each referral threshold
- class-specific coverage
- error rate by agreement stratum

No HIBA-derived threshold tuning, model weighting, model removal, or method selection is allowed.

## 15. Data integrity gates before computation

Implementation must fail if any of the following occurs:

- a required backbone prediction file is missing
- duplicate sample/image IDs are present
- the seven model ID sets differ within a cohort
- ground-truth labels disagree between model files
- unexpected class labels occur
- internal ISIC row count is not 3,668
- HIBA row count is not 1,232
- HIBA patient ID is missing
- model list differs from the frozen seven-model set

Input file hashes should be recorded in the ITT result manifest.

## 16. Leakage and selection prohibitions

Do not:

- tune ensemble weights on ISIC test
- tune ensemble weights on HIBA
- choose a referral threshold from ISIC test
- choose a referral threshold from HIBA
- drop a constituent model after seeing test/HIBA ensemble results
- redefine the primary ensemble after seeing results
- redefine DenseNet169 as comparator based on anything other than frozen validation evidence
- use Shared-Hard or oracle predictions in the ITT consensus
- retrain constituent CNNs during the initial ITT experiment
- overwrite `results/paper/`, `configs/paper/`, or ICCIT manuscript artifacts

## 17. Output boundary

New ITT artifacts must go only under:

- `configs/extensions/itt2026_consensus/`
- `results/extensions/itt2026_consensus/`
- `docs/extensions/itt2026_consensus/`
- `scripts/itt2026/`

## 18. Phase ITT-01 closure condition

Phase ITT-01 is closed once this protocol and its machine-readable configuration are committed before any ITT ensemble computation.

The next phase is **ITT-02 — Implementation**, where reusable code is created and tested against synthetic/unit-test fixtures first. Running the frozen ISIC test or HIBA ensemble evaluation is not part of ITT-02.
