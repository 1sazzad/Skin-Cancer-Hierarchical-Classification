# ITT 2026 — Phase ITT-06 Closure

**Status:** PASS / CLOSED  
**Date:** 2026-09-21

## Primary inferential comparison

Primary comparison was frozen as:

**Seven-model hard majority vote vs DenseNet169**

DenseNet169 was prospectively selected as the individual comparator because it had the highest frozen validation macro-F1.

### ISIC 2019 internal test

- N = 3,668
- Majority-vote macro-F1 = 0.6847936721
- DenseNet169 macro-F1 = 0.6493306153
- Difference = **+0.0354630569**
- Paired image-bootstrap 95% CI = **[0.0130134727, 0.0581974018]**
- CI excludes zero
- Exact two-sided McNemar p = **4.0016359464581125e-09**
- Majority correct / DenseNet169 wrong = **248**
- Majority wrong / DenseNet169 correct = **133**

### HIBA external zero-shot cohort

- Images = 1,232
- Patients = 568
- Majority-vote macro-F1 = 0.6621826511
- DenseNet169 macro-F1 = 0.6333103424
- Difference = **+0.0288723087**
- Patient-cluster bootstrap 95% CI = **[0.0002677991, 0.0573841144]**
- CI excludes zero

The HIBA lower confidence bound is close to zero and should be interpreted conservatively.

## Secondary weighted ensemble

Validation-macro-F1-weighted hard voting produced the same point-estimate macro-F1 as the primary majority ensemble on both cohorts:

- ISIC weighted-minus-majority macro-F1 = 0.000000
- HIBA weighted-minus-majority macro-F1 = 0.000000

This does not imply prediction identity on every sample unless confirmed from per-sample outputs; it means the cohort-level macro-F1 point estimates were equal.

## Selective-referral evidence

The complete prospectively specified referral series was retained. No operating threshold was selected from either evaluation cohort.

### ISIC

| Rule | Coverage | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Full coverage | 1.000000 | [see full table] | 0.684794 |
| 7/7 | 0.518539 | 0.931651 | 0.825686 |
| >=6/7 | 0.705016 | 0.897912 | 0.778362 |
| >=5/7 | 0.848419 | 0.853792 | 0.728883 |
| >=4/7 | 0.975463 | 0.811627 | 0.689313 |

### HIBA

| Rule | Coverage | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Full coverage | 1.000000 | [see full table] | 0.662183 |
| 7/7 | 0.617695 | 0.925099 | 0.755570 |
| >=6/7 | 0.737825 | 0.886689 | 0.723554 |
| >=5/7 | 0.847403 | 0.848659 | 0.697422 |
| >=4/7 | 0.961039 | 0.804899 | 0.668536 |

These results support a descriptive coverage-performance relationship: higher cross-backbone agreement is associated with higher retained performance.

## Agreement distributions

### ISIC

- 7 votes: 1,902
- 6 votes: 684
- 5 votes: 526
- 4 votes: 466
- 3 votes: 84
- 2 votes: 6
- tie-resolution cases: 46

### HIBA

- 7 votes: 761
- 6 votes: 148
- 5 votes: 135
- 4 votes: 140
- 3 votes: 45
- 2 votes: 3
- tie-resolution cases: 26

## Statistical interpretation

Supported:

- the primary majority ensemble had higher macro-F1 than the prospectively selected DenseNet169 comparator on both evaluation cohorts;
- the prespecified paired bootstrap CI excluded zero on the internal cohort;
- the patient-cluster bootstrap CI also excluded zero on HIBA, with a lower bound very close to zero;
- internal paired correctness favored the majority ensemble under exact McNemar testing;
- higher model agreement was associated descriptively with higher retained accuracy/macro-F1.

Not supported:

- universal superiority over every possible individual model or ensemble;
- clinical benefit or deployment readiness;
- a claim that one referral threshold is optimal;
- causal interpretation of disagreement;
- claims beyond the evaluated ISIC and HIBA cohorts.

## Phase status

**ITT-06: PASS / CLOSED.**

Next: **ITT-07 — Manuscript synthesis and submission-ready evidence package.**
