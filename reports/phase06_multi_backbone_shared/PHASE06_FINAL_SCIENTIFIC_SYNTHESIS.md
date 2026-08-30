# Phase 06 — Final Scientific Synthesis

## Status

**CLOSED / PASS**

Phase 06 extended the original flat-versus-shared-hard-hierarchy comparison from one backbone to seven matched backbone families and then tested the same frozen models under zero-shot external generalization on HIBA.

No HIBA sample was used for training, fine-tuning, threshold selection, checkpoint selection, calibration fitting, preprocessing changes, or backbone reselection.

## Research questions

Phase 06 addressed three questions:

1. Does the flat-versus-hard-hierarchy result depend strongly on backbone choice?
2. Is routing loss consistently observable across multiple matched backbone families?
3. Do the internal findings persist on an independent external dermoscopic cohort?

## Backbones

Seven matched flat/shared backbone pairs were evaluated:

- DenseNet121
- DenseNet169
- ResNet50
- MobileNetV3-Large
- EfficientNet-B0
- EfficientNet-B2
- EfficientNet-B3

All checkpoints were selected before the relevant locked evaluation from internal validation only.

---

# 1. Locked internal-test results

Frozen ISIC 2019 internal test: **N = 3,668**.

| Backbone | Flat macro-F1 | Shared hard-routing macro-F1 | Oracle-routing macro-F1 | Hard − Flat | 95% paired-bootstrap CI |
|---|---:|---:|---:|---:|---:|
| DenseNet121 | 0.635107 | 0.560878 | 0.765942 | -0.074229 | [-0.103054, -0.043784] |
| DenseNet169 | 0.649331 | 0.542656 | 0.725136 | -0.106675 | [-0.136078, -0.076100] |
| EfficientNet-B0 | 0.619222 | 0.568591 | 0.776976 | -0.050632 | [-0.075993, -0.024827] |
| EfficientNet-B2 | 0.619951 | 0.580799 | 0.749212 | -0.039153 | [-0.067643, -0.010772] |
| EfficientNet-B3 | 0.626040 | 0.594386 | 0.796147 | -0.031653 | [-0.062006, -0.001032] |
| MobileNetV3-Large | 0.619004 | 0.636972 | 0.806338 | +0.017968 | [-0.009606, +0.045891] |
| ResNet50 | 0.640989 | 0.546998 | 0.745743 | -0.093991 | [-0.121167, -0.067112] |

### Internal interpretation

Hard routing underperformed the matched flat classifier for six of seven backbones. MobileNetV3-Large was the only numerical exception, with a +0.0180 macro-F1 difference, but its paired-bootstrap interval crossed zero.

Therefore the internal evidence does **not** support a claim that hierarchy-minus-flat direction is mathematically invariant to backbone choice. The correct conclusion is that hard routing generally reduced end-to-end performance, while the magnitude and even numerical direction could depend on backbone.

The stronger cross-backbone result is the oracle-routing diagnostic: oracle routing substantially exceeded predicted hard routing for every backbone. This indicates that routing errors consistently prevented the downstream subtype classifier from expressing its available predictive capability.

---

# 2. Frozen HIBA external cohort

The external cohort was frozen prospectively before inference.

Final HIBA cohort:

- total images: **1,232**
- unique patients: **568**
- unique lesions: **1,158**
- non_malignant: **696**
- melanoma: **196**
- BCC: **229**
- SCC: **111**

External leakage audit against the complete frozen ISIC 2019 manifest found:

- ISIC-ID overlap: **0**
- exact image-content SHA-256 overlap: **0**

Two exact duplicate HIBA image copies were removed deterministically before freezing the final cohort.

The external comparison used the same frozen seven flat and seven shared checkpoints. Uncertainty for the external hierarchy-minus-flat macro-F1 difference used a **patient-cluster bootstrap**, accounting for patients represented by multiple images.

---

# 3. Zero-shot HIBA external results

| Backbone | Flat macro-F1 | Shared hard-routing macro-F1 | Oracle-routing macro-F1 | Hard − Flat | Patient-cluster 95% CI |
|---|---:|---:|---:|---:|---:|
| DenseNet121 | 0.586650 | 0.553003 | 0.671661 | -0.033647 | [-0.082341, +0.015894] |
| DenseNet169 | 0.633310 | 0.527826 | 0.647657 | -0.105484 | [-0.150203, -0.059934] |
| EfficientNet-B0 | 0.634546 | 0.598343 | 0.702269 | -0.036202 | [-0.072710, +0.001705] |
| EfficientNet-B2 | 0.631638 | 0.539516 | 0.673364 | -0.092122 | [-0.138625, -0.043868] |
| EfficientNet-B3 | 0.652885 | 0.594882 | 0.704156 | -0.058003 | [-0.097674, -0.019215] |
| MobileNetV3-Large | 0.634262 | 0.617092 | 0.735248 | -0.017170 | [-0.055841, +0.023876] |
| ResNet50 | 0.592521 | 0.476445 | 0.633597 | -0.116076 | [-0.158543, -0.071668] |

## External interpretation

Unlike the internal test, the independent HIBA evaluation produced a flat-favoring numerical direction for **all seven backbones**.

Four backbone pairs had patient-cluster bootstrap intervals that excluded zero in the flat-favoring direction:

- DenseNet169
- ResNet50
- EfficientNet-B2
- EfficientNet-B3

Three intervals included zero:

- DenseNet121
- EfficientNet-B0
- MobileNetV3-Large

Accordingly, the defensible external claim is not that every individual backbone establishes a statistically non-zero difference. Rather, the external experiment shows a uniform 7/7 numerical direction, with four backbone-specific intervals excluding zero and three remaining statistically inconclusive at the 95% level.

---

# 4. External routing-loss evidence

Oracle-minus-hard-routing macro-F1 gaps on HIBA were:

| Backbone | Oracle − hard macro-F1 |
|---|---:|
| EfficientNet-B0 | +0.103925 |
| DenseNet121 | +0.118658 |
| DenseNet169 | +0.119831 |
| ResNet50 | +0.157153 |
| MobileNetV3-Large | +0.118156 |
| EfficientNet-B2 | +0.133848 |
| EfficientNet-B3 | +0.109275 |

Oracle routing exceeded hard routing for **all seven external backbone pairs**.

This is the most consistent diagnostic result across both evaluation domains. The external routing gaps range from approximately +0.104 to +0.157 macro-F1, demonstrating that imperfect first-stage routing remains a substantial bottleneck after distribution shift.

Oracle routing is a diagnostic upper-bound experiment, not a deployable classifier, because it uses the true top-level branch to remove routing mistakes.

---

# 5. Combined scientific interpretation

The multi-backbone and external evidence changes the strength of the original conclusion.

The original EfficientNet-B0 experiment showed that the flat classifier outperformed the deployed shared hard-routing hierarchy and that oracle routing recovered substantial performance. Phase 06 demonstrates that this phenomenon is not adequately explained as an EfficientNet-B0-specific observation.

Across the locked ISIC internal test, hard routing was worse for six of seven backbone pairs, while MobileNetV3-Large produced a small statistically inconclusive reversal. Across the independent zero-shot HIBA cohort, hard routing was numerically worse for all seven backbone pairs. Most importantly, oracle routing improved over predicted hard routing for every backbone in both evaluation domains.

The resulting evidence supports the following central conclusion:

> A clinically intuitive hierarchy does not automatically improve end-to-end lesion classification. In a hard top-down system, first-stage routing errors can dominate the benefit of downstream specialization, and this routing bottleneck can persist across architecture families and under external distribution shift.

This is stronger and more precise than claiming simply that "hierarchical classification is worse." The experiments evaluate one specific shared-representation, hard-routing hierarchy under a fixed training protocol. They do not establish that all hierarchical methods are inferior to flat classification.

---

# 6. Publication-ready findings

The paper may state the following:

1. **Backbone robustness:** on the locked internal test, flat classification exceeded the hard hierarchy for six of seven backbones; the single MobileNetV3-Large reversal was statistically inconclusive.
2. **External generalization:** on the frozen HIBA cohort, the flat model numerically exceeded hard routing for all seven matched backbones.
3. **External uncertainty:** four of seven HIBA patient-cluster bootstrap intervals excluded zero in the flat-favoring direction; three remained inconclusive.
4. **Routing bottleneck:** oracle routing improved macro-F1 over hard routing for all seven backbones internally and externally.
5. **External independence:** no ISIC-ID or exact image-content overlap was detected between the HIBA external cohort and the frozen ISIC 2019 dataset.
6. **Zero-shot protocol:** HIBA was never used for model development, checkpoint selection, fine-tuning, threshold tuning, or preprocessing adaptation.

---

# 7. Important limitations

Publication claims must remain bounded by the following limitations:

- Only one fixed training seed was used per backbone, so training-run uncertainty is not estimated.
- The study evaluates a specific shared hard-routing hierarchy rather than all possible hierarchical formulations.
- Oracle routing is diagnostic and cannot be deployed because it requires the true upstream branch.
- HIBA contains multiple images from some patients; this was addressed for uncertainty estimation using patient-cluster bootstrap, but the primary metrics remain image-level.
- External labels and acquisition conditions differ from ISIC, and zero-shot performance therefore combines model generalization with dataset shift.
- The HIBA cohort contains 111 SCC images, still fewer than the other principal classes.
- No prospective clinical validation is presented.
- Calibration, uncertainty-aware routing, soft routing, and explainability were not evaluated in this phase.

These limitations do not negate the routing finding; they define its scope.

---

# 8. Recommended ICCIT paper framing

The manuscript should now be framed around **routing robustness rather than raw state-of-the-art accuracy**.

A suitable central narrative is:

1. construct a matched flat and shared hard-routing hierarchy for four-class dermoscopic lesion classification;
2. show the original internal flat-versus-hierarchy difference;
3. expand the comparison to seven modern CNN backbones;
4. use oracle routing to isolate routing-error effects;
5. test the same frozen models zero-shot on an independent HIBA cohort;
6. show that the routing bottleneck persists across backbone families and external distribution shift.

The primary novelty should therefore be presented as a controlled empirical analysis of **when and why a hard diagnostic hierarchy fails to translate downstream specialization into end-to-end gains**, rather than as a claim of achieving the highest classification accuracy.

---

# Phase 06 closure verdict

**PASS / CLOSED.**

No further model training or experiment expansion is required for the ICCIT submission version unless a correctness defect is discovered.

## Exact next task

**ICCIT manuscript finalization.**

Update the title, abstract, methodology, experimental setup, results, discussion, figures/tables, limitations/future-work text, and conclusion to incorporate the seven-backbone internal comparison and frozen HIBA zero-shot external-generalization evidence.