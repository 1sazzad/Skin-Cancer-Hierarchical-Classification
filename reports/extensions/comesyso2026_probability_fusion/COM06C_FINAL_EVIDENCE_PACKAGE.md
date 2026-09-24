# COM-06C Final Publication Evidence Package

Status: **PASS**

This package is a publication-facing synthesis of frozen COM-03/04/05 evidence.
It performs no scientific recomputation.

## 1. Frozen study identity

- Seeds: 42, 123, 2026.
- Primary endpoint: four-class macro-F1.
- Primary comparison: Probability Fusion − Hard routing.
- Systems: Flat, Hard, Path-Soft, Probability Fusion, Oracle.
- Internal evaluation: frozen ISIC 2019 internal test.
- External evaluation: frozen HIBA cohort, zero-shot transfer.
- HIBA uncertainty: patient-cluster bootstrap.
- Oracle is diagnostic only and is not a deployable method.

## 2. Core publication numbers

| Evidence | ISIC internal test | HIBA zero-shot |
|---|---:|---:|
| Flat macro-F1 | 0.6504 ± 0.0179 | 0.6109 ± 0.0165 |
| Hard macro-F1 | 0.3808 ± 0.1189 | 0.3624 ± 0.1113 |
| Path-Soft macro-F1 | 0.3344 ± 0.1621 | 0.2822 ± 0.1403 |
| Probability Fusion macro-F1 | 0.3469 ± 0.1177 | 0.2730 ± 0.1042 |
| Oracle macro-F1 | 0.5929 ± 0.1136 | 0.5491 ± 0.1169 |
| Fusion − Hard | -0.0340 ± 0.0057 | -0.0894 ± 0.0210 |
| Path-Soft − Hard | -0.0464 ± 0.0545 | -0.0802 ± 0.0416 |
| Fusion − Flat | -0.3036 ± 0.1163 | -0.3379 ± 0.1118 |
| Oracle − Fusion | 0.2460 ± 0.0522 | 0.2762 ± 0.0729 |

Values are mean ± sample SD across the three frozen seeds; seed-image pairs are not pooled.

## 3. Seed-level primary result

- ISIC Fusion − Hard: 42=-0.0280, 123=-0.0393, 2026=-0.0345.
- HIBA Fusion − Hard: 42=-0.0662, 123=-0.1070, 2026=-0.0951.
- The primary macro-F1 contrast is negative for every frozen seed on both cohorts.

## 4. Minority-class evidence

| Dataset | Fusion Non-malignant F1 | Fusion Melanoma F1 | Fusion BCC F1 | Fusion SCC F1 |
|---|---:|---:|---:|---:|
| ISIC internal test | 0.8051 ± 0.0238 | 0.3845 ± 0.1096 | 0.1979 ± 0.3427 | 0.0000 ± 0.0000 |
| HIBA zero-shot | 0.7455 ± 0.0362 | 0.1921 ± 0.1137 | 0.1543 ± 0.2672 | 0.0000 ± 0.0000 |

## 5. Claims supported by the frozen evidence

1. Under the frozen shared-backbone hierarchy, the prespecified Probability Fusion model did not improve four-class macro-F1 over Hard routing on the internal ISIC test or the zero-shot HIBA cohort.
2. The fixed Path-Soft probability product also did not improve macro-F1 over Hard routing on either cohort.
3. The Oracle diagnostic remained materially above Probability Fusion, supporting routing decisions as an important source of lost end-to-end performance in this setup.
4. External HIBA transfer exposed substantial minority-class weakness in Probability Fusion; aggregate accuracy must therefore not be used as a substitute for balanced four-class evaluation.
5. The HIBA result is a true zero-shot transfer of the ISIC-validation-fitted fusion models; no HIBA fitting or tuning was performed.

## 6. Claims NOT supported

- Do not claim that hierarchical classification is inherently inferior to flat classification.
- Do not claim that all soft-routing or probability-fusion methods fail.
- Do not claim external calibration, because the fusion model was fitted on the ISIC validation partition rather than an independent calibration cohort.
- Do not present Oracle as a practical deployment method.
- Do not present accuracy gains, if any, as evidence of balanced four-class improvement when macro-F1 and minority-class F1 decline.
- Do not pool the three seeds into one enlarged sample.

## 7. Recommended manuscript evidence layout

**Main Table 1 — System comparison.** Report Flat, Hard, Path-Soft, Probability Fusion, and Oracle macro-F1 for ISIC and HIBA as mean ± sample SD.

**Main Table 2 — Primary contrast and uncertainty.** Report Fusion − Hard by seed together with the frozen 95% bootstrap intervals. State explicitly that ISIC uses paired image bootstrap and HIBA uses patient-cluster bootstrap.

**Main Table 3 — Per-class F1.** At minimum show Probability Fusion and Hard per-class F1 so the minority-class behavior is visible.

**Figure 1 — Method diagram.** Show shared model outputs P(Task1 malignant) and P(Task2 MEL/BCC/SCC), the fixed Path-Soft product path, and the validation-only multinomial logistic Probability Fusion path. Mark HIBA as zero-shot and unchanged.

**Figure 2 — Primary macro-F1 comparison.** Plot the five systems for ISIC and HIBA, using the frozen three-seed summaries. Oracle should be visually labeled diagnostic.

**Figure 3 — Routing-rescue diagnostic.** Visualize, by seed, Hard-wrong/Fusion-correct versus Hard-correct/Fusion-wrong counts and malignant→NM routing failures.

## 8. Manuscript-ready interpretation

Probability Fusion reduced macro-F1 relative to Hard routing by -0.0340 ± 0.0057 on the frozen ISIC internal test and -0.0894 ± 0.0210 on the zero-shot HIBA cohort. The direction was consistent across all three seeds on both datasets. Path-Soft likewise remained below Hard by -0.0464 ± 0.0545 on ISIC and -0.0802 ± 0.0416 on HIBA. At the same time, Oracle exceeded Probability Fusion by 0.2460 ± 0.0522 on ISIC and 0.2762 ± 0.0729 on HIBA, indicating that the available downstream predictions retained information that was not recovered by the tested routing alternatives. The result therefore narrows the conclusion: within this frozen shared-backbone hierarchy, neither the deterministic probability-product formulation nor the prespecified validation-trained multinomial fusion layer recovered the macro-F1 degradation associated with routing.

## 9. Limitations to retain

- Three frozen seeds only.
- Shared-backbone hierarchy; dedicated branch models were not tested here.
- Hard, fixed Path-Soft, and one prespecified multinomial fusion formulation only.
- ISIC validation was used for checkpoint selection and fusion fitting; it is not an independent calibration cohort.
- No HIBA adaptation or recalibration.
- External cohort is the frozen historical 1,232-image HIBA cohort with patient-cluster bootstrap.
- The study diagnoses routing-associated degradation; it does not exhaust the broader design space of hierarchical classifiers.

## 10. Provenance

- COM-06A inventory SHA256: 5dbe66cd0918d58b006abf76f4d6f955aa990cf3f8c6c313b4ffcf9cd9536fa4
- COM-06B synthesis SHA256: 0ba52320aeb423747ccbb6f18eae462c2ce2cc0edf524c26a8ace092d0f502e2
- external_hiba_complete_sha256: e46113cfb1c6e4a01f0b0157c3fbf2059f5a95e5188166c7c8141d56a67b235d
- external_hiba_inference_complete_sha256: 107f51b01f2f56fd3f0b23264ee7f13bc14ee723c419459b22b1cd0aa69661c1
- internal_isic_complete_sha256: bd8183c9573cf2f117f1de72197016d80df9945aedb12d6bb51366f4e046a530
- validation_fit_complete_sha256: cb94a2dc626b17917d38f29d8f17a4e560defd2a16ccbfb28d749dcc6e406e81

COM-06 scientific synthesis is complete and ready for manuscript drafting.
