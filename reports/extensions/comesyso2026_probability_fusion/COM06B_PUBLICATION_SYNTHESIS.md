# COM-06B Publication Synthesis

Status: **PASS**

Source: validated com06a_evidence_inventory.json only.
No scientific metric, confidence interval, prediction, or rescue count is recomputed by this script.

## Table 1. Four-class macro-F1 across systems

| Dataset | Flat | Hard | Path-Soft | Probability Fusion | Oracle |
|---|---:|---:|---:|---:|---:|
| ISIC internal test | 0.6504 ± 0.0179 | 0.3808 ± 0.1189 | 0.3344 ± 0.1621 | 0.3469 ± 0.1177 | 0.5929 ± 0.1136 |
| HIBA zero-shot | 0.6109 ± 0.0165 | 0.3624 ± 0.1113 | 0.2822 ± 0.1403 | 0.2730 ± 0.1042 | 0.5491 ± 0.1169 |

Values are mean ± sample SD across seeds 42, 123, and 2026; seeds are not pooled.
Oracle is a diagnostic condition rather than a deployable system.

## Table 2. Primary comparison: Probability Fusion − Hard

| Dataset | Seed 42 | Seed 123 | Seed 2026 | Mean ± SD |
|---|---:|---:|---:|---:|
| ISIC internal test | -0.0280 | -0.0393 | -0.0345 | -0.0340 ± 0.0057 |
| HIBA zero-shot | -0.0662 | -0.1070 | -0.0951 | -0.0894 ± 0.0210 |

Negative values indicate lower four-class macro-F1 for Probability Fusion than Hard routing.

## Table 3. Secondary macro-F1 contrasts

| Dataset | Path-Soft − Hard | Fusion − Flat | Oracle − Fusion |
|---|---:|---:|---:|
| ISIC internal test | -0.0464 ± 0.0545 | -0.3036 ± 0.1163 | 0.2460 ± 0.0522 |
| HIBA zero-shot | -0.0802 ± 0.0416 | -0.3379 ± 0.1118 | 0.2762 ± 0.0729 |

## Table 4. Probability Fusion per-class F1

| Dataset | Non-malignant | Melanoma | BCC | SCC |
|---|---:|---:|---:|---:|
| ISIC internal test | 0.8051 ± 0.0238 | 0.3845 ± 0.1096 | 0.1979 ± 0.3427 | 0.0000 ± 0.0000 |
| HIBA zero-shot | 0.7455 ± 0.0362 | 0.1921 ± 0.1137 | 0.1543 ± 0.2672 | 0.0000 ± 0.0000 |

## Table 5. Seed-level paired-bootstrap uncertainty for Fusion − Hard macro-F1

| Dataset | Seed | Point difference | 95% CI | Resampling unit | Replicates |
|---|---:|---:|---:|---|---:|
| ISIC internal test | 42 | -0.0280 | [-0.0379, -0.0184] | image | 10000 |
| ISIC internal test | 123 | -0.0393 | [-0.0520, -0.0266] | image | 10000 |
| ISIC internal test | 2026 | -0.0345 | [-0.0545, -0.0140] | image | 10000 |
| HIBA zero-shot | 42 | -0.0662 | [-0.0870, -0.0447] | patient_cluster | 5000 |
| HIBA zero-shot | 123 | -0.1070 | [-0.1307, -0.0831] | patient_cluster | 5000 |
| HIBA zero-shot | 2026 | -0.0951 | [-0.1337, -0.0558] | patient_cluster | 5000 |

ISIC uses the frozen paired image bootstrap. HIBA uses the frozen patient-cluster bootstrap.

## Table 6. Routing-rescue accounting for Probability Fusion versus Hard

| Dataset | Seed | Malignant→NM failures | Rescued | Harmed | NM→malignant failures | Corrections | New errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| ISIC internal test | 42 | 808 | 0 | 0 | 462 | 308 | 0 |
| ISIC internal test | 123 | 635 | 0 | 0 | 725 | 626 | 0 |
| ISIC internal test | 2026 | 314 | 0 | 0 | 622 | 327 | 0 |
| HIBA zero-shot | 42 | 454 | 0 | 0 | 47 | 42 | 0 |
| HIBA zero-shot | 123 | 363 | 0 | 0 | 69 | 61 | 0 |
| HIBA zero-shot | 2026 | 241 | 0 | 0 | 44 | 29 | 0 |

Counts are copied from the frozen rescue-analysis artifacts. They are not re-derived here.

## Manuscript-ready scientific synthesis

Across the three frozen seeds, Probability Fusion did not improve the primary endpoint over Hard routing on either evaluation cohort. On the ISIC internal test, Fusion − Hard macro-F1 was -0.0340 ± 0.0057, with a negative difference for every seed. On the zero-shot HIBA cohort, the corresponding difference was -0.0894 ± 0.0210, again negative for every seed.

The same direction was observed for the fixed Path-Soft construction. Path-Soft − Hard macro-F1 was -0.0464 ± 0.0545 on ISIC and -0.0802 ± 0.0416 on HIBA. Thus, neither propagating conditional probabilities through the hierarchy nor fitting the prespecified validation-only multinomial fusion layer removed the observed end-to-end degradation relative to Hard routing.

The Oracle diagnostic remained substantially above Probability Fusion. Oracle − Fusion macro-F1 was 0.2460 ± 0.0522 on ISIC and 0.2762 ± 0.0729 on HIBA. Because the downstream shared-model outputs are unchanged in the Oracle condition, this gap is consistent with routing decisions remaining an important source of lost end-to-end performance.

External transfer exposed a severe minority-class weakness in the fixed fusion formulation. On HIBA, Probability Fusion SCC F1 was 0.0000 ± 0.0000 across the three seeds. The seed-level rescue tables should therefore be reported alongside aggregate accuracy so that majority-class behavior is not mistaken for balanced four-class improvement.

The HIBA evaluation is zero-shot: the fusion models were fitted on the frozen ISIC validation partition and transferred unchanged to HIBA. HIBA uncertainty is based on the prespecified patient-cluster bootstrap, while the ISIC internal analysis uses the frozen paired image bootstrap.

These results support a narrow conclusion: under this shared-backbone, hard-routing experimental setup, the two prespecified probability-propagation alternatives did not recover the primary macro-F1 loss associated with the hierarchy. They do not establish that hierarchical classification is inherently inferior, nor do they evaluate tuned soft-routing variants, dedicated branch models, or alternative calibration cohorts.
