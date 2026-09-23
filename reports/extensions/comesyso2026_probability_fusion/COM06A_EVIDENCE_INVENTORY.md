# COM-06A Evidence Inventory

Status: **PASS**

This inventory is read-only. It validates the frozen COM-03/04/05 completion
markers and summarizes already-published metrics; it does not rerun inference,
bootstrap, fitting, training, or metric computation.

## Primary system results

| Dataset | Flat | Hard | Path-Soft | Fusion | Oracle |
|---|---:|---:|---:|---:|---:|
| ISIC internal test | 0.6504 ± 0.0179 | 0.3808 ± 0.1189 | 0.3344 ± 0.1621 | 0.3469 ± 0.1177 | 0.5929 ± 0.1136 |
| HIBA zero-shot | 0.6109 ± 0.0165 | 0.3624 ± 0.1113 | 0.2822 ± 0.1403 | 0.2730 ± 0.1042 | 0.5491 ± 0.1169 |

Values are four-class macro-F1, mean ± sample SD across seeds 42, 123, 2026; seeds are not pooled.

## Primary contrast: Fusion - Hard

| Dataset | Seed 42 | Seed 123 | Seed 2026 | Mean ± SD |
|---|---:|---:|---:|---:|
| ISIC internal test | -0.0280 | -0.0393 | -0.0345 | -0.0340 ± 0.0057 |
| HIBA zero-shot | -0.0662 | -0.1070 | -0.0951 | -0.0894 ± 0.0210 |

## Secondary macro-F1 contrasts

| Dataset | Path-Soft - Hard | Fusion - Flat | Oracle - Fusion |
|---|---:|---:|---:|
| ISIC internal test | -0.0464 ± 0.0545 | -0.3036 ± 0.1163 | 0.2460 ± 0.0522 |
| HIBA zero-shot | -0.0802 ± 0.0416 | -0.3379 ± 0.1118 | 0.2762 ± 0.0729 |

## Fusion per-class F1

| Dataset | Non-malignant | Melanoma | BCC | SCC |
|---|---:|---:|---:|---:|
| ISIC internal test | 0.8051 ± 0.0238 | 0.3845 ± 0.1096 | 0.1979 ± 0.3427 | 0.0000 ± 0.0000 |
| HIBA zero-shot | 0.7455 ± 0.0362 | 0.1921 ± 0.1137 | 0.1543 ± 0.2672 | 0.0000 ± 0.0000 |

## Seed-level primary uncertainty

The JSON inventory preserves, for every seed and dataset, the frozen paired-bootstrap
95% confidence interval for Fusion - Hard macro-F1. ISIC also preserves the declared
image-level exact McNemar correctness test; HIBA intentionally has no additional
image-level McNemar test under the frozen protocol.

## Routing rescue evidence

The JSON inventory preserves each seed's complete frozen rescue-analysis object,
including the five-way Hard/Fusion correctness partition, malignant→NM routing
failures, NM→malignant routing failures, rescues, harms, and subtype counts.

## Provenance

- \`validation_fit_complete_sha256\`: \`cb94a2dc626b17917d38f29d8f17a4e560defd2a16ccbfb28d749dcc6e406e81\`
- \`internal_isic_complete_sha256\`: \`bd8183c9573cf2f117f1de72197016d80df9945aedb12d6bb51366f4e046a530\`
- \`external_hiba_inference_complete_sha256\`: \`107f51b01f2f56fd3f0b23264ee7f13bc14ee723c419459b22b1cd0aa69661c1\`
- \`external_hiba_complete_sha256\`: \`e46113cfb1c6e4a01f0b0157c3fbf2059f5a95e5188166c7c8141d56a67b235d\`

COM-06A is ready for publication-table and narrative synthesis.
