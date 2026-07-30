# Phase 09 ISIC-Derived Stage-3 Fast-Track Result

## Status and scope

Phase 09 is complete. This result is standalone feasibility evidence for
five-class melanoma T-category classification. It is not an integrated
three-stage system, a statistical-superiority result, or evidence suitable for
clinical deployment.

The paper-facing dataset name is **ISIC-derived melanoma T-category subset**.
It must not be called the EMB dataset.

## Provenance and licensing

The EMB repository at commit
`3ec674f43e73cb08682b99b7fb996aca5f8040d8` had no identifiable licence.
Consequently, no EMB images were acquired and no Atlas images were used. The
EMB CSV served only as an index of candidate public ISIC identifiers.

Every candidate was independently resolved through the official ISIC Archive
API v2. Official metadata supplied the authoritative diagnosis, Breslow
thickness, modality, patient and lesion identifiers, public status, per-image
licence, and attribution. The exact accepted licence and attribution remain in
the audit inventory. The eligible licences were CC-0 (586), CC-BY (247), and
CC-BY-NC (15); no eligible image lacked attribution.

## Label derivation

Official ISIC metadata defines the broad labels:

- melanoma in situ: Tis
- invasive melanoma with thickness greater than 0 and at most 1.0 mm: T1
- thickness greater than 1.0 and at most 2.0 mm: T2
- thickness greater than 2.0 and at most 4.0 mm: T3
- thickness greater than 4.0 mm: T4

Ulceration is retained as provenance but does not change these broad labels.
The original EMB `stage_ajcc` value is not an authoritative training label.

## Dataset audit

Of 856 candidates, 848 were eligible and 8 were excluded. The eligible class
counts were Tis 509, T1 264, T2 47, T3 14, and T4 14. All 848 eligible images
were present and readable. Missing attribution, exact duplicate hash groups,
conflicting duplicate labels, ISIC 2019 image-ID overlap, and ISIC 2019
SHA-256 overlap were all zero.

## Deterministic leakage-safe split

The seed-42 split contains 594 train, 127 validation, and 127 untouched test
images:

| Split | Tis | T1 | T2 | T3 | T4 | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 355 | 184 | 33 | 10 | 12 | 594 |
| Validation | 77 | 40 | 7 | 2 | 1 | 127 |
| Test | 77 | 40 | 7 | 2 | 1 | 127 |

All non-empty patient IDs, lesion IDs, and image SHA-256 relations were joined
by transitive connected components and assigned intact. There were 452
components: 443 single-label, 9 multi-label patient components, and 126
multi-image components; the maximum component size was 28. Cross-split overlap
counts were zero for patient ID, lesion ID, SHA-256, and split-group ID.

Patient and lesion identifiers are incomplete. A patient may have different
lesions with different T-categories, so patient-safe components may be
multi-label. Same-lesion or same-hash label conflicts remain fatal.

## Validation selection

Both candidates used EfficientNet-B0, ImageNet initialization, seed 42, the
same data pipeline and split, AdamW, cosine annealing, AMP, at most 30 epochs,
early stopping, and validation macro-F1 checkpoint selection.

The ordinary cross-entropy baseline selected epoch 2 with validation macro-F1
`0.36561465460163317` (protocol threshold recorded as `0.365615`). Its locked
test was then consumed. It showed strong Tis majority collapse.

Exactly one imbalance-aware candidate was permitted. Its only controlled
change was train-only inverse-frequency weighted cross-entropy with weights:
Tis `0.063475735584673`, T1 `0.12246677245955931`, T2
`0.682845034319967`, T3 `2.253388613255891`, and T4
`1.8778238443799093`. It selected epoch 12 with validation macro-F1
`0.43657311157311157`. Because this was strictly greater than `0.365615`, it
was selected before its internal test was accessed.

## Locked internal-test comparison

| Metric | Ordinary CE | Weighted CE |
| --- | ---: | ---: |
| Sample count | 127 | 127 |
| Mean loss | 0.9036416960513498 | 1.1099267015306968 |
| Accuracy | 0.6062992125984252 | 0.5433070866141733 |
| Balanced accuracy | 0.20240259740259742 | 0.38603896103896107 |
| Macro-F1 | 0.16283767911674887 | 0.2756106656721984 |
| Weighted-F1 | 0.48009115139677305 | 0.4932469633725395 |

Weighted cross-entropy improved macro-F1, balanced accuracy, and weighted-F1,
while overall accuracy decreased.

### Per-class test comparison

| Class | Support | Baseline precision | Baseline recall | Baseline F1 | Weighted precision | Weighted recall | Weighted F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tis | 77 | 0.628099173553719 | 0.987012987012987 | 0.7676767676767676 | 0.6458333333333334 | 0.8051948051948052 | 0.7167630057803468 |
| T1 | 40 | 0.3333333333333333 | 0.025 | 0.046511627906976744 | 0.22727272727272727 | 0.125 | 0.16129032258064516 |
| T2 | 7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| T3 | 2 | 0.0 | 0.0 | 0.0 | 0.3333333333333333 | 1.0 | 0.5 |
| T4 | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

T2 and T4 recall remained zero. Weighted CE classified both T3 samples
correctly, but support was only two, so this is fragile descriptive evidence.

### Ordinary cross-entropy confusion matrix

Rows are actual and columns are predicted in `[Tis, T1, T2, T3, T4]` order.

| Actual \ Predicted | Tis | T1 | T2 | T3 | T4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tis | 76 | 1 | 0 | 0 | 0 |
| T1 | 36 | 1 | 0 | 3 | 0 |
| T2 | 6 | 1 | 0 | 0 | 0 |
| T3 | 2 | 0 | 0 | 0 | 0 |
| T4 | 1 | 0 | 0 | 0 | 0 |

### Weighted cross-entropy confusion matrix

| Actual \ Predicted | Tis | T1 | T2 | T3 | T4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tis | 62 | 14 | 1 | 0 | 0 |
| T1 | 31 | 5 | 2 | 2 | 0 |
| T2 | 3 | 3 | 0 | 1 | 0 |
| T3 | 0 | 0 | 0 | 2 | 0 |
| T4 | 0 | 0 | 0 | 1 | 0 |

## Limitations and claims lock

The subset is highly imbalanced, with very small T3 and T4 validation/test
support. Metadata-derived broad T-categories do not constitute clinical
staging, and the study has one seed and one internal split. Patient/lesion
metadata is incomplete, although every known relation was kept leakage-safe.
The two locked test evaluations support descriptive comparison only.

Allowed claims are that a licensed, officially relabelled ISIC-derived
standalone Stage-3 subset was feasible; the preselected weighted candidate
improved locked-test macro-F1 and balanced accuracy relative to the baseline;
and rare-class performance remains inadequate and uncertain.

Prohibited claims include statistical superiority, external generalisation,
an integrated three-stage result, comprehensive AJCC staging, or readiness for
clinical deployment.

## Evidence hashes

| Evidence | SHA-256 |
| --- | --- |
| Split manifest | `12c9379ff1f7e2098d6b84554ca69f30c56fc451e0b33238866c2c02c5a17396` |
| Split audit | `b6e88fc7e2784efbba9f63a1de9e8a51718e8648408b6938d082207fd57bbc9c` |
| Baseline internal-test metrics | `3e101d72d145d8cbc570d78356acf40d8b13775fb116ef36dd7a29004cf06e7b` |
| Baseline predictions | `27aaf9f52bdfccbf82ba5e547551bd812123e4c6db8b98144e23faf4eb398212` |
| Baseline evaluation summary | `0e09ea527e8b0ca9c60e1ef277b8e34897412c141e715f569f669f4d58671cf0` |
| Baseline confusion matrix | `ce72a12b87ab5a42aa5e8de7553ef590e48d40786ce899ef477ed52fe75f0acc` |
| Baseline per-class metrics | `cdb6ff90bdf0f90a25ee898ccbdf873e38efdb6d0d5a09e3d2acaf2aa8dacbdb` |
| Weighted internal-test metrics | `bbca6d888339cf9a6224d791d3c72f4841a9c397e3f9f74b516fd564926116f3` |
| Weighted predictions | `445a633248e95644647fae0818b59b1f939fda57ba289b1ccdc59c4e3aef9cc0` |
| Weighted evaluation summary | `bb005f8b024fcaadf0e593464c976437681dcb3f35d9d2b907866db3ac45bffa` |
| Weighted confusion matrix | `0fa151622be27ddbd82efd283f9a7c110ad7c5e14460401c13a112b2feaf9ae0` |
| Weighted per-class metrics | `fb4205a8ca4e3cfc8e6a4b2427ab38b8c8aa342212a77e9306f85990b2df8089` |

## Final lock

Phase 09 is closed. Both internal-test evaluations are consumed and
`rerun_allowed=false`. There will be no further Stage-3 tuning or internal-test
rerun. Future work must reuse these immutable evidence files and preserve the
validation-first selection record.
