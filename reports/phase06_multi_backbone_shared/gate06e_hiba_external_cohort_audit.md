# Gate 06E — HIBA External Cohort Audit and Freeze

## Verdict

**PASS / CLOSED.**

A frozen zero-shot external evaluation cohort was constructed from the HIBA Skin Lesions dataset without using HIBA for training, fine-tuning, threshold tuning, checkpoint selection, preprocessing changes, or model reselection.

## Source provenance

Dataset: HIBA Skin Lesions

Source archive files downloaded directly on the GPU VM from the official ISIC Archive distribution.

Source metadata SHA-256:

`57d0b754effb4d066bc7e75f964522fa01867ebea67a633b0eb652d742aebcc1`

Source ZIP SHA-256:

`e5635c842be4a9e0614dbf551a2830ae93157d8ce27a34076d6a131e6718bd2f`

The separately downloaded metadata CSV and the metadata CSV extracted from the ZIP had identical SHA-256 values.

Archive integrity check completed with no compressed-data errors. The extracted archive contained 1,635 images.

## Source metadata audit

Total metadata rows: 1,635

Image type distribution:

- dermoscopic: 1,280
- clinical overview: 349
- clinical close-up: 6

Overall benign/malignant counts:

- benign: 871
- malignant: 764

Dermoscopic diagnosis distribution:

- nevus: 555
- basal cell carcinoma: 230
- melanoma: 196
- squamous cell carcinoma: 111
- seborrheic keratosis: 47
- actinic keratosis: 46
- vascular lesion: 41
- dermatofibroma: 39
- solar lentigo: 14
- lichenoid keratosis: 1

Dermoscopic diagnosis confirmation:

- histopathology: 762
- missing: 516
- single-image expert consensus: 2

All dermoscopic rows had both lesion and patient identifiers.

## Frozen target mapping

The external cohort matches the frozen four-class ISIC task:

`[non_malignant, melanoma, bcc, scc]`

Mapping:

- melanoma -> melanoma
- basal cell carcinoma -> bcc
- squamous cell carcinoma -> scc
- nevus -> non_malignant
- seborrheic keratosis -> non_malignant
- dermatofibroma -> non_malignant
- vascular lesion -> non_malignant
- solar lentigo -> non_malignant
- lichenoid keratosis -> non_malignant

Actinic keratosis was excluded to preserve consistency with the frozen ISIC four-class task construction.

All non-dermoscopic images were excluded to avoid conflating external-domain shift with an imaging-modality shift.

## Initial eligible cohort

After modality and diagnosis filtering:

- included: 1,234
- excluded: 401

Initial class distribution:

- non_malignant: 697
- melanoma: 196
- bcc: 230
- scc: 111

Exclusions:

- 355 non-dermoscopic images
- 46 dermoscopic actinic-keratosis images

## External leakage audit

The 1,234 eligible images were compared against the frozen ISIC 2019 manifest using both ISIC IDs and exact image-content SHA-256 hashes.

Results:

- ISIC 2019 manifest rows: 25,331
- HIBA eligible rows: 1,234
- missing HIBA image files: 0
- HIBA vs ISIC 2019 ID overlap: **0**
- HIBA vs ISIC 2019 exact content-hash overlap: **0**

Therefore no exact image leakage from the frozen ISIC 2019 dataset was detected.

## HIBA internal duplicate audit

Among the 1,234 eligible images:

- unique image-content SHA-256 values: 1,232
- exact-content duplicate groups: 2
- duplicate rows involved: 4
- maximum copies of a single exact hash: 2

Duplicate group 1 contained two BCC rows with identical image bytes but different HIBA patient and lesion identifiers.

Duplicate group 2 contained two nevus rows with identical image bytes but different HIBA patient and lesion identifiers.

In both groups, diagnosis and four-class target labels were consistent.

To prevent exact duplicate content from receiving repeated metric weight, the frozen deterministic rule is:

> For identical image SHA-256 content, retain the lexicographically smallest ISIC ID and exclude the other copy.

Two duplicate copies were therefore removed.

## Final frozen external cohort

Final external cohort size: **1,232 images**

Class distribution:

- non_malignant: 696
- melanoma: 196
- bcc: 229
- scc: 111

Cohort structure:

- unique patients: 568
- unique lesions: 1,158
- unique image-content hashes: 1,232

Final diagnosis confirmation distribution:

- histopathology: 738
- missing: 492
- single-image expert consensus: 2

Final exclusions from the complete 1,635-image HIBA archive:

- 355 non-dermoscopic images
- 46 dermoscopic actinic-keratosis images
- 2 exact-content duplicate copies
- total excluded: 403

Thus:

`1635 - 403 = 1232`

## Frozen artifacts

Final manifest:

`data/external/hiba/manifests/hiba_external_dermoscopic_4class_final.csv`

SHA-256:

`a2f30f14a249d8acb2bd9f03884e5e41c773ddbee19910d5b739407521e3706a`

Final exclusion manifest:

`data/external/hiba/manifests/hiba_external_exclusions_final.csv`

SHA-256:

`49cf31b49fa58159f745671636548c2c50f1a3a18f6bca09b128fff307d1c3a9`

Cohort audit JSON:

`data/external/hiba/manifests/hiba_external_cohort_audit.json`

SHA-256 at creation:

`a8c777924f002c7ce47fb9cbbbfd11328cc5449bf9d2bd4abbd08f1db207dd9d`

The final manifest stores repo-root-relative image paths and per-image SHA-256 values.

## Frozen external-evaluation governance

From this point onward:

1. The 1,232-image HIBA cohort must not be filtered, relabeled, expanded, or reduced based on model performance.
2. HIBA must not be used for training or fine-tuning.
3. HIBA must not be used for checkpoint selection, backbone selection, threshold tuning, calibration fitting, routing-rule tuning, or preprocessing changes.
4. All external models must use the already frozen internal-validation-selected checkpoints.
5. External inference must use the frozen preprocessing semantics from the ISIC evaluation protocol.
6. All predeclared model/backbone results must be reported; unfavorable external results may not be omitted.
7. Patient clustering should be respected for uncertainty estimation because multiple images may belong to the same patient.
8. Raw HIBA images must not be committed to Git; only manifests, audit evidence, code, and permitted derived evaluation evidence should be versioned.

## Scientific role

The HIBA evaluation is a strict zero-shot external-generalization experiment. Its purpose is to test whether the internal flat-versus-hard-hierarchy findings and routing-associated behavior persist under an independent dermoscopic source distribution.

It is not a new model-development stage.

## Gate conclusion

**Gate 06E: PASS / CLOSED.**

The HIBA external cohort is integrity-checked, leakage-audited, exact-content deduplicated, mapped to the frozen four-class task, and prospectively frozen at 1,232 dermoscopic images.

## Exact next task

**Gate 06F — Frozen HIBA external evaluator implementation and preflight.**

Implement an architecture-aware zero-shot evaluator that loads the frozen seven flat and seven shared checkpoints, verifies the final HIBA manifest and image hashes, performs no training or tuning, supports hard and oracle routing diagnostics where labels permit, and computes external metrics plus patient-clustered uncertainty from saved predictions. Preflight must complete before the one-time full GPU inference run.