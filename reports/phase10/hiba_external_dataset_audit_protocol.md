# Phase 10A HIBA External-Dataset Audit Protocol

## Status and scientific role

Protocol status: `candidate_pending_official_acquisition_audit`.

HIBA is mandatory only as a frozen, zero-shot external evaluation dataset. It
must not influence training, fine-tuning, checkpoint or architecture
selection, calibration, thresholds, augmentation, or preprocessing. No HIBA
performance may be viewed during this audit. The flat and hierarchical models
will use the exact same approved manifest rows.

The intended official release is **Hospital Italiano de Buenos Aires - Skin
Lesions Images (2019-2022)**, ISIC collection `251`, dataset DOI
`10.34970/587329`, described as containing 1,616 images. It must not be
substituted with older collection `175` / DOI `10.34970/559884`. These
identifiers select the intended source; they do not assert that the files
currently present are official or complete. Acquisition must preserve the
release title, collection identifier, DOI/source URL, per-image source
reference, licence, and attribution supplied by the official ISIC record.

## Expected inputs and outputs

Raw files, if separately acquired by an authorized later task, belong under
`data/external/hiba/`. Suggested immutable locations are:

- `data/external/hiba/images/`
- `data/external/hiba/metadata/`
- `data/external/hiba/source/` for downloaded release notices and attribution

The audit writes no raw file. Its controlled outputs are:

- `data/manifests/hiba_dataset_manifest.csv`
- `data/manifests/hiba_dataset_manifest.audit.json`
- `data/checksums/hiba_sha256.txt`

HIBA is one external cohort. No train, validation, or test split may be
created.

## Metadata and licence gate

Official metadata must identify every image and provide diagnosis, modality,
licence, attribution, and source reference. Patient and lesion identifiers
must be preserved when supplied; genuinely missing values remain empty rather
than being synthesized. The exact official release identity, metadata
headers, support, and licence terms must be inspected after acquisition.

Use is permitted only where the official image record authorizes the intended
research use and its attribution can be retained. A bare paper licence must
not be assumed to license every dataset image. Missing, conflicting, or
unsupported licence or attribution blocks approval. The prospective accepted
licence string for the selected ISIC collection 251 / DOI `10.34970/587329`
release is explicitly declared in the mapping YAML as `CC-BY`, matching the
current official ISIC Archive description. Comparison uses only Unicode
case-folding and whitespace normalization; punctuation is not altered. The
exact original value is always retained in the manifest. `CC BY 4.0`,
`Creative Commons Attribution 4.0 International`, and every other different
string remain unsupported unless that exact value is later observed in the
acquired official per-image metadata and approved by human review before
inference. Redistribution must follow the official terms.

## Modality and label gate

The primary project modality is dermoscopic imaging. Only rows explicitly
identified by official metadata as dermoscopic are eligible. Clinical,
smartphone clinical, mixed/unclear, and missing modalities are excluded with
an explicit reason.

Original diagnosis text is immutable in the manifest. Mapping uses only exact
entries in `configs/datasets/hiba_external_label_mapping.yaml` after Unicode
case-folding and whitespace normalization. Substring inference is forbidden.
Unknown official vocabulary is recorded as unresolved, excluded, and blocks
approval until a mapping decision is documented before inference.

Melanoma, basal cell carcinoma, and squamous cell carcinoma map to `melanoma`,
`bcc`, and `scc`. No benign diagnosis mapping is currently declared: the
official benign vocabulary remains unresolved until the acquired official
metadata inventory is inspected and a human freezes explicit compatible
entries. Actinic keratosis is excluded, consistently with the locked ISIC 2019
primary mapping. Ambiguous, unsupported, unknown, and non-diagnostic labels
are never forced into the four classes.

Every approved row must have exactly one of `non_malignant`, `melanoma`, `bcc`,
or `scc`, with consistent Stage-1 and Stage-2 labels.

## Required manifest columns

`dataset`, `release_id`, `image_id`, `image_path`, `patient_id`, `lesion_id`,
`original_diagnosis`, `canonical_diagnosis`, `modality`,
`mapped_final_label`, `mapped_stage_1_label`, `mapped_stage_2_label`,
`include_primary_evaluation`, `exclusion_reason`, `attribution`, `license`,
`source_reference`, `file_size_bytes`, and `file_sha256`.

## Integrity, leakage, and approval gates

The audit must:

1. validate required metadata columns and unique, non-empty image IDs;
2. resolve paths beneath `data/external/hiba/`, reject missing/empty files,
   and stream SHA-256 without modifying files;
3. detect identical hashes and conflicting labels within an identical hash or
   non-empty lesion ID;
4. count original/canonical diagnosis, mapped class, modality, licence,
   inclusion, and exclusion reason;
5. compare HIBA image IDs and SHA-256 values with
   `data/manifests/isic2019_dataset_manifest.csv`;
6. report all overlaps; any overlap blocks approval pending a separately
   documented resolution;
7. block approval for unresolved labels, metadata/licence/attribution defects,
   modality uncertainty, conflicts, or integrity failures.

The audit also reports approved-row support for all four mapped classes. Zero
approved rows blocks approval. Any absent class blocks evaluation approval
pending human feasibility review; no numeric minimum-support threshold is
invented.

Passing the automated audit is necessary but not sufficient: a human must
verify the official metadata, release, licence, attribution, modality
semantics, label support, integrity findings, and overlap disposition.
Therefore this protocol and the evaluation YAML remain pending until that
review is recorded. Audit approval does not itself authorize inference.
