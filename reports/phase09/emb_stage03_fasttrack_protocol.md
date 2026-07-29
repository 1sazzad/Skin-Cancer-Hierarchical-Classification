# ISIC-Derived Melanoma T-Category Fast-Track Protocol

## Emergency ICCIT scope

This is a standalone five-class melanoma severity experiment, not the future
shared three-task model. The paper-facing dataset name is **ISIC-derived
melanoma T-category subset**.

## Source and licensing decision

The EMB repository at commit `3ec674f43e73cb08682b99b7fb996aca5f8040d8`
exposed no identifiable licence. Full EMB/Atlas acquisition was therefore
rejected and Dermoscopy Atlas is excluded. The EMB CSV is used only as an index
of candidate public ISIC identifiers. Its `stage_ajcc` value is retained solely
for agreement auditing and is never an authoritative training label.

Every candidate identifier is independently revalidated through the official
ISIC Archive API. Official API responses supply public status, full image URL,
per-image licence, attribution, modality, diagnosis, diagnosis-confirmation
method, Breslow thickness, ulceration, patient ID, and lesion ID. Unknown,
missing, or unsupported licences are excluded. Accepted licences are CC-0,
CC0, CC-BY, and CC-BY-NC; the exact per-image value and attribution are
retained.

## Candidate and eligibility gates

Only EMB-index rows with `source=ISIC`, `type=dermoscopic`, and a syntactically
valid ISIC identifier become candidates. Eligibility requires a successful
matching API response, public visibility, official dermoscopic modality,
histopathology confirmation, supported licence, recorded attribution, a full
image URL, and an independently derivable T-category.

Metadata access, downloads, audit, split generation, training, inference, and
evaluation are Azure GPU VM-only. Raw API responses, inventories, and images
remain ignored under `data/raw/emb/`.

## Official labels

The authoritative fields are `derived_stage_ajcc` and `t_category`. Official
`diagnosis_3` containing “melanoma in situ” maps to `0/Tis`. Official invasive
melanoma requires positive numeric `mel_thick_mm`: `(0,1] -> 1/T1`,
`(1,2] -> 2/T2`, `(2,4] -> 3/T3`, and `>4 -> 4/T4`. Melanoma NOS,
non-melanoma, missing/invalid thickness, and contradictory metadata are
rejected. Ulceration is retained for reporting but does not alter these broad
categories. T3 and especially T4 support is expected to be small.

## Audit and overlap policy

The metadata audit records operational results, exclusions, licences,
attributions, official modality/diagnosis/T-category distributions, grouping
coverage, and original-versus-official disagreement. It reports ISIC 2019 ID
overlap overall and by official T-category. No overlap is removed yet; the
overlap policy remains pending review of the VM metadata audit. Image audit
later adds readability, SHA-256 duplicates/conflicts, and ISIC 2019 hash
overlap.

## Split and model protocol

After the overlap decision, the seed-42 dermoscopic split is 70% train, 15%
validation, and 15% untouched test, stratified by official T-category. Complete
connected components are built from every available non-empty patient ID,
non-empty lesion ID, and exact SHA-256 relation, including their transitive
closure. Components receive deterministic identifiers and are assigned intact
using five-class image-count vectors while targeting the requested per-class
and overall ratios. A patient may have several distinct melanoma lesions with
different T-categories; this makes a patient-safe component multi-label, not
invalid, and patient grouping is retained to prevent leakage. Conflicting
T-categories within one lesion ID or one exact image hash remain fatal. Actual
image ratios are reported because unequal component sizes can prevent exact
70/15/15 allocation. Patient/lesion metadata is incomplete, but every known
relation is preserved; blank identifiers never connect images. Patient,
lesion, identical hash, and connected-component groups cannot cross
partitions. Optional class weights derive from training only.

EfficientNet-B0 uses ImageNet initialization, five logits ordered
`[Tis, T1, T2, T3, T4]`, cross entropy, AdamW, CUDA mixed precision, at most
30 epochs, early stopping, and seed 42. Validation macro-F1 alone selects the
checkpoint. The frozen test split is used only after selection.

## Locked baseline and single imbalance-aware candidate

The frozen cross-entropy baseline selected epoch 2 with validation macro-F1
`0.365615`. On its one locked internal-test evaluation (127 images), accuracy
was `0.6062992125984252`, balanced accuracy `0.20240259740259742`, macro-F1
`0.16283767911674887`, and weighted-F1 `0.48009115139677305`. Test F1 was
`0.7676767676767676` for Tis, `0.046511627906976744` for T1, and zero for T2,
T3, and T4. Predictions were Tis 121, T1 3, T3 3, T2 0, and T4 0, indicating
strong majority-class collapse rather than adequate five-class discrimination.

Exactly one imbalance-aware candidate is permitted: train-only
inverse-frequency weighted cross-entropy, with fixed train counts Tis 355, T1
184, T2 33, T3 10, and T4 12. The corresponding weights, normalized to sum to
five, are Tis `0.063475735584673`, T1 `0.12246677245955931`, T2
`0.682845034319967`, T3 `2.253388613255891`, and T4
`1.8778238443799093`. T3 and T4 support remains especially limited.

The candidate may be selected only if its best validation macro-F1 is strictly
greater than `0.365615`. The internal test remains untouched until that
validation-only decision; test artifacts cannot guide tuning, candidate
changes, or checkpoint selection.
