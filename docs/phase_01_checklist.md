# Phase 01 — ISIC 2019 Data Foundation Checklist

## Acquisition and Integrity

- [x] Official training image archive downloaded
- [x] Official ground-truth CSV downloaded
- [x] Official metadata CSV downloaded
- [x] ZIP structure validated
- [x] 25,331 images confirmed
- [x] 25,331 ground-truth rows confirmed
- [x] 25,331 metadata rows confirmed
- [x] Image-label-metadata identifiers matched
- [x] Source SHA-256 checksums generated
- [x] All images decode successfully

## Manifest and Label Audit

- [x] Dataset manifest generated
- [x] Original class distribution recorded
- [x] Stage 1 mapping generated
- [x] Stage 2 mapping generated
- [x] AK exclusion policy recorded
- [x] Metadata missingness recorded
- [x] Lesion-ID availability recorded

## Leakage and Split Audit

- [x] Exact duplicate images audited
- [x] Cross-diagnosis duplicate component identified
- [x] Four conflicting images excluded
- [x] Lesion/hash connected components generated
- [x] Deterministic seed-42 split generated
- [x] Group overlap validation passed
- [x] Lesion-ID overlap validation passed
- [x] Exact-hash overlap validation passed

## Final State

- Phase 01 status: Complete
- Model training status: Not started
- Next phase: Preprocessing, dataloaders, and baseline experiment preparation
