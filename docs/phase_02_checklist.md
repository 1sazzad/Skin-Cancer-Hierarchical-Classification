# Phase 02 Completion Checklist

## Scope

Phase 02 covers the reproducible preprocessing pipeline, Stage 1 and Stage 2
manifest-driven datasets, dataloaders, class statistics, tests, and preparation
of baseline experiment configurations. Full model training is outside this
phase until the pipeline passes all checks.

## Environment

- [x] Active Python environment inspected
- [x] Runtime and development dependencies installed
- [x] Local CPU environment snapshot recorded
- [ ] Azure GPU environment audited separately before GPU training

## Preprocessing

- [x] 224 x 224 input policy implemented
- [x] ImageNet normalization implemented
- [x] Moderate train-only augmentation implemented
- [x] Deterministic validation and internal-test transforms implemented
- [x] Raw images remain unmodified

## Dataset and loaders

- [x] Frozen split manifest used as the single loader source
- [x] `split_included == 1` enforced
- [x] stage-specific inclusion flags enforced
- [x] Stage 1 label encoding implemented
- [x] Stage 2 label encoding implemented
- [x] train, validation, and internal-test dataloaders implemented
- [x] reproducible worker and sampler seeding implemented

## Audits and tests

- [x] class-statistics exporter implemented
- [x] transform tests implemented
- [x] dataset-filtering tests implemented
- [x] dataloader tests implemented
- [x] reproducibility tests implemented
- [x] complete pytest suite passes locally
- [x] real-data CPU smoke test passes
- [x] generated class statistics match Phase 01 locked counts

## Baseline preparation

- [x] Stage 1 baseline configuration prepared
- [x] Stage 2 baseline configuration prepared
- [x] ordinary cross-entropy retained as the clean baseline
- [x] weighted sampling and focal loss disabled for the clean baseline
- [x] EfficientNet-B0 model factory implemented
- [x] reusable classification epoch engine implemented
- [x] imbalance-sensitive primary metric exporter implemented
- [x] offline model and engine tests implemented
- [ ] one-batch real-data baseline forward/backward smoke test passes
- [ ] full training unlocked only after all Phase 02 checks pass

