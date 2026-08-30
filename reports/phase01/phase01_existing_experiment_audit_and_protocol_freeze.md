# Phase 01 — Existing Experiment Audit and Protocol Freeze

## A. Executive status

**Status: PASS.** Phase 01 inventoried the preserved ISIC 2019 experiments,
audited their provenance and selection rules, verified the locked split and the
Phase 11 paired analysis, classified reusable evidence, and froze the Phase 02
seven-backbone benchmark protocol. No training, fine-tuning, inference, GPU
work, checkpoint regeneration, split regeneration, or internal-test evaluation
was performed.

## B. Repository provenance

- Starting branch: `main`.
- Starting HEAD: `6fc5463dba9175643943de23bbdc15b348acfb65`.
- `origin/main`: `c6c8824eeefadeb006a000098df8cd5de78853ac`.
- Local `main` was two commits ahead: `700f987` (DenseNet paired statistics)
  and `6fc5463` (ICCIT manuscript finalization).
- Phase 01 branch: `phase01-existing-experiment-audit-protocol-freeze`, created
  directly from the starting HEAD.
- The initial worktree was clean. The six anticipated paper deletions were not
  present, so no files required restoration. No history was rewritten and no
  remote operation was performed.

Relevant implementation/evidence commits include `08b7604` (locked hierarchy
evaluation), `550e7cd` (selected flat internal-test evaluation),
`133442a185060d78f23019d6997b12526e6cef3c` (DenseNet implementation/run),
`700f987210ddcb20f1e77da25421da027daed501` (DenseNet paired statistics), and
`6fc5463dba9175643943de23bbdc15b348acfb65` (starting evidence state).

## C. Dataset freeze

The authoritative manifest is
`data/manifests/isic2019_train_val_test_split_seed42.csv`; the independent audit
records are `reports/dataset_audits/isic2019_split_audit_seed42.json` and
`reports/phase06/flat_four_class_label_audit.json`.

| Partition | non_malignant | melanoma | bcc | scc | Total |
|---|---:|---:|---:|---:|---:|
| Train | 11,193 | 3,164 | 2,327 | 440 | 17,124 |
| Validation | 2,398 | 678 | 498 | 94 | 3,668 |
| Test | 2,398 | 678 | 498 | 94 | 3,668 |
| Full flat cohort | 15,989 | 4,520 | 3,323 | 628 | 24,460 |

The split uses seed 42 and intended ratios 70/15/15. Shared non-empty lesion
IDs and shared exact file SHA-256 values form grouping relations; transitive
closure is applied. Missing lesion IDs use an exact-hash group when duplicated
and otherwise a singleton. Audit counts are zero for split-group overlap,
lesion-ID overlap, and exact-hash overlap. Four images in one MEL/NV
cross-diagnosis exact-duplicate component were excluded, as were 867 AK images
outside the locked Stage-1/flat scope. The manifest was audited, not regenerated.

The exact flat class order is `[non_malignant, melanoma, bcc, scc]`, mapped to
indices `[0, 1, 2, 3]`. ISIC 2019 supplies no patient ID, so patient-independent
splitting cannot be guaranteed.

## D. Experiment inventory

`—` means the metric is not applicable, not evaluated, or not preserved in a
directly comparable form. Hierarchical rows are not flat-backbone benchmark rows.

| Experiment | Architecture | Task | Loss | Resolution | Seed | Best Val Macro-F1 | Test Macro-F1 | Artifact status | Decision |
|---|---|---|---|---:|---:|---:|---:|---|---|
| Phase 06 flat clean CE | EfficientNet-B0 | flat 4-class | CE | 224 | 42 | 0.6535716654 | 0.6192224685 | Run metadata/history tracked; test evidence represented in Phase 07; checkpoint path/hash recorded but checkpoint absent here | REUSE |
| Phase 06B flat CB-focal | EfficientNet-B0 | flat 4-class | CB focal | 224 | 42 | 0.6490067298 | not accessed | Config/registry/hash and backup checksum retained; extracted run/checkpoint absent here | REFERENCE ONLY |
| Phase 11 flat clean CE | DenseNet121 | flat 4-class | CE | 224 | 42 | 0.6449820791 | 0.6351074532 | Run metadata/history and compact test evidence tracked; checkpoint path/hash recorded but checkpoint absent here | REUSE |
| Phase 03 Stage 1 | EfficientNet-B0 | non_malignant vs malignant | CE | 224 | 42 | 0.808693 | 0.774783 | Config, resolved config, history, validation/test metrics and predictions retained; checkpoint not in checkout | HISTORICAL BASELINE |
| Phase 03 Stage 2 clean | EfficientNet-B0 | malignant subtype | CE | 224 | 42 | 0.763902 | 0.712918 | Config, resolved config, history, validation/test metrics and predictions retained; checkpoint not in checkout | HISTORICAL BASELINE |
| Phase 04 Stage 2 weighted | EfficientNet-B0 | malignant subtype | weighted CE | 224 | 42 | 0.764999 | not selected for final locked comparison | Config, resolved config, history retained | HISTORICAL COMPARATOR |
| Phase 04 Stage 2 CB-focal | EfficientNet-B0 | malignant subtype | CB focal | 224 | 42 | 0.776307 | selected Stage-2 evidence retained | Config, resolved config, history, test metrics/predictions retained | HISTORICAL COMPARATOR |
| Phase 05 predicted-gate | two EfficientNet-B0 stages | end-to-end hierarchy | CE gate + CB-focal subtype | 224 | 42 | — | 0.6053674006 | Locked metrics, predictions, matrices, protocol and checkpoint hashes retained | HISTORICAL COMPARATOR |
| Phase 05 oracle routing | two EfficientNet-B0 stages | oracle diagnostic | same frozen stages | 224 | 42 | — | 0.7936564676 | Derived from locked hierarchy predictions | DIAGNOSTIC ONLY |
| Phase 07 paired analysis | flat EfficientNet-B0 vs predicted gate | stored-prediction statistics | — | — | 42 | — | flat-minus-hierarchy +0.0138550680 | Protocol, pairing audits, generated statistics and reports retained | HISTORICAL COMPARATOR |
| Phase 11 paired analysis | DenseNet121 vs EfficientNet-B0 and predicted gate | stored-prediction statistics | — | — | 42 | — | see Section F | Code, tests, CSV/JSON and report retained | REUSE |

## E. Artifact traceability for reusable runs

### EfficientNet-B0 clean CE

- Config: `configs/experiments/phase06_flat_four_class_isic2019_efficientnet_b0_cross_entropy.yaml`
- Resolved config/history/best validation metrics: `runs/phase06_full/full__phase06_flat_four_class_isic2019_efficientnet_b0_cross_entropy_seed42__20260726T232308Z/`
- Checkpoint path: the same run directory, `best_checkpoint.pt` (not present in this checkout)
- Checkpoint SHA-256: `f3d8b8b0e5ef42e3c287a2377b5570411d442246acd16cb874ccf903facdc7a7`
- Test metrics/report: Phase 06C registry entry and
  `reports/phase06/phase06c_selected_flat_internal_test_result.md`
- Preserved paired predictions: `reports/phase07/generated/paired_prediction_manifest.csv`
  (the canonical Phase 06C extracted prediction path is absent; a verified
  archive/checksum is referenced by the Phase 07 evidence)
- Confusion matrix/statistics: `reports/phase07/generated/` and
  `reports/phase07/phase07_statistical_analysis_results.md`
- Registry: `phase06_flat_four_class_clean_ce_seed42` and
  `phase06c_selected_flat_internal_test_seed42`
- Evaluation implementation commit: `550e7cdb1144f059c940d4240fe4579e0280a803`

### DenseNet121 clean CE

- Config: `configs/experiments/phase11_flat_four_class_isic2019_densenet121_cross_entropy.yaml`
- Resolved config/history/best validation metrics: `runs/phase11/full__phase11_flat_four_class_isic2019_densenet121_cross_entropy_seed42__20260730T155147Z/`
- Checkpoint path: the same run directory, `best_checkpoint.pt` (not present in this checkout)
- Checkpoint SHA-256: `97f50dd5fb6b8d5a65b1c08035f07bab2bb5683e5647e519527c9cb56afbaa01`
- Test metrics/predictions/matrix: `experiments/evaluations/phase11_densenet121_internal_test_seed42__best_epoch04/`
- Reports/statistics: `reports/phase11/phase11_final_densenet_baseline_result.md`,
  `reports/phase11/phase11_densenet121_paired_statistical_analysis.md`, and
  `reports/phase11/generated/`
- Registry: `phase11_flat_four_class_densenet121_seed42`
- Run implementation commit: `133442a185060d78f23019d6997b12526e6cef3c`
- Paired-statistics implementation commit: `700f987210ddcb20f1e77da25421da027daed501`

Reuse means that the preserved result is protocol-compatible and need not be
trained again for the Phase 02 evidence table. Actual checkpoint-dependent new
operations would first require restoring the verified checkpoint; Phase 01 did
not do so.

## F. Statistical evidence

Phase 07 paired 3,668 samples by identifier. Flat EfficientNet-B0 minus the
predicted-gate hierarchy had Macro-F1 difference `+0.01385506796`, paired 95%
CI `[-0.0142546488, 0.0419633760]`, and exact McNemar `p=0.8207415883`.
The interval crosses zero; no Macro-F1 superiority is established.

The Phase 11 implementation validates 3,668 unique matched IDs, rejects missing
or duplicate IDs, target mismatches, unsupported indices/labels and non-finite
probabilities, stable-sorts by sample ID, uses fixed labels with zero division
0, and runs a ground-truth-class-stratified paired percentile bootstrap with
10,000 replicates and seed 42. Linear unrounded float64 quantiles provide the
confidence limits. The exact two-sided McNemar calculation uses the discordant
correctness pairs through an exact binomial test.

DenseNet121 minus EfficientNet-B0 reproduces:

| Metric | Difference | Paired 95% CI |
|---|---:|---:|
| Accuracy | +0.0493456925 | [0.0351690294, 0.0632497274] |
| Balanced accuracy | -0.0334843128 | [-0.0635792038, -0.0037129085] |
| Macro-F1 | +0.0158849846 | [-0.0148333731, 0.0463207941] |
| Weighted-F1 | +0.0336056426 | [0.0198483321, 0.0470811848] |

McNemar counts are 475 DenseNet-only correct and 294 EfficientNet-only correct
(`p=7.007954295e-11`). DenseNet has higher paired accuracy and weighted-F1;
EfficientNet has higher balanced accuracy. DenseNet Macro-F1 superiority is not
established because its paired CI crosses zero.

Against the predicted-gate hierarchy, DenseNet differences are accuracy
`+0.0512540894` (CI `[0.0370774264, 0.0654307525]`), balanced accuracy
`-0.0143706973` (CI `[-0.0448635404, 0.0168743400]`), Macro-F1
`+0.0297400526` (CI `[0.0003352759, 0.0588147619]`), and weighted-F1
`+0.0358307946` (CI `[0.0221293000, 0.0494743004]`); McNemar uses 472 versus
284 discordant wins (`p=8.153500949e-12`).

The statement in `phase11_final_densenet_baseline_result.md` that no DenseNet
paired test existed was correct when that earlier evidence was written, but it
was superseded by commit `700f987`. The still-generated
`final_model_comparison.csv` likewise contains stale wording, “Descriptive
locked point estimate; no new paired hypothesis test.” Both are preserved as
chronological evidence and are not silently rewritten.

## G. Test-set leakage and model-selection audit

No evidence was found that internal-test results selected flat CE versus focal:
CE won on validation Macro-F1 (`0.6535716654` vs `0.6490067298`), and the
rejected flat focal run records `internal test accessed=false`. Checkpoints are
selected by validation Macro-F1 and early stopping observes validation only.
Stage-2 CB-focal was selected over clean and weighted CE using validation before
its one-time test evaluation. The hierarchy used already-frozen stage
checkpoints; oracle routing is explicitly diagnostic, not a selectable system.
DenseNet was an approved protocol-matched comparator and its epoch 4 checkpoint
was validation-selected before its locked evaluation; no evidence of test-based
checkpoint or post-test variant selection was found.

These legitimate one-time post-selection evaluations do not authorize new
selection from their test values. In particular, DenseNet and EfficientNet test
results must not choose the Phase 02 winner.

## H. Reuse matrix

| Evidence/candidate | Classification | Reason |
|---|---|---|
| EfficientNet-B0 flat CE | REUSE | Protocol-compatible preserved validation and locked-test evidence |
| DenseNet121 flat CE | REUSE | Same frozen protocol except backbone; preserved validation and locked-test evidence |
| EfficientNet-B0 flat CB-focal | REFERENCE ONLY | Validation-rejected; test remains untouched |
| Stage 1 and Stage 2 clean CE | HISTORICAL BASELINE | Historical hierarchy components, not flat candidates |
| Stage 2 weighted CE and CB-focal | HISTORICAL COMPARATOR | Historical imbalance variants, not flat candidates |
| Predicted-gate hierarchy | HISTORICAL COMPARATOR | End-to-end historical comparison, not a flat backbone row |
| Oracle-routing hierarchy | DIAGNOSTIC ONLY | Uses ground-truth routing and is not deployable/selectable |
| Phase 07 paired statistics | HISTORICAL COMPARATOR | Locked flat-vs-hierarchy comparison |
| Phase 11 paired statistics | REUSE | Verified stored-prediction evidence |
| DenseNet169 | RERUN REQUIRED | No protocol-compatible preserved run; this is a new run |
| ResNet50 | RERUN REQUIRED | No protocol-compatible preserved run; this is a new run |
| MobileNetV3-Large | RERUN REQUIRED | No protocol-compatible preserved run; this is a new run |
| EfficientNet-B2 | RERUN REQUIRED | No protocol-compatible preserved run; this is a new run |
| EfficientNet-B3 | RERUN REQUIRED | No protocol-compatible preserved run; this is a new run |

## I. Frozen Phase 02 protocol

Phase 02 uses ISIC 2019, the existing seed-42 manifest, the flat four-class
task, and the exact class order `[non_malignant, melanoma, bcc, scc]`.
ImageNet-pretrained models use 224×224 input and dropout 0.2. Training uses
ordinary cross-entropy, no class weights, no weighted sampler, batch size 64,
and AMP when running on CUDA. Optimization is AdamW (`lr=3e-4`, weight decay
`1e-4`) with `CosineAnnealingLR`, `T_max` equal to the configured 30-epoch
budget and `eta_min=1e-6`. Maximum epochs are 30, early-stopping patience is 7,
and checkpoint selection is validation Macro-F1. Seed is 42.

The exact training transform is: `ToImage`; bilinear antialiased
`RandomResizedCrop(224×224, scale=[0.85,1.0], ratio=[0.90,1.10])`; horizontal
flip 0.5; vertical flip 0.5; bilinear random rotation ±15°; `ColorJitter` with
brightness/contrast/saturation 0.10 and hue 0.02; conversion to float32 with
unit scaling; ImageNet normalization mean `[0.485,0.456,0.406]`, standard
deviation `[0.229,0.224,0.225]`. Validation/test use bilinear antialiased resize
to 256 (shorter-edge torchvision integer-resize behavior), center crop 224×224,
the same float32 scaling, and the same normalization.

Primary ranking is **Validation Macro-F1 only**. Secondary reporting is balanced
accuracy, macro precision, macro recall, weighted-F1, accuracy, per-class
precision/recall/F1/support, and confusion matrix. Fixed labels and
`zero_division=0` apply; absent classes are retained. Reports must emphasize SCC
and malignant-to-non_malignant errors.

The internal test must not be used for architecture, backbone, hyperparameter,
augmentation, loss, resolution, ensemble-weight, checkpoint, early-stopping,
or threshold selection. Development uses training and validation only. The
locked internal test is evaluation-only after configuration selection. Existing
test evidence remains preserved but cannot influence Phase 02 ranking.

The machine-readable companion is
`configs/protocols/phase02_flat_four_class_backbone_benchmark.yaml`. It is a
protocol record, not a runnable multi-backbone training configuration.

## J. Phase 02 execution list

- Reuse preserved runs: DenseNet121 and EfficientNet-B0.
- New controlled runs required: DenseNet169, ResNet50, MobileNetV3-Large,
  EfficientNet-B2, and EfficientNet-B3.
- Current model factory supports only EfficientNet-B0 and DenseNet121. Adding
  the other five architectures is a Phase 02 preparation task, not Phase 01 work.

## K. Known limitations

- All reusable results use one seed (42).
- Evaluation is on one internal dataset/cohort; external generalization is not established.
- SCC support is rare (440 train, 94 validation, 94 test), limiting precision.
- Patient-independent splitting cannot be guaranteed because patient IDs are absent.
- The internal test cohort has already been consumed by historical locked evaluations.
- Historical test results cannot be used for new architecture selection.
- Several checkpoint/archive paths are provenance references rather than files present in this checkout.
- Macro precision and macro recall are prospective additions; historical JSON was not regenerated.
