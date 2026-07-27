# Phase 06B Class-Balanced Focal Validation Result

## Objective and scope

Phase 06B tested whether changing only the loss in the flat four-class
EfficientNet-B0 improved the predeclared validation selection metric. It used
the locked ISIC 2019 seed-42 manifest, the `phase06_flat_four_class_v1` mapping,
and class order `[non_malignant, melanoma, bcc, scc]`. The model was
EfficientNet-B0 with a four-logit head. The loss was class-balanced focal loss
with effective-number beta `0.9999` and gamma `2.0`.

The full experiment ran on the Azure Tesla T4 VM with Python `3.12.3`,
PyTorch `2.13.0+cu130`, CUDA available, and GPU `Tesla T4`. Before training,
all `82` VM tests passed in `18.32` seconds with exit code `0`. The Phase 06
label audit also exited `0`; a regenerated path-only JSON difference was
restored, leaving the repository clean.

## Non-reportable sanity run

The CUDA sanity run was:

`runs/phase06b/sanity/sanity__phase06b_flat_four_class_isic2019_efficientnet_b0_class_balanced_focal_loss_seed42__20260727T120357Z`

It completed one training batch, one validation batch, and one epoch with
training loss `0.301803`, validation loss `0.162430`, and validation macro-F1
`0.1263157895`. It is explicitly marked `sanity_run: true` and
`reportable_as_full_result: false`; internal-test metrics are absent. A
scikit-learn warning about predicted classes absent from `y_true` was expected
because this single validation batch contained no BCC or SCC examples. These
numbers are not a scientific experiment result.

## Full training

- Run: `runs/phase06b/full/full__phase06b_flat_four_class_isic2019_efficientnet_b0_class_balanced_focal_loss_seed42__20260727T120615Z`
- Control directory: `runs/phase06b/full_control/20260727T120611Z`
- Exit code: `0`
- Configured/completed epochs: `30` / `18`
- Early stopping: `true`, patience `7`
- Best epoch: `11`
- Total training time: `1692.106839308` seconds
- Selection metric: validation macro-F1
- `sanity_run: false`
- `reportable_as_full_result: true`
- `internal_test_accessed: false`

## Best validation metrics

The best checkpoint was evaluated on all `3668` validation samples.

| Metric | Value |
|---|---:|
| Accuracy | 0.7570883315 |
| Balanced accuracy | 0.6793461931 |
| Macro-F1 | 0.6490067298 |
| Weighted F1 | 0.7671360749 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| non_malignant | 0.9222560976 | 0.7568807339 | 0.8314246450 | 2398 |
| melanoma | 0.5311890838 | 0.8038348083 | 0.6396713615 | 678 |
| bcc | 0.6375838926 | 0.7630522088 | 0.6946983547 | 498 |
| scc | 0.4743589744 | 0.3936170213 | 0.4302325581 | 94 |

## Validation-only model selection

The predeclared policy selects the highest validation macro-F1, uses
validation balanced accuracy only as a tie-breaker, and prefers the simpler
clean cross-entropy model if both metrics are exactly tied.

| Candidate | Accuracy | Balanced accuracy | Macro-F1 | Weighted F1 | SCC F1 |
|---|---:|---:|---:|---:|---:|
| Phase 06A clean CE | 0.7729007634 | 0.6796986150 | 0.6535716654 | 0.7806549873 | 0.3870967742 |
| Phase 06B focal | 0.7570883315 | 0.6793461931 | 0.6490067298 | 0.7671360749 | 0.4302325581 |

Focal minus clean-CE macro-F1 was `-0.0045649356`. Therefore Phase 06A clean
cross-entropy is the validation-selected flat model. No tie-break was required.
Phase 06B is a rejected candidate under the predeclared primary metric, not a
failed experiment.

The secondary SCC F1 improvement was `0.0431357839` (`0.3870967742` to
`0.4302325581`). This is a meaningful minority-class observation, but it does
not override validation macro-F1. No statistical-significance claim is made.

## Frozen identities and backup

The selected checkpoint frozen before internal-test access is:

`runs/phase06_full/full__phase06_flat_four_class_isic2019_efficientnet_b0_cross_entropy_seed42__20260726T232308Z/best_checkpoint.pt`

SHA-256:
`f3d8b8b0e5ef42e3c287a2377b5570411d442246acd16cb874ccf903facdc7a7`

The rejected focal checkpoint is preserved in the verified archive and was
recorded at:

`runs/phase06b/full/full__phase06b_flat_four_class_isic2019_efficientnet_b0_class_balanced_focal_loss_seed42__20260727T120615Z/best_checkpoint.pt`

SHA-256:
`07586d515cd9378e05831ca542f391e32b3b7a6c669c7dd83ce1df219b2af015`

The verified local backup is
`runs/backups/phase06b/phase06b_cb_focal_training_backup_verified_20260727T123825Z.tar.gz`,
SHA-256
`bbaf3f385a2acf2c028243a1e1f73a2b0b5e0914e46bb803ce0e79b4de909f8b`.
Archive transfer and extraction exited `0`; the archive and extracted focal
checkpoint hashes matched, no required file was missing, final training status
was `0`, and `internal_test_accessed=false`.

## Lock and limitations

Selection used validation results only. The internal test remained untouched,
and no internal-test metric was used for model selection. Only the selected
Phase 06A clean-CE checkpoint is eligible for the Phase 06C one-time
internal-test evaluation; the Phase 06B candidate is prohibited.

This single-seed internal-dataset validation comparison does not establish
statistical significance, clinical readiness, fairness, external
generalisation, state-of-the-art performance, or superiority of the flat model
over the hierarchy. No post-test tuning or candidate switching is permitted.
