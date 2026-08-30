# Phase 04 Final Internal-Test Preflight

Date: 2026-08-28 (Asia/Dhaka)

## Readiness

- Gate04D: **PASS**
- Gate04E: **PASS**
- Protocol: **FROZEN**
- Internal test executed: **NO**
- Final runner: **READY**
- Final config: **READY**

This preflight performed static inspection, configuration comparison, unit tests, compile checks, and recomputation from existing validation predictions only. It did not construct or access an internal-test dataset, run model inference, train, tune thresholds, or select checkpoints.

## Evidence verification

Gate04D records `execution_split=validation`, `internal_test_executed=false`, 3,668 unique paired sample IDs, identical shared/flat order and ground truth, and Tesla T4 execution. Recorded sample counts are 3,668 flat, 3,668 Task 1, 1,270 malignant-subset Task 2, and 127 Task 3.

Independent recomputation from `paired_validation_predictions.csv` reproduced:

- Flat Macro-F1: `0.6535716653605627`; accuracy: `0.7729007633587787`
- Shared predicted-gate Macro-F1: `0.5832871340331853`; accuracy: `0.7091057797164667`
- McNemar discordant counts: hierarchy-only correct `235`; flat-only correct `469`

Gate04E records 3,668 validation samples, `internal_test_executed=false`, Macro-F1 delta `-0.07028453132737733` with paired-bootstrap 95% CI `[-0.10189398304158802, -0.03864787451662513]`, accuracy delta `-0.06379498364231195` with 95% CI `[-0.07797164667393675, -0.04961832061068705]`, and exact two-sided McNemar p-value `7.723251233116162e-19`.

The protocol freeze is `FROZEN`, prohibits further model selection, hyperparameter tuning, threshold tuning, checkpoint reselection, and preprocessing changes, and records that internal test was not executed before freeze.

## Frozen manifests

- `data/manifests/isic2019_train_val_test_split_seed42.csv`
- `data/manifests/emb_stage03_dermoscopic_split_seed42.csv`

## Frozen checkpoints

- Shared: `runs/phase03_shared_three_task/seed_42/best_checkpoint.pt`; SHA-256 `2f1c2393c5c9de15dfa4a1a132a31b9a5b8ede07d7ed6e07ab90918fc2aaa9eb`; epoch 6
- Task 1: `runs/phase03_full/full__stage01_isic2019_efficientnet_b0_cross_entropy_seed42__20260724T190600Z/best_checkpoint.pt`; SHA-256 `95e02c26b1ea4a0dba17016313c81f97c9c2635270a37b4debbee0f84e07ba3b`; epoch 5
- Task 2: `runs/phase04_cb_focal_full/full__stage02_isic2019_efficientnet_b0_class_balanced_focal_loss_seed42__20260726T064808Z/best_checkpoint.pt`; SHA-256 `10986d41b64a685fcd8fe166623c5b1c7fd2f21bdad7cf4d55dedc3967a397fd`; epoch 8
- Task 3: `experiments/runs/full__stage03_isic_derived_dermoscopic_efficientnet_b0_weighted_cross_entropy_seed42__20260729T204920Z/best_checkpoint.pt`; SHA-256 `71bfda5f7a19333947e1c13f4e1c5ed45e9a827c447fc9bcd6fd9ddc999f8692`; epoch 12
- Flat: `runs/phase06_full/full__phase06_flat_four_class_isic2019_efficientnet_b0_cross_entropy_seed42__20260726T232308Z/best_checkpoint.pt`; SHA-256 `f3d8b8b0e5ef42e3c287a2377b5570411d442246acd16cb874ccf903facdc7a7`; epoch 2

The final runner verifies every checkpoint SHA-256, epoch, model kind, and class order before inference. Shared and Task 3 checkpoints are intentionally not present in the recovered local code workspace; they must remain available at the frozen paths on the VM. This does not affect the static local preflight.

## Static and local tests

- `python -m pytest -q tests/test_phase04_comparative_harness.py tests/test_phase04_final_internal_test_preflight.py`: **13 passed**
- All `tests/test_phase04*.py`: **28 passed**
- `py_compile` on both runners, the preflight script, comparative harness, adapter, and relevant tests: **PASS**
- Final/validation YAML semantic comparison: only `execution_split` and `internal_test_execution_allowed` differ
- Paired validation CSV recomputation: **PASS**

The final-runner tests use configuration parsing and AST/static inspection. They do not call `main`, construct datasets, load checkpoints, or run inference.

## One-time VM execution

Synchronize the committed Phase 04 preparation files, verify the five frozen checkpoints exist at the paths above, then run exactly once from the VM repository root:

```bash
python scripts/run_phase04_final_internal_test.py --config configs/evaluation/phase04_controlled_comparative_internal_test.yaml --device cuda
```

Expected output artifacts:

- `reports/phase04_controlled_comparative/final_internal_test/final_internal_test_summary.json`
- `reports/phase04_controlled_comparative/final_internal_test/paired_internal_test_predictions.csv`

The summary records `internal_test_executed=true` only after collection, paired-integrity checks, schema validation, and efficiency benchmarking all complete successfully.

## Unresolved issues

None for local preparation. Before the one-time VM command, confirm the Tesla T4 environment and all frozen manifest/checkpoint paths are present. Do not rerun after a successful execution.
