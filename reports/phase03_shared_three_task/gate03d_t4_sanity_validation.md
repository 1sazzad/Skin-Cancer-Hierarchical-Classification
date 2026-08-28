# Gate 03D — T4 Sanity Validation

## Verdict

**PASS**

Gate 03D verified that the shared three-task baseline is executable end-to-end on the authorized Tesla T4 environment using real image payloads.

No full training was performed and the internal test set remained closed.

## Environment

- GPU: NVIDIA Tesla T4
- PyTorch: 2.13.0+cu130
- Input resolution: 224 x 224
- Training batch size: 64
- Branch: phase03-shared-three-task-hierarchical-baseline
- Baseline implementation commit: e036c352c62f2c0d7337ae7afe414fac2641f8b3

## Dataset payload verification

The Phase 03 worktree uses the existing VM dataset payload through local symlinks.

Verified source payloads:

- ISIC 2019 image payload: 25,331 JPG images
- ISIC-derived melanoma T-category payload: 848 JPG images

The repository manifests themselves were not modified.

## Image-backed loader verification

Verified cohort sizes:

- Combined training pool: 17,718
- Task 1 validation: 3,668
- Task 2 malignant-only validation: 1,270
- Task 3 validation: 127

Real image-backed batches loaded successfully.

Observed tensor shapes:

- image: [64, 3, 224, 224]
- targets: [64, 3]
- task_mask: [64, 3]

## GPU forward smoke test

A real 64-image batch was transferred to the Tesla T4 and passed through the shared model.

Output shapes:

- Task 1: [64, 2]
- Task 2: [64, 3]
- Task 3: [64, 5]

Model parameters:

- Total parameters: 4,020,358
- Trainable parameters: 4,020,358

Forward smoke test verdict: PASS.

The observed GPU memory allocation from this smoke test is not treated as a final efficiency benchmark.

## One-step optimization smoke test

A real naturally sampled training batch was used.

Active samples:

- Task 1: 63
- Task 2: 21
- Task 3: 1

Observed losses:

- Total loss: 0.9492905139923096
- Task 1 loss: 0.7283908724784851
- Task 2 loss: 0.3659829795360565
- Task 3 loss: 1.75349760055542

Gradient norms:

- Shared encoder: 4.12223986607194
- Task 1 head: 0.6318535503995649
- Task 2 head: 0.9704609636264316
- Task 3 head: 4.696210045756092

The shared encoder parameter changed after the single optimizer step:

- maximum observed parameter delta: 0.0003001689910888672

This verifies that all active task losses propagate gradients through their heads into the shared encoder.

The observed peak GPU allocation during this one-step smoke test is not treated as a final efficiency benchmark.

## Validation-pipeline smoke test

The full validation-only pipeline completed successfully for:

- Task 1 validation
- Task 2 malignant-only validation
- Task 3 validation

The shared validation score was also computed successfully as the arithmetic mean of the three task Macro-F1 values.

The untrained smoke-test metrics were:

- Task 1 Macro-F1: 0.3067031308
- Task 2 Macro-F1: 0.2389474094
- Task 3 Macro-F1: 0.1907159177
- Shared validation score: 0.2454554860

**These values are NOT experimental performance results.**

They were produced from a freshly initialized shared baseline before training and exist only as execution-pipeline evidence. They must not be used for model comparison, paper performance tables, or scientific conclusions.

## Safety / protocol confirmation

During Gate 03D:

- no full training was performed
- no training epoch was completed
- only one optimizer step was used as a gradient smoke test
- no checkpoint was selected
- no hyperparameter tuning was performed
- no internal-test loader was constructed
- no internal-test predictions were generated
- no internal-test metrics were calculated
- no historical locked evidence was modified

## Gate conclusion

Gate 03D: **PASS**

The shared three-task baseline has passed:

- real-image loading
- real-batch collation
- T4 execution
- three-head forward propagation
- masked multitask loss computation
- shared-encoder gradient propagation
- one-step optimizer execution
- validation-only metric pipeline execution

The implementation is therefore technically ready for Gate 03E full shared-model training under the already frozen protocol.
