# Gate 06C — Shared Multi-Backbone Training Closure

## Verdict

**PASS.**

The frozen Phase 06 shared three-task training campaign is complete. Six new shared backbones were trained under the frozen protocol and the existing EfficientNet-B0 shared checkpoint was reused without retraining.

No new run evaluated the internal-test split during training or validation selection.

## Frozen model family

All shared models use one encoder and three task heads:

- Task 1: non-malignant vs malignant
- Task 2: melanoma vs BCC vs SCC
- Task 3: Tis vs T1 vs T2 vs T3 vs T4 auxiliary task

Checkpoint selection used the predeclared arithmetic mean of Task-1, Task-2, and Task-3 validation macro-F1.

## Completed shared runs

| Backbone | Best epoch | Task-1 val macro-F1 | Task-2 val macro-F1 | Task-3 val macro-F1 | Shared validation score | Training status |
|---|---:|---:|---:|---:|---:|---|
| EfficientNet-B0 | 6 | 0.762418 | 0.698750 | 0.436684 | 0.632618 | reused frozen Phase 03 checkpoint |
| DenseNet121 | 13 | 0.776769 | 0.687256 | 0.461740 | 0.641922 | completed early stopping |
| DenseNet169 | 7 | 0.797087 | 0.642164 | 0.350027 | 0.596426 | completed early stopping |
| ResNet50 | 13 | 0.768655 | 0.681232 | 0.412977 | 0.620955 | completed early stopping |
| MobileNetV3-Large | 10 | 0.803726 | 0.711111 | 0.428857 | 0.647898 | completed early stopping |
| EfficientNet-B2 | 4 | 0.790577 | 0.643516 | 0.451439 | 0.628510 | completed early stopping |
| EfficientNet-B3 | 19 | 0.790003 | 0.764184 | 0.458769 | **0.670986** | completed early stopping |

Validation-only ranking placed EfficientNet-B3 first, followed by MobileNetV3-Large and DenseNet121. This ranking was frozen before Gate 06D internal-test execution.

## New checkpoint provenance

| Backbone | Checkpoint SHA-256 |
|---|---|
| DenseNet121 | `fc12adcea9494809b389a16cc1b795e0ea6be443613462ccaada7de7cae0ea13` |
| DenseNet169 | `f4fd80e72b720e54a1d6f63a71a01454f61c3eede03c2d3cde74591cdecd0300` |
| ResNet50 | `8a7923b658b8793cd08b483a91a7fcb8bc28b63bcc95f3d5190b511acf9661b0` |
| MobileNetV3-Large | `da439f8d1093bc06654b50e5fd3e198e7662d6321a16dcc7577b5e06ffa05ed2` |
| EfficientNet-B2 | `21b13d567f0f81c8db779881ad67569f1b7351fed06c362073a2d4e2d631febd` |
| EfficientNet-B3 | `e178526f54b1e1bc8a5da6d728c7199eab302bc412855ba1a59d62dbb41e0cd8` |

Existing EfficientNet-B0 shared checkpoint SHA-256:

`2f1c2393c5c9de15dfa4a1a132a31b9a5b8ede07d7ed6e07ab90918fc2aaa9eb`

All six new runs recorded `internal_test_evaluated=false` and were produced from implementation commit `f43e7d9309e101e0a8ceda9b3d6df5630ee29ad8`.

## Verification

On the Tesla T4 VM:

- Phase 06 shared-backbone tests: 10 passed
- Combined Phase 03 + Phase 06 shared tests: 31 passed
- architecture preflight passed before every full new run
- fixed train/validation counts were preserved
- no internal-test loader was constructed by the training launcher

## Scientific selection statement

Gate 06C is a validation-only model-development stage. No internal-test result was used to choose, retrain, tune, reject, or replace a shared backbone.

## Gate conclusion

**Gate 06C: PASS / CLOSED.**

All seven shared backbone checkpoints required for the controlled comparative campaign are available. Gate 06D may therefore execute the prospectively frozen one-time matched internal-test comparison across all seven backbone pairs.