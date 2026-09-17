# Gate 06B — Multi-Backbone Shared Implementation

## Verdict

**PASS, with mandatory Gate 06C preflight before full GPU training.**

The shared three-task model is no longer EfficientNet-B0-only. The implementation now supports all seven frozen backbone identifiers while preserving the existing one-encoder / three-head task structure and the Phase 03 scientific protocol.

## Supported shared backbones

- efficientnet_b0
- densenet121
- densenet169
- resnet50
- mobilenet_v3_large
- efficientnet_b2
- efficientnet_b3

## Implementation changes

### `src/models/shared_three_task.py`

Added:

- `SUPPORTED_SHARED_ARCHITECTURES`
- generic `SharedThreeTaskModel`
- generic `build_shared_three_task_model(architecture, ...)`
- architecture-specific encoder extraction for DenseNet121, DenseNet169, ResNet50, MobileNetV3-Large, EfficientNet-B0, EfficientNet-B2, and EfficientNet-B3
- DenseNet post-feature ReLU before global average pooling to match torchvision DenseNet forward semantics

Preserved:

- exactly one shared encoder pass
- Task 1 head = Dropout(0.2) -> Linear(..., 2)
- Task 2 head = Dropout(0.2) -> Linear(..., 3)
- Task 3 head = Dropout(0.2) -> Linear(..., 5)
- unchanged task label mappings
- legacy `SharedThreeTaskEfficientNetB0`
- legacy `build_shared_three_task_efficientnet_b0(...)`

The EfficientNet-B0 compatibility wrapper intentionally retains the same registered module names (`encoder`, `pool`, `task1_head`, `task2_head`, `task3_head`), so the frozen Phase 03 checkpoint state-dict key structure remains compatible.

### `src/models/__init__.py`

Exported the new shared architecture registry and generic builder while retaining existing builders.

### `scripts/train_phase06_shared_three_task.py`

Added a Phase 06 launcher that reuses the already validated Phase 03 training loop rather than duplicating scientific logic.

The launcher permits exactly one experimental variable:

`model architecture`

For all other fields it reuses the strict Phase 03 frozen-protocol validator. It then writes a resolved run config containing the selected architecture and invokes the unchanged masked losses, optimizer, scheduler, validation metrics, checkpoint selection, and early-stopping logic.

No internal-test loader or test-evaluation path is introduced.

### `tests/test_phase06_shared_backbone_support.py`

Added tests that require every approved architecture to:

- construct without pretrained downloads using `pretrained="none"`
- return Task-1 shape `(N, 2)`
- return Task-2 shape `(N, 3)`
- return Task-3 shape `(N, 5)`
- call the shared encoder once per forward pass
- preserve dropout 0.2
- reject unsupported architectures explicitly
- preserve the legacy EfficientNet-B0 builder

## Architecture smoke validation

The encoder extraction logic was separately smoke-validated with torchvision CPU models using a synthetic `1 x 3 x 64 x 64` input.

Observed pre-head feature dimensions and output shapes:

| Backbone | Feature dimension | Task output shapes |
|---|---:|---|
| EfficientNet-B0 | 1280 | 2 / 3 / 5 |
| DenseNet121 | 1024 | 2 / 3 / 5 |
| DenseNet169 | 1664 | 2 / 3 / 5 |
| ResNet50 | 2048 | 2 / 3 / 5 |
| MobileNetV3-Large | 960 | 2 / 3 / 5 |
| EfficientNet-B2 | 1408 | 2 / 3 / 5 |
| EfficientNet-B3 | 1536 | 2 / 3 / 5 |

All seven architecture paths produced valid Task-1/Task-2/Task-3 logits.

## Verification limitation

The connected GitHub repository has no CI run associated with the Gate 06B commits, and the execution container could not clone GitHub because outbound DNS/network access is unavailable. Therefore the committed repository pytest suite could not be executed end-to-end in this session.

This does not authorize skipping verification before GPU use.

## Mandatory Gate 06C preflight

Before each new full T4 run, execute the Phase 06 launcher with `--preflight` on the actual project checkout. A failed preflight blocks that architecture from training.

Example:

```bash
python scripts/train_phase06_shared_three_task.py \
  --architecture densenet169 \
  --run-directory runs/phase06_shared_three_task/densenet169/seed_42 \
  --device cuda \
  --preflight
```

The preflight must confirm:

- dataloaders resolve
- source counts match the frozen config
- selected model constructs
- losses construct
- AdamW and cosine scheduler construct
- internal-test loader is not constructed

The full local pytest suite should also be run on the project checkout before the first full training command.

## Training scope after Gate 06B

Reuse without retraining:

- Shared EfficientNet-B0 frozen Phase 03 checkpoint

New full runs permitted only after successful preflight:

1. Shared DenseNet121
2. Shared DenseNet169
3. Shared ResNet50
4. Shared MobileNetV3-Large
5. Shared EfficientNet-B2
6. Shared EfficientNet-B3

## Gate conclusion

Gate 06B implementation is complete and protocol-preserving. The new architecture factory supports all seven approved encoders, the old EfficientNet-B0 API remains available, and a Phase 06 launcher isolates architecture as the only permitted model-level experimental variable.

**Gate 06B: PASS, conditional on mandatory local/T4 preflight before each Gate 06C full run.**

## Exact next task

**Gate 06C — GPU training.**

First run the repository tests and six architecture preflights. Only architectures whose preflight passes may start full training. The existing EfficientNet-B0 shared checkpoint must be reused rather than retrained.