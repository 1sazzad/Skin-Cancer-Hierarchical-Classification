# Phase 02 / Gate 02A — Local Backbone Support and Protocol Compliance

## Status and scope

**Gate 02A status: PASS.** This gate adds local implementation and configuration
support for the five new Phase 02 backbones. No full or dataset-backed training,
GPU execution, pretrained-weight download, internal-test inference, split
regeneration, hyperparameter tuning, VM work, push, or merge was performed.

Starting provenance was the clean Phase 01 branch
`phase01-existing-experiment-audit-protocol-freeze` at
`7c4460fd339ca13bf3d4b47122181648fa7fefdb`. Work proceeded on
`phase02-controlled-seven-backbone-benchmark`.

## Implementation inspected

The audit inspected the generic factory, EfficientNet-B0 and DenseNet121
builders, configuration loader, training runner, transforms, metrics evaluator,
internal-test evaluator, training CLI, model/config tests, existing experiment
configs, and `configs/protocols/phase02_flat_four_class_backbone_benchmark.yaml`.

Existing behavior was:

- EfficientNet-B0 uses torchvision `EfficientNet_B0_Weights.DEFAULT` and a
  replacement `Dropout → Linear` classifier.
- DenseNet121 uses `DenseNet121_Weights.DEFAULT` and a replacement
  `Dropout → Linear` classifier.
- The generic factory passes the class count, pretrained mode and dropout.
- The configuration loader validates architecture membership and previously
  restricted DenseNet121 to the flat task.
- `run_baseline_experiment` builds through the generic factory and iterates only
  `dataloaders["train"]` and `dataloaders["validation"]`. It never invokes the
  separate internal-test evaluator. Internal-test evaluation remains an
  explicit, separate script/API operation.

## Architecture support and classifier heads

The factory now recognizes, in its locked order:

`efficientnet_b0`, `densenet121`, `densenet169`, `resnet50`,
`mobilenet_v3_large`, `efficientnet_b2`, and `efficientnet_b3`.

All use torchvision official default ImageNet weight enums when
`pretrained="imagenet"`, accept `pretrained="none"` for offline tests, and
produce `[B, 4]` logits for the Phase 02 task.

| Architecture | Head handling | Effective classifier dropout |
|---|---|---:|
| EfficientNet-B0 | Existing classifier replaced by `Dropout → Linear(4)` | 0.2 |
| DenseNet121 | Existing `.classifier` replaced by `Dropout → Linear(4)` | 0.2 |
| DenseNet169 | `.classifier` replaced by the same DenseNet convention | 0.2 |
| ResNet50 | Native `.fc` wrapped as `Dropout → Linear(4)` | 0.2 |
| MobileNetV3-Large | Native hidden Linear and Hardswish retained; native dropout set to 0.2; only final Linear replaced | 0.2 |
| EfficientNet-B2 | Classifier replaced by the existing EfficientNet convention | 0.2 |
| EfficientNet-B3 | Classifier replaced by the existing EfficientNet convention | 0.2 |

The unavoidable distinction is MobileNetV3-Large's architecture-native
multi-layer classifier, which is retained rather than flattened into the
two-layer project convention. ResNet50 has no native classifier dropout, so a
dropout layer is added immediately before the replacement linear layer.

## Runnable configuration strategy

Five new configs were created under `configs/experiments/`:

- `phase02_flat_four_class_isic2019_densenet169_cross_entropy.yaml`
- `phase02_flat_four_class_isic2019_resnet50_cross_entropy.yaml`
- `phase02_flat_four_class_isic2019_mobilenet_v3_large_cross_entropy.yaml`
- `phase02_flat_four_class_isic2019_efficientnet_b2_cross_entropy.yaml`
- `phase02_flat_four_class_isic2019_efficientnet_b3_cross_entropy.yaml`

The configuration loader recognizes the research-stage marker
`phase02_controlled_backbone_benchmark` and rejects drift in scientific fields.
It also restricts every newly supported non-B0 architecture to the flat
four-class task. Each new config explicitly sets
`evaluate_internal_test_after_training: false`.

## Test-set quarantine

**PASS.** Completing a baseline training run does not automatically evaluate
the internal test. The internal-test evaluator and CLI remain available for a
separately authorized post-selection evaluation, preserving other phases'
capability. A regression test supplies an internal-test loader that raises if
iterated and verifies that the runner passes only train and validation loaders
to the epoch engine. No historical test results appear in configuration
validation or ranking logic.

## Protocol compliance matrix

The five configs were parsed and compared after normalizing only run name and
architecture identity. They are otherwise structurally equal. Strict loader
validation also rejects frozen-field changes.

| Frozen field | DenseNet169 | ResNet50 | MobileNetV3-Large | EfficientNet-B2 | EfficientNet-B3 |
|---|---|---|---|---|---|
| ISIC 2019 / frozen seed-42 split | PASS | PASS | PASS | PASS | PASS |
| Class order 0–3 | PASS | PASS | PASS | PASS | PASS |
| Seed 42 | PASS | PASS | PASS | PASS | PASS |
| 224×224 / locked transforms / ImageNet normalization | PASS | PASS | PASS | PASS | PASS |
| ImageNet pretraining / dropout 0.2 | PASS | PASS | PASS | PASS | PASS |
| Ordinary CE / no weights / no weighted sampler / no focal | PASS | PASS | PASS | PASS | PASS |
| AdamW / LR 0.0003 / weight decay 0.0001 | PASS | PASS | PASS | PASS | PASS |
| Cosine annealing / eta_min 0.000001 | PASS | PASS | PASS | PASS | PASS |
| Batch 64 / loader settings frozen | PASS | PASS | PASS | PASS | PASS |
| 30 epochs / patience 7 / validation Macro-F1 | PASS | PASS | PASS | PASS | PASS |
| CUDA AMP enabled | PASS | PASS | PASS | PASS | PASS |
| Automatic internal-test evaluation disabled | PASS | PASS | PASS | PASS | PASS |

Permitted differences are limited to architecture/model identity, run name,
and the eventual run-directory name derived from the run name.

## Local tests and sanity checks

Targeted command:

```powershell
.\tmp\python-3.11.9-embed-amd64\python.exe -m pytest -q tests/test_phase02_backbone_support.py tests/test_baseline_model.py tests/test_phase11_densenet_baseline.py tests/test_phase03_baseline_experiment.py tests/test_transforms.py tests/test_classification_metrics.py
```

Result: **42 passed in 33.18 seconds**, with no warnings. The dedicated Phase 02
file also passed 21/21 during its focused iteration.

Full-suite command:

```powershell
.\tmp\python-3.11.9-embed-amd64\python.exe -m pytest -q
```

Result: **326 passed in 60.11 seconds**.

The temporary interpreter was necessary because the checked-in `.venv` launcher
references a Python installation no longer present on this machine. It used the
existing `.venv` packages and was removed after testing.

CPU-only offline forward passes constructed all seven architectures with
`weights=None`, confirmed rank-2 four-logit output, final output dimension four,
and effective classifier dropout 0.2. All five new configs resolved. The runner
quarantine test performed no optimization or dataset inference.

## Files changed

- Added `src/models/phase02_backbones.py`.
- Extended `src/models/classification_backbone.py`.
- Extended `src/training/baseline_experiment.py` with Phase 02 validation.
- Added five Phase 02 experiment configs.
- Added `tests/test_phase02_backbone_support.py`.
- Updated the explicit architecture-list assertion in
  `tests/test_phase11_densenet_baseline.py`.
- Added this Gate 02A report.

## Unresolved issues and stop gate

No Gate 02A blocker remains. The local `.venv` launcher is stale, but the test
suite passes using the matching temporary Python 3.11.9 runtime and its existing
packages. This preparation does not authorize training or internal-test access.

Recommended next action: review this focused local commit, then separately
authorize Phase 02 controlled training of only the five new backbones using
validation-only architecture ranking. Stop before any internal-test evaluation.
