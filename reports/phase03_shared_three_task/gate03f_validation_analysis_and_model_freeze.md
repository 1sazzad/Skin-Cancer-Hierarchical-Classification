# Gate 03F — Validation Analysis and Model Freeze

## Verdict

**Gate 03F: PASS.**

Gate 03E completed the frozen shared three-task training protocol successfully,
and Gate 03F freezes the validation-selected epoch-6 checkpoint. The internal
test remained closed throughout training, validation, selection, and this
analysis.

## Gate 03E completion

| Field | Value |
|---|---|
| Environment | NVIDIA Tesla T4 |
| Execution commit | `0b466ff7cde630867192fe029f6164afbca8c4e6` |
| Seed | 42 |
| Configured maximum epochs | 30 |
| Early-stopping patience | 7 |
| Completion status | `completed_early_stopping` |
| Epochs completed | 13 |
| Training time | 1436.8322077899938 seconds |
| Internal test evaluated | false |

## Epoch-level development summary

Each row lists total/Task-1/Task-2/Task-3 training loss, followed by the three
validation Macro-F1 values, shared validation score, and patience counter.

| Epoch | Total loss | Task 1 loss | Task 2 loss | Task 3 loss | Task 1 val F1 | Task 2 val F1 | Task 3 val F1 | Shared score | Patience |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.514761 | 0.460307 | 0.173184 | 1.009452 | 0.774783 | 0.627559 | 0.261335 | 0.554559 | 0 |
| 2 | 0.424740 | 0.404028 | 0.132763 | 0.785627 | 0.758432 | 0.656226 | 0.215710 | 0.543456 | 1 |
| 3 | 0.377618 | 0.385215 | 0.124361 | 0.694649 | 0.797543 | 0.651931 | 0.217479 | 0.555651 | 0 |
| 4 | 0.310489 | 0.367819 | 0.104857 | 0.476357 | 0.792356 | 0.654679 | 0.371825 | 0.606286 | 0 |
| 5 | 0.272423 | 0.347936 | 0.095765 | 0.383875 | 0.784699 | 0.679071 | 0.336307 | 0.600025 | 1 |
| 6 | 0.290280 | 0.347482 | 0.086692 | 0.451454 | 0.762418 | 0.698750 | 0.436684 | 0.632618 | 0 |
| 7 | 0.260697 | 0.331859 | 0.085931 | 0.379052 | 0.755953 | 0.721245 | 0.287951 | 0.588383 | 1 |
| 8 | 0.227353 | 0.321585 | 0.074465 | 0.284093 | 0.798960 | 0.685011 | 0.283658 | 0.589210 | 2 |
| 9 | 0.200949 | 0.309713 | 0.066850 | 0.203236 | 0.763390 | 0.682273 | 0.248441 | 0.564701 | 3 |
| 10 | 0.200061 | 0.311128 | 0.065582 | 0.259632 | 0.758626 | 0.656013 | 0.384279 | 0.599639 | 4 |
| 11 | 0.214779 | 0.304204 | 0.068438 | 0.278572 | 0.790519 | 0.661061 | 0.399333 | 0.616971 | 5 |
| 12 | 0.169077 | 0.282098 | 0.055114 | 0.159578 | 0.784171 | 0.658908 | 0.359271 | 0.600783 | 6 |
| 13 | 0.151237 | 0.270609 | 0.050555 | 0.114345 | 0.781986 | 0.683542 | 0.402900 | 0.622809 | 7 |

The shared score improved through epochs 1, 3, 4, and 6. No later epoch
exceeded epoch 6. Seven consecutive non-improving epochs then accumulated from
epochs 7 through 13, correctly triggering patience-7 early stopping.

## Frozen checkpoint selection

Checkpoint selection used only the arithmetic mean of the three validation
Macro-F1 values. Epoch 6 achieved the highest shared score:

| Validation measure | Exact value |
|---|---:|
| Task 1 Macro-F1 | 0.7624178951977814 |
| Task 2 Macro-F1 | 0.6987503899226807 |
| Task 3 Macro-F1 | 0.4366844746640764 |
| Shared validation score | 0.6326175865948461 |

The epoch-6 checkpoint is now **frozen**:

- Path: `runs/phase03_shared_three_task/seed_42/best_checkpoint.pt`
- SHA-256:
  `2f1c2393c5c9de15dfa4a1a132a31b9a5b8ede07d7ed6e07ab90918fc2aaa9eb`
- Size: 48,737,992 bytes

Later test evidence must not reopen checkpoint selection or trigger Phase 03
retuning.

## Descriptive historical validation context

| Task | Shared model | Historical standalone EfficientNet-B0 | Approximate delta |
|---|---:|---:|---:|
| Task 1 | 0.7624178952 | 0.808693 | -0.0463 |
| Task 2 | 0.6987503899 | 0.776307 | -0.0776 |
| Task 3 | 0.4366844747 | approximately 0.436573 | +0.0001 |

The shared encoder shows a validation trade-off: Task 1 and Task 2 are lower
than their separately trained historical models, while Task 3 is essentially
tied with historical standalone validation performance. This is descriptive
validation evidence only. It does not establish statistical significance, test
performance, overall clinical superiority, or an efficiency advantage.

## Limitations

- Task-3 validation contains only 127 samples and is severely imbalanced.
- T3 and T4 validation support is especially small.
- Training and selection used a single random seed.
- No internal-test evaluation has occurred.
- Historical validation comparisons are descriptive and not paired
  significance analyses.
- Efficiency has not yet been benchmarked under the final matched protocol.

## Gate conclusion

Gate 03F is **PASS**. Gate 03E completed under the frozen protocol, patience-7
early stopping behaved as specified, and the highest shared-validation
checkpoint from epoch 6 is frozen by path and SHA-256. The internal test remains
closed and did not influence any Phase 03 decision.
