# Gate 06A — Existing Evidence Audit

## Verdict

**PASS**

Gate 06A confirms that the existing seven-backbone flat evidence is sufficient for Phase 06 and that the frozen EfficientNet-B0 shared three-task checkpoint is reusable. No flat model needs retraining. The only new full training required is for the six missing shared three-task backbones after Gate 06B generalizes the shared implementation.

## Audit baseline

Phase 06 starts from Phase 05 closure commit:

`fff4afef72c210837b1ad1c352e9c64b1994aeeb`

Frozen ISIC split:

`data/manifests/isic2019_train_val_test_split_seed42.csv`

Seed: `42`

Flat class order:

`[non_malignant, melanoma, bcc, scc]`

The frozen Phase 02 backbone protocol already declares the seven intended candidates:

- DenseNet121
- DenseNet169
- ResNet50
- MobileNetV3-Large
- EfficientNet-B0
- EfficientNet-B2
- EfficientNet-B3

The protocol fixes ImageNet initialization, 224x224 input, the same moderate augmentation family, batch size 64, AdamW at 3e-4, weight decay 1e-4, cosine annealing, maximum 30 epochs, patience 7, seed 42, and validation Macro-F1 checkpoint selection.

## Flat-backbone evidence inventory

| Backbone | Runnable config | Frozen checkpoint | Validation evidence | Existing internal-test prediction | Retrain? |
|---|---|---|---|---|---|
| DenseNet121 | yes | yes | yes | yes | no |
| DenseNet169 | yes | yes | yes | yes | no |
| ResNet50 | yes | yes | yes | no | no |
| MobileNetV3-Large | yes | yes | yes | no | no |
| EfficientNet-B0 | yes | yes | yes | yes | no |
| EfficientNet-B2 | yes | yes | yes | no | no |
| EfficientNet-B3 | yes | yes | yes | no | no |

### Existing flat validation results

| Backbone | Validation Macro-F1 | Evidence status |
|---|---:|---|
| DenseNet121 | 0.6449820791 | completed, protocol-matched historical run |
| DenseNet169 | 0.6659647727 | completed, Phase 02 validation-selected winner |
| ResNet50 | 0.6533194436 | completed, not selected |
| MobileNetV3-Large | 0.6582407229 | completed, not selected |
| EfficientNet-B0 | 0.6535716654 | completed, protocol-compatible historical run |
| EfficientNet-B2 | 0.6544148913 | completed, not selected |
| EfficientNet-B3 | 0.6583008001 | completed, not selected |

DenseNet169 remains the validation-selected flat architecture winner.

Important: the absence of internal-test predictions for ResNet50, MobileNetV3-Large, EfficientNet-B2, and EfficientNet-B3 is intentional. Those candidates were not evaluated on the internal test during the original architecture-selection phase. This is not missing training evidence and does not justify retraining.

## EfficientNet-B0 shared three-task audit

The frozen shared baseline is reusable.

Model:

- one shared EfficientNet-B0 encoder pass
- ImageNet initialization
- dropout probability 0.2
- Task 1 head: 2 logits
- Task 2 head: 3 logits
- Task 3 head: 5 logits

Losses:

- Task 1: Cross-Entropy
- Task 2: Class-Balanced Focal Loss, beta=0.9999, gamma=2.0
- Task 3: inverse-frequency Weighted Cross-Entropy
- task weights: 1:1:1

Training protocol:

- seed 42
- batch size 64
- maximum 30 epochs
- patience 7
- AdamW, learning rate 0.0003
- weight decay 0.0001
- cosine annealing
- validation-only checkpoint selection
- shared validation score = arithmetic mean of Task-1, Task-2, and Task-3 Macro-F1

Frozen checkpoint:

`runs/phase03_shared_three_task/seed_42/best_checkpoint.pt`

SHA-256:

`2f1c2393c5c9de15dfa4a1a132a31b9a5b8ede07d7ed6e07ab90918fc2aaa9eb`

Frozen epoch: `6`

Shared validation score: `0.6326175865948461`

Task validation Macro-F1 values:

- Task 1: `0.7624178951977814`
- Task 2: `0.6987503899226807`
- Task 3: `0.4366844746640764`

Therefore EfficientNet-B0 must **not** be retrained for Phase 06 unless a later integrity check proves the frozen checkpoint unavailable or corrupted.

## Shared implementation audit

The current shared model implementation is architecture-specific.

`src/models/shared_three_task.py` directly imports and constructs `efficientnet_b0`, extracts its `features` and `avgpool`, and defines `SharedThreeTaskEfficientNetB0`.

Therefore Gate 06B is necessary before any new GPU training. Gate 06B must generalize only the encoder construction while preserving the three heads, task mappings, loss behavior, preprocessing, training budget, and validation-selection rule.

The existing flat backbone factory in `src/models/phase02_backbones.py` provides reusable architecture-handling evidence for the seven encoder families.

## Phase 06 training decision

### Reuse without retraining

- Flat DenseNet121
- Flat DenseNet169
- Flat ResNet50
- Flat MobileNetV3-Large
- Flat EfficientNet-B0
- Flat EfficientNet-B2
- Flat EfficientNet-B3
- Shared EfficientNet-B0

### New shared training required

1. Shared DenseNet121
2. Shared DenseNet169
3. Shared ResNet50
4. Shared MobileNetV3-Large
5. Shared EfficientNet-B2
6. Shared EfficientNet-B3

Expected number of new full GPU runs: **6**.

## Internal-test governance clarification

The earlier flat architecture-selection protocol intentionally withheld the internal test from losing candidates. Phase 06 asks a different, predeclared scientific question: whether the flat-versus-shared hierarchy effect and routing-associated degradation are consistent across backbone families.

To avoid post-hoc model selection, the following rule is frozen now, before Gate 06B implementation and before any new shared training:

1. All seven shared checkpoints are selected using validation evidence only.
2. No Phase 06 internal-test result may trigger retraining, hyperparameter changes, backbone switching, threshold tuning, loss changes, or checkpoint reselection.
3. Existing flat internal-test predictions must be reused where they already exist; they must not be rerun merely for convenience.
4. For flat backbones without stored internal-test predictions, any Gate 06D inference is a one-time, predeclared comparative analysis, not architecture selection.
5. All seven matched backbone pairs must be reported; no pair may be omitted because its test result is unfavorable.
6. Primary architecture selection remains validation-only.
7. Oracle routing is diagnostic and must not be presented as a deployable model.

This prospective rule prevents Phase 06 from turning the internal test into a tuning set while still allowing the new backbone-invariance research question to be evaluated.

## Frozen Phase 06 protocol summary

Do not change:

- dataset split
- class order
- train/validation/test partitions
- input resolution
- preprocessing family
- seed-42 primary protocol
- optimizer family
- scheduler family
- maximum training budget
- early stopping patience
- test-set non-selection rule
- task definitions
- task losses
- 1:1:1 task weighting
- validation-only checkpoint selection

Only the shared encoder architecture may vary across the seven backbones.

## Gate conclusion

**Gate 06A: PASS.**

No flat retraining is justified. The existing EfficientNet-B0 shared checkpoint is reusable. Six new shared full-training runs remain. Before those runs, Gate 06B must generalize the shared three-task encoder implementation and pass local/unit sanity checks without accessing the internal test.

## Exact next task

**Gate 06B — Multi-backbone shared implementation.**

Generalize the shared three-task model to support all seven frozen backbone names while preserving identical task heads and training semantics. Run local/unit sanity tests only; do not start full GPU training in Gate 06B.