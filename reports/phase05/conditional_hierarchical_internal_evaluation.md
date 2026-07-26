# Phase 05 - Conditional Hierarchical Internal Evaluation

## Experimental status

- Dataset: ISIC 2019.
- Split: frozen leakage-aware seed-42 internal-test partition.
- Internal-test samples: 3,668.
- Stage 1 task: non-malignant versus malignant.
- Stage 2 task: melanoma versus BCC versus SCC.
- Final classes: non-malignant, melanoma, BCC, and SCC.
- Stage 1 gate policy: argmax.
- Stage 2 production-style execution: Stage 1 predicted malignant only.
- Stage 2 evaluation union: true malignant or Stage 1 predicted malignant.
- Internal-test evaluation was locked before execution.
- Primary result is reportable and must not be rerun or used for tuning.

## Frozen model provenance

### Stage 1

- Architecture: EfficientNet-B0.
- Loss: ordinary cross-entropy.
- Frozen epoch: 5.
- Validation macro-F1: 0.808693.
- Checkpoint SHA-256:
  `95e02c26b1ea4a0dba17016313c81f97c9c2635270a37b4debbee0f84e07ba3b`.

### Stage 2

- Architecture: EfficientNet-B0.
- Loss: class-balanced focal loss.
- Effective-number beta: 0.9999.
- Focal gamma: 2.0.
- Frozen epoch: 8.
- Validation macro-F1: 0.776307.
- Checkpoint SHA-256:
  `10986d41b64a685fcd8fe166623c5b1c7fd2f21bdad7cf4d55dedc3967a397fd`.

## Locked execution provenance

- Repository commit:
  `08b76044c52d5fe3e3b7082c5be73298c271ba77`.
- Device: NVIDIA Tesla T4.
- PyTorch: 2.13.0+cu130.
- Python: 3.12.3.
- Batch size: 64.
- DataLoader workers: 4.
- Evaluation duration: 39.847 seconds.
- Throughput: 92.051 images per second.
- Successful attempt completed at:
  `2026-07-26T20:51:10Z`.
- Evaluation exit code: 0.

## Headline results

| Evaluation view | Samples | Accuracy | Balanced accuracy | Macro-F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| Standalone Stage 1 | 3,668 | 0.786260 | 0.789306 | 0.774009 | 0.790190 |
| Oracle-gated Stage 2 | 1,270 | 0.833858 | 0.722716 | 0.724875 | 0.832915 |
| Oracle-gate four-class | 3,668 | 0.942475 | 0.792037 | 0.793656 | 0.942149 |
| Predicted-gate end-to-end | 3,668 | 0.740185 | 0.631199 | 0.605367 | 0.750332 |

The predicted-gate four-class macro-F1 of 0.605367 is the primary
end-to-end result.

## Predicted-gate four-class confusion matrix

Rows are actual classes and columns are predicted classes in the order
`non_malignant`, `melanoma`, `bcc`, and `scc`.

| Actual / Predicted | Non-malignant | Melanoma | BCC | SCC |
|---|---:|---:|---:|---:|
| Non-malignant | 1869 | 385 | 101 | 43 |
| Melanoma | 177 | 463 | 18 | 20 |
| BCC | 66 | 60 | 349 | 23 |
| SCC | 12 | 15 | 33 | 34 |

## Predicted-gate per-class performance

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Non-malignant | 0.879944 | 0.779399 | 0.826625 | 2,398 |
| Melanoma | 0.501625 | 0.682891 | 0.578389 | 678 |
| BCC | 0.696607 | 0.700803 | 0.698699 | 498 |
| SCC | 0.283333 | 0.361702 | 0.317757 | 94 |

SCC remains the weakest final class. Its low support and confusion with BCC
continue to limit macro-F1.

## Oracle-gated Stage 2 performance

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Melanoma | 0.874116 | 0.911504 | 0.892419 | 678 |
| BCC | 0.846809 | 0.799197 | 0.822314 | 498 |
| SCC | 0.462366 | 0.457447 | 0.459893 | 94 |

Even with perfect routing, SCC remains difficult. This shows that Stage 2
minority-class discrimination is a secondary bottleneck independent of Stage 1
routing.

## Routing analysis

| Routing quantity | Count or rate |
|---|---:|
| True malignant samples | 1,270 |
| Correctly routed malignant samples | 1,015 |
| Malignant samples blocked by Stage 1 | 255 |
| Malignant block rate | 20.079% |
| True non-malignant samples | 2,398 |
| Non-malignant samples incorrectly routed | 529 |
| Incorrect non-malignant route rate | 22.060% |
| Predicted malignant samples | 1,544 |
| Evaluation-union Stage 2 executions | 1,799 |
| Correct subtype predictions after correct routing | 846 |
| Subtype errors after correct routing | 169 |
| Subtype error rate after correct routing | 16.650% |

The production-style predicted gate invokes Stage 2 for 1,544 of 3,668
samples, or approximately 42.09% of the internal-test set. The larger union
count of 1,799 is an evaluation-only mechanism used to support both oracle and
predicted-gate analyses.

This conditional execution result must not be described as a 57.91% reduction
in total computation because Stage 1 is still executed for every image.

## Error propagation

| Metric | Oracle gate | Predicted gate | Absolute loss |
|---|---:|---:|---:|
| Accuracy | 0.942475 | 0.740185 | 0.202290 |
| Balanced accuracy | 0.792037 | 0.631199 | 0.160838 |
| Macro-F1 | 0.793656 | 0.605367 | 0.188289 |
| Weighted F1 | 0.942149 | 0.750332 | 0.191817 |

The 0.188289 absolute macro-F1 gap demonstrates that Stage 1 routing error is
the dominant source of end-to-end performance loss.

## Stage 1 numerical consistency audit

The Phase 05 hierarchy used the same frozen Stage 1 epoch-5 checkpoint as the
Phase 03 standalone evaluation. Five of 3,668 predictions differed between the
two saved evaluations.

All five affected samples were true non-malignant cases with probabilities
close to the 0.5 decision boundary. Four changed from non-malignant in Phase 03
to malignant in Phase 05, while one changed in the opposite direction.

The net confusion-matrix difference was:

- three fewer true-negative predictions;
- three additional false-positive malignant predictions;
- no change in malignant true positives or false negatives.

This behaviour is consistent with small floating-point differences under
different inference execution configurations, including batch size, worker
configuration, and numerical execution path. It does not indicate a checkpoint,
label, model-selection, or routing-policy change.

The five affected records are stored in:

`reports/phase05/stage01_numerical_consistency_audit.csv`

For the hierarchical paper analysis, the locked Phase 05 outputs are the
authoritative end-to-end results. The previously locked Phase 03 standalone
result remains the authoritative Phase 03 baseline result.

## Failed attempt and recovery audit

The first Phase 05 launch failed before metric calculation and before creation
of the locked output directory. CUDA autocast produced half-precision Stage 2
probabilities that could not be assigned to a float32 collection tensor.

The failure was preserved under:

`runs/phase05_hierarchical_internal_test/failed_attempt_01_amp_dtype_mismatch`

The fix promoted Stage 1 and Stage 2 logits to float32 before softmax and
probability collection. The following checks passed before the recovery run:

- targeted inference-engine regression tests;
- complete test suite;
- synthetic CUDA half-precision regression;
- frozen-checkpoint SHA and output-shape preflight.

The fix did not change checkpoint selection, model weights, hierarchy mapping,
gate policy, loss parameters, or test labels. The successful recovery run is
therefore an implementation-failure recovery rather than result-driven model
retuning.

## Reproducibility artifacts

The locked output contains:

- hierarchical metrics;
- four confusion matrices;
- per-class metric tables;
- routing analysis;
- error-propagation analysis;
- 3,668 per-image predictions;
- checkpoint provenance;
- execution environment;
- locked protocol snapshot;
- evaluation summary.

All 16 result files, the completion log, and exit-code file were protected by
an 18-entry SHA-256 manifest. The transferred local archive matched SHA-256:

`48455c488ecc74f5d859f796a343399ff9653eaf8b439de38d478bfc4362475a`

## Scientific interpretation

The hierarchy provides a clinically interpretable decomposition between
malignancy screening and malignant-subtype recognition. However, the actual
end-to-end result is substantially lower than the oracle-gate diagnostic
ceiling.

The primary bottleneck is Stage 1 routing:

- 20.08% of malignant lesions were blocked;
- 22.06% of non-malignant lesions were unnecessarily routed;
- routing reduced four-class macro-F1 by 0.188289.

Stage 2 remains a secondary bottleneck for SCC, whose oracle-gated F1 was only
0.459893.

The result supports a routing-error-analysis contribution. It does not yet
support a claim that the hierarchy outperforms direct flat four-class
classification.

## ICCIT claim boundaries

Supported claims:

- the conditional hierarchy achieved four-class macro-F1 0.605367 and accuracy
  0.740185 on the frozen ISIC 2019 internal-test split;
- oracle routing increased four-class macro-F1 to 0.793656;
- Stage 1 routing produced an absolute macro-F1 loss of 0.188289;
- predicted routing invoked Stage 2 for 42.09% of samples;
- routing errors and subtype errors were measured separately;
- class-balanced focal loss produced a modest Stage 2 improvement over clean
  cross-entropy.

Unsupported claims:

- state-of-the-art performance;
- superiority over a flat classifier before Phase 06;
- clinical readiness;
- dermatologist-level performance;
- cross-dataset generalisation;
- fairness across skin tones;
- total-compute reduction of 57.91%;
- elimination of all possible patient-level leakage.

## Phase 05 outcome

Phase 05 is complete. The locked hierarchical internal-test result is
reportable, locally backed up, checksum verified, and must not be rerun or used
for further model selection.

The next experiment is a fair flat four-class EfficientNet-B0 comparator using
the same frozen split, preprocessing policy, seed, validation-based checkpoint
selection, and one-time internal-test protocol.
