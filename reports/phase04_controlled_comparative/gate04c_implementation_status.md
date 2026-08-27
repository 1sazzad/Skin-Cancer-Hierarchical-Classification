# Gate 04C — Evaluation Harness Implementation Status

Date: 2026-08-28

## Status

**IMPLEMENTATION COMPLETE / EXECUTION VERIFICATION REQUIRED BEFORE PASS**

Gate04C must not be marked PASS until the repository tests and frozen-checkpoint
interface preflight have succeeded on the authoritative project workspace.
No internal-test dataset loader or inference is executed by the Gate04C preflight.

## Branch

`phase04-controlled-comparative-evaluation`

Base:

`phase03-shared-three-task-hierarchical-baseline` at Phase03 closure commit
`aea3f1f7fc89ce6b46d491bba19f39d19a7f6a28`.

## Implemented

- Frozen SHA-256 checkpoint verification.
- Strict shared three-task checkpoint loading.
- Strict standalone Task1/Task2/Task3 and flat four-class checkpoint loading.
- Frozen EfficientNet-B0 architecture/dropout enforcement in loaders.
- Shared Task1 and malignant-subset Task2 metric calculation.
- Generic Task3/standalone/flat metric calculation using the repository metric implementation.
- Shared one-pass Task1/Task2 ISIC collection.
- Existing locked hierarchical routing reused for predicted-gate and oracle-gate four-class outputs.
- Routing-loss Macro-F1 and malignant blocking statistics.
- Stable sample-ID paired export for shared-vs-flat bootstrap/McNemar analysis.
- Parameter count and verified checkpoint-byte hooks.
- Optional FLOP hook via fvcore when available.
- Matched-device latency, throughput and peak CUDA-memory benchmark hook.
- Environment/Git provenance capture.
- Validation-only frozen configuration with `internal_test_execution_allowed: false`.
- Gate04C preflight script that constructs no dataset loader.
- Synthetic unit tests for routing loss, blocking rate, paired export integrity, metrics,
  SHA verification, parameter counting and CPU benchmark execution.

## Reused frozen repository logic

- `src/evaluation/classification_metrics.py`
- `src/evaluation/hierarchical_evaluator.py`
- `src/models/shared_three_task.py`
- `src/models/classification_backbone.py`
- `src/data/transforms.py`

The existing deterministic evaluation preprocessing remains Resize(256), CenterCrop(224),
float conversion and ImageNet normalization. No new preprocessing policy was introduced.

## Files added/changed

- `src/evaluation/phase04_comparative_harness.py`
- `src/evaluation/shared_task_head_adapter.py`
- `scripts/preflight_phase04_evaluation.py`
- `configs/evaluation/phase04_controlled_comparative_validation.yaml`
- `tests/test_phase04_comparative_harness.py`
- this status report

## Required closure commands on authoritative workspace

```bash
git fetch origin
git switch phase04-controlled-comparative-evaluation
git pull --ff-only

python -m pytest -q tests/test_phase04_comparative_harness.py
python scripts/preflight_phase04_evaluation.py \
  --project-root "$PWD" \
  --device cuda
```

The preflight must report `checkpoint_interface_validation: PASS` and
`internal_test_executed: false`. It writes:

`reports/phase04_controlled_comparative/gate04c_preflight.json`

## Gate restriction

Do **not** run validation inference or any internal-test inference while closing Gate04C.
Gate04D begins only after the two closure commands above pass.

## Gate04D after PASS

Gate04D is validation-only dry-run execution of the frozen harness. Resolve the authoritative
ISIC and ISIC-derived melanoma T-category manifest paths from the existing frozen Phase03
configs, construct validation-only loaders with `build_eval_transform()`, run the five frozen
comparators/shared heads, verify metric/export schemas and run the T4 efficiency hooks.
The internal test remains closed throughout Gate04D.
