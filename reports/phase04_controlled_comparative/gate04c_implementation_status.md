# Gate 04C — Evaluation Harness Closure

Date: 2026-08-28

## Status

**PASS**

Gate04C is closed. The repository tests and frozen-checkpoint/interface preflight both succeeded on the authoritative Azure Tesla T4 workspace. No internal-test dataset loader or inference was executed.

## Branch and provenance

- Starting branch: `phase03-shared-three-task-hierarchical-baseline`
- Starting commit: `aea3f1f7fc89ce6b46d491bba19f39d19a7f6a28`
- Final branch: `phase04-controlled-comparative-evaluation`
- Preflight code commit: `14851847634d1c7409578cceb46b18b6f348f8d0`
- Python: `3.11.9`
- PyTorch: `2.13.0+cu130`
- Device: CUDA / Tesla T4

## Verification evidence

### Unit tests

Command:

```bash
python -m pytest -q tests/test_phase04_comparative_harness.py
```

Result:

```text
6 passed in 2.98s
```

### Frozen checkpoint/interface preflight

Command:

```bash
python scripts/preflight_phase04_evaluation.py \
  --project-root "$PWD" \
  --device cuda
```

Result:

```text
Gate04C checkpoint/interface preflight PASS
```

The generated evidence is retained in:

`reports/phase04_controlled_comparative/gate04c_preflight.json`

Required fields were verified:

- `checkpoint_interface_validation: PASS`
- `internal_test_executed: false`
- `device: cuda`
- `gpu_name: Tesla T4`

## Frozen artifacts verified

| Group | Expected epoch | SHA-256 | Parameters | Checkpoint bytes |
|---|---:|---|---:|---:|
| Shared three-task | 6 | `2f1c2393c5c9de15dfa4a1a132a31b9a5b8ede07d7ed6e07ab90918fc2aaa9eb` | 4,020,358 | 48,737,992 |
| Standalone Task1 | 5 | `95e02c26b1ea4a0dba17016313c81f97c9c2635270a37b4debbee0f84e07ba3b` | 4,010,110 | 48,609,369 |
| Standalone Task2 | 8 | `10986d41b64a685fcd8fe166623c5b1c7fd2f21bdad7cf4d55dedc3967a397fd` | 4,011,391 | 48,625,369 |
| Standalone Task3 | 12 | `71bfda5f7a19333947e1c13f4e1c5ed45e9a827c447fc9bcd6fd9ddc999f8692` | 4,013,953 | 48,656,473 |
| Flat four-class | 2 | `f3d8b8b0e5ef42e3c287a2377b5570411d442246acd16cb874ccf903facdc7a7` | 4,012,672 | 48,640,921 |

Optional FLOP/MAC reporting was unavailable because the optional backend was not present; this is explicitly recorded as unsupported rather than estimated.

## Shared checkpoint recovery note

The shared Phase03 checkpoint was absent from the active Phase04 worktree at its canonical path. The exact frozen artifact was found in the preserved Phase03 worktree at:

`/home/m-sazzad-h/projects/Skin-Cancer-Hierarchical-Classification-phase03/runs/phase03_shared_three_task/seed_42/best_checkpoint.pt`

Its SHA-256 exactly matched the locked value and it was copied into the Phase04 worktree canonical path before preflight. No checkpoint was retrained or modified.

## Implemented Gate04C scope

- Frozen SHA-256 checkpoint verification.
- Strict shared three-task checkpoint loading.
- Strict standalone Task1/Task2/Task3 and flat four-class checkpoint loading.
- Frozen EfficientNet-B0 architecture/dropout enforcement in loaders.
- Shared Task1 and malignant-subset Task2 metric calculation.
- Generic Task3/standalone/flat metric calculation using repository metric logic.
- Shared one-pass Task1/Task2 ISIC collection.
- Locked predicted-gate and oracle-gate hierarchical routing reuse.
- Routing-loss Macro-F1 and malignant blocking statistics.
- Stable sample-ID paired export for shared-vs-flat bootstrap/McNemar analysis.
- Parameter count and verified checkpoint-byte hooks.
- Optional FLOP hook with explicit unsupported state when unavailable.
- Matched-device latency, throughput and peak CUDA-memory benchmark hook.
- Environment/Git provenance capture.
- Validation-only frozen configuration with `internal_test_execution_allowed: false`.
- Gate04C preflight that constructs no dataset loader.
- Synthetic unit tests for routing, paired export integrity, metrics, SHA verification and benchmarking helpers.

## Files added/changed during Gate04C

- `src/evaluation/phase04_comparative_harness.py`
- `src/evaluation/shared_task_head_adapter.py`
- `scripts/preflight_phase04_evaluation.py`
- `configs/evaluation/phase04_controlled_comparative_validation.yaml`
- `tests/test_phase04_comparative_harness.py`
- `reports/phase04_controlled_comparative/gate04c_preflight.json`
- this closure report

## Workspace state at verification

The active branch matched `origin/phase04-controlled-comparative-evaluation`. The VM still contained an unrelated untracked `backups/` directory. The locally generated preflight JSON was untracked before being committed as closure evidence. Phase11 artifacts that previously blocked branch checkout were preserved in the workspace-local `.workspace_local_backup/` directory and excluded through `.git/info/exclude`.

## Restrictions preserved

- No validation inference was run in Gate04C.
- No internal-test inference was run.
- No frozen split, label order, preprocessing, checkpoint, architecture or loss definition was changed.
- No retraining or retuning occurred.

## Exact next task — Gate04D only

Gate04D is the **validation-only dry run** of the frozen comparative harness.

Required actions:

1. Resolve the authoritative ISIC and ISIC-derived melanoma T-category manifest paths from the frozen Phase03 configuration/evidence already in the repository.
2. Construct validation-only loaders using the existing deterministic `build_eval_transform()` preprocessing.
3. Run the shared model and all frozen comparators on validation only.
4. Verify Task1, malignant-subset Task2, Task3, predicted-gate four-class, oracle-gate four-class, routing-loss and routing-count metric schemas.
5. Verify stable sample-ID paired export for shared-vs-flat analysis.
6. Exercise the Tesla T4 latency/throughput/peak-memory hooks under the locked efficiency settings.
7. Do not open or execute the internal test. Gate04E remains the first authorized internal-test execution gate.
