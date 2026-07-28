# Phase 07 Gate 5A — Efficiency Evidence Audit

## Decision and method

PASS. Evidence was collected without training, inference, evaluation, model construction, forward passes, benchmarking or GPU work. Parameter counts are Grade C static measurements from verified checkpoints loaded CPU-only with `weights_only=True`; standard BatchNorm buffers were separated. Checkpoints contain optimizer and scheduler state, so file size is storage evidence only.

## Static artifacts and conditional compute

| Role | Parameters | Checkpoint bytes | MiB |
|---|---:|---:|---:|
| Flat | 4,012,672 | 48,640,921 | 46.387597 |
| Stage 1 | 4,010,110 | 48,609,369 | 46.357507 |
| Stage 2 | 4,011,391 | 48,625,369 | 46.372766 |
| Hierarchy combined | 8,021,501 | 97,234,738 | 92.730272 |

Stage 2 invocation was 1,799/3,668 (49.045802%); bypass was 1,869/3,668 (50.954198%). The hierarchy therefore used 1–2 passes and a derived mean of 1.490458015. Its average conditionally active parameter-pass count was 5977528.868321; this is a workload proxy, not FLOPs, memory or latency.

## Stored timing

Flat: 30.6784900749999 seconds, 119.56259877956239 samples/s, 8.363819541 ms/sample. Hierarchy: 39.8473870979997 seconds, 92.0512050383382 samples/s, 10.863518838 ms/sample.

Both are stored Tesla T4, batch-64, CUDA-float16-autocast evaluator-loop measurements. The timer includes dataloader iteration, transfers, model work and CPU collection but excludes post-loop metrics/writes. Neither path documents warm-up or explicit CUDA synchronization, and the evaluator paths differ. Classification: comparable with limitations. No speed ratio or claim that either system is faster is authorized.

FLOPs/MACs, peak GPU/CPU memory, energy, power, and model-loading time are Grade U unavailable and were not estimated.
