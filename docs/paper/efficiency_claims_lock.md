# Phase 07 Efficiency Claims Lock

## Supported

- The flat system uses one model decision path per image.
- The hierarchical system stores and coordinates two component models.
- Stage 2 was invoked for 1,799 of 3,668 samples on the locked split.
- The hierarchy required one to two model passes per image and a derived mean of 1 + 1799/3668.
- Static checkpoint and parameter footprints differ by the audited values.
- Stored evaluator-loop timing may be reported under its documented conditions.

## Qualified

- Timing values are comparable with limitations; evaluator paths differ and neither documents warm-up or explicit CUDA synchronization, so no speed ratio or faster-system claim is authorized.
- Average conditionally active parameter-pass count is an architecture workload proxy, not FLOPs, latency, memory, or energy.

## Prohibited formulations

- `real-time`
- `faster`
- `more memory-efficient`
- `energy-efficient`
- `mobile-ready`
- `lower FLOPs`
- `lower latency`
- `scalable`
- `production-ready`
