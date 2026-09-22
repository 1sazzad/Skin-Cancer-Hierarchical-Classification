#!/usr/bin/env python3
"""Run Phase ITT-02 synthetic-only implementation smoke checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.itt2026_consensus import (  # noqa: E402
    CohortSpec,
    FROZEN_MODELS,
    build_consensus,
    selective_referral_metrics,
    validate_and_align_prediction_frames,
)


def main() -> int:
    target = [0, 0, 1, 1, 2, 2, 3, 3]
    model_predictions = {
        "densenet121":       [0, 0, 1, 0, 2, 1, 3, 2],
        "densenet169":       [0, 0, 1, 1, 2, 2, 3, 3],
        "resnet50":          [0, 1, 1, 1, 2, 2, 3, 2],
        "mobilenet_v3_large":[0, 0, 1, 0, 2, 2, 3, 3],
        "efficientnet_b0":   [0, 0, 1, 1, 2, 1, 3, 3],
        "efficientnet_b2":   [0, 0, 1, 1, 2, 2, 2, 3],
        "efficientnet_b3":   [0, 0, 1, 1, 2, 2, 3, 3],
    }
    frames = {
        model: pd.DataFrame(
            {
                "sample_id": [f"synthetic_{index}" for index in range(len(target))],
                "target": target,
                "prediction": model_predictions[model],
            }
        )
        for model in FROZEN_MODELS
    }
    aligned = validate_and_align_prediction_frames(
        frames,
        CohortSpec(
            id_field="sample_id",
            target_field="target",
            prediction_field="prediction",
            expected_rows=len(target),
        ),
    )
    matrix = aligned[[f"pred__{model}" for model in FROZEN_MODELS]].to_numpy(dtype=np.int64)
    consensus = build_consensus(matrix)
    referral = selective_referral_metrics(
        aligned["target"].to_numpy(dtype=np.int64),
        consensus.majority_prediction,
        consensus.max_vote_count,
    )
    summary = {
        "status": "PASS",
        "scope": "synthetic_only",
        "sample_count": len(aligned),
        "majority_prediction": consensus.majority_prediction.tolist(),
        "weighted_prediction": consensus.weighted_prediction.tolist(),
        "max_vote_count": consensus.max_vote_count.tolist(),
        "thresholds_checked": [row["threshold_votes"] for row in referral],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
