#!/usr/bin/env python3
"""Analyze already-stored Phase 04 final internal-test evidence; never run inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.phase04_final_internal_test import analyze, audit_inputs, write_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "reports/phase04_controlled_comparative/final_internal_test"
    summary, frame, recomputed, integrity = audit_inputs(
        output / "final_internal_test_summary.json",
        output / "paired_internal_test_predictions.csv",
        root / "configs/evaluation/phase04_controlled_comparative_internal_test.yaml",
    )
    result = analyze(summary, frame, recomputed, integrity)
    write_outputs(output, result)
    print(f"PASS: analyzed {len(frame)} stored paired predictions; no inference performed.")


if __name__ == "__main__":
    main()
