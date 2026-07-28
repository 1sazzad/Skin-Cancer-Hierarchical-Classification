"""Validate and materialize the deterministic Phase 07 protocol lock."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.analysis.paired_model_comparison import write_json  # noqa: E402
from src.analysis.statistical_protocol import load_and_validate_protocol  # noqa: E402


def main() -> int:
    """Validate configuration without executing statistical analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/analysis/phase07_paired_model_comparison.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/phase07/generated/statistical_protocol_lock.json"),
    )
    args = parser.parse_args()
    protocol = load_and_validate_protocol(args.config)
    write_json(args.output, protocol)
    print(f"Validated frozen protocol: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
