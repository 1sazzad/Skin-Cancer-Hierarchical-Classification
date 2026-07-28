"""Generate deterministic Phase 07 Gate 4 evidence and claims artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.analysis.phase07_evidence_review import generate_gate4  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("reports/phase07/generated")
    )
    parser.add_argument(
        "--generated-output", type=Path, default=Path("reports/phase07/generated")
    )
    parser.add_argument(
        "--report-output", type=Path, default=Path("reports/phase07")
    )
    args = parser.parse_args()
    outputs = generate_gate4(args.source, args.generated_output, args.report_output)
    print(f"Generated {len(outputs)} Gate 4 evidence artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
