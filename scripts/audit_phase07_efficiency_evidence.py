"""Generate the deterministic Phase 07 Gate 5A efficiency audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.efficiency_evidence import generate_efficiency  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--generated-output", type=Path, default=Path("reports/phase07/generated"))
    parser.add_argument("--report-output", type=Path, default=Path("reports/phase07"))
    args = parser.parse_args()
    outputs = generate_efficiency(args.repository, args.generated_output, args.report_output)
    print(f"Generated {len(outputs)} Gate 5A artifacts without model execution.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
