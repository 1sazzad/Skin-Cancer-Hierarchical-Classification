"""Generate deterministic Phase 07 paper figures from committed evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.paper_figures import generate_figures  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("reports/phase07/generated"))
    parser.add_argument("--figure-output", type=Path, default=Path("reports/phase07/figures"))
    parser.add_argument("--audit-output", type=Path, default=Path("reports/phase07/generated"))
    args = parser.parse_args()
    command = (
        ".\\.venv\\Scripts\\python.exe scripts/generate_phase07_paper_figures.py "
        "--source reports/phase07/generated --figure-output reports/phase07/figures "
        "--audit-output reports/phase07/generated"
    )
    outputs = generate_figures(args.source, args.figure_output, args.audit_output, command)
    print(f"Generated {len(outputs)} deterministic figure artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
