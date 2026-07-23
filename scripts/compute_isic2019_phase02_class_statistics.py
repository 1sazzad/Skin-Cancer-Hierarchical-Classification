"""Export Stage 1 and Stage 2 class statistics from the frozen split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.class_statistics import compute_class_statistics, save_class_statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/isic2019_train_val_test_split_seed42.csv"),
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path(
            "reports/dataset_audits/isic2019_phase02_class_statistics_seed42.csv"
        ),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path(
            "reports/dataset_audits/isic2019_phase02_class_statistics_seed42.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    statistics = compute_class_statistics(args.manifest)
    save_class_statistics(statistics, args.csv_output, args.json_output)
    print(statistics.to_string(index=False))
    print(f"\nSaved: {args.csv_output}")
    print(f"Saved: {args.json_output}")


if __name__ == "__main__":
    main()
