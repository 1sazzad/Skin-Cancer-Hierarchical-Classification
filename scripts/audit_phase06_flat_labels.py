"""Write machine- and human-readable Phase 06 flat-label audits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.flat_four_class_audit import audit_flat_four_class_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT
        / "data/manifests/isic2019_train_val_test_split_seed42.csv",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "reports/phase06",
    )
    return parser.parse_args()


def render_markdown(audit: dict[str, object]) -> str:
    lines = [
        "# Phase 06 Flat Four-Class Label Audit",
        "",
        f"- Manifest rows: {audit['manifest_row_count']}",
        f"- Mapped rows: {audit['mapped_row_count']}",
        f"- Excluded rows: {audit['excluded_rows']}",
        f"- Reconciled: {audit['reconciled']}",
        f"- Class order: `{audit['class_order']}`",
        "",
        "## Counts",
        "",
        "| Split | non_malignant | melanoma | bcc | scc | Total |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    counts = audit["counts"]
    assert isinstance(counts, dict)
    for split in ("full_dataset", "train", "validation", "test"):
        entry = counts[split]
        classes = entry["classes"]
        cells = [
            f"{classes[name]['count']} ({classes[name]['percentage']:.4f}%)"
            for name in ("non_malignant", "melanoma", "bcc", "scc")
        ]
        lines.append(f"| {split} | {' | '.join(cells)} | {entry['total']} |")
    lines.extend(["", "## Mapping", ""])
    for source, target in audit["diagnosis_to_class"].items():
        lines.append(f"- `{source}` -> `{target}`")
    lines.extend(["", "## Explicit exclusions", ""])
    for exclusion in audit["exclusions"]:
        lines.append(
            f"- `{exclusion['diagnosis_canonical']}`: {exclusion['count']} rows; "
            f"`{exclusion['reason']}`."
        )
    lines.extend(
        [
            "",
            "## Leakage checks",
            "",
            "- Split-group cross-split count: "
            f"{audit['leakage']['split_group_id_cross_split_count']}",
            "- Exact-hash cross-split count: "
            f"{audit['leakage']['file_sha256_cross_split_count']}",
            f"- Passed: {audit['leakage']['passed']}",
            "",
            "This audit reads labels only. It performs no model loading, "
            "inference, or metrics.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    audit = audit_flat_four_class_manifest(args.manifest)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "flat_four_class_label_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_directory / "flat_four_class_label_audit.md").write_text(
        render_markdown(audit),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
