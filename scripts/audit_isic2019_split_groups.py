#!/usr/bin/env python3
"""
Audit ISIC 2019 grouping constraints before train/validation/test splitting.

Standard-library only.

Outputs:
- reports/dataset_audits/isic2019_split_group_audit.json
- reports/dataset_audits/isic2019_exact_duplicate_hash_groups.csv
- reports/dataset_audits/isic2019_lesion_label_conflicts.csv
- reports/dataset_audits/isic2019_missing_lesion_id_distribution.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

EXPECTED_COUNT = 25_331


class DisjointSet:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)

        if left_root == right_root:
            return

        left_rank = self.rank[left_root]
        right_rank = self.rank[right_root]

        if left_rank < right_rank:
            self.parent[left_root] = right_root
        elif left_rank > right_rank:
            self.parent[right_root] = left_root
        else:
            self.parent[right_root] = left_root
            self.rank[left_root] += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit ISIC 2019 split-group leakage constraints."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root. Defaults to the parent of the scripts directory.",
    )
    return parser.parse_args()


def resolve_project_root(args: argparse.Namespace) -> Path:
    if args.project_root is not None:
        return args.project_root.expanduser().resolve()
    return Path(__file__).resolve().parent.parent


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")

        required = {
            "image_id",
            "diagnosis_original",
            "stage_1_label",
            "stage_2_label",
            "lesion_id",
            "file_sha256",
        }
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(
                f"Manifest is missing required columns: {missing}"
            )

        rows = [dict(row) for row in reader]

    if len(rows) != EXPECTED_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_COUNT} rows; found {len(rows)}."
        )

    image_ids = [row["image_id"].strip() for row in rows]
    duplicate_image_ids = [
        image_id
        for image_id, count in Counter(image_ids).items()
        if count > 1
    ]
    if duplicate_image_ids:
        raise ValueError(
            "Duplicate image IDs found. Examples: "
            + ", ".join(sorted(duplicate_image_ids)[:10])
        )

    return rows


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=path.parent,
        prefix=f"{path.name}.",
        suffix=".tmp",
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    os.replace(temp_path, path)


def atomic_write_csv(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        delete=False,
        dir=path.parent,
        prefix=f"{path.name}.",
        suffix=".tmp",
    ) as handle:
        temp_path = Path(handle.name)
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)

    os.replace(temp_path, path)


def joined(values: Iterable[str]) -> str:
    return "|".join(sorted({value for value in values if value}))


def main() -> int:
    args = parse_args()
    project_root = resolve_project_root(args)

    manifest_path = (
        project_root
        / "data"
        / "manifests"
        / "isic2019_dataset_manifest.csv"
    )
    output_root = project_root / "reports" / "dataset_audits"

    audit_path = output_root / "isic2019_split_group_audit.json"
    duplicate_path = (
        output_root / "isic2019_exact_duplicate_hash_groups.csv"
    )
    conflict_path = (
        output_root / "isic2019_lesion_label_conflicts.csv"
    )
    missing_path = (
        output_root / "isic2019_missing_lesion_id_distribution.csv"
    )

    rows = read_manifest(manifest_path)
    rows_by_id = {
        row["image_id"].strip(): row
        for row in rows
    }

    hash_groups: dict[str, list[str]] = defaultdict(list)
    lesion_groups: dict[str, list[str]] = defaultdict(list)

    for row in rows:
        image_id = row["image_id"].strip()
        file_hash = row["file_sha256"].strip().lower()
        lesion_id = row["lesion_id"].strip()

        if not file_hash:
            raise ValueError(
                f"Missing file_sha256 for image: {image_id}"
            )

        hash_groups[file_hash].append(image_id)

        if lesion_id:
            lesion_groups[lesion_id].append(image_id)

    duplicate_hash_groups = {
        file_hash: image_ids
        for file_hash, image_ids in hash_groups.items()
        if len(image_ids) > 1
    }

    duplicate_rows: list[dict[str, Any]] = []
    cross_lesion_duplicate_group_count = 0
    cross_diagnosis_duplicate_group_count = 0

    for file_hash, image_ids in sorted(duplicate_hash_groups.items()):
        grouped_rows = [rows_by_id[image_id] for image_id in image_ids]
        lesion_ids = {
            row["lesion_id"].strip()
            for row in grouped_rows
            if row["lesion_id"].strip()
        }
        diagnoses = {
            row["diagnosis_original"].strip()
            for row in grouped_rows
        }

        cross_lesion = len(lesion_ids) > 1
        cross_diagnosis = len(diagnoses) > 1

        cross_lesion_duplicate_group_count += int(cross_lesion)
        cross_diagnosis_duplicate_group_count += int(cross_diagnosis)

        duplicate_rows.append(
            {
                "file_sha256": file_hash,
                "image_count": len(image_ids),
                "image_ids": joined(image_ids),
                "lesion_ids": joined(lesion_ids),
                "diagnoses": joined(diagnoses),
                "cross_lesion_id": int(cross_lesion),
                "cross_diagnosis": int(cross_diagnosis),
            }
        )

    conflict_rows: list[dict[str, Any]] = []

    for lesion_id, image_ids in sorted(lesion_groups.items()):
        grouped_rows = [rows_by_id[image_id] for image_id in image_ids]
        diagnoses = {
            row["diagnosis_original"].strip()
            for row in grouped_rows
        }
        stage_1_labels = {
            row["stage_1_label"].strip()
            for row in grouped_rows
            if row["stage_1_label"].strip()
        }
        stage_2_labels = {
            row["stage_2_label"].strip()
            for row in grouped_rows
            if row["stage_2_label"].strip()
        }

        if (
            len(diagnoses) > 1
            or len(stage_1_labels) > 1
            or len(stage_2_labels) > 1
        ):
            conflict_rows.append(
                {
                    "lesion_id": lesion_id,
                    "image_count": len(image_ids),
                    "image_ids": joined(image_ids),
                    "diagnoses": joined(diagnoses),
                    "stage_1_labels": joined(stage_1_labels),
                    "stage_2_labels": joined(stage_2_labels),
                }
            )

    missing_lesion_rows = [
        row
        for row in rows
        if not row["lesion_id"].strip()
    ]
    missing_distribution = Counter(
        row["diagnosis_original"].strip()
        for row in missing_lesion_rows
    )
    missing_distribution_rows = [
        {
            "diagnosis_original": diagnosis,
            "missing_lesion_id_count": count,
        }
        for diagnosis, count in sorted(missing_distribution.items())
    ]

    image_ids = sorted(rows_by_id)
    disjoint_set = DisjointSet(image_ids)

    for grouped_image_ids in lesion_groups.values():
        anchor = grouped_image_ids[0]
        for image_id in grouped_image_ids[1:]:
            disjoint_set.union(anchor, image_id)

    for grouped_image_ids in duplicate_hash_groups.values():
        anchor = grouped_image_ids[0]
        for image_id in grouped_image_ids[1:]:
            disjoint_set.union(anchor, image_id)

    component_members: dict[str, list[str]] = defaultdict(list)
    for image_id in image_ids:
        component_members[disjoint_set.find(image_id)].append(image_id)

    component_sizes = Counter(
        len(members)
        for members in component_members.values()
    )
    repeated_components = [
        members
        for members in component_members.values()
        if len(members) > 1
    ]

    cross_diagnosis_components = 0
    cross_stage_1_components = 0
    cross_stage_2_components = 0

    for members in component_members.values():
        grouped_rows = [rows_by_id[image_id] for image_id in members]
        diagnoses = {
            row["diagnosis_original"].strip()
            for row in grouped_rows
        }
        stage_1_labels = {
            row["stage_1_label"].strip()
            for row in grouped_rows
            if row["stage_1_label"].strip()
        }
        stage_2_labels = {
            row["stage_2_label"].strip()
            for row in grouped_rows
            if row["stage_2_label"].strip()
        }

        cross_diagnosis_components += int(len(diagnoses) > 1)
        cross_stage_1_components += int(len(stage_1_labels) > 1)
        cross_stage_2_components += int(len(stage_2_labels) > 1)

    audit_payload = {
        "dataset": "isic2019",
        "status": "complete",
        "manifest_rows": len(rows),
        "exact_duplicate_hash_audit": {
            "unique_hash_count": len(hash_groups),
            "duplicate_hash_group_count": len(
                duplicate_hash_groups
            ),
            "images_in_duplicate_hash_groups": sum(
                len(grouped_image_ids)
                for grouped_image_ids in duplicate_hash_groups.values()
            ),
            "cross_lesion_duplicate_group_count": (
                cross_lesion_duplicate_group_count
            ),
            "cross_diagnosis_duplicate_group_count": (
                cross_diagnosis_duplicate_group_count
            ),
        },
        "lesion_group_audit": {
            "non_empty_lesion_id_rows": sum(
                len(grouped_image_ids)
                for grouped_image_ids in lesion_groups.values()
            ),
            "unique_non_empty_lesion_ids": len(lesion_groups),
            "missing_lesion_id_rows": len(missing_lesion_rows),
            "lesion_label_conflict_group_count": len(conflict_rows),
        },
        "connected_component_grouping": {
            "strategy": (
                "connected_components_of_shared_lesion_id_"
                "or_exact_file_sha256"
            ),
            "component_count": len(component_members),
            "repeated_component_count": len(repeated_components),
            "images_in_repeated_components": sum(
                len(members)
                for members in repeated_components
            ),
            "maximum_component_size": max(
                (len(members) for members in component_members.values()),
                default=0,
            ),
            "component_size_distribution": {
                str(size): count
                for size, count in sorted(component_sizes.items())
            },
            "cross_diagnosis_component_count": (
                cross_diagnosis_components
            ),
            "cross_stage_1_label_component_count": (
                cross_stage_1_components
            ),
            "cross_stage_2_label_component_count": (
                cross_stage_2_components
            ),
        },
        "recommended_split_policy": {
            "primary_grouping": (
                "connected component based on lesion_id and exact "
                "image SHA-256"
            ),
            "missing_lesion_id_fallback": (
                "exact-hash component when duplicated; otherwise "
                "image-level singleton"
            ),
            "patient_level_split_possible": False,
            "limitation": (
                "ISIC 2019 metadata has no patient_id column. "
                "Lesion-aware grouping reduces known leakage but "
                "cannot guarantee patient-independent splits."
            ),
        },
        "outputs": {
            "duplicate_hash_groups": duplicate_path.relative_to(
                project_root
            ).as_posix(),
            "lesion_label_conflicts": conflict_path.relative_to(
                project_root
            ).as_posix(),
            "missing_lesion_distribution": missing_path.relative_to(
                project_root
            ).as_posix(),
            "audit": audit_path.relative_to(project_root).as_posix(),
        },
    }

    atomic_write_csv(
        duplicate_path,
        [
            "file_sha256",
            "image_count",
            "image_ids",
            "lesion_ids",
            "diagnoses",
            "cross_lesion_id",
            "cross_diagnosis",
        ],
        duplicate_rows,
    )
    atomic_write_csv(
        conflict_path,
        [
            "lesion_id",
            "image_count",
            "image_ids",
            "diagnoses",
            "stage_1_labels",
            "stage_2_labels",
        ],
        conflict_rows,
    )
    atomic_write_csv(
        missing_path,
        [
            "diagnosis_original",
            "missing_lesion_id_count",
        ],
        missing_distribution_rows,
    )
    atomic_write_json(audit_path, audit_payload)

    print()
    print("==========================================")
    print("ISIC 2019 split-group audit completed")
    print("==========================================")
    print(f"Manifest rows:                    {len(rows)}")
    print(
        "Duplicate hash groups:            "
        f"{len(duplicate_hash_groups)}"
    )
    print(
        "Cross-diagnosis duplicate groups: "
        f"{cross_diagnosis_duplicate_group_count}"
    )
    print(f"Lesion label conflicts:           {len(conflict_rows)}")
    print(f"Missing lesion IDs:               {len(missing_lesion_rows)}")
    print(f"Connected components:             {len(component_members)}")
    print(
        "Maximum connected component size: "
        f"{max((len(m) for m in component_members.values()), default=0)}"
    )
    print(f"Audit: {audit_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"[FAIL] {exc}")
        raise SystemExit(1) from exc
