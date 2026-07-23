#!/usr/bin/env python3
"""
Create deterministic leakage-aware ISIC 2019 train/validation/test splits.

Grouping constraints:
1. Images sharing a non-empty lesion_id stay in the same split.
2. Images sharing an exact file_sha256 stay in the same split.
3. The transitive closure of those relations forms one split group.
4. Any component containing conflicting diagnoses or hierarchy labels is
   excluded from the primary split.

The default split is 70/15/15 with seed 42. Assignment is performed at
connected-component level and stratified by original diagnosis.

Outputs:
- data/manifests/isic2019_split_groups_seed42.csv
- data/manifests/isic2019_train_val_test_split_seed42.csv
- reports/dataset_audits/isic2019_split_class_distribution_seed42.csv
- reports/dataset_audits/isic2019_split_audit_seed42.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

EXPECTED_COUNT = 25_331
PARTITIONS = ("train", "validation", "test")


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


@dataclass(frozen=True)
class SplitRatios:
    train: float
    validation: float
    test: float

    def as_dict(self) -> dict[str, float]:
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }


@dataclass
class Component:
    group_id: str
    image_ids: list[str]
    diagnosis: str
    stage_1_label: str
    stage_2_label: str
    lesion_ids: list[str]
    file_hashes: list[str]
    excluded: bool
    exclusion_reason: str

    @property
    def size(self) -> int:
        return len(self.image_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create deterministic group-stratified ISIC 2019 splits."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help=(
            "Project root. Defaults to the parent directory of the "
            "scripts directory."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    return parser.parse_args()


def resolve_project_root(args: argparse.Namespace) -> Path:
    if args.project_root is not None:
        return args.project_root.expanduser().resolve()
    return Path(__file__).resolve().parent.parent


def validate_ratios(args: argparse.Namespace) -> SplitRatios:
    ratios = SplitRatios(
        train=args.train_ratio,
        validation=args.validation_ratio,
        test=args.test_ratio,
    )

    values = ratios.as_dict()

    if any(value <= 0.0 for value in values.values()):
        raise ValueError("All split ratios must be greater than zero.")

    if not math.isclose(sum(values.values()), 1.0, abs_tol=1e-9):
        raise ValueError(
            "Split ratios must sum to 1.0. "
            f"Received: {sum(values.values()):.12f}"
        )

    return ratios


def read_manifest(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")

        headers = list(reader.fieldnames)
        required = {
            "image_id",
            "diagnosis_original",
            "stage_1_label",
            "stage_2_label",
            "include_stage_1",
            "include_stage_2",
            "lesion_id",
            "file_sha256",
        }
        missing = sorted(required - set(headers))

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
    duplicate_ids = [
        image_id
        for image_id, count in Counter(image_ids).items()
        if count > 1
    ]

    if duplicate_ids:
        raise ValueError(
            "Duplicate image IDs found. Examples: "
            + ", ".join(sorted(duplicate_ids)[:10])
        )

    return rows, headers


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
        temporary_path = Path(handle.name)
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)

    os.replace(temporary_path, path)


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
        temporary_path = Path(handle.name)
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    os.replace(temporary_path, path)


def stable_group_id(image_ids: Iterable[str]) -> str:
    joined_ids = "|".join(sorted(image_ids))
    digest = hashlib.sha256(joined_ids.encode("utf-8")).hexdigest()
    return f"isic2019_group_{digest[:16]}"


def stable_tie_key(seed: int, group_id: str) -> str:
    value = f"{seed}:{group_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def build_components(
    rows: list[dict[str, str]],
) -> tuple[list[Component], dict[str, str]]:
    rows_by_id = {
        row["image_id"].strip(): row
        for row in rows
    }
    image_ids = sorted(rows_by_id)

    lesion_groups: dict[str, list[str]] = defaultdict(list)
    hash_groups: dict[str, list[str]] = defaultdict(list)

    for row in rows:
        image_id = row["image_id"].strip()
        lesion_id = row["lesion_id"].strip()
        file_hash = row["file_sha256"].strip().lower()

        if not file_hash:
            raise ValueError(
                f"Missing file_sha256 for image: {image_id}"
            )

        if lesion_id:
            lesion_groups[lesion_id].append(image_id)

        hash_groups[file_hash].append(image_id)

    disjoint_set = DisjointSet(image_ids)

    for grouped_ids in lesion_groups.values():
        anchor = grouped_ids[0]
        for image_id in grouped_ids[1:]:
            disjoint_set.union(anchor, image_id)

    for grouped_ids in hash_groups.values():
        if len(grouped_ids) <= 1:
            continue
        anchor = grouped_ids[0]
        for image_id in grouped_ids[1:]:
            disjoint_set.union(anchor, image_id)

    members_by_root: dict[str, list[str]] = defaultdict(list)

    for image_id in image_ids:
        members_by_root[disjoint_set.find(image_id)].append(image_id)

    components: list[Component] = []
    image_to_group: dict[str, str] = {}

    for members in members_by_root.values():
        members = sorted(members)
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
        lesion_ids = sorted(
            {
                row["lesion_id"].strip()
                for row in grouped_rows
                if row["lesion_id"].strip()
            }
        )
        file_hashes = sorted(
            {
                row["file_sha256"].strip().lower()
                for row in grouped_rows
            }
        )

        conflict_reasons: list[str] = []

        if len(diagnoses) > 1:
            conflict_reasons.append("cross_diagnosis")
        if len(stage_1_labels) > 1:
            conflict_reasons.append("cross_stage_1_label")
        if len(stage_2_labels) > 1:
            conflict_reasons.append("cross_stage_2_label")

        excluded = bool(conflict_reasons)
        exclusion_reason = (
            "cross_diagnosis_exact_duplicate_component"
            if "cross_diagnosis" in conflict_reasons
            else "|".join(conflict_reasons)
        )

        group_id = stable_group_id(members)

        component = Component(
            group_id=group_id,
            image_ids=members,
            diagnosis=(
                next(iter(diagnoses))
                if len(diagnoses) == 1
                else "|".join(sorted(diagnoses))
            ),
            stage_1_label=(
                next(iter(stage_1_labels))
                if len(stage_1_labels) == 1
                else "|".join(sorted(stage_1_labels))
            ),
            stage_2_label=(
                next(iter(stage_2_labels))
                if len(stage_2_labels) == 1
                else "|".join(sorted(stage_2_labels))
            ),
            lesion_ids=lesion_ids,
            file_hashes=file_hashes,
            excluded=excluded,
            exclusion_reason=exclusion_reason,
        )
        components.append(component)

        for image_id in members:
            image_to_group[image_id] = group_id

    components.sort(key=lambda component: component.group_id)
    return components, image_to_group


def assignment_score(
    partition: str,
    component_size: int,
    diagnosis: str,
    *,
    ratios: dict[str, float],
    class_targets: dict[str, dict[str, float]],
    class_counts: dict[str, Counter[str]],
    total_targets: dict[str, float],
    total_counts: Counter[str],
) -> tuple[float, float, int]:
    class_score = 0.0

    for candidate_partition in PARTITIONS:
        candidate_count = class_counts[diagnosis][candidate_partition]

        if candidate_partition == partition:
            candidate_count += component_size

        target = class_targets[diagnosis][candidate_partition]
        denominator = max(target, 1.0)
        class_score += ((candidate_count - target) / denominator) ** 2

    global_score = 0.0

    for candidate_partition in PARTITIONS:
        candidate_count = total_counts[candidate_partition]

        if candidate_partition == partition:
            candidate_count += component_size

        target = total_targets[candidate_partition]
        denominator = max(target, 1.0)
        global_score += ((candidate_count - target) / denominator) ** 2

    overshoot = max(
        0.0,
        (
            class_counts[diagnosis][partition]
            + component_size
            - class_targets[diagnosis][partition]
        ),
    )
    overshoot_ratio = overshoot / max(
        class_targets[diagnosis][partition],
        1.0,
    )

    partition_order = {
        "train": 0,
        "validation": 1,
        "test": 2,
    }

    combined_score = (
        (10.0 * class_score)
        + global_score
        + (2.0 * overshoot_ratio)
    )

    return (
        combined_score,
        -ratios[partition],
        partition_order[partition],
    )


def assign_components(
    components: list[Component],
    ratios: SplitRatios,
    seed: int,
) -> tuple[dict[str, str], dict[str, Counter[str]], Counter[str]]:
    eligible_components = [
        component
        for component in components
        if not component.excluded
    ]

    diagnosis_totals = Counter()

    for component in eligible_components:
        diagnosis_totals[component.diagnosis] += component.size

    ratio_map = ratios.as_dict()

    class_targets = {
        diagnosis: {
            partition: total * ratio_map[partition]
            for partition in PARTITIONS
        }
        for diagnosis, total in diagnosis_totals.items()
    }

    total_eligible = sum(diagnosis_totals.values())
    total_targets = {
        partition: total_eligible * ratio_map[partition]
        for partition in PARTITIONS
    }

    class_counts = {
        diagnosis: Counter()
        for diagnosis in diagnosis_totals
    }
    total_counts: Counter[str] = Counter()
    assignments: dict[str, str] = {}

    components_by_diagnosis: dict[str, list[Component]] = defaultdict(list)

    for component in eligible_components:
        components_by_diagnosis[component.diagnosis].append(component)

    diagnosis_order = sorted(
        components_by_diagnosis,
        key=lambda diagnosis: (
            diagnosis_totals[diagnosis],
            diagnosis,
        ),
    )

    for diagnosis in diagnosis_order:
        diagnosis_components = sorted(
            components_by_diagnosis[diagnosis],
            key=lambda component: (
                -component.size,
                stable_tie_key(seed, component.group_id),
            ),
        )

        for component in diagnosis_components:
            selected_partition = min(
                PARTITIONS,
                key=lambda partition: assignment_score(
                    partition,
                    component.size,
                    diagnosis,
                    ratios=ratio_map,
                    class_targets=class_targets,
                    class_counts=class_counts,
                    total_targets=total_targets,
                    total_counts=total_counts,
                ),
            )

            assignments[component.group_id] = selected_partition
            class_counts[diagnosis][selected_partition] += component.size
            total_counts[selected_partition] += component.size

    return assignments, class_counts, total_counts


def ratio_error(
    actual: int,
    total: int,
    target_ratio: float,
) -> float:
    if total == 0:
        return 0.0
    return (actual / total) - target_ratio


def main() -> int:
    args = parse_args()
    ratios = validate_ratios(args)
    ratio_map = ratios.as_dict()
    project_root = resolve_project_root(args)

    input_manifest = (
        project_root
        / "data"
        / "manifests"
        / "isic2019_dataset_manifest.csv"
    )
    split_groups_path = (
        project_root
        / "data"
        / "manifests"
        / f"isic2019_split_groups_seed{args.seed}.csv"
    )
    split_manifest_path = (
        project_root
        / "data"
        / "manifests"
        / f"isic2019_train_val_test_split_seed{args.seed}.csv"
    )
    distribution_path = (
        project_root
        / "reports"
        / "dataset_audits"
        / f"isic2019_split_class_distribution_seed{args.seed}.csv"
    )
    audit_path = (
        project_root
        / "reports"
        / "dataset_audits"
        / f"isic2019_split_audit_seed{args.seed}.json"
    )

    rows, input_headers = read_manifest(input_manifest)
    rows_by_id = {
        row["image_id"].strip(): row
        for row in rows
    }

    components, image_to_group = build_components(rows)
    assignments, diagnosis_counts, total_counts = assign_components(
        components,
        ratios,
        args.seed,
    )

    component_by_group = {
        component.group_id: component
        for component in components
    }

    group_rows: list[dict[str, Any]] = []

    for component in components:
        partition = (
            "excluded"
            if component.excluded
            else assignments[component.group_id]
        )

        group_rows.append(
            {
                "split_group_id": component.group_id,
                "split": partition,
                "image_count": component.size,
                "diagnosis_original": component.diagnosis,
                "stage_1_label": component.stage_1_label,
                "stage_2_label": component.stage_2_label,
                "lesion_ids": "|".join(component.lesion_ids),
                "file_sha256_values": "|".join(component.file_hashes),
                "image_ids": "|".join(component.image_ids),
                "excluded": int(component.excluded),
                "exclusion_reason": component.exclusion_reason,
                "split_seed": args.seed,
                "split_ratio": (
                    f"{ratios.train:.6f}/"
                    f"{ratios.validation:.6f}/"
                    f"{ratios.test:.6f}"
                ),
            }
        )

    split_manifest_rows: list[dict[str, Any]] = []
    stage_1_counts_by_split: dict[str, Counter[str]] = {
        partition: Counter()
        for partition in (*PARTITIONS, "excluded")
    }
    stage_2_counts_by_split: dict[str, Counter[str]] = {
        partition: Counter()
        for partition in (*PARTITIONS, "excluded")
    }
    diagnosis_counts_by_split: dict[str, Counter[str]] = {
        partition: Counter()
        for partition in (*PARTITIONS, "excluded")
    }

    for row in rows:
        image_id = row["image_id"].strip()
        group_id = image_to_group[image_id]
        component = component_by_group[group_id]
        partition = (
            "excluded"
            if component.excluded
            else assignments[group_id]
        )

        diagnosis = row["diagnosis_original"].strip()
        stage_1_label = row["stage_1_label"].strip()
        stage_2_label = row["stage_2_label"].strip()

        diagnosis_counts_by_split[partition][diagnosis] += 1

        if row["include_stage_1"].strip() == "1":
            stage_1_counts_by_split[partition][stage_1_label] += 1

        if row["include_stage_2"].strip() == "1":
            stage_2_counts_by_split[partition][stage_2_label] += 1

        output_row = dict(row)
        output_row.update(
            {
                "split_group_id": group_id,
                "split": partition,
                "split_seed": args.seed,
                "split_ratio": (
                    f"{ratios.train:.6f}/"
                    f"{ratios.validation:.6f}/"
                    f"{ratios.test:.6f}"
                ),
                "split_included": int(not component.excluded),
                "split_exclusion_reason": (
                    component.exclusion_reason
                    if component.excluded
                    else ""
                ),
            }
        )
        split_manifest_rows.append(output_row)

    distribution_rows: list[dict[str, Any]] = []

    for level, counts_by_split in (
        ("diagnosis_original", diagnosis_counts_by_split),
        ("stage_1_label", stage_1_counts_by_split),
        ("stage_2_label", stage_2_counts_by_split),
    ):
        labels = sorted(
            {
                label
                for counter in counts_by_split.values()
                for label in counter
                if label
            }
        )

        for label in labels:
            eligible_total = sum(
                counts_by_split[partition][label]
                for partition in PARTITIONS
            )

            for partition in (*PARTITIONS, "excluded"):
                count = counts_by_split[partition][label]
                observed_ratio = (
                    count / eligible_total
                    if eligible_total > 0
                    and partition in PARTITIONS
                    else ""
                )
                target_ratio = (
                    ratio_map[partition]
                    if partition in PARTITIONS
                    else ""
                )
                distribution_rows.append(
                    {
                        "level": level,
                        "label": label,
                        "split": partition,
                        "count": count,
                        "eligible_total": eligible_total,
                        "observed_ratio": (
                            f"{observed_ratio:.8f}"
                            if observed_ratio != ""
                            else ""
                        ),
                        "target_ratio": (
                            f"{target_ratio:.8f}"
                            if target_ratio != ""
                            else ""
                        ),
                        "ratio_error": (
                            f"{ratio_error(count, eligible_total, target_ratio):.8f}"
                            if partition in PARTITIONS
                            and eligible_total > 0
                            else ""
                        ),
                    }
                )

    excluded_components = [
        component
        for component in components
        if component.excluded
    ]
    excluded_image_count = sum(
        component.size
        for component in excluded_components
    )

    lesion_to_splits: dict[str, set[str]] = defaultdict(set)
    hash_to_splits: dict[str, set[str]] = defaultdict(set)
    group_to_splits: dict[str, set[str]] = defaultdict(set)

    for row in split_manifest_rows:
        if row["split"] == "excluded":
            continue

        split_name = str(row["split"])
        group_to_splits[str(row["split_group_id"])].add(split_name)

        lesion_id = str(row["lesion_id"]).strip()
        if lesion_id:
            lesion_to_splits[lesion_id].add(split_name)

        file_hash = str(row["file_sha256"]).strip().lower()
        if file_hash:
            hash_to_splits[file_hash].add(split_name)

    group_leaks = {
        group_id: sorted(splits)
        for group_id, splits in group_to_splits.items()
        if len(splits) > 1
    }
    lesion_leaks = {
        lesion_id: sorted(splits)
        for lesion_id, splits in lesion_to_splits.items()
        if len(splits) > 1
    }
    hash_leaks = {
        file_hash: sorted(splits)
        for file_hash, splits in hash_to_splits.items()
        if len(splits) > 1
    }

    if group_leaks or lesion_leaks or hash_leaks:
        raise ValueError(
            "Leakage validation failed: "
            f"group_leaks={len(group_leaks)}, "
            f"lesion_leaks={len(lesion_leaks)}, "
            f"hash_leaks={len(hash_leaks)}"
        )

    split_totals = Counter(
        row["split"]
        for row in split_manifest_rows
    )

    original_diagnosis_totals = Counter(
        row["diagnosis_original"].strip()
        for row in rows
        if not component_by_group[
            image_to_group[row["image_id"].strip()]
        ].excluded
    )

    class_ratio_errors: dict[str, dict[str, float]] = {}

    for diagnosis, total in sorted(original_diagnosis_totals.items()):
        class_ratio_errors[diagnosis] = {}

        for partition in PARTITIONS:
            actual = diagnosis_counts_by_split[partition][diagnosis]
            class_ratio_errors[diagnosis][partition] = ratio_error(
                actual,
                total,
                ratio_map[partition],
            )

    audit_payload: dict[str, Any] = {
        "dataset": "isic2019",
        "status": "complete",
        "seed": args.seed,
        "requested_ratios": ratio_map,
        "grouping_policy": {
            "relations": [
                "shared_non_empty_lesion_id",
                "shared_exact_file_sha256",
            ],
            "transitive_closure": True,
            "patient_level_split_possible": False,
            "missing_lesion_id_fallback": (
                "exact-hash group when duplicated; otherwise singleton"
            ),
        },
        "component_counts": {
            "total_components": len(components),
            "assigned_components": (
                len(components) - len(excluded_components)
            ),
            "excluded_components": len(excluded_components),
            "excluded_images": excluded_image_count,
            "maximum_component_size": max(
                (component.size for component in components),
                default=0,
            ),
        },
        "exclusions": [
            {
                "split_group_id": component.group_id,
                "image_count": component.size,
                "image_ids": component.image_ids,
                "lesion_ids": component.lesion_ids,
                "diagnosis": component.diagnosis,
                "reason": component.exclusion_reason,
            }
            for component in excluded_components
        ],
        "split_counts_all_non_conflict_rows": {
            partition: split_totals[partition]
            for partition in PARTITIONS
        },
        "excluded_row_count": split_totals["excluded"],
        "primary_stage_1_counts": {
            partition: dict(
                sorted(stage_1_counts_by_split[partition].items())
            )
            for partition in PARTITIONS
        },
        "primary_stage_2_counts": {
            partition: dict(
                sorted(stage_2_counts_by_split[partition].items())
            )
            for partition in PARTITIONS
        },
        "class_ratio_errors_original_diagnosis": class_ratio_errors,
        "leakage_validation": {
            "split_group_overlap_count": len(group_leaks),
            "lesion_id_overlap_count": len(lesion_leaks),
            "exact_hash_overlap_count": len(hash_leaks),
            "passed": True,
        },
        "limitations": [
            (
                "ISIC 2019 metadata does not provide patient_id, so "
                "patient-independent splitting cannot be guaranteed."
            ),
            (
                "Rows with missing lesion_id are grouped by exact hash "
                "when duplicated and otherwise treated as singletons."
            ),
        ],
        "outputs": {
            "split_groups": split_groups_path.relative_to(
                project_root
            ).as_posix(),
            "split_manifest": split_manifest_path.relative_to(
                project_root
            ).as_posix(),
            "class_distribution": distribution_path.relative_to(
                project_root
            ).as_posix(),
            "audit": audit_path.relative_to(project_root).as_posix(),
        },
    }

    atomic_write_csv(
        split_groups_path,
        [
            "split_group_id",
            "split",
            "image_count",
            "diagnosis_original",
            "stage_1_label",
            "stage_2_label",
            "lesion_ids",
            "file_sha256_values",
            "image_ids",
            "excluded",
            "exclusion_reason",
            "split_seed",
            "split_ratio",
        ],
        group_rows,
    )
    atomic_write_csv(
        split_manifest_path,
        input_headers
        + [
            "split_group_id",
            "split",
            "split_seed",
            "split_ratio",
            "split_included",
            "split_exclusion_reason",
        ],
        split_manifest_rows,
    )
    atomic_write_csv(
        distribution_path,
        [
            "level",
            "label",
            "split",
            "count",
            "eligible_total",
            "observed_ratio",
            "target_ratio",
            "ratio_error",
        ],
        distribution_rows,
    )
    atomic_write_json(audit_path, audit_payload)

    print()
    print("===================================================")
    print("ISIC 2019 leakage-aware dataset split successfully")
    print("===================================================")
    print(f"Seed:                 {args.seed}")
    print(
        "Ratios:               "
        f"{ratios.train:.2f}/{ratios.validation:.2f}/{ratios.test:.2f}"
    )
    print(f"Total components:     {len(components)}")
    print(f"Excluded components:  {len(excluded_components)}")
    print(f"Excluded images:      {excluded_image_count}")
    print(f"Train rows:           {split_totals['train']}")
    print(f"Validation rows:      {split_totals['validation']}")
    print(f"Test rows:            {split_totals['test']}")
    print("Leakage checks:       PASSED")
    print(f"Split manifest:       {split_manifest_path}")
    print(f"Split audit:          {audit_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"[FAIL] {exc}")
        raise SystemExit(1) from exc
