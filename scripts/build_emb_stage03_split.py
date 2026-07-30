#!/usr/bin/env python3
"""Build deterministic 70/15/15 official ISIC-derived Stage-3 splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.emb_stage03 import inverse_frequency_class_weights, map_stage_ajcc

SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
T_CATEGORIES = ("Tis", "T1", "T2", "T3", "T4")
GROUPING_STRATEGY = "patient_id + lesion_id + file_sha256 connected components"


def add_transitive_group_ids(selected: pd.DataFrame) -> pd.DataFrame:
    """Build connected components across every known identity and exact hash."""

    selected = selected.reset_index(drop=True)
    parent = list(range(len(selected)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for name in ("patient_id", "lesion_id", "file_sha256"):
        if name not in selected:
            continue
        values = selected[name].fillna("").astype(str).str.strip()
        for identifier in sorted(value for value in values.unique() if value):
            indices = values.index[values.eq(identifier)].tolist()
            members = list(indices)
            for member in members[1:]:
                union(members[0], member)
    component_members: dict[int, list[str]] = {}
    for position, image_id in enumerate(selected["image_id"]):
        component_members.setdefault(find(position), []).append(str(image_id))
    identifiers = {
        root: "isic_stage03_group_"
        + hashlib.sha256("|".join(sorted(members)).encode()).hexdigest()[:16]
        for root, members in component_members.items()
    }
    result = selected.reset_index(drop=True).copy()
    result["split_group_id"] = [identifiers[find(index)] for index in range(len(result))]
    return result


def validate_relation_labels(selected: pd.DataFrame, field: str, label: str) -> None:
    values = selected[field].fillna("").astype(str).str.strip()
    for identifier in sorted(value for value in values.unique() if value):
        group = selected.loc[values.eq(identifier)]
        labels = sorted(group["t_category"].astype(str).str.strip().unique())
        if len(labels) > 1:
            image_ids = sorted(group["image_id"].astype(str))
            raise ValueError(
                f"Conflicting T-categories for {label} {identifier}: "
                f"image_ids={image_ids}, labels={labels}"
            )


def component_frame(selected: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for group_id, group in selected.groupby("split_group_id", sort=True):
        labels = sorted(
            group["t_category"].astype(str).str.strip().unique(),
            key=T_CATEGORIES.index,
        )
        record: dict[str, object] = {
            "split_group_id": str(group_id),
            "image_count": int(len(group)),
            "image_ids": sorted(group["image_id"].astype(str)),
            "patient_ids": sorted(
                value
                for value in group.get(
                    "patient_id", pd.Series("", index=group.index)
                ).fillna("").astype(str).str.strip().unique()
                if value
            ),
            "lesion_ids": sorted(
                value
                for value in group.get(
                    "lesion_id", pd.Series("", index=group.index)
                ).fillna("").astype(str).str.strip().unique()
                if value
            ),
            "class_count_vector": [
                int(group["t_category"].astype(str).str.strip().eq(category).sum())
                for category in T_CATEGORIES
            ],
            "labels_present": labels,
            "is_multi_label": len(labels) > 1,
        }
        if "split" in group:
            record["split"] = str(group["split"].iloc[0])
        records.append(record)
    return pd.DataFrame(records)


def allocation_objective(
    totals: dict[str, int],
    class_totals: dict[str, list[int]],
    target_total: dict[str, float],
    target_class: dict[str, list[float]],
    global_class_counts: list[int],
) -> float:
    score = sum(
        (totals[split] - target_total[split]) ** 2
        / max(target_total[split], 1.0)
        for split in SPLIT_RATIOS
    )
    largest_class = max(global_class_counts)
    for split in SPLIT_RATIOS:
        for index, count in enumerate(global_class_counts):
            rare_weight = (largest_class / count) ** 0.5
            target = target_class[split][index]
            score += (
                rare_weight
                * (class_totals[split][index] - target) ** 2
                / max(target, 1.0)
            )
            if class_totals[split][index] == 0:
                score += 1_000_000.0
    return score


def assign_component_splits(
    selected: pd.DataFrame, seed: int
) -> dict[str, str]:
    components = component_frame(selected)
    class_component_counts = [
        int(components["class_count_vector"].map(lambda vector: vector[index] > 0).sum())
        for index in range(len(T_CATEGORIES))
    ]
    insufficient = {
        category: class_component_counts[index]
        for index, category in enumerate(T_CATEGORIES)
        if class_component_counts[index] < 3
    }
    if insufficient:
        raise ValueError(
            "Three-way grouped class coverage is impossible: each T-category "
            "must occur in at least three distinct connected components; "
            f"insufficient: {insufficient}"
        )

    split_names = tuple(SPLIT_RATIOS)
    global_class_counts = [
        int(selected["t_category"].astype(str).str.strip().eq(category).sum())
        for category in T_CATEGORIES
    ]
    target_total = {
        split: len(selected) * ratio for split, ratio in SPLIT_RATIOS.items()
    }
    target_class = {
        split: [count * ratio for count in global_class_counts]
        for split, ratio in SPLIT_RATIOS.items()
    }
    best: tuple[float, tuple[tuple[str, str], ...], dict[str, str]] | None = None

    for restart in range(32):
        ordered = components.copy()
        ordered["rarity"] = ordered["class_count_vector"].map(
            lambda vector: sum(
                value / global_class_counts[index]
                for index, value in enumerate(vector)
                if value
            )
        )
        ordered["tie_break"] = ordered["split_group_id"].map(
            lambda value: hashlib.sha256(
                f"{seed}|{restart}|{value}".encode()
            ).hexdigest()
        )
        ordered = ordered.sort_values(
            ["rarity", "image_count", "tie_break", "split_group_id"],
            ascending=[False, False, True, True],
        )
        totals = {split: 0 for split in split_names}
        class_totals = {
            split: [0] * len(T_CATEGORIES) for split in split_names
        }
        assignment: dict[str, str] = {}
        for component in ordered.to_dict("records"):
            vector = [int(value) for value in component["class_count_vector"]]
            size = int(component["image_count"])
            split = min(
                split_names,
                key=lambda candidate: (
                    allocation_objective(
                        {
                            name: totals[name] + (size if name == candidate else 0)
                            for name in split_names
                        },
                        {
                            name: [
                                class_totals[name][index]
                                + (vector[index] if name == candidate else 0)
                                for index in range(len(T_CATEGORIES))
                            ]
                            for name in split_names
                        },
                        target_total,
                        target_class,
                        global_class_counts,
                    ),
                    split_names.index(candidate),
                ),
            )
            assignment[str(component["split_group_id"])] = split
            totals[split] += size
            class_totals[split] = [
                class_totals[split][index] + vector[index]
                for index in range(len(T_CATEGORIES))
            ]
        score = allocation_objective(
            totals,
            class_totals,
            target_total,
            target_class,
            global_class_counts,
        )
        signature = tuple(sorted(assignment.items()))
        candidate = (score, signature, assignment)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    assert best is not None
    return best[2]


def cross_split_overlap_count(frame: pd.DataFrame, field: str) -> int:
    values = frame[field].fillna("").astype(str).str.strip()
    nonempty = frame.loc[values.ne("")].copy()
    if nonempty.empty:
        return 0
    return int((nonempty.groupby(field)["split"].nunique() > 1).sum())


def split_audit(
    manifest: pd.DataFrame,
    weights: dict[str, float],
    limitations: list[str],
    manifest_sha256: str,
    seed: int = 42,
) -> dict[str, object]:
    components = component_frame(manifest)
    split_counts = manifest["split"].value_counts().reindex(SPLIT_RATIOS, fill_value=0)
    total = len(manifest)
    image_counts = manifest.groupby(
        ["split", "t_category"]
    ).size().unstack(fill_value=0).reindex(
        index=SPLIT_RATIOS, columns=T_CATEGORIES, fill_value=0
    ).astype(int)
    class_counts = manifest["t_category"].value_counts().reindex(
        T_CATEGORIES, fill_value=0
    )
    class_component_counts = {
        category: int(
            components["class_count_vector"].map(lambda vector: vector[index] > 0).sum()
        )
        for index, category in enumerate(T_CATEGORIES)
    }
    components_containing_by_split = {
        split: {
            category: int(
                components.loc[components["split"].eq(split), "class_count_vector"]
                .map(lambda vector: vector[index] > 0)
                .sum()
            )
            for index, category in enumerate(T_CATEGORIES)
        }
        for split in SPLIT_RATIOS
    }
    return {
        "seed": seed,
        "grouping_strategy": GROUPING_STRATEGY,
        "eligible_image_count": total,
        "patient_id_available_count": int(
            manifest.get("patient_id", pd.Series("", index=manifest.index))
            .fillna("").astype(str).str.strip().ne("").sum()
        ),
        "lesion_id_available_count": int(
            manifest.get("lesion_id", pd.Series("", index=manifest.index))
            .fillna("").astype(str).str.strip().ne("").sum()
        ),
        "connected_component_count": int(manifest["split_group_id"].nunique()),
        "single_label_component_count": int((~components["is_multi_label"]).sum()),
        "multi_label_component_count": int(components["is_multi_label"].sum()),
        "maximum_labels_in_component": int(
            components["labels_present"].map(len).max()
        ),
        "multi_image_component_count": int(
            (manifest.groupby("split_group_id").size() > 1).sum()
        ),
        "maximum_component_size": int(manifest.groupby("split_group_id").size().max()),
        "class_component_counts": class_component_counts,
        "component_counts_by_class": class_component_counts,
        "component_label_set_counts": {
            key: int(value)
            for key, value in components["labels_present"]
            .map(lambda labels: "|".join(labels))
            .value_counts().sort_index().items()
        },
        "components": components[
            [
                "split_group_id",
                "split",
                "image_count",
                "image_ids",
                "patient_ids",
                "lesion_ids",
                "class_count_vector",
                "labels_present",
                "is_multi_label",
            ]
        ].to_dict("records"),
        "image_counts_by_split_and_class": image_counts.to_dict("index"),
        "component_counts_by_split": {
            key: int(value)
            for key, value in components["split"].value_counts()
            .reindex(SPLIT_RATIOS, fill_value=0).items()
        },
        "components_containing_each_class_by_split": components_containing_by_split,
        "component_counts_by_split_and_class": components_containing_by_split,
        "requested_ratios": SPLIT_RATIOS,
        "requested_image_ratios": SPLIT_RATIOS,
        "ratios": SPLIT_RATIOS,
        "counts": {key: int(value) for key, value in split_counts.items()},
        "stage_counts": image_counts.to_dict("index"),
        "actual_image_ratios": {
            split: float(count / total) for split, count in split_counts.items()
        },
        "actual_overall_image_ratios": {
            split: float(count / total) for split, count in split_counts.items()
        },
        "actual_per_class_image_ratios": {
            category: {
                split: float(image_counts.loc[split, category] / class_counts[category])
                for split in SPLIT_RATIOS
            }
            for category in T_CATEGORIES
        },
        "patient_id_overlap_count": cross_split_overlap_count(manifest, "patient_id")
        if "patient_id" in manifest else 0,
        "lesion_id_overlap_count": cross_split_overlap_count(manifest, "lesion_id")
        if "lesion_id" in manifest else 0,
        "sha256_overlap_count": cross_split_overlap_count(manifest, "file_sha256"),
        "split_group_overlap_count": cross_split_overlap_count(
            manifest, "split_group_id"
        ),
        "train_only_class_weights": weights,
        "manifest_sha256": manifest_sha256,
        "limitations": limitations,
    }


def build_split(frame: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, dict[str, float], list[str]]:
    required = {
        "image_id", "image_path", "derived_stage_ajcc", "t_category",
        "modality", "file_sha256", "eligible",
    }
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"Audit inventory is missing columns: {missing}")
    selected = frame.loc[
        frame["eligible"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes"})
        & (
            frame["modality"].astype(str).str.strip().str.lower()
            == "dermoscopic"
        )
    ].copy()
    if selected.empty:
        raise ValueError("No dermoscopic rows available.")
    if selected["file_sha256"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("Every eligible image must have a SHA-256 before splitting.")
    mapped = selected["derived_stage_ajcc"].map(map_stage_ajcc)
    if not mapped.equals(selected["t_category"].astype(str).str.strip()):
        raise ValueError("derived_stage_ajcc disagrees with t_category.")
    if "lesion_id" in selected:
        validate_relation_labels(selected, "lesion_id", "lesion_id")
    validate_relation_labels(selected, "file_sha256", "SHA-256")
    limitations = [
        "Patient/lesion metadata is incomplete; all known patient and lesion "
        "relations are nevertheless preserved through connected components.",
        "Patient-safe components may contain multiple distinct lesions and "
        "T-categories; they are assigned intact using five-class count vectors.",
    ]
    selected = add_transitive_group_ids(selected)
    assignment = assign_component_splits(selected, seed)
    selected["split"] = selected["split_group_id"].map(assignment)
    for field in ("image_id", "patient_id", "lesion_id", "file_sha256", "split_group_id"):
        if field in selected and cross_split_overlap_count(selected, field):
            raise ValueError(f"{field} overlaps across splits.")
    missing_class_splits = {
        split: sorted(set(T_CATEGORIES) - set(group["t_category"].astype(str)))
        for split, group in selected.groupby("split")
        if set(group["t_category"].astype(str)) != set(T_CATEGORIES)
    }
    if missing_class_splits:
        raise ValueError(
            "Grouped stratification failed required three-way class coverage: "
            f"{missing_class_splits}"
        )
    selected["dataset"] = "isic_stage03"
    selected["split_seed"] = seed
    weights = inverse_frequency_class_weights(
        selected.loc[selected["split"] == "train", "t_category"].tolist()
    )
    return selected.sort_values(["split", "image_id"]).reset_index(drop=True), weights, limitations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-non-vm-fixture", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.allow_non_vm_fixture and (
        os.name != "posix"
        or not (Path("/proc/driver/nvidia").exists() or shutil.which("nvidia-smi"))
    ):
        raise SystemExit(
            "NO-GO: real ISIC Stage-3 split generation is Azure GPU VM-only."
        )
    root = args.project_root.resolve()
    output = args.output or root / "data/manifests/emb_stage03_dermoscopic_split_seed42.csv"
    manifest, weights, limitations = build_split(pd.read_csv(args.input), args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    audit = split_audit(
        manifest,
        weights,
        limitations,
        hashlib.sha256(output.read_bytes()).hexdigest(),
        args.seed,
    )
    output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print("GO", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
