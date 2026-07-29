#!/usr/bin/env python3
"""Build deterministic 70/15/15 official ISIC-derived Stage-3 splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from itertools import permutations
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


def assign_component_splits(
    selected: pd.DataFrame, seed: int
) -> dict[str, str]:
    groups = (
        selected.groupby("split_group_id", sort=True)
        .agg(t_category=("t_category", "first"), image_count=("image_id", "size"))
        .reset_index()
    )
    counts = groups["t_category"].value_counts()
    insufficient = {
        category: int(counts.get(category, 0))
        for category in T_CATEGORIES
        if counts.get(category, 0) < 3
    }
    if insufficient:
        raise ValueError(
            "Each T-category needs at least three connected components to preserve "
            f"all classes in train/validation/test; insufficient: {insufficient}"
        )

    assignment: dict[str, str] = {}
    split_names = tuple(SPLIT_RATIOS)
    for category in T_CATEGORIES:
        class_groups = groups.loc[groups["t_category"].eq(category)].copy()
        class_groups["tie_break"] = class_groups["split_group_id"].map(
            lambda value: hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()
        )
        class_groups = class_groups.sort_values(
            ["image_count", "tie_break", "split_group_id"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
        targets = {
            split: len(selected.loc[selected["t_category"].eq(category)]) * ratio
            for split, ratio in SPLIT_RATIOS.items()
        }
        totals = {split: 0 for split in split_names}

        first = class_groups.iloc[:3].to_dict("records")
        best_permutation = min(
            permutations(split_names),
            key=lambda order: (
                sum(
                    (
                        int(component["image_count"])
                        - targets[split]
                    )
                    ** 2
                    / max(targets[split], 1)
                    for component, split in zip(first, order)
                ),
                order,
            ),
        )
        for component, split in zip(first, best_permutation):
            group_id = str(component["split_group_id"])
            assignment[group_id] = split
            totals[split] += int(component["image_count"])

        for component in class_groups.iloc[3:].to_dict("records"):
            size = int(component["image_count"])
            split = min(
                split_names,
                key=lambda candidate: (
                    sum(
                        (
                            totals[name]
                            + (size if name == candidate else 0)
                            - targets[name]
                        )
                        ** 2
                        / max(targets[name], 1)
                        for name in split_names
                    ),
                    split_names.index(candidate),
                ),
            )
            assignment[str(component["split_group_id"])] = split
            totals[split] += size
    return assignment


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
    component_rows = manifest.drop_duplicates("split_group_id")
    split_counts = manifest["split"].value_counts().reindex(SPLIT_RATIOS, fill_value=0)
    total = len(manifest)
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
        "multi_image_component_count": int(
            (manifest.groupby("split_group_id").size() > 1).sum()
        ),
        "maximum_component_size": int(manifest.groupby("split_group_id").size().max()),
        "component_counts_by_class": {
            key: int(value)
            for key, value in component_rows["t_category"]
            .value_counts().reindex(T_CATEGORIES, fill_value=0).items()
        },
        "image_counts_by_split_and_class": manifest.groupby(
            ["split", "t_category"]
        ).size().unstack(fill_value=0).reindex(
            index=SPLIT_RATIOS, columns=T_CATEGORIES, fill_value=0
        ).astype(int).to_dict("index"),
        "component_counts_by_split_and_class": component_rows.groupby(
            ["split", "t_category"]
        ).size().unstack(fill_value=0).reindex(
            index=SPLIT_RATIOS, columns=T_CATEGORIES, fill_value=0
        ).astype(int).to_dict("index"),
        "requested_ratios": SPLIT_RATIOS,
        "ratios": SPLIT_RATIOS,
        "counts": {key: int(value) for key, value in split_counts.items()},
        "stage_counts": manifest.groupby(
            ["split", "t_category"]
        ).size().unstack(fill_value=0).reindex(
            index=SPLIT_RATIOS, columns=T_CATEGORIES, fill_value=0
        ).astype(int).to_dict("index"),
        "actual_image_ratios": {
            split: float(count / total) for split, count in split_counts.items()
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
    limitations = [
        "Patient/lesion metadata is incomplete; all known patient and lesion "
        "relations are nevertheless preserved through connected components."
    ]
    selected = add_transitive_group_ids(selected)
    groups = selected.groupby("split_group_id", sort=True)
    for component_id, component in groups:
        labels = sorted(component["t_category"].astype(str).str.strip().unique())
        if len(labels) != 1:
            image_ids = sorted(component["image_id"].astype(str))
            raise ValueError(
                f"Conflicting T-categories in component {component_id}: "
                f"image_ids={image_ids}, labels={labels}"
            )
    assignment = assign_component_splits(selected, seed)
    selected["split"] = selected["split_group_id"].map(assignment)
    for field in ("image_id", "patient_id", "lesion_id", "file_sha256", "split_group_id"):
        if field in selected and cross_split_overlap_count(selected, field):
            raise ValueError(f"{field} overlaps across splits.")
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
