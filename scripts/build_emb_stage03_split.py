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
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.emb_stage03 import inverse_frequency_class_weights, map_stage_ajcc


def add_transitive_group_ids(selected: pd.DataFrame, identity_column: str | None) -> pd.DataFrame:
    """Keep identity groups and exact hashes together through transitive closure."""

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

    grouped_columns = ["file_sha256"]
    if identity_column:
        grouped_columns.insert(0, identity_column)
    for name in grouped_columns:
        for indices in selected.groupby(name, sort=True).indices.values():
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
    if selected["file_sha256"].astype(str).str.strip().eq("").any():
        raise ValueError("Every eligible image must have a SHA-256 before splitting.")
    mapped = selected["derived_stage_ajcc"].map(map_stage_ajcc)
    if not mapped.equals(selected["t_category"].astype(str).str.strip()):
        raise ValueError("derived_stage_ajcc disagrees with t_category.")
    group_candidates = [
        name for name in ("patient_id", "lesion_id")
        if name in selected and selected[name].astype(str).str.strip().ne("").all()
    ]
    limitations: list[str] = []
    identity_column = group_candidates[0] if group_candidates else None
    if not identity_column:
        limitations.append("No complete valid patient/lesion grouping identifier; split uses image/hash groups.")
    selected = add_transitive_group_ids(selected, identity_column)
    groups = selected.groupby("split_group_id", sort=True)
    if groups["t_category"].nunique().max() > 1:
        raise ValueError("Conflicting stages within a patient/lesion/hash component.")
    group_frame = groups.first().reset_index()
    train_groups, rest_groups = train_test_split(
        group_frame, test_size=0.30, random_state=seed,
        stratify=group_frame["t_category"],
    )
    val_groups, test_groups = train_test_split(
        rest_groups, test_size=0.50, random_state=seed,
        stratify=rest_groups["t_category"],
    )
    assignment = {
        **{value: "train" for value in train_groups["split_group_id"]},
        **{value: "validation" for value in val_groups["split_group_id"]},
        **{value: "test" for value in test_groups["split_group_id"]},
    }
    selected["split"] = selected["split_group_id"].map(assignment)
    for field in ("image_id", "file_sha256", "split_group_id"):
        if selected.groupby(field)["split"].nunique().max() > 1:
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
    audit = {
        "seed": args.seed, "ratios": {"train": .70, "validation": .15, "test": .15},
        "counts": manifest["split"].value_counts().to_dict(),
        "stage_counts": manifest.groupby(["split", "t_category"]).size().unstack(fill_value=0).to_dict("index"),
        "train_only_class_weights": weights, "limitations": limitations,
        "manifest_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print("GO", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
