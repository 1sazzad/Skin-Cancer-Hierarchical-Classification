#!/usr/bin/env python3
"""Build deterministic 70/15/15 dermoscopic EMB splits on the VM."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.emb_stage03 import inverse_frequency_class_weights, map_stage_ajcc


def build_split(frame: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, dict[str, float], list[str]]:
    required = {"image_id", "image_path", "stage_ajcc", "modality", "file_sha256"}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"Audit inventory is missing columns: {missing}")
    selected = frame.loc[
        frame["modality"].astype(str).str.strip().str.lower() == "dermoscopic"
    ].copy()
    if selected.empty:
        raise ValueError("No dermoscopic rows available.")
    selected["t_category"] = selected["stage_ajcc"].map(map_stage_ajcc)
    group_candidates = [name for name in ("patient_id", "lesion_id") if name in selected and selected[name].astype(str).str.strip().ne("").all()]
    limitations: list[str] = []
    if group_candidates:
        group_col = group_candidates[0]
        groups = selected.groupby(group_col, sort=True)
        if groups["t_category"].nunique().max() > 1:
            raise ValueError(f"Conflicting stages within {group_col}.")
        group_frame = groups.first().reset_index()
        train_groups, rest_groups = train_test_split(
            group_frame, test_size=0.30, random_state=seed, stratify=group_frame["t_category"]
        )
        val_groups, test_groups = train_test_split(
            rest_groups, test_size=0.50, random_state=seed, stratify=rest_groups["t_category"]
        )
        assignment = {
            **{value: "train" for value in train_groups[group_col]},
            **{value: "validation" for value in val_groups[group_col]},
            **{value: "test" for value in test_groups[group_col]},
        }
        selected["split"] = selected[group_col].map(assignment)
        selected["split_group_id"] = selected[group_col].map(lambda value: f"{group_col}:{value}")
    else:
        limitations.append("No complete valid patient/lesion grouping identifier; split uses image/hash groups.")
        if selected["file_sha256"].duplicated().any():
            conflicts = selected.groupby("file_sha256")["t_category"].nunique()
            if (conflicts > 1).any():
                raise ValueError("Conflicting labels in exact duplicate groups.")
            deduplicated = selected.drop_duplicates("file_sha256").copy()
        else:
            deduplicated = selected
        train, rest = train_test_split(
            deduplicated, test_size=0.30, random_state=seed, stratify=deduplicated["t_category"]
        )
        validation, test = train_test_split(
            rest, test_size=0.50, random_state=seed, stratify=rest["t_category"]
        )
        hash_split = {
            **{value: "train" for value in train["file_sha256"]},
            **{value: "validation" for value in validation["file_sha256"]},
            **{value: "test" for value in test["file_sha256"]},
        }
        selected["split"] = selected["file_sha256"].map(hash_split)
        selected["split_group_id"] = selected["file_sha256"].map(lambda value: f"sha256:{value}")
    for field in ("image_id", "file_sha256", "split_group_id"):
        if selected.groupby(field)["split"].nunique().max() > 1:
            raise ValueError(f"{field} overlaps across splits.")
    selected["dataset"] = "emb"
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
    if not args.allow_non_vm_fixture and (os.name != "posix" or not Path("/proc/driver/nvidia").exists()):
        raise SystemExit("NO-GO: real EMB split generation is Azure GPU VM-only.")
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
