#!/usr/bin/env python3
"""Audit local storage for recoverable ITT validation evidence without using test/HIBA results.

This script scans only user-supplied roots. It looks for:
1) checkpoint files whose SHA-256 matches the seven locked Flat checkpoints;
2) CSV files whose names suggest validation predictions.

It does not load models, run inference, or read ISIC internal-test/HIBA predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


CHECKPOINT_EXTENSIONS = {".pt", ".pth", ".ckpt"}
VALIDATION_NAME_TOKENS = ("val", "validation")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def looks_like_validation_prediction(path: Path) -> bool:
    name = path.name.lower()
    return path.suffix.lower() == ".csv" and any(token in name for token in VALIDATION_NAME_TOKENS) and "pred" in name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="+",
        type=Path,
        help="Local directories to scan, e.g. project root, old phase02 copy, backup folder.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/extensions/itt2026_consensus/checkpoint_recovery_manifest.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/extensions/itt2026_consensus/itt03_validation_recovery_audit.json"),
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    models = config["models"]
    expected_by_hash = {
        item["sha256"].lower(): {
            "model": model,
            "expected_epoch": int(item["expected_epoch"]),
            "original_path": item["original_path"],
        }
        for model, item in models.items()
    }

    roots = [root.resolve() for root in args.roots]
    missing_roots = [str(root) for root in roots if not root.exists()]
    if missing_roots:
        raise SystemExit(f"Missing scan roots: {missing_roots}")

    matches: dict[str, list[dict[str, object]]] = {model: [] for model in models}
    validation_candidates: list[str] = []
    scanned_checkpoint_files = 0
    scanned_csv_files = 0
    hash_errors: list[dict[str, str]] = []

    seen_paths: set[Path] = set()
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)

            suffix = path.suffix.lower()
            if suffix in CHECKPOINT_EXTENSIONS:
                scanned_checkpoint_files += 1
                try:
                    digest = sha256_file(path).lower()
                except (OSError, PermissionError) as exc:
                    hash_errors.append({"path": str(path), "error": str(exc)})
                    continue
                if digest in expected_by_hash:
                    expected = expected_by_hash[digest]
                    matches[expected["model"]].append(
                        {
                            "path": str(path),
                            "sha256": digest,
                            "expected_epoch": expected["expected_epoch"],
                            "original_vm_path": expected["original_path"],
                        }
                    )
            elif suffix == ".csv":
                scanned_csv_files += 1
                if looks_like_validation_prediction(path):
                    validation_candidates.append(str(path))

    recovered_models = [model for model, found in matches.items() if found]
    result = {
        "status": "PASS" if len(recovered_models) == len(models) else "INCOMPLETE",
        "scope": "local_recovery_audit_only",
        "roots": [str(root) for root in roots],
        "expected_checkpoint_count": len(models),
        "recovered_checkpoint_count": len(recovered_models),
        "recovered_models": recovered_models,
        "missing_models": [model for model in models if not matches[model]],
        "checkpoint_matches": matches,
        "validation_prediction_candidates": sorted(validation_candidates),
        "scanned_checkpoint_files": scanned_checkpoint_files,
        "scanned_csv_files": scanned_csv_files,
        "hash_errors": hash_errors,
        "note": (
            "No model inference or test/HIBA ensemble computation was performed. "
            "Validation CSV candidates are filename candidates only and require schema/provenance verification."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
