"""Generate publication-ready CoMeSySo Hard-vs-Fusion confusion matrices.

The script reads the frozen seed-level metric artifacts already committed for the
CoMeSySo 2026 probability-fusion experiment. It does not recompute predictions
or pool seed-image pairs.

For each dataset/system/seed:
1. load the stored 4x4 confusion matrix,
2. row-normalize it independently,
3. summarize each cell across seeds 42, 123, and 2026 using mean and sample SD.

Outputs:
- fig4_confusion_matrices.png
- fig4_confusion_matrices.pdf
- fig4_confusion_matrices_summary.json

Default execution from the repository root:
    python scripts/generate_comesyso_confusion_matrices.py

Optional:
    python scripts/generate_comesyso_confusion_matrices.py --annotate-sd
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


SEEDS = (42, 123, 2026)
CLASS_NAMES = ("NM", "MEL", "BCC", "SCC")
SYSTEMS = (
    ("hard", "Hard Routing"),
    ("fusion", "Probability Fusion"),
)

DATASETS = {
    "isic": {
        "display_name": "ISIC internal test",
        "path_template": (
            "results/extensions/comesyso2026_probability_fusion/"
            "internal_isic/seed{seed}/metrics_and_statistics.json"
        ),
    },
    "hiba": {
        "display_name": "HIBA zero-shot",
        "path_template": (
            "results/extensions/comesyso2026_probability_fusion/"
            "external_hiba/analysis/seed{seed}/metrics_and_statistics.json"
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate row-normalized Hard Routing vs Probability Fusion "
            "confusion matrices from frozen CoMeSySo result artifacts."
        )
    )
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=default_root,
        help="Repository root. Defaults to the parent of the scripts directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to <repo-root>/figures. "
            "The directory is created if missing."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution in dots per inch (default: 300).",
    )
    parser.add_argument(
        "--annotate-sd",
        action="store_true",
        help="Show mean ± sample SD in each matrix cell instead of mean only.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_confusion_matrix(
    matrix: np.ndarray,
    *,
    source: Path,
    system_key: str,
    class_names: list[str],
    labels: list[int],
) -> None:
    if matrix.shape != (4, 4):
        raise ValueError(
            f"{source}: system={system_key} expected a 4x4 confusion matrix, "
            f"got {matrix.shape}"
        )
    if class_names != ["non_malignant", "melanoma", "bcc", "scc"]:
        raise ValueError(
            f"{source}: system={system_key} unexpected class_names={class_names}"
        )
    if labels != [0, 1, 2, 3]:
        raise ValueError(
            f"{source}: system={system_key} unexpected labels={labels}"
        )
    if np.any(matrix < 0):
        raise ValueError(
            f"{source}: system={system_key} confusion matrix contains "
            "negative counts"
        )
    if np.any(matrix.sum(axis=1) == 0):
        raise ValueError(
            f"{source}: system={system_key} contains an empty true-class row"
        )


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    """Convert a count matrix to row-normalized percentages."""
    row_totals = matrix.sum(axis=1, keepdims=True)
    return (matrix / row_totals) * 100.0


def collect_summaries(repo_root: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "experiment": "comesyso2026_probability_fusion",
        "seeds": list(SEEDS),
        "class_order": list(CLASS_NAMES),
        "normalization": (
            "Each seed-level count matrix is row-normalized independently; "
            "cell means and sample SDs (ddof=1) are then computed across seeds."
        ),
        "seed_image_pair_pooling": False,
        "datasets": {},
        "sources": {},
    }

    for dataset_key, dataset_cfg in DATASETS.items():
        dataset_result: dict[str, Any] = {
            "display_name": dataset_cfg["display_name"],
            "systems": {},
        }
        source_records: dict[str, Any] = {}

        per_system: dict[str, list[np.ndarray]] = {
            system_key: [] for system_key, _ in SYSTEMS
        }

        for seed in SEEDS:
            rel_path = Path(dataset_cfg["path_template"].format(seed=seed))
            source_path = repo_root / rel_path
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"Required frozen result artifact not found: {source_path}"
                )

            payload = load_json(source_path)
            if int(payload.get("seed", -1)) != seed:
                raise ValueError(
                    f"{source_path}: expected seed {seed}, "
                    f"found {payload.get('seed')}"
                )

            source_records[str(seed)] = {
                "path": rel_path.as_posix(),
                "sha256": sha256_file(source_path),
            }

            systems_payload = payload.get("systems")
            if not isinstance(systems_payload, dict):
                raise ValueError(f"{source_path}: missing systems mapping")

            for system_key, _ in SYSTEMS:
                system_payload = systems_payload.get(system_key)
                if not isinstance(system_payload, dict):
                    raise ValueError(
                        f"{source_path}: missing system '{system_key}'"
                    )

                matrix = np.asarray(
                    system_payload.get("confusion_matrix"),
                    dtype=np.float64,
                )
                class_names = system_payload.get("class_names")
                labels = system_payload.get("labels")

                validate_confusion_matrix(
                    matrix,
                    source=source_path,
                    system_key=system_key,
                    class_names=class_names,
                    labels=labels,
                )
                per_system[system_key].append(row_normalize(matrix))

        for system_key, system_display in SYSTEMS:
            stack = np.stack(per_system[system_key], axis=0)
            mean = stack.mean(axis=0)
            sd = stack.std(axis=0, ddof=1)

            dataset_result["systems"][system_key] = {
                "display_name": system_display,
                "mean_row_normalized_percent": mean.tolist(),
                "sample_sd_row_normalized_percent": sd.tolist(),
                "individual_seed_row_normalized_percent": {
                    str(seed): per_system[system_key][idx].tolist()
                    for idx, seed in enumerate(SEEDS)
                },
            }

        summary["datasets"][dataset_key] = dataset_result
        summary["sources"][dataset_key] = source_records

    return summary


def add_matrix_panel(
    ax: plt.Axes,
    matrix_mean: np.ndarray,
    matrix_sd: np.ndarray,
    title: str,
    *,
    annotate_sd: bool,
) -> Any:
    image = ax.imshow(matrix_mean, vmin=0.0, vmax=100.0)

    ax.set_title(title, fontsize=10)
    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax.set_xlabel("Predicted class", fontsize=9)
    ax.set_ylabel("True class", fontsize=9)

    # Annotate using contrasting text chosen from the cell magnitude.
    threshold = 50.0
    for row in range(4):
        for col in range(4):
            mean = matrix_mean[row, col]
            sd = matrix_sd[row, col]
            if annotate_sd:
                label = f"{mean:.1f}%\n±{sd:.1f}"
            else:
                label = f"{mean:.1f}%"
            text_color = "white" if mean >= threshold else "black"
            ax.text(
                col,
                row,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    return image


def generate_figure(
    summary: dict[str, Any],
    output_dir: Path,
    *,
    dpi: int,
    annotate_sd: bool,
) -> tuple[Path, Path]:
    panels = (
        ("isic", "hard", "(a) ISIC — Hard Routing"),
        ("isic", "fusion", "(b) ISIC — Probability Fusion"),
        ("hiba", "hard", "(c) HIBA — Hard Routing"),
        ("hiba", "fusion", "(d) HIBA — Probability Fusion"),
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.6))
    last_image = None

    for ax, (dataset_key, system_key, title) in zip(axes.flat, panels):
        system_summary = summary["datasets"][dataset_key]["systems"][system_key]
        mean = np.asarray(
            system_summary["mean_row_normalized_percent"],
            dtype=np.float64,
        )
        sd = np.asarray(
            system_summary["sample_sd_row_normalized_percent"],
            dtype=np.float64,
        )
        last_image = add_matrix_panel(
            ax,
            mean,
            sd,
            title,
            annotate_sd=annotate_sd,
        )

    if last_image is None:
        raise RuntimeError("No confusion-matrix panels were generated")

    fig.subplots_adjust(
        left=0.09,
        right=0.88,
        bottom=0.08,
        top=0.95,
        wspace=0.33,
        hspace=0.34,
    )
    colorbar_ax = fig.add_axes([0.91, 0.17, 0.025, 0.66])
    colorbar = fig.colorbar(last_image, cax=colorbar_ax)
    colorbar.set_label("Mean row-normalized predictions (%)", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "fig4_confusion_matrices.png"
    pdf_path = output_dir / "fig4_confusion_matrices.pdf"

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else repo_root / "figures"
    )

    summary = collect_summaries(repo_root)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "fig4_confusion_matrices_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    png_path, pdf_path = generate_figure(
        summary,
        output_dir,
        dpi=args.dpi,
        annotate_sd=args.annotate_sd,
    )

    print("CoMeSySo confusion-matrix figure generated successfully.")
    print(f"PNG:     {png_path}")
    print(f"PDF:     {pdf_path}")
    print(f"Summary: {summary_path}")
    print("Seed-image pairs pooled: false")


if __name__ == "__main__":
    main()
