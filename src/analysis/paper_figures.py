"""Deterministic Phase 07 ICCIT figure generation from committed evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.analysis.stored_prediction_statistics import sha256_file, write_json


CLASSES = ("non-malignant", "melanoma", "BCC", "SCC")
SOURCE_CLASSES = ("non_malignant", "melanoma", "bcc", "scc")
SUPPORT = (2398, 678, 498, 94)
FIGURE_SPECS = {
    "figure01_architecture": (7.16, 3.6),
    "figure02_confusion_matrix_comparison": (7.16, 3.55),
    "figure03_per_class_f1": (7.16, 3.75),
}


class FigureEvidenceError(ValueError):
    """Raised when committed figure evidence is missing or inconsistent."""


def normalize_confusion(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.shape != (4, 4) or (matrix < 0).any():
        raise FigureEvidenceError("Confusion matrix must be nonnegative 4x4.")
    support = matrix.sum(axis=1)
    if (support == 0).any():
        raise FigureEvidenceError("Confusion rows cannot be empty.")
    return matrix / support[:, None]


def _source_hashes(source: Path) -> dict[str, str]:
    names = (
        "confusion_matrix_flat.csv",
        "confusion_matrix_hierarchical.csv",
        "paper_table_per_class_f1.csv",
        "bootstrap_confidence_intervals.csv",
        "claims_lock.json",
        "efficiency_claims_lock.json",
    )
    hashes = {}
    for name in names:
        path = source / name
        if not path.is_file():
            raise FigureEvidenceError(f"Missing figure source: {path}")
        hashes[path.as_posix()] = sha256_file(path)
    return hashes


def _read_confusion(path: Path) -> np.ndarray:
    frame = pd.read_csv(path)
    if frame["true_label"].tolist() != list(SOURCE_CLASSES):
        raise FigureEvidenceError("Confusion class order changed.")
    matrix = frame.drop(columns="true_label").to_numpy(dtype=np.int64)
    if tuple(matrix.sum(axis=1)) != SUPPORT:
        raise FigureEvidenceError("Confusion support changed.")
    return matrix


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "svg.fonttype": "none",
            "svg.hashsalt": "phase07-iccit",
            "pdf.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        base.with_suffix(".svg"),
        format="svg",
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "Phase 07 deterministic figure generator"},
    )
    svg_path = base.with_suffix(".svg")
    svg_path.write_text(
        "\n".join(
            line.rstrip()
            for line in svg_path.read_text(encoding="utf-8").splitlines()
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    fig.savefig(
        base.with_suffix(".pdf"),
        format="pdf",
        bbox_inches="tight",
        metadata={
            "Creator": "Phase 07 deterministic figure generator",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        base.with_suffix(".png"),
        format="png",
        dpi=600,
        bbox_inches="tight",
        metadata={"Software": "Phase 07 deterministic figure generator"},
    )
    plt.close(fig)


def _box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, text: str, color: str) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor="#202020",
        linewidth=0.9,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=8)


def architecture_figure(output: Path) -> None:
    fig, ax = plt.subplots(figsize=FIGURE_SPECS["figure01_architecture"])
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.axis("off")
    _box(ax, (0.1, 2.5), 1.75, 0.8, "Input dermoscopic\nimage", "#f2f2f2")
    ax.text(4.25, 5.72, "Conditional hierarchical system", ha="center", weight="bold")
    _box(ax, (2.35, 4.4), 1.7, 0.8, "Stage 1\nEfficientNet-B0", "#d9e6f2")
    _box(ax, (4.75, 4.4), 1.5, 0.8, "Binary decision", "#f2f2f2")
    _box(ax, (7.15, 4.95), 1.7, 0.75, "non-malignant\nprediction", "#e8e8e8")
    _box(ax, (6.8, 3.25), 1.7, 0.8, "Stage 2\nEfficientNet-B0", "#d9e6f2")
    _box(ax, (9.15, 3.25), 1.75, 0.8, "melanoma / BCC /\nSCC prediction", "#e8e8e8")
    ax.text(3.2, 5.34, "runs for every image", ha="center", fontsize=7)
    ax.text(7.65, 4.34, "conditional", ha="center", fontsize=7, style="italic")
    ax.text(6.58, 5.58, "non-malignant", fontsize=7, ha="center")
    ax.text(6.25, 3.38, "malignant", fontsize=7, ha="center")
    ax.text(4.25, 2.08, "Flat comparison system", ha="center", weight="bold")
    _box(ax, (2.35, 0.7), 2.2, 0.8, "Four-class\nEfficientNet-B0", "#eadfca")
    _box(ax, (5.55, 0.7), 2.7, 0.8, "non-malignant / melanoma /\nBCC / SCC prediction", "#e8e8e8")
    ax.text(3.45, 0.46, "one direct decision path", ha="center", fontsize=7)
    arrows = [
        ((1.85, 3.05), (2.35, 4.8)),
        ((4.05, 4.8), (4.75, 4.8)),
        ((6.25, 4.98), (7.15, 5.32)),
        ((6.25, 4.58), (6.8, 3.65)),
        ((8.5, 3.65), (9.15, 3.65)),
        ((1.85, 2.75), (2.35, 1.1)),
        ((4.55, 1.1), (5.55, 1.1)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, color="#202020", linewidth=1))
    fig.tight_layout()
    _save(fig, output / "figure01_architecture")


def confusion_figure(flat: np.ndarray, hierarchical: np.ndarray, output: Path) -> tuple[np.ndarray, np.ndarray]:
    normalized = (normalize_confusion(flat), normalize_confusion(hierarchical))
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SPECS["figure02_confusion_matrix_comparison"], sharex=True, sharey=True)
    image = None
    for ax, raw, values, title in zip(
        axes, (flat, hierarchical), normalized, ("(a) Flat model", "(b) Conditional hierarchy")
    ):
        image = ax.imshow(values, cmap="Greys", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xticks(range(4), CLASSES, rotation=30, ha="right")
        ax.set_yticks(range(4), [f"{name}\n(n={n:,})" for name, n in zip(CLASSES, SUPPORT)])
        ax.set_xlabel("Predicted class")
        for row in range(4):
            for column in range(4):
                color = "white" if values[row, column] >= 0.55 else "black"
                ax.text(column, row, f"{values[row,column]*100:.1f}%\n({raw[row,column]})", ha="center", va="center", fontsize=6.5, color=color)
    axes[0].set_ylabel("True class")
    assert image is not None
    fig.subplots_adjust(left=0.13, right=0.84, bottom=0.2, top=0.88, wspace=0.25)
    colorbar_axis = fig.add_axes((0.875, 0.20, 0.022, 0.68))
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Row-normalized proportion")
    colorbar.set_ticks(
        [0, 0.25, 0.5, 0.75, 1],
        labels=["0%", "25%", "50%", "75%", "100%"],
    )
    _save(fig, output / "figure02_confusion_matrix_comparison")
    return normalized


def f1_figure(table: pd.DataFrame, output: Path) -> None:
    if table["class"].tolist() != list(SOURCE_CLASSES):
        raise FigureEvidenceError("Per-class F1 class order changed.")
    x = np.arange(4)
    width = 0.34
    fig, ax = plt.subplots(figsize=FIGURE_SPECS["figure03_per_class_f1"])
    colors = ("#4c78a8", "#e39c37")
    for offset, model, label, color in (
        (-width / 2, "flat", "Flat", colors[0]),
        (width / 2, "hierarchical", "Hierarchical", colors[1]),
    ):
        points = table[f"{model}_f1"].to_numpy(float)
        lower = table[f"{model}_ci_lower"].to_numpy(float)
        upper = table[f"{model}_ci_upper"].to_numpy(float)
        ax.bar(x + offset, points, width, label=label, color=color, edgecolor="black", linewidth=0.6)
        ax.errorbar(x + offset, points, yerr=np.vstack((points - lower, upper - points)), fmt="none", ecolor="black", capsize=3, linewidth=0.9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 score")
    ax.set_xticks(x, [f"{name}\n(n={n:,})" for name, n in zip(CLASSES, SUPPORT)])
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_title(
        "Exploratory comparisons; SCC support is 94 and uncertainty is high.",
        loc="left",
        fontsize=7,
        pad=5,
    )
    fig.tight_layout()
    _save(fig, output / "figure03_per_class_f1")


def generate_figures(source: Path, output: Path, generated: Path, command: str) -> list[Path]:
    """Generate required figures plus numerical audit and artifact manifest."""
    _style()
    source_hashes = _source_hashes(source)
    flat = _read_confusion(source / "confusion_matrix_flat.csv")
    hierarchical = _read_confusion(source / "confusion_matrix_hierarchical.csv")
    f1_table = pd.read_csv(source / "paper_table_per_class_f1.csv")
    architecture_figure(output)
    flat_norm, hierarchy_norm = confusion_figure(flat, hierarchical, output)
    f1_figure(f1_table, output)
    figure_files = [
        output / f"{name}.{extension}"
        for name in FIGURE_SPECS
        for extension in ("svg", "pdf", "png")
    ]
    audits = {
        "class_order": list(CLASSES),
        "source_class_order": list(SOURCE_CLASSES),
        "support": list(SUPPORT),
        "source_hashes": source_hashes,
        "confusion": {
            "normalization": "row normalized",
            "flat_raw": flat.tolist(),
            "hierarchical_raw": hierarchical.tolist(),
            "flat_normalized": flat_norm.tolist(),
            "hierarchical_normalized": hierarchy_norm.tolist(),
            "row_sums_equal_one_tolerance": 1e-12,
            "row_sum_check": bool(
                np.allclose(flat_norm.sum(axis=1), 1, atol=1e-12)
                and np.allclose(hierarchy_norm.sum(axis=1), 1, atol=1e-12)
            ),
        },
        "per_class_f1": {
            "values": f1_table.to_dict(orient="records"),
            "interval_type": "model-specific 95% paired-class-stratified bootstrap intervals",
            "paired_difference_intervals_used": False,
            "exploratory": True,
            "scc_support": 94,
        },
        "architecture": {
            "nodes": ["input", "stage1", "binary decision", "non-malignant output", "stage2", "malignant subtype output", "flat four-class model", "flat four-class output"],
            "stage1_runs_for_every_image": True,
            "stage2_is_conditional": True,
            "flat_direct_path": True,
        },
        "outputs": {
            f"reports/phase07/figures/{path.name}": {
                "sha256": sha256_file(path),
                "format": path.suffix[1:],
                "figure_inches": list(FIGURE_SPECS[path.stem]),
                "png_dpi": 600 if path.suffix == ".png" else None,
            }
            for path in figure_files
        },
        "generation_command": command,
        "optional_figure04": "omitted: duplicates the main paired-comparison evidence and is not essential within six pages",
    }
    audit_path = generated / "figure_data_audit.json"
    manifest_path = generated / "figure_artifact_manifest.txt"
    write_json(audit_path, audits)
    manifest_files = [*figure_files, audit_path]
    manifest_path.write_text(
        "".join(
            f"{sha256_file(path)}  "
            f"{'reports/phase07/figures/' + path.name if path in figure_files else 'reports/phase07/generated/figure_data_audit.json'}\n"
            for path in manifest_files
        ),
        encoding="utf-8",
        newline="\n",
    )
    return [*figure_files, audit_path, manifest_path]
