#!/usr/bin/env python3
"""Cross-model transfer heatmap: two-panel (ordinal accuracy + retention).

Input:  embedded paper data or artifacts/*transfer*.json
Output: artifacts/figures/fig3_transfer_heatmap.pdf
Deps:   matplotlib, numpy
"""

import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

MODELS = ["Qwen-rw", "Pythia", "OLMo", "Qwen-cont"]

ORDINAL_ACC = np.array([
    [0.423, 0.259, 0.265, 0.347],
    [0.272, 0.386, 0.353, 0.329],
    [0.277, 0.345, 0.382, 0.324],
    [0.344, 0.328, 0.311, 0.388],
])


def compute_retention(acc):
    diag = np.diag(acc)
    return acc / diag[:, None] * 100


def parse_args():
    parser = argparse.ArgumentParser(description="Plot cross-model transfer heatmap")
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.data_dir

    acc = ORDINAL_ACC
    retention = compute_retention(acc)
    n = len(MODELS)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)

    # --- Panel (a): Ordinal Accuracy ---
    cmap_acc = plt.cm.RdYlGn
    vmin_a, vmax_a = 0.25, max(0.43, np.nanmax(acc) + 0.01)
    im1 = ax1.imshow(acc, cmap=cmap_acc, vmin=vmin_a, vmax=vmax_a, aspect="equal")

    for i in range(n):
        for j in range(n):
            val = acc[i, j]
            color = "white" if val > (vmin_a + vmax_a) / 2 else "black"
            weight = "bold" if i == j else "normal"
            ax1.text(j, i, f"{val:.3f}", ha="center", va="center",
                     fontsize=10, color=color, fontweight=weight)

    for i in range(n):
        rect = plt.Rectangle((i - 0.5, i - 0.5), 1, 1, linewidth=2.0,
                              edgecolor="#333333", facecolor="none", zorder=5)
        ax1.add_patch(rect)

    ax1.set_xticks(range(n))
    ax1.set_xticklabels(MODELS, fontsize=9)
    ax1.set_yticks(range(n))
    ax1.set_yticklabels(MODELS, fontsize=9)
    ax1.set_xlabel("Test Model", fontsize=10)
    ax1.set_ylabel("Train Model", fontsize=10)
    ax1.set_title("(a) Ordinal Accuracy", fontsize=11, pad=6)
    ax1.tick_params(direction="in", length=0)

    cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.82, pad=0.02)
    cbar1.set_label("Accuracy", fontsize=10)
    cbar1.ax.tick_params(labelsize=9)

    # --- Panel (b): Transfer Retention ---
    cmap_ret = plt.cm.RdYlGn
    vmin_b, vmax_b = 55, 100
    im2 = ax2.imshow(retention, cmap=cmap_ret, vmin=vmin_b, vmax=vmax_b, aspect="equal")

    for i in range(n):
        for j in range(n):
            val = retention[i, j]
            color = "white" if val > (vmin_b + vmax_b) / 2 else "black"
            weight = "bold" if i == j else "normal"
            ax2.text(j, i, f"{val:.0f}%", ha="center", va="center",
                     fontsize=10, color=color, fontweight=weight)

    for i in range(n):
        rect = plt.Rectangle((i - 0.5, i - 0.5), 1, 1, linewidth=2.0,
                              edgecolor="#333333", facecolor="none", zorder=5)
        ax2.add_patch(rect)

    ax2.set_xticks(range(n))
    ax2.set_xticklabels(MODELS, fontsize=9)
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(MODELS, fontsize=9)
    ax2.set_xlabel("Test Model", fontsize=10)
    ax2.set_ylabel("Train Model", fontsize=10)
    ax2.set_title("(b) Transfer Retention", fontsize=11, pad=6)
    ax2.tick_params(direction="in", length=0)

    cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.82, pad=0.02)
    cbar2.set_label("Retention (%)", fontsize=10)
    cbar2.ax.tick_params(labelsize=9)

    # --- Save ---
    if args.output:
        out_path = args.output
    else:
        out_dir = root / "artifacts" / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "fig3_transfer_heatmap.pdf"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    png_path = out_path.with_suffix(".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    pdf_size = out_path.stat().st_size
    png_size = png_path.stat().st_size
    print(f"Models: {MODELS}")
    print(f"PDF: {pdf_size/1024:.1f} KB | PNG: {png_size/1024:.1f} KB")
    assert pdf_size > 3 * 1024, f"PDF too small: {pdf_size}"
    print("OK")


if __name__ == "__main__":
    main()
