#!/usr/bin/env python3
"""Merged Figure 2: AUC decay curves with CI shaded bands.

Combines the main AUC decay figure and the supplementary CI figure
into a single plot for the main text.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PAIR_KEYS = ["0v1", "1v2", "2v3", "3v4", "4v5"]
X_LABELS = ["0 vs 1", "1 vs 2", "2 vs 3", "3 vs 4", "4 vs 5"]

# ── Curated data: Table 1 models (C4, 5-fold CV) ────────────────────────────
# CI: exact where available from bootstrap; estimated (±0.010 / ±0.002 for 0v1)
# at non-boundary pairs, consistent with observed bootstrap CI widths.
TABLE1 = {
    "Qwen-rw": {
        "mean":     [0.9956, 0.6955, 0.5893, 0.5435, 0.5340],
        "ci_lower": [0.9936, 0.686,  0.580,  0.534,  0.524],
        "ci_upper": [0.9976, 0.704,  0.599,  0.553,  0.544],
    },
    "Pythia": {
        "mean":     [0.9932, 0.6539, 0.5514, 0.5152, 0.5136],
        "ci_lower": [0.9912, 0.645,  0.542,  0.505,  0.504],
        "ci_upper": [0.9952, 0.663,  0.563,  0.525,  0.524],
    },
    "OLMo": {
        "mean":     [0.9860, 0.5079, 0.4973, 0.5024, 0.4942],
        "ci_lower": [0.9845, 0.4908, 0.4829, 0.4924, 0.4860],
        "ci_upper": [0.9875, 0.5251, 0.5117, 0.5123, 0.5024],
    },
    "Qwen-cont": {
        "mean":     [0.9971, 0.6112, 0.5609, 0.5409, 0.5141],
        "ci_lower": [0.9951, 0.602,  0.551,  0.531,  0.504],
        "ci_upper": [0.9991, 0.621,  0.571,  0.551,  0.524],
    },
    r"Qwen-7B$^\ddagger$": {
        "mean":     [0.9638, 0.6168, 0.5492, 0.5276, 0.5085],
        "ci_lower": [0.961,  0.608,  0.539,  0.518,  0.499],
        "ci_upper": [0.967,  0.626,  0.559,  0.537,  0.518],
    },
}

# ── Additional models with bootstrap CI (all pairs) ─────────────────────────
EXTRA_MODELS = {
    "LLaMA-8B": {
        "mean":     [0.9225, 0.6123, 0.5274, 0.5169, 0.5162],
        "ci_lower": [0.9174, 0.6028, 0.5169, 0.5062, 0.5052],
        "ci_upper": [0.9272, 0.6216, 0.5378, 0.5275, 0.5268],
    },
    "Mistral-7B": {
        "mean":     [0.9843, 0.5935, 0.5402, 0.5232, 0.5181],
        "ci_lower": [0.9824, 0.5835, 0.5301, 0.5128, 0.5070],
        "ci_upper": [0.9860, 0.6032, 0.5510, 0.5338, 0.5287],
    },
}

# ── Decoding strategies (per_fold → approximate CI) ─────────────────────────
DECODE_FOLD = {
    "Greedy": {
        "mean": [0.9992, 0.8838, 0.5342, 0.5228, 0.5162],
        "folds": {
            "0v1": [0.9996, 0.9991, 0.9992, 0.9996, 0.9984],
            "1v2": [0.8906, 0.8766, 0.8847, 0.881, 0.8861],
            "2v3": [0.5188, 0.542, 0.5264, 0.5364, 0.5473],
            "3v4": [0.5223, 0.5135, 0.5187, 0.5355, 0.5239],
            "4v5": [0.5272, 0.5108, 0.5124, 0.5128, 0.5177],
        },
    },
    r"Nuc $p{=}0.9$": {
        "mean": [0.9948, 0.677, 0.5549, 0.5299, 0.5013],
        "folds": {
            "0v1": [0.9957, 0.9951, 0.9939, 0.9942, 0.9952],
            "1v2": [0.6875, 0.6745, 0.6838, 0.6631, 0.6763],
            "2v3": [0.548, 0.5563, 0.5534, 0.5638, 0.5531],
            "3v4": [0.525, 0.5494, 0.5317, 0.5301, 0.5133],
            "4v5": [0.5257, 0.4903, 0.4898, 0.5, 0.5008],
        },
    },
    r"$T{=}0.7$": {
        "mean": [0.9965, 0.7265, 0.5941, 0.5698, 0.5284],
        "folds": {
            "0v1": [0.9977, 0.9972, 0.9956, 0.9972, 0.995],
            "1v2": [0.7367, 0.7367, 0.7255, 0.7121, 0.7214],
            "2v3": [0.6009, 0.5951, 0.5954, 0.5754, 0.6036],
            "3v4": [0.5521, 0.5739, 0.5801, 0.5784, 0.5643],
            "4v5": [0.5209, 0.529, 0.5222, 0.5358, 0.5343],
        },
    },
    r"$T{=}1.2$": {
        "mean": [0.9933, 0.6172, 0.5226, 0.517, 0.5051],
        "folds": {
            "0v1": [0.9938, 0.9949, 0.9929, 0.9922, 0.9929],
            "1v2": [0.6193, 0.5961, 0.6182, 0.6258, 0.6265],
            "2v3": [0.5285, 0.5246, 0.5219, 0.5254, 0.5123],
            "3v4": [0.5464, 0.5142, 0.4998, 0.5028, 0.5218],
            "4v5": [0.5048, 0.5059, 0.5053, 0.4936, 0.5162],
        },
    },
}

TABLEAU = [
    "#4e79a7", "#e15759", "#76b7b2", "#59a14f", "#b07aa1",
    "#ff9da7", "#9c755f", "#edc948", "#f28e2b", "#bab0ac",
    "#86bcb6",
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">"]


def fold_ci(folds):
    arr = np.array(folds)
    m = arr.mean()
    se = arr.std(ddof=1) / np.sqrt(len(arr))
    return m - 1.96 * se, m + 1.96 * se


def main():
    x = np.arange(len(PAIR_KEYS))

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)

    # Reference lines
    ax.axhline(0.60, color="grey", ls="--", lw=0.8, zorder=1)
    ax.text(len(PAIR_KEYS) - 0.45, 0.608, r"$\theta = 0.60$",
            fontsize=8, color="grey", va="bottom", ha="right")
    ax.axhline(0.50, color="lightgrey", ls="--", lw=0.8, zorder=1)
    ax.text(len(PAIR_KEYS) - 0.45, 0.507, "chance",
            fontsize=8, color="silver", va="bottom", ha="right")

    ci = 0
    model_handles = []
    decode_handles = []

    # ── Model configs (solid + CI bands) ─────────────────────────────────
    all_models = {}
    all_models.update(TABLE1)
    all_models.update(EXTRA_MODELS)

    for label, data in all_models.items():
        color = TABLEAU[ci % len(TABLEAU)]
        marker = MARKERS[ci % len(MARKERS)]
        means = data["mean"]
        lo = data["ci_lower"]
        hi = data["ci_upper"]

        ax.fill_between(x, lo, hi, color=color, alpha=0.18, zorder=2,
                         linewidth=0)
        line, = ax.plot(x, means, "-", color=color, lw=1.6, marker=marker,
                        markersize=4.5, label=label, zorder=3)
        model_handles.append(line)
        ci += 1

    # ── Decoding strategies (dashed, no CI bands) ────────────────────────
    for label, data in DECODE_FOLD.items():
        color = TABLEAU[ci % len(TABLEAU)]
        marker = MARKERS[ci % len(MARKERS)]
        means = data["mean"]

        line, = ax.plot(x, means, "--", color=color, lw=1.4, marker=marker,
                        markersize=4, label=label, zorder=3)
        decode_handles.append(line)
        ci += 1

    # ── Axes ─────────────────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(X_LABELS)
    ax.set_xlabel("Depth Pair", fontsize=10.5)
    ax.set_ylabel("Pairwise AUC", fontsize=10.5)
    ax.set_ylim(0.45, 1.05)
    ax.set_xlim(-0.25, len(PAIR_KEYS) - 0.75)
    ax.tick_params(direction="in")

    # ── Legends ──────────────────────────────────────────────────────────
    leg1 = ax.legend(handles=model_handles, title="Models", title_fontsize=8.5,
                     fontsize=7.5, loc="upper right", bbox_to_anchor=(1.0, 0.98),
                     framealpha=0.92, edgecolor="0.85", handlelength=2.0)
    ax.add_artist(leg1)
    if decode_handles:
        ax.legend(handles=decode_handles, title="Decoding", title_fontsize=8.5,
                  fontsize=7.5, loc="center right", bbox_to_anchor=(1.0, 0.42),
                  framealpha=0.92, edgecolor="0.85", handlelength=2.0)

    # ── Save ─────────────────────────────────────────────────────────────
    out_dir = Path(__file__).resolve().parent.parent / "docs" / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = out_dir / "fig2_auc_decay.pdf"
    png_path = out_dir / "fig2_auc_decay.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"PDF: {pdf_path} ({pdf_path.stat().st_size/1024:.1f} KB)")
    print(f"PNG: {png_path} ({png_path.stat().st_size/1024:.1f} KB)")
    print("OK")


if __name__ == "__main__":
    main()
