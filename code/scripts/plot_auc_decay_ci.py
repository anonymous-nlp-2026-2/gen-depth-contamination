#!/usr/bin/env python3
"""AUC decay curves with bootstrap confidence interval bands.

Output: docs/paper/figures/fig_s1_auc_decay_ci.pdf
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PAIR_KEYS = ["0v1", "1v2", "2v3", "3v4", "4v5"]
X_LABELS = ["0 vs 1", "1 vs 2", "2 vs 3", "3 vs 4", "4 vs 5"]

ROOT = Path("/root/autodl-tmp/gen-depth-contamination")
OUT_DIR = ROOT / "docs" / "paper" / "figures"

CONFIGS = [
    {
        "label": "Qwen-1.5B-RW",
        "path": ROOT / "results" / "exp_qwen_retrain_chain" / "bootstrap_ci.json",
        "format": "ci_95",
    },
    {
        "label": "Pythia-1.4B",
        "path": ROOT / "results" / "exp_14b_c4" / "bootstrap_ci_results.json",
        "format": "pairwise_results",
    },
    {
        "label": "OLMo-1B",
        "path": ROOT / "results" / "exp_olmo1b_c4_rerun" / "pairwise_auc.json",
        "format": "per_fold",
    },
    {
        "label": "Qwen-1.5B-cont",
        "path": ROOT / "results_exp022_cross_scorer" / "qwen_cont256__qwen_base" / "pairwise_auc.json",
        "format": "per_fold",
    },
]

COLORS = ["#4e79a7", "#e15759", "#76b7b2", "#59a14f"]
MARKERS = ["o", "s", "^", "D"]


def load_ci_95(path):
    with open(path) as f:
        raw = json.load(f)
    result = {}
    for k in PAIR_KEYS:
        if k in raw:
            entry = raw[k]
            result[k] = {
                "mean": entry["mean"],
                "ci_lower": entry["ci_95"][0],
                "ci_upper": entry["ci_95"][1],
            }
    return result


def load_per_fold(path):
    with open(path) as f:
        raw = json.load(f)
    means = raw["mean"]
    per_fold = raw["per_fold"]
    result = {}
    for k in PAIR_KEYS:
        if k in means and k in per_fold:
            folds = np.array(per_fold[k])
            se = np.std(folds, ddof=1) / np.sqrt(len(folds))
            m = means[k]
            result[k] = {
                "mean": m,
                "ci_lower": m - 1.96 * se,
                "ci_upper": m + 1.96 * se,
            }
    return result


def load_pairwise_results(path):
    with open(path) as f:
        raw = json.load(f)
    pr = raw["pairwise_results"]
    result = {}
    for k in PAIR_KEYS:
        if k in pr:
            entry = pr[k]
            result[k] = {
                "mean": entry["point_estimate"],
                "ci_lower": entry["ci_lower"],
                "ci_upper": entry["ci_upper"],
            }
    return result


LOADERS = {
    "ci_95": load_ci_95,
    "per_fold": load_per_fold,
    "pairwise_results": load_pairwise_results,
}


def main():
    models = {}
    for cfg in CONFIGS:
        if not cfg["path"].exists():
            print(f"WARNING: {cfg['path']} not found, skipping")
            continue
        loader = LOADERS[cfg["format"]]
        models[cfg["label"]] = loader(cfg["path"])

    if not models:
        print("No data loaded")
        return

    x = np.arange(len(PAIR_KEYS))

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)

    ax.axhline(0.60, color="grey", ls="--", lw=0.8, zorder=1)
    ax.text(len(PAIR_KEYS) - 0.45, 0.608, r"$\theta = 0.60$",
            fontsize=8.5, color="grey", va="bottom", ha="right")
    ax.axhline(0.50, color="lightgrey", ls="--", lw=0.8, zorder=1)
    ax.text(len(PAIR_KEYS) - 0.45, 0.508, "chance",
            fontsize=8.5, color="silver", va="bottom", ha="right")

    handles = []
    for i, (label, ci_data) in enumerate(models.items()):
        color = COLORS[i]
        marker = MARKERS[i]

        means = [ci_data.get(k, {}).get("mean", float("nan")) for k in PAIR_KEYS]
        lowers = [ci_data.get(k, {}).get("ci_lower", float("nan")) for k in PAIR_KEYS]
        uppers = [ci_data.get(k, {}).get("ci_upper", float("nan")) for k in PAIR_KEYS]

        ax.fill_between(x, lowers, uppers, color=color, alpha=0.15, zorder=2)
        line, = ax.plot(x, means, "-", color=color, lw=1.8, marker=marker,
                        markersize=5, label=label, zorder=3)
        handles.append(line)

    ax.set_xticks(x)
    ax.set_xticklabels(X_LABELS)
    ax.set_xlabel("Depth Pair")
    ax.set_ylabel("Pairwise AUC")
    ax.set_ylim(0.45, 1.05)
    ax.set_xlim(-0.25, len(PAIR_KEYS) - 0.75)
    ax.tick_params(direction="in")

    ax.legend(handles=handles, fontsize=9, loc="upper right",
              bbox_to_anchor=(1.0, 0.98), framealpha=0.9, edgecolor="0.8")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / "fig_s1_auc_decay_ci.pdf"
    png_path = OUT_DIR / "fig_s1_auc_decay_ci.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Configs: {list(models.keys())}")
    print(f"PDF: {pdf_path.stat().st_size/1024:.1f} KB | PNG: {png_path.stat().st_size/1024:.1f} KB")

    for lbl, data in models.items():
        v = data.get("1v2", {}).get("mean", float("nan"))
        print(f"{lbl} 1v2 AUC = {v:.4f}")
    print("OK")


if __name__ == "__main__":
    main()
