#!/usr/bin/env python3
"""Transfer heatmap v2: K* accuracy + AUC(0v1) for 3-model cross-transfer."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    data_path = ROOT / "results" / "exp013_olmo_rerun_v2" / "transfer_matrix.json"
    with open(data_path) as f:
        data = json.load(f)

    models = data["models"]
    self_kstar = data["self_kstar"]
    n = len(models)

    auc_matrix = np.zeros((n, n))
    kstar_matrix = np.zeros((n, n), dtype=int)
    correct_matrix = np.zeros((n, n), dtype=bool)

    for i, train_m in enumerate(models):
        for j, test_m in enumerate(models):
            cell = data["matrix"][train_m][test_m]
            auc_matrix[i, j] = cell["pairwise_auc"]["0v1"]
            kstar_matrix[i, j] = cell["kstar"]
            if i == j:
                correct_matrix[i, j] = True
            else:
                correct_matrix[i, j] = (cell["kstar"] == self_kstar[test_m])

    agg_acc = data["aggregate_accuracy"]
    correct_count = data["correct"]
    total_count = data["total"]

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "mathtext.fontset": "cm",
    })

    fig, ax = plt.subplots(figsize=(3.3, 3.0))

    DIAG_COLOR = "#D0D0D0"
    CORRECT_COLOR = "#8FCA8F"
    INCORRECT_COLOR = "#E8908A"

    for i in range(n):
        for j in range(n):
            if i == j:
                color = DIAG_COLOR
            elif correct_matrix[i, j]:
                color = CORRECT_COLOR
            else:
                color = INCORRECT_COLOR

            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                 facecolor=color, edgecolor="white", linewidth=2)
            ax.add_patch(rect)

            auc_val = auc_matrix[i, j]
            ax.text(j, i - 0.13, f"{auc_val:.3f}", ha="center", va="center",
                    fontsize=9, fontweight="bold" if i == j else "normal",
                    color="#1a1a1a")

            k_pred = kstar_matrix[i, j]
            if i == j:
                label = r"$K\!^*\!=\!%d$" % k_pred
            else:
                if correct_matrix[i, j]:
                    mark = r"$\checkmark$"
                else:
                    mark = r"$\times$"
                label = r"$K\!^*\!=\!%d$ %s" % (k_pred, mark)

            ax.text(j, i + 0.23, label, ha="center", va="center",
                    fontsize=7.5, color="#555555")

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels(models, fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(models, fontsize=9)
    ax.set_xlabel("Test model", fontsize=9, labelpad=4)
    ax.set_ylabel("Train model", fontsize=9, labelpad=4)
    ax.tick_params(direction="in", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(
        r"Cross-model $K\!^*$ transfer  (acc = %d/%d = %.1f%%)"
        % (correct_count, total_count, agg_acc * 100),
        fontsize=9, pad=8)

    legend_patches = [
        mpatches.Patch(facecolor=DIAG_COLOR, edgecolor="#999", linewidth=0.5, label="Self"),
        mpatches.Patch(facecolor=CORRECT_COLOR, edgecolor="#999", linewidth=0.5, label="Correct"),
        mpatches.Patch(facecolor=INCORRECT_COLOR, edgecolor="#999", linewidth=0.5, label="Incorrect"),
    ]
    ax.legend(handles=legend_patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.28), ncol=3, fontsize=7.5,
              frameon=False, handlelength=1.2, handletextpad=0.4,
              columnspacing=1.0)

    out_dir = ROOT / "docs" / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    for suffix in [".pdf", ".png"]:
        out_path = out_dir / f"fig3_transfer_heatmap{suffix}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        size_kb = out_path.stat().st_size / 1024
        print(f"Saved: {out_path} ({size_kb:.1f} KB)")

    plt.close(fig)
    print("OK")


if __name__ == "__main__":
    main()
