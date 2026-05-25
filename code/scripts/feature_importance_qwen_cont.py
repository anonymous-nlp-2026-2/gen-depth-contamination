"""
Permutation Importance Analysis for 15D OBD Features
Qwen-1.5B continuation mode (retrain chain) on C4
"""

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["OMP_NUM_THREADS"] = "4"

DATA_PATH = Path("/root/autodl-tmp/gen-depth-contamination/data/exp_qwen_retrain_chain/features.csv")
OUT_DIR = Path("/root/autodl-tmp/gen-depth-contamination/results/feature_importance_qwen_cont")
ART_DIR = Path("/root/autodl-tmp/gen-depth-contamination/artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR.mkdir(parents=True, exist_ok=True)

ALL_FEATURES = [
    "surp_mean", "surp_std", "surp_skew", "surp_kurt",
    "surp_d1_mean", "surp_d1_std", "surp_d1_skew",
    "surp_d2_mean", "surp_d2_std",
    "ttr", "hapax_ratio", "self_bleu",
    "freq_kurtosis", "freq_entropy", "low_freq_ratio",
]

FEATURE_GROUPS = {
    "surp_mean": "surprisal_base", "surp_std": "surprisal_base",
    "surp_skew": "surprisal_base", "surp_kurt": "surprisal_base",
    "surp_d1_mean": "surprisal_deriv", "surp_d1_std": "surprisal_deriv",
    "surp_d1_skew": "surprisal_deriv",
    "surp_d2_mean": "surprisal_deriv", "surp_d2_std": "surprisal_deriv",
    "ttr": "lexical_diversity", "hapax_ratio": "lexical_diversity",
    "self_bleu": "lexical_diversity",
    "freq_kurtosis": "freq_distribution", "freq_entropy": "freq_distribution",
    "low_freq_ratio": "freq_distribution",
}

DISPLAY_NAMES = {
    "surp_mean": "Mean Surprisal",
    "surp_std": "Surprisal Std",
    "surp_skew": "Surprisal Skew",
    "surp_kurt": "Surprisal Kurtosis",
    "surp_d1_mean": "Δ¹ Surprisal Mean",
    "surp_d1_std": "Δ¹ Surprisal Std",
    "surp_d1_skew": "Δ¹ Surprisal Skew",
    "surp_d2_mean": "Δ² Surprisal Mean",
    "surp_d2_std": "Δ² Surprisal Std",
    "ttr": "TTR",
    "hapax_ratio": "Hapax Ratio",
    "self_bleu": "Self-BLEU",
    "freq_kurtosis": "Freq Kurtosis",
    "freq_entropy": "Freq Entropy",
    "low_freq_ratio": "Low-Freq Ratio",
}

LGB_PARAMS = dict(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    num_leaves=31, verbose=-1, n_jobs=4,
)

N_REPEATS = 10
N_FOLDS = 5


def load_data():
    with open(DATA_PATH) as f:
        rows = list(csv.DictReader(f))
    doc_ids = np.array([int(r["doc_id"]) for r in rows])
    depths = np.array([int(r["depth"]) for r in rows])
    feat_dict = {}
    for col in ALL_FEATURES:
        vals = []
        for r in rows:
            v = r[col]
            vals.append(float(v) if v != "" and v != "nan" else np.nan)
        feat_dict[col] = np.array(vals)
    return doc_ids, depths, feat_dict


def get_pair_data(doc_ids, depths, feat_dict, d_low, d_high):
    mask = (depths == d_low) | (depths == d_high)
    ids = doc_ids[mask]
    y = (depths[mask] == d_high).astype(int)
    X = np.column_stack([feat_dict[col][mask] for col in ALL_FEATURES])
    nan_mask = np.isnan(X).any(axis=1)
    if nan_mask.any():
        valid = ~nan_mask
        X, y, ids = X[valid], y[valid], ids[valid]
    return ids, X, y


def compute_permutation_importance(X, y, groups):
    gkf = GroupKFold(n_splits=N_FOLDS)
    all_importances = []
    fold_aucs = []

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(X_train, y_train)

        baseline_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        fold_aucs.append(baseline_auc)

        result = permutation_importance(
            model, X_test, y_test,
            n_repeats=N_REPEATS,
            scoring="roc_auc",
            random_state=42,
        )
        all_importances.append(result.importances_mean)

    mean_importance = np.mean(all_importances, axis=0)
    std_importance = np.std(all_importances, axis=0)
    mean_auc = np.mean(fold_aucs)

    return mean_importance, std_importance, mean_auc


GROUP_COLORS = {
    "surprisal_base": "#2196F3",
    "surprisal_deriv": "#4CAF50",
    "lexical_diversity": "#FF9800",
    "freq_distribution": "#9C27B0",
}

GROUP_LABELS = {
    "surprisal_base": "Surprisal (base)",
    "surprisal_deriv": "Surprisal (derivatives)",
    "lexical_diversity": "Lexical diversity",
    "freq_distribution": "Frequency distribution",
}


def make_barplot(res_0v1, res_1v2, save_path):
    mean_01 = res_0v1["mean"]
    std_01 = res_0v1["std"]
    mean_12 = res_1v2["mean"]
    std_12 = res_1v2["std"]

    order_01 = np.argsort(mean_01)[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, mean_vals, std_vals, title in [
        (axes[0], mean_01, std_01, "0 vs 1 (human vs gen-1)"),
        (axes[1], mean_12, std_12, "1 vs 2 (K* boundary)"),
    ]:
        display = [DISPLAY_NAMES[ALL_FEATURES[i]] for i in order_01]
        vals = mean_vals[order_01]
        errs = std_vals[order_01]
        colors = [GROUP_COLORS[FEATURE_GROUPS[ALL_FEATURES[i]]] for i in order_01]

        bars = ax.barh(range(len(display)), vals, xerr=errs, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(display)))
        ax.set_yticklabels(display, fontsize=9)
        ax.set_xlabel("Permutation Importance (ΔAUC)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.invert_yaxis()
        ax.axvline(x=0, color="gray", linewidth=0.5, linestyle="--")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=GROUP_LABELS[g]) for g, c in GROUP_COLORS.items()]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Qwen-1.5B Continuation Mode (C4) — Permutation Feature Importance", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def make_latex_table(res_0v1, res_1v2, save_path):
    mean_01 = res_0v1["mean"]
    std_01 = res_0v1["std"]
    mean_12 = res_1v2["mean"]
    std_12 = res_1v2["std"]

    order_01 = np.argsort(mean_01)[::-1]

    lines = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Group & Feature & 0\,vs\,1 & 1\,vs\,2 \\",
        r"\midrule",
    ]

    prev_group = None
    for i in order_01:
        feat = ALL_FEATURES[i]
        group = FEATURE_GROUPS[feat]
        group_label = GROUP_LABELS[group] if group != prev_group else ""
        prev_group = group

        val_01 = f"{mean_01[i]:.4f} \\pm {std_01[i]:.4f}"
        val_12 = f"{mean_12[i]:.4f} \\pm {std_12[i]:.4f}"

        lines.append(f"  {group_label} & {DISPLAY_NAMES[feat]} & ${val_01}$ & ${val_12}$ \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {save_path}")


def main():
    print(f"Loading data from {DATA_PATH}")
    doc_ids, depths, feat_dict = load_data()
    print(f"Loaded {len(doc_ids)} samples, depths: {sorted(set(depths))}")

    # 0 vs 1
    print("\n=== 0 vs 1 (human vs first generation) ===")
    ids_01, X_01, y_01 = get_pair_data(doc_ids, depths, feat_dict, 0, 1)
    print(f"  Samples: {len(y_01)} (class 0: {(y_01==0).sum()}, class 1: {(y_01==1).sum()})")
    mean_01, std_01, auc_01 = compute_permutation_importance(X_01, y_01, ids_01)
    print(f"  Baseline AUC: {auc_01:.4f}")
    print("  Top-5 features:")
    top5_01 = np.argsort(mean_01)[::-1][:5]
    for rank, i in enumerate(top5_01, 1):
        print(f"    {rank}. {ALL_FEATURES[i]:20s} {mean_01[i]:.4f} ± {std_01[i]:.4f}")

    # 1 vs 2
    print("\n=== 1 vs 2 (K* boundary) ===")
    ids_12, X_12, y_12 = get_pair_data(doc_ids, depths, feat_dict, 1, 2)
    print(f"  Samples: {len(y_12)} (class 0: {(y_12==0).sum()}, class 1: {(y_12==1).sum()})")
    mean_12, std_12, auc_12 = compute_permutation_importance(X_12, y_12, ids_12)
    print(f"  Baseline AUC: {auc_12:.4f}")
    print("  Top-5 features:")
    top5_12 = np.argsort(mean_12)[::-1][:5]
    for rank, i in enumerate(top5_12, 1):
        print(f"    {rank}. {ALL_FEATURES[i]:20s} {mean_12[i]:.4f} ± {std_12[i]:.4f}")

    # 2 vs 3
    print("\n=== 2 vs 3 (post-boundary, expected indistinguishable) ===")
    ids_23, X_23, y_23 = get_pair_data(doc_ids, depths, feat_dict, 2, 3)
    print(f"  Samples: {len(y_23)} (class 0: {(y_23==0).sum()}, class 1: {(y_23==1).sum()})")
    mean_23, std_23, auc_23 = compute_permutation_importance(X_23, y_23, ids_23)
    print(f"  Baseline AUC: {auc_23:.4f}")
    print("  Top-5 features:")
    top5_23 = np.argsort(mean_23)[::-1][:5]
    for rank, i in enumerate(top5_23, 1):
        print(f"    {rank}. {ALL_FEATURES[i]:20s} {mean_23[i]:.4f} ± {std_23[i]:.4f}")

    # Save results
    results = {
        "data_path": str(DATA_PATH),
        "model": "Qwen-1.5B-cont",
        "domain": "C4",
        "mode": "continuation (retrain chain)",
        "n_repeats": N_REPEATS,
        "n_folds": N_FOLDS,
        "lgb_params": LGB_PARAMS,
        "features": ALL_FEATURES,
        "0v1": {
            "baseline_auc": round(auc_01, 4),
            "importance": {ALL_FEATURES[i]: {"mean": round(float(mean_01[i]), 6), "std": round(float(std_01[i]), 6)} for i in range(len(ALL_FEATURES))},
            "ranking": [ALL_FEATURES[i] for i in np.argsort(mean_01)[::-1]],
        },
        "1v2": {
            "baseline_auc": round(auc_12, 4),
            "importance": {ALL_FEATURES[i]: {"mean": round(float(mean_12[i]), 6), "std": round(float(std_12[i]), 6)} for i in range(len(ALL_FEATURES))},
            "ranking": [ALL_FEATURES[i] for i in np.argsort(mean_12)[::-1]],
        },
        "2v3": {
            "baseline_auc": round(auc_23, 4),
            "importance": {ALL_FEATURES[i]: {"mean": round(float(mean_23[i]), 6), "std": round(float(std_23[i]), 6)} for i in range(len(ALL_FEATURES))},
            "ranking": [ALL_FEATURES[i] for i in np.argsort(mean_23)[::-1]],
        },
    }

    results_path = OUT_DIR / "importance_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {results_path}")

    # Visualizations
    res_0v1 = {"mean": mean_01, "std": std_01}
    res_1v2 = {"mean": mean_12, "std": std_12}

    pdf_path = ART_DIR / "feature_importance_qwen_1.5b_cont.pdf"
    make_barplot(res_0v1, res_1v2, pdf_path)

    tex_path = ART_DIR / "feature_importance_qwen_1.5b_cont.tex"
    make_latex_table(res_0v1, res_1v2, tex_path)

    # Summary
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    print(f"\n1. 0v1 classification AUC: {auc_01:.4f}")
    print(f"   1v2 classification AUC: {auc_12:.4f}")
    print(f"   2v3 classification AUC: {auc_23:.4f}")

    surp_mean_idx = ALL_FEATURES.index("surp_mean")
    print(f"\n2. Mean Surprisal contribution to 0v1: {mean_01[surp_mean_idx]:.4f} "
          f"({mean_01[surp_mean_idx]/mean_01.sum()*100:.1f}% of total positive importance)")

    print(f"\n3. Top feature for 1v2 boundary: {ALL_FEATURES[np.argmax(mean_12)]} "
          f"(importance: {mean_12.max():.4f})")

    print(f"\n4. Importance concentration (Gini-like):")
    for name, imp in [("0v1", mean_01), ("1v2", mean_12), ("2v3", mean_23)]:
        pos_imp = np.maximum(imp, 0)
        if pos_imp.sum() > 0:
            normed = pos_imp / pos_imp.sum()
            gini = 1 - (normed**2).sum()
            print(f"   {name}: entropy={gini:.3f} (higher=more distributed)")


if __name__ == "__main__":
    main()
