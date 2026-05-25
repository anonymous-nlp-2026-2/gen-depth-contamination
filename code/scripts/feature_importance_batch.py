"""
Batch Feature Importance Re-run with Corrected Data Sources
Runs permutation importance for 4 models: Qwen-base, OLMo, Pythia, Qwen-cont
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

os.environ["OMP_NUM_THREADS"] = "4"

ROOT = Path("/root/autodl-tmp/gen-depth-contamination")
ART_DIR = ROOT / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "qwen_base": {
        "data": ROOT / "data_exp016_qwen_base" / "features.csv",
        "out": ROOT / "results" / "feature_importance_qwen_base",
        "target_1v2": 0.6262,
        "label": "Qwen-1.5B (rewrite)",
    },
    "olmo": {
        "data": ROOT / "data" / "olmo_c4_qwen_scorer" / "features.csv",
        "out": ROOT / "results" / "feature_importance_olmo_corrected",
        "target_1v2": 0.6145,
        "label": "OLMo-1B",
    },
    "pythia": {
        "data": ROOT / "data" / "pythia_c4_qwen_scorer" / "features.csv",
        "out": ROOT / "results" / "feature_importance_pythia_corrected",
        "target_1v2": 0.6539,
        "label": "Pythia-1.4B",
    },
    "qwen_cont": {
        "data": ROOT / "data_exp022_cross_scorer" / "qwen_cont256__qwen_base" / "features.csv",
        "out": ROOT / "results" / "feature_importance_qwen_cont_corrected",
        "target_1v2": 0.6112,
        "label": "Qwen-1.5B (cont)",
    },
}

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
    "surp_mean": "Mean Surprisal", "surp_std": "Surprisal Std",
    "surp_skew": "Surprisal Skew", "surp_kurt": "Surprisal Kurtosis",
    "surp_d1_mean": "Δ¹ Surprisal Mean", "surp_d1_std": "Δ¹ Surprisal Std",
    "surp_d1_skew": "Δ¹ Surprisal Skew",
    "surp_d2_mean": "Δ² Surprisal Mean", "surp_d2_std": "Δ² Surprisal Std",
    "ttr": "TTR", "hapax_ratio": "Hapax Ratio", "self_bleu": "Self-BLEU",
    "freq_kurtosis": "Freq Kurtosis", "freq_entropy": "Freq Entropy",
    "low_freq_ratio": "Low-Freq Ratio",
}

GROUP_COLORS = {
    "surprisal_base": "#2196F3",
    "surprisal_deriv": "#1565C0",
    "lexical_diversity": "#4CAF50",
    "freq_distribution": "#FF9800",
}

LGB_PARAMS = dict(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    num_leaves=31, verbose=-1, n_jobs=4,
)

N_REPEATS = 10
N_FOLDS = 5


def load_data(data_path):
    with open(data_path) as f:
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
            n_jobs=4,
        )
        all_importances.append(result.importances_mean)

    mean_imp = np.mean(all_importances, axis=0)
    std_imp = np.std(all_importances, axis=0)
    mean_auc = np.mean(fold_aucs)
    return mean_imp, std_imp, mean_auc


def make_barplot(res_0v1, res_1v2, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, res, title in zip(axes, [res_0v1, res_1v2], ["0 vs 1", "1 vs 2"]):
        mean, std = res["mean"], res["std"]
        order = np.argsort(mean)[::-1]
        names = [DISPLAY_NAMES[ALL_FEATURES[i]] for i in order]
        colors = [GROUP_COLORS[FEATURE_GROUPS[ALL_FEATURES[i]]] for i in order]
        
        ax.barh(range(len(names)), mean[order], xerr=std[order],
                color=colors, edgecolor="white", linewidth=0.5, capsize=2)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Permutation Importance (ΔAUC)")
        ax.set_title(f"Depth {title}")
        ax.axvline(x=0, color="gray", linewidth=0.5, linestyle="--")
    
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def make_latex_table(res_0v1, res_1v2, save_path):
    order_12 = np.argsort(res_1v2["mean"])[::-1]
    
    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Feature & 0\,vs\,1 & 1\,vs\,2 \\",
        r"\midrule",
    ]
    for i in order_12:
        name = DISPLAY_NAMES[ALL_FEATURES[i]]
        v01 = res_0v1["mean"][i]
        v12 = res_1v2["mean"][i]
        s01 = res_0v1["std"][i]
        s12 = res_1v2["std"][i]
        lines.append(f"  {name} & {v01:.4f} $\\pm$ {s01:.4f} & {v12:.4f} $\\pm$ {s12:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    
    with open(save_path, "w") as f:
        f.write("\n".join(lines))


def run_single_model(model_key, model_cfg):
    print(f"\n{'='*60}")
    print(f"Model: {model_cfg['label']}")
    print(f"Data: {model_cfg['data']}")
    print(f"{'='*60}")
    
    if not model_cfg["data"].exists():
        print(f"  SKIPPED: features file not found")
        return None
    
    out_dir = model_cfg["out"]
    out_dir.mkdir(parents=True, exist_ok=True)
    
    doc_ids, depths, feat_dict = load_data(model_cfg["data"])
    print(f"  Loaded {len(doc_ids)} rows, depths {sorted(set(depths))}")
    
    # 0 vs 1
    ids_01, X_01, y_01 = get_pair_data(doc_ids, depths, feat_dict, 0, 1)
    mean_01, std_01, auc_01 = compute_permutation_importance(X_01, y_01, ids_01)
    print(f"  0v1 AUC: {auc_01:.4f}")
    
    # 1 vs 2
    ids_12, X_12, y_12 = get_pair_data(doc_ids, depths, feat_dict, 1, 2)
    mean_12, std_12, auc_12 = compute_permutation_importance(X_12, y_12, ids_12)
    print(f"  1v2 AUC: {auc_12:.4f} (target: {model_cfg['target_1v2']:.4f}, Δ={abs(auc_12 - model_cfg['target_1v2']):.4f})")
    
    auc_ok = bool(abs(auc_12 - model_cfg["target_1v2"]) <= 0.01)
    print(f"  AUC check: {'PASS' if auc_ok else 'FAIL'} (tolerance ±0.01)")
    
    # 2 vs 3
    ids_23, X_23, y_23 = get_pair_data(doc_ids, depths, feat_dict, 2, 3)
    mean_23, std_23, auc_23 = compute_permutation_importance(X_23, y_23, ids_23)
    print(f"  2v3 AUC: {auc_23:.4f}")
    
    # Save results
    results = {
        "model": model_key,
        "label": model_cfg["label"],
        "data_path": str(model_cfg["data"]),
        "n_repeats": N_REPEATS,
        "n_folds": N_FOLDS,
        "lgb_params": LGB_PARAMS,
        "features": ALL_FEATURES,
        "target_1v2": model_cfg["target_1v2"],
        "auc_check_passed": auc_ok,
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
    
    results_path = out_dir / "importance_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {results_path}")
    
    # Barplot
    res_0v1 = {"mean": mean_01, "std": std_01}
    res_1v2 = {"mean": mean_12, "std": std_12}
    
    pdf_path = out_dir / "importance_barplot.pdf"
    make_barplot(res_0v1, res_1v2, pdf_path)
    print(f"  Plot: {pdf_path}")
    
    # LaTeX table
    tex_path = out_dir / "importance_table.tex"
    make_latex_table(res_0v1, res_1v2, tex_path)
    print(f"  Table: {tex_path}")
    
    # Top-5 summary
    print(f"\n  Top-5 features for 1v2:")
    top5 = np.argsort(mean_12)[::-1][:5]
    for rank, i in enumerate(top5, 1):
        print(f"    {rank}. {ALL_FEATURES[i]:20s} {mean_12[i]:.4f} ± {std_12[i]:.4f}")
    
    return results


def main():
    # Run specific models if given as arguments, else all
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(MODELS.keys())
    
    all_results = {}
    for key in targets:
        if key not in MODELS:
            print(f"Unknown model: {key}")
            continue
        result = run_single_model(key, MODELS[key])
        if result:
            all_results[key] = result
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for key, res in all_results.items():
        label = res["label"]
        auc_12 = res["1v2"]["baseline_auc"]
        target = res["target_1v2"]
        status = "✓" if res["auc_check_passed"] else "✗"
        top_feat = res["1v2"]["ranking"][0]
        print(f"  {status} {label:25s} 1v2={auc_12:.4f} (target {target:.4f}) top={top_feat}")


if __name__ == "__main__":
    main()
