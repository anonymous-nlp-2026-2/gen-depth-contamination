"""
OLMo-1B C4 Rerun: Per-pair permutation importance analysis
Compares with existing K*=2 models (LLaMA-8B, Pythia, Qwen) and K*=1 models (old OLMo, Gemma)
"""

import csv
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"

DATA_PATH = Path("/root/autodl-tmp/gen-depth-contamination/data/exp_olmo1b_c4_rerun/features.csv")
OUT_DIR = Path("/root/autodl-tmp/gen-depth-contamination/results/olmo_feature_importance")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_FEATURES = [
    "surp_mean", "surp_std", "surp_skew", "surp_kurt",
    "surp_d1_mean", "surp_d1_std", "surp_d1_skew",
    "surp_d2_mean", "surp_d2_std",
    "ttr", "hapax_ratio", "self_bleu",
    "freq_kurtosis", "freq_entropy", "low_freq_ratio",
]

DISPLAY_NAMES = {
    "surp_mean": r"Surprisal $\mu$",
    "surp_std": r"Surprisal $\sigma$",
    "surp_skew": "Surprisal Skew",
    "surp_kurt": "Surprisal Kurt.",
    "surp_d1_mean": r"$\Delta^1$ Surprisal $\mu$",
    "surp_d1_std": r"$\Delta^1$ Surprisal $\sigma$",
    "surp_d1_skew": r"$\Delta^1$ Surprisal Skew",
    "surp_d2_mean": r"$\Delta^2$ Surprisal $\mu$",
    "surp_d2_std": r"$\Delta^2$ Surprisal $\sigma$",
    "ttr": "TTR",
    "hapax_ratio": "Hapax Ratio",
    "self_bleu": "Self-BLEU",
    "freq_kurtosis": "Freq. Kurt.",
    "freq_entropy": "Freq. Entropy",
    "low_freq_ratio": "Low-Freq Ratio",
}

FEATURE_GROUPS = {
    "surp_mean": "surprisal", "surp_std": "surprisal",
    "surp_skew": "surprisal", "surp_kurt": "surprisal",
    "surp_d1_mean": "delta_surprisal", "surp_d1_std": "delta_surprisal",
    "surp_d1_skew": "delta_surprisal",
    "surp_d2_mean": "delta_surprisal", "surp_d2_std": "delta_surprisal",
    "ttr": "lexical", "hapax_ratio": "lexical", "self_bleu": "lexical",
    "freq_kurtosis": "freq_dist", "freq_entropy": "freq_dist",
    "low_freq_ratio": "freq_dist",
}

GROUP_COLORS = {
    "surprisal": "#2166ac",
    "delta_surprisal": "#4393c3",
    "lexical": "#d6604d",
    "freq_dist": "#f4a582",
}

LGB_PARAMS = dict(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    num_leaves=31, verbose=-1, n_jobs=4,
)

N_REPEATS = 10
N_FOLDS = 5


def load_data(path):
    with open(path) as f:
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

    mean_importance = np.mean(all_importances, axis=0)
    std_importance = np.std(all_importances, axis=0)
    mean_auc = np.mean(fold_aucs)

    return mean_importance, std_importance, mean_auc


def load_comparison_results():
    base = Path("/root/autodl-tmp/gen-depth-contamination/results")
    comparisons = {}

    files = {
        "LLaMA-8B\n(K*=2)": base / "feature_importance" / "importance_results.json",
        "Pythia-1.4B\n(K*=2)": base / "feature_importance_pythia" / "importance_results.json",
        "Qwen-1.5B\n(K*=2)": base / "feature_importance_qwen_cont_corrected" / "importance_results.json",
        "OLMo-1B old\n(K*=1)": base / "feature_importance_olmo" / "importance_results.json",
    }

    for label, path in files.items():
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            comparisons[label] = data
            print(f"  Loaded: {label} -> {path.name}")

    return comparisons


def make_olmo_barplot(res_0v1, res_1v2, res_2v3, pdf_path):
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 8,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    })

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))

    for ax, (key, res, title_label) in zip(axes, [
        ("0v1", res_0v1, "Gen-0 vs Gen-1"),
        ("1v2", res_1v2, "Gen-1 vs Gen-2"),
        ("2v3", res_2v3, "Gen-2 vs Gen-3"),
    ]):
        means = res["mean"]
        stds = res["std"]
        auc = res["auc"]

        sorted_idx = np.argsort(means)
        y_pos = np.arange(len(ALL_FEATURES))

        colors = [GROUP_COLORS[FEATURE_GROUPS[ALL_FEATURES[i]]] for i in sorted_idx]

        ax.barh(y_pos, means[sorted_idx], xerr=stds[sorted_idx], height=0.65,
                color=colors, edgecolor='black', linewidth=0.4,
                error_kw={'linewidth': 0.6, 'capsize': 1.5, 'capthick': 0.6},
                zorder=3)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([DISPLAY_NAMES[ALL_FEATURES[i]] for i in sorted_idx])
        ax.set_xlabel(r'$\Delta$AUC')
        ax.set_title(f'{title_label}\n(AUC = {auc:.3f})', fontsize=9, fontweight='bold')
        ax.axvline(x=0, color='black', linewidth=0.5, zorder=2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout(w_pad=2.0)
    plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
    plt.savefig(str(pdf_path).replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {pdf_path}")


def make_cross_model_comparison(olmo_results, comparisons, pdf_path):
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 8,
        'axes.linewidth': 0.8,
    })

    all_models = {}
    all_models["OLMo-1B rerun\n(K*=1)"] = olmo_results

    for label, data in comparisons.items():
        model_res = {}
        for pair in ["0v1", "1v2"]:
            if pair in data:
                imp = data[pair]["importance"]
                means = np.array([imp[f]["mean"] for f in ALL_FEATURES])
                stds = np.array([imp[f]["std"] for f in ALL_FEATURES])
                auc = data[pair]["baseline_auc"]
                model_res[pair] = {"mean": means, "std": stds, "auc": auc}
        all_models[label] = model_res

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5))

    model_labels = list(all_models.keys())
    n_models = len(model_labels)
    n_features = len(ALL_FEATURES)

    for ax_idx, pair in enumerate(["0v1", "1v2"]):
        ax = axes[ax_idx]

        bar_width = 0.8 / n_models
        feature_order = list(range(n_features))

        if pair == "1v2":
            ref_means = all_models[model_labels[0]][pair]["mean"] if pair in all_models[model_labels[0]] else np.zeros(n_features)
            feature_order = sorted(range(n_features), key=lambda i: ref_means[i])

        model_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

        for m_idx, label in enumerate(model_labels):
            if pair not in all_models[label]:
                continue
            means = all_models[label][pair]["mean"]
            stds = all_models[label][pair]["std"]

            y_positions = np.arange(n_features) + (m_idx - n_models/2 + 0.5) * bar_width

            ax.barh(y_positions, means[feature_order], xerr=stds[feature_order],
                    height=bar_width * 0.9,
                    color=model_colors[m_idx % len(model_colors)],
                    edgecolor='black', linewidth=0.3,
                    error_kw={'linewidth': 0.4, 'capsize': 1.0, 'capthick': 0.4},
                    label=f"{label} (AUC={all_models[label][pair]['auc']:.3f})",
                    zorder=3)

        ax.set_yticks(np.arange(n_features))
        ax.set_yticklabels([DISPLAY_NAMES[ALL_FEATURES[i]] for i in feature_order], fontsize=7)
        ax.set_xlabel(r'$\Delta$AUC (permutation importance)')
        pair_label = "Gen-0 vs Gen-1" if pair == "0v1" else "Gen-1 vs Gen-2"
        ax.set_title(pair_label, fontsize=10, fontweight='bold')
        ax.axvline(x=0, color='black', linewidth=0.5, zorder=2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='lower right', fontsize=6, framealpha=0.9)

    plt.tight_layout(w_pad=2.0)
    plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
    plt.savefig(str(pdf_path).replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {pdf_path}")


def main():
    print("Loading OLMo-1B C4 rerun features...")
    doc_ids, depths, feat_dict = load_data(DATA_PATH)
    print(f"  Total samples: {len(doc_ids)}, depths: {sorted(set(depths))}")

    results = {
        "data_path": str(DATA_PATH),
        "model": "OLMo-1B",
        "dataset": "C4",
        "label": "OLMo-1B C4 Rerun",
        "n_repeats": N_REPEATS,
        "n_folds": N_FOLDS,
        "lgb_params": LGB_PARAMS,
        "features": ALL_FEATURES,
    }

    olmo_results = {}

    for d_low, d_high, pair_key in [(0, 1, "0v1"), (1, 2, "1v2"), (2, 3, "2v3")]:
        print(f"\n=== {pair_key}: depth {d_low} vs {d_high} ===")
        ids, X, y = get_pair_data(doc_ids, depths, feat_dict, d_low, d_high)
        print(f"  Samples: {len(y)} (class 0: {(y==0).sum()}, class 1: {(y==1).sum()})")

        mean_imp, std_imp, auc = compute_permutation_importance(X, y, ids)
        print(f"  Baseline AUC: {auc:.4f}")

        olmo_results[pair_key] = {"mean": mean_imp, "std": std_imp, "auc": auc}

        print("  Top-5 features:")
        top5 = np.argsort(mean_imp)[::-1][:5]
        for rank, i in enumerate(top5, 1):
            print(f"    {rank}. {ALL_FEATURES[i]:20s} {mean_imp[i]:.6f} +/- {std_imp[i]:.6f}")

        print("  Bottom-5 features (negative/zero importance):")
        bot5 = np.argsort(mean_imp)[:5]
        for rank, i in enumerate(bot5, 1):
            print(f"    {rank}. {ALL_FEATURES[i]:20s} {mean_imp[i]:.6f} +/- {std_imp[i]:.6f}")

        results[pair_key] = {
            "baseline_auc": round(auc, 4),
            "importance": {
                ALL_FEATURES[i]: {"mean": round(float(mean_imp[i]), 6), "std": round(float(std_imp[i]), 6)}
                for i in range(len(ALL_FEATURES))
            },
            "ranking": [ALL_FEATURES[i] for i in np.argsort(mean_imp)[::-1]],
        }

    # Save JSON
    results_path = OUT_DIR / "importance_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {results_path}")

    # Figure 1: OLMo barplots
    make_olmo_barplot(olmo_results["0v1"], olmo_results["1v2"], olmo_results["2v3"],
                      OUT_DIR / "olmo_rerun_feature_importance.pdf")

    # Load comparison data
    print("\nLoading comparison models...")
    comparisons = load_comparison_results()

    # Figure 2: Cross-model comparison
    make_cross_model_comparison(olmo_results, comparisons,
                                OUT_DIR / "cross_model_feature_comparison.pdf")

    # Analysis summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)

    print(f"\n1. OLMo Rerun AUC: 0v1={olmo_results['0v1']['auc']:.4f}, "
          f"1v2={olmo_results['1v2']['auc']:.4f}, 2v3={olmo_results['2v3']['auc']:.4f}")

    print("\n2. 1v2 Feature Importance Pattern (OLMo rerun):")
    imp_1v2 = olmo_results["1v2"]["mean"]
    n_positive = (imp_1v2 > 0).sum()
    n_negative = (imp_1v2 < 0).sum()
    n_near_zero = ((np.abs(imp_1v2) < 0.002)).sum()
    print(f"   Positive: {n_positive}, Negative: {n_negative}, Near-zero (<0.002): {n_near_zero}")
    print(f"   Max importance: {imp_1v2.max():.6f} ({ALL_FEATURES[np.argmax(imp_1v2)]})")
    print(f"   Total positive importance: {imp_1v2[imp_1v2>0].sum():.6f}")

    print("\n3. Comparison with K*=2 models (1v2 total positive importance):")
    for label, data in comparisons.items():
        if "1v2" in data:
            imp = data["1v2"]["importance"]
            total_pos = sum(v["mean"] for v in imp.values() if v["mean"] > 0)
            auc = data["1v2"]["baseline_auc"]
            print(f"   {label}: AUC={auc:.4f}, total_pos_imp={total_pos:.6f}")

    olmo_total = imp_1v2[imp_1v2>0].sum()
    print(f"   OLMo rerun: AUC={olmo_results['1v2']['auc']:.4f}, total_pos_imp={olmo_total:.6f}")

    print("\n4. Group-level importance comparison (1v2):")
    groups = {"surprisal": [], "delta_surprisal": [], "lexical": [], "freq_dist": []}
    for i, f in enumerate(ALL_FEATURES):
        groups[FEATURE_GROUPS[f]].append(imp_1v2[i])
    for g, vals in groups.items():
        print(f"   {g}: mean={np.mean(vals):.6f}, max={np.max(vals):.6f}")


if __name__ == "__main__":
    main()
