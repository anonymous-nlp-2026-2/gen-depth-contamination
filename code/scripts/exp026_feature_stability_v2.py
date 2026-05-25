"""exp-026: Feature Subset Stability Ablation"""
import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

ALL_FEAT_COLS = [
    "surp_mean", "surp_std", "surp_skew", "surp_kurt",
    "surp_d1_mean", "surp_d1_std", "surp_d1_skew",
    "surp_d2_mean", "surp_d2_std",
    "ttr", "hapax_ratio", "self_bleu",
    "freq_kurtosis", "freq_entropy", "low_freq_ratio",
]
TOP3_FEATURES = ["surp_d1_std", "surp_std", "surp_d2_std"]
THETA = 0.60
SEED = 42
DATA_PATH = Path("/root/autodl-tmp/gen-depth-contamination/data_exp016_qwen_base/features.csv")
OUT_PATH = Path("/root/autodl-tmp/gen-depth-contamination/artifacts/exp026_feature_stability_v2.json")

SUBSET_CONFIGS = [
    (15, 1),
    (10, 5),
    (8, 5),
    (5, 5),
    (3, 1),
]


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df[ALL_FEAT_COLS] = df[ALL_FEAT_COLS].fillna(df[ALL_FEAT_COLS].mean())
    print(f"Loaded {len(df)} rows, depths: {sorted(df['depth'].unique())}")

    rng = np.random.default_rng(SEED)
    all_indices = list(range(len(ALL_FEAT_COLS)))
    results = []

    for subset_size, n_trials in SUBSET_CONFIGS:
        print(f"\n--- subset_size={subset_size}, n_trials={n_trials} ---")
        
        if subset_size == 15:
            subsets = [all_indices]
        elif subset_size == 3:
            subsets = [[ALL_FEAT_COLS.index(f) for f in TOP3_FEATURES]]
        else:
            subsets = [sorted(rng.choice(all_indices, size=subset_size, replace=False).tolist()) for _ in range(n_trials)]

        trial_results = []
        for i, feat_idx in enumerate(subsets):
            feat_names = [ALL_FEAT_COLS[j] for j in feat_idx]
            pairwise = {}
            for k in range(1, 5):
                pair_df = df[df["depth"].isin([k-1, k])]
                X = pair_df[feat_names].values
                y = (pair_df["depth"].values == k).astype(int)
                
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
                fold_aucs = []
                for train_idx, test_idx in skf.split(X, y):
                    clf = lgb.LGBMClassifier(
                        n_estimators=100, max_depth=5, learning_rate=0.1,
                        num_leaves=31, verbose=-1, n_jobs=4,
                    )
                    clf.fit(X[train_idx], y[train_idx])
                    proba = clf.predict_proba(X[test_idx])[:, 1]
                    fold_aucs.append(roc_auc_score(y[test_idx], proba))
                pairwise[f"{k-1}v{k}"] = float(np.mean(fold_aucs))

            kstar = 0
            for k in range(1, 5):
                if pairwise[f"{k-1}v{k}"] > THETA:
                    kstar = k
                else:
                    break

            trial_results.append({
                "features": feat_names,
                "pairwise_auc": pairwise,
                "kstar": kstar,
                "auc_1v2": pairwise["1v2"],
            })
            print(f"  trial {i}: K*={kstar}, 1v2={pairwise['1v2']:.4f}, 0v1={pairwise['0v1']:.4f}")

        num_kstar2 = sum(1 for t in trial_results if t["kstar"] == 2)
        auc_1v2_vals = [t["auc_1v2"] for t in trial_results]
        entry = {
            "subset_size": subset_size,
            "num_tested": len(subsets),
            "num_kstar_2": num_kstar2,
            "min_1v2_auc": round(min(auc_1v2_vals), 4),
            "mean_1v2_auc": round(float(np.mean(auc_1v2_vals)), 4),
            "max_1v2_auc": round(max(auc_1v2_vals), 4),
            "trials": trial_results,
        }
        results.append(entry)

    print("\n" + "="*70)
    print(f"{'subset_size':<12}| {'num_tested':<11}| {'num_K*=2':<9}| {'min_1v2_AUC':<12}| {'mean_1v2_AUC':<13}")
    print("-"*70)
    for r in results:
        print(f"{r['subset_size']:<12}| {r['num_tested']:<11}| {r['num_kstar_2']:<9}| {r['min_1v2_auc']:<12.4f}| {r['mean_1v2_auc']:<13.4f}")
    print("="*70)
    print(f"Theta = {THETA}")
    all_pass = all(r["min_1v2_auc"] > THETA for r in results)
    print(f"All subsets 1v2 AUC > theta? {'YES' if all_pass else 'NO'}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "exp_026_feature_stability",
        "theta": THETA,
        "seed": SEED,
        "data_path": str(DATA_PATH),
        "all_1v2_above_theta": all_pass,
        "summary": [{k: v for k, v in r.items() if k != "trials"} for r in results],
        "details": results,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
