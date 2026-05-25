"""
C4 subsample control: sample n=2200 doc chains from C4 (exp016),
run GroupKFold(5) pairwise LightGBM, report K* with bootstrap CI.
Validates that C4 K*=2 is not an artifact of larger sample size vs arXiv n=2200.
"""
import csv
import json
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

FEAT_COLS = [
    "surp_mean", "surp_std", "surp_skew", "surp_kurt",
    "surp_d1_mean", "surp_d1_std", "surp_d1_skew",
    "surp_d2_mean", "surp_d2_std",
    "ttr", "hapax_ratio", "self_bleu",
    "freq_kurtosis", "freq_entropy", "low_freq_ratio",
]

DATA_PATH = Path("/root/autodl-tmp/gen-depth-contamination/data_exp016_qwen_base/features.csv")
OUT_PATH = Path("/root/autodl-tmp/gen-depth-contamination/results/c4_subsample_control.json")
SUBSAMPLE_N = 2200
MAX_DEPTH = 5
THETA = 0.60
N_BOOT = 10000
SEED = 42

def load_data():
    with open(DATA_PATH) as f:
        rows = list(csv.DictReader(f))
    depths = np.array([int(r["depth"]) for r in rows])
    doc_ids = np.array([int(r["doc_id"]) for r in rows])
    X = np.array(
        [[float(r[c]) if r[c] not in ("", "nan") else np.nan for c in FEAT_COLS] for r in rows]
    )
    col_means = np.nanmean(X, axis=0)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        X[mask, j] = col_means[j]
    return X, depths, doc_ids

def get_oof_predictions(X, y, groups):
    gkf = GroupKFold(n_splits=5)
    oof_proba = np.zeros(len(y))
    fold_aucs = []
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        clf = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            num_leaves=31, verbose=-1, n_jobs=-1,
        )
        clf.fit(X[train_idx], y[train_idx])
        proba = clf.predict_proba(X[test_idx])[:, 1]
        oof_proba[test_idx] = proba
        fold_aucs.append(roc_auc_score(y[test_idx], proba))
    return oof_proba, fold_aucs

def bootstrap_ci(y, oof_proba, doc_ids, n_boot, rng):
    unique_docs = np.unique(doc_ids)
    n_docs = len(unique_docs)
    doc_indices = [np.where(doc_ids == d)[0] for d in unique_docs]
    point_auc = roc_auc_score(y, oof_proba)
    boot_aucs = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.randint(0, n_docs, size=n_docs)
        indices = np.concatenate([doc_indices[s] for s in sampled])
        try:
            boot_aucs[b] = roc_auc_score(y[indices], oof_proba[indices])
        except ValueError:
            boot_aucs[b] = np.nan
    boot_aucs = boot_aucs[~np.isnan(boot_aucs)]
    ci_lower = float(np.percentile(boot_aucs, 2.5))
    ci_upper = float(np.percentile(boot_aucs, 97.5))
    return point_auc, ci_lower, ci_upper

def main():
    t_start = time.time()
    print("Loading data...", flush=True)
    X, depths, doc_ids = load_data()
    print(f"Full dataset: {len(depths)} samples, {len(np.unique(doc_ids))} unique docs", flush=True)

    rng = np.random.RandomState(SEED)
    unique_docs = np.unique(doc_ids)
    sampled_docs = set(rng.choice(unique_docs, SUBSAMPLE_N, replace=False))
    mask = np.array([d in sampled_docs for d in doc_ids])
    X, depths, doc_ids = X[mask], depths[mask], doc_ids[mask]
    print(f"Subsampled: {len(depths)} samples, {len(np.unique(doc_ids))} unique docs", flush=True)
    print(f"Docs per depth: { {int(d): int((depths==d).sum()) for d in sorted(np.unique(depths))} }", flush=True)

    results = {}
    print(f"\n{'Pair':<8} {'AUC':<8} {'95% CI':<22} {'CI_lo > theta?'}", flush=True)
    print("-" * 55, flush=True)

    k_star = 0
    for k in range(1, MAX_DEPTH + 1):
        pair = f"{k-1}v{k}"
        pair_mask = (depths == k - 1) | (depths == k)
        X_pair = X[pair_mask]
        y_pair = (depths[pair_mask] == k).astype(int)
        groups_pair = doc_ids[pair_mask]

        oof_proba, fold_aucs = get_oof_predictions(X_pair, y_pair, groups_pair)
        point_auc, ci_lower, ci_upper = bootstrap_ci(
            y_pair, oof_proba, groups_pair, N_BOOT, rng
        )

        passes = ci_lower > THETA
        if passes:
            k_star = k

        results[pair] = {
            "auc": round(point_auc, 4),
            "auc_mean_folds": round(float(np.mean(fold_aucs)), 4),
            "fold_aucs": [round(a, 4) for a in fold_aucs],
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
        }

        status = "PASS" if passes else "FAIL"
        print(f"{pair:<8} {point_auc:.4f}   [{ci_lower:.4f}, {ci_upper:.4f}]   {status}", flush=True)

    total_time = time.time() - t_start
    print(f"\nK* = {k_star} (theta={THETA}, n={SUBSAMPLE_N})", flush=True)
    print(f"Total time: {total_time:.1f}s", flush=True)

    output = {
        "experiment": "c4_subsample_control",
        "source": "exp016_qwen_base (C4)",
        "subsample_n": SUBSAMPLE_N,
        "full_n": 5000,
        "seed": SEED,
        "theta": THETA,
        "n_bootstrap": N_BOOT,
        "k_star": k_star,
        "total_seconds": round(total_time, 1),
        "pairs": results,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUT_PATH}", flush=True)

if __name__ == "__main__":
    main()
