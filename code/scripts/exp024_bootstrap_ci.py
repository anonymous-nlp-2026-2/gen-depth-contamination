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

DATA_DIR = Path("/root/autodl-tmp/gen-depth-contamination/data/exp_024_arxiv")
OUT_PATH = Path("/root/autodl-tmp/gen-depth-contamination/artifacts/exp024_bootstrap_ci.json")
N_BOOT = 10000
N_PERM = 10000
MAX_DEPTH = 5
SEED = 42
THETA = 0.60

def load_data():
    with open(DATA_DIR / "features.csv") as f:
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
    return point_auc, ci_lower, ci_upper, boot_aucs

def permutation_test(y, oof_proba, doc_ids, n_perm, rng, observed_auc):
    unique_docs, inverse = np.unique(doc_ids, return_inverse=True)
    n_docs = len(unique_docs)
    
    count_ge = 0
    for _ in range(n_perm):
        doc_labels = rng.randint(0, 2, size=n_docs)
        perm_y = doc_labels[inverse]
        if perm_y.sum() == 0 or perm_y.sum() == len(perm_y):
            continue
        perm_auc = roc_auc_score(perm_y, oof_proba)
        if perm_auc >= observed_auc:
            count_ge += 1
    return (count_ge + 1) / (n_perm + 1)

def bootstrap_test_vs_theta(boot_aucs, theta=THETA):
    p_below = float(np.mean(boot_aucs <= theta))
    return p_below

def main():
    t_start = time.time()
    print("Loading data...", flush=True)
    X, depths, doc_ids = load_data()
    print(f"Loaded {len(depths)} samples, depths 0-{depths.max()}", flush=True)
    
    rng = np.random.RandomState(SEED)
    results = {}
    
    for k in range(1, MAX_DEPTH + 1):
        pair = f"{k-1}v{k}"
        t0 = time.time()
        print(f"\n{'='*50}", flush=True)
        print(f"Processing {pair}...", flush=True)
        
        mask = (depths == k - 1) | (depths == k)
        X_pair = X[mask]
        y_pair = (depths[mask] == k).astype(int)
        groups_pair = doc_ids[mask]
        
        print(f"  Samples: {len(y_pair)} (class 0: {(y_pair==0).sum()}, class 1: {(y_pair==1).sum()})", flush=True)
        
        print("  Training 5-fold GroupKFold...", flush=True)
        oof_proba, fold_aucs = get_oof_predictions(X_pair, y_pair, groups_pair)
        print(f"  Fold AUCs: {[round(a,4) for a in fold_aucs]}", flush=True)
        print(f"  Mean AUC: {np.mean(fold_aucs):.4f}", flush=True)
        
        print(f"  Running {N_BOOT} bootstrap resamples...", flush=True)
        point_auc, ci_lower, ci_upper, boot_aucs = bootstrap_ci(
            y_pair, oof_proba, groups_pair, N_BOOT, rng
        )
        print(f"  Point AUC (OOF): {point_auc:.4f}", flush=True)
        print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]", flush=True)
        
        print(f"  Running {N_PERM} permutation resamples...", flush=True)
        perm_p = permutation_test(y_pair, oof_proba, groups_pair, N_PERM, rng, point_auc)
        print(f"  Permutation p-value (AUC >= observed under H0): {perm_p:.6f}", flush=True)
        
        p_vs_theta = bootstrap_test_vs_theta(boot_aucs, THETA)
        print(f"  Bootstrap p(AUC <= {THETA}): {p_vs_theta:.4f}", flush=True)
        
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s", flush=True)
        
        results[pair] = {
            "auc": round(point_auc, 4),
            "auc_mean_folds": round(float(np.mean(fold_aucs)), 4),
            "fold_aucs": [round(a, 4) for a in fold_aucs],
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "ci_lower_above_theta": bool(ci_lower > THETA),
            "perm_p": round(perm_p, 6),
            "p_vs_theta_0.60": round(p_vs_theta, 4),
            "boot_mean": round(float(np.mean(boot_aucs)), 4),
            "boot_std": round(float(np.std(boot_aucs)), 4),
        }
    
    total_time = time.time() - t_start
    
    kstar = 0
    for k in range(1, MAX_DEPTH + 1):
        pair = f"{k-1}v{k}"
        if results[pair]["ci_lower_above_theta"]:
            kstar = k
        else:
            break
    
    output = {
        "experiment": "exp-024 Qwen2.5-1.5B-base arXiv continuation",
        "method": "GroupKFold(5) OOF + doc-level bootstrap + label permutation",
        "n_bootstrap": N_BOOT,
        "n_permutation": N_PERM,
        "seed": SEED,
        "theta": THETA,
        "kstar_ci": kstar,
        "total_seconds": round(total_time, 1),
        "pairs": results,
    }
    
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUT_PATH}", flush=True)
    
    print("\n" + "="*60, flush=True)
    print("SUMMARY", flush=True)
    print("="*60, flush=True)
    print(f"Theta = {THETA}", flush=True)
    print(f"K* (CI-based) = {kstar}", flush=True)
    print(flush=True)
    for pair, r in results.items():
        if r["ci_lower_above_theta"]:
            flag = " [CI lower > theta -> PASS]"
        else:
            flag = " [CI lower <= theta -> FAIL]"
        print(f"  {pair}: AUC={r['auc']:.4f}  95% CI=[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]  p_perm={r['perm_p']:.6f}  p_vs_theta={r['p_vs_theta_0.60']:.4f}{flag}", flush=True)
    
    print(f"\nTotal time: {total_time:.1f}s", flush=True)

if __name__ == "__main__":
    main()
