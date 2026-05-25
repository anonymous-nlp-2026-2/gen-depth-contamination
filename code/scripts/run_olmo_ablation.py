"""OLMo-1B C4 selfbleu ablation (one-off script)."""
import csv, json, sys, time
from pathlib import Path
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

FEAT_COLS_15D = [
    "surp_mean", "surp_std", "surp_skew", "surp_kurt",
    "surp_d1_mean", "surp_d1_std", "surp_d1_skew",
    "surp_d2_mean", "surp_d2_std",
    "ttr", "hapax_ratio", "self_bleu",
    "freq_kurtosis", "freq_entropy", "low_freq_ratio",
]
FEAT_COLS_14D = [c for c in FEAT_COLS_15D if c != "self_bleu"]

DATA_PATH = Path("/root/autodl-tmp/gen-depth-contamination/data/exp_olmo1b_c4_rerun/features.csv")
N_BOOT = 10000
MAX_DEPTH = 5
THETA = 0.60
SEED = 42

def load_data(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    depths = np.array([int(r["depth"]) for r in rows])
    doc_ids = np.array([int(r["doc_id"]) for r in rows])
    X_15d = np.array(
        [[float(r[c]) if r[c] not in ("", "nan") else np.nan for c in FEAT_COLS_15D] for r in rows]
    )
    X_14d = np.array(
        [[float(r[c]) if r[c] not in ("", "nan") else np.nan for c in FEAT_COLS_14D] for r in rows]
    )
    for X in [X_15d, X_14d]:
        col_means = np.nanmean(X, axis=0)
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            X[mask, j] = col_means[j]
    return X_15d, X_14d, depths, doc_ids

def get_oof_predictions(X, y, groups):
    gkf = GroupKFold(n_splits=5)
    oof_proba = np.zeros(len(y))
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        clf = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            num_leaves=31, verbose=-1, n_jobs=-1,
        )
        clf.fit(X[train_idx], y[train_idx])
        oof_proba[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
    return oof_proba

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

def analyze_pair(X_15d, X_14d, depths, doc_ids, d_low, d_high, rng):
    mask = (depths == d_low) | (depths == d_high)
    X15 = X_15d[mask]
    X14 = X_14d[mask]
    y = (depths[mask] == d_high).astype(int)
    groups = doc_ids[mask]
    oof_15 = get_oof_predictions(X15, y, groups)
    oof_14 = get_oof_predictions(X14, y, groups)
    auc_15, ci15_lo, ci15_hi = bootstrap_ci(y, oof_15, groups, N_BOOT, rng)
    auc_14, ci14_lo, ci14_hi = bootstrap_ci(y, oof_14, groups, N_BOOT, rng)
    return {
        "auc_15d": round(auc_15, 4), "ci_15d_lower": round(ci15_lo, 4), "ci_15d_upper": round(ci15_hi, 4),
        "auc_14d": round(auc_14, 4), "ci_14d_lower": round(ci14_lo, 4), "ci_14d_upper": round(ci14_hi, 4),
        "delta": round(auc_14 - auc_15, 4),
    }

def compute_kstar(pair_results, key="auc_15d"):
    kstar = 0
    for k in range(1, MAX_DEPTH + 1):
        pair_key = f"{k-1}v{k}"
        if pair_key in pair_results and pair_results[pair_key][key] > THETA:
            kstar = k
        else:
            break
    return kstar

def main():
    t0 = time.time()
    print("Loading OLMo-1B C4 rerun data...", flush=True)
    X_15d, X_14d, depths, doc_ids = load_data(DATA_PATH)
    print(f"  {len(depths)} samples, depths 0-{depths.max()}", flush=True)

    rng = np.random.RandomState(SEED)
    pairs = {}
    for d in range(MAX_DEPTH):
        pk = f"{d}v{d+1}"
        print(f"  {pk} ...", end=" ", flush=True)
        t1 = time.time()
        res = analyze_pair(X_15d, X_14d, depths, doc_ids, d, d+1, rng)
        pairs[pk] = res
        print(f"15D={res['auc_15d']:.4f} [{res['ci_15d_lower']:.4f},{res['ci_15d_upper']:.4f}]  "
              f"14D={res['auc_14d']:.4f} [{res['ci_14d_lower']:.4f},{res['ci_14d_upper']:.4f}]  "
              f"Δ={res['delta']:+.4f}  ({time.time()-t1:.1f}s)", flush=True)

    k15 = compute_kstar(pairs, "auc_15d")
    k14 = compute_kstar(pairs, "auc_14d")
    print(f"\nK*_15D={k15}  K*_14D={k14}  ΔK*={k14-k15}", flush=True)
    print(f"Total: {time.time()-t0:.1f}s", flush=True)

    result = {"pairs": pairs, "kstar_15d": k15, "kstar_14d": k14}
    out = Path("/root/autodl-tmp/gen-depth-contamination/artifacts/olmo_selfbleu_ablation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out}", flush=True)

if __name__ == "__main__":
    main()
