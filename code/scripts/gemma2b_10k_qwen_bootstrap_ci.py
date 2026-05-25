"""Bootstrap CI + permutation test for exp_gemma2b_10k_qwen_scorer (all pairs).
10000 bootstrap iterations, 95% CI + Bonferroni 99.5% CI (10 configs, alpha=0.005, z=2.807).
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

DATA_DIR = Path("/root/autodl-tmp/gen-depth-contamination/data/exp_gemma2b_10k_qwen_scorer")
OUT_PATH = Path("/root/autodl-tmp/gen-depth-contamination/artifacts/gemma2b_10k_qwen_bootstrap_ci.json")
SEED = 42
N_BOOT = 10000
N_PERM = 10000
THETA = 0.60

BONFERRONI_Z = 2.807


def load_data(data_dir):
    with open(Path(data_dir) / "features.csv") as f:
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
    boot_aucs = []
    for _ in range(n_boot):
        boot_docs = rng.choice(n_docs, size=n_docs, replace=True)
        idx = np.concatenate([doc_indices[d] for d in boot_docs])
        try:
            auc = roc_auc_score(y[idx], oof_proba[idx])
            boot_aucs.append(auc)
        except ValueError:
            continue

    boot_aucs = np.array(boot_aucs)
    ci_lower_95 = float(np.percentile(boot_aucs, 2.5))
    ci_upper_95 = float(np.percentile(boot_aucs, 97.5))

    # Bonferroni 99.5% CI (percentile method)
    ci_lower_bonf_pct = float(np.percentile(boot_aucs, 0.25))
    ci_upper_bonf_pct = float(np.percentile(boot_aucs, 99.75))

    # Bonferroni 99.5% CI (normal approximation)
    boot_mean = float(np.mean(boot_aucs))
    boot_std = float(np.std(boot_aucs))
    ci_lower_bonf_norm = boot_mean - BONFERRONI_Z * boot_std
    ci_upper_bonf_norm = boot_mean + BONFERRONI_Z * boot_std

    return (point_auc, ci_lower_95, ci_upper_95,
            ci_lower_bonf_pct, ci_upper_bonf_pct,
            ci_lower_bonf_norm, ci_upper_bonf_norm,
            boot_aucs)


def permutation_test(y, oof_proba, doc_ids, n_perm, rng, observed_auc):
    unique_docs = np.unique(doc_ids)
    doc_indices = {d: np.where(doc_ids == d)[0] for d in unique_docs}
    count_ge = 0
    for _ in range(n_perm):
        perm_y = y.copy()
        for d in unique_docs:
            idx = doc_indices[d]
            if rng.random() < 0.5:
                perm_y[idx] = 1 - perm_y[idx]
        perm_auc = roc_auc_score(perm_y, oof_proba)
        if perm_auc >= observed_auc:
            count_ge += 1
    return (count_ge + 1) / (n_perm + 1)


def main():
    t_start = time.time()
    rng = np.random.RandomState(SEED)

    print("Loading data ...", flush=True)
    X, depths, doc_ids = load_data(DATA_DIR)
    max_depth = int(depths.max())
    print(f"Loaded {len(depths)} samples, depths 0-{max_depth}", flush=True)

    pairs = [(k-1, k) for k in range(1, max_depth + 1)]
    all_results = {}

    for d0, d1 in pairs:
        pair_name = f"{d0}v{d1}"
        print(f"\n{'='*60}", flush=True)
        print(f"Pair: {pair_name}", flush=True)
        print(f"{'='*60}", flush=True)

        t0 = time.time()
        mask = (depths == d0) | (depths == d1)
        X_pair = X[mask]
        y_pair = (depths[mask] == d1).astype(int)
        groups_pair = doc_ids[mask]

        n0 = int((y_pair == 0).sum())
        n1 = int((y_pair == 1).sum())
        print(f"  Samples: {len(y_pair)} (class 0: {n0}, class 1: {n1})", flush=True)

        oof_proba, fold_aucs = get_oof_predictions(X_pair, y_pair, groups_pair)
        print(f"  Fold AUCs: {[round(a,4) for a in fold_aucs]}", flush=True)
        print(f"  Mean AUC: {np.mean(fold_aucs):.4f}", flush=True)

        (point_auc, ci95_lo, ci95_hi,
         ci_bonf_pct_lo, ci_bonf_pct_hi,
         ci_bonf_norm_lo, ci_bonf_norm_hi,
         boot_aucs) = bootstrap_ci(y_pair, oof_proba, groups_pair, N_BOOT, rng)

        print(f"  Point AUC (OOF): {point_auc:.4f}", flush=True)
        print(f"  Bootstrap mean: {np.mean(boot_aucs):.4f}, std: {np.std(boot_aucs):.4f}", flush=True)
        print(f"  95% CI: [{ci95_lo:.4f}, {ci95_hi:.4f}]", flush=True)
        print(f"  Bonferroni 99.5% CI (percentile): [{ci_bonf_pct_lo:.4f}, {ci_bonf_pct_hi:.4f}]", flush=True)
        print(f"  Bonferroni 99.5% CI (normal): [{ci_bonf_norm_lo:.4f}, {ci_bonf_norm_hi:.4f}]", flush=True)

        perm_p = permutation_test(y_pair, oof_proba, groups_pair, N_PERM, rng, point_auc)
        print(f"  Permutation p-value: {perm_p:.6f}", flush=True)

        bonf_pass_pct = bool(ci_bonf_pct_lo > THETA) if point_auc > THETA else False
        bonf_pass_norm = bool(ci_bonf_norm_lo > THETA) if point_auc > THETA else False
        ci95_pass = bool(ci95_lo > THETA) if point_auc > THETA else False

        print(f"  95% CI lower > θ=0.60: {ci95_pass}", flush=True)
        print(f"  Bonferroni lower (pct) > θ=0.60: {bonf_pass_pct}", flush=True)
        print(f"  Bonferroni lower (norm) > θ=0.60: {bonf_pass_norm}", flush=True)

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s", flush=True)

        all_results[pair_name] = {
            "depth_pair": pair_name,
            "auc_oof": round(point_auc, 4),
            "auc_mean_folds": round(float(np.mean(fold_aucs)), 4),
            "fold_aucs": [round(a, 4) for a in fold_aucs],
            "boot_mean": round(float(np.mean(boot_aucs)), 4),
            "boot_std": round(float(np.std(boot_aucs)), 4),
            "ci_95_lower": round(ci95_lo, 4),
            "ci_95_upper": round(ci95_hi, 4),
            "ci_995_bonferroni_pct_lower": round(ci_bonf_pct_lo, 4),
            "ci_995_bonferroni_pct_upper": round(ci_bonf_pct_hi, 4),
            "ci_995_bonferroni_norm_lower": round(ci_bonf_norm_lo, 4),
            "ci_995_bonferroni_norm_upper": round(ci_bonf_norm_hi, 4),
            "perm_p": round(perm_p, 6),
            "ci95_pass": ci95_pass,
            "bonferroni_pass_pct": bonf_pass_pct,
            "bonferroni_pass_norm": bonf_pass_norm,
        }

    total_time = time.time() - t_start

    output = {
        "experiment": "exp_gemma2b_10k_qwen_scorer",
        "description": "Bootstrap CI (10000 iters) + Bonferroni 99.5% CI + permutation test for Gemma-2-2B 10K Qwen cross-scored",
        "method": "GroupKFold(5) OOF + doc-level bootstrap(10000) + doc-level label permutation(10000)",
        "scorer": "Qwen (cross-scorer)",
        "bonferroni": {"n_configs": 10, "alpha_corrected": 0.005, "z": BONFERRONI_Z},
        "theta": THETA,
        "seed": SEED,
        "n_bootstrap": N_BOOT,
        "n_permutation": N_PERM,
        "total_seconds": round(total_time, 1),
        "pairs": all_results,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUT_PATH}", flush=True)

    print("\n" + "="*70, flush=True)
    print("FINAL SUMMARY", flush=True)
    print("="*70, flush=True)
    for name, r in all_results.items():
        ci95_flag = "PASS" if r["ci95_pass"] else "FAIL"
        bonf_flag = "PASS" if r["bonferroni_pass_pct"] else "FAIL"
        print(f"  {name}: AUC={r['auc_oof']:.4f}  std={r['boot_std']:.4f}  "
              f"95%CI=[{r['ci_95_lower']:.4f}, {r['ci_95_upper']:.4f}] [{ci95_flag}]  "
              f"Bonf99.5%=[{r['ci_995_bonferroni_pct_lower']:.4f}, {r['ci_995_bonferroni_pct_upper']:.4f}] [{bonf_flag}]  "
              f"p={r['perm_p']:.6f}", flush=True)

    print(f"\nTotal time: {total_time:.1f}s", flush=True)


if __name__ == "__main__":
    main()
