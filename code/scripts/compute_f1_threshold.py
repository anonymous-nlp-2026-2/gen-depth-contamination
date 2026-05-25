"""
Compute F1@optimal-threshold for all depth pairs using OOF predictions.
Also computes F1@0.50 (natural threshold) and excess F1 above random.
"""

import csv
import json
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

BASE = Path("/root/autodl-tmp/gen-depth-contamination")
OUT_DIR = BASE / "results" / "f1_optimal_threshold"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEAT_COLS = [
    "surp_mean", "surp_std", "surp_skew", "surp_kurt",
    "surp_d1_mean", "surp_d1_std", "surp_d1_skew",
    "surp_d2_mean", "surp_d2_std",
    "ttr", "hapax_ratio", "self_bleu",
    "freq_kurtosis", "freq_entropy", "low_freq_ratio",
]

CONFIGS = {
    "Qwen-base":      BASE / "data_exp016_qwen_base" / "features.csv",
    "Pythia-nuc0.9":   BASE / "data_exp020_pythia_nuc09" / "features.csv",
    "Pythia-greedy":   BASE / "data_exp020_pythia_greedy" / "features.csv",
    "Pythia-T0.7":     BASE / "data_exp020_pythia_t07" / "features.csv",
    "Pythia-T1.2":     BASE / "data_exp020_pythia_t12" / "features.csv",
    "Mistral-7B":      BASE / "data_exp019_mistral7b" / "features.csv",
    "LLaMA-8B":        BASE / "data" / "exp_018_llama8b" / "features.csv",
}

DEPTH_PAIRS = [(0,1), (1,2), (2,3), (3,4), (4,5)]
THRESHOLDS = np.arange(0.01, 1.00, 0.01)
RANDOM_F1_BALANCED = 2/3  # trivial all-positive F1 for 50/50 classes


def load_features(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    depths = np.array([int(r["depth"]) for r in rows])
    doc_ids = np.array([int(r["doc_id"]) for r in rows])
    X = np.array(
        [[float(r[c]) if r[c] not in ("", "nan") else np.nan for c in FEAT_COLS]
         for r in rows]
    )
    col_means = np.nanmean(X, axis=0)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        X[mask, j] = col_means[j]
    return X, depths, doc_ids


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


def optimal_f1(y_true, proba, thresholds):
    best_f1, best_thr, best_p, best_r = 0.0, 0.5, 0.0, 0.0
    for thr in thresholds:
        y_pred = (proba >= thr).astype(int)
        if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
            continue
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            best_p = precision_score(y_true, y_pred)
            best_r = recall_score(y_true, y_pred)
    return best_f1, best_thr, best_p, best_r


def f1_at_050(y_true, proba):
    y_pred = (proba >= 0.50).astype(int)
    if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
        return 0.0, 0.0, 0.0
    f1 = f1_score(y_true, y_pred)
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return f1, p, r


def main():
    t0 = time.time()
    all_results = {}

    for config_name, feat_path in CONFIGS.items():
        if not feat_path.exists():
            print(f"SKIP {config_name}: {feat_path} not found", flush=True)
            continue
        print(f"\nConfig: {config_name}", flush=True)

        X_all, depths_all, doc_ids_all = load_features(feat_path)
        config_results = {}

        for d_lo, d_hi in DEPTH_PAIRS:
            pair_key = f"{d_lo}v{d_hi}"
            mask = (depths_all == d_lo) | (depths_all == d_hi)
            if mask.sum() == 0:
                continue

            X = X_all[mask]
            y = (depths_all[mask] == d_hi).astype(int)
            groups = doc_ids_all[mask]

            oof_proba = get_oof_predictions(X, y, groups)
            auc = roc_auc_score(y, oof_proba)
            f1_opt, thr_opt, p_opt, r_opt = optimal_f1(y, oof_proba, THRESHOLDS)
            f1_50, p_50, r_50 = f1_at_050(y, oof_proba)

            # "Nontrivial" optimal: require threshold >= 0.35 to avoid trivial all-positive
            f1_nt, thr_nt, p_nt, r_nt = 0.0, 0.5, 0.0, 0.0
            for thr in THRESHOLDS:
                if thr < 0.35:
                    continue
                y_pred = (oof_proba >= thr).astype(int)
                if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
                    continue
                f1 = f1_score(y, y_pred)
                if f1 > f1_nt:
                    f1_nt = f1
                    thr_nt = thr
                    p_nt = precision_score(y, y_pred)
                    r_nt = recall_score(y, y_pred)

            config_results[pair_key] = {
                "auc": round(auc, 4),
                "f1_optimal": round(f1_opt, 4),
                "thr_optimal": round(thr_opt, 2),
                "p_optimal": round(p_opt, 4),
                "r_optimal": round(r_opt, 4),
                "f1_at_050": round(f1_50, 4),
                "p_at_050": round(p_50, 4),
                "r_at_050": round(r_50, 4),
                "f1_nontrivial": round(f1_nt, 4),
                "thr_nontrivial": round(thr_nt, 2),
                "p_nontrivial": round(p_nt, 4),
                "r_nontrivial": round(r_nt, 4),
            }
            print(f"  {pair_key}: AUC={auc:.4f}  F1*={f1_opt:.4f}@{thr_opt:.2f}  F1@0.50={f1_50:.4f}  F1_nt={f1_nt:.4f}@{thr_nt:.2f}", flush=True)

        all_results[config_name] = config_results

    # Save JSON
    out_json = OUT_DIR / "f1_optimal_results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)

    # === PRINT TABLES ===

    # Table 1: F1@0.50 (natural threshold, most interpretable)
    print(f"\n{'='*110}")
    print("TABLE 1: F1 @ threshold=0.50  (random baseline ~ 0.50 for balanced classes)")
    print(f"{'='*110}")
    header = f"{'Config':<18}"
    for d_lo, d_hi in DEPTH_PAIRS:
        header += f"  {d_lo}v{d_hi:>7}"
    print(header)
    print("-" * 70)
    for cfg, res in all_results.items():
        row = f"{cfg:<18}"
        for d_lo, d_hi in DEPTH_PAIRS:
            pk = f"{d_lo}v{d_hi}"
            if pk in res:
                row += f"  {res[pk]['f1_at_050']:>7.4f}"
            else:
                row += f"  {'---':>7}"
        print(row)

    # Table 2: F1@nontrivial-optimal (threshold >= 0.35)
    print(f"\n{'='*110}")
    print("TABLE 2: F1 @ nontrivial-optimal (sweep thr >= 0.35 to avoid trivial all-positive)")
    print(f"{'='*110}")
    header = f"{'Config':<18}"
    for d_lo, d_hi in DEPTH_PAIRS:
        header += f" {d_lo}v{d_hi} F1(thr){'>':>4}"
    print(header)
    print("-" * 110)
    for cfg, res in all_results.items():
        row = f"{cfg:<18}"
        for d_lo, d_hi in DEPTH_PAIRS:
            pk = f"{d_lo}v{d_hi}"
            if pk in res:
                r = res[pk]
                row += f"  {r['f1_nontrivial']:.3f}({r['thr_nontrivial']:.2f})"
            else:
                row += f"  {'---':>12}"
        print(row)

    # Table 3: AUC comparison
    print(f"\n{'='*110}")
    print("TABLE 3: Pairwise AUC (reference)")
    print(f"{'='*110}")
    header = f"{'Config':<18}"
    for d_lo, d_hi in DEPTH_PAIRS:
        header += f"  {d_lo}v{d_hi:>7}"
    print(header)
    print("-" * 70)
    for cfg, res in all_results.items():
        row = f"{cfg:<18}"
        for d_lo, d_hi in DEPTH_PAIRS:
            pk = f"{d_lo}v{d_hi}"
            if pk in res:
                row += f"  {res[pk]['auc']:>7.4f}"
            else:
                row += f"  {'---':>7}"
        print(row)

    # Key finding
    print(f"\n{'='*110}")
    print("KEY FINDING: F1 boundary analysis")
    print(f"{'='*110}")
    for cfg, res in all_results.items():
        f1_1v2 = res.get("1v2", {}).get("f1_at_050", 0)
        f1_2v3 = res.get("2v3", {}).get("f1_at_050", 0)
        auc_1v2 = res.get("1v2", {}).get("auc", 0)
        auc_2v3 = res.get("2v3", {}).get("auc", 0)
        boundary_consistent = f1_1v2 > 0.55 and f1_2v3 < 0.55
        kstar_auc = 2 if auc_1v2 >= 0.60 and auc_2v3 < 0.60 else (1 if auc_1v2 < 0.60 else ">2")
        print(f"  {cfg:<18}: F1@0.50 1v2={f1_1v2:.4f} 2v3={f1_2v3:.4f}  |  AUC K*={kstar_auc}  |  F1 boundary@2v3: {'YES' if boundary_consistent else 'NO'}")

    print(f"\nTotal time: {time.time()-t0:.1f}s")
    print(f"Saved: {out_json}", flush=True)


if __name__ == "__main__":
    main()
