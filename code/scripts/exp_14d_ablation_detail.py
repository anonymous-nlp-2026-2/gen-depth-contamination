"""
14D vs 15D Feature Ablation: self_bleu removal impact on AUC and K*.
Compares 15D (all features) vs 14D (drop self_bleu) for each model x depth pair.
Bootstrap CI (10000 resamples, document-level) for each setting.
"""

import csv
import json
import sys
import time
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

MODELS = {
    "Qwen-base": Path("/root/autodl-tmp/gen-depth-contamination/data_exp016_qwen_base/features.csv"),
    "Pythia": Path("/root/autodl-tmp/gen-depth-contamination/data_exp020_pythia_nuc09/features.csv"),
    "LLaMA-8B": Path("/root/autodl-tmp/gen-depth-contamination/data/exp_018_llama8b/features.csv"),
    "Mistral-7B": Path("/root/autodl-tmp/gen-depth-contamination/data_exp019_mistral7b/features.csv"),
}

OUT_DIR = Path("/root/autodl-tmp/gen-depth-contamination/artifacts")
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

    delta = auc_14 - auc_15

    return {
        "auc_15d": round(auc_15, 4),
        "ci_15d_lower": round(ci15_lo, 4),
        "ci_15d_upper": round(ci15_hi, 4),
        "auc_14d": round(auc_14, 4),
        "ci_14d_lower": round(ci14_lo, 4),
        "ci_14d_upper": round(ci14_hi, 4),
        "delta": round(delta, 4),
    }


def compute_kstar(pair_results, theta=THETA):
    kstar = 0
    for k in range(1, MAX_DEPTH + 1):
        pair_key = f"{k-1}v{k}"
        if pair_key in pair_results and pair_results[pair_key]["auc_15d"] > theta:
            kstar = k
        else:
            break
    return kstar


def compute_kstar_14d(pair_results, theta=THETA):
    kstar = 0
    for k in range(1, MAX_DEPTH + 1):
        pair_key = f"{k-1}v{k}"
        if pair_key in pair_results and pair_results[pair_key]["auc_14d"] > theta:
            kstar = k
        else:
            break
    return kstar


def main():
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    csv_rows = []

    for model_name, fpath in MODELS.items():
        print(f"\n{'='*60}", flush=True)
        print(f"Model: {model_name} ({fpath})", flush=True)
        print(f"{'='*60}", flush=True)

        X_15d, X_14d, depths, doc_ids = load_data(fpath)
        print(f"  Loaded {len(depths)} samples, depths 0-{depths.max()}", flush=True)

        model_pairs = {}
        rng = np.random.RandomState(SEED)

        for d in range(MAX_DEPTH):
            pair_key = f"{d}v{d+1}"
            print(f"  Pair {pair_key} ...", end=" ", flush=True)
            t0 = time.time()
            result = analyze_pair(X_15d, X_14d, depths, doc_ids, d, d+1, rng)
            elapsed = time.time() - t0
            model_pairs[pair_key] = result
            print(f"AUC_15D={result['auc_15d']:.4f} [{result['ci_15d_lower']:.4f}, {result['ci_15d_upper']:.4f}]  "
                  f"AUC_14D={result['auc_14d']:.4f} [{result['ci_14d_lower']:.4f}, {result['ci_14d_upper']:.4f}]  "
                  f"Delta={result['delta']:+.4f}  ({elapsed:.1f}s)", flush=True)

        kstar_15d = compute_kstar(model_pairs)
        kstar_14d = compute_kstar_14d(model_pairs)
        print(f"  K*_15D={kstar_15d}  K*_14D={kstar_14d}", flush=True)

        all_results[model_name] = {
            "pairs": model_pairs,
            "kstar_15d": kstar_15d,
            "kstar_14d": kstar_14d,
        }

        for pair_key, res in model_pairs.items():
            csv_rows.append({
                "Model": model_name,
                "Pair": pair_key,
                "AUC_15D": res["auc_15d"],
                "CI_15D_lower": res["ci_15d_lower"],
                "CI_15D_upper": res["ci_15d_upper"],
                "AUC_14D": res["auc_14d"],
                "CI_14D_lower": res["ci_14d_lower"],
                "CI_14D_upper": res["ci_14d_upper"],
                "Delta": res["delta"],
                "Kstar_15D": "",
                "Kstar_14D": "",
            })
        csv_rows.append({
            "Model": model_name,
            "Pair": "K*",
            "AUC_15D": "",
            "CI_15D_lower": "",
            "CI_15D_upper": "",
            "AUC_14D": "",
            "CI_14D_lower": "",
            "CI_14D_upper": "",
            "Delta": "",
            "Kstar_15D": kstar_15d,
            "Kstar_14D": kstar_14d,
        })

    total_time = time.time() - t_start

    output = {
        "description": "14D vs 15D Feature Ablation (self_bleu removal)",
        "method": "GroupKFold(5) OOF + doc-level bootstrap(10000)",
        "theta": THETA,
        "n_bootstrap": N_BOOT,
        "seed": SEED,
        "total_seconds": round(total_time, 1),
        "models": all_results,
    }

    json_path = OUT_DIR / "exp_14d_ablation_detail.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON saved to {json_path}", flush=True)

    csv_path = OUT_DIR / "exp_14d_ablation_detail.csv"
    fieldnames = ["Model", "Pair", "AUC_15D", "CI_15D_lower", "CI_15D_upper",
                   "AUC_14D", "CI_14D_lower", "CI_14D_upper", "Delta",
                   "Kstar_15D", "Kstar_14D"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"CSV saved to {csv_path}", flush=True)

    print(f"\nTotal time: {total_time:.1f}s", flush=True)

    print("\n\n=== SUMMARY TABLE ===", flush=True)
    print(f"{'Model':<12} {'Pair':<6} {'AUC_15D':>8} {'CI_15D':>20} {'AUC_14D':>8} {'CI_14D':>20} {'Delta':>8}", flush=True)
    print("-" * 90, flush=True)
    for model_name, mdata in all_results.items():
        for pair_key, res in mdata["pairs"].items():
            ci15 = f"[{res['ci_15d_lower']:.4f}, {res['ci_15d_upper']:.4f}]"
            ci14 = f"[{res['ci_14d_lower']:.4f}, {res['ci_14d_upper']:.4f}]"
            print(f"{model_name:<12} {pair_key:<6} {res['auc_15d']:>8.4f} {ci15:>20} {res['auc_14d']:>8.4f} {ci14:>20} {res['delta']:>+8.4f}", flush=True)
        print(f"{model_name:<12} {'K*':<6} {mdata['kstar_15d']:>8} {'':>20} {mdata['kstar_14d']:>8}", flush=True)
        print("-" * 90, flush=True)


if __name__ == "__main__":
    main()
