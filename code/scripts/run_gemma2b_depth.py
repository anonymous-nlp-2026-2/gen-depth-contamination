#!/usr/bin/env python3
"""
exp_gemma2b_depth: Gemma-2-2B C4 Depth Chain (continuation mode)
Full pipeline + bootstrap CI + permutation test.
"""
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/root/autodl-tmp/gen-depth-contamination")
MODEL_DIR = Path("/root/autodl-tmp/.hf_cache/gemma-2-2b-local")
SCORER_PATH = "/root/autodl-tmp/.hf_cache/Qwen/Qwen2.5-1.5B"
DATA_DIR = PROJECT_ROOT / "data" / "exp_gemma2b_depth"
RESULTS_DIR = PROJECT_ROOT / "results" / "exp_gemma2b_depth"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

N_BOOT = 10000
N_PERM = 10000
MAX_DEPTH = 5
SEED = 42

FEAT_COLS = [
    "surp_mean", "surp_std", "surp_skew", "surp_kurt",
    "surp_d1_mean", "surp_d1_std", "surp_d1_skew",
    "surp_d2_mean", "surp_d2_std",
    "ttr", "hapax_ratio", "self_bleu",
    "freq_kurtosis", "freq_entropy", "low_freq_ratio",
]


def ensure_model():
    """Download model if not already present."""
    marker = MODEL_DIR / "model.safetensors.index.json"
    safetensors = list(MODEL_DIR.glob("model-*.safetensors")) if MODEL_DIR.exists() else []
    if marker.exists() and len(safetensors) >= 3:
        print(f"Model already present at {MODEL_DIR} ({len(safetensors)} shards)", flush=True)
        return str(MODEL_DIR)

    print("Downloading Gemma-2-2B via ModelScope...", flush=True)
    from modelscope import snapshot_download
    local_path = snapshot_download(
        "google/gemma-2-2b",
        cache_dir="/root/autodl-tmp/.hf_cache/gemma-2-2b-ms",
    )
    print(f"Model downloaded to: {local_path}", flush=True)
    return local_path


def run_pipeline(model_path):
    """Run generation + feature extraction + basic analysis via run_pipeline.py."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_pipeline.py"),
        "--model_path", model_path,
        "--data_dir", str(DATA_DIR),
        "--results_dir", str(RESULTS_DIR),
        "--num_samples", "5000",
        "--max_depth", str(MAX_DEPTH),
        "--generation_mode", "continuation",
        "--top_p", "0.95",
        "--temperature", "1.0",
        "--scorer_path", SCORER_PATH,
    ]
    print(f"Running pipeline: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


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
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score

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
    from sklearn.metrics import roc_auc_score

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
    from sklearn.metrics import roc_auc_score

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


def run_bootstrap_ci():
    """Bootstrap CI + permutation test on generated features."""
    t_start = time.time()
    print("\n" + "=" * 60, flush=True)
    print("Bootstrap CI + Permutation Test", flush=True)
    print("=" * 60, flush=True)

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
        print(f"  Permutation p-value: {perm_p:.6f}", flush=True)

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s", flush=True)

        results[pair] = {
            "auc": round(point_auc, 4),
            "auc_mean_folds": round(float(np.mean(fold_aucs)), 4),
            "fold_aucs": [round(a, 4) for a in fold_aucs],
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "perm_p": round(perm_p, 6),
            "boot_mean": round(float(np.mean(boot_aucs)), 4),
            "boot_std": round(float(np.std(boot_aucs)), 4),
        }

    total_time = time.time() - t_start
    output = {
        "experiment": "exp_gemma2b_depth: Gemma-2-2B continuation, scorer=Qwen2.5-1.5B base",
        "method": "GroupKFold(5) OOF + doc-level bootstrap + label permutation",
        "n_bootstrap": N_BOOT,
        "n_permutation": N_PERM,
        "seed": SEED,
        "total_seconds": round(total_time, 1),
        "pairs": results,
    }

    out_path = ARTIFACTS_DIR / "exp_gemma2b_depth_bootstrap_ci.json"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nBootstrap CI results saved to {out_path}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for pair, r in results.items():
        flag = ""
        if pair == "1v2":
            flag = " *** KEY: CI lower vs theta=0.60 ***"
            if r["ci_lower"] > 0.60:
                flag += " -> PASS"
            else:
                flag += " -> FAIL"
        print(f"  {pair}: AUC={r['auc']:.4f}  95% CI=[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]  p={r['perm_p']:.6f}{flag}", flush=True)


if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("exp_gemma2b_depth: Gemma-2-2B C4 Depth Chain", flush=True)
    print("=" * 60, flush=True)

    model_path = ensure_model()
    print(f"Model path: {model_path}", flush=True)
    print(f"Scorer: {SCORER_PATH}", flush=True)
    print(f"Data: {DATA_DIR}", flush=True)
    print(f"Results: {RESULTS_DIR}", flush=True)

    run_pipeline(model_path)
    run_bootstrap_ci()

    print("\n=== ALL DONE ===", flush=True)
