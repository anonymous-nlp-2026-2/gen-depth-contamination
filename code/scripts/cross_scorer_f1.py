#!/usr/bin/env python3
"""Compute pairwise AUC + F1@0.50 for all cross-scorer configs."""
import os
os.environ["PYTHONWARNINGS"] = "ignore"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import json, csv, time, sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, f1_score
from lightgbm import LGBMClassifier
import pandas as pd

FEAT_COLS = [
    "surp_mean", "surp_std", "surp_skew", "surp_kurt",
    "surp_d1_mean", "surp_d1_std", "surp_d1_skew",
    "surp_d2_mean", "surp_d2_std",
    "ttr", "hapax_ratio", "self_bleu",
    "freq_kurtosis", "freq_entropy", "low_freq_ratio",
]

DATA_ROOT = Path("/root/autodl-tmp/gen-depth-contamination/data_exp022_cross_scorer")
OUT_DIR = Path("/root/autodl-tmp/gen-depth-contamination/results/cross_scorer_f1")
THRESHOLD = 0.50
SEED = 42
MAX_DEPTH = 6

def load_data(data_dir):
    df = pd.read_csv(data_dir / "features.csv")
    for c in FEAT_COLS:
        df[c] = df[c].fillna(df[c].mean())
    return df

def evaluate_pair(df, depth_lo, depth_hi):
    mask = df['depth'].isin([depth_lo, depth_hi])
    sub = df[mask].copy()
    if len(sub) < 20:
        return None
    sub['label'] = (sub['depth'] == depth_hi).astype(int)
    if sub['label'].nunique() < 2:
        return None

    X = sub[FEAT_COLS]
    y = sub['label'].values
    groups = sub['doc_id'].values

    gkf = GroupKFold(n_splits=5)
    oof_proba = np.zeros(len(y))
    fold_aucs = []

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        clf = LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            num_leaves=31, verbose=-1, n_jobs=-1, random_state=SEED,
        )
        clf.fit(X.iloc[train_idx], y[train_idx])
        proba = clf.predict_proba(X.iloc[test_idx])[:, 1]
        oof_proba[test_idx] = proba
        fold_aucs.append(roc_auc_score(y[test_idx], proba))

    oof_auc = roc_auc_score(y, oof_proba)
    oof_pred = (oof_proba >= THRESHOLD).astype(int)
    oof_f1 = f1_score(y, oof_pred)
    mean_fold_auc = float(np.mean(fold_aucs))

    return {
        "auc_oof": round(oof_auc, 4),
        "auc_mean_fold": round(mean_fold_auc, 4),
        "f1_at_050": round(oof_f1, 4),
        "fold_aucs": [round(a, 4) for a in fold_aucs],
    }

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configs = sorted([d.name for d in DATA_ROOT.iterdir() if d.is_dir() and (d / "features.csv").exists()])
    print(f"Found {len(configs)} configs", flush=True)

    all_results = {}
    for config_name in configs:
        t0 = time.time()
        df = load_data(DATA_ROOT / config_name)
        max_d = int(df['depth'].max())
        print(f"{config_name}: {len(df)} samples, depths 0-{max_d}", flush=True)

        pairs = {}
        for k in range(min(max_d, MAX_DEPTH)):
            pair_name = f"{k}v{k+1}"
            result = evaluate_pair(df, k, k+1)
            if result:
                pairs[pair_name] = result
                print(f"  {pair_name}: AUC={result['auc_mean_fold']:.4f} F1={result['f1_at_050']:.4f}", flush=True)

        elapsed = time.time() - t0
        all_results[config_name] = {"pairs": pairs, "seconds": round(elapsed, 1)}
        print(f"  [{elapsed:.1f}s]", flush=True)

    with open(OUT_DIR / "cross_scorer_f1_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    depth_pairs = ["0v1", "1v2", "2v3", "3v4", "4v5"]
    md_lines = ["| Config | " + " | ".join(f"{p} AUC | {p} F1@0.50" for p in depth_pairs) + " |"]
    md_lines.append("|" + "|".join(["---"] * (1 + 2 * len(depth_pairs))) + "|")
    for config_name in configs:
        pairs = all_results[config_name]["pairs"]
        cells = [config_name]
        for p in depth_pairs:
            if p in pairs:
                cells.append(f"{pairs[p]['auc_mean_fold']:.4f}")
                cells.append(f"{pairs[p]['f1_at_050']:.4f}")
            else:
                cells.extend(["N/A", "N/A"])
        md_lines.append("| " + " | ".join(cells) + " |")

    with open(OUT_DIR / "cross_scorer_f1_table.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")

    # Verify against existing
    print("\n=== VERIFICATION ===", flush=True)
    ref_root = Path("/root/autodl-tmp/gen-depth-contamination/results_exp022_cross_scorer")
    for config_name in configs:
        ref_file = ref_root / config_name / "pairwise_auc.json"
        if ref_file.exists():
            with open(ref_file) as f:
                ref = json.load(f)
            ok = True
            for p in depth_pairs:
                if p in all_results[config_name]["pairs"] and p in ref.get("mean", {}):
                    diff = abs(all_results[config_name]["pairs"][p]["auc_mean_fold"] - ref["mean"][p])
                    if diff > 0.005:
                        print(f"  MISMATCH {config_name} {p}: {diff:.4f}", flush=True)
                        ok = False
            if ok:
                print(f"  {config_name}: OK", flush=True)

    print("\n=== TABLE ===", flush=True)
    for line in md_lines:
        print(line, flush=True)
    print(f"\nDone. Saved to {OUT_DIR}", flush=True)

if __name__ == "__main__":
    main()
