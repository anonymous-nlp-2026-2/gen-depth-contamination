#!/usr/bin/env python3
"""
exp013 Cross-Transfer (OLMo rerun): Train OBD classifier on model A's features,
test on model B's features. CPU-only, uses LightGBM.

Data:
  - OLMo:  data/exp_olmo1b_c4_rerun/features.csv  (K*=1)
  - Pythia: data/exp_pythia14b_c4_rerun/features.csv (K*=2)
  - Qwen:  data_exp016_qwen_base/features.csv       (K*=2)
"""

import csv
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path("/root/autodl-tmp/gen-depth-contamination")

MODELS = {
    "OLMo":   ROOT / "data" / "exp_olmo1b_c4_rerun" / "features.csv",
    "Pythia": ROOT / "data" / "exp_pythia14b_c4_rerun" / "features.csv",
    "Qwen":   ROOT / "data_exp016_qwen_base" / "features.csv",
}

FEAT_COLS = [
    "surp_mean", "surp_std", "surp_skew", "surp_kurt",
    "surp_d1_mean", "surp_d1_std", "surp_d1_skew",
    "surp_d2_mean", "surp_d2_std",
    "ttr", "hapax_ratio", "self_bleu",
    "freq_kurtosis", "freq_entropy", "low_freq_ratio",
]

MAX_DEPTH = 5
THETA = 0.60
SEED = 42
N_BOOT = 1000


def load_features(csv_path):
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    depths = np.array([int(r["depth"]) for r in rows])
    doc_ids = np.array([int(r["doc_id"]) for r in rows])
    feat_matrix = np.array(
        [[float(r[c]) if r[c] not in ("", "nan") else np.nan for c in FEAT_COLS] for r in rows]
    )
    col_means = np.nanmean(feat_matrix, axis=0)
    for j in range(feat_matrix.shape[1]):
        mask = np.isnan(feat_matrix[:, j])
        feat_matrix[mask, j] = col_means[j]
    return depths, doc_ids, feat_matrix


def compute_kstar(pairwise_auc, theta=THETA):
    k_star = 0
    for k in range(1, MAX_DEPTH + 1):
        key = f"{k-1}v{k}"
        if pairwise_auc.get(key, 0) > theta:
            k_star = k
    return k_star


def bootstrap_auc_ci(y_true, y_score, n_boot=N_BOOT, alpha=0.05):
    rng = np.random.RandomState(SEED)
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    aucs = sorted(aucs)
    lo = aucs[int(alpha / 2 * len(aucs))]
    hi = aucs[int((1 - alpha / 2) * len(aucs))]
    return lo, hi


def cross_transfer_pair(train_depths, train_feats, test_depths, test_feats):
    pairwise_auc = {}
    pairwise_ci = {}

    for k in range(1, MAX_DEPTH + 1):
        key = f"{k-1}v{k}"

        # Training data
        tr_mask = (train_depths == k - 1) | (train_depths == k)
        X_train = train_feats[tr_mask]
        y_train = (train_depths[tr_mask] == k).astype(int)

        # Test data
        te_mask = (test_depths == k - 1) | (test_depths == k)
        X_test = test_feats[te_mask]
        y_test = (test_depths[te_mask] == k).astype(int)

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            pairwise_auc[key] = float("nan")
            pairwise_ci[key] = [float("nan"), float("nan")]
            continue

        clf = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            num_leaves=31, verbose=-1, n_jobs=-1, random_state=SEED,
        )
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)

        lo, hi = bootstrap_auc_ci(y_test, proba)
        pairwise_auc[key] = round(float(auc), 4)
        pairwise_ci[key] = [round(lo, 4), round(hi, 4)]

        log.info(f"    AUC({key}) = {auc:.4f}  [{lo:.4f}, {hi:.4f}]")

    kstar = compute_kstar(pairwise_auc)
    return {"pairwise_auc": pairwise_auc, "pairwise_ci": pairwise_ci, "kstar": kstar}


def self_transfer_pair(depths, doc_ids, feats):
    from sklearn.model_selection import GroupKFold
    pairwise_auc = {}
    pairwise_ci = {}

    for k in range(1, MAX_DEPTH + 1):
        key = f"{k-1}v{k}"
        mask = (depths == k - 1) | (depths == k)
        X = feats[mask]
        y = (depths[mask] == k).astype(int)
        groups = doc_ids[mask]

        if len(np.unique(y)) < 2:
            pairwise_auc[key] = float("nan")
            pairwise_ci[key] = [float("nan"), float("nan")]
            continue

        gkf = GroupKFold(n_splits=5)
        fold_aucs = []
        all_proba = np.zeros(len(y))
        for train_idx, test_idx in gkf.split(X, y, groups=groups):
            clf = lgb.LGBMClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                num_leaves=31, verbose=-1, n_jobs=-1, random_state=SEED,
            )
            clf.fit(X[train_idx], y[train_idx])
            proba = clf.predict_proba(X[test_idx])[:, 1]
            fold_aucs.append(roc_auc_score(y[test_idx], proba))
            all_proba[test_idx] = proba

        auc_mean = float(np.mean(fold_aucs))
        lo, hi = bootstrap_auc_ci(y, all_proba)
        pairwise_auc[key] = round(auc_mean, 4)
        pairwise_ci[key] = [round(lo, 4), round(hi, 4)]

        log.info(f"    AUC({key}) = {auc_mean:.4f}  [{lo:.4f}, {hi:.4f}]  folds={[round(a,4) for a in fold_aucs]}")

    kstar = compute_kstar(pairwise_auc)
    return {"pairwise_auc": pairwise_auc, "pairwise_ci": pairwise_ci, "kstar": kstar}


def main():
    np.random.seed(SEED)

    out_dir = ROOT / "results" / "exp013_olmo_rerun"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all model features
    model_data = {}
    for name, path in MODELS.items():
        log.info(f"Loading {name}: {path}")
        depths, doc_ids, feats = load_features(path)
        model_data[name] = {"depths": depths, "doc_ids": doc_ids, "feats": feats}
        log.info(f"  {len(depths)} samples, depths {depths.min()}-{depths.max()}")

    model_names = list(MODELS.keys())
    matrix = {}
    all_results = {}

    for tr_name in model_names:
        matrix[tr_name] = {}
        for te_name in model_names:
            pair_key = f"{tr_name}_to_{te_name}"
            log.info(f"\n{'='*60}")
            log.info(f"  {tr_name} -> {te_name}")

            if tr_name == te_name:
                d = model_data[tr_name]
                result = self_transfer_pair(d["depths"], d["doc_ids"], d["feats"])
            else:
                tr_d = model_data[tr_name]
                te_d = model_data[te_name]
                result = cross_transfer_pair(
                    tr_d["depths"], tr_d["feats"],
                    te_d["depths"], te_d["feats"],
                )

            all_results[pair_key] = result
            matrix[tr_name][te_name] = {
                "kstar": result["kstar"],
                "pairwise_auc": result["pairwise_auc"],
            }

            with open(out_dir / f"{pair_key}.json", "w") as f:
                json.dump(result, f, indent=2)

            log.info(f"  K* = {result['kstar']}")

    # Compute aggregate accuracy
    correct = 0
    total = 0
    for tr_name in model_names:
        for te_name in model_names:
            predicted_kstar = matrix[tr_name][te_name]["kstar"]
            true_kstar = matrix[te_name][te_name]["kstar"]
            if predicted_kstar == true_kstar:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0

    summary = {
        "models": model_names,
        "matrix": matrix,
        "self_kstar": {name: matrix[name][name]["kstar"] for name in model_names},
        "aggregate_accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "theta": THETA,
    }

    with open(out_dir / "transfer_matrix.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print matrix
    log.info(f"\n{'='*80}")
    log.info("K* Transfer Matrix (train=row, test=col)")
    header = f"{'':>10}" + "".join(f"{m:>10}" for m in model_names)
    log.info(header)
    for tr in model_names:
        row = f"{tr:>10}"
        for te in model_names:
            kstar = matrix[tr][te]["kstar"]
            self_kstar = matrix[te][te]["kstar"]
            marker = " *" if kstar != self_kstar and tr != te else ""
            row += f"{str(kstar) + marker:>10}"
        log.info(row)

    log.info(f"\nSelf K*: {summary['self_kstar']}")
    log.info(f"Aggregate accuracy (predicted K* == true K*): {correct}/{total} = {accuracy:.1%}")

    # Print full AUC matrix for 0v1
    log.info(f"\n{'='*80}")
    log.info("AUC(0v1) Transfer Matrix")
    header = f"{'':>10}" + "".join(f"{m:>10}" for m in model_names)
    log.info(header)
    for tr in model_names:
        row = f"{tr:>10}"
        for te in model_names:
            auc = matrix[tr][te]["pairwise_auc"]["0v1"]
            row += f"{auc:>10.4f}"
        log.info(row)

    log.info(f"\nAll results saved to {out_dir}/")


if __name__ == "__main__":
    main()
