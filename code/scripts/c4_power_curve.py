"""
C4 Power Curve: how many docs are needed to reliably detect K*=2?

For each n in {1000..5000}, subsample n docs 10 times, run GroupKFold(5)
LightGBM pairwise classification on 1v2, record AUC.
Plot power curve with error bars.
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
OUT_DIR = Path("/root/autodl-tmp/gen-depth-contamination/results/c4_power_curve")

SAMPLE_SIZES = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
N_TRIALS = 10
THETA = 0.60
MASTER_SEED = 2024


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


def run_pairwise_cv(X, y, groups):
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
    overall_auc = roc_auc_score(y, oof_proba)
    return overall_auc, fold_aucs


def main():
    t0 = time.time()
    print("Loading features...", flush=True)
    X_all, depths_all, doc_ids_all = load_data()
    unique_docs_all = np.unique(doc_ids_all)
    print(f"Total: {len(depths_all)} rows, {len(unique_docs_all)} docs", flush=True)

    # Pre-index: for each doc_id, row indices
    doc_row_map = {}
    for i, d in enumerate(doc_ids_all):
        doc_row_map.setdefault(d, []).append(i)

    # Focus on depth 1 vs 2 pair
    pair_mask_all = (depths_all == 1) | (depths_all == 2)

    rng = np.random.RandomState(MASTER_SEED)
    results = {}

    print(f"\n{'n':>5}  {'trial':>5}  {'AUC':>7}  {'fold_aucs'}", flush=True)
    print("-" * 70, flush=True)

    for n in SAMPLE_SIZES:
        trial_aucs = []
        trial_fold_aucs = []
        for trial in range(N_TRIALS):
            sampled_docs = set(rng.choice(unique_docs_all, n, replace=False))
            row_mask = np.array([d in sampled_docs for d in doc_ids_all])
            combined_mask = row_mask & pair_mask_all

            X_sub = X_all[combined_mask]
            y_sub = (depths_all[combined_mask] == 2).astype(int)
            groups_sub = doc_ids_all[combined_mask]

            auc, fold_aucs = run_pairwise_cv(X_sub, y_sub, groups_sub)
            trial_aucs.append(auc)
            trial_fold_aucs.append(fold_aucs)
            print(f"{n:>5}  {trial:>5}  {auc:.4f}  {[round(a,4) for a in fold_aucs]}", flush=True)

        mean_auc = float(np.mean(trial_aucs))
        std_auc = float(np.std(trial_aucs))
        detect_prob = float(np.mean([a > THETA for a in trial_aucs]))

        results[str(n)] = {
            "n": n,
            "trial_aucs": [round(a, 4) for a in trial_aucs],
            "mean_auc": round(mean_auc, 4),
            "std_auc": round(std_auc, 4),
            "min_auc": round(float(np.min(trial_aucs)), 4),
            "max_auc": round(float(np.max(trial_aucs)), 4),
            "detection_probability": round(detect_prob, 2),
            "trial_fold_aucs": [[round(a, 4) for a in fa] for fa in trial_fold_aucs],
        }
        print(f"  => n={n}: mean={mean_auc:.4f} +/- {std_auc:.4f}, P(AUC>{THETA})={detect_prob:.0%}", flush=True)

    elapsed = time.time() - t0

    output = {
        "experiment": "c4_power_curve",
        "description": "Subsample power curve for 1v2 AUC (K*=2 detection)",
        "source": "exp016_qwen_base (C4)",
        "full_n": int(len(unique_docs_all)),
        "theta": THETA,
        "n_trials": N_TRIALS,
        "sample_sizes": SAMPLE_SIZES,
        "total_seconds": round(elapsed, 1),
        "results": results,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "power_curve_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {out_path}", flush=True)
    print(f"Total time: {elapsed:.1f}s", flush=True)

    # --- Plot ---
    plot_power_curve(results)


def plot_power_curve(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = sorted([int(k) for k in results.keys()])
    means = [results[str(n)]["mean_auc"] for n in ns]
    stds = [results[str(n)]["std_auc"] for n in ns]
    det_probs = [results[str(n)]["detection_probability"] for n in ns]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left panel: AUC vs n
    ax1.errorbar(ns, means, yerr=stds, fmt="o-", color="#2563EB", capsize=4,
                 capthick=1.5, linewidth=1.5, markersize=5, label="1v2 AUC (mean $\\pm$ std)")
    ax1.axhline(y=THETA, color="#DC2626", linestyle="--", linewidth=1.2, label=f"$\\theta$ = {THETA}")
    ax1.fill_between(ns, THETA - 0.005, THETA + 0.005, color="#DC2626", alpha=0.08)
    ax1.set_xlabel("Number of documents ($n$)", fontsize=11)
    ax1.set_ylabel("1v2 AUC", fontsize=11)
    ax1.set_title("(a) AUC vs. sample size", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9, loc="lower right")
    ax1.set_xlim(800, 5200)
    ax1.set_ylim(0.54, 0.68)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=9)

    # Right panel: Detection probability vs n
    ax2.plot(ns, det_probs, "s-", color="#059669", linewidth=1.5, markersize=6)
    ax2.axhline(y=0.8, color="#9333EA", linestyle=":", linewidth=1.0, alpha=0.7, label="80% power")
    ax2.set_xlabel("Number of documents ($n$)", fontsize=11)
    ax2.set_ylabel(f"P(AUC > {THETA})", fontsize=11)
    ax2.set_title("(b) Detection probability", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.set_xlim(800, 5200)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=9)

    plt.tight_layout()

    for ext in ["pdf", "png"]:
        out_path = OUT_DIR / f"fig_power_curve.{ext}"
        fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
        print(f"Saved {out_path}", flush=True)
    plt.close()


if __name__ == "__main__":
    main()
