"""
LOFO (Leave-One-Feature-Out) Feature Importance Analysis

For each domain config and depth pair:
1. Baseline: 5-fold GroupKFold LightGBM with all 15 features
2. LOFO: remove one feature at a time, retrain, measure AUC drop
3. Paired t-test for statistical significance
"""

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "4"

DOMAINS = {
    "C4":    Path("/root/autodl-tmp/gen-depth-contamination/data/exp_018_llama8b/features.csv"),
    "arXiv": Path("/root/autodl-tmp/gen-depth-contamination/data/exp_024_arxiv/features.csv"),
    "Wiki":  Path("/root/autodl-tmp/gen-depth-contamination/data/exp_023_wiki/features.csv"),
}

ALL_FEATURES = [
    "surp_mean", "surp_std", "surp_skew", "surp_kurt",
    "surp_d1_mean", "surp_d1_std", "surp_d1_skew",
    "surp_d2_mean", "surp_d2_std",
    "ttr", "hapax_ratio", "self_bleu",
    "freq_kurtosis", "freq_entropy", "low_freq_ratio",
]

FEATURE_GROUPS = {
    "surp_mean": "surprisal_base", "surp_std": "surprisal_base",
    "surp_skew": "surprisal_base", "surp_kurt": "surprisal_base",
    "surp_d1_mean": "surprisal_deriv", "surp_d1_std": "surprisal_deriv",
    "surp_d1_skew": "surprisal_deriv",
    "surp_d2_mean": "surprisal_deriv", "surp_d2_std": "surprisal_deriv",
    "ttr": "lexical_diversity", "hapax_ratio": "lexical_diversity",
    "self_bleu": "lexical_diversity",
    "freq_kurtosis": "freq_distribution", "freq_entropy": "freq_distribution",
    "low_freq_ratio": "freq_distribution",
}

LGB_PARAMS = dict(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    num_leaves=31, verbose=-1, n_jobs=4,
)

N_FOLDS = 5


def load_data(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    doc_ids = np.array([int(r["doc_id"]) for r in rows])
    depths = np.array([int(r["depth"]) for r in rows])
    feat_dict = {}
    for col in ALL_FEATURES:
        vals = []
        for r in rows:
            v = r[col]
            vals.append(float(v) if v != "" and v != "nan" else np.nan)
        feat_dict[col] = np.array(vals)
    return doc_ids, depths, feat_dict


def get_pair_data(doc_ids, depths, feat_dict, d_low, d_high):
    mask = (depths == d_low) | (depths == d_high)
    ids = doc_ids[mask]
    y = (depths[mask] == d_high).astype(int)
    sub_feat = {col: feat_dict[col][mask] for col in ALL_FEATURES}
    return ids, y, sub_feat


def build_X(feat_dict, cols):
    X = np.column_stack([feat_dict[c] for c in cols])
    col_means = np.nanmean(X, axis=0)
    for j in range(X.shape[1]):
        nan_mask = np.isnan(X[:, j])
        if nan_mask.any():
            X[nan_mask, j] = col_means[j]
    return X


def run_cv(X, y, groups, n_folds=N_FOLDS):
    gkf = GroupKFold(n_splits=n_folds)
    fold_aucs = []
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        clf = lgb.LGBMClassifier(**LGB_PARAMS)
        clf.fit(X[train_idx], y[train_idx])
        proba = clf.predict_proba(X[test_idx])[:, 1]
        fold_aucs.append(roc_auc_score(y[test_idx], proba))
    return np.array(fold_aucs)


def main():
    import time
    t0 = time.time()

    out_dir = Path("/root/autodl-tmp/gen-depth-contamination/results/lofo_importance")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for domain, csv_path in DOMAINS.items():
        print(f"\n{'='*60}", flush=True)
        print(f"Domain: {domain} ({csv_path})", flush=True)
        print(f"{'='*60}", flush=True)

        doc_ids, depths, feat_dict = load_data(csv_path)
        unique_depths = sorted(set(depths))
        max_depth = max(unique_depths)
        depth_pairs = [(k-1, k) for k in range(1, max_depth+1)]

        domain_results = {}

        for d_low, d_high in depth_pairs:
            pair_name = f"{d_low}v{d_high}"
            ids, y, sub_feat = get_pair_data(doc_ids, depths, feat_dict, d_low, d_high)

            if len(np.unique(y)) < 2:
                print(f"  {pair_name}: skipped (single class)", flush=True)
                continue

            n0 = np.sum(y == 0)
            n1 = np.sum(y == 1)
            print(f"\n  --- {pair_name} (n={len(y)}, class0={n0}, class1={n1}) ---", flush=True)

            X_full = build_X(sub_feat, ALL_FEATURES)
            baseline_folds = run_cv(X_full, y, ids)
            baseline_mean = float(np.mean(baseline_folds))
            baseline_std = float(np.std(baseline_folds))
            print(f"  Baseline (15D): AUC = {baseline_mean:.4f} +/- {baseline_std:.4f}", flush=True)

            pair_results = {
                "baseline": {
                    "mean": round(baseline_mean, 6),
                    "std": round(baseline_std, 6),
                    "folds": [round(x, 6) for x in baseline_folds],
                },
                "lofo": {},
            }

            for feat in ALL_FEATURES:
                remaining = [f for f in ALL_FEATURES if f != feat]
                X_lofo = build_X(sub_feat, remaining)
                lofo_folds = run_cv(X_lofo, y, ids)

                delta_folds = baseline_folds - lofo_folds
                delta_mean = float(np.mean(delta_folds))
                delta_std = float(np.std(delta_folds))

                if np.std(delta_folds) > 1e-10:
                    t_stat, p_value = sp_stats.ttest_rel(baseline_folds, lofo_folds)
                else:
                    t_stat, p_value = 0.0, 1.0

                pair_results["lofo"][feat] = {
                    "lofo_auc_mean": round(float(np.mean(lofo_folds)), 6),
                    "lofo_auc_std": round(float(np.std(lofo_folds)), 6),
                    "lofo_folds": [round(x, 6) for x in lofo_folds],
                    "delta_mean": round(delta_mean, 6),
                    "delta_std": round(delta_std, 6),
                    "t_stat": round(float(t_stat), 4),
                    "p_value": round(float(p_value), 6),
                    "significant": bool(p_value < 0.05),
                    "group": FEATURE_GROUPS[feat],
                }

                sig_marker = " *" if p_value < 0.05 else ""
                print(f"    -{feat:20s}: AUC={np.mean(lofo_folds):.4f}  "
                      f"delta={delta_mean:+.4f} +/- {delta_std:.4f}  "
                      f"p={p_value:.4f}{sig_marker}", flush=True)

            domain_results[pair_name] = pair_results

        all_results[domain] = domain_results

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s", flush=True)

    with open(out_dir / "lofo_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Full results saved to {out_dir / 'lofo_results.json'}", flush=True)

    print_ranking_table(all_results, out_dir)
    generate_latex_table(all_results, out_dir)


def print_ranking_table(all_results, out_dir):
    lines = []
    lines.append("\n" + "="*80)
    lines.append("LOFO Feature Importance Ranking (sorted by 1v2 delta AUC)")
    lines.append("="*80)

    for domain, domain_results in all_results.items():
        if "1v2" not in domain_results:
            continue
        pair_data = domain_results["1v2"]
        lines.append(f"\n--- {domain} (1v2 baseline AUC = {pair_data['baseline']['mean']:.4f}) ---")
        lines.append(f"{'Feature':25s} {'Group':20s} {'Delta AUC':>12s} {'Std':>8s} {'p-value':>10s} {'Sig':>5s}")
        lines.append("-" * 82)

        ranked = sorted(pair_data["lofo"].items(),
                       key=lambda x: x[1]["delta_mean"], reverse=True)
        for feat, info in ranked:
            sig = "***" if info["p_value"] < 0.001 else "**" if info["p_value"] < 0.01 else "*" if info["p_value"] < 0.05 else ""
            lines.append(f"{feat:25s} {info['group']:20s} {info['delta_mean']:+12.6f} "
                        f"{info['delta_std']:8.6f} {info['p_value']:10.6f} {sig:>5s}")

    report = "\n".join(lines)
    print(report, flush=True)

    with open(out_dir / "lofo_ranking.txt", "w") as f:
        f.write(report)
    print(f"\nRanking saved to {out_dir / 'lofo_ranking.txt'}", flush=True)

    summary_lines = []
    summary_lines.append("\n" + "="*80)
    summary_lines.append("Cross-domain Summary: All depth pairs")
    summary_lines.append("="*80)

    domains = list(all_results.keys())
    all_pairs_set = set()
    for d in domains:
        all_pairs_set.update(all_results[d].keys())
    all_pairs = sorted(all_pairs_set)

    for pair_name in all_pairs:
        summary_lines.append(f"\n--- {pair_name} ---")
        header = f"{'Feature':25s}"
        for domain in domains:
            header += f" | {domain:>20s}"
        summary_lines.append(header)
        summary_lines.append("-" * (25 + 23 * len(domains)))

        baselines = {}
        for domain in domains:
            if pair_name in all_results[domain]:
                baselines[domain] = all_results[domain][pair_name]["baseline"]["mean"]

        bline = f"{'BASELINE':25s}"
        for domain in domains:
            if domain in baselines:
                bline += f" | {baselines[domain]:20.4f}"
            else:
                bline += f" | {'N/A':>20s}"
        summary_lines.append(bline)
        summary_lines.append("")

        for feat in ALL_FEATURES:
            row = f"{feat:25s}"
            for domain in domains:
                if pair_name in all_results[domain] and feat in all_results[domain][pair_name]["lofo"]:
                    info = all_results[domain][pair_name]["lofo"][feat]
                    sig = "*" if info["significant"] else " "
                    row += f" | {info['delta_mean']:+.4f}({info['p_value']:.3f}){sig:1s}"
                else:
                    row += f" | {'N/A':>20s}"
            summary_lines.append(row)

    summary = "\n".join(summary_lines)
    print(summary, flush=True)

    with open(out_dir / "lofo_cross_domain_summary.txt", "w") as f:
        f.write(summary)
    print(f"\nCross-domain summary saved to {out_dir / 'lofo_cross_domain_summary.txt'}", flush=True)


def generate_latex_table(all_results, out_dir):
    domains = list(all_results.keys())
    pair_name = "1v2"

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{LOFO feature importance for depth 1 vs.\ 2 classification. "
                 r"$\Delta$AUC is the drop when removing that feature (positive = feature helps). "
                 r"Significance: {*} $p<0.05$, {**} $p<0.01$, {***} $p<0.001$.}")
    lines.append(r"\label{tab:lofo}")
    lines.append(r"\small")
    cols = "l" + "r" * len(domains)
    lines.append(r"\begin{tabular}{" + cols + "}")
    lines.append(r"\toprule")

    header = "Feature"
    for d in domains:
        header += f" & {d}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    bline = "Baseline AUC"
    for d in domains:
        if pair_name in all_results[d]:
            v = all_results[d][pair_name]["baseline"]["mean"]
            bline += f" & {v:.4f}"
        else:
            bline += " & ---"
    bline += r" \\"
    lines.append(bline)
    lines.append(r"\midrule")

    avg_deltas = {}
    for feat in ALL_FEATURES:
        deltas = []
        for d in domains:
            if pair_name in all_results[d] and feat in all_results[d][pair_name]["lofo"]:
                deltas.append(all_results[d][pair_name]["lofo"][feat]["delta_mean"])
        avg_deltas[feat] = np.mean(deltas) if deltas else 0.0

    ranked_feats = sorted(ALL_FEATURES, key=lambda f: avg_deltas[f], reverse=True)

    for feat in ranked_feats:
        row = feat.replace("_", r"\_")
        for d in domains:
            if pair_name in all_results[d] and feat in all_results[d][pair_name]["lofo"]:
                info = all_results[d][pair_name]["lofo"][feat]
                delta = info["delta_mean"]
                p = info["p_value"]
                sig = "^{***}" if p < 0.001 else "^{**}" if p < 0.01 else "^{*}" if p < 0.05 else ""
                row += f" & ${delta:+.4f}{sig}$"
            else:
                row += " & ---"
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)
    with open(out_dir / "lofo_table.tex", "w") as f:
        f.write(latex)
    print(f"\nLaTeX table saved to {out_dir / 'lofo_table.tex'}", flush=True)


if __name__ == "__main__":
    main()
