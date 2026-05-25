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

THETA = 0.60
SEED = 42
B_VALUES = [1000, 5000, 10000]

CASES = [
    {
        "name": "Qwen-1.5B C4 (exp_016)",
        "data_dir": "/root/autodl-tmp/gen-depth-contamination/data_exp016_qwen_base",
        "pairs": [(0,1),(1,2),(2,3),(3,4),(4,5)],
    },
    {
        "name": "LLaMA-3.1-8B (exp_018)",
        "data_dir": "/root/autodl-tmp/gen-depth-contamination/data/exp_018_llama8b",
        "pairs": [(0,1),(1,2),(2,3)],
    },
    {
        "name": "Gemma-2-2B (exp_gemma2b)",
        "data_dir": "/root/autodl-tmp/gen-depth-contamination/data/exp_gemma2b_depth",
        "pairs": [(0,1),(1,2),(2,3)],
    },
    {
        "name": "Mistral-7B (exp_019)",
        "data_dir": "/root/autodl-tmp/gen-depth-contamination/data_exp019_mistral7b",
        "pairs": [(0,1),(1,2),(2,3)],
    },
]

OUT_DIR = Path("/root/autodl-tmp/gen-depth-contamination/results/bootstrap_stability")


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


def get_oof_predictions(X, y, groups, seed):
    gkf = GroupKFold(n_splits=5)
    oof_proba = np.zeros(len(y))
    fold_aucs = []
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        clf = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            num_leaves=31, verbose=-1, n_jobs=-1, random_state=seed,
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
    return point_auc, ci_lower, ci_upper, float(np.std(boot_aucs))


def verdict(ci_lower, ci_upper, point_auc, theta=THETA):
    if ci_lower > theta:
        return "DETECT"
    elif ci_upper < theta:
        return "NOT_DETECT"
    elif point_auc > theta:
        return "BORDERLINE"
    else:
        return "BORDERLINE_LOW"


def main():
    t_start = time.time()
    all_results = {}
    borderline_summary = []

    for case in CASES:
        name = case["name"]
        print(f"\n{'='*70}", flush=True)
        print(f"Model: {name}", flush=True)
        print(f"{'='*70}", flush=True)

        try:
            X, depths, doc_ids = load_data(case["data_dir"])
        except FileNotFoundError:
            print(f"  SKIPPED: features.csv not found", flush=True)
            continue

        avail_depths = sorted(set(depths))
        print(f"  Available depths: {avail_depths}, total samples: {len(depths)}", flush=True)

        model_results = {}
        for d0, d1 in case["pairs"]:
            if d0 not in avail_depths or d1 not in avail_depths:
                print(f"  {d0}v{d1}: SKIPPED (depth not available)", flush=True)
                continue

            pair_key = f"{d0}v{d1}"
            t0 = time.time()
            print(f"\n  --- {pair_key} ---", flush=True)

            mask = (depths == d0) | (depths == d1)
            X_pair = X[mask]
            y_pair = (depths[mask] == d1).astype(int)
            groups_pair = doc_ids[mask]
            print(f"  Samples: {len(y_pair)} (d{d0}={int((y_pair==0).sum())}, d{d1}={int((y_pair==1).sum())})", flush=True)

            oof_proba, fold_aucs = get_oof_predictions(X_pair, y_pair, groups_pair, SEED)
            print(f"  Fold AUCs: {[round(a,4) for a in fold_aucs]}", flush=True)

            pair_result = {
                "fold_aucs": [round(a, 4) for a in fold_aucs],
                "mean_fold_auc": round(float(np.mean(fold_aucs)), 4),
            }

            for B in B_VALUES:
                rng = np.random.RandomState(SEED)
                point_auc, ci_lo, ci_hi, boot_std = bootstrap_ci(
                    y_pair, oof_proba, groups_pair, B, rng
                )
                v = verdict(ci_lo, ci_hi, point_auc)
                ci_width = ci_hi - ci_lo

                bkey = f"B{B}"
                pair_result[bkey] = {
                    "auc": round(point_auc, 4),
                    "ci": [round(ci_lo, 4), round(ci_hi, 4)],
                    "ci_width": round(ci_width, 4),
                    "boot_std": round(boot_std, 4),
                    "verdict": v,
                }
                print(f"  B={B:>5d}: AUC={point_auc:.4f}  CI=[{ci_lo:.4f}, {ci_hi:.4f}]  w={ci_width:.4f}  -> {v}", flush=True)

            # Check convergence: width change from 1K to 5K and 5K to 10K
            w1k = pair_result["B1000"]["ci_width"]
            w5k = pair_result["B5000"]["ci_width"]
            w10k = pair_result["B10000"]["ci_width"]
            pair_result["convergence"] = {
                "width_change_1k_to_5k_pct": round((w5k - w1k) / w1k * 100, 2) if w1k > 0 else 0,
                "width_change_5k_to_10k_pct": round((w10k - w5k) / w5k * 100, 2) if w5k > 0 else 0,
                "width_change_1k_to_10k_pct": round((w10k - w1k) / w1k * 100, 2) if w1k > 0 else 0,
                "verdict_stable_1k_5k": pair_result["B1000"]["verdict"] == pair_result["B5000"]["verdict"],
                "verdict_stable_5k_10k": pair_result["B5000"]["verdict"] == pair_result["B10000"]["verdict"],
                "verdict_stable_all": (pair_result["B1000"]["verdict"] == pair_result["B5000"]["verdict"] == pair_result["B10000"]["verdict"]),
            }

            # Flag borderline cases
            is_borderline = any(
                pair_result[f"B{B}"]["verdict"].startswith("BORDERLINE")
                or abs(pair_result[f"B{B}"]["ci"][0] - THETA) < 0.02
                for B in B_VALUES
            )
            pair_result["is_borderline"] = is_borderline
            if is_borderline:
                borderline_summary.append({
                    "model": name,
                    "pair": pair_key,
                    "auc": pair_result["B10000"]["auc"],
                    "ci_1k_lower": pair_result["B1000"]["ci"][0],
                    "ci_5k_lower": pair_result["B5000"]["ci"][0],
                    "ci_10k_lower": pair_result["B10000"]["ci"][0],
                    "verdict_1k": pair_result["B1000"]["verdict"],
                    "verdict_5k": pair_result["B5000"]["verdict"],
                    "verdict_10k": pair_result["B10000"]["verdict"],
                    "verdict_stable": pair_result["convergence"]["verdict_stable_all"],
                })

            elapsed = time.time() - t0
            print(f"  Convergence: w_change 1K->5K={pair_result['convergence']['width_change_1k_to_5k_pct']:.1f}%  5K->10K={pair_result['convergence']['width_change_5k_to_10k_pct']:.1f}%  verdict_stable={pair_result['convergence']['verdict_stable_all']}", flush=True)
            print(f"  Done in {elapsed:.1f}s", flush=True)

            model_results[pair_key] = pair_result

        all_results[name] = model_results

    total_time = time.time() - t_start

    output = {
        "description": "Bootstrap CI convergence: B=1000 vs B=5000 vs B=10000",
        "method": "GroupKFold(5) OOF + doc-level bootstrap",
        "theta": THETA,
        "seed": SEED,
        "B_values": B_VALUES,
        "total_seconds": round(total_time, 1),
        "results": all_results,
        "borderline_summary": borderline_summary,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "stability_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    # Generate LaTeX table
    latex_lines = []
    latex_lines.append(r"\begin{table}[t]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Bootstrap CI stability: $B{=}1\text{K}$ vs $B{=}5\text{K}$ vs $B{=}10\text{K}$}")
    latex_lines.append(r"\label{tab:bootstrap-stability}")
    latex_lines.append(r"\small")
    latex_lines.append(r"\begin{tabular}{llccccc}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"Model & Pair & AUC & CI$_{1\text{K}}$ & CI$_{5\text{K}}$ & CI$_{10\text{K}}$ & $\Delta w$ (\%) \\")
    latex_lines.append(r"\midrule")

    for model_name, model_res in all_results.items():
        short_name = model_name.split("(")[0].strip()
        first_row = True
        for pair_key, pr in model_res.items():
            mname = short_name if first_row else ""
            auc = pr["B10000"]["auc"]
            ci1 = pr["B1000"]["ci"]
            ci5 = pr["B5000"]["ci"]
            ci10 = pr["B10000"]["ci"]
            dw = pr["convergence"]["width_change_1k_to_10k_pct"]

            bold = pr["is_borderline"]
            fmt = lambda lo, hi: f"[{lo:.3f}, {hi:.3f}]"

            row = f"  {mname} & {pair_key} & {auc:.3f} & {fmt(ci1[0], ci1[1])} & {fmt(ci5[0], ci5[1])} & {fmt(ci10[0], ci10[1])} & {dw:+.1f} \\\\"
            if bold:
                row = row.replace(pair_key, r"\textbf{" + pair_key + "}")
            latex_lines.append(row)
            first_row = False

        latex_lines.append(r"\midrule")

    latex_lines[-1] = r"\bottomrule"
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")

    latex_path = "/root/autodl-tmp/gen-depth-contamination/artifacts/bootstrap_stability_table.tex"
    Path(latex_path).parent.mkdir(parents=True, exist_ok=True)
    with open(latex_path, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"LaTeX table saved to {latex_path}", flush=True)

    # Final summary
    print(f"\n{'='*70}", flush=True)
    print("BORDERLINE CASES SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    for bs in borderline_summary:
        stable = "STABLE" if bs["verdict_stable"] else "UNSTABLE"
        print(f"  {bs['model']} {bs['pair']}: AUC={bs['auc']:.4f}", flush=True)
        print(f"    CI lower: 1K={bs['ci_1k_lower']:.4f}  5K={bs['ci_5k_lower']:.4f}  10K={bs['ci_10k_lower']:.4f}", flush=True)
        print(f"    Verdict:  1K={bs['verdict_1k']}  5K={bs['verdict_5k']}  10K={bs['verdict_10k']}  -> {stable}", flush=True)

    total_pairs = sum(len(mr) for mr in all_results.values())
    unstable = sum(1 for bs in borderline_summary if not bs["verdict_stable"])
    print(f"\nTotal pairs analyzed: {total_pairs}", flush=True)
    print(f"Borderline cases: {len(borderline_summary)}", flush=True)
    print(f"Verdict unstable across B: {unstable}", flush=True)
    print(f"Total time: {total_time:.1f}s", flush=True)


if __name__ == "__main__":
    main()
