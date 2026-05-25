"""Compute missing table data: bootstrap CIs for Table S10 + Three-way ANOVA for Table 24."""
import csv, json, time, sys
from pathlib import Path
import numpy as np

FEAT_COLS = [
    "surp_mean", "surp_std", "surp_skew", "surp_kurt",
    "surp_d1_mean", "surp_d1_std", "surp_d1_skew",
    "surp_d2_mean", "surp_d2_std",
    "ttr", "hapax_ratio", "self_bleu",
    "freq_kurtosis", "freq_entropy", "low_freq_ratio",
]
DEPTH_PAIRS = [(0,1),(1,2),(2,3),(3,4),(4,5)]
SEED = 42
N_BOOT = 10000

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

def compute_bootstrap_ci(data_dir, label):
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score

    print(f"\n{'='*60}", flush=True)
    print(f"Processing: {label}", flush=True)
    print(f"Data dir: {data_dir}", flush=True)

    X, depths, doc_ids = load_data(data_dir)
    results = {}

    for d0, d1 in DEPTH_PAIRS:
        pair_name = f"{d0}v{d1}"
        t0 = time.time()
        mask = (depths == d0) | (depths == d1)
        X_pair = X[mask]
        y_pair = (depths[mask] == d1).astype(int)
        groups_pair = doc_ids[mask]

        # OOF predictions
        gkf = GroupKFold(n_splits=5)
        oof_proba = np.zeros(len(y_pair))
        fold_aucs = []
        for train_idx, test_idx in gkf.split(X_pair, y_pair, groups=groups_pair):
            clf = lgb.LGBMClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                num_leaves=31, verbose=-1, n_jobs=-1,
            )
            clf.fit(X_pair[train_idx], y_pair[train_idx])
            proba = clf.predict_proba(X_pair[test_idx])[:, 1]
            oof_proba[test_idx] = proba
            fold_aucs.append(roc_auc_score(y_pair[test_idx], proba))

        point_auc = roc_auc_score(y_pair, oof_proba)

        # Bootstrap CI
        unique_docs = np.unique(groups_pair)
        n_docs = len(unique_docs)
        doc_indices = {d: np.where(groups_pair == d)[0] for d in unique_docs}
        doc_list = list(unique_docs)
        rng = np.random.RandomState(SEED)
        boot_aucs = np.empty(N_BOOT)
        for b in range(N_BOOT):
            sampled = rng.randint(0, n_docs, size=n_docs)
            indices = np.concatenate([doc_indices[doc_list[s]] for s in sampled])
            try:
                boot_aucs[b] = roc_auc_score(y_pair[indices], oof_proba[indices])
            except ValueError:
                boot_aucs[b] = np.nan
        boot_aucs = boot_aucs[~np.isnan(boot_aucs)]
        ci_lower = float(np.percentile(boot_aucs, 2.5))
        ci_upper = float(np.percentile(boot_aucs, 97.5))

        elapsed = time.time() - t0
        print(f"  {pair_name}: AUC={point_auc:.4f} [{ci_lower:.4f}, {ci_upper:.4f}] ({elapsed:.1f}s)", flush=True)

        results[pair_name] = {
            "auc": round(point_auc, 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "fold_aucs": [round(a, 4) for a in fold_aucs],
        }

    return results

def compute_anova():
    """Three-way ANOVA: Generator × Scorer × Depth-Pair on per-fold AUC."""
    print(f"\n{'='*60}", flush=True)
    print("Computing Three-way ANOVA", flush=True)

    base = Path("/root/autodl-tmp/gen-depth-contamination/results_exp022_cross_scorer")
    generators = ["pythia_greedy", "qwen_base", "qwen_cont256"]
    scorers = ["olmo", "pythia", "qwen_base", "qwen_instruct"]
    pairs = ["0v1", "1v2", "2v3", "3v4", "4v5"]

    rows = []
    for gen in generators:
        for scorer in scorers:
            d = base / f"{gen}__{scorer}"
            with open(d / "pairwise_auc.json") as f:
                data = json.load(f)
            for pair in pairs:
                folds = data["per_fold"][pair]
                for fold_idx, auc_val in enumerate(folds):
                    rows.append({
                        "generator": gen,
                        "scorer": scorer,
                        "depth_pair": pair,
                        "fold": fold_idx,
                        "auc": auc_val,
                    })

    print(f"  Total observations: {len(rows)}", flush=True)

    try:
        import pandas as pd
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        from statsmodels.stats.anova import anova_lm

        df = pd.DataFrame(rows)
        model = ols("auc ~ C(generator) * C(scorer) * C(depth_pair)", data=df).fit()
        anova_table = anova_lm(model, typ=2)
        print("\nANOVA Table (Type II):", flush=True)
        print(anova_table, flush=True)

        # Compute eta-squared partial
        ss_resid = anova_table.loc["Residual", "sum_sq"]
        results = {}
        for source in anova_table.index:
            if source == "Residual":
                continue
            ss = anova_table.loc[source, "sum_sq"]
            df_val = anova_table.loc[source, "df"]
            f_val = anova_table.loc[source, "F"]
            p_val = anova_table.loc[source, "PR(>F)"]
            eta_sq_p = ss / (ss + ss_resid)
            results[source] = {
                "df": int(df_val),
                "F": round(float(f_val), 2) if not np.isnan(f_val) else None,
                "p": float(p_val) if not np.isnan(p_val) else None,
                "eta_sq_p": round(float(eta_sq_p), 3),
                "SS": round(float(ss), 6),
            }
            print(f"  {source}: df={int(df_val)}, F={f_val:.2f}, p={p_val:.2e}, eta²_p={eta_sq_p:.3f}", flush=True)

        results["Residual"] = {
            "df": int(anova_table.loc["Residual", "df"]),
            "SS": round(float(ss_resid), 6),
        }
        return results

    except ImportError as e:
        print(f"  ERROR: {e}", flush=True)
        print("  Falling back to manual computation...", flush=True)
        return {"error": str(e)}

def main():
    t_start = time.time()
    all_results = {}

    # Part 1: Bootstrap CIs for Pythia decoding ablation (Table S10)
    configs = {
        "pythia_greedy": "/root/autodl-tmp/gen-depth-contamination/data_exp020_pythia_greedy",
        "pythia_nuc09": "/root/autodl-tmp/gen-depth-contamination/data_exp020_pythia_nuc09",
        "pythia_t07": "/root/autodl-tmp/gen-depth-contamination/data_exp020_pythia_t07",
        "pythia_t12": "/root/autodl-tmp/gen-depth-contamination/data_exp020_pythia_t12",
    }
    for name, data_dir in configs.items():
        all_results[name] = compute_bootstrap_ci(data_dir, name)

    # Part 2: Three-way ANOVA (Table 24)
    all_results["anova"] = compute_anova()

    # Save
    out_path = Path("/root/autodl-tmp/gen-depth-contamination/results/missing_tables_data.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)
    print(f"Total time: {time.time() - t_start:.1f}s", flush=True)

if __name__ == "__main__":
    main()
