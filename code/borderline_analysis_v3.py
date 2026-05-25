import pandas as pd
import numpy as np
import json
import os
from collections import OrderedDict
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

BASE = "/root/autodl-tmp/gen-depth-contamination"
OUT = os.path.join(BASE, "artifacts", "borderline_analysis")
os.makedirs(OUT, exist_ok=True)

EXPERIMENTS = OrderedDict({
    "Pythia-greedy":  {"path": f"{BASE}/data_exp020_pythia_greedy/features.csv",  "auc": 0.8838, "cat": "strong"},
    "Pythia-nuc09":   {"path": f"{BASE}/data_exp020_pythia_nuc09/features.csv",   "auc": 0.6770, "cat": "strong"},
    "Qwen-1.5B-base": {"path": f"{BASE}/data_exp016_qwen_base/features.csv",     "auc": 0.6262, "cat": "strong"},
    "Pythia-t12":     {"path": f"{BASE}/data_exp020_pythia_t12/features.csv",     "auc": 0.6172, "cat": "borderline"},
    "LLaMA-3.1-8B":  {"path": f"{BASE}/data/exp_018_llama8b/features.csv",       "auc": 0.6123, "cat": "borderline"},
    "Mistral-7B":     {"path": f"{BASE}/data_exp019_mistral7b/features.csv",      "auc": 0.5935, "cat": "below_theta"},
})

FEATURE_COLS = [
    "surp_mean", "surp_std", "surp_skew", "surp_kurt",
    "surp_d1_mean", "surp_d1_std", "surp_d1_skew",
    "surp_d2_mean", "surp_d2_std",
    "ttr", "hapax_ratio", "self_bleu",
    "freq_kurtosis", "freq_entropy", "low_freq_ratio"
]

SURP_FEATURES = ["surp_mean", "surp_std", "surp_d1_std", "surp_d2_std"]
LEX_FEATURES = ["ttr", "hapax_ratio", "freq_entropy", "low_freq_ratio"]

def load_clean(path):
    df = pd.read_csv(path)
    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    return df

def load_pair(path, d_low, d_high):
    """Load depth pair: d_low vs d_high. Label: 0=d_low, 1=d_high."""
    df = load_clean(path)
    df = df[df["depth"].isin([d_low, d_high])].copy()
    df["label"] = (df["depth"] == d_high).astype(int)
    return df

def cohens_d(g0, g1):
    g0 = g0.dropna()
    g1 = g1.dropna()
    n0, n1 = len(g0), len(g1)
    if n0 < 2 or n1 < 2:
        return np.nan
    m0, m1 = g0.mean(), g1.mean()
    s0, s1 = g0.std(ddof=1), g1.std(ddof=1)
    sp = np.sqrt(((n0-1)*s0**2 + (n1-1)*s1**2) / (n0+n1-2))
    if sp < 1e-12:
        return 0.0
    return (m1 - m0) / sp

def train_lgbm_importance(df, features):
    df_clean = df.copy()
    for c in features:
        df_clean[c] = df_clean[c].fillna(df_clean[c].median())
    
    X = df_clean[features].values
    y = df_clean["label"].values
    groups = df_clean["doc_id"].values
    
    gkf = GroupKFold(n_splits=5)
    importances = np.zeros(len(features))
    aucs = []
    
    for train_idx, val_idx in gkf.split(X, y, groups):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            importance_type='gain', verbose=-1, random_state=42
        )
        model.fit(X_tr, y_tr, feature_name=features)
        importances += model.feature_importances_
        preds = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, preds))
    
    importances /= 5
    return importances, np.mean(aucs), aucs

# ============ MAIN ANALYSIS: 1v2 pair ============
results = {}

print("=" * 70)
print("C4 Borderline Feature-Level Error Analysis (1v2 pair)")
print("=" * 70)

for name, info in EXPERIMENTS.items():
    print(f"\n--- {name} (reported 1v2 AUC={info['auc']:.4f}, {info['cat']}) ---")
    
    # 1v2 = depth 1 vs depth 2
    df = load_pair(info["path"], d_low=1, d_high=2)
    n1 = sum(df.label==0)
    n2 = sum(df.label==1)
    print(f"  Samples: depth1={n1}, depth2={n2}")
    
    imp, auc_check, fold_aucs = train_lgbm_importance(df, FEATURE_COLS)
    imp_dict = {f: float(v) for f, v in zip(FEATURE_COLS, imp)}
    sorted_feats = sorted(imp_dict.items(), key=lambda x: -x[1])
    print(f"  Reproduced AUC: {auc_check:.4f} (fold: {[round(a,4) for a in fold_aucs]})")
    print(f"  vs reported:     {info['auc']:.4f}")
    
    # Cohen's d: depth1 vs depth2
    d1_data = df[df.label == 0]
    d2_data = df[df.label == 1]
    cd = {}
    for f in FEATURE_COLS:
        cd[f] = cohens_d(d1_data[f], d2_data[f])
    
    sorted_cd = sorted(cd.items(), key=lambda x: -abs(x[1]) if not np.isnan(x[1]) else 0)
    print(f"  Top-5 |Cohen's d|: {[(f, round(v,4)) for f,v in sorted_cd[:5]]}")
    
    # Feature importance top-5
    imp_total = sum(imp_dict.values())
    top5 = [(f, v/imp_total*100) for f, v in sorted_feats[:5]]
    print(f"  Top-5 importance: {[(f, round(p,1)) for f,p in top5]}")
    
    # Full depth profile for surprisal features
    df_full = load_clean(info["path"])
    surp_by_depth = df_full.groupby("depth")["surp_mean"].mean()
    surp_std_by_depth = df_full.groupby("depth")["surp_std"].mean()
    surp_d1std_by_depth = df_full.groupby("depth")["surp_d1_std"].mean()
    ttr_by_depth = df_full.groupby("depth")["ttr"].mean()
    
    delta_surp_01 = float(surp_by_depth.get(1, np.nan) - surp_by_depth.get(0, np.nan))
    delta_surp_12 = float(surp_by_depth.get(2, np.nan) - surp_by_depth.get(1, np.nan))
    
    mean_abs_d = np.nanmean([abs(v) for v in cd.values()])
    
    results[name] = {
        "cat": info["cat"],
        "auc_1v2_reported": info["auc"],
        "auc_1v2_reproduced": round(auc_check, 4),
        "fold_aucs": [round(a, 4) for a in fold_aucs],
        "feature_importance": imp_dict,
        "feature_importance_pct": {f: round(v/imp_total*100, 2) for f, v in imp_dict.items()},
        "top5_features": [(f, round(p, 1)) for f, p in top5],
        "cohens_d": {k: (round(v, 4) if not np.isnan(v) else None) for k, v in cd.items()},
        "mean_abs_d": round(mean_abs_d, 4),
        "surp_mean_by_depth": {int(k): round(float(v), 4) for k, v in surp_by_depth.items()},
        "surp_std_by_depth": {int(k): round(float(v), 4) for k, v in surp_std_by_depth.items()},
        "surp_d1std_by_depth": {int(k): round(float(v), 4) for k, v in surp_d1std_by_depth.items()},
        "ttr_by_depth": {int(k): round(float(v), 4) for k, v in ttr_by_depth.items()},
        "surp_delta_01": round(delta_surp_01, 4),
        "surp_delta_12": round(delta_surp_12, 4),
    }

# Save full results
with open(os.path.join(OUT, "full_results.json"), "w") as fp:
    json.dump(results, fp, indent=2, default=str)
print(f"\nFull results -> {OUT}/full_results.json")

# ============ Print comparison tables ============
print("\n" + "=" * 70)
print("CROSS-MODEL COMPARISON (1v2)")
print("=" * 70)

print("\n--- Feature Importance (%) ---")
hdr = f"{'Feature':<18}"
for n in EXPERIMENTS:
    hdr += f"  {n[:14]:>14}"
print(hdr)
print("-" * len(hdr))
for f in FEATURE_COLS:
    row = f"{f:<18}"
    for n in EXPERIMENTS:
        pct = results[n]["feature_importance_pct"][f]
        row += f"  {pct:>13.1f}%"
    print(row)

print("\n--- Cohen's d (depth1 vs depth2) ---")
hdr = f"{'Feature':<18}"
for n in EXPERIMENTS:
    hdr += f"  {n[:14]:>14}"
print(hdr)
print("-" * len(hdr))
for f in FEATURE_COLS:
    row = f"{f:<18}"
    for n in EXPERIMENTS:
        d = results[n]["cohens_d"].get(f)
        if d is None:
            row += f"  {'nan':>14}"
        else:
            row += f"  {d:>+14.4f}"
    print(row)

print("\n--- Mean |d| ---")
for n in EXPERIMENTS:
    print(f"  {n:20s} ({results[n]['cat']:12s}): mean|d| = {results[n]['mean_abs_d']:.4f}, AUC = {results[n]['auc_1v2_reported']:.4f}")

print("\n--- Surprisal Deltas ---")
print(f"  {'Model':20s} {'delta(0->1)':>12} {'delta(1->2)':>12} {'ratio':>8}")
for n in EXPERIMENTS:
    d01 = results[n]["surp_delta_01"]
    d12 = results[n]["surp_delta_12"]
    ratio = d12 / d01 if abs(d01) > 1e-6 else float('inf')
    print(f"  {n:20s} {d01:>+12.4f} {d12:>+12.4f} {ratio:>8.3f}")

# Surprisal depth profiles
print("\n--- surp_mean depth profile ---")
for n in EXPERIMENTS:
    depths = results[n]["surp_mean_by_depth"]
    s = f"  {n:20s}:"
    for k in sorted(depths):
        s += f"  d{k}={depths[k]:.3f}"
    print(s)

print("\n--- surp_d1_std depth profile ---")
for n in EXPERIMENTS:
    depths = results[n]["surp_d1std_by_depth"]
    s = f"  {n:20s}:"
    for k in sorted(depths):
        s += f"  d{k}={depths[k]:.3f}"
    print(s)

# ============ LaTeX Tables ============

# Table 1: Main summary
lines = []
lines.append(r"\begin{table*}[t]")
lines.append(r"\centering")
lines.append(r"\small")
lines.append(r"\caption{Feature-level analysis of C4 1v2 classification across model categories. ``Imp.'' = normalized LightGBM gain importance (\%). $d$ = Cohen's $d$ between depth-1 and depth-2 samples. $\bar{|d|}$ = mean absolute $d$ over all features.}")
lines.append(r"\label{tab:borderline_features}")
lines.append(r"\begin{tabular}{@{}llc lcc@{}}")
lines.append(r"\toprule")
lines.append(r"Category & Model & AUC & Top-3 Features (Imp.\%) & $\bar{|d|}$ & $\Delta\text{surp}_{1\to2}$ \\")
lines.append(r"\midrule")

prev_cat = None
for name, r in results.items():
    cat = r["cat"]
    if cat != prev_cat and prev_cat is not None:
        lines.append(r"\midrule")
    prev_cat = cat
    
    cat_label = {"strong": "Strong", "borderline": "Borderline", "below_theta": r"Below $\theta$"}[cat]
    auc = r["auc_1v2_reported"]
    sorted_feats = sorted(r["feature_importance"].items(), key=lambda x: -x[1])
    imp_total = sum(r["feature_importance"].values())
    top3_parts = []
    for f, v in sorted_feats[:3]:
        fname = f.replace("_", r"\_")
        pct = v / imp_total * 100
        top3_parts.append(f"{fname} ({pct:.0f})")
    top3_str = ", ".join(top3_parts)
    
    lines.append(f"{cat_label} & {name} & {auc:.3f} & {top3_str} & {r['mean_abs_d']:.3f} & {r['surp_delta_12']:+.3f} \\\\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table*}")

with open(os.path.join(OUT, "borderline_features.tex"), "w") as fp:
    fp.write("\n".join(lines))
print(f"\nLaTeX main table -> {OUT}/borderline_features.tex")

# Table 2: Full Cohen's d
lines2 = []
lines2.append(r"\begin{table*}[t]")
lines2.append(r"\centering")
lines2.append(r"\scriptsize")
lines2.append(r"\caption{Cohen's $d$ (depth-1 vs.\ depth-2) for all features across C4 models. \textbf{Bold}: largest $|d|$ in each row.}")
lines2.append(r"\label{tab:cohens_d_full}")
ncols = len(EXPERIMENTS)
lines2.append(r"\begin{tabular}{@{}l" + "r" * ncols + r"@{}}")
lines2.append(r"\toprule")
header = "Feature"
for n in EXPERIMENTS:
    header += f" & \\rotatebox{{60}}{{{n}}}"
header += r" \\"
lines2.append(header)
lines2.append(r"\midrule")

for f in FEATURE_COLS:
    vals = [results[n]["cohens_d"].get(f) for n in EXPERIMENTS]
    valid_vals = [v for v in vals if v is not None]
    max_abs = max((abs(v) for v in valid_vals), default=0)
    
    cells = []
    for v in vals:
        if v is None:
            cells.append("---")
        elif abs(v) >= max_abs - 1e-6 and max_abs > 0.01:
            cells.append(f"\\textbf{{{v:+.3f}}}")
        else:
            cells.append(f"{v:+.3f}")
    
    fname = f.replace("_", "\\_")
    lines2.append(f"{fname} & {' & '.join(cells)} \\\\")

lines2.append(r"\midrule")
mean_cells = [f"{results[n]['mean_abs_d']:.3f}" for n in EXPERIMENTS]
lines2.append(f"$\\bar{{|d|}}$ & {' & '.join(mean_cells)} \\\\")

lines2.append(r"\bottomrule")
lines2.append(r"\end{tabular}")
lines2.append(r"\end{table*}")

with open(os.path.join(OUT, "cohens_d_full.tex"), "w") as fp:
    fp.write("\n".join(lines2))
print(f"Cohen's d table -> {OUT}/cohens_d_full.tex")

print("\n=== ALL DONE ===")
