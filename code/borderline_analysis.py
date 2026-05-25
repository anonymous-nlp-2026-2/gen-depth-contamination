import pandas as pd
import numpy as np
import json
import os
from collections import OrderedDict

# Check if lightgbm is available
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("WARNING: lightgbm not available, will use sklearn GradientBoosting")
    from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

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

def load_1v2(path):
    df = pd.read_csv(path)
    return df[df["depth"].isin([0, 1])].copy()

def cohens_d(g0, g1):
    n0, n1 = len(g0), len(g1)
    m0, m1 = g0.mean(), g1.mean()
    s0, s1 = g0.std(ddof=1), g1.std(ddof=1)
    sp = np.sqrt(((n0-1)*s0**2 + (n1-1)*s1**2) / (n0+n1-2))
    if sp == 0:
        return 0.0
    return (m1 - m0) / sp

def train_lgbm_importance(df, features):
    X = df[features].values
    y = df["depth"].values
    groups = df["doc_id"].values
    
    gkf = GroupKFold(n_splits=5)
    importances = np.zeros(len(features))
    aucs = []
    
    for train_idx, val_idx in gkf.split(X, y, groups):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        if HAS_LGB:
            model = lgb.LGBMClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                importance_type='gain', verbose=-1, random_state=42
            )
            model.fit(X_tr, y_tr)
            importances += model.feature_importances_
        else:
            model = GradientBoostingClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42
            )
            model.fit(X_tr, y_tr)
            importances += model.feature_importances_
        
        preds = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, preds))
    
    importances /= 5
    return importances, np.mean(aucs)

# ============ MAIN ANALYSIS ============
results = {}

print("=" * 70)
print("C4 Borderline Feature-Level Error Analysis")
print("=" * 70)

for name, info in EXPERIMENTS.items():
    print(f"\n--- {name} (1v2 AUC={info['auc']:.4f}, {info['cat']}) ---")
    df = load_1v2(info["path"])
    print(f"  Samples: depth0={sum(df.depth==0)}, depth1={sum(df.depth==1)}")
    
    # Feature importance
    imp, auc_check = train_lgbm_importance(df, FEATURE_COLS)
    print(f"  Reproduced AUC: {auc_check:.4f}")
    
    imp_dict = {f: float(v) for f, v in zip(FEATURE_COLS, imp)}
    sorted_feats = sorted(imp_dict.items(), key=lambda x: -x[1])
    top5 = [f for f, _ in sorted_feats[:5]]
    print(f"  Top-5 features: {top5}")
    
    # Cohen's d for each feature
    d0 = df[df.depth == 0]
    d1 = df[df.depth == 1]
    cd = {}
    for f in FEATURE_COLS:
        cd[f] = cohens_d(d0[f].values, d1[f].values)
    
    sorted_cd = sorted(cd.items(), key=lambda x: -abs(x[1]))
    print(f"  Top-5 Cohen's d: {[(f, round(v,4)) for f,v in sorted_cd[:5]]}")
    print(f"  Bottom-5 Cohen's d: {[(f, round(v,4)) for f,v in sorted_cd[-5:]]}")
    
    # Surprisal signals across all available depths
    df_full = pd.read_csv(info["path"])
    surp_by_depth = df_full.groupby("depth")["surp_mean"].mean()
    print(f"  surp_mean by depth: {dict(surp_by_depth.round(4))}")
    
    if 0 in surp_by_depth.index and 1 in surp_by_depth.index:
        delta_01 = surp_by_depth[1] - surp_by_depth[0]
        print(f"  surp_mean delta(0->1): {delta_01:.4f}")
    else:
        delta_01 = None
    
    results[name] = {
        "cat": info["cat"],
        "auc_1v2": info["auc"],
        "auc_reproduced": round(auc_check, 4),
        "feature_importance": imp_dict,
        "top5_features": top5,
        "cohens_d": cd,
        "surp_mean_by_depth": {int(k): round(float(v), 4) for k, v in surp_by_depth.items()},
        "surp_delta_01": round(float(delta_01), 4) if delta_01 is not None else None,
    }

# ============ CROSS-MODEL COMPARISON ============
print("\n" + "=" * 70)
print("CROSS-MODEL COMPARISON")
print("=" * 70)

# Feature importance comparison table
print("\n--- Feature Importance (normalized per model) ---")
header = f"{'Feature':<18}"
for name in EXPERIMENTS:
    header += f"  {name[:12]:>12}"
print(header)
print("-" * len(header))

for f in FEATURE_COLS:
    row = f"{f:<18}"
    for name in EXPERIMENTS:
        imp_total = sum(results[name]["feature_importance"].values())
        pct = results[name]["feature_importance"][f] / imp_total * 100
        row += f"  {pct:>11.1f}%"
    print(row)

# Cohen's d comparison
print("\n--- Cohen's d (1v2) ---")
header = f"{'Feature':<18}"
for name in EXPERIMENTS:
    header += f"  {name[:12]:>12}"
print(header)
print("-" * len(header))

for f in FEATURE_COLS:
    row = f"{f:<18}"
    for name in EXPERIMENTS:
        d = results[name]["cohens_d"][f]
        row += f"  {d:>12.4f}"
    print(row)

# Surprisal delta comparison
print("\n--- Surprisal Delta (depth 0 -> 1) ---")
for name in EXPERIMENTS:
    cat = results[name]["cat"]
    delta = results[name]["surp_delta_01"]
    print(f"  {name:<20} ({cat:<12}): delta={delta:.4f}")

# Aggregate: mean |Cohen's d| per category
print("\n--- Mean |Cohen's d| by category ---")
for cat in ["strong", "borderline", "below_theta"]:
    models_in_cat = [n for n, r in results.items() if r["cat"] == cat]
    if not models_in_cat:
        continue
    mean_d = {}
    for f in FEATURE_COLS:
        vals = [abs(results[n]["cohens_d"][f]) for n in models_in_cat]
        mean_d[f] = np.mean(vals)
    sorted_md = sorted(mean_d.items(), key=lambda x: -x[1])
    print(f"\n  {cat} ({', '.join(models_in_cat)}):")
    for f, v in sorted_md[:5]:
        print(f"    {f:<18}: {v:.4f}")
    print(f"    ... weakest:")
    for f, v in sorted_md[-3:]:
        print(f"    {f:<18}: {v:.4f}")

# Save full results
with open(os.path.join(OUT, "full_results.json"), "w") as fp:
    json.dump(results, fp, indent=2, default=str)
print(f"\nFull results saved to {OUT}/full_results.json")

# ============ Generate LaTeX Table ============
print("\n--- Generating LaTeX table ---")

# Table: model x top-5 features (importance %) + Cohen's d
latex_lines = []
latex_lines.append(r"\begin{table*}[t]")
latex_lines.append(r"\centering")
latex_lines.append(r"\small")
latex_lines.append(r"\caption{Feature-level analysis of borderline vs.\ strong models on C4 1v2 classification. Imp.\% = normalized gain importance; $d$ = Cohen's $d$ (depth-0 vs.\ depth-1).}")
latex_lines.append(r"\label{tab:borderline_features}")
latex_lines.append(r"\begin{tabular}{ll r rrrrr rr}")
latex_lines.append(r"\toprule")
latex_lines.append(r"Category & Model & AUC & \multicolumn{5}{c}{Top-5 Features (Imp.\%)} & $\bar{|d|}$ & $\Delta$surp \\")
latex_lines.append(r"\midrule")

for name, r in results.items():
    cat = r["cat"]
    auc = r["auc_1v2"]
    imp_total = sum(r["feature_importance"].values())
    sorted_feats = sorted(r["feature_importance"].items(), key=lambda x: -x[1])
    top5_str = " & ".join([f"\\texttt{{{f.replace('_','-')}}} ({v/imp_total*100:.1f})" for f, v in sorted_feats[:5]])
    
    mean_abs_d = np.mean([abs(v) for v in r["cohens_d"].values()])
    delta_surp = r["surp_delta_01"] if r["surp_delta_01"] is not None else 0
    
    cat_label = {"strong": "Strong", "borderline": "Borderline", "below_theta": r"Below $\theta$"}[cat]
    
    latex_lines.append(f"{cat_label} & {name} & {auc:.4f} & {top5_str} & {mean_abs_d:.3f} & {delta_surp:+.3f} \\\\")

latex_lines.append(r"\bottomrule")
latex_lines.append(r"\end{tabular}")
latex_lines.append(r"\end{table*}")

latex_str = "\n".join(latex_lines)

with open(os.path.join(OUT, "borderline_features.tex"), "w") as fp:
    fp.write(latex_str)
print(f"LaTeX table saved to {OUT}/borderline_features.tex")

# ============ DETAILED PER-FEATURE TABLE ============
latex2 = []
latex2.append(r"\begin{table*}[t]")
latex2.append(r"\centering")
latex2.append(r"\scriptsize")
latex2.append(r"\caption{Cohen's $d$ (depth-0 vs.\ depth-1) for all 15 features across models. Bold = largest $|d|$ per row. Gray = $|d| < 0.05$.}")
latex2.append(r"\label{tab:cohens_d_full}")
latex2.append(r"\begin{tabular}{l" + "r" * len(EXPERIMENTS) + "}")
latex2.append(r"\toprule")
header_names = " & ".join([f"\\rotatebox{{60}}{{{n}}}" for n in EXPERIMENTS])
latex2.append(f"Feature & {header_names} \\\\")
latex2.append(r"\midrule")

for f in FEATURE_COLS:
    vals = [results[n]["cohens_d"][f] for n in EXPERIMENTS]
    max_abs = max(abs(v) for v in vals)
    cells = []
    for v in vals:
        if abs(v) == max_abs and max_abs > 0:
            cells.append(f"\\textbf{{{v:.3f}}}")
        elif abs(v) < 0.05:
            cells.append(f"\\color{{gray}}{{{v:.3f}}}")
        else:
            cells.append(f"{v:.3f}")
    fname = f.replace("_", "\\_")
    latex2.append(f"{fname} & {' & '.join(cells)} \\\\")

latex2.append(r"\bottomrule")
latex2.append(r"\end{tabular}")
latex2.append(r"\end{table*}")

with open(os.path.join(OUT, "cohens_d_full.tex"), "w") as fp:
    fp.write("\n".join(latex2))
print(f"Cohen's d table saved to {OUT}/cohens_d_full.tex")

print("\n=== DONE ===")
