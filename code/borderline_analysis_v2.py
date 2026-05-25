import pandas as pd
import numpy as np
import json
import os
from collections import OrderedDict
import lightgbm as lgb
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

def load_clean(path):
    df = pd.read_csv(path)
    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    return df

def load_1v2(path):
    df = load_clean(path)
    return df[df["depth"].isin([0, 1])].copy()

def cohens_d(g0, g1):
    g0 = g0.dropna()
    g1 = g1.dropna()
    n0, n1 = len(g0), len(g1)
    if n0 < 2 or n1 < 2:
        return np.nan
    m0, m1 = g0.mean(), g1.mean()
    s0, s1 = g0.std(ddof=1), g1.std(ddof=1)
    sp = np.sqrt(((n0-1)*s0**2 + (n1-1)*s1**2) / (n0+n1-2))
    if sp == 0:
        return 0.0
    return (m1 - m0) / sp

def train_lgbm_importance(df, features):
    df_clean = df.copy()
    for c in features:
        df_clean[c] = df_clean[c].fillna(df_clean[c].median())
    
    X = df_clean[features].values
    y = df_clean["depth"].values
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
    return importances, np.mean(aucs)

results = {}

for name, info in EXPERIMENTS.items():
    print(f"\n--- {name} (1v2 AUC={info['auc']:.4f}, {info['cat']}) ---")
    df = load_1v2(info["path"])
    n0 = sum(df.depth==0)
    n1 = sum(df.depth==1)
    print(f"  Samples: depth0={n0}, depth1={n1}")
    
    imp, auc_check = train_lgbm_importance(df, FEATURE_COLS)
    imp_dict = {f: float(v) for f, v in zip(FEATURE_COLS, imp)}
    sorted_feats = sorted(imp_dict.items(), key=lambda x: -x[1])
    top5 = [(f, v) for f, v in sorted_feats[:5]]
    print(f"  Reproduced AUC: {auc_check:.4f}")
    
    d0 = df[df.depth == 0]
    d1 = df[df.depth == 1]
    cd = {}
    for f in FEATURE_COLS:
        cd[f] = cohens_d(d0[f], d1[f])
    
    df_full = load_clean(info["path"])
    surp_by_depth = df_full.groupby("depth")["surp_mean"].mean()
    delta_01 = float(surp_by_depth.get(1, np.nan) - surp_by_depth.get(0, np.nan))
    
    surp_std_by_depth = df_full.groupby("depth")["surp_std"].mean()
    surp_d1std_by_depth = df_full.groupby("depth")["surp_d1_std"].mean()
    
    mean_abs_d = np.nanmean([abs(v) for v in cd.values()])
    
    results[name] = {
        "cat": info["cat"],
        "auc_1v2": info["auc"],
        "auc_reproduced": round(auc_check, 4),
        "feature_importance": imp_dict,
        "top5_features": top5,
        "cohens_d": {k: (round(v, 4) if not np.isnan(v) else None) for k, v in cd.items()},
        "mean_abs_d": round(mean_abs_d, 4),
        "surp_mean_by_depth": {int(k): round(float(v), 4) for k, v in surp_by_depth.items()},
        "surp_std_by_depth": {int(k): round(float(v), 4) for k, v in surp_std_by_depth.items()},
        "surp_d1std_by_depth": {int(k): round(float(v), 4) for k, v in surp_d1std_by_depth.items()},
        "surp_delta_01": round(delta_01, 4),
    }

with open(os.path.join(OUT, "full_results.json"), "w") as fp:
    json.dump(results, fp, indent=2, default=str)

# ========== LaTeX: Main summary table ==========
lines = []
lines.append(r"\begin{table*}[t]")
lines.append(r"\centering")
lines.append(r"\small")
lines.append(r"\caption{Feature-level analysis for C4 1v2 classification across strong, borderline, and below-threshold models. Imp.\% denotes normalized LightGBM gain importance. $\bar{|d|}$ is the mean absolute Cohen's $d$ across all features. $\Delta_{\mathrm{surp}}$ is the shift in mean surprisal from depth 0 to depth 1.}")
lines.append(r"\label{tab:borderline_features}")

lines.append(r"\begin{tabular}{@{}ll c l c c@{}}")
lines.append(r"\toprule")
lines.append(r"Category & Model & AUC$_{1\text{v}2}$ & Top-3 Features (Imp.\%) & $\bar{|d|}$ & $\Delta_{\mathrm{surp}}$ \\")
lines.append(r"\midrule")

prev_cat = None
for name, r in results.items():
    cat = r["cat"]
    if cat != prev_cat and prev_cat is not None:
        lines.append(r"\midrule")
    prev_cat = cat
    
    cat_label = {"strong": "Strong", "borderline": "Borderline", "below_theta": r"Below $\theta$"}[cat]
    auc = r["auc_1v2"]
    imp_total = sum(r["feature_importance"].values())
    sorted_feats = sorted(r["feature_importance"].items(), key=lambda x: -x[1])
    top3_str = ", ".join([f"\\texttt{{{f.replace('_', '-')}}} ({v/imp_total*100:.0f}\\%)" for f, v in sorted_feats[:3]])
    
    lines.append(f"{cat_label} & {name} & {auc:.3f} & {top3_str} & {r['mean_abs_d']:.2f} & {r['surp_delta_01']:+.2f} \\\\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table*}")

with open(os.path.join(OUT, "borderline_features.tex"), "w") as fp:
    fp.write("\n".join(lines))
print("Main table written.")

# ========== LaTeX: Cohen's d full table ==========
lines2 = []
lines2.append(r"\begin{table*}[t]")
lines2.append(r"\centering")
lines2.append(r"\scriptsize")
lines2.append(r"\caption{Cohen's $d$ (depth 0 vs.\ depth 1) for all features. \textbf{Bold}: largest $|d|$ per row. \colorbox{gray!15}{Shaded}: $|d| < 0.10$.}")
lines2.append(r"\label{tab:cohens_d_full}")

ncols = len(EXPERIMENTS)
lines2.append(r"\begin{tabular}{@{}l" + "r" * ncols + r"@{}}")
lines2.append(r"\toprule")
header = "Feature"
for n in EXPERIMENTS:
    short = n.replace("Pythia-", "Py-").replace("Qwen-1.5B-", "Qw-").replace("LLaMA-3.1-", "LL-").replace("Mistral-", "Mi-")
    header += f" & \\rotatebox{{60}}{{{n}}}"
header += r" \\"
lines2.append(header)
lines2.append(r"\midrule")

for f in FEATURE_COLS:
    vals = []
    for n in EXPERIMENTS:
        v = results[n]["cohens_d"].get(f)
        vals.append(v)
    
    valid_vals = [v for v in vals if v is not None]
    if not valid_vals:
        max_abs = 0
    else:
        max_abs = max(abs(v) for v in valid_vals)
    
    cells = []
    for v in vals:
        if v is None:
            cells.append("---")
        elif abs(v) == max_abs and max_abs > 0.01:
            cells.append(f"\\textbf{{{v:+.3f}}}")
        elif abs(v) < 0.10:
            cells.append(f"\\cellcolor{{gray!15}}{{{v:+.3f}}}")
        else:
            cells.append(f"{v:+.3f}")
    
    fname = f.replace("_", "\\_")
    lines2.append(f"{fname} & {' & '.join(cells)} \\\\")

lines2.append(r"\midrule")
# Mean |d| row
cells_mean = []
for n in EXPERIMENTS:
    cells_mean.append(f"{results[n]['mean_abs_d']:.2f}")
lines2.append(f"$\\bar{{|d|}}$ & {' & '.join(cells_mean)} \\\\")

lines2.append(r"\bottomrule")
lines2.append(r"\end{tabular}")
lines2.append(r"\end{table*}")

with open(os.path.join(OUT, "cohens_d_full.tex"), "w") as fp:
    fp.write("\n".join(lines2))
print("Cohen's d table written.")

# ========== Print key findings for summary ==========
print("\n" + "=" * 60)
print("KEY FINDINGS SUMMARY")
print("=" * 60)

# 1. Feature importance dominance
print("\n1. FEATURE IMPORTANCE DOMINANCE:")
for name, r in results.items():
    imp_total = sum(r["feature_importance"].values())
    surp_d1_std_pct = r["feature_importance"]["surp_d1_std"] / imp_total * 100
    print(f"  {name:20s}: surp_d1_std = {surp_d1_std_pct:.1f}%")

# 2. Cohen's d decay with AUC
print("\n2. COHEN'S D DECAY (key features):")
print(f"  {'Model':20s} {'AUC':>6} {'surp_mean':>10} {'surp_std':>10} {'surp_d1_std':>12} {'ttr':>8} {'freq_ent':>10}")
for name, r in results.items():
    cd = r["cohens_d"]
    print(f"  {name:20s} {r['auc_1v2']:>6.4f} {cd.get('surp_mean',0):>+10.3f} {cd.get('surp_std',0):>+10.3f} {cd.get('surp_d1_std',0):>+12.3f} {cd.get('ttr',0):>+8.3f} {cd.get('freq_entropy',0):>+10.3f}")

# 3. Surprisal signal depth profile
print("\n3. SURPRISAL DEPTH PROFILE (surp_mean):")
for name, r in results.items():
    depths = r["surp_mean_by_depth"]
    profile = " | ".join([f"d{k}={v:.3f}" for k, v in sorted(depths.items())])
    print(f"  {name:20s}: {profile}")

# 4. Why Mistral fails despite similar |d|
print("\n4. MISTRAL vs LLaMA COMPARISON:")
m = results["Mistral-7B"]
l = results["LLaMA-3.1-8B"]
for f in ["surp_mean", "surp_std", "surp_d1_std", "surp_d1_mean", "ttr", "freq_entropy"]:
    md = m["cohens_d"].get(f, 0) or 0
    ld = l["cohens_d"].get(f, 0) or 0
    mi = m["feature_importance"][f] / sum(m["feature_importance"].values()) * 100
    li = l["feature_importance"][f] / sum(l["feature_importance"].values()) * 100
    print(f"  {f:18s}: Mistral d={md:+.3f} imp={mi:.1f}%  |  LLaMA d={ld:+.3f} imp={li:.1f}%")

# 5. surp_d1_std importance vs AUC correlation
print("\n5. FEATURE CONCENTRATION INDEX:")
for name, r in results.items():
    imp_total = sum(r["feature_importance"].values())
    sorted_imp = sorted(r["feature_importance"].values(), reverse=True)
    top1_pct = sorted_imp[0] / imp_total * 100
    top3_pct = sum(sorted_imp[:3]) / imp_total * 100
    print(f"  {name:20s}: top-1={top1_pct:.1f}%, top-3={top3_pct:.1f}%")

print("\nDONE")
