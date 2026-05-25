"""McNemar test: OBD-LightGBM vs baseline classifiers on 1v2 binary task."""
import csv, json, os, sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from scipy.stats import chi2
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

BASE = Path("/root/autodl-tmp/gen-depth-contamination")

CONFIGS = {
    "Qwen-1.5B-base / C4":  BASE / "data_exp016_qwen_base" / "features.csv",
    "Pythia-1.4B / C4":      BASE / "data_exp020_pythia_nuc09" / "features.csv",
    "LLaMA-8B / C4":         BASE / "data" / "exp_018_llama8b" / "features.csv",
    "Qwen-1.5B / Wiki":      BASE / "data" / "exp_023_wiki" / "features.csv",
    "Qwen-1.5B / arXiv":     BASE / "data" / "exp_024_arxiv" / "features.csv",
    "Pythia / Wiki":          BASE / "data" / "exp_wiki_pythia" / "features.csv",
    "OLMo / Wiki":            BASE / "data" / "exp_wiki_olmo" / "features.csv",
    "Qwen-7B / arXiv":       BASE / "data" / "exp_arXiv_qwen7b" / "features.csv",
    "Mistral-7B / arXiv":    BASE / "data" / "exp_arXiv_mistral7b" / "features.csv",
    "Pythia / arXiv":         BASE / "data" / "exp_025_pythia_arxiv" / "features.csv",
    "OLMo / arXiv":           BASE / "data" / "exp_arXiv_olmo" / "features.csv",
    "LLaMA-8B / Wiki":       BASE / "data" / "exp_wiki_llama8b" / "features.csv",
}

FEAT_COLS = [
    "surp_mean","surp_std","surp_skew","surp_kurt",
    "surp_d1_mean","surp_d1_std","surp_d1_skew",
    "surp_d2_mean","surp_d2_std",
    "ttr","hapax_ratio","self_bleu",
    "freq_kurtosis","freq_entropy","low_freq_ratio",
]

def P(msg):
    print(msg, flush=True)

def load_1v2(csv_path):
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if int(row["depth"]) in (1, 2):
                rows.append(row)
    if not rows:
        return None, None, None
    doc_ids = np.array([int(r["doc_id"]) for r in rows])
    y = (np.array([int(r["depth"]) for r in rows]) == 2).astype(int)
    X = np.zeros((len(rows), len(FEAT_COLS)))
    for j, c in enumerate(FEAT_COLS):
        for i, r in enumerate(rows):
            try: X[i,j] = float(r[c])
            except: X[i,j] = np.nan
    return X, y, doc_ids

def mcnemar_test(y_true, pred_a, pred_b):
    ca, cb = (pred_a == y_true), (pred_b == y_true)
    b, c = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    if b + c == 0: return 1.0, b, c
    stat = (abs(b - c) - 1)**2 / (b + c)
    return float(1 - chi2.cdf(stat, df=1)), b, c

def oof(X, y, groups, factory, pipe=False):
    preds = np.zeros_like(y)
    for tri, tei in GroupKFold(5).split(X, y, groups=groups):
        if pipe:
            p = Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler()), ("clf", factory())])
            p.fit(X[tri], y[tri]); preds[tei] = p.predict(X[tei])
        else:
            clf = factory(); clf.fit(X[tri], y[tri]); preds[tei] = clf.predict(X[tei])
    return preds

results = {}
for name, path in CONFIGS.items():
    if not path.exists():
        P(f"SKIP {name}"); continue
    P(f"\n=== {name} ===")
    X, y, g = load_1v2(path)
    if X is None: P("  no data"); continue
    P(f"  n={len(y)} d1={int((y==0).sum())} d2={int((y==1).sum())} nan={int(np.isnan(X).sum())}")

    p_obd = oof(X, y, g, lambda: lgb.LGBMClassifier(n_estimators=200, max_depth=6,
        learning_rate=0.05, num_leaves=31, verbose=-1, n_jobs=-1))
    a_obd = accuracy_score(y, p_obd)
    P(f"  OBD acc={a_obd:.4f}")

    res = {"n": int(len(y)), "acc_obd": round(float(a_obd), 4)}
    for bl, fac in [("LogReg", lambda: LogisticRegression(max_iter=2000, C=1.0)),
                    ("LinearSVC", lambda: LinearSVC(max_iter=10000, C=1.0)),
                    ("RF", lambda: RandomForestClassifier(n_estimators=100, max_depth=8, n_jobs=-1, random_state=42))]:
        p_bl = oof(X, y, g, fac, pipe=True)
        a_bl = accuracy_score(y, p_bl)
        pv, b, c = mcnemar_test(y, p_obd, p_bl)
        sig = "***" if pv<.001 else ("**" if pv<.01 else ("*" if pv<.05 else "ns"))
        P(f"  vs {bl}: acc={a_bl:.4f} p={pv:.2e} [{sig}] b={b} c={c}")
        res[bl] = {"acc": round(float(a_bl),4), "p": round(pv,6), "b": b, "c": c}
    results[name] = res

out = BASE / "results" / "mcnemar_obd_vs_baselines"
out.mkdir(parents=True, exist_ok=True)
with open(out / "mcnemar_results.json", "w") as f:
    json.dump(results, f, indent=2)

P("\n" + "="*60 + "\nSUMMARY\n" + "="*60)
all_p = [(cfg,bl,r[bl]["p"]) for cfg,r in results.items() for bl in ["LogReg","LinearSVC","RF"] if bl in r]
if all_p:
    mx = max(all_p, key=lambda x: x[2])
    P(f"Comparisons: {len(all_p)}")
    P(f"Max p: {mx[2]:.2e} ({mx[0]} vs {mx[1]})")
    P(f"All<0.01: {all(x[2]<0.01 for x in all_p)}")
    P(f"All<0.05: {all(x[2]<0.05 for x in all_p)}")
    bad = [x for x in all_p if x[2]>=0.01]
    if bad:
        P(f"p>=0.01 ({len(bad)}):")
        for x in sorted(bad, key=lambda x:-x[2]):
            P(f"  {x[0]} vs {x[1]}: p={x[2]:.4e}")
P(f"\nSaved: {out/'mcnemar_results.json'}")
