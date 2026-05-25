#!/usr/bin/env python3
"""exp_027 Bootstrap CI: doc-level bootstrap on OOF predictions + permutation test."""

import json, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

DATA_DIR = Path("/root/autodl-tmp/gen-depth-contamination/data/exp_027_latex_ablation")
RESULTS_DIR = Path("/root/autodl-tmp/gen-depth-contamination/results/exp_027_latex_ablation")
N_BOOT = 10_000
N_PERM = 1_000
SEED = 42
THETA = 0.60

LGB_PARAMS = dict(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    num_leaves=31, verbose=-1, n_jobs=-1,
)


def main():
    t0 = time.time()
    df = pd.read_csv(DATA_DIR / "features.csv")
    feat_cols = [c for c in df.columns if c not in ("doc_id", "depth")]
    print(f"Data: {len(df)} rows, {len(feat_cols)} features, depths {sorted(df.depth.unique())}", flush=True)

    pairs = ["0v1", "1v2", "2v3", "3v4", "4v5"]
    results = {}

    for pair in pairs:
        lo, hi = int(pair[0]), int(pair[2])
        sub = df[df["depth"].isin([lo, hi])].copy().reset_index(drop=True)
        X = sub[feat_cols].values
        y = (sub["depth"] == hi).astype(int).values
        doc_ids = sub["doc_id"].values
        unique_docs = np.unique(doc_ids)

        print(f"\n--- {pair}: {len(sub)} samples, {len(unique_docs)} docs ---", flush=True)

        # Step 1: GroupKFold(5) OOF predictions
        oof = np.zeros(len(X))
        fold_aucs = []
        gkf = GroupKFold(n_splits=5)
        for fold_i, (tr, te) in enumerate(gkf.split(X, y, groups=doc_ids)):
            clf = lgb.LGBMClassifier(**LGB_PARAMS)
            clf.fit(X[tr], y[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
            fold_aucs.append(roc_auc_score(y[te], oof[te]))

        point_est = roc_auc_score(y, oof)
        print(f"  OOF AUC = {point_est:.4f}  folds={[round(a,4) for a in fold_aucs]}", flush=True)

        # Step 2: Doc-level bootstrap CI on OOF predictions
        doc_to_idx = {d: np.where(doc_ids == d)[0] for d in unique_docs}
        rng = np.random.RandomState(SEED)
        boot_aucs = []
        t1 = time.time()
        for bi in range(N_BOOT):
            bd = rng.choice(unique_docs, len(unique_docs), replace=True)
            idx = np.concatenate([doc_to_idx[d] for d in bd])
            yb, pb = y[idx], oof[idx]
            if len(np.unique(yb)) < 2:
                continue
            boot_aucs.append(roc_auc_score(yb, pb))
            if (bi + 1) % 2000 == 0:
                print(f"    bootstrap {bi+1}/{N_BOOT} ({time.time()-t1:.1f}s)", flush=True)

        ci_lo = float(np.percentile(boot_aucs, 2.5))
        ci_hi = float(np.percentile(boot_aucs, 97.5))
        print(f"  CI95 = [{ci_lo:.4f}, {ci_hi:.4f}]  boot_mean={np.mean(boot_aucs):.4f}  ({len(boot_aucs)}/{N_BOOT} valid, {time.time()-t1:.1f}s)", flush=True)

        result = {
            "point_estimate": round(point_est, 4),
            "boot_mean": round(float(np.mean(boot_aucs)), 4),
            "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4),
            "per_fold": [round(a, 4) for a in fold_aucs],
            "n_boot_valid": len(boot_aucs),
        }

        # Step 3: Permutation test (2v3 only)
        if pair == "2v3":
            print(f"  Permutation test ({N_PERM} perms)...", flush=True)
            rng_p = np.random.RandomState(SEED + 1)
            perm_aucs = []
            for pi in range(N_PERM):
                y_perm = rng_p.permutation(y)
                if len(np.unique(y_perm)) < 2:
                    continue
                perm_aucs.append(roc_auc_score(y_perm, oof))

            p_val = float(np.mean([a >= point_est for a in perm_aucs]))
            result["p_value"] = round(p_val, 6)
            result["perm_null_mean"] = round(float(np.mean(perm_aucs)), 4)
            result["perm_null_std"] = round(float(np.std(perm_aucs)), 4)
            result["n_perm"] = len(perm_aucs)
            print(f"  Perm: p={p_val:.6f}  null={np.mean(perm_aucs):.4f}+/-{np.std(perm_aucs):.4f}", flush=True)

        results[pair] = result

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("RESULTS SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    for pair, r in results.items():
        s = f"  {pair}: AUC={r['point_estimate']:.4f} [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]"
        if "p_value" in r:
            s += f"  p={r['p_value']:.6f}"
        s += f"  ci_lo {'>' if r['ci_lower'] > THETA else '<='} theta={THETA}"
        print(s, flush=True)

    r23 = results["2v3"]
    verdict = r23["ci_lower"] > THETA
    print(f"\n*** 2v3 CI lower = {r23['ci_lower']:.4f} {'>' if verdict else '<='} theta={THETA}", flush=True)
    print(f"    -> {'CONFIRMED: signal above threshold' if verdict else 'BELOW threshold'}", flush=True)
    print(f"\nTotal: {time.time()-t0:.1f}s", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "bootstrap_ci_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
