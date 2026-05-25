#!/usr/bin/env python3
"""7B Cross-Model Transfer Matrix.
Train OBD-LightGBM on model A's features, test on model B's.
Diagonal: 5-fold GroupKFold CV. Off-diagonal: 100% train -> 100% test.
"""
import sys, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

def load_features(data_dir):
    df = pd.read_csv(Path(data_dir) / "features.csv")
    feature_cols = [c for c in df.columns if c not in ('doc_id', 'depth')]
    return df, feature_cols

def train_obd_predict(X_train, y_train, X_test, y_test, max_depth_val):
    results = {}
    for k in range(max_depth_val):
        d_lo, d_hi = k, k + 1
        mask_train = np.isin(y_train, [d_lo, d_hi])
        mask_test = np.isin(y_test, [d_lo, d_hi])
        if mask_train.sum() < 10 or mask_test.sum() < 10:
            continue
        yt = (y_train[mask_train] == d_hi).astype(int)
        yv = (y_test[mask_test] == d_hi).astype(int)
        clf = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                             num_leaves=31, verbose=-1, n_jobs=-1)
        clf.fit(X_train[mask_train], yt)
        pred = clf.predict_proba(X_test[mask_test])[:, 1]
        auc = roc_auc_score(yv, pred)
        results[f"{d_lo}v{d_hi}"] = auc
    return results

def run_self_transfer(df, feature_cols, n_folds=5):
    X = df[feature_cols].values
    y = df['depth'].values
    groups = df['doc_id'].values
    max_d = int(y.max())
    all_aucs = {f"{k}v{k+1}": [] for k in range(max_d)}
    gkf = GroupKFold(n_splits=n_folds)
    for train_idx, test_idx in gkf.split(X, y, groups):
        fold_aucs = train_obd_predict(X[train_idx], y[train_idx], X[test_idx], y[test_idx], max_d)
        for pair, auc in fold_aucs.items():
            all_aucs[pair].append(auc)
    return {pair: np.mean(aucs) for pair, aucs in all_aucs.items() if aucs}

def run_cross_transfer(df_train, df_test, feature_cols):
    X_tr = df_train[feature_cols].values
    y_tr = df_train['depth'].values
    X_te = df_test[feature_cols].values
    y_te = df_test['depth'].values
    max_d = min(int(y_tr.max()), int(y_te.max()))
    return train_obd_predict(X_tr, y_tr, X_te, y_te, max_d)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dirs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    assert len(args.data_dirs) == len(args.labels)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    datasets = {}
    for d, label in zip(args.data_dirs, args.labels):
        df, fcols = load_features(d)
        datasets[label] = (df, fcols)
        print(f"Loaded {label}: {len(df)} rows, depths {sorted(df['depth'].unique())}")

    common_cols = None
    for label, (df, fcols) in datasets.items():
        if common_cols is None:
            common_cols = set(fcols)
        else:
            common_cols &= set(fcols)
    common_cols = sorted(common_cols)
    print(f"Common features: {len(common_cols)}")

    labels = args.labels
    matrix = {}

    for train_label in labels:
        for test_label in labels:
            if train_label == test_label:
                aucs = run_self_transfer(datasets[train_label][0], common_cols)
            else:
                aucs = run_cross_transfer(datasets[train_label][0], datasets[test_label][0], common_cols)
            mean_auc = np.mean(list(aucs.values())) if aucs else 0
            matrix[(train_label, test_label)] = {"pairwise": aucs, "mean_auc": mean_auc}
            print(f"  {train_label} -> {test_label}: mean_auc={mean_auc:.4f}, pairs={aucs}")

    out = {"matrix": {f"{k[0]}->{k[1]}": v for k, v in matrix.items()}}
    with open(Path(args.out_dir) / "cross_transfer_7b_c4.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== TRANSFER MATRIX (mean_pairwise_auc) ===")
    header = "Train\\Test\t" + "\t".join(labels)
    print(header)
    for tl in labels:
        row = [tl]
        for tst in labels:
            row.append(f"{matrix[(tl, tst)]['mean_auc']:.4f}")
        print("\t".join(row))

    print("\n=== RETENTION RATES ===")
    for tl in labels:
        diag = matrix[(tl, tl)]['mean_auc']
        off_diag = [matrix[(tl, tst)]['mean_auc'] for tst in labels if tst != tl]
        if diag > 0:
            retention = np.mean(off_diag) / diag * 100
            print(f"{tl}: diag={diag:.4f}, off_diag_mean={np.mean(off_diag):.4f}, retention={retention:.1f}%")

    print("\n=== PAIRWISE AUC TRANSFER (each depth pair) ===")
    for pair_name in ["0v1", "1v2", "2v3", "3v4", "4v5"]:
        print(f"\n--- {pair_name} ---")
        header = "Train\\Test\t" + "\t".join(labels)
        print(header)
        for tl in labels:
            row = [tl]
            for tst in labels:
                auc_val = matrix[(tl, tst)]['pairwise'].get(pair_name, 'N/A')
                row.append(f"{auc_val:.4f}" if isinstance(auc_val, float) else auc_val)
            print("\t".join(row))

if __name__ == "__main__":
    main()
