import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

os.environ["OMP_NUM_THREADS"] = "4"

ROOT = Path("/root/autodl-tmp/gen-depth-contamination")
OUT_DIR = ROOT / "results" / "feature_importance_correct"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "olmo": {
        "data": ROOT / "data" / "exp_olmo1b_c4_rerun" / "features.csv",
        "label": "OLMo-1B",
        "target_1v2_raw": 0.508,
    },
    "pythia": {
        "data": ROOT / "data_exp020_pythia_nuc09" / "features.csv",
        "label": "Pythia-1.4B",
        "target_1v2_raw": 0.654,
    },
    "qwen_cont": {
        "data": ROOT / "data_exp022_cross_scorer" / "qwen_cont256__qwen_base" / "features.csv",
        "label": "Qwen-1.5B (cont)",
        "target_1v2_raw": 0.611,
    },
}

ALL_FEATURES = [
    "surp_mean", "surp_std", "surp_skew", "surp_kurt",
    "surp_d1_mean", "surp_d1_std", "surp_d1_skew",
    "surp_d2_mean", "surp_d2_std",
    "ttr", "hapax_ratio", "self_bleu",
    "freq_kurtosis", "freq_entropy", "low_freq_ratio",
]

DISPLAY_NAMES = {
    "surp_mean": "Surprisal μ", "surp_std": "Surprisal σ",
    "surp_skew": "Surprisal Skew", "surp_kurt": "Surprisal Kurt",
    "surp_d1_mean": "Δ¹ Surp μ", "surp_d1_std": "Δ¹ Surp σ",
    "surp_d1_skew": "Δ¹ Surp Skew",
    "surp_d2_mean": "Δ² Surp μ", "surp_d2_std": "Δ² Surp σ",
    "ttr": "TTR", "hapax_ratio": "Hapax Ratio", "self_bleu": "Self-BLEU",
    "freq_kurtosis": "Freq Kurt", "freq_entropy": "Freq Entropy",
    "low_freq_ratio": "Low-Freq Ratio",
}

FEATURE_GROUPS = {
    "surp_mean": "surprisal_base", "surp_std": "surprisal_base",
    "surp_skew": "surprisal_base", "surp_kurt": "surprisal_base",
    "surp_d1_mean": "surprisal_deriv", "surp_d1_std": "surprisal_deriv",
    "surp_d1_skew": "surprisal_deriv",
    "surp_d2_mean": "surprisal_deriv", "surp_d2_std": "surprisal_deriv",
    "ttr": "lexical_div", "hapax_ratio": "lexical_div",
    "self_bleu": "lexical_div",
    "freq_kurtosis": "freq_dist", "freq_entropy": "freq_dist",
    "low_freq_ratio": "freq_dist",
}

GROUP_COLORS = {
    "surprisal_base": "#2196F3",
    "surprisal_deriv": "#1565C0",
    "lexical_div": "#4CAF50",
    "freq_dist": "#FF9800",
}

LGB_PARAMS = dict(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    num_leaves=31, verbose=-1, n_jobs=4,
)

THETA = 0.60
N_REPEATS = 10
N_FOLDS = 5
SEED = 42


def load_data(data_path):
    with open(data_path) as f:
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
    X = np.column_stack([feat_dict[col][mask] for col in ALL_FEATURES])
    nan_mask = np.isnan(X).any(axis=1)
    if nan_mask.any():
        valid = ~nan_mask
        X, y, ids = X[valid], y[valid], ids[valid]
    return ids, X, y


def compute_permutation_importance(X, y, groups):
    gkf = GroupKFold(n_splits=N_FOLDS)
    all_importances = []
    fold_aucs = []

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(X_train, y_train)

        baseline_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        fold_aucs.append(baseline_auc)

        result = permutation_importance(
            model, X_test, y_test,
            n_repeats=N_REPEATS,
            scoring="roc_auc",
            random_state=SEED,
            n_jobs=4,
        )
        all_importances.append(result.importances_mean)

    mean_imp = np.mean(all_importances, axis=0)
    std_imp = np.std(all_importances, axis=0)
    mean_auc = np.mean(fold_aucs)
    return mean_imp, std_imp, mean_auc


def run_model(model_key, model_cfg):
    print(f"\n{'='*60}")
    print(f"Model: {model_cfg['label']} ({model_key})")
    print(f"Data:  {model_cfg['data']}")
    print(f"{'='*60}")

    model_dir = OUT_DIR / model_key
    model_dir.mkdir(parents=True, exist_ok=True)

    doc_ids, depths, feat_dict = load_data(model_cfg["data"])
    print(f"  Loaded {len(doc_ids)} rows, depths {sorted(set(depths))}")

    results = {
        "model": model_key,
        "label": model_cfg["label"],
        "data_path": str(model_cfg["data"]),
        "n_repeats": N_REPEATS,
        "n_folds": N_FOLDS,
        "theta": THETA,
        "seed": SEED,
        "features": ALL_FEATURES,
    }

    for d_low, d_high in [(0, 1), (1, 2)]:
        pair_name = f"{d_low}v{d_high}"
        ids, X, y = get_pair_data(doc_ids, depths, feat_dict, d_low, d_high)
        print(f"\n  {pair_name}: n={len(y)}, pos_rate={y.mean():.3f}")

        mean_imp, std_imp, raw_auc = compute_permutation_importance(X, y, ids)
        smoothed_auc = THETA * raw_auc + (1 - THETA) * 0.5

        print(f"  raw_AUC={raw_auc:.4f}, smoothed_AUC={smoothed_auc:.4f}")

        if pair_name == "1v2":
            target = model_cfg["target_1v2_raw"]
            print(f"  target_1v2={target:.3f}, delta={abs(raw_auc - target):.4f}")

        ranking = np.argsort(mean_imp)[::-1]
        print(f"  Top-5 features:")
        for rank, i in enumerate(ranking[:5], 1):
            print(f"    {rank}. {ALL_FEATURES[i]:18s} {mean_imp[i]:+.4f} ± {std_imp[i]:.4f}")

        results[pair_name] = {
            "raw_auc": round(raw_auc, 4),
            "smoothed_auc": round(smoothed_auc, 4),
            "importance": {
                ALL_FEATURES[i]: {
                    "mean": round(float(mean_imp[i]), 6),
                    "std": round(float(std_imp[i]), 6),
                    "rank": int(np.where(ranking == i)[0][0]) + 1,
                } for i in range(len(ALL_FEATURES))
            },
            "ranking": [ALL_FEATURES[i] for i in ranking],
        }

    with open(model_dir / "importance_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {model_dir / 'importance_results.json'}")
    return results


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(MODELS.keys())

    all_results = {}
    for key in targets:
        if key not in MODELS:
            print(f"Unknown model: {key}")
            continue
        all_results[key] = run_model(key, MODELS[key])

    # Cross-model summary
    print(f"\n{'='*60}")
    print("CROSS-MODEL SUMMARY (1v2)")
    print(f"{'='*60}")

    header = f"{'Feature':20s}"
    for key in all_results:
        header += f" | {all_results[key]['label']:>20s}"
    print(header)
    print("-" * len(header))

    for feat in ALL_FEATURES:
        row = f"{DISPLAY_NAMES[feat]:20s}"
        for key in all_results:
            info = all_results[key]["1v2"]["importance"][feat]
            row += f" | {info['mean']:+.4f} (#{info['rank']:2d})"
        print(row)

    # Save combined JSON
    with open(OUT_DIR / "all_models_importance.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results: {OUT_DIR / 'all_models_importance.json'}")


if __name__ == "__main__":
    main()
