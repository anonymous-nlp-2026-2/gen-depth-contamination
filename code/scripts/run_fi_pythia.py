"""Wrapper: run feature_importance_analysis for Pythia-1.4B (wiki) data."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"

# Monkey-patch the paths before importing
import importlib.util
spec = importlib.util.spec_from_file_location(
    "fi", "/root/autodl-tmp/gen-depth-contamination/scripts/feature_importance_analysis.py"
)
mod = importlib.util.module_from_spec(spec)

# Override constants
from pathlib import Path
import feature_importance_analysis as fi

fi.DATA_PATH = Path("/root/autodl-tmp/gen-depth-contamination/data/exp_wiki_pythia/features.csv")
fi.OUT_DIR = Path("/root/autodl-tmp/gen-depth-contamination/results/feature_importance_pythia")
fi.ART_DIR = Path("/root/autodl-tmp/gen-depth-contamination/artifacts")
fi.OUT_DIR.mkdir(parents=True, exist_ok=True)
fi.ART_DIR.mkdir(parents=True, exist_ok=True)

# Patch the output file names in main
import json, numpy as np

def main_pythia():
    doc_ids, depths, feat_dict = fi.load_data()
    print(f"Loaded {len(doc_ids)} samples from {fi.DATA_PATH}")
    unique_depths = sorted(set(depths))
    print(f"Depths: {unique_depths}")

    # 0 vs 1
    print("\n=== 0 vs 1 (distinguishable) ===")
    ids_01, X_01, y_01 = fi.get_pair_data(doc_ids, depths, feat_dict, 0, 1)
    print(f"  Samples: {len(y_01)} (class 0: {(y_01==0).sum()}, class 1: {(y_01==1).sum()})")
    mean_01, std_01, auc_01 = fi.compute_permutation_importance(X_01, y_01, ids_01)
    print(f"  Baseline AUC: {auc_01:.4f}")
    print("  Top-5 features:")
    top5_01 = np.argsort(mean_01)[::-1][:5]
    for rank, i in enumerate(top5_01, 1):
        print(f"    {rank}. {fi.ALL_FEATURES[i]:20s} {mean_01[i]:.4f} +/- {std_01[i]:.4f}")

    # 1 vs 2
    print("\n=== 1 vs 2 (K* boundary) ===")
    ids_12, X_12, y_12 = fi.get_pair_data(doc_ids, depths, feat_dict, 1, 2)
    print(f"  Samples: {len(y_12)} (class 0: {(y_12==0).sum()}, class 1: {(y_12==1).sum()})")
    mean_12, std_12, auc_12 = fi.compute_permutation_importance(X_12, y_12, ids_12)
    print(f"  Baseline AUC: {auc_12:.4f}")
    print("  Top-5 features:")
    top5_12 = np.argsort(mean_12)[::-1][:5]
    for rank, i in enumerate(top5_12, 1):
        print(f"    {rank}. {fi.ALL_FEATURES[i]:20s} {mean_12[i]:.4f} +/- {std_12[i]:.4f}")

    # 2 vs 3
    print("\n=== 2 vs 3 (post-boundary) ===")
    ids_23, X_23, y_23 = fi.get_pair_data(doc_ids, depths, feat_dict, 2, 3)
    print(f"  Samples: {len(y_23)} (class 0: {(y_23==0).sum()}, class 1: {(y_23==1).sum()})")
    mean_23, std_23, auc_23 = fi.compute_permutation_importance(X_23, y_23, ids_23)
    print(f"  Baseline AUC: {auc_23:.4f}")
    print("  Top-5 features:")
    top5_23 = np.argsort(mean_23)[::-1][:5]
    for rank, i in enumerate(top5_23, 1):
        print(f"    {rank}. {fi.ALL_FEATURES[i]:20s} {mean_23[i]:.4f} +/- {std_23[i]:.4f}")

    # Save results
    results = {
        "data_path": str(fi.DATA_PATH),
        "model": "pythia-1.4b",
        "domain": "wiki",
        "n_repeats": fi.N_REPEATS,
        "n_folds": fi.N_FOLDS,
        "lgb_params": fi.LGB_PARAMS,
        "features": fi.ALL_FEATURES,
    }
    for label, mean_arr, std_arr, auc_val in [
        ("0v1", mean_01, std_01, auc_01),
        ("1v2", mean_12, std_12, auc_12),
        ("2v3", mean_23, std_23, auc_23),
    ]:
        results[label] = {
            "baseline_auc": round(auc_val, 4),
            "importance": {
                fi.ALL_FEATURES[i]: {"mean": round(float(mean_arr[i]), 6), "std": round(float(std_arr[i]), 6)}
                for i in range(len(fi.ALL_FEATURES))
            },
            "ranking": [fi.ALL_FEATURES[i] for i in np.argsort(mean_arr)[::-1]],
        }

    out_json = fi.OUT_DIR / "importance_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_json}")

    # Barplot
    res_0v1 = {"mean": mean_01, "std": std_01}
    res_1v2 = {"mean": mean_12, "std": std_12}
    pdf_path = fi.ART_DIR / "feature_importance_pythia_1.4b.pdf"
    fi.make_barplot(res_0v1, res_1v2, pdf_path)
    print(f"Saved barplot: {pdf_path}")

    tex_path = fi.ART_DIR / "feature_importance_pythia_1.4b.tex"
    fi.make_latex_table(res_0v1, res_1v2, tex_path)
    print(f"Saved table: {tex_path}")

if __name__ == "__main__":
    main_pythia()
