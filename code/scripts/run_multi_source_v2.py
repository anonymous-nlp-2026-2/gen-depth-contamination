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

SOURCES = {
    "qwen1b5": Path("/root/autodl-tmp/gen-depth-contamination/data_exp016_qwen_base/features.csv"),
    "pythia1b4": Path("/root/autodl-tmp/gen-depth-contamination/data_exp020_pythia_greedy/features.csv"),
    "llama8b": Path("/root/autodl-tmp/gen-depth-contamination/data/exp_018_llama8b/features.csv"),
}

DOCS_PER_GEN = 1000
MAX_DEPTH = 5
N_BOOT = 10000
THETA = 0.60
SEED = 42
OUT_DIR = Path("/root/autodl-tmp/gen-depth-contamination/results/exp_multi_source_detection_v2")


def load_features(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    doc_ids = np.array([int(r["doc_id"]) for r in rows])
    depths = np.array([int(r["depth"]) for r in rows])
    X = np.array(
        [[float(r[c]) if r[c] not in ("", "nan") else np.nan for c in FEAT_COLS] for r in rows]
    )
    col_means = np.nanmean(X, axis=0)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        X[mask, j] = col_means[j]
    return X, depths, doc_ids


def sample_and_merge(rng):
    all_X, all_depths, all_doc_ids, all_gen_ids = [], [], [], []
    doc_offset = 0

    for gen_idx, (gen_name, path) in enumerate(SOURCES.items()):
        X, depths, doc_ids = load_features(path)
        unique_docs = np.unique(doc_ids)
        sampled_docs = rng.choice(unique_docs, size=DOCS_PER_GEN, replace=False)
        sampled_set = set(sampled_docs)
        mask = np.array([d in sampled_set for d in doc_ids])

        X_s = X[mask]
        depths_s = depths[mask]
        doc_ids_s = doc_ids[mask] + doc_offset
        gen_ids_s = np.full(mask.sum(), gen_idx)

        all_X.append(X_s)
        all_depths.append(depths_s)
        all_doc_ids.append(doc_ids_s)
        all_gen_ids.append(gen_ids_s)

        doc_offset += unique_docs.max() + 1
        print(f"  {gen_name}: sampled {DOCS_PER_GEN} docs -> {mask.sum()} rows")

    X = np.vstack(all_X)
    depths = np.concatenate(all_depths)
    doc_ids = np.concatenate(all_doc_ids)
    gen_ids = np.concatenate(all_gen_ids)
    return X, depths, doc_ids, gen_ids


def get_oof_predictions(X, y, groups):
    gkf = GroupKFold(n_splits=5)
    oof_proba = np.zeros(len(y))
    fold_aucs = []
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        clf = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            num_leaves=31, verbose=-1, n_jobs=-1,
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
    return point_auc, ci_lower, ci_upper, boot_aucs


def run_pairwise(X, depths, doc_ids, label, gen_ids=None):
    results = {}
    for k in range(1, MAX_DEPTH + 1):
        pair = f"{k-1}v{k}"
        t0 = time.time()
        print(f"\n  [{label}] {pair}...", flush=True)

        mask = (depths == k - 1) | (depths == k)
        X_pair = X[mask]
        y_pair = (depths[mask] == k).astype(int)
        groups_pair = doc_ids[mask]

        if gen_ids is not None:
            X_pair = np.column_stack([X_pair, gen_ids[mask]])

        n0, n1 = (y_pair == 0).sum(), (y_pair == 1).sum()
        print(f"    samples: {len(y_pair)} (d{k-1}={n0}, d{k}={n1})", flush=True)

        oof_proba, fold_aucs = get_oof_predictions(X_pair, y_pair, groups_pair)
        rng_boot = np.random.RandomState(SEED + k)
        point_auc, ci_lower, ci_upper, boot_aucs = bootstrap_ci(
            y_pair, oof_proba, groups_pair, N_BOOT, rng_boot
        )

        elapsed = time.time() - t0
        print(f"    AUC={point_auc:.4f}  CI=[{ci_lower:.4f}, {ci_upper:.4f}]  ({elapsed:.1f}s)", flush=True)

        results[pair] = {
            "auc": round(point_auc, 4),
            "fold_aucs": [round(a, 4) for a in fold_aucs],
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "boot_mean": round(float(np.mean(boot_aucs)), 4),
            "boot_std": round(float(np.std(boot_aucs)), 4),
        }
    return results


def compute_kstar(results, theta):
    kstar = 0
    for k in range(1, MAX_DEPTH + 1):
        pair = f"{k-1}v{k}"
        if results[pair]["ci_lower"] > theta:
            kstar = k
        else:
            break
    return kstar


def main():
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(SEED)

    print("=== Sampling and merging data ===", flush=True)
    X, depths, doc_ids, gen_ids = sample_and_merge(rng)
    print(f"Total: {len(depths)} rows, {len(np.unique(doc_ids))} unique docs", flush=True)

    for d in range(MAX_DEPTH + 1):
        n = (depths == d).sum()
        print(f"  depth {d}: {n} rows", flush=True)

    # --- Main experiment: mixed-source, no generator label ---
    print("\n=== Experiment 1: Mixed-source (no generator label) ===", flush=True)
    results_no_label = run_pairwise(X, depths, doc_ids, "no-gen-label")
    kstar_no_label = compute_kstar(results_no_label, THETA)
    print(f"\n  K*(mixed, no label, theta={THETA}) = {kstar_no_label}", flush=True)

    # --- Experiment 2: with generator label as 16th feature ---
    print("\n=== Experiment 2: Mixed-source (with generator label) ===", flush=True)
    results_with_label = run_pairwise(X, depths, doc_ids, "with-gen-label", gen_ids=gen_ids)
    kstar_with_label = compute_kstar(results_with_label, THETA)
    print(f"\n  K*(mixed, with label, theta={THETA}) = {kstar_with_label}", flush=True)

    total_time = time.time() - t_start

    output = {
        "experiment": "exp_multi_source_detection_v2",
        "description": "Multi-source contamination detection v2: Qwen-1.5B + Pythia-1.4B + LLaMA-3.1-8B",
        "generators": list(SOURCES.keys()),
        "docs_per_generator": DOCS_PER_GEN,
        "total_docs": int(len(np.unique(doc_ids))),
        "n_bootstrap": N_BOOT,
        "theta": THETA,
        "seed": SEED,
        "total_seconds": round(total_time, 1),
        "no_generator_label": {
            "pairs": results_no_label,
            "kstar": kstar_no_label,
        },
        "with_generator_label": {
            "pairs": results_with_label,
            "kstar": kstar_with_label,
        },
        "single_source_kstar_reference": {
            "qwen1b5": 2,
            "pythia1b4": 2,
            "llama8b": 2,
        },
        "v1_comparison": {
            "v1_generators": ["qwen1b5", "pythia1b4", "olmo1b"],
            "v1_kstar_no_label": 2,
            "v1_kstar_with_label": 2,
            "v1_1v2_auc_no_label": 0.7007,
            "v1_1v2_auc_with_label": 0.7015,
        },
    }

    out_path = OUT_DIR / "multi_source_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Generators: Qwen-1.5B + Pythia-1.4B + LLaMA-3.1-8B", flush=True)
    print(f"Single-source K* (reference): Qwen=2, Pythia=2, LLaMA-8B=2", flush=True)
    print(f"Mixed-source K* (no label):   {kstar_no_label}", flush=True)
    print(f"Mixed-source K* (with label): {kstar_with_label}", flush=True)
    print(f"\nv1 (OLMo) K*: no_label=2, with_label=2", flush=True)
    print(f"\nPairwise AUC comparison:", flush=True)
    print(f"{'Pair':<8} {'NoLabel AUC':<14} {'NoLabel CI':<22} {'WithLabel AUC':<16} {'WithLabel CI':<22}", flush=True)
    for k in range(1, MAX_DEPTH + 1):
        pair = f"{k-1}v{k}"
        r1 = results_no_label[pair]
        r2 = results_with_label[pair]
        print(f"{pair:<8} {r1['auc']:<14.4f} [{r1['ci_lower']:.4f}, {r1['ci_upper']:.4f}]   {r2['auc']:<16.4f} [{r2['ci_lower']:.4f}, {r2['ci_upper']:.4f}]", flush=True)

    print(f"\nTotal time: {total_time:.1f}s", flush=True)


if __name__ == "__main__":
    main()
