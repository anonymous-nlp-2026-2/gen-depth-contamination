import csv
import json
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

FEAT_COLS = ["ttr", "hapax_ratio", "self_bleu", "freq_kurtosis", "freq_entropy", "low_freq_ratio"]

SOURCES_V1 = {
    "qwen1b5": Path("/root/autodl-tmp/gen-depth-contamination/data_exp016_qwen_base/features.csv"),
    "pythia1b4": Path("/root/autodl-tmp/gen-depth-contamination/data_exp020_pythia_greedy/features.csv"),
    "olmo1b": Path("/root/autodl-tmp/gen-depth-contamination/data/olmo_c4_cont/features.csv"),
}

SOURCES_V2 = {
    "qwen1b5": Path("/root/autodl-tmp/gen-depth-contamination/data_exp016_qwen_base/features.csv"),
    "pythia1b4": Path("/root/autodl-tmp/gen-depth-contamination/data_exp020_pythia_greedy/features.csv"),
    "llama8b": Path("/root/autodl-tmp/gen-depth-contamination/data/exp_018_llama8b/features.csv"),
}

DOCS_PER_GEN = 1000
MAX_DEPTH = 5
N_BOOT = 10000
THETA = 0.60
SEED = 42
OUT_DIR = Path("/root/autodl-tmp/gen-depth-contamination/results/exp_multi_source_detection")


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


def sample_and_merge(sources, rng):
    all_X, all_depths, all_doc_ids, all_gen_ids = [], [], [], []
    doc_offset = 0

    for gen_idx, (gen_name, path) in enumerate(sources.items()):
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


def run_pairwise(X, depths, doc_ids, label):
    results = {}
    for k in range(1, MAX_DEPTH + 1):
        pair = f"{k-1}v{k}"
        t0 = time.time()
        print(f"\n  [{label}] Pair {pair}:", end="", flush=True)
        mask = (depths == k - 1) | (depths == k)
        X_pair = X[mask]
        y_pair = (depths[mask] == k).astype(int)
        doc_pair = doc_ids[mask]

        oof_proba, fold_aucs = get_oof_predictions(X_pair, y_pair, doc_pair)
        rng_boot = np.random.RandomState(SEED + k)
        point_auc, ci_lower, ci_upper, boot_aucs = bootstrap_ci(
            y_pair, oof_proba, doc_pair, N_BOOT, rng_boot
        )

        elapsed = time.time() - t0
        print(f" AUC={point_auc:.4f}  CI=[{ci_lower:.4f}, {ci_upper:.4f}]  ({elapsed:.1f}s)", flush=True)

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


def run_version(version_name, sources):
    print(f"\n{'='*60}")
    print(f"  {version_name}")
    print(f"{'='*60}")
    rng = np.random.RandomState(SEED)

    print("Sampling and merging data...", flush=True)
    X, depths, doc_ids, gen_ids = sample_and_merge(sources, rng)
    print(f"Total: {len(depths)} rows, {len(np.unique(doc_ids))} unique docs", flush=True)

    results = run_pairwise(X, depths, doc_ids, version_name)
    kstar = compute_kstar(results, THETA)
    print(f"\n  K*(theta={THETA}) = {kstar}", flush=True)
    return results, kstar


def main():
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Features: {FEAT_COLS}")
    print(f"Feature count: {len(FEAT_COLS)} (vocab-only, scorer-independent)")

    results_v1, kstar_v1 = run_version("v1 (Qwen+Pythia+OLMo)", SOURCES_V1)
    results_v2, kstar_v2 = run_version("v2 (Qwen+Pythia+LLaMA-8B)", SOURCES_V2)

    total_time = time.time() - t_start

    output = {
        "experiment": "exp_multi_source_vocab_only",
        "description": "Vocab-only (scorer-independent) multi-source detection",
        "features": FEAT_COLS,
        "n_features": len(FEAT_COLS),
        "n_bootstrap": N_BOOT,
        "theta": THETA,
        "seed": SEED,
        "total_seconds": round(total_time, 1),
        "v1_qwen_pythia_olmo": {
            "generators": list(SOURCES_V1.keys()),
            "pairs": results_v1,
            "kstar": kstar_v1,
        },
        "v2_qwen_pythia_llama8b": {
            "generators": list(SOURCES_V2.keys()),
            "pairs": results_v2,
            "kstar": kstar_v2,
        },
    }

    out_path = OUT_DIR / "multi_source_vocab_only_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print("\n" + "=" * 60)
    print("=== Vocab-Only Multi-Source Results ===")
    print("=" * 60)
    print(f"\nv1 (Qwen+Pythia+OLMo):")
    for k in range(1, 4):
        pair = f"{k-1}v{k}"
        r = results_v1[pair]
        print(f"  {pair} AUC: {r['auc']:.4f}  CI=[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]")
    print(f"  K* (theta=0.60): {kstar_v1}")

    print(f"\nv2 (Qwen+Pythia+LLaMA-8B):")
    for k in range(1, 4):
        pair = f"{k-1}v{k}"
        r = results_v2[pair]
        print(f"  {pair} AUC: {r['auc']:.4f}  CI=[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]")
    print(f"  K* (theta=0.60): {kstar_v2}")

    print(f"\nFull-feature comparison:")
    print(f"  v1 full K*: 2 vs vocab-only K*: {kstar_v1}")
    print(f"  v2 full K*: (run full-feature v2 for comparison) vs vocab-only K*: {kstar_v2}")

    print(f"\nTotal time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
