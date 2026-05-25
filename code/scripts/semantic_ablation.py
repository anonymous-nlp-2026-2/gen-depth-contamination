"""
Semantic Feature Ablation: compute semantic features (TF-IDF cosine sim + entity diversity),
merge with existing 15D features, retrain LightGBM with 15D vs 17D, compare K* values.

Uses TF-IDF sentence embeddings (no external model download needed) to compute
inter-sentence semantic coherence.
"""
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

os.environ["CUDA_VISIBLE_DEVICES"] = ""

DATA_DIR = Path("/root/autodl-tmp/gen-depth-contamination/data/exp_018_llama8b")
ARTIFACTS_DIR = Path("/root/autodl-tmp/gen-depth-contamination/artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MAX_DEPTH = 5

FEAT_15D = [
    "surp_mean", "surp_std", "surp_skew", "surp_kurt",
    "surp_d1_mean", "surp_d1_std", "surp_d1_skew",
    "surp_d2_mean", "surp_d2_std",
    "ttr", "hapax_ratio", "self_bleu",
    "freq_kurtosis", "freq_entropy", "low_freq_ratio",
]
SEMANTIC_COLS = ["semantic_cosine_sim", "entity_diversity"]
FEAT_17D = FEAT_15D + SEMANTIC_COLS

SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def simple_sent_tokenize(text):
    sents = SENT_SPLIT_RE.split(text.strip())
    return [s for s in sents if len(s.strip()) > 5]


def load_texts(data_dir, max_depth):
    texts_by_depth = {}
    for d in range(max_depth + 1):
        path = data_dir / f"depth_{d}.jsonl"
        texts = []
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                texts.append((obj["doc_id"], obj["text"]))
        texts_by_depth[d] = texts
        print(f"  depth {d}: {len(texts)} docs", flush=True)
    return texts_by_depth


def compute_semantic_cosine_sim(texts_by_depth):
    """Compute average cosine similarity between adjacent sentences using TF-IDF."""
    print("  Building TF-IDF vocabulary from all documents...", flush=True)
    all_sents = []
    for depth, doc_list in texts_by_depth.items():
        for doc_id, text in doc_list:
            sents = simple_sent_tokenize(text)
            all_sents.extend(sents)

    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', sublinear_tf=True)
    vectorizer.fit(all_sents)
    print(f"  TF-IDF vocab size: {len(vectorizer.vocabulary_)}", flush=True)

    results = {}
    for depth, doc_list in texts_by_depth.items():
        print(f"  Computing cosine sim for depth {depth} ({len(doc_list)} docs)...", flush=True)
        depth_scores = []
        for doc_id, text in doc_list:
            sents = simple_sent_tokenize(text)
            if len(sents) < 2:
                depth_scores.append((doc_id, 1.0))
                continue
            tfidf_matrix = vectorizer.transform(sents)
            cosines = []
            for i in range(tfidf_matrix.shape[0] - 1):
                cos = sklearn_cosine(tfidf_matrix[i:i+1], tfidf_matrix[i+1:i+2])[0, 0]
                cosines.append(float(cos))
            depth_scores.append((doc_id, float(np.mean(cosines))))
        results[depth] = depth_scores
    return results


def compute_entity_diversity(texts_by_depth):
    """Compute entity diversity using regex NER (capitalized multi-word phrases)."""
    cap_pattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
    print("  Using regex NER (capitalized multi-word phrases)", flush=True)

    results = {}
    for depth, doc_list in texts_by_depth.items():
        print(f"  Computing entity diversity for depth {depth}...", flush=True)
        depth_scores = []
        for doc_id, text in doc_list:
            ents = cap_pattern.findall(text)
            if len(ents) == 0:
                depth_scores.append((doc_id, 1.0))
            else:
                ents_lower = [e.lower() for e in ents]
                diversity = len(set(ents_lower)) / len(ents_lower)
                depth_scores.append((doc_id, diversity))
        results[depth] = depth_scores
    return results


def load_features_csv(data_dir):
    feat_path = data_dir / "features.csv"
    rows = []
    with open(feat_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def merge_features(existing_rows, cosine_results, entity_results):
    cosine_lookup = {}
    entity_lookup = {}
    for depth, scores in cosine_results.items():
        for doc_id, val in scores:
            cosine_lookup[(depth, doc_id)] = val
    for depth, scores in entity_results.items():
        for doc_id, val in scores:
            entity_lookup[(depth, doc_id)] = val

    for row in existing_rows:
        key = (int(row["depth"]), int(row["doc_id"]))
        row["semantic_cosine_sim"] = cosine_lookup.get(key, np.nan)
        row["entity_diversity"] = entity_lookup.get(key, np.nan)
    return existing_rows


def build_arrays(rows, feat_cols):
    depths = np.array([int(r["depth"]) for r in rows])
    doc_ids = np.array([int(r["doc_id"]) for r in rows])
    X = np.array([
        [float(r[c]) if r[c] != "" and str(r[c]) != "nan" else np.nan for c in feat_cols]
        for r in rows
    ])
    col_means = np.nanmean(X, axis=0)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        X[mask, j] = col_means[j]
    return X, depths, doc_ids


def run_pairwise_auc(X, depths, doc_ids, max_depth):
    aucs = {}
    for k in range(1, max_depth + 1):
        mask = (depths == k - 1) | (depths == k)
        Xm = X[mask]
        ym = (depths[mask] == k).astype(int)
        if len(np.unique(ym)) < 2:
            aucs[f"{k-1}v{k}"] = float("nan")
            continue
        gkf = GroupKFold(n_splits=5)
        groups = doc_ids[mask]
        fold_aucs = []
        for train_idx, test_idx in gkf.split(Xm, ym, groups=groups):
            clf = lgb.LGBMClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                num_leaves=31, verbose=-1, n_jobs=-1,
            )
            clf.fit(Xm[train_idx], ym[train_idx])
            proba = clf.predict_proba(Xm[test_idx])[:, 1]
            fold_aucs.append(roc_auc_score(ym[test_idx], proba))
        aucs[f"{k-1}v{k}"] = round(float(np.mean(fold_aucs)), 4)
    return aucs


def compute_k_star(aucs, max_depth):
    k_star = 0
    for k in range(1, max_depth + 1):
        key = f"{k-1}v{k}"
        if aucs.get(key, 0) >= 0.60:
            k_star = k
    return k_star


def get_feature_importance(X, depths, doc_ids, feat_cols, max_depth):
    importances = np.zeros(len(feat_cols))
    count = 0
    for k in range(1, max_depth + 1):
        mask = (depths == k - 1) | (depths == k)
        Xm = X[mask]
        ym = (depths[mask] == k).astype(int)
        if len(np.unique(ym)) < 2:
            continue
        clf = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            num_leaves=31, verbose=-1, n_jobs=-1,
        )
        clf.fit(Xm, ym)
        importances += clf.feature_importances_
        count += 1
    if count > 0:
        importances /= count
    ranked = np.argsort(-importances)
    rank_map = {}
    for rank, idx in enumerate(ranked, 1):
        rank_map[feat_cols[idx]] = rank
    return rank_map, {feat_cols[i]: float(importances[i]) for i in range(len(feat_cols))}


def main():
    print("=" * 60, flush=True)
    print("Semantic Feature Ablation Experiment", flush=True)
    print("=" * 60, flush=True)

    print("\n[1/5] Loading texts...", flush=True)
    texts_by_depth = load_texts(DATA_DIR, MAX_DEPTH)

    print("\n[2/5] Computing semantic cosine similarity (TF-IDF)...", flush=True)
    cosine_results = compute_semantic_cosine_sim(texts_by_depth)

    print("\n[3/5] Computing entity diversity...", flush=True)
    entity_results = compute_entity_diversity(texts_by_depth)

    print("\n[4/5] Loading and merging features...", flush=True)
    rows = load_features_csv(DATA_DIR)
    rows = merge_features(rows, cosine_results, entity_results)

    X_15d, depths, doc_ids = build_arrays(rows, FEAT_15D)
    X_17d, _, _ = build_arrays(rows, FEAT_17D)
    print(f"  15D shape: {X_15d.shape}, 17D shape: {X_17d.shape}", flush=True)

    print("\n[5/5] Training LightGBM classifiers...", flush=True)
    print("  --- 15D ---", flush=True)
    aucs_15d = run_pairwise_auc(X_15d, depths, doc_ids, MAX_DEPTH)
    for pair, auc in aucs_15d.items():
        print(f"    {pair}: AUC = {auc}", flush=True)
    k_star_15d = compute_k_star(aucs_15d, MAX_DEPTH)
    print(f"  K* (15D) = {k_star_15d}", flush=True)

    print("  --- 17D ---", flush=True)
    aucs_17d = run_pairwise_auc(X_17d, depths, doc_ids, MAX_DEPTH)
    for pair, auc in aucs_17d.items():
        print(f"    {pair}: AUC = {auc}", flush=True)
    k_star_17d = compute_k_star(aucs_17d, MAX_DEPTH)
    print(f"  K* (17D) = {k_star_17d}", flush=True)

    print("\n  Feature importance (17D):", flush=True)
    rank_map, raw_imp = get_feature_importance(X_17d, depths, doc_ids, FEAT_17D, MAX_DEPTH)
    for feat in SEMANTIC_COLS:
        print(f"    {feat}: rank {rank_map[feat]}/{len(FEAT_17D)}, importance={raw_imp[feat]:.1f}", flush=True)

    auc_diff = {}
    for pair in aucs_15d:
        auc_diff[pair] = round(aucs_17d.get(pair, 0) - aucs_15d.get(pair, 0), 4)

    if k_star_15d == k_star_17d:
        conclusion = f"K* unchanged at {k_star_15d}. Semantic features do not shift the discrimination boundary."
    else:
        conclusion = f"K* changed from {k_star_15d} (15D) to {k_star_17d} (17D). Semantic features shift the discrimination boundary."

    sem_ranks = {feat: rank_map[feat] for feat in SEMANTIC_COLS}
    avg_rank = np.mean([sem_ranks[f] for f in SEMANTIC_COLS])
    if avg_rank > len(FEAT_17D) * 0.5:
        conclusion += " Semantic features rank low in importance, surface features dominate."
    else:
        conclusion += " Semantic features rank high, contributing meaningful discriminative signal."

    report = {
        "experiment": "semantic_feature_ablation",
        "data": "exp_018_llama8b (C4, 5000 docs/depth, depths 0-5)",
        "semantic_method": "TF-IDF cosine similarity (adjacent sentences) + regex entity diversity",
        "15d_aucs": aucs_15d,
        "17d_aucs": aucs_17d,
        "auc_diff": auc_diff,
        "k_star_15d": k_star_15d,
        "k_star_17d": k_star_17d,
        "feature_importance_rank": sem_ranks,
        "feature_importance_raw": {f: round(raw_imp[f], 2) for f in SEMANTIC_COLS},
        "all_feature_ranks": rank_map,
        "conclusion": conclusion,
    }

    out_path = ARTIFACTS_DIR / "semantic_ablation_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {out_path}", flush=True)
    print("\n" + json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
