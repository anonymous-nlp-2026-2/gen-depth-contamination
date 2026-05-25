"""
arXiv Multi-Source Detection: Vocab-Only + Unified Qwen2.5-1.5B Scorer.
Parallels C4 multi-source experiments but on arXiv domain.
"""
import csv
import json
import time
import logging
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

SCORER_PATH = "/root/autodl-tmp/.hf_cache/Qwen/Qwen2___5-1___5B"
SOURCES = {
    "qwen1b5": {
        "data_dir": Path("/root/autodl-tmp/gen-depth-contamination/data/exp_024_arxiv"),
        "features_csv": Path("/root/autodl-tmp/gen-depth-contamination/data/exp_024_arxiv/features.csv"),
        "needs_rescore": False,
    },
    "pythia1b4": {
        "data_dir": Path("/root/autodl-tmp/gen-depth-contamination/data/exp_025_pythia_arxiv"),
        "features_csv": Path("/root/autodl-tmp/gen-depth-contamination/data/exp_025_pythia_arxiv/features.csv"),
        "needs_rescore": True,
    },
    "llama8b": {
        "data_dir": Path("/root/autodl-tmp/gen-depth-contamination/data/exp_arXiv_llama8b"),
        "features_csv": Path("/root/autodl-tmp/gen-depth-contamination/data/exp_arXiv_llama8b/features.csv"),
        "needs_rescore": True,
    },
}

SURPRISAL_COLS = ["surp_mean","surp_std","surp_skew","surp_kurt","surp_d1_mean","surp_d1_std","surp_d1_skew","surp_d2_mean","surp_d2_std"]
TEXT_COLS = ["ttr","hapax_ratio","self_bleu","freq_kurtosis","freq_entropy","low_freq_ratio"]
ALL_FEAT_COLS = SURPRISAL_COLS + TEXT_COLS

DOCS_PER_GEN = 1000
MAX_DEPTH = 5
N_BOOT = 10000
THETA = 0.60
SEED = 42
OUT_DIR = Path("/root/autodl-tmp/gen-depth-contamination/results/exp_multi_source_arxiv")


def load_features_csv(path, feat_cols):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    doc_ids_raw = [r["doc_id"] for r in rows]
    try:
        doc_ids = np.array([int(d) for d in doc_ids_raw])
    except (ValueError, TypeError):
        uid_map = {d: i for i, d in enumerate(sorted(set(doc_ids_raw)))}
        doc_ids = np.array([uid_map[d] for d in doc_ids_raw])
    depths = np.array([int(r["depth"]) for r in rows])
    X = np.array(
        [[float(r[c]) if r[c] not in ("", "nan") else np.nan for c in feat_cols] for r in rows]
    )
    col_means = np.nanmean(X, axis=0)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if mask.any():
            X[mask, j] = col_means[j]
    return X, depths, doc_ids


def load_text_features(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    result = {}
    for r in rows:
        key = (str(r["doc_id"]), int(r["depth"]))
        result[key] = {col: r[col] for col in TEXT_COLS}
    return result


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def sample_and_merge(all_data, feat_cols, rng):
    all_X, all_depths, all_doc_ids, all_gen_ids = [], [], [], []
    doc_offset = 0
    for gen_idx, (gen_name, data) in enumerate(all_data.items()):
        X, depths, doc_ids = data["X"], data["depths"], data["doc_ids"]
        unique_docs = np.unique(doc_ids)
        n_sample = min(DOCS_PER_GEN, len(unique_docs))
        sampled_docs = rng.choice(unique_docs, size=n_sample, replace=False)
        sampled_set = set(sampled_docs.tolist())
        mask = np.array([d in sampled_set for d in doc_ids])

        all_X.append(X[mask])
        all_depths.append(depths[mask])
        all_doc_ids.append(doc_ids[mask] + doc_offset)
        all_gen_ids.append(np.full(mask.sum(), gen_idx))
        doc_offset += int(unique_docs.max()) + 1
        print(f"  {gen_name}: sampled {n_sample} docs -> {mask.sum()} rows", flush=True)

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
        mask = (depths == k - 1) | (depths == k)
        X_pair = X[mask]
        y_pair = (depths[mask] == k).astype(int)
        doc_pair = doc_ids[mask]
        if gen_ids is not None:
            X_pair = np.hstack([X_pair, gen_ids[mask].reshape(-1, 1)])
        rng = np.random.RandomState(SEED + k)
        oof_proba, fold_aucs = get_oof_predictions(X_pair, y_pair, doc_pair)
        point_auc, ci_lower, ci_upper, boot_aucs = bootstrap_ci(
            y_pair, oof_proba, doc_pair, N_BOOT, rng
        )
        elapsed = time.time() - t0
        print(f"  {label} {pair}: AUC={point_auc:.4f}  CI=[{ci_lower:.4f}, {ci_upper:.4f}]  ({elapsed:.1f}s)", flush=True)
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


def run_vocab_only():
    print("\n" + "=" * 60, flush=True)
    print("  PATH A: arXiv Vocab-Only (6 scorer-independent features)", flush=True)
    print("=" * 60, flush=True)
    print(f"Features: {TEXT_COLS}", flush=True)

    all_data = {}
    for gen_name, src in SOURCES.items():
        X, depths, doc_ids = load_features_csv(src["features_csv"], TEXT_COLS)
        all_data[gen_name] = {"X": X, "depths": depths, "doc_ids": doc_ids}
        print(f"  Loaded {gen_name}: {len(depths)} rows, {len(np.unique(doc_ids))} docs", flush=True)

    rng = np.random.RandomState(SEED)
    print("\nSampling and merging...", flush=True)
    X, depths, doc_ids, gen_ids = sample_and_merge(all_data, TEXT_COLS, rng)
    print(f"Total: {len(depths)} rows, {len(np.unique(doc_ids))} unique docs\n", flush=True)

    print("--- no_label (vocab-only) ---", flush=True)
    results_no_label = run_pairwise(X, depths, doc_ids, "vocab-no-label")
    kstar_no_label = compute_kstar(results_no_label, THETA)
    print(f"  K*(vocab-only, no label, theta={THETA}) = {kstar_no_label}\n", flush=True)

    print("--- with_label (vocab-only + generator identity) ---", flush=True)
    results_with_label = run_pairwise(X, depths, doc_ids, "vocab-with-label", gen_ids=gen_ids)
    kstar_with_label = compute_kstar(results_with_label, THETA)
    print(f"  K*(vocab-only, with label, theta={THETA}) = {kstar_with_label}", flush=True)

    return {
        "no_label": {"pairs": results_no_label, "kstar": kstar_no_label},
        "with_label": {"pairs": results_with_label, "kstar": kstar_with_label},
    }


def compute_surprisal_features(model, tokenizer, texts, batch_size=16):
    from scipy import stats as sp_stats
    from tqdm import tqdm
    import torch

    all_features = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Surprisal", leave=False):
        batch_texts = texts[start:start + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits
        log_probs = torch.log_softmax(logits, dim=-1)
        input_ids = inputs["input_ids"]
        for i in range(len(batch_texts)):
            attn_mask = inputs["attention_mask"][i].bool()
            seq_len = attn_mask.sum().item()
            if seq_len < 3:
                all_features.append([0.0] * 9)
                continue
            offset = input_ids.shape[1] - seq_len
            token_ids = input_ids[i, offset:]
            token_log_probs = []
            for t in range(1, seq_len):
                lp = log_probs[i, offset + t - 1, token_ids[t]].item()
                token_log_probs.append(-lp)
            arr = np.array(token_log_probs)
            if len(arr) < 3:
                all_features.append([0.0] * 9)
                continue
            d1 = np.diff(arr)
            d2 = np.diff(d1) if len(d1) > 1 else np.array([0.0])
            feats = [
                float(np.mean(arr)), float(np.std(arr)),
                float(sp_stats.skew(arr)), float(sp_stats.kurtosis(arr)),
                float(np.mean(d1)), float(np.std(d1)),
                float(sp_stats.skew(d1)) if len(d1) >= 3 else 0.0,
                float(np.mean(d2)), float(np.std(d2)),
            ]
            all_features.append(feats)
        torch.cuda.empty_cache()
    return all_features


def run_unified_scorer():
    import torch

    print("\n" + "=" * 60, flush=True)
    print("  PATH B: arXiv Unified Qwen2.5-1.5B Scorer (15D OBD features)", flush=True)
    print("=" * 60, flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    log.info(f"Loading scorer: {SCORER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(SCORER_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        SCORER_PATH, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    log.info(f"Scorer loaded on {model.device}")

    all_gen_features = {}
    for gen_name, src in SOURCES.items():
        data_dir = src["data_dir"]
        csv_path = src["features_csv"]
        needs_rescore = src["needs_rescore"]

        log.info(f"\n{'='*40}")
        log.info(f"Processing {gen_name} (needs_rescore={needs_rescore})")

        text_feats = load_text_features(csv_path)
        log.info(f"  Loaded {len(text_feats)} text-based feature rows")

        if needs_rescore:
            rows = []
            for d in range(MAX_DEPTH + 1):
                records = load_jsonl(data_dir / f"depth_{d}.jsonl")
                texts = [r["text"] for r in records]
                doc_ids_list = [r["doc_id"] for r in records]
                log.info(f"  depth {d}: computing surprisal for {len(texts)} texts")
                surp_feats = compute_surprisal_features(model, tokenizer, texts, batch_size=16)
                for i in range(len(texts)):
                    key = (str(doc_ids_list[i]), d)
                    tf = text_feats.get(key, {col: "0.0" for col in TEXT_COLS})
                    row = {"doc_id": doc_ids_list[i], "depth": d}
                    for j, col in enumerate(SURPRISAL_COLS):
                        row[col] = surp_feats[i][j]
                    for col in TEXT_COLS:
                        row[col] = float(tf[col]) if tf[col] not in ("", "nan") else 0.0
                    rows.append(row)

            feat_path = OUT_DIR / f"features_{gen_name}_qwen_scorer.csv"
            with open(feat_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["doc_id", "depth"] + ALL_FEAT_COLS)
                writer.writeheader()
                writer.writerows(rows)
            log.info(f"  Saved {len(rows)} rows to {feat_path}")
        else:
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            for r in rows:
                for col in ALL_FEAT_COLS:
                    r[col] = float(r[col]) if r[col] not in ("", "nan") else 0.0
            log.info(f"  Reusing existing features ({gen_name} is self-scorer)")

        all_gen_features[gen_name] = rows

    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()
    log.info("Scorer unloaded")

    rng = np.random.RandomState(SEED)
    all_X, all_depths, all_doc_ids, all_gen_ids = [], [], [], []
    doc_offset = 0

    for gen_idx, (gen_name, feat_rows) in enumerate(all_gen_features.items()):
        doc_ids_raw = [r["doc_id"] for r in feat_rows]
        try:
            doc_ids = np.array([int(d) for d in doc_ids_raw])
        except (ValueError, TypeError):
            uid_map = {d: i for i, d in enumerate(sorted(set(doc_ids_raw)))}
            doc_ids = np.array([uid_map[d] for d in doc_ids_raw])
        depths = np.array([int(r["depth"]) for r in feat_rows])
        X = np.array([[float(r[c]) for c in ALL_FEAT_COLS] for r in feat_rows])
        col_means = np.nanmean(X, axis=0)
        for j in range(X.shape[1]):
            m = np.isnan(X[:, j])
            if m.any():
                X[m, j] = col_means[j]

        unique_docs = np.unique(doc_ids)
        n_sample = min(DOCS_PER_GEN, len(unique_docs))
        sampled_docs = rng.choice(unique_docs, size=n_sample, replace=False)
        sampled_set = set(sampled_docs.tolist())
        mask = np.array([d in sampled_set for d in doc_ids])

        all_X.append(X[mask])
        all_depths.append(depths[mask])
        all_doc_ids.append(doc_ids[mask] + doc_offset)
        all_gen_ids.append(np.full(mask.sum(), gen_idx))
        doc_offset += int(unique_docs.max()) + 1
        print(f"  {gen_name}: sampled {n_sample} docs -> {mask.sum()} rows", flush=True)

    X = np.vstack(all_X)
    depths = np.concatenate(all_depths)
    doc_ids = np.concatenate(all_doc_ids)
    gen_ids = np.concatenate(all_gen_ids)

    print("\n=== Unified Scorer: no_label ===", flush=True)
    results_no_label = run_pairwise(X, depths, doc_ids, "unified-no-label")
    kstar_no_label = compute_kstar(results_no_label, THETA)
    print(f"  K*(unified, no label, theta={THETA}) = {kstar_no_label}\n", flush=True)

    print("=== Unified Scorer: with_label ===", flush=True)
    results_with_label = run_pairwise(X, depths, doc_ids, "unified-with-label", gen_ids=gen_ids)
    kstar_with_label = compute_kstar(results_with_label, THETA)
    print(f"  K*(unified, with label, theta={THETA}) = {kstar_with_label}", flush=True)

    return {
        "no_label": {"pairs": results_no_label, "kstar": kstar_no_label},
        "with_label": {"pairs": results_with_label, "kstar": kstar_with_label},
    }


def print_summary(vocab_results, unified_results):
    print("\n" + "=" * 60, flush=True)
    print("=== arXiv Multi-Source Results ===", flush=True)
    print("=" * 60, flush=True)

    print("\n--- Vocab-Only (6 scorer-independent features) ---", flush=True)
    for cond_name, cond_data in vocab_results.items():
        print(f"  [{cond_name}]", flush=True)
        for k in range(1, 4):
            pair = f"{k-1}v{k}"
            r = cond_data["pairs"][pair]
            print(f"    {pair} AUC: {r['auc']:.4f} [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]", flush=True)
        print(f"    K* (theta=0.60): {cond_data['kstar']}", flush=True)

    if unified_results:
        print("\n--- Unified Qwen Scorer (15D OBD features) ---", flush=True)
        for cond_name, cond_data in unified_results.items():
            print(f"  [{cond_name}]", flush=True)
            for k in range(1, 4):
                pair = f"{k-1}v{k}"
                r = cond_data["pairs"][pair]
                print(f"    {pair} AUC: {r['auc']:.4f} [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]", flush=True)
            print(f"    K* (theta=0.60): {cond_data['kstar']}", flush=True)

    print("\n--- vs C4 Multi-Source (comparison) ---", flush=True)
    print("  C4 unified no_label  K*: 2  (1v2 AUC: 0.7105)", flush=True)
    print("  C4 unified with_label K*: 2  (1v2 AUC: 0.7183)", flush=True)
    print("  C4 vocab-only K*: 2  (1v2 AUC: 0.6830)", flush=True)
    vocab_kstar = vocab_results["no_label"]["kstar"]
    print(f"  arXiv vocab-only K*: {vocab_kstar}", flush=True)
    if unified_results:
        uni_kstar = unified_results["no_label"]["kstar"]
        print(f"  arXiv unified K*: {uni_kstar}", flush=True)
    print("  arXiv single-source K*: 3 (from exp_deberta_arxiv)", flush=True)


def main():
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    vocab_results = run_vocab_only()

    vocab_time = time.time() - t_start
    log.info(f"Vocab-only completed in {vocab_time:.1f}s")

    interim_output = {
        "experiment": "exp_multi_source_arxiv",
        "domain": "arXiv",
        "generators": list(SOURCES.keys()),
        "docs_per_generator": DOCS_PER_GEN,
        "n_bootstrap": N_BOOT,
        "theta": THETA,
        "seed": SEED,
        "vocab_only": vocab_results,
        "vocab_only_seconds": round(vocab_time, 1),
    }
    interim_path = OUT_DIR / "results_vocab_only.json"
    with open(interim_path, "w") as f:
        json.dump(interim_output, f, indent=2)
    log.info(f"Vocab-only results saved to {interim_path}")

    print_summary(vocab_results, None)

    unified_results = run_unified_scorer()

    total_time = time.time() - t_start
    full_output = {
        "experiment": "exp_multi_source_arxiv",
        "domain": "arXiv",
        "scorer": "Qwen2.5-1.5B",
        "generators": list(SOURCES.keys()),
        "docs_per_generator": DOCS_PER_GEN,
        "n_bootstrap": N_BOOT,
        "theta": THETA,
        "seed": SEED,
        "total_seconds": round(total_time, 1),
        "vocab_only": vocab_results,
        "unified_scorer_no_label": unified_results["no_label"],
        "unified_scorer_with_label": unified_results["with_label"],
        "c4_comparison": {
            "c4_unified_no_label_kstar": 2,
            "c4_unified_with_label_kstar": 2,
            "c4_vocab_only_kstar": 2,
        },
    }
    out_path = OUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(full_output, f, indent=2)
    log.info(f"Full results saved to {out_path}")

    print_summary(vocab_results, unified_results)
    print(f"\nTotal time: {total_time:.1f}s", flush=True)


if __name__ == "__main__":
    main()
