import os, json, re, warnings, time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import nltk
from sentence_transformers import SentenceTransformer
import spacy

warnings.filterwarnings("ignore")
t0 = time.time()

BASE = Path("/root/autodl-tmp/gen-depth-contamination")

MODELS = {
    "pythia_1.4b": {
        "data_dir": BASE / "data" / "exp_008_retrain_chain",
        "max_depth": 4,
    },
    "qwen_cont": {
        "data_dir": BASE / "data_exp022_cross_scorer" / "qwen_cont256__qwen_base",
        "max_depth": 5,
    },
}

K_STAR_THRESHOLD = 0.60

print(f"[{time.time()-t0:.0f}s] Loading sentence-transformers model (CPU)...")
st_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
print(f"[{time.time()-t0:.0f}s] Loading spaCy model...")
nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "textcat"])
sent_tokenizer = nltk.data.load("tokenizers/punkt_tab/english.pickle")
print(f"[{time.time()-t0:.0f}s] Models loaded.")


def compute_semantic_features(texts):
    cosine_sims = []
    entity_divs = []
    
    all_sentences = []
    doc_sentence_counts = []
    
    for text in texts:
        sents = sent_tokenizer.tokenize(text)
        if len(sents) < 2:
            sents = text.split("\n")
            sents = [s.strip() for s in sents if s.strip()]
        if len(sents) < 2:
            sents = [text[:len(text)//2], text[len(text)//2:]]
        all_sentences.extend(sents)
        doc_sentence_counts.append(len(sents))
    
    print(f"    Encoding {len(all_sentences)} sentences...")
    all_embeddings = st_model.encode(
        all_sentences, batch_size=512,
        show_progress_bar=False, normalize_embeddings=True
    )
    
    idx = 0
    for count in doc_sentence_counts:
        embs = all_embeddings[idx:idx+count]
        idx += count
        if count >= 2:
            cos_sims = np.array([
                float(np.dot(embs[i], embs[i+1]))
                for i in range(len(embs) - 1)
            ])
            cosine_sims.append(float(np.mean(cos_sims)))
        else:
            cosine_sims.append(0.0)
    
    print(f"    Computing entity diversity...")
    for doc in nlp.pipe(texts, batch_size=1000, n_process=1):
        ents = [ent.text for ent in doc.ents]
        if len(ents) > 0:
            entity_divs.append(len(set(ents)) / len(ents))
        else:
            entity_divs.append(0.0)
    
    return cosine_sims, entity_divs


def run_lgbm_cv(X, y, groups):
    gkf = GroupKFold(n_splits=5)
    aucs = []
    importances = np.zeros(X.shape[1])
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "seed": 42,
        }
        
        model = lgb.train(
            params, dtrain,
            num_boost_round=300,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        
        y_pred = model.predict(X_val)
        auc = roc_auc_score(y_val, y_pred)
        aucs.append(auc)
        importances += model.feature_importance(importance_type="gain")
    
    importances /= 5
    return np.mean(aucs), aucs, importances


def get_k_star(aucs_dict):
    for pair in sorted(aucs_dict.keys()):
        k = int(pair.split("v")[0])
        if aucs_dict[pair] < K_STAR_THRESHOLD:
            return k
    return "> max_depth"


results = {}

for model_name, cfg in MODELS.items():
    print(f"\n{'='*60}")
    print(f"[{time.time()-t0:.0f}s] Processing: {model_name}")
    
    features_csv = cfg["data_dir"] / "features.csv"
    existing_df = pd.read_csv(features_csv)
    feature_cols_15d = [c for c in existing_df.columns if c not in ("doc_id", "depth")]
    print(f"  Existing features: {existing_df.shape}, {len(feature_cols_15d)}D")
    
    all_texts = {}
    for d in range(cfg["max_depth"] + 1):
        jsonl_path = cfg["data_dir"] / f"depth_{d}.jsonl"
        texts = []
        with open(jsonl_path) as f:
            for line in f:
                obj = json.loads(line)
                texts.append(obj["text"])
        all_texts[d] = texts
    
    semantic_records = []
    for d in range(cfg["max_depth"] + 1):
        print(f"  [{time.time()-t0:.0f}s] depth {d} ({len(all_texts[d])} docs)...")
        cosine_sims, entity_divs = compute_semantic_features(all_texts[d])
        for i, (cs, ed) in enumerate(zip(cosine_sims, entity_divs)):
            semantic_records.append({
                "doc_id": i, "depth": d,
                "semantic_cosine_sim": cs, "entity_diversity": ed,
            })
    
    sem_df = pd.DataFrame(semantic_records)
    merged_df = existing_df.merge(sem_df, on=["doc_id", "depth"], how="left")
    feature_cols_17d = feature_cols_15d + ["semantic_cosine_sim", "entity_diversity"]
    
    aucs_15d = {}
    aucs_17d = {}
    importance_17d_all = {}
    
    pairs = [f"{k}v{k+1}" for k in range(cfg["max_depth"])]
    
    for pair in pairs:
        k1, k2 = int(pair.split("v")[0]), int(pair.split("v")[1])
        pair_df = merged_df[merged_df["depth"].isin([k1, k2])].copy()
        pair_df["label"] = (pair_df["depth"] == k2).astype(int)
        groups = pair_df["doc_id"].values
        
        X_15 = pair_df[feature_cols_15d].values
        X_17 = pair_df[feature_cols_17d].values
        y = pair_df["label"].values
        
        auc_15, _, _ = run_lgbm_cv(X_15, y, groups)
        auc_17, _, imp_17 = run_lgbm_cv(X_17, y, groups)
        
        aucs_15d[pair] = round(auc_15, 4)
        aucs_17d[pair] = round(auc_17, 4)
        importance_17d_all[pair] = dict(zip(feature_cols_17d, imp_17.tolist()))
        
        print(f"  [{time.time()-t0:.0f}s] {pair}: 15D={auc_15:.4f} 17D={auc_17:.4f} diff={auc_17-auc_15:+.4f}")
    
    avg_importance = {}
    for feat in feature_cols_17d:
        vals = [importance_17d_all[p][feat] for p in pairs]
        avg_importance[feat] = np.mean(vals)
    
    sorted_feats = sorted(avg_importance.items(), key=lambda x: -x[1])
    feat_ranks = {f: i+1 for i, (f, _) in enumerate(sorted_feats)}
    
    k_star_15 = get_k_star(aucs_15d)
    k_star_17 = get_k_star(aucs_17d)
    
    print(f"  K* 15D={k_star_15}, 17D={k_star_17}")
    print(f"  semantic_cosine_sim rank={feat_ranks.get('semantic_cosine_sim')}")
    print(f"  entity_diversity rank={feat_ranks.get('entity_diversity')}")
    
    results[model_name] = {
        "15d_aucs": aucs_15d,
        "17d_aucs": aucs_17d,
        "k_star_15d": k_star_15,
        "k_star_17d": k_star_17,
        "feature_importance_rank": {
            "semantic_cosine_sim": feat_ranks.get("semantic_cosine_sim"),
            "entity_diversity": feat_ranks.get("entity_diversity"),
        },
        "feature_importance_raw": {
            "semantic_cosine_sim": round(avg_importance.get("semantic_cosine_sim", 0), 1),
            "entity_diversity": round(avg_importance.get("entity_diversity", 0), 1),
        },
        "all_feature_ranks": feat_ranks,
    }

k_stars_unchanged = all(
    results[m]["k_star_15d"] == results[m]["k_star_17d"] for m in results
)
results["summary"] = (
    f"Cross-model confirmation: K* {'unchanged' if k_stars_unchanged else 'CHANGED'} "
    f"for all models tested. "
    f"Pythia-1.4B: K*={results['pythia_1.4b']['k_star_15d']}, "
    f"Qwen-cont: K*={results['qwen_cont']['k_star_15d']}."
)

out_path = BASE / "artifacts" / "semantic_ablation_multi_model.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n[{time.time()-t0:.0f}s] DONE. Saved to {out_path}")
print(f"Summary: {results['summary']}")
