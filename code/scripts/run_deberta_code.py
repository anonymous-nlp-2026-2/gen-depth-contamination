#!/usr/bin/env python3
"""
DeBERTa-v3-base upper-bound baseline for code domain (Qwen-1.5B scorer).
6-class ordinal depth classification, 5-fold GroupKFold, pairwise AUC with
doc-level bootstrap CI, K* determination.
"""

import gc
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

ROOT = Path("/root/autodl-tmp/gen-depth-contamination")
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "exp_deberta_code_qwen1b5"

EXPERIMENTS = {
    "code_qwen1b5": DATA_DIR / "exp_code_qwen1b5",
}

OBD_MAP = {
    "code_qwen1b5": ROOT / "results" / "exp_code_qwen1b5" / "pairwise_auc.json",
}

MODEL_NAME = "microsoft/deberta-v3-base"
HF_CACHE = "/root/autodl-tmp/.hf_cache"
MAX_LEN = 512
N_CLASSES = 6
N_FOLDS = 5
EPOCHS = 3
BS = 16
EVAL_BS = 8
GRAD_ACCUM = 2
LR = 2e-5
WARMUP = 0.1
WD = 0.01
N_BOOT = 10000
THETA = 0.60
SEED = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def load_data(exp_dir):
    texts, labels, doc_ids = [], [], []
    for d in range(N_CLASSES):
        with open(exp_dir / f"depth_{d}.jsonl") as f:
            for line in f:
                obj = json.loads(line)
                texts.append(obj["text"])
                labels.append(d)
                doc_ids.append(obj["doc_id"])
    return texts, np.array(labels), np.array(doc_ids)


class DS(Dataset):
    def __init__(self, ids, masks, labels):
        self.ids, self.masks, self.labels = ids, masks, labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return self.ids[i], self.masks[i], self.labels[i]


def collate(batch):
    ids_l, mask_l, lab_l = zip(*batch)
    mx = max(len(x) for x in ids_l)
    bsz = len(ids_l)
    ids = torch.zeros(bsz, mx, dtype=torch.long)
    masks = torch.zeros(bsz, mx, dtype=torch.long)
    for i, (x, m) in enumerate(zip(ids_l, mask_l)):
        n = len(x)
        ids[i, :n] = torch.tensor(x, dtype=torch.long)
        masks[i, :n] = torch.tensor(m, dtype=torch.long)
    return ids, masks, torch.tensor(lab_l, dtype=torch.long)


def train_fold(tr_ids, tr_masks, tr_labels, va_ids, va_masks, va_labels, device, fold):
    log.info(f"  Fold {fold}: train={len(tr_labels)}, val={len(va_labels)}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=N_CLASSES, cache_dir=HF_CACHE,
        local_files_only=True, torch_dtype=torch.float32,
    ).to(device)

    tr_dl = DataLoader(DS(tr_ids, tr_masks, tr_labels), batch_size=BS, shuffle=True, collate_fn=collate, num_workers=2, pin_memory=True)
    va_dl = DataLoader(DS(va_ids, va_masks, va_labels), batch_size=EVAL_BS, collate_fn=collate, num_workers=2, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    total_opt = (len(tr_dl) // GRAD_ACCUM) * EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(total_opt * WARMUP), total_opt)

    model.train()
    for ep in range(EPOCHS):
        ep_loss = 0.0
        opt.zero_grad()
        for step, (ids, mask, lab) in enumerate(tr_dl):
            ids, mask, lab = ids.to(device), mask.to(device), lab.to(device)
            out = model(input_ids=ids, attention_mask=mask, labels=lab)
            (out.loss / GRAD_ACCUM).backward()
            ep_loss += out.loss.item()
            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
            if step % 200 == 0 and step > 0:
                log.info(f"    Ep {ep+1} step {step}/{len(tr_dl)} loss={ep_loss/(step+1):.4f}")
        log.info(f"    Ep {ep+1}/{EPOCHS} avg_loss={ep_loss/len(tr_dl):.4f}")

    torch.cuda.empty_cache()
    model.eval()
    probs = []
    with torch.no_grad():
        for ids, mask, _ in va_dl:
            ids, mask = ids.to(device), mask.to(device)
            out = model(input_ids=ids, attention_mask=mask)
            probs.append(torch.softmax(out.logits, dim=-1).cpu().numpy())
    del model, opt, sched
    torch.cuda.empty_cache()
    gc.collect()
    return np.concatenate(probs)


def pairwise_auc_ci(labels, probs, doc_ids):
    rng = np.random.RandomState(SEED)
    results = {}
    for k in range(N_CLASSES - 1):
        key = f"{k}v{k+1}"
        mask = (labels == k) | (labels == k + 1)
        y = (labels[mask] == k + 1).astype(int)
        s = probs[mask][:, k+1:].sum(axis=1)
        docs = doc_ids[mask]
        auc = float(roc_auc_score(y, s))
        doc2idx = {}
        for i, d in enumerate(docs):
            doc2idx.setdefault(d, []).append(i)
        idx_arr = [np.array(v) for v in doc2idx.values()]
        n_docs = len(idx_arr)
        boots = []
        for _ in range(N_BOOT):
            si = rng.randint(0, n_docs, n_docs)
            idx = np.concatenate([idx_arr[j] for j in si])
            try: boots.append(roc_auc_score(y[idx], s[idx]))
            except ValueError: continue
        boots = np.array(boots)
        ci_lo, ci_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        results[key] = {"auc": round(auc,4), "ci_lower": round(ci_lo,4), "ci_upper": round(ci_hi,4),
                         "ci_lower_above_theta": bool(ci_lo > THETA), "boot_mean": round(float(boots.mean()),4), "boot_std": round(float(boots.std()),4)}
    kstar = 0
    for k in range(N_CLASSES - 1):
        if results[f"{k}v{k+1}"]["ci_lower_above_theta"]: kstar = k + 1
        else: break
    return results, kstar


def load_obd(name):
    p = OBD_MAP.get(name)
    if p and p.exists():
        data = json.load(open(p))
        return data.get("mean", data)
    return None


def main():
    os.environ["HF_HOME"] = HF_CACHE
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    device = torch.device("cuda:0")
    log.info(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    log.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE, use_fast=False, local_files_only=True)
    log.info("Tokenizer loaded")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_res = {}

    for exp_name, exp_dir in EXPERIMENTS.items():
        log.info(f"\n{'='*60}\n{exp_name}\n{'='*60}")
        t0 = time.time()
        texts, labels, doc_ids = load_data(exp_dir)
        N = len(texts)
        log.info(f"Loaded {N} samples ({N // N_CLASSES} docs x {N_CLASSES} depths)")

        log.info("Tokenizing...")
        enc = tokenizer(texts, truncation=True, max_length=MAX_LEN, padding=False)
        all_ids, all_masks = enc["input_ids"], enc["attention_mask"]
        log.info("Tokenization done")

        gkf = GroupKFold(n_splits=N_FOLDS)
        oof = np.zeros((N, N_CLASSES))

        for fi, (tr_i, va_i) in enumerate(gkf.split(np.arange(N), labels, doc_ids)):
            oof[va_i] = train_fold(
                [all_ids[i] for i in tr_i], [all_masks[i] for i in tr_i], labels[tr_i].tolist(),
                [all_ids[i] for i in va_i], [all_masks[i] for i in va_i], labels[va_i].tolist(),
                device, fi,
            )

        np.save(RESULTS_DIR / "oof.npy", oof)
        pw, kstar = pairwise_auc_ci(labels, oof, doc_ids)
        elapsed = time.time() - t0

        log.info(f"\nPairwise AUC ({exp_name}):")
        for pk, v in pw.items():
            log.info(f"  {pk}: {v['auc']:.4f} [{v['ci_lower']:.4f}, {v['ci_upper']:.4f}]")
        log.info(f"  K* (theta={THETA}): {kstar}  Time: {elapsed:.0f}s")

        obd = load_obd(exp_name)
        exp_res = {"experiment": exp_name, "model": MODEL_NAME, "method": "DeBERTa-v3-base fine-tuned 6-class",
                    "n_samples": N, "epochs": EPOCHS, "lr": LR, "batch_size": BS, "max_length": MAX_LEN,
                    "theta": THETA, "kstar": kstar, "pairwise_auc": pw, "elapsed_seconds": round(elapsed, 1)}
        if obd: exp_res["obd_baseline"] = obd
        all_res[exp_name] = exp_res
        json.dump(exp_res, open(RESULTS_DIR / f"{exp_name}.json", "w"), indent=2)
        log.info(f"Saved: {RESULTS_DIR / f'{exp_name}.json'}")

    json.dump(all_res, open(RESULTS_DIR / "summary.json", "w"), indent=2)

    log.info(f"\n{'='*80}\nDeBERTa vs OBD-LightGBM\n{'='*80}")
    for name, r in all_res.items():
        obd = r.get("obd_baseline", {})
        for pk in ["0v1", "1v2", "2v3", "3v4", "4v5"]:
            da = r["pairwise_auc"][pk]["auc"]
            cl, ch = r["pairwise_auc"][pk]["ci_lower"], r["pairwise_auc"][pk]["ci_upper"]
            oa = obd.get(pk)
            log.info(f"{name:<18} {pk} DeBERTa={da:.4f} [{cl:.4f},{ch:.4f}] OBD={oa if oa else 'N/A'} delta={da-oa:+.4f}" if oa else f"{name:<18} {pk} DeBERTa={da:.4f} [{cl:.4f},{ch:.4f}]")
        log.info(f"  K*(DeBERTa)={r['kstar']}")
    log.info("All done.")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    main()
