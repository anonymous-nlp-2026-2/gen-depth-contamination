#!/usr/bin/env python3
"""
DeBERTa cross-model transfer test.

Train DeBERTa-v3-base on one generator model's depth chain data,
evaluate on another model's data. 5-fold GroupKFold ensemble.

Usage:
  # Single pair
  python scripts/run_deberta_cross_transfer.py \
      --train_model wiki_pythia --test_model wiki_qwen7b --gpu 2

  # Full N*N matrix (all off-diagonal pairs)
  python scripts/run_deberta_cross_transfer.py --all_pairs --gpu 2

Input:  data/exp_{model}/depth_{0..5}.jsonl  (each line: {"text":..., "doc_id":...})
Output: results/deberta_cross_transfer/{train}__to__{test}.json
        results/deberta_cross_transfer/matrix_summary.json  (--all_pairs)

Dependencies: torch, transformers, sklearn, numpy
"""

import argparse
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
from sklearn.metrics import roc_auc_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

ROOT = Path("/root/autodl-tmp/gen-depth-contamination")
DATA_DIR = ROOT / "data"

EXPERIMENTS = {
    "wiki_pythia": DATA_DIR / "exp_wiki_pythia",
    "wiki_olmo": DATA_DIR / "exp_wiki_olmo",
    "wiki_qwen7b": DATA_DIR / "exp_wiki_qwen7b",
    "wiki_llama8b": DATA_DIR / "exp_wiki_llama8b",
    "wiki_mistral7b": DATA_DIR / "exp_wiki_mistral7b",
}

MODEL_NAME = "microsoft/deberta-v3-base"
HF_CACHE = "/root/autodl-tmp/.hf_cache"
MAX_LEN = 512
N_CLASSES = 6
N_FOLDS = 5
EPOCHS = 3
BS = 16
GRAD_ACCUM = 2
LR = 2e-5
WARMUP = 0.1
WD = 0.01
N_BOOT = 1000
THETA = 0.60
SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
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

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.ids[i], self.masks[i], self.labels[i]


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


def train_and_predict(tr_ids, tr_masks, tr_labels,
                      te_ids, te_masks, te_labels, device, fold):
    log.info(f"  Fold {fold}: train={len(tr_labels)}, test={len(te_labels)}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=N_CLASSES, cache_dir=HF_CACHE,
        local_files_only=True, torch_dtype=torch.float32,
    ).to(device)

    tr_dl = DataLoader(
        DS(tr_ids, tr_masks, tr_labels), batch_size=BS,
        shuffle=True, collate_fn=collate, num_workers=2, pin_memory=True,
    )
    te_dl = DataLoader(
        DS(te_ids, te_masks, te_labels), batch_size=BS * 2,
        collate_fn=collate, num_workers=2, pin_memory=True,
    )

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
                log.info(
                    f"    Ep {ep+1} step {step}/{len(tr_dl)} "
                    f"loss={ep_loss/(step+1):.4f}"
                )
        log.info(f"    Ep {ep+1}/{EPOCHS} avg_loss={ep_loss/len(tr_dl):.4f}")

    model.eval()
    probs = []
    with torch.no_grad():
        for ids, mask, _ in te_dl:
            ids, mask = ids.to(device), mask.to(device)
            out = model(input_ids=ids, attention_mask=mask)
            probs.append(torch.softmax(out.logits, dim=-1).cpu().numpy())
    del model, opt, sched
    torch.cuda.empty_cache()
    gc.collect()
    return np.concatenate(probs)


def pairwise_metrics(labels, probs, doc_ids):
    rng = np.random.RandomState(SEED)
    results = {}
    for k in range(N_CLASSES - 1):
        key = f"{k}v{k+1}"
        mask = (labels == k) | (labels == k + 1)
        y = (labels[mask] == k + 1).astype(int)
        s = probs[mask][:, k + 1 :].sum(axis=1)
        docs = doc_ids[mask]

        auc = float(roc_auc_score(y, s))

        preds = (s >= 0.5).astype(int)
        f1 = float(f1_score(y, preds))

        doc2idx = {}
        for i, d in enumerate(docs):
            doc2idx.setdefault(int(d), []).append(i)
        idx_arr = [np.array(v) for v in doc2idx.values()]
        n_docs = len(idx_arr)
        boots = []
        for _ in range(N_BOOT):
            si = rng.randint(0, n_docs, n_docs)
            idx = np.concatenate([idx_arr[j] for j in si])
            try:
                boots.append(roc_auc_score(y[idx], s[idx]))
            except ValueError:
                continue
        boots = np.array(boots)
        ci_lo = float(np.percentile(boots, 2.5))
        ci_hi = float(np.percentile(boots, 97.5))

        results[key] = {
            "auc": round(auc, 4),
            "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4),
            "ci_lower_above_theta": bool(ci_lo > THETA),
            "f1_at_050": round(f1, 4),
        }

    kstar = 0
    for k in range(N_CLASSES - 1):
        if results[f"{k}v{k+1}"]["ci_lower_above_theta"]:
            kstar = k + 1
        else:
            break
    return results, kstar


def run_pair(train_name, test_name, tokenizer, device, output_dir):
    log.info(f"\n{'='*60}")
    log.info(f"TRAIN: {train_name}  ->  TEST: {test_name}")
    log.info(f"{'='*60}")
    t0 = time.time()

    tr_texts, tr_labels, tr_doc_ids = load_data(EXPERIMENTS[train_name])
    log.info(f"Train: {len(tr_texts)} samples from {train_name}")

    te_texts, te_labels, te_doc_ids = load_data(EXPERIMENTS[test_name])
    log.info(f"Test:  {len(te_texts)} samples from {test_name}")

    log.info("Tokenizing train...")
    tr_enc = tokenizer(tr_texts, truncation=True, max_length=MAX_LEN, padding=False)
    tr_all_ids, tr_all_masks = tr_enc["input_ids"], tr_enc["attention_mask"]

    log.info("Tokenizing test...")
    te_enc = tokenizer(te_texts, truncation=True, max_length=MAX_LEN, padding=False)
    te_all_ids, te_all_masks = te_enc["input_ids"], te_enc["attention_mask"]

    gkf = GroupKFold(n_splits=N_FOLDS)
    ensemble_probs = np.zeros((len(te_texts), N_CLASSES))

    for fi, (tr_i, _va_i) in enumerate(
        gkf.split(np.arange(len(tr_texts)), tr_labels, tr_doc_ids)
    ):
        fold_probs = train_and_predict(
            [tr_all_ids[i] for i in tr_i],
            [tr_all_masks[i] for i in tr_i],
            tr_labels[tr_i].tolist(),
            te_all_ids,
            te_all_masks,
            te_labels.tolist(),
            device,
            fi,
        )
        ensemble_probs += fold_probs

    ensemble_probs /= N_FOLDS

    pw, kstar = pairwise_metrics(te_labels, ensemble_probs, te_doc_ids)
    elapsed = time.time() - t0

    log.info(f"\nPairwise metrics ({train_name} -> {test_name}):")
    for pk, v in pw.items():
        log.info(
            f"  {pk}: AUC={v['auc']:.4f} [{v['ci_lower']:.4f}, "
            f"{v['ci_upper']:.4f}]  F1@0.50={v['f1_at_050']:.4f}"
        )
    log.info(f"  K* (theta={THETA}): {kstar}  Time: {elapsed:.0f}s")

    result = {
        "train_model": train_name,
        "test_model": test_name,
        "method": "DeBERTa-v3-base cross-transfer (5-fold ensemble)",
        "n_train": len(tr_texts),
        "n_test": len(te_texts),
        "epochs": EPOCHS,
        "lr": LR,
        "batch_size": BS,
        "max_length": MAX_LEN,
        "theta": THETA,
        "kstar": kstar,
        "pairwise": pw,
        "elapsed_seconds": round(elapsed, 1),
    }
    out_path = output_dir / f"{train_name}__to__{test_name}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"Saved: {out_path}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="DeBERTa cross-model transfer test"
    )
    parser.add_argument(
        "--train_model", type=str,
        help=f"Train model key. Available: {list(EXPERIMENTS.keys())}",
    )
    parser.add_argument(
        "--test_model", type=str,
        help=f"Test model key. Available: {list(EXPERIMENTS.keys())}",
    )
    parser.add_argument(
        "--all_pairs", action="store_true",
        help="Run full N*N matrix (all off-diagonal pairs)",
    )
    parser.add_argument("--gpu", type=int, default=2, help="GPU index (default: 2)")
    parser.add_argument(
        "--output_dir", type=str,
        default=str(ROOT / "results" / "deberta_cross_transfer"),
    )
    args = parser.parse_args()

    if not args.all_pairs and (not args.train_model or not args.test_model):
        parser.error("Specify --train_model and --test_model, or use --all_pairs")

    available = list(EXPERIMENTS.keys())
    if args.train_model and args.train_model not in EXPERIMENTS:
        parser.error(f"Unknown train_model: {args.train_model}. Available: {available}")
    if args.test_model and args.test_model not in EXPERIMENTS:
        parser.error(f"Unknown test_model: {args.test_model}. Available: {available}")

    os.environ["HF_HOME"] = HF_CACHE
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    device = torch.device(f"cuda:{args.gpu}")
    log.info(f"Device: {device} ({torch.cuda.get_device_name(args.gpu)})")

    log.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=HF_CACHE, use_fast=False, local_files_only=True,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all_pairs:
        pairs = [(tr, te) for tr in available for te in available if tr != te]
        log.info(f"Running {len(pairs)} cross-transfer pairs")
        all_results = {}
        for tr, te in pairs:
            res = run_pair(tr, te, tokenizer, device, output_dir)
            all_results[f"{tr}__to__{te}"] = res

        matrix = {}
        for tr in available:
            matrix[tr] = {}
            for te in available:
                if tr == te:
                    matrix[tr][te] = "self"
                else:
                    key = f"{tr}__to__{te}"
                    matrix[tr][te] = {
                        "kstar": all_results[key]["kstar"],
                        "pairwise_auc": {
                            k: v["auc"]
                            for k, v in all_results[key]["pairwise"].items()
                        },
                        "pairwise_f1": {
                            k: v["f1_at_050"]
                            for k, v in all_results[key]["pairwise"].items()
                        },
                    }

        summary = {"matrix": matrix, "models": available, "n_pairs": len(pairs)}
        with open(output_dir / "matrix_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        log.info(f"\nMatrix summary saved: {output_dir / 'matrix_summary.json'}")

        log.info(f"\n{'='*80}\nK* Matrix (train=row, test=col)")
        header = f"{'':>18}" + "".join(f"{m:>18}" for m in available)
        log.info(header)
        for tr in available:
            row = f"{tr:>18}"
            for te in available:
                if tr == te:
                    row += f"{'---':>18}"
                else:
                    row += f"{matrix[tr][te]['kstar']:>18}"
            log.info(row)
    else:
        run_pair(args.train_model, args.test_model, tokenizer, device, output_dir)

    log.info("All done.")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    main()
