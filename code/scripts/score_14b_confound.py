"""
14B Scorer Confound Test

Re-scores depth chain data using Qwen2.5-14B-Instruct-AWQ as scorer,
to rule out "1.5B scorer too weak" confound.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/score_14b_confound.py \
        --src_data_dir data/exp_14b_c4 \
        --results_dir results_14b_scorer_confound \
        --max_depth 5
"""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_pipeline import (
    load_jsonl, compute_surprisal_features, compute_vocab_diversity,
    compute_tail_features, run_analysis,
    SURPRISAL_COLS, VOCAB_COLS, TAIL_COLS, ALL_FEAT_COLS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SCORER_MODEL_PATH = "/root/autodl-tmp/.hf_cache/Qwen2.5-14B-Instruct-AWQ"


def load_14b_scorer():
    from transformers import AutoModelForCausalLM, AutoTokenizer, AwqConfig

    log.info(f"Loading tokenizer from {SCORER_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(SCORER_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    log.info("Loading Qwen2.5-14B-Instruct-AWQ model...")
    qc = AwqConfig(bits=4, group_size=128, zero_point=True, backend="autoawq", version="gemm")
    model = AutoModelForCausalLM.from_pretrained(
        SCORER_MODEL_PATH,
        quantization_config=qc,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    log.info(f"Model loaded. GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="14B Scorer Confound Test")
    parser.add_argument("--src_data_dir", type=str, required=True,
                        help="Source data dir with depth_0..5.jsonl (e.g. data/exp_14b_c4)")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    src_dir = Path(args.src_data_dir)
    results_dir = Path(args.results_dir)
    work_dir = results_dir / "data_14b_scored"
    work_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    for d in range(args.max_depth + 1):
        src = src_dir / f"depth_{d}.jsonl"
        dst = work_dir / f"depth_{d}.jsonl"
        if not src.exists():
            log.error(f"Missing {src}. Run the generator experiment first.")
            sys.exit(1)
        if not dst.exists():
            os.symlink(src.resolve(), dst)
        n = sum(1 for _ in open(dst))
        log.info(f"  depth_{d}.jsonl: {n} samples")

    model, tokenizer = load_14b_scorer()

    log.info("=" * 40 + " Feature Extraction (14B scorer) " + "=" * 40)

    depth_texts = {}
    for d in range(args.max_depth + 1):
        depth_texts[d] = load_jsonl(work_dir / f"depth_{d}.jsonl")

    rows = []
    for d in range(args.max_depth + 1):
        records = depth_texts[d]
        texts = [r["text"] for r in records]
        doc_ids = [r["doc_id"] for r in records]
        log.info(f"  depth {d}: {len(texts)} texts")

        surp = compute_surprisal_features(model, tokenizer, texts, batch_size=args.batch_size)
        vocab = compute_vocab_diversity(texts, tokenizer, depth_texts, d)
        tail = compute_tail_features(texts)

        for i in range(len(texts)):
            row = {"doc_id": doc_ids[i], "depth": d}
            for j, col in enumerate(SURPRISAL_COLS):
                row[col] = surp[i][j]
            for j, col in enumerate(VOCAB_COLS):
                row[col] = vocab[i][j]
            for j, col in enumerate(TAIL_COLS):
                row[col] = tail[i][j]
            rows.append(row)

    feat_path = work_dir / "features.csv"
    with open(feat_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "depth"] + ALL_FEAT_COLS)
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Saved {len(rows)} feature rows to {feat_path}")

    del model
    torch.cuda.empty_cache()

    log.info("=" * 40 + " Analysis " + "=" * 40)
    run_analysis(work_dir, results_dir, args.max_depth)

    log.info("14B scorer confound test complete.")


if __name__ == "__main__":
    main()
