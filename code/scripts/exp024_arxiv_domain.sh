#!/bin/bash
set -e

# exp-024: arXiv abstracts domain ablation
# 目的: 验证 K*=2 跨数据域（arXiv abstracts vs C4/Wikipedia）的稳定性
# 模型: Qwen2.5-1.5B (base), continuation mode, nucleus p=0.95 T=1.0
# 数据: arXiv abstracts 5K docs, depth 0-5
# 输入: data/exp_024_arxiv/depth_0.jsonl
# 输出: results/exp_024_arxiv/

# Block all HF network access — data is local
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/root/autodl-tmp/.hf_cache
export PYTHONHASHSEED=42

cd /root/autodl-tmp/gen-depth-contamination
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

MODEL=/root/autodl-tmp/.hf_cache/Qwen/Qwen2.5-1.5B

echo "=== exp-024: arXiv abstracts domain ablation ==="
echo "Start time: $(date)"
echo "Model: $MODEL"
echo "Data: data/exp_024_arxiv/depth_0.jsonl"

# Verify seed data exists
if [ ! -f data/exp_024_arxiv/depth_0.jsonl ]; then
  echo "ERROR: depth_0.jsonl not found!"
  exit 1
fi

SEED_COUNT=$(wc -l < data/exp_024_arxiv/depth_0.jsonl)
echo "Seed data: $SEED_COUNT lines"

CUDA_VISIBLE_DEVICES=2 python scripts/run_pipeline.py \
  --model_path $MODEL \
  --generation_mode continuation \
  --num_samples 2200 \
  --max_depth 5 \
  --batch_size 16 \
  --max_new_tokens 256 \
  --data_dir data/exp_024_arxiv \
  --results_dir results/exp_024_arxiv

echo "=== exp-024 DONE ==="
echo "End time: $(date)"
