#!/bin/bash
set -e

# exp-023: Wikipedia domain ablation
# 目的: 验证 K*=2 跨数据域（Wikipedia vs C4）的稳定性
# 模型: Qwen2.5-1.5B (base), continuation mode
# 数据: Wikipedia 5K docs, depth 0-5, max_new_tokens=256
# 输入: data/exp_023_wiki/depth_0.jsonl (pre-downloaded)
# 输出: results/exp_023_wiki/
# GPU: cuda:3 (via CUDA_VISIBLE_DEVICES)

# Block all HF network access — data is local
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/root/autodl-tmp/.hf_cache
export PYTHONHASHSEED=42

cd /root/autodl-tmp/gen-depth-contamination
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

MODEL=/root/autodl-tmp/.hf_cache/Qwen/Qwen2.5-1.5B

echo "=== exp-023: Wikipedia domain ablation ==="
echo "Start time: $(date)"
echo "Model: $MODEL"
echo "Data: data/exp_023_wiki/depth_0.jsonl"

# Verify seed data exists
if [ ! -f data/exp_023_wiki/depth_0.jsonl ]; then
  echo "ERROR: depth_0.jsonl not found!"
  exit 1
fi

SEED_COUNT=$(wc -l < data/exp_023_wiki/depth_0.jsonl)
echo "Seed data: $SEED_COUNT lines"

CUDA_VISIBLE_DEVICES=1 python scripts/run_pipeline.py \
  --model_path $MODEL \
  --generation_mode continuation \
  --num_samples 5000 \
  --max_depth 5 \
  --batch_size 16 \
  --max_new_tokens 256 \
  --data_dir data/exp_023_wiki \
  --results_dir results/exp_023_wiki

echo "=== exp-023 DONE ==="
echo "End time: $(date)"
