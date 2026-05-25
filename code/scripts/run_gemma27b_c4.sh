#!/bin/bash
set -e

source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

export HF_HOME=/root/autodl-tmp/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

cd /root/autodl-tmp/gen-depth-contamination

MODEL_PATH="/root/autodl-tmp/.hf_cache/google/gemma-2-27b"
SCORER_PATH="/root/autodl-tmp/.hf_cache/Qwen/Qwen2___5-1___5B-Instruct"
DATA_DIR="data/exp_gemma27b_c4"
RESULTS_DIR="results/exp_gemma27b_c4"

echo "=== Phase 1: Gemma-2-27B Generation + Self-Features ==="
echo "Start: $(date)"
python3 scripts/run_pipeline.py \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --results_dir "$RESULTS_DIR" \
    --generation_mode continuation \
    --top_p 0.95 \
    --temperature 1.0 \
    --num_samples 5000 \
    --max_depth 5 \
    --batch_size 8 \
    --seed 42 \
    --dtype bf16

echo "=== Phase 2: Qwen-1.5B Cross-Scoring ==="
echo "Start: $(date)"
python3 scripts/run_pipeline.py \
    --model_path "$SCORER_PATH" \
    --data_dir "$DATA_DIR" \
    --results_dir "$RESULTS_DIR" \
    --skip_generation \
    --num_samples 5000 \
    --max_depth 5 \
    --seed 42

echo "=== All Done ==="
echo "End: $(date)"
