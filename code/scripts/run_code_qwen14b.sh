#!/bin/bash
set -e

cd /root/autodl-tmp/gen-depth-contamination
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

export HF_HOME=/root/autodl-tmp/.hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export PYTHONHASHSEED=42
# Two free GPUs for 14B fp16 model sharding (~28GB across 48GB total)
export CUDA_VISIBLE_DEVICES=0,2

MODEL_PATH="/root/autodl-tmp/.hf_cache/Qwen/Qwen2___5-14B"
SCORER_PATH="/root/autodl-tmp/.hf_cache/Qwen/Qwen2___5-1___5B"
SRC_DATA="data/exp_code_qwen1b5"
DATA_DIR="data/exp_code_qwen14b"
RESULTS_DIR="results/exp_code_qwen14b"

# Verify model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Download first: HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen2.5-14B --local-dir $MODEL_PATH"
    exit 1
fi

# Create data dir and symlink seed data from exp_code_qwen1b5
mkdir -p "$DATA_DIR"
if [ ! -e "${DATA_DIR}/depth_0.jsonl" ]; then
    ln -s "$(pwd)/${SRC_DATA}/depth_0.jsonl" "${DATA_DIR}/depth_0.jsonl"
fi

echo "=== exp_code_qwen14b: Qwen2.5-14B Code domain K* measurement ==="
echo "Generator: $MODEL_PATH"
echo "Scorer:    $SCORER_PATH"
echo "Data:      $DATA_DIR"
echo "Results:   $RESULTS_DIR"
echo "GPUs:      $CUDA_VISIBLE_DEVICES"

python scripts/run_pipeline.py \
  --model_path "$MODEL_PATH" \
  --data_dir "$DATA_DIR" \
  --results_dir "$RESULTS_DIR" \
  --scorer_path "$SCORER_PATH" \
  --num_samples 5000 \
  --max_depth 5 \
  --batch_size 4 \
  --feat_batch_size 4 \
  --generation_mode continuation \
  --max_new_tokens 256

echo "=== DONE ==="
