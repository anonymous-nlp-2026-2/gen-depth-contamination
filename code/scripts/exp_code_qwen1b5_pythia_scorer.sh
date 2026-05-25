#!/bin/bash
set -e

cd /root/autodl-tmp/gen-depth-contamination
source /root/miniconda3/etc/profile.d/conda.sh
export CUDA_VISIBLE_DEVICES=0
conda activate base

export HF_HOME=/root/autodl-tmp/.hf_cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export PYTHONHASHSEED=42

SCORER_PATH="/root/autodl-tmp/.hf_cache/models--EleutherAI--pythia-1.4b/snapshots/fedc38a16eea3bd36a96b906d78d11d2ce18ed79"
SRC_DATA="data/exp_code_qwen1b5"
DATA_DIR="data/exp_code_qwen1b5_pythia_scorer"
RESULTS_DIR="results/exp_code_qwen1b5_pythia_scorer"

# Create data dir with symlinks to Qwen-generated depth chain
mkdir -p "$DATA_DIR"
for depth in 0 1 2 3 4 5; do
  src="$(pwd)/${SRC_DATA}/depth_${depth}.jsonl"
  dst="${DATA_DIR}/depth_${depth}.jsonl"
  if [ ! -e "$dst" ]; then
    ln -s "$src" "$dst"
  fi
done

# Remove stale features.csv to force re-extraction with Pythia scorer
rm -f "${DATA_DIR}/features.csv"

echo "=== exp_code_qwen1b5_pythia_scorer: Pythia-1.4B scoring Qwen Code generations ==="
echo "Scorer: $SCORER_PATH"
echo "Data:   $DATA_DIR"
echo "Output: $RESULTS_DIR"

python scripts/run_pipeline.py \
  --model_path "$SCORER_PATH" \
  --data_dir "$DATA_DIR" \
  --results_dir "$RESULTS_DIR" \
  --num_samples 5500 \
  --max_depth 5 \
  --batch_size 16 \
  --feat_batch_size 8 \
  --generation_mode continuation \
  --skip_generation

echo "=== DONE ==="
