#!/bin/bash
set -e

cd /root/autodl-tmp/gen-depth-contamination
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

export HF_HOME=/root/autodl-tmp/.hf_cache
export PYTHONHASHSEED=42
export CUDA_VISIBLE_DEVICES=0

# Source data: exp_14b_c4 depth chain output
# Update this path when exp_14b_c4 data is ready
SRC_DATA_DIR="data/exp_14b_c4"
RESULTS_DIR="results_14b_scorer_confound"

echo "=== 14B Scorer Confound Test ==="
echo "Source data: $SRC_DATA_DIR"
echo "Results:     $RESULTS_DIR"

python scripts/score_14b_confound.py \
    --src_data_dir "$SRC_DATA_DIR" \
    --results_dir "$RESULTS_DIR" \
    --max_depth 5 \
    --batch_size 8

echo "=== DONE ==="
