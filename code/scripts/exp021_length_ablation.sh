#!/bin/bash
set -e

# Network proxy for data download
source /etc/network_turbo

# HuggingFace cache on data disk
export HF_HOME=/root/autodl-tmp/.hf_cache

# SSL certificates for proxy
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

cd /root/autodl-tmp/gen-depth-contamination
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

export HF_HUB_DISABLE_XET=1
export PYTHONHASHSEED=42

MODEL=/root/autodl-tmp/.hf_cache/Qwen/Qwen2.5-1.5B

for LEN in 128 256 512; do
  echo "=== Running length=$LEN ==="
  CUDA_VISIBLE_DEVICES=3 python scripts/run_pipeline.py \
    --model_path $MODEL \
    --generation_mode continuation \
    --num_samples 5000 \
    --max_depth 5 \
    --batch_size 16 \
    --max_new_tokens $LEN \
    --data_dir data/exp_021_length_${LEN} \
    --results_dir results/exp_021_length_${LEN}
  echo "=== Done length=$LEN ==="
done
echo "ALL_DONE"
