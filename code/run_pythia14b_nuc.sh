#!/bin/bash
set -e
cd /root/autodl-tmp/gen-depth-contamination
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
source /etc/network_turbo

export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/root/autodl-tmp/.hf_cache
export HF_HUB_DISABLE_XET=1
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

echo "=== Pythia-1.4B C4 Nucleus Sampling Pipeline ==="
echo "Start: $(date)"
echo ""

python scripts/run_pipeline.py \
  --model_path /root/autodl-tmp/.hf_cache/models--EleutherAI--pythia-1.4b/snapshots/fedc38a16eea3bd36a96b906d78d11d2ce18ed79 \
  --num_samples 5000 \
  --max_depth 5 \
  --batch_size 8 \
  --feat_batch_size 4 \
  --generation_mode continuation \
  --data_dir data/exp_pythia14b_c4_nuc \
  --results_dir results/exp_pythia14b_c4_nuc

echo ""
echo "=== Done: $(date) ==="
