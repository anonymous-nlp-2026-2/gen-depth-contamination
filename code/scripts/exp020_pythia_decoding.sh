#!/bin/bash
# exp-020: Pythia-1.4B decoding ablation
# Parallel to exp-003 (Qwen decoding ablation) — validates K*=2 finding across model families.
# 4 configs: greedy, nucleus p=0.9/T=1.0, nucleus p=0.95/T=0.7, nucleus p=0.95/T=1.2
set -e

cd /root/autodl-tmp/gen-depth-contamination
MODEL="/root/autodl-tmp/.hf_cache/models--EleutherAI--pythia-1.4b/snapshots/fedc38a16eea3bd36a96b906d78d11d2ce18ed79"
COMMON="--num_samples 5000 --max_depth 5 --batch_size 16 --generation_mode continuation"

# Config 1: Greedy decoding (deterministic baseline, matches exp-003 config 1)
echo "=== [exp-020] Config 1/4: Greedy ==="
python scripts/run_pipeline.py --model_path $MODEL \
  --data_dir data_exp020_pythia_greedy --results_dir results_exp020_pythia_greedy \
  $COMMON --no_sample

# Config 2: Nucleus p=0.9, T=1.0 (matches exp-003 config 2)
echo "=== [exp-020] Config 2/4: Nucleus p=0.9 T=1.0 ==="
python scripts/run_pipeline.py --model_path $MODEL \
  --data_dir data_exp020_pythia_nuc09 --results_dir results_exp020_pythia_nuc09 \
  $COMMON --top_p 0.9 --temperature 1.0

# Config 3: Nucleus p=0.95, T=0.7 (matches exp-003 config 3)
echo "=== [exp-020] Config 3/4: Nucleus p=0.95 T=0.7 ==="
python scripts/run_pipeline.py --model_path $MODEL \
  --data_dir data_exp020_pythia_t07 --results_dir results_exp020_pythia_t07 \
  $COMMON --top_p 0.95 --temperature 0.7

# Config 4: Nucleus p=0.95, T=1.2 (matches exp-003 config 4)
echo "=== [exp-020] Config 4/4: Nucleus p=0.95 T=1.2 ==="
python scripts/run_pipeline.py --model_path $MODEL \
  --data_dir data_exp020_pythia_t12 --results_dir results_exp020_pythia_t12 \
  $COMMON --top_p 0.95 --temperature 1.2

echo "=== [exp-020] All 4 configs complete. ==="
