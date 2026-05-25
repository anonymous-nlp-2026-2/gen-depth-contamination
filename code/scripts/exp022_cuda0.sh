#!/bin/bash
set -e
cd /root/autodl-tmp/gen-depth-contamination
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

export HF_HOME=/root/autodl-tmp/.hf_cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export PYTHONHASHSEED=42
export CUDA_VISIBLE_DEVICES=0

declare -A GEN_DATA
GEN_DATA[qwen_base]="data_exp016_qwen_base"
GEN_DATA[qwen_cont256]="data/exp_021_length_256"

declare -A SCORER_MODELS
SCORER_MODELS[qwen_instruct]="/root/autodl-tmp/.hf_cache/Qwen/Qwen2.5-1.5B-Instruct"
SCORER_MODELS[qwen_base]="/root/autodl-tmp/.hf_cache/Qwen/Qwen2.5-1.5B"
SCORER_MODELS[pythia]="/root/autodl-tmp/.hf_cache/models--EleutherAI--pythia-1.4b/snapshots/fedc38a16eea3bd36a96b906d78d11d2ce18ed79"
SCORER_MODELS[olmo]="/root/autodl-tmp/.hf_cache/allenai/OLMo-1B-hf"

OUT_BASE="results_exp022_cross_scorer"
DATA_BASE="data_exp022_cross_scorer"

COMBOS=(
  "qwen_base:qwen_instruct"
  "qwen_cont256:qwen_instruct"
  "qwen_cont256:qwen_base"
  "qwen_cont256:pythia"
  "qwen_cont256:olmo"
)

echo "=== exp022 cuda:0 — ${#COMBOS[@]} combos ==="

for combo_spec in "${COMBOS[@]}"; do
  gen_name="${combo_spec%%:*}"
  scorer_name="${combo_spec##*:}"
  gen_dir="${GEN_DATA[$gen_name]}"
  scorer_path="${SCORER_MODELS[$scorer_name]}"
  combo="${gen_name}__${scorer_name}"
  data_dir="${DATA_BASE}/${combo}"
  results_dir="${OUT_BASE}/${combo}"

  if [ -f "${results_dir}/summary.txt" ]; then
    echo "SKIP (done): ${combo}"
    continue
  fi

  echo "START: ${combo}"

  mkdir -p "$data_dir"
  for depth in 0 1 2 3 4 5; do
    src="$(pwd)/${gen_dir}/depth_${depth}.jsonl"
    dst="${data_dir}/depth_${depth}.jsonl"
    if [ ! -e "$dst" ]; then
      ln -s "$src" "$dst"
    fi
  done

  rm -f "${data_dir}/features.csv"

  python scripts/run_pipeline.py \
    --model_path "$scorer_path" \
    --data_dir "$data_dir" \
    --results_dir "$results_dir" \
    --num_samples 5000 \
    --max_depth 5 \
    --batch_size 16 \
    --generation_mode continuation

  echo "DONE: ${combo}"
  echo "---"
done

echo "=== cuda:0 ALL_DONE ==="
