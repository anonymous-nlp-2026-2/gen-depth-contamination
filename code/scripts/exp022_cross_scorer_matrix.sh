#!/bin/bash
set -e

cd /root/autodl-tmp/gen-depth-contamination
source /root/miniconda3/etc/profile.d/conda.sh
export CUDA_VISIBLE_DEVICES=3
conda activate base

export HF_HOME=/root/autodl-tmp/.hf_cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export PYTHONHASHSEED=42

# ---- Generator data directories ----
declare -A GEN_DATA
GEN_DATA[qwen_base]="data_exp016_qwen_base"
GEN_DATA[pythia_greedy]="data_exp020_pythia_greedy"
GEN_DATA[qwen_cont256]="data/exp_021_length_256"

# ---- Scorer model paths ----
declare -A SCORER_MODELS
SCORER_MODELS[qwen_instruct]="/root/autodl-tmp/.hf_cache/Qwen/Qwen2.5-1.5B-Instruct"
SCORER_MODELS[qwen_base]="/root/autodl-tmp/.hf_cache/Qwen/Qwen2.5-1.5B"
SCORER_MODELS[pythia]="/root/autodl-tmp/.hf_cache/models--EleutherAI--pythia-1.4b/snapshots/fedc38a16eea3bd36a96b906d78d11d2ce18ed79"
SCORER_MODELS[olmo]="/root/autodl-tmp/.hf_cache/allenai/OLMo-1B-hf"

OUT_BASE="results_exp022_cross_scorer"
DATA_BASE="data_exp022_cross_scorer"

TOTAL=0
DONE=0

# Count combos
for gen_name in qwen_base pythia_greedy qwen_cont256; do
  for scorer_name in qwen_instruct qwen_base pythia olmo; do
    TOTAL=$((TOTAL + 1))
  done
done

echo "=== exp-022 cross-scorer matrix: ${TOTAL} combos ==="

for gen_name in qwen_base pythia_greedy qwen_cont256; do
  gen_dir="${GEN_DATA[$gen_name]}"
  for scorer_name in qwen_instruct qwen_base pythia olmo; do
    scorer_path="${SCORER_MODELS[$scorer_name]}"
    combo="${gen_name}__${scorer_name}"
    data_dir="${DATA_BASE}/${combo}"
    results_dir="${OUT_BASE}/${combo}"
    DONE=$((DONE + 1))

    # Skip if results already exist
    if [ -f "${results_dir}/summary.txt" ]; then
      echo "[$DONE/$TOTAL] SKIP (done): ${combo}"
      continue
    fi

    # Verify scorer model exists
    if [ ! -f "${scorer_path}/config.json" ]; then
      echo "[$DONE/$TOTAL] SKIP (no model): ${combo} — ${scorer_path}"
      continue
    fi

    echo "[$DONE/$TOTAL] START: gen=${gen_name} scorer=${scorer_name}"

    # Create data dir with symlinks to depth chain files
    mkdir -p "$data_dir"
    for depth in 0 1 2 3 4 5; do
      src="$(pwd)/${gen_dir}/depth_${depth}.jsonl"
      dst="${data_dir}/depth_${depth}.jsonl"
      if [ ! -e "$dst" ]; then
        ln -s "$src" "$dst"
      fi
    done

    # Remove stale features.csv to force re-extraction with new scorer
    rm -f "${data_dir}/features.csv"

    # Run pipeline — steps 2-3 skip (data exists), only 4-5 execute
    python scripts/run_pipeline.py \
      --model_path "$scorer_path" \
      --data_dir "$data_dir" \
      --results_dir "$results_dir" \
      --num_samples 5000 \
      --max_depth 5 \
      --batch_size 16 \
      --generation_mode continuation

    echo "[$DONE/$TOTAL] DONE: ${combo}"
    echo "---"
  done
done

echo "=== ALL_DONE ==="
