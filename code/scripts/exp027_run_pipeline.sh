#!/bin/bash
set -e
cd /root/autodl-tmp/gen-depth-contamination
source /root/miniconda3/bin/activate

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/root/autodl-tmp/.hf_cache
export PYTHONHASHSEED=42
export CUDA_VISIBLE_DEVICES=3

echo "=== exp-027: LaTeX ablation pipeline ==="
echo "Start: $(date)"

python3 -c "
import sys, os
sys.path.insert(0, 'scripts')

import torch
print(f'CUDA visible: {torch.cuda.device_count()} GPUs')
print(f'GPU 0 name: {torch.cuda.get_device_name(0)}')
print(f'GPU 0 free: {torch.cuda.mem_get_info(0)[0]/1e9:.1f} GB')

from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = '/root/autodl-tmp/.hf_cache/Qwen/Qwen2.5-1.5B'
print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'left'

print('Loading model to cuda:0...')
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map={'': 'cuda:0'},
    trust_remote_code=True,
)
model.eval()
print(f'Model on device: {model.device}')

from pathlib import Path
from run_pipeline import extract_all_features, run_analysis

data_dir = Path('data/exp_027_latex_ablation')
results_dir = Path('results/exp_027_latex_ablation')
results_dir.mkdir(parents=True, exist_ok=True)

# Delete stale features.csv to force re-extraction
feat_path = data_dir / 'features.csv'
if feat_path.exists():
    feat_path.unlink()
    print('Deleted stale features.csv')

print('=== Step 4: Feature Extraction ===')
extract_all_features(model, tokenizer, data_dir, max_depth=5, batch_size=16)

del model
torch.cuda.empty_cache()

print('=== Step 5: Analysis ===')
run_analysis(data_dir, results_dir, max_depth=5)

print('=== DONE ===')
"

echo "End: $(date)"
