"""Download Wikipedia seed data for exp-023 domain ablation."""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/root/autodl-tmp/.hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from transformers import AutoTokenizer
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained(
    '/root/autodl-tmp/.hf_cache/Qwen/Qwen2.5-1.5B', trust_remote_code=True
)

out_dir = Path('/root/autodl-tmp/gen-depth-contamination/data/exp_023_wiki')
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "depth_0.jsonl"

num_samples = 5000
max_tokens = 256
min_text_len = 500  # skip stubs

print("Loading Wikipedia dataset (streaming)...", flush=True)
try:
    ds = load_dataset('wikimedia/wikipedia', '20231101.en', split='train', streaming=True)
except Exception as e:
    print(f"wikimedia/wikipedia failed: {e}", flush=True)
    print("Trying fallback: wikipedia 20220301.en...", flush=True)
    ds = load_dataset('wikipedia', '20220301.en', split='train', streaming=True)

records = []
scanned = 0
for example in ds:
    scanned += 1
    if len(records) >= num_samples:
        break
    text = example['text'].strip()
    if len(text) < min_text_len:
        continue
    ids = tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
    truncated = tokenizer.decode(ids, skip_special_tokens=True)
    records.append({'text': truncated, 'doc_id': len(records)})
    if len(records) % 500 == 0:
        print(f"  Collected {len(records)}/{num_samples} (scanned {scanned})", flush=True)

with open(out_path, 'w') as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print(f"DONE: {len(records)} docs saved to {out_path}")
print(f"Scanned {scanned} articles total, skipped {scanned - len(records)} short ones")
print(f"File size: {out_path.stat().st_size / 1024:.1f} KB")
