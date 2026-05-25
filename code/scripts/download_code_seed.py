import json, os, sys

os.environ['HF_HOME'] = '/root/autodl-tmp/.hf_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset

out_dir = '/root/autodl-tmp/gen-depth-contamination/data/code_python_seed'
os.makedirs(out_dir, exist_ok=True)

# code_search_net Python subset (non-gated)
print("Loading code_search_net python...", flush=True)
try:
    ds = load_dataset('code_search_net', 'python', split='train', streaming=True)
    first = next(iter(ds))
    print(f"  OK, keys: {list(first.keys())}", flush=True)
    # find text field
    text_field = None
    for f in ['whole_func_string', 'func_code_string', 'code', 'content', 'text']:
        if f in first and first[f] and len(first[f]) > 10:
            text_field = f
            break
    if text_field is None:
        print(f"  Trying all string fields...")
        for k, v in first.items():
            if isinstance(v, str) and len(v) > 50:
                text_field = k
                break
    print(f"  Using field: {text_field}", flush=True)
    print(f"  Sample length: {len(first[text_field])}", flush=True)
except Exception as e:
    print(f"code_search_net failed: {e}", flush=True)
    sys.exit(1)

count = 0
skipped = 0
with open(f'{out_dir}/depth_0.jsonl', 'w') as f:
    for item in ds:
        text = item.get(text_field, '')
        if len(text) < 200 or len(text) > 2000:
            skipped += 1
            continue
        if not text.isascii():
            skipped += 1
            continue
        f.write(json.dumps({'text': text, 'doc_id': f'code_py_{count}'}) + '\n')
        count += 1
        if count % 1000 == 0:
            print(f"  collected {count} docs (skipped {skipped})...", flush=True)
        if count >= 5500:
            break

print(f"Done! Saved {count} Python code documents (skipped {skipped})")
