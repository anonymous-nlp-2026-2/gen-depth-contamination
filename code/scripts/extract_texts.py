import json

BASE = "/root/autodl-tmp/gen-depth-contamination"

def get_text(path, doc_id, max_words=120):
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d['doc_id'] == doc_id:
                words = d['text'].split()[:max_words]
                return ' '.join(words)
    return None

cases = {
    'case1_c4_typical': {'doc_id': 2468, 'base': f'{BASE}/data_exp016_qwen_base'},
    'case2_c4_borderline': {'doc_id': 2403, 'base': f'{BASE}/data_exp016_qwen_base'},
    'case3_arxiv': {'doc_id': 1079, 'base': f'{BASE}/data/exp_024_arxiv'},
}

for case_name, info in cases.items():
    print(f"\n{'='*60}")
    print(f"  {case_name} (doc_id={info['doc_id']})")
    print(f"{'='*60}")
    for depth in range(4):
        path = f"{info['base']}/depth_{depth}.jsonl"
        text = get_text(path, info['doc_id'])
        if text:
            print(f"\n--- Depth {depth} ---")
            print(text[:500])
        else:
            print(f"\n--- Depth {depth}: NOT FOUND ---")
