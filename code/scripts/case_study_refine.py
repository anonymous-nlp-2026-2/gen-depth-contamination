import json
BASE = "/root/autodl-tmp/gen-depth-contamination"

# Check a few more C4 doc_ids with interesting depth 0 content
# Look for docs where depth 0 has distinctive style (news, informal, etc.)
candidates = [100, 500, 1000, 1500, 2000, 3000, 4000]

for did in candidates:
    with open(f"{BASE}/data_exp016_qwen_base/depth_0.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d['doc_id'] == did:
                text = d['text'][:200]
                print(f"doc_id={did}: {text[:150]}...")
                break
