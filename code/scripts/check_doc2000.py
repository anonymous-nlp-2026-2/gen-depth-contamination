import json, pandas as pd
BASE = "/root/autodl-tmp/gen-depth-contamination"

# Text
for depth in range(4):
    with open(f"{BASE}/data_exp016_qwen_base/depth_{depth}.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d['doc_id'] == 2000:
                print(f"\n--- Depth {depth} ---")
                print(d['text'][:400])
                break

# Features
feat = pd.read_csv(f"{BASE}/data_exp016_qwen_base/features.csv")
doc = feat[feat['doc_id'] == 2000].sort_values('depth')
print("\n--- Features ---")
for _, r in doc[doc['depth'] <= 3].iterrows():
    print(f"  d{int(r['depth'])}: TTR={r['ttr']:.4f} surp={r['surp_mean']:.4f} ent={r['freq_entropy']:.4f} bleu={r['self_bleu']:.6f} hapax={r['hapax_ratio']:.4f}")
