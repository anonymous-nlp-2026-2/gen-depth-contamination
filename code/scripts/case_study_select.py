import pandas as pd
import json
import numpy as np

BASE = "/root/autodl-tmp/gen-depth-contamination"
key_features = ['surp_mean', 'surp_std', 'ttr', 'hapax_ratio', 'freq_entropy', 'self_bleu']

def load_and_pivot(path):
    df = pd.read_csv(path)
    df = df[df['depth'] <= 3]
    return df

def compute_deltas_vectorized(df, features):
    pivoted = {}
    for d in range(4):
        sub = df[df['depth'] == d].set_index('doc_id')[features]
        pivoted[d] = sub
    
    common_ids = pivoted[0].index
    for d in range(1, 4):
        common_ids = common_ids.intersection(pivoted[d].index)
    
    for d in range(4):
        pivoted[d] = pivoted[d].loc[common_ids]
    
    deltas = {}
    for d in range(3):
        diff = pivoted[d+1].values - pivoted[d].values
        deltas[f'd{d}_{d+1}'] = np.sqrt(np.sum(diff**2, axis=1))
    
    result = pd.DataFrame({'doc_id': common_ids})
    for k, v in deltas.items():
        result[k] = v
    
    # Add per-depth feature values for key display
    for d in range(4):
        for f in ['ttr', 'surp_mean', 'freq_entropy', 'self_bleu']:
            result[f'{f}_d{d}'] = pivoted[d][f].values
    
    result['ratio_01_vs_23'] = result['d0_1'] / result['d2_3'].clip(lower=1e-10)
    return result

print("Loading C4...")
c4_feat = load_and_pivot(f"{BASE}/data_exp016_qwen_base/features.csv")
print(f"C4 rows: {len(c4_feat)}")
c4_deltas = compute_deltas_vectorized(c4_feat, key_features)

print("Loading arXiv...")
arxiv_feat = load_and_pivot(f"{BASE}/data/exp_024_arxiv/features.csv")
arxiv_deltas = compute_deltas_vectorized(arxiv_feat, key_features)

c4_deltas = c4_deltas.dropna()
arxiv_deltas = arxiv_deltas.dropna()

# Case 1: Typical K*=2 (C4) - high ratio
p80 = c4_deltas['ratio_01_vs_23'].quantile(0.80)
p95 = c4_deltas['ratio_01_vs_23'].quantile(0.95)
pool1 = c4_deltas[(c4_deltas['ratio_01_vs_23'] >= p80) & (c4_deltas['ratio_01_vs_23'] <= p95)]
case1 = pool1.iloc[len(pool1)//2]
print(f"\nCase 1 (Typical K*=2): doc={int(case1['doc_id'])}, d01={case1['d0_1']:.3f}, d12={case1['d1_2']:.3f}, d23={case1['d2_3']:.3f}, ratio={case1['ratio_01_vs_23']:.1f}")

# Case 2: Borderline - d1_2 still notable
c4_deltas['d12_rel'] = c4_deltas['d1_2'] / c4_deltas['d0_1'].clip(lower=1e-10)
pool2 = c4_deltas[(c4_deltas['d12_rel'] > 0.4) & (c4_deltas['d12_rel'] < 0.7) & (c4_deltas['d0_1'] > c4_deltas['d0_1'].quantile(0.3))]
if len(pool2) == 0:
    pool2 = c4_deltas.sort_values('d12_rel', ascending=False).head(200)
case2 = pool2.iloc[len(pool2)//2]
print(f"Case 2 (Borderline): doc={int(case2['doc_id'])}, d01={case2['d0_1']:.3f}, d12={case2['d1_2']:.3f}, d23={case2['d2_3']:.3f}")

# Case 3: arXiv K*=3
arxiv_deltas['d23_rel'] = arxiv_deltas['d2_3'] / arxiv_deltas['d0_1'].clip(lower=1e-10)
pool3 = arxiv_deltas[arxiv_deltas['d23_rel'] > arxiv_deltas['d23_rel'].quantile(0.7)]
case3 = pool3.iloc[len(pool3)//2]
print(f"Case 3 (arXiv K*=3): doc={int(case3['doc_id'])}, d01={case3['d0_1']:.3f}, d12={case3['d1_2']:.3f}, d23={case3['d2_3']:.3f}")

selected = {
    'case1_typical': int(case1['doc_id']),
    'case2_borderline': int(case2['doc_id']),
    'case3_arxiv': int(case3['doc_id']),
}

# Print feature values
for name, did in selected.items():
    df = arxiv_feat if 'arxiv' in name else c4_feat
    doc = df[df['doc_id'] == did].sort_values('depth')
    print(f"\n--- {name} (doc_id={did}) ---")
    for _, r in doc.iterrows():
        print(f"  d{int(r['depth'])}: TTR={r['ttr']:.4f} surp={r['surp_mean']:.4f} ent={r['freq_entropy']:.4f} bleu={r['self_bleu']:.6f}")

with open(f"{BASE}/scripts/selected_cases.json", 'w') as f:
    json.dump(selected, f, indent=2)
print("\nDone.")
