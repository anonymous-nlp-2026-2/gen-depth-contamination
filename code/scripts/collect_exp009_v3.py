import json, os, glob
import numpy as np
from scipy import stats
from collections import defaultdict

base_dir = "/root/autodl-tmp/gen-depth-contamination/results/exp_009_dose_response"
artifact_dir = "/root/autodl-tmp/gen-depth-contamination/artifacts"
os.makedirs(artifact_dir, exist_ok=True)

strategies = ["no_filter", "binary_ours", "graduated"]
ratios = ["ratio_10", "ratio_30", "ratio_50"]
seeds = list(range(5))

# Collect all results
raw_data = []
missing = []
grouped = defaultdict(lambda: defaultdict(lambda: {"mmlu": [], "hellaswag": []}))

for ratio in ratios:
    ratio_dir = os.path.join(base_dir, ratio)
    for strategy in strategies:
        for seed in seeds:
            fname = f"{strategy}_seed{seed}_eval.json"
            fpath = os.path.join(ratio_dir, fname)
            if os.path.exists(fpath):
                with open(fpath) as fp:
                    data = json.load(fp)
                mmlu = data["scores"]["mmlu"]
                hellaswag = data["scores"]["hellaswag"]
                raw_data.append({
                    "ratio": ratio,
                    "strategy": strategy,
                    "seed": seed,
                    "mmlu": mmlu,
                    "hellaswag": hellaswag
                })
                grouped[ratio][strategy]["mmlu"].append(mmlu)
                grouped[ratio][strategy]["hellaswag"].append(hellaswag)
            else:
                missing.append(f"{ratio}/{fname}")

# Summary stats
summary = {}
for ratio in ratios:
    summary[ratio] = {}
    for strategy in strategies:
        vals = grouped[ratio][strategy]
        n = len(vals["mmlu"])
        summary[ratio][strategy] = {
            "n": n,
            "mmlu_mean": float(np.mean(vals["mmlu"])) if n > 0 else None,
            "mmlu_std": float(np.std(vals["mmlu"], ddof=1)) if n > 1 else None,
            "hellaswag_mean": float(np.mean(vals["hellaswag"])) if n > 0 else None,
            "hellaswag_std": float(np.std(vals["hellaswag"], ddof=1)) if n > 1 else None,
        }

# Paired t-tests (only if both have 5 seeds)
comparisons = [
    ("binary_ours", "no_filter"),
    ("graduated", "binary_ours"),
    ("graduated", "no_filter"),
]
ttest_results = {}
for ratio in ratios:
    ttest_results[ratio] = []
    for s1, s2 in comparisons:
        v1 = grouped[ratio][s1]
        v2 = grouped[ratio][s2]
        n1, n2 = len(v1["mmlu"]), len(v2["mmlu"])
        entry = {"comparison": f"{s1} vs {s2}", "n1": n1, "n2": n2}
        if n1 == 5 and n2 == 5:
            # Paired by seed
            mmlu_diff = [v1["mmlu"][i] - v2["mmlu"][i] for i in range(5)]
            hella_diff = [v1["hellaswag"][i] - v2["hellaswag"][i] for i in range(5)]
            t_mmlu, p_mmlu = stats.ttest_rel(v1["mmlu"], v2["mmlu"])
            t_hella, p_hella = stats.ttest_rel(v1["hellaswag"], v2["hellaswag"])
            entry.update({
                "delta_mmlu": float(np.mean(mmlu_diff)),
                "p_mmlu": float(p_mmlu),
                "t_mmlu": float(t_mmlu),
                "delta_hellaswag": float(np.mean(hella_diff)),
                "p_hellaswag": float(p_hella),
                "t_hellaswag": float(t_hella),
                "significant_mmlu_005": bool(p_mmlu < 0.05),
                "significant_hellaswag_005": bool(p_hella < 0.05),
            })
        else:
            entry["note"] = f"Incomplete data (n1={n1}, n2={n2}), t-test skipped"
        ttest_results[ratio].append(entry)

# Build full analysis JSON
analysis = {
    "experiment": "exp-009 v3 dose-response",
    "date": "2026-05-17",
    "total_files_expected": 45,
    "total_files_found": len(raw_data),
    "missing_files": missing,
    "raw_data": raw_data,
    "summary": summary,
    "paired_ttests": ttest_results,
}

out_path = os.path.join(artifact_dir, "exp009_v3_full_analysis.json")
with open(out_path, "w") as fp:
    json.dump(analysis, fp, indent=2)
print(f"Saved to {out_path}")

# Print human-readable tables
print(f"\n{'='*80}")
print(f"EXP-009 v3 DOSE-RESPONSE RESULTS")
print(f"Total: {len(raw_data)}/45 files collected. Missing: {len(missing)}")
if missing:
    print(f"Missing files: {missing}")
print(f"{'='*80}")

print(f"\n## Table 1: Mean±Std")
print(f"{'Ratio':<12} {'Strategy':<15} {'N':>3} {'MMLU':>18} {'HellaSwag':>18}")
print("-" * 70)
for ratio in ratios:
    for strategy in strategies:
        s = summary[ratio][strategy]
        n = s["n"]
        if n > 1:
            mmlu_str = f"{s['mmlu_mean']:.4f}±{s['mmlu_std']:.4f}"
            hella_str = f"{s['hellaswag_mean']:.4f}±{s['hellaswag_std']:.4f}"
        elif n == 1:
            mmlu_str = f"{s['mmlu_mean']:.4f} (n=1)"
            hella_str = f"{s['hellaswag_mean']:.4f} (n=1)"
        else:
            mmlu_str = "N/A"
            hella_str = "N/A"
        print(f"{ratio:<12} {strategy:<15} {n:>3} {mmlu_str:>18} {hella_str:>18}")

print(f"\n## Table 2: Paired t-tests")
print(f"{'Ratio':<12} {'Comparison':<30} {'Δ MMLU':>10} {'p(MMLU)':>10} {'Δ Hella':>10} {'p(Hella)':>10}")
print("-" * 85)
for ratio in ratios:
    for t in ttest_results[ratio]:
        if "delta_mmlu" in t:
            sig_m = "*" if t["significant_mmlu_005"] else ""
            sig_h = "*" if t["significant_hellaswag_005"] else ""
            print(f"{ratio:<12} {t['comparison']:<30} {t['delta_mmlu']:>+.4f}{sig_m:1} {t['p_mmlu']:>9.4f} {t['delta_hellaswag']:>+.4f}{sig_h:1} {t['p_hellaswag']:>9.4f}")
        else:
            print(f"{ratio:<12} {t['comparison']:<30} {'INCOMPLETE':>10} {'':>10} {'INCOMPLETE':>10} {'':>10}")

print(f"\n## Table 3: Raw Data")
print(f"{'Ratio':<12} {'Strategy':<15} {'Seed':>5} {'MMLU':>10} {'HellaSwag':>10}")
print("-" * 55)
for d in raw_data:
    print(f"{d['ratio']:<12} {d['strategy']:<15} {d['seed']:>5} {d['mmlu']:>10.4f} {d['hellaswag']:>10.4f}")

print(f"\n{'='*80}")
print("* = significant at p<0.05")
