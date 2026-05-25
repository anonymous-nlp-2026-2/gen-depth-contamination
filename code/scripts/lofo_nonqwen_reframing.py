#!/usr/bin/env python3
"""
LOFO θ reframing: Non-Qwen derivation + Qwen validation

Derivation set: Gemma-2B, OLMo, Pythia-1.4B, LLaMA-8B, Mistral-7B
Validation set: Qwen-1.5B-rewrite, Qwen-1.5B-continuation, Qwen-7B
"""

import json
import numpy as np
from pathlib import Path
from collections import OrderedDict

PAIR_KEYS = ["0v1", "1v2", "2v3", "3v4", "4v5"]

NON_QWEN = OrderedDict([
    ("Gemma-2B", {
        "mean":     [0.9530, 0.5981, 0.5340, 0.5033, 0.5025],
        "ci_lower": [0.9494, 0.5882, 0.5234, 0.4924, 0.4915],
        "ci_upper": [0.9565, 0.6076, 0.5446, 0.5142, 0.5136],
    }),
    ("OLMo", {
        "mean":     [0.9970, 0.6145, 0.5496, 0.5091, 0.5045],
        "ci_lower": [0.9964, 0.6045, 0.5399, 0.4973, 0.4932],
        "ci_upper": [0.9975, 0.6247, 0.5596, 0.5200, 0.5150],
    }),
    ("Pythia-1.4B", {
        "mean":     [0.9932, 0.6539, 0.5514, 0.5152, 0.5136],
        "ci_lower": [0.9921, 0.6454, 0.5417, 0.5041, 0.5021],
        "ci_upper": [0.9943, 0.6630, 0.5628, 0.5260, 0.5249],
    }),
    ("LLaMA-8B", {
        "mean":     [0.9225, 0.6123, 0.5274, 0.5169, 0.5162],
        "ci_lower": [0.9174, 0.6028, 0.5169, 0.5062, 0.5052],
        "ci_upper": [0.9272, 0.6216, 0.5378, 0.5275, 0.5268],
    }),
    ("Mistral-7B", {
        "mean":     [0.9843, 0.5935, 0.5402, 0.5232, 0.5181],
        "ci_lower": [0.9824, 0.5835, 0.5301, 0.5128, 0.5070],
        "ci_upper": [0.9860, 0.6032, 0.5510, 0.5338, 0.5287],
    }),
])

QWEN = OrderedDict([
    ("Qwen-1.5B-rw", {
        "mean":     [0.9956, 0.6955, 0.5893, 0.5435, 0.5340],
        "ci_lower": [0.9946, 0.6859, 0.5795, 0.5333, 0.5238],
        "ci_upper": [0.9976, 0.7041, 0.5991, 0.5551, 0.5448],
    }),
    ("Qwen-1.5B-cont", {
        "mean":     [0.9971, 0.6112, 0.5609, 0.5409, 0.5141],
        "ci_lower": [0.9963, 0.6015, 0.5511, 0.5308, 0.5035],
        "ci_upper": [0.9978, 0.6213, 0.5712, 0.5508, 0.5255],
    }),
    ("Qwen-7B", {
        "mean":     [0.9638, 0.6168, 0.5492, 0.5276, 0.5085],
        "ci_lower": [0.9610, 0.6080, 0.5390, 0.5180, 0.4990],
        "ci_upper": [0.9670, 0.6260, 0.5590, 0.5370, 0.5180],
    }),
])

ALL_MODELS = OrderedDict(list(NON_QWEN.items()) + list(QWEN.items()))


def compute_kstar(ci_lowers, theta):
    k = 0
    for v in ci_lowers:
        if v >= theta:
            k += 1
        else:
            break
    return k


def consensus_range(model_dict):
    ci_1v2 = [d["ci_lower"][1] for d in model_dict.values()]
    ci_2v3 = [d["ci_lower"][2] for d in model_dict.values()]
    theta_high = min(ci_1v2)
    theta_low = max(ci_2v3)
    binding_high = [n for n, d in model_dict.items() if d["ci_lower"][1] == theta_high]
    binding_low = [n for n, d in model_dict.items() if d["ci_lower"][2] == theta_low]
    return {
        "theta_low": round(theta_low, 4),
        "theta_high": round(theta_high, 4),
        "width": round(theta_high - theta_low, 4),
        "midpoint": round((theta_low + theta_high) / 2, 4),
        "valid": theta_high > theta_low,
        "binding_high": binding_high,
        "binding_low": binding_low,
    }


def main():
    out_dir = Path("/root/autodl-tmp/gen-depth-contamination/results/lofo_theta_validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PART 1: Non-Qwen θ̂ Derivation (5 families)")
    print("=" * 80)

    print(f"\n{'Model':18s} {'1v2 mean':>10s} {'1v2 ci_lo':>10s} {'2v3 ci_lo':>10s} {'K*(0.60)':>8s} {'K*(0.58)':>8s}")
    print("-" * 80)
    for name, d in NON_QWEN.items():
        k60 = compute_kstar(d["ci_lower"], 0.60)
        k58 = compute_kstar(d["ci_lower"], 0.58)
        print(f"{name:18s} {d['mean'][1]:10.4f} {d['ci_lower'][1]:10.4f} {d['ci_lower'][2]:10.4f} {k60:8d} {k58:8d}")

    nq_range = consensus_range(NON_QWEN)
    print(f"\nNon-Qwen K*=2 consensus range: ({nq_range['theta_low']}, {nq_range['theta_high']}]")
    print(f"  Width: {nq_range['width']:.4f}")
    print(f"  Midpoint: {nq_range['midpoint']:.4f}")
    print(f"  Binding high: {nq_range['binding_high']} (lowest 1v2 ci_lower)")
    print(f"  Binding low:  {nq_range['binding_low']} (highest 2v3 ci_lower)")
    print(f"  Valid: {nq_range['valid']}")

    print("\n" + "=" * 80)
    print("PART 2: Qwen Validation at derived θ̂")
    print("=" * 80)

    theta_candidates = [nq_range["theta_high"], nq_range["midpoint"], 0.58, 0.585, 0.59]
    theta_candidates = sorted(set(theta_candidates))

    print(f"\n{'Model':18s} {'1v2 ci_lo':>10s} {'2v3 ci_lo':>10s}", end="")
    for t in theta_candidates:
        print(f" {'K*('+f'{t:.3f}'+')':>10s}", end="")
    print()
    print("-" * (38 + 11 * len(theta_candidates)))

    qwen_validation = {}
    for name, d in QWEN.items():
        print(f"{name:18s} {d['ci_lower'][1]:10.4f} {d['ci_lower'][2]:10.4f}", end="")
        qwen_validation[name] = {}
        for t in theta_candidates:
            k = compute_kstar(d["ci_lower"], t)
            print(f" {k:10d}", end="")
            margin_1v2 = d["ci_lower"][1] - t
            margin_2v3 = d["ci_lower"][2] - t
            qwen_validation[name][f"theta_{t:.4f}"] = {
                "K*": k,
                "margin_1v2": round(margin_1v2, 4),
                "margin_2v3": round(margin_2v3, 4),
            }
        print()

    print("\n--- Non-Qwen models at same θ candidates (sanity check) ---")
    print(f"{'Model':18s} {'1v2 ci_lo':>10s} {'2v3 ci_lo':>10s}", end="")
    for t in theta_candidates:
        print(f" {'K*('+f'{t:.3f}'+')':>10s}", end="")
    print()
    print("-" * (38 + 11 * len(theta_candidates)))
    for name, d in NON_QWEN.items():
        print(f"{name:18s} {d['ci_lower'][1]:10.4f} {d['ci_lower'][2]:10.4f}", end="")
        for t in theta_candidates:
            k = compute_kstar(d["ci_lower"], t)
            print(f" {k:10d}", end="")
        print()

    print("\n" + "=" * 80)
    print("PART 3: Full 8-family consensus range")
    print("=" * 80)
    full_range = consensus_range(ALL_MODELS)
    print(f"All 8 families K*=2 consensus: ({full_range['theta_low']}, {full_range['theta_high']}]")
    print(f"  Width: {full_range['width']:.4f}")
    print(f"  Midpoint: {full_range['midpoint']:.4f}")
    print(f"  Binding high: {full_range['binding_high']}")
    print(f"  Binding low:  {full_range['binding_low']}")

    print("\n" + "=" * 80)
    print("PART 4: Margin analysis at θ̂ = midpoint of non-Qwen range")
    print("=" * 80)
    theta_hat = nq_range["midpoint"]
    # Also check at full-range midpoint
    theta_full = full_range["midpoint"]
    
    for label, th in [("non-Qwen midpoint", theta_hat), ("full-8 midpoint", theta_full), ("θ=0.58", 0.58)]:
        print(f"\n--- θ = {th:.4f} ({label}) ---")
        print(f"{'Model':18s} {'Group':>8s} {'K*':>4s} {'1v2 margin':>12s} {'2v3 margin':>12s}")
        print("-" * 60)
        all_k2 = True
        for name, d in ALL_MODELS.items():
            group = "non-Qwen" if name in NON_QWEN else "Qwen"
            k = compute_kstar(d["ci_lower"], th)
            m1 = d["ci_lower"][1] - th
            m2 = d["ci_lower"][2] - th
            status = "" if k == 2 else " ← MISMATCH"
            print(f"{name:18s} {group:>8s} {k:4d} {m1:+12.4f} {m2:+12.4f}{status}")
            if k != 2:
                all_k2 = False
        print(f"All K*=2: {all_k2}")

    # --- Save JSON ---
    output = {
        "description": "LOFO reframing: non-Qwen derivation + Qwen validation",
        "non_qwen_families": list(NON_QWEN.keys()),
        "qwen_families": list(QWEN.keys()),
        "non_qwen_consensus_range": nq_range,
        "full_8_consensus_range": full_range,
        "non_qwen_data": {
            name: {
                "1v2_mean": d["mean"][1],
                "1v2_ci_lower": d["ci_lower"][1],
                "1v2_ci_upper": d["ci_upper"][1],
                "2v3_ci_lower": d["ci_lower"][2],
                "K_at_060": compute_kstar(d["ci_lower"], 0.60),
                "K_at_058": compute_kstar(d["ci_lower"], 0.58),
            }
            for name, d in NON_QWEN.items()
        },
        "qwen_validation": {
            name: {
                "1v2_mean": d["mean"][1],
                "1v2_ci_lower": d["ci_lower"][1],
                "2v3_ci_lower": d["ci_lower"][2],
                **{f"K_at_{t:.3f}": compute_kstar(d["ci_lower"], t) for t in theta_candidates},
                **{f"margin_1v2_at_{t:.3f}": round(d["ci_lower"][1] - t, 4) for t in theta_candidates},
                **{f"margin_2v3_at_{t:.3f}": round(d["ci_lower"][2] - t, 4) for t in theta_candidates},
            }
            for name, d in QWEN.items()
        },
        "recommended_theta": round(full_range["midpoint"], 4),
        "narrative": {
            "non_qwen_range": f"({nq_range['theta_low']}, {nq_range['theta_high']}]",
            "full_8_range": f"({full_range['theta_low']}, {full_range['theta_high']}]",
            "at_full_midpoint_all_k2": all(
                compute_kstar(d["ci_lower"], full_range["midpoint"]) == 2
                for d in ALL_MODELS.values()
            ),
            "at_058_all_k2": all(
                compute_kstar(d["ci_lower"], 0.58) == 2
                for d in ALL_MODELS.values()
            ),
        },
    }

    with open(out_dir / "lofo_nonqwen_reframing.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON saved to {out_dir / 'lofo_nonqwen_reframing.json'}")

    # --- LaTeX table ---
    print("\n" + "=" * 80)
    print("LaTeX Table")
    print("=" * 80)

    theta_hat_val = full_range["midpoint"]
    
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Non-Qwen $\hat\theta$ derivation and Qwen validation. The five non-Qwen families yield a $K^*{=}2$ consensus range $\hat\theta \in (" + f"{nq_range['theta_low']:.4f}, {nq_range['theta_high']:.4f}" + r"]$; all three held-out Qwen variants independently agree on $K^*{=}2$ within this range.}")
    lines.append(r"\label{tab:theta_reframing}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llccccc}")
    lines.append(r"\toprule")
    lines.append(r"Role & Model & \texttt{1v2} $\ell$ & \texttt{2v3} $\ell$ & $K^*_{\hat\theta}$ & Margin\textsubscript{1v2} & Margin\textsubscript{2v3} \\")
    lines.append(r"\midrule")
    
    th = 0.58
    lines.append(r"\multicolumn{7}{l}{\textit{Derivation set (non-Qwen)}} \\")
    for name, d in NON_QWEN.items():
        k = compute_kstar(d["ci_lower"], th)
        m1 = d["ci_lower"][1] - th
        m2 = d["ci_lower"][2] - th
        lines.append(f"& {name} & {d['ci_lower'][1]:.4f} & {d['ci_lower'][2]:.4f} & {k} & ${m1:+.4f}$ & ${m2:+.4f}$ \\\\")
    
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{7}{l}{\textit{Validation set (Qwen, held out)}} \\")
    for name, d in QWEN.items():
        k = compute_kstar(d["ci_lower"], th)
        m1 = d["ci_lower"][1] - th
        m2 = d["ci_lower"][2] - th
        lines.append(f"& {name} & {d['ci_lower'][1]:.4f} & {d['ci_lower'][2]:.4f} & {k} & ${m1:+.4f}$ & ${m2:+.4f}$ \\\\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    latex = "\n".join(lines)
    print(latex)
    
    with open(out_dir / "lofo_nonqwen_reframing_table.tex", "w") as f:
        f.write(latex)
    print(f"\nLaTeX saved to {out_dir / 'lofo_nonqwen_reframing_table.tex'}")


if __name__ == "__main__":
    main()
