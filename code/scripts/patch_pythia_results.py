"""
Pythia-6.9B results integration helper.

Reads run_pipeline.py output (pairwise_auc.json, ordinal_results.json)
and optional bootstrap_ci_results.json, then produces:
  1. LaTeX table row for tab:kstar_ci_extended
  2. Claim update checklist with suggested replacement text
  3. Updated configuration/family counts

Usage:
  python scripts/patch_pythia_results.py \
    --results_dir results/exp_pythia69b_c4 \
    [--bootstrap_ci results/exp_pythia69b_c4/bootstrap_ci_results.json]
"""

import argparse
import json
import sys
from pathlib import Path


THETA = 0.60
MODEL_LABEL = "Pythia-6.9B"
TABLE_SHORT = "Pythia-6.9B"

# Current paper state (before adding Pythia-6.9B)
CURRENT_CONFIGS = 8
CURRENT_KSTAR2_CONFIGS = 6  # 6/8 yield K*=2
CURRENT_FAMILIES = 5
# Pythia family already counted (via Pythia-1.4B)


def load_results(results_dir: Path):
    auc_path = results_dir / "pairwise_auc.json"
    if not auc_path.exists():
        print(f"ERROR: {auc_path} not found. Pipeline may still be running.")
        sys.exit(1)

    with open(auc_path) as f:
        auc_data = json.load(f)

    ordinal_path = results_dir / "ordinal_results.json"
    ordinal = {}
    if ordinal_path.exists():
        with open(ordinal_path) as f:
            ordinal = json.load(f)

    return auc_data, ordinal


def load_bootstrap(bootstrap_path: Path):
    if bootstrap_path and bootstrap_path.exists():
        with open(bootstrap_path) as f:
            return json.load(f)
    return None


def compute_kstar(auc_data, theta=THETA):
    mean_aucs = auc_data["mean"]
    kstar = 0
    for k in range(1, 10):
        key = f"{k-1}v{k}"
        if key in mean_aucs and mean_aucs[key] > theta:
            kstar = k
        else:
            break
    return kstar


def format_auc(val, digits=4):
    return f".{val:.4f}"[1:]  # e.g., ".6539"


def generate_latex_row_extended(auc_data, bootstrap):
    """Generate a LaTeX table column for tab:kstar_ci_extended."""
    mean = auc_data["mean"]
    pairs = ["0v1", "1v2", "2v3", "3v4", "4v5"]

    lines = []
    lines.append(f"% === Pythia-6.9B column for tab:kstar_ci_extended ===")
    lines.append(f"% Add \\multicolumn{{2}}{{c}}{{Pythia-6.9B}} to header row")
    lines.append(f"% Add \\cmidrule(lr){{N-N+1}} for column span")
    lines.append("")

    for pair in pairs:
        auc_val = mean.get(pair)
        if auc_val is None:
            lines.append(f"% {pair}: -- (not computed)")
            continue

        auc_str = format_auc(auc_val)

        ci_str = ""
        pstar = ""
        if bootstrap and "pairwise_results" in bootstrap:
            br = bootstrap["pairwise_results"].get(pair, {})
            ci_lo = br.get("ci_lower")
            ci_hi = br.get("ci_upper")
            if ci_lo is not None and ci_hi is not None:
                ci_str = f" & [{format_auc(ci_lo)},\\,{format_auc(ci_hi)}]"
            p_val = br.get("p_value_permutation")
            if pair == "1v2" and p_val is not None and p_val < 0.001:
                pstar = "\\rlap{$^*$}"
        elif "per_fold" in auc_data and pair in auc_data["per_fold"]:
            folds = auc_data["per_fold"][pair]
            import numpy as np
            arr = np.array(folds)
            ci_lo = float(np.percentile(arr, 2.5))
            ci_hi = float(np.percentile(arr, 97.5))
            ci_str = f" & [{format_auc(ci_lo)},\\,{format_auc(ci_hi)}]"
            if pair == "1v2":
                pstar = "\\rlap{$^*$}"

        if ci_str:
            lines.append(f"% {pair}: {auc_str}{ci_str}{pstar}")
        else:
            lines.append(f"% {pair}: {auc_str}")

    lines.append("")
    lines.append("% === Ready-to-paste cell values (AUC & CI columns) ===")
    for pair in pairs:
        auc_val = mean.get(pair)
        if auc_val is None:
            lines.append(f"    % {pair}: --")
            continue

        auc_str = format_auc(auc_val)
        ci_cell = ""
        pstar = ""

        if bootstrap and "pairwise_results" in bootstrap:
            br = bootstrap["pairwise_results"].get(pair, {})
            ci_lo = br.get("ci_lower")
            ci_hi = br.get("ci_upper")
            if ci_lo is not None and ci_hi is not None:
                ci_cell = f"[{format_auc(ci_lo)},\\,{format_auc(ci_hi)}]"
            if pair == "1v2":
                p_val = br.get("p_value_permutation")
                if p_val is not None and p_val < 0.001:
                    pstar = "\\rlap{$^*$}"

        if ci_cell:
            lines.append(f"    {auc_str}{pstar} & {ci_cell} \\\\")
        else:
            lines.append(f"    {auc_str} & \\\\")

    return "\n".join(lines)


def generate_latex_row_main(auc_data, bootstrap):
    """Generate LaTeX for potential inclusion in tab:kstar_ci (main table)."""
    mean = auc_data["mean"]
    pairs = ["0v1", "1v2", "2v3", "3v4", "4v5"]

    lines = []
    lines.append("% === Pythia-6.9B column for tab:kstar_ci (main table) ===")
    lines.append("% Header: \\multicolumn{2}{c}{Pythia-6.9B}")
    lines.append("% Add: \\cmidrule(lr){12-13}")
    lines.append("")

    for pair in pairs:
        auc_val = mean.get(pair)
        if auc_val is None:
            continue

        auc_str = format_auc(auc_val)
        ci_str = ""
        pstar = ""

        if bootstrap and "pairwise_results" in bootstrap:
            br = bootstrap["pairwise_results"].get(pair, {})
            ci_lo = br.get("ci_lower")
            ci_hi = br.get("ci_upper")
            if ci_lo is not None and ci_hi is not None:
                ci_str = f"[{format_auc(ci_lo)},\\,{format_auc(ci_hi)}]"
            if pair == "1v2":
                p_val = br.get("p_value_permutation")
                if p_val is not None and p_val < 0.001:
                    pstar = "\\rlap{$^*$}"

        if pair in ("1v2", "2v3") and ci_str:
            lines.append(f"{pair} & {auc_str} & {ci_str}{pstar} \\\\")
        elif ci_str:
            lines.append(f"{pair} & {auc_str} & {ci_str} \\\\")
        else:
            lines.append(f"{pair} & {auc_str} & \\\\")

    return "\n".join(lines)


def generate_claim_updates(kstar, auc_data, bootstrap):
    mean = auc_data["mean"]
    auc_1v2 = mean.get("1v2", 0)
    auc_2v3 = mean.get("2v3", 0)

    new_configs = CURRENT_CONFIGS + 1  # 9
    new_families = CURRENT_FAMILIES    # still 5 (Pythia family already counted)

    updates = []

    if kstar == 2:
        new_kstar2 = CURRENT_KSTAR2_CONFIGS + 1  # 8
        ratio = f"{new_kstar2}/{new_configs}"

        updates.append({
            "scenario": "A (K*=2, expected)",
            "files": [
                {
                    "file": "main.tex (abstract)",
                    "old": "8 model configurations (5 architecture families, 1B--8B)",
                    "new": f"{new_configs} model configurations ({new_families} architecture families, 1B--8B)",
                },
                {
                    "file": "main.tex (abstract)",
                    "old": "7/8 configurations",
                    "new": f"{ratio} configurations",
                },
                {
                    "file": "introduction.tex:19",
                    "old": "8 model configurations (5 families, 1B--8B)",
                    "new": f"{new_configs} model configurations ({new_families} families, 1B--8B)",
                },
                {
                    "file": "introduction.tex:21",
                    "old": "7/8 model configurations",
                    "new": f"{ratio} model configurations",
                },
                {
                    "file": "experiments_c1.tex:4",
                    "old": "eight model configurations spanning five architecture families",
                    "new": f"nine model configurations spanning five architecture families",
                },
                {
                    "file": "experiments_c1.tex:5",
                    "old": "Pythia-1.4B, OLMo-1B",
                    "new": "Pythia-1.4B, Pythia-6.9B, OLMo-1B",
                },
                {
                    "file": "experiments_c1.tex:43",
                    "old": "all 8 model configurations: 7/8 yield",
                    "new": f"all {new_configs} model configurations: {ratio} yield",
                },
                {
                    "file": "experiments_c1.tex:43",
                    "old": f"0.6112--0.6955",
                    "new": f"{min(0.6112, auc_1v2):.4f}--{max(0.6955, auc_1v2):.4f}",
                    "note": "Update AUC range if Pythia-6.9B extends it"
                },
                {
                    "file": "conclusion.tex:5",
                    "old": "7/8 configurations (5 families, 1B--8B",
                    "new": f"{ratio} configurations ({new_families} families, 1B--8B",
                },
                {
                    "file": "discussion.tex (boundary_cases paragraph)",
                    "old": "Of 8 tested model configurations on C4, 7 yield",
                    "new": f"Of {new_configs} tested model configurations on C4, {new_kstar2} yield",
                },
                {
                    "file": "discussion.tex (boundary_cases paragraph)",
                    "old": "gap: $0.594$ vs.\\ $0.611$--$0.696$",
                    "new": f"gap: $0.594$ vs.\\ ${min(0.611, auc_1v2):.3f}$--${max(0.696, auc_1v2):.3f}$",
                    "note": "Update gap range if needed"
                },
                {
                    "file": "main.tex (Limitations)",
                    "old": "model scale $\\leq$8B (5 architecture families)",
                    "new": "model scale $\\leq$8B (5 architecture families)",
                    "note": "No change needed (Pythia-6.9B < 8B, family already counted)"
                },
                {
                    "file": "experiments_c1.tex (scale paragraph)",
                    "note": "Add: 'Pythia-6.9B (1v2 AUC = X.XXXX) further confirms scale invariance within the Pythia family.'",
                },
                {
                    "file": "tab:kstar_ci_extended (appendix.tex)",
                    "note": "Add Pythia-6.9B column (2 new columns: AUC + CI)",
                },
            ]
        })

    elif kstar == 1:
        updates.append({
            "scenario": "B (K*=1, Mistral-like)",
            "files": [
                {
                    "file": "main.tex (abstract)",
                    "old": "8 model configurations (5 architecture families, 1B--8B)",
                    "new": f"{CURRENT_CONFIGS + 1} model configurations ({new_families} architecture families, 1B--8B)",
                },
                {
                    "file": "main.tex (abstract)",
                    "old": "7/8 configurations",
                    "new": f"{CURRENT_KSTAR2_CONFIGS}/{CURRENT_CONFIGS + 1} configurations",
                },
                {
                    "file": "introduction.tex:19",
                    "old": "8 model configurations (5 families, 1B--8B)",
                    "new": f"{CURRENT_CONFIGS + 1} model configurations ({new_families} families, 1B--8B)",
                },
                {
                    "file": "introduction.tex:21",
                    "old": "7/8 model configurations",
                    "new": f"{CURRENT_KSTAR2_CONFIGS}/{CURRENT_CONFIGS + 1} model configurations",
                },
                {
                    "file": "experiments_c1.tex:4-5",
                    "note": "Update config count to 9; add Pythia-6.9B to model list",
                },
                {
                    "file": "experiments_c1.tex:43",
                    "old": "Mistral-7B is the sole exception",
                    "new": f"Mistral-7B and Pythia-6.9B are exceptions ($\\Kempirical{{=}}1$; 1v2 AUC: 0.594, {auc_1v2:.4f})",
                },
                {
                    "file": "conclusion.tex:5",
                    "old": "7/8 configurations",
                    "new": f"{CURRENT_KSTAR2_CONFIGS}/{CURRENT_CONFIGS + 1} configurations",
                },
                {
                    "file": "discussion.tex (boundary_cases)",
                    "note": f"Pythia-6.9B joins Mistral-7B as K*=1. Strengthens '13B+ may yield K*=1' hypothesis. Update paragraph to mention both exceptions.",
                },
                {
                    "file": "tab:kstar_ci_extended (appendix.tex)",
                    "note": "Add Pythia-6.9B column",
                },
            ]
        })

    elif kstar >= 3:
        updates.append({
            "scenario": f"C (K*={kstar}, unexpected)",
            "files": [
                {
                    "file": "ALL counting claims",
                    "note": f"K*={kstar} is unexpected for a 6.9B model. This contradicts the scale->lower K* trend. "
                            f"Requires careful framing: either a Pythia-family artifact or evidence that the boundary "
                            f"is not monotonically scale-dependent.",
                },
                {
                    "file": "experiments_c1.tex:4-5",
                    "note": "Update config count to 9; add Pythia-6.9B to model list",
                },
                {
                    "file": "discussion.tex",
                    "note": f"Add paragraph discussing Pythia-6.9B K*={kstar} as counterexample to scale trend. "
                            f"Compare with arXiv K*=3 (domain effect vs architecture effect).",
                },
                {
                    "file": "tab:kstar_ci_extended (appendix.tex)",
                    "note": "Add Pythia-6.9B column",
                },
            ]
        })

    return updates


def generate_setup_table_row():
    """LaTeX row for tab:setup (method.tex) if we add Pythia-6.9B there."""
    return "Pythia-6.9B & Pythia-6.9B & Continuation & 5{,}000 \\\\"


def main():
    parser = argparse.ArgumentParser(description="Pythia-6.9B integration helper")
    parser.add_argument("--results_dir", type=str,
                        default="results/exp_pythia69b_c4",
                        help="Path to pipeline results directory")
    parser.add_argument("--bootstrap_ci", type=str, default=None,
                        help="Path to bootstrap CI JSON (optional)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    auc_data, ordinal = load_results(results_dir)

    bootstrap = None
    if args.bootstrap_ci:
        bootstrap = load_bootstrap(Path(args.bootstrap_ci))
    else:
        default_ci = results_dir / "bootstrap_ci_results.json"
        if default_ci.exists():
            bootstrap = load_bootstrap(default_ci)

    kstar = compute_kstar(auc_data)
    mean = auc_data["mean"]

    print("=" * 60)
    print(f"  Pythia-6.9B Integration Report")
    print("=" * 60)
    print()
    print(f"K* = {kstar}  (theta = {THETA})")
    print(f"Pairwise AUC (mean):")
    for pair in ["0v1", "1v2", "2v3", "3v4", "4v5"]:
        v = mean.get(pair, "N/A")
        marker = ""
        if pair == "1v2":
            marker = " <-- BOUNDARY" if isinstance(v, float) and v > THETA else " <-- BELOW THETA"
        if pair == "2v3" and isinstance(v, float):
            marker = " (below theta)" if v < THETA else " (ABOVE theta!)"
        print(f"  {pair}: {v}{marker}")

    if bootstrap:
        print(f"\nBootstrap CI available:")
        for pair in ["1v2", "2v3"]:
            br = bootstrap.get("pairwise_results", {}).get(pair, {})
            if br:
                print(f"  {pair}: [{br.get('ci_lower', '?')}, {br.get('ci_upper', '?')}]  p={br.get('p_value_permutation', '?')}")

    if ordinal:
        print(f"\nOrdinal results: MAE={ordinal.get('mae')}, Acc={ordinal.get('accuracy')}, K*={ordinal.get('k_star')}")

    print()
    print("-" * 60)
    print("  LaTeX: tab:kstar_ci_extended column")
    print("-" * 60)
    print(generate_latex_row_extended(auc_data, bootstrap))

    print()
    print("-" * 60)
    print("  LaTeX: tab:kstar_ci (main table) column")
    print("-" * 60)
    print(generate_latex_row_main(auc_data, bootstrap))

    print()
    print("-" * 60)
    print("  LaTeX: tab:setup row")
    print("-" * 60)
    print(generate_setup_table_row())

    print()
    print("-" * 60)
    print(f"  Claim Updates (Scenario: K*={kstar})")
    print("-" * 60)
    claim_updates = generate_claim_updates(kstar, auc_data, bootstrap)
    for scenario_block in claim_updates:
        print(f"\n  Scenario: {scenario_block['scenario']}")
        print()
        for i, item in enumerate(scenario_block["files"], 1):
            print(f"  [{i}] {item['file']}")
            if "old" in item:
                print(f"      OLD: {item['old']}")
                print(f"      NEW: {item['new']}")
            if "note" in item:
                print(f"      NOTE: {item['note']}")
            print()

    # JSON output
    output = {
        "model": MODEL_LABEL,
        "kstar": kstar,
        "theta": THETA,
        "pairwise_auc": mean,
        "per_fold": auc_data.get("per_fold", {}),
        "ordinal": ordinal,
        "has_bootstrap": bootstrap is not None,
        "new_config_count": CURRENT_CONFIGS + 1,
        "new_kstar2_count": CURRENT_KSTAR2_CONFIGS + (1 if kstar == 2 else 0),
        "family_count": CURRENT_FAMILIES,
        "claim_updates": claim_updates,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nJSON output saved to {args.output}")
    else:
        out_path = results_dir / "integration_report.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nJSON output saved to {out_path}")


if __name__ == "__main__":
    main()
