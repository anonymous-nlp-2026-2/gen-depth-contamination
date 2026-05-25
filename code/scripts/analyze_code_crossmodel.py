#!/usr/bin/env python3
"""Analyze code-domain cross-model pairwise AUC results and generate LaTeX table + JSON."""

import json
import math
import os
import sys
from pathlib import Path

PROJ_ROOT = Path("/root/autodl-tmp/gen-depth-contamination")
RESULTS_DIR = PROJ_ROOT / "results"
ARTIFACTS_DIR = PROJ_ROOT / "artifacts"
THETA = 0.60

EXPERIMENTS = {
    "Qwen2.5-1.5B": "exp_code_qwen1b5",
    "Pythia-1.4B": "exp_code_pythia1b4",
    "Qwen2.5-7B": "exp_code_qwen7b",
    "Llama-3.1-8B": "exp_code_llama8b",
    "Mistral-7B-v0.3": "exp_code_mistral7b",
}

PAIRS = ["0v1", "1v2", "2v3", "3v4", "4v5"]


def load_experiment(exp_dir: Path) -> dict | None:
    auc_file = exp_dir / "pairwise_auc.json"
    if not auc_file.exists():
        return None
    with open(auc_file) as f:
        data = json.load(f)
    return data


def compute_ci(fold_values: list[float]) -> tuple[float, float, float]:
    n = len(fold_values)
    mean = sum(fold_values) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in fold_values) / (n - 1))
    ci = 1.96 * std / math.sqrt(n)
    return mean, std, ci


def determine_kstar(mean_aucs: dict[str, float], theta: float = THETA) -> int:
    kstar = 0
    for pair in PAIRS:
        k = int(pair.split("v")[1])
        if mean_aucs.get(pair, 0) > theta:
            kstar = max(kstar, k)
        else:
            break
    return kstar


def build_results(experiments: dict[str, str]) -> dict:
    all_results = {}
    for model_name, exp_name in experiments.items():
        exp_dir = RESULTS_DIR / exp_name
        data = load_experiment(exp_dir)
        if data is None:
            print(f"[WARN] {exp_name} not found or incomplete, skipping", file=sys.stderr)
            all_results[model_name] = None
            continue

        model_result = {"pairs": {}}
        for pair in PAIRS:
            folds = data["per_fold"][pair]
            mean, std, ci = compute_ci(folds)
            model_result["pairs"][pair] = {
                "mean": round(mean, 4),
                "std": round(std, 4),
                "ci": round(ci, 4),
                "folds": folds,
            }

        mean_aucs = {p: model_result["pairs"][p]["mean"] for p in PAIRS}
        model_result["kstar"] = determine_kstar(mean_aucs)
        all_results[model_name] = model_result

    return all_results


def generate_latex(all_results: dict) -> str:
    model_names = list(EXPERIMENTS.keys())
    n_models = len(model_names)

    lines = []
    lines.append("% Code Domain Cross-Model Pairwise AUC Results")
    lines.append(f"% Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"% Threshold theta = {THETA}")
    lines.append("")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Pairwise AUC ($k$ vs $k{+}1$) for code-domain contamination detection "
        "across five generator models trained on the CodeSearchNet Python subset "
        "\\citep{husain2019codesearchnet}. "
        "CI columns show 95\\% confidence intervals from 5-fold CV. "
        "Underlined values mark the $K^*$ boundary (last pair with AUC $> \\theta$).}"
    )
    lines.append("\\label{tab:code_crossmodel_auc}")
    lines.append("\\resizebox{\\columnwidth}{!}{%")

    col_spec = "l" + " cc" * n_models
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    header1_parts = [""]
    for name in model_names:
        header1_parts.append(f"\\multicolumn{{2}}{{c}}{{{name}}}")
    lines.append(" & ".join(header1_parts) + " \\\\")

    cmidrules = []
    for i in range(n_models):
        start = 2 + i * 2
        end = start + 1
        cmidrules.append(f"\\cmidrule(lr){{{start}-{end}}}")
    lines.append(" ".join(cmidrules))

    header2 = "Pair"
    for _ in model_names:
        header2 += " & AUC & CI"
    lines.append(header2 + " \\\\")
    lines.append("\\midrule")

    for pair in PAIRS:
        k_from, k_to = pair.split("v")
        row_label = f"${k_from}{{\\to}}{k_to}$"
        row_parts = [row_label]

        for model_name in model_names:
            result = all_results.get(model_name)
            if result is None:
                row_parts.extend(["TBD", ""])
            else:
                pair_data = result["pairs"][pair]
                kstar = result["kstar"]
                k = int(k_from)
                is_boundary = (k + 1 == kstar) and (pair_data["mean"] > THETA)

                auc_str = f"{pair_data['mean']:.4f}"
                ci_str = f"$\\pm${pair_data['ci']:.4f}"

                if is_boundary:
                    auc_str = f"\\underline{{{auc_str}}}"
                    ci_str = f"\\underline{{{ci_str}}}"

                row_parts.extend([auc_str, ci_str])

        lines.append(" & ".join(row_parts) + " \\\\")

    lines.append("\\midrule")

    kstar_parts = ["$K^*$"]
    for model_name in model_names:
        result = all_results.get(model_name)
        if result is None:
            kstar_parts.extend(["TBD", ""])
        else:
            kstar_parts.extend([str(result["kstar"]), ""])
    lines.append(" & ".join(kstar_parts) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}}")
    lines.append("")
    lines.append("\\vspace{2pt}")
    lines.append(
        f"{{\\footnotesize $\\theta = {THETA}$; $K^*$ = max $k$ s.t.\\ "
        f"AUC($k{{-}}1, k$) $> \\theta$. Domain: Code (CodeSearchNet Python subset). 5-fold CV.}}"
    )
    lines.append("\\end{table}")

    return "\n".join(lines)


def main():
    all_results = build_results(EXPERIMENTS)

    latex = generate_latex(all_results)
    print(latex)
    print()

    json_out = {}
    for model_name, result in all_results.items():
        if result is None:
            json_out[model_name] = None
        else:
            json_out[model_name] = {
                "kstar": result["kstar"],
                "pairs": {
                    p: {"mean": v["mean"], "ci": v["ci"]}
                    for p, v in result["pairs"].items()
                },
            }

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    tex_path = ARTIFACTS_DIR / "code_crossmodel_table.tex"
    json_path = ARTIFACTS_DIR / "code_crossmodel_results.json"

    with open(tex_path, "w") as f:
        f.write(latex + "\n")
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)

    print(f"[INFO] LaTeX table written to {tex_path}", file=sys.stderr)
    print(f"[INFO] JSON results written to {json_path}", file=sys.stderr)

    for model_name, result in all_results.items():
        if result is not None:
            print(f"[CHECK] {model_name}: K*={result['kstar']}, 0v1 AUC={result['pairs']['0v1']['mean']:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
