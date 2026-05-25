#!/usr/bin/env python3
"""Compute theta sensitivity for C4 and Wiki domains (DeBERTa vs OBD)."""

import json
import numpy as np
from pathlib import Path

ROOT = Path("/root/autodl-tmp/gen-depth-contamination")
PAIRS = ["0v1", "1v2", "2v3", "3v4", "4v5"]
THETAS = [0.55, 0.60, 0.625, 0.65, 0.70]
N_BOOT = 10000
SEED = 42


def bootstrap_ci_from_folds(per_fold, n_boot=N_BOOT, seed=SEED):
    """Compute bootstrap CI from per-fold AUC values."""
    rng = np.random.RandomState(seed)
    vals = np.array(per_fold)
    n = len(vals)
    boots = np.array([vals[rng.randint(0, n, n)].mean() for _ in range(n_boot)])
    return {
        "auc": round(float(vals.mean()), 4),
        "ci_lower": round(float(np.percentile(boots, 2.5)), 4),
        "ci_upper": round(float(np.percentile(boots, 97.5)), 4),
    }


def compute_kstar(pairwise, theta):
    """K* = largest k s.t. pair (k-1)v(k) has ci_lower > theta."""
    kstar = 0
    pivotal = None
    margin = None
    for i, pair_key in enumerate(PAIRS):
        ci_lo = pairwise[pair_key]["ci_lower"]
        if ci_lo > theta:
            kstar = i + 1
            pivotal = pair_key
            margin = round(ci_lo - theta, 4)
        else:
            break
    if kstar == 0:
        pivotal = PAIRS[0]
        margin = round(pairwise[PAIRS[0]]["ci_lower"] - theta, 4)
    return {
        "K_star": kstar,
        "pivotal_pair": pivotal,
        "auc": pairwise[pivotal]["auc"],
        "ci_lower": pairwise[pivotal]["ci_lower"],
        "ci_upper": pairwise[pivotal]["ci_upper"],
        "margin": margin,
    }


def process_domain(domain, deberta_path, obd_path, output_path):
    # Load DeBERTa data (has bootstrap CI)
    with open(deberta_path) as f:
        deb_raw = json.load(f)
    if "pairwise_auc" in deb_raw:
        deb_pairwise = {k: {"auc": v["auc"], "ci_lower": v["ci_lower"], "ci_upper": v["ci_upper"]}
                        for k, v in deb_raw["pairwise_auc"].items() if k in PAIRS}
    else:
        deb_pairwise = deb_raw

    # Load OBD data (per-fold) and compute bootstrap CI
    with open(obd_path) as f:
        obd_raw = json.load(f)
    obd_pairwise = {}
    for pair in PAIRS:
        if pair in obd_raw.get("per_fold", {}):
            obd_pairwise[pair] = bootstrap_ci_from_folds(obd_raw["per_fold"][pair])
        elif pair in obd_raw.get("mean", {}):
            obd_pairwise[pair] = {"auc": obd_raw["mean"][pair], "ci_lower": obd_raw["mean"][pair], "ci_upper": obd_raw["mean"][pair]}

    # Compute K* for each theta
    result = {
        "domain": domain,
        "thresholds": THETAS,
        "deberta": {},
        "obd": {},
        "deberta_pairwise": deb_pairwise,
        "obd_pairwise": obd_pairwise,
    }
    for theta in THETAS:
        key = str(theta)
        result["deberta"][key] = compute_kstar(deb_pairwise, theta)
        result["obd"][key] = compute_kstar(obd_pairwise, theta)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {output_path}")
    return result


def print_table(domain, result):
    print(f"\n### {domain} DeBERTa vs OBD")
    print("| θ | DeBERTa K* | OBD K* | ΔK* | DeBERTa pivotal (margin) | OBD pivotal (margin) |")
    print("|---|-----------|--------|-----|--------------------------|----------------------|")
    for theta in THETAS:
        key = str(theta)
        d = result["deberta"][key]
        o = result["obd"][key]
        delta = d["K_star"] - o["K_star"]
        d_info = f"{d['pivotal_pair']} ({d['margin']:+.4f})"
        o_info = f"{o['pivotal_pair']} ({o['margin']:+.4f})"
        print(f"| {theta} | {d['K_star']} | {o['K_star']} | {delta:+d} | {d_info} | {o_info} |")

    print(f"\nDeBERTa pairwise AUC [ci_lower, ci_upper]:")
    for pair in PAIRS:
        p = result["deberta_pairwise"][pair]
        print(f"  {pair}: {p['auc']:.4f} [{p['ci_lower']:.4f}, {p['ci_upper']:.4f}]")

    print(f"OBD pairwise AUC [ci_lower, ci_upper]:")
    for pair in PAIRS:
        p = result["obd_pairwise"][pair]
        print(f"  {pair}: {p['auc']:.4f} [{p['ci_lower']:.4f}, {p['ci_upper']:.4f}]")


if __name__ == "__main__":
    # C4 (Qwen-1.5B)
    c4_result = process_domain(
        "c4",
        ROOT / "results" / "exp_deberta_c4_1b5" / "qwen_cont256.json",
        ROOT / "results" / "exp_021_length_256" / "pairwise_auc.json",
        ROOT / "results" / "exp_deberta_c4_1b5" / "theta_sensitivity.json",
    )
    print_table("C4", c4_result)

    # Wiki (Pythia)
    wiki_result = process_domain(
        "wiki",
        ROOT / "results" / "exp_deberta_upperbound" / "wiki_pythia.json",
        ROOT / "results" / "exp_023_wiki" / "pairwise_auc.json",
        ROOT / "results" / "exp_deberta_upperbound" / "theta_sensitivity_wiki_pythia.json",
    )
    print_table("Wiki", wiki_result)
