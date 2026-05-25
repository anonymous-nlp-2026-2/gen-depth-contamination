#!/usr/bin/env python3
"""Changepoint-based K* analysis: replace fixed theta with data-driven breakpoint detection."""

import json
import glob
import os
import numpy as np
import ruptures as rpt
from pathlib import Path
from collections import OrderedDict

RESULTS_ROOT = Path("/root/autodl-tmp/gen-depth-contamination/results")
OUTPUT_DIR = RESULTS_ROOT / "exp_changepoint_kstar"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THETA = 0.60
PENALTY_VALUES = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
METHODS = ["Pelt", "Binseg", "BottomUp"]


def load_all_experiments():
    """Load all pairwise_auc.json files."""
    experiments = {}
    for fp in sorted(glob.glob(str(RESULTS_ROOT / "*/pairwise_auc.json"))):
        exp_name = Path(fp).parent.name
        with open(fp) as f:
            data = json.load(f)
        pair_keys = sorted([k for k in data["mean"].keys() if "v" in k],
                           key=lambda x: int(x.split("v")[0]))
        mean_aucs = [data["mean"][k] for k in pair_keys]
        per_fold = {k: data["per_fold"][k] for k in pair_keys} if "per_fold" in data else None
        experiments[exp_name] = {
            "pair_keys": pair_keys,
            "mean_aucs": mean_aucs,
            "per_fold": per_fold,
            "n_pairs": len(pair_keys),
        }
    return experiments


def compute_kstar_theta(auc_list, theta=THETA):
    """K* via fixed threshold: consecutive pairs from start with AUC >= theta."""
    k = 0
    for auc in auc_list:
        if auc >= theta:
            k += 1
        else:
            break
    return k


def detect_changepoint(signal, method="Pelt", pen=None, model="l2", min_size=1):
    """Run changepoint detection, return list of breakpoint positions (excluding endpoint)."""
    signal = np.array(signal, dtype=float).reshape(-1, 1)
    n = len(signal)
    if n < 2:
        return []

    if method == "Pelt":
        algo = rpt.Pelt(model=model, min_size=min_size, jump=1)
    elif method == "Binseg":
        algo = rpt.Binseg(model=model, min_size=min_size, jump=1)
    elif method == "BottomUp":
        algo = rpt.BottomUp(model=model, min_size=min_size, jump=1)
    else:
        raise ValueError(f"Unknown method: {method}")

    algo.fit(signal)

    if pen is not None:
        bkps = algo.predict(pen=pen)
    else:
        bkps = algo.predict(pen=1.0)

    # ruptures returns breakpoints with the last being n; remove it
    bkps = [b for b in bkps if b < n]
    return bkps


def changepoint_to_kstar(bkps, auc_list):
    """Convert changepoint positions to K*.
    
    Logic:
    - If no changepoint: check if sequence is overall high or low
      - If mean AUC > 0.55 (above chance): K* = len(sequence) (all detectable)
      - Else: K* = 0
    - If changepoint(s): first changepoint where AUC drops = K*
      We take the first breakpoint where the segment before has higher mean
      than the segment after.
    """
    n = len(auc_list)
    if not bkps:
        mean_all = np.mean(auc_list)
        if mean_all > 0.55:
            return n
        else:
            return 0

    # For multiple breakpoints, find the first "drop" point
    prev_start = 0
    for bp in sorted(bkps):
        seg_before = np.mean(auc_list[prev_start:bp])
        seg_after_end = sorted(bkps)[sorted(bkps).index(bp) + 1] if bp != sorted(bkps)[-1] else n
        seg_after = np.mean(auc_list[bp:seg_after_end])
        if seg_before > seg_after:
            return bp
        prev_start = bp
    
    # No drop detected, all in one regime
    return n


def bootstrap_changepoint(per_fold, pair_keys, method="Pelt", pen=1.0, n_bootstrap=2000, seed=42):
    """Bootstrap changepoint: resample folds, compute mean AUC, detect changepoint, aggregate."""
    rng = np.random.RandomState(seed)
    n_folds = len(per_fold[pair_keys[0]])
    n_pairs = len(pair_keys)
    
    # Build fold matrix: (n_folds, n_pairs)
    fold_matrix = np.array([[per_fold[k][f] for k in pair_keys] for f in range(n_folds)])
    
    kstar_samples = []
    changepoint_positions = []  # raw positions
    
    for _ in range(n_bootstrap):
        idx = rng.choice(n_folds, size=n_folds, replace=True)
        boot_mean = fold_matrix[idx].mean(axis=0)
        bkps = detect_changepoint(boot_mean, method=method, pen=pen)
        ks = changepoint_to_kstar(bkps, boot_mean.tolist())
        kstar_samples.append(ks)
        changepoint_positions.append(bkps[0] if bkps else -1)
    
    kstar_samples = np.array(kstar_samples)
    changepoint_positions = np.array(changepoint_positions)
    
    return {
        "kstar_median": int(np.median(kstar_samples)),
        "kstar_mean": float(np.mean(kstar_samples)),
        "kstar_mode": int(np.bincount(kstar_samples.astype(int).clip(0, n_pairs)).argmax()),
        "kstar_distribution": {int(k): int(v) for k, v in 
                               zip(*np.unique(kstar_samples, return_counts=True))},
        "changepoint_distribution": {int(k): int(v) for k, v in 
                                     zip(*np.unique(changepoint_positions, return_counts=True))},
        "n_bootstrap": n_bootstrap,
    }


def run_analysis():
    experiments = load_all_experiments()
    print(f"Loaded {len(experiments)} experiments")

    # ---- 1. Main changepoint analysis (PELT, pen via BIC approximation) ----
    # For short signals, BIC penalty ≈ log(n) * sigma^2
    # We'll use pen=0.05 as primary (good for 5-point signals) and sweep others
    
    primary_pen = 0.05
    primary_method = "Pelt"
    
    changepoint_results = OrderedDict()
    
    for exp_name, exp_data in sorted(experiments.items()):
        aucs = exp_data["mean_aucs"]
        n = exp_data["n_pairs"]
        
        # K* via theta
        kstar_theta = compute_kstar_theta(aucs)
        
        # K* via changepoint (primary)
        bkps = detect_changepoint(aucs, method=primary_method, pen=primary_pen)
        kstar_cp = changepoint_to_kstar(bkps, aucs)
        
        # Bootstrap K* (if per_fold available)
        bootstrap_result = None
        if exp_data["per_fold"] is not None:
            bootstrap_result = bootstrap_changepoint(
                exp_data["per_fold"], exp_data["pair_keys"],
                method=primary_method, pen=primary_pen, n_bootstrap=2000
            )
        
        # Multi-method comparison
        method_results = {}
        for method in METHODS:
            bkps_m = detect_changepoint(aucs, method=method, pen=primary_pen)
            kstar_m = changepoint_to_kstar(bkps_m, aucs)
            method_results[method] = {
                "breakpoints": bkps_m,
                "kstar": kstar_m,
            }
        
        changepoint_results[exp_name] = {
            "auc_sequence": aucs,
            "n_pairs": n,
            "pair_keys": exp_data["pair_keys"],
            "kstar_theta_0.60": kstar_theta,
            "kstar_changepoint": kstar_cp,
            "breakpoints": bkps,
            "kstar_bootstrap": bootstrap_result,
            "method_comparison": method_results,
            "agreement_with_theta": kstar_cp == kstar_theta,
        }
        
        boot_str = ""
        if bootstrap_result:
            boot_str = f"  boot_median={bootstrap_result['kstar_median']}"
        print(f"  {exp_name:45s} AUC={[f'{a:.3f}' for a in aucs]}")
        print(f"    K*_theta={kstar_theta}  K*_cp={kstar_cp}  bkps={bkps}  "
              f"{'AGREE' if kstar_cp == kstar_theta else 'DISAGREE'}{boot_str}")
    
    # ---- 2. Agreement summary ----
    total = len(changepoint_results)
    agree = sum(1 for v in changepoint_results.values() if v["agreement_with_theta"])
    disagree_cases = {k: v for k, v in changepoint_results.items() if not v["agreement_with_theta"]}
    
    # Bootstrap agreement
    boot_agree = 0
    boot_total = 0
    for v in changepoint_results.values():
        if v["kstar_bootstrap"] is not None:
            boot_total += 1
            if v["kstar_bootstrap"]["kstar_median"] == v["kstar_theta_0.60"]:
                boot_agree += 1
    
    agreement_summary = {
        "total_experiments": total,
        "agreement_count": agree,
        "agreement_rate": round(agree / total, 4) if total > 0 else 0,
        "disagreement_count": total - agree,
        "disagreement_cases": {
            k: {
                "kstar_theta": v["kstar_theta_0.60"],
                "kstar_changepoint": v["kstar_changepoint"],
                "auc_sequence": v["auc_sequence"],
                "breakpoints": v["breakpoints"],
            }
            for k, v in disagree_cases.items()
        },
        "bootstrap_agreement": {
            "total": boot_total,
            "agree": boot_agree,
            "rate": round(boot_agree / boot_total, 4) if boot_total > 0 else 0,
        },
        "primary_method": primary_method,
        "primary_penalty": primary_pen,
    }
    
    print(f"\n=== Agreement Summary ===")
    print(f"Direct changepoint: {agree}/{total} = {agree/total:.1%}")
    print(f"Bootstrap median:   {boot_agree}/{boot_total} = {boot_agree/boot_total:.1%}" if boot_total > 0 else "")
    if disagree_cases:
        print(f"\nDisagreement cases:")
        for k, v in disagree_cases.items():
            print(f"  {k}: K*_theta={v['kstar_theta_0.60']}, K*_cp={v['kstar_changepoint']}, "
                  f"AUC={[f'{a:.3f}' for a in v['auc_sequence']]}")

    # ---- 3. Penalty sensitivity sweep ----
    print(f"\n=== Penalty Sensitivity ===")
    penalty_sensitivity = {}
    
    for exp_name, exp_data in sorted(experiments.items()):
        aucs = exp_data["mean_aucs"]
        kstar_theta = compute_kstar_theta(aucs)
        pen_results = {}
        
        for pen in PENALTY_VALUES:
            method_kstars = {}
            for method in METHODS:
                bkps = detect_changepoint(aucs, method=method, pen=pen)
                ks = changepoint_to_kstar(bkps, aucs)
                method_kstars[method] = {"kstar": ks, "breakpoints": bkps}
            pen_results[str(pen)] = method_kstars
        
        # Stability: fraction of (pen, method) combos agreeing with theta
        all_kstars = [pen_results[str(p)][m]["kstar"] 
                      for p in PENALTY_VALUES for m in METHODS]
        agree_rate = sum(1 for ks in all_kstars if ks == kstar_theta) / len(all_kstars)
        
        # Range of K* across all settings
        unique_kstars = sorted(set(all_kstars))
        
        penalty_sensitivity[exp_name] = {
            "kstar_theta": kstar_theta,
            "penalty_sweep": pen_results,
            "agreement_with_theta_rate": round(agree_rate, 4),
            "kstar_range": unique_kstars,
            "is_stable": len(unique_kstars) <= 2,
        }
        
        print(f"  {exp_name:45s} K*_theta={kstar_theta}  "
              f"range={unique_kstars}  agree={agree_rate:.0%}  "
              f"{'STABLE' if len(unique_kstars) <= 2 else 'UNSTABLE'}")
    
    # Overall stability
    stable_count = sum(1 for v in penalty_sensitivity.values() if v["is_stable"])
    
    # Agreement rate at each penalty value (PELT only)
    pen_agreement_rates = {}
    for pen in PENALTY_VALUES:
        agree_at_pen = sum(
            1 for exp in penalty_sensitivity.values()
            if exp["penalty_sweep"][str(pen)]["Pelt"]["kstar"] == exp["kstar_theta"]
        )
        pen_agreement_rates[str(pen)] = round(agree_at_pen / total, 4)
    
    penalty_summary = {
        "stable_experiments": stable_count,
        "total_experiments": total,
        "stability_rate": round(stable_count / total, 4),
        "pelt_agreement_by_penalty": pen_agreement_rates,
        "best_penalty": max(pen_agreement_rates, key=pen_agreement_rates.get),
        "best_agreement_rate": max(pen_agreement_rates.values()),
    }
    
    print(f"\nStability: {stable_count}/{total} experiments have K* range <= 2")
    print(f"PELT agreement by penalty: {json.dumps(pen_agreement_rates, indent=2)}")
    print(f"Best penalty: {penalty_summary['best_penalty']} "
          f"(agreement={penalty_summary['best_agreement_rate']:.1%})")

    # ---- 4. Bootstrap vs direct comparison ----
    print(f"\n=== Bootstrap vs Direct ===")
    bootstrap_comparison = {}
    for exp_name, result in changepoint_results.items():
        if result["kstar_bootstrap"] is None:
            continue
        boot = result["kstar_bootstrap"]
        bootstrap_comparison[exp_name] = {
            "kstar_theta": result["kstar_theta_0.60"],
            "kstar_direct_cp": result["kstar_changepoint"],
            "kstar_boot_median": boot["kstar_median"],
            "kstar_boot_mode": boot["kstar_mode"],
            "boot_distribution": boot["kstar_distribution"],
            "direct_agrees_theta": result["kstar_changepoint"] == result["kstar_theta_0.60"],
            "boot_median_agrees_theta": boot["kstar_median"] == result["kstar_theta_0.60"],
            "boot_mode_agrees_theta": boot["kstar_mode"] == result["kstar_theta_0.60"],
        }
        print(f"  {exp_name:45s} theta={result['kstar_theta_0.60']}  "
              f"direct={result['kstar_changepoint']}  "
              f"boot_med={boot['kstar_median']}  boot_mode={boot['kstar_mode']}  "
              f"dist={boot['kstar_distribution']}")

    # ---- 5. Save all results ----
    # Serialize changepoint_results (convert numpy types)
    def serialize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    with open(OUTPUT_DIR / "changepoint_results.json", "w") as f:
        json.dump(changepoint_results, f, indent=2, cls=NumpyEncoder)
    
    with open(OUTPUT_DIR / "agreement_summary.json", "w") as f:
        json.dump(agreement_summary, f, indent=2, cls=NumpyEncoder)
    
    with open(OUTPUT_DIR / "penalty_sensitivity.json", "w") as f:
        json.dump({"experiments": penalty_sensitivity, "summary": penalty_summary}, 
                  f, indent=2, cls=NumpyEncoder)
    
    with open(OUTPUT_DIR / "bootstrap_comparison.json", "w") as f:
        json.dump(bootstrap_comparison, f, indent=2, cls=NumpyEncoder)
    
    print(f"\nAll results saved to {OUTPUT_DIR}/")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Experiments analyzed:     {total}")
    print(f"Direct CP agreement:     {agree}/{total} = {agree/total:.1%}")
    if boot_total > 0:
        print(f"Bootstrap agreement:     {boot_agree}/{boot_total} = {boot_agree/boot_total:.1%}")
    print(f"Stable (K* range ≤ 2):   {stable_count}/{total} = {stable_count/total:.1%}")
    print(f"Best PELT penalty:       {penalty_summary['best_penalty']} "
          f"(agreement={penalty_summary['best_agreement_rate']:.1%})")


if __name__ == "__main__":
    run_analysis()
