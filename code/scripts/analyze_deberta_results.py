#!/usr/bin/env python3
"""
Post-training analysis for DeBERTa upper-bound experiment.
Compares DeBERTa 6-class classifier vs LightGBM-15D OBD pairwise AUC.

Usage:
    python scripts/analyze_deberta_results.py
    python scripts/analyze_deberta_results.py --recompute   # re-derive metrics from OOF npz files
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

ROOT = Path("/root/autodl-tmp/gen-depth-contamination")
DEBERTA_DIR = ROOT / "results" / "exp_deberta_upperbound"
RESULTS_DIR = ROOT / "results"

PAIRS = ["0v1", "1v2", "2v3", "3v4", "4v5"]
N_CLASSES = 6
THETA = 0.60
N_BOOT = 10000
SEED = 42

EXPERIMENTS = ["wiki_pythia", "wiki_olmo", "wiki_qwen7b", "wiki_llama8b", "wiki_mistral7b"]


# ── data loading ──────────────────────────────────────────────────────────────

def load_deberta_results():
    summary = DEBERTA_DIR / "summary.json"
    if summary.exists():
        return json.load(open(summary))
    out = {}
    for exp in EXPERIMENTS:
        p = DEBERTA_DIR / f"{exp}.json"
        if p.exists():
            out[exp] = json.load(open(p))
    return out


def load_lgbm_results():
    out = {}
    for exp in EXPERIMENTS:
        p = RESULTS_DIR / f"exp_{exp}" / "pairwise_auc.json"
        if p.exists():
            out[exp] = json.load(open(p))
    return out


def load_oof(exp_name):
    p = DEBERTA_DIR / f"{exp_name}_oof.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return d["probs"], d["labels"], d["doc_ids"]


def load_raw_data(exp_name):
    exp_dir = ROOT / "data" / f"exp_{exp_name}"
    if not exp_dir.exists():
        return None, None, None
    labels, doc_ids = [], []
    for d in range(N_CLASSES):
        fp = exp_dir / f"depth_{d}.jsonl"
        if not fp.exists():
            return None, None, None
        with open(fp) as f:
            for line in f:
                obj = json.loads(line)
                labels.append(d)
                doc_ids.append(obj["doc_id"])
    return np.array(labels), np.array(doc_ids), len(labels)


# ── metrics ───────────────────────────────────────────────────────────────────

def pairwise_auc_bootstrap(labels, probs, doc_ids, n_boot=N_BOOT, theta=THETA, seed=SEED):
    rng = np.random.RandomState(seed)
    results = {}
    for k in range(N_CLASSES - 1):
        key = f"{k}v{k+1}"
        mask = (labels == k) | (labels == k + 1)
        y = (labels[mask] == k + 1).astype(int)
        s = probs[mask][:, k+1:].sum(axis=1)
        docs = doc_ids[mask]
        auc = float(roc_auc_score(y, s))

        doc2idx = {}
        for i, d in enumerate(docs):
            doc2idx.setdefault(int(d), []).append(i)
        idx_arr = [np.array(v) for v in doc2idx.values()]
        n_docs = len(idx_arr)

        boots = []
        for _ in range(n_boot):
            si = rng.randint(0, n_docs, n_docs)
            idx = np.concatenate([idx_arr[j] for j in si])
            try:
                boots.append(roc_auc_score(y[idx], s[idx]))
            except ValueError:
                continue
        boots = np.array(boots)
        ci_lo = float(np.percentile(boots, 2.5))
        ci_hi = float(np.percentile(boots, 97.5))
        results[key] = {
            "auc": round(auc, 4),
            "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4),
            "ci_lower_above_theta": bool(ci_lo > theta),
            "boot_mean": round(float(boots.mean()), 4),
            "boot_std": round(float(boots.std()), 4),
        }
    kstar = 0
    for k in range(N_CLASSES - 1):
        if results[f"{k}v{k+1}"]["ci_lower_above_theta"]:
            kstar = k + 1
        else:
            break
    return results, kstar


def compute_f1_at_050(labels, probs):
    results = {}
    for k in range(N_CLASSES - 1):
        key = f"{k}v{k+1}"
        mask = (labels == k) | (labels == k + 1)
        y = (labels[mask] == k + 1).astype(int)
        s = probs[mask][:, k+1:].sum(axis=1)
        pred = (s >= 0.50).astype(int)
        f1 = float(f1_score(y, pred, zero_division=0))
        prec = float(precision_score(y, pred, zero_division=0))
        rec = float(recall_score(y, pred, zero_division=0))
        results[key] = {"f1": round(f1, 4), "precision": round(prec, 4), "recall": round(rec, 4)}
    return results


def lgbm_kstar(mean_auc, theta=THETA):
    kstar = 0
    for k in range(5):
        key = f"{k}v{k+1}"
        if key in mean_auc and mean_auc[key] > theta:
            kstar = k + 1
        else:
            break
    return kstar


# ── formatting ────────────────────────────────────────────────────────────────

def fmt(v, w=7):
    if v is None:
        return " " * w
    return f"{v:>{w}.4f}"


def print_comparison_table(deberta, lgbm_raw):
    pairs_hdr = "  ".join(f"{p:>9}" for p in PAIRS)
    print(f"{'Config':<16} {'Method':<10} {pairs_hdr}  {'K*':>3}")
    print("-" * 90)

    for exp in EXPERIMENTS:
        d = deberta.get(exp, {})
        d_pw = d.get("pairwise_auc", {})
        d_kstar = d.get("kstar", "?")

        l_data = lgbm_raw.get(exp, {})
        l_mean = l_data.get("mean", l_data) if isinstance(l_data, dict) else {}
        l_kstar = lgbm_kstar(l_mean)

        # DeBERTa AUC row
        d_cells = "  ".join(fmt(d_pw[p]["auc"], 9) if p in d_pw else " " * 9 for p in PAIRS)
        print(f"{exp:<16} {'DeBERTa':<10} {d_cells}  {d_kstar:>3}")

        # DeBERTa CI row
        ci_cells = []
        for p in PAIRS:
            if p in d_pw:
                ci_cells.append(f"[{d_pw[p]['ci_lower']:.3f},{d_pw[p]['ci_upper']:.3f}]")
            else:
                ci_cells.append(" " * 13)
        print(f"{'':<16} {'  95%CI':<10} {'  '.join(ci_cells)}")

        # LightGBM AUC row
        l_cells = "  ".join(fmt(l_mean.get(p), 9) for p in PAIRS)
        print(f"{'':<16} {'LGBM-15D':<10} {l_cells}  {l_kstar:>3}")

        # Delta row
        delta_cells = []
        for p in PAIRS:
            da = d_pw.get(p, {}).get("auc")
            la = l_mean.get(p)
            if da is not None and la is not None:
                delta_cells.append(f"{da - la:>+9.4f}")
            else:
                delta_cells.append(" " * 9)
        print(f"{'':<16} {'  delta':<10} {'  '.join(delta_cells)}")
        print("-" * 90)


def print_f1_table(f1_results):
    pairs_hdr = "  ".join(f"{p:>9}" for p in PAIRS)
    print(f"{'Config':<16} {'Metric':<10} {pairs_hdr}")
    print("-" * 80)
    for exp in EXPERIMENTS:
        f1 = f1_results.get(exp)
        if f1 is None:
            continue
        f1_cells = "  ".join(fmt(f1[p]["f1"], 9) if p in f1 else " " * 9 for p in PAIRS)
        p_cells = "  ".join(fmt(f1[p]["precision"], 9) if p in f1 else " " * 9 for p in PAIRS)
        r_cells = "  ".join(fmt(f1[p]["recall"], 9) if p in f1 else " " * 9 for p in PAIRS)
        print(f"{exp:<16} {'F1':<10} {f1_cells}")
        print(f"{'':<16} {'Prec':<10} {p_cells}")
        print(f"{'':<16} {'Recall':<10} {r_cells}")
        print("-" * 80)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true",
                        help="Re-derive metrics from OOF npz files (10K bootstrap, F1)")
    args = parser.parse_args()

    deberta = load_deberta_results()
    lgbm_raw = load_lgbm_results()

    if not deberta:
        print("ERROR: No DeBERTa results found in", DEBERTA_DIR)
        sys.exit(1)

    print(f"Found DeBERTa results for: {', '.join(deberta.keys())}")
    print(f"Found LightGBM baselines for: {', '.join(lgbm_raw.keys())}")

    # If --recompute: re-derive from OOF files with 10K bootstrap
    if args.recompute:
        recomputed = {}
        for exp in EXPERIMENTS:
            oof_data = load_oof(exp)
            if oof_data is None:
                print(f"  WARN: No OOF file for {exp}, using saved results")
                continue
            probs, labels, doc_ids = oof_data
            print(f"  Recomputing {exp} with {N_BOOT} bootstrap resamples...")
            pw, kstar = pairwise_auc_bootstrap(labels, probs, doc_ids)
            recomputed[exp] = {
                "pairwise_auc": pw,
                "kstar": kstar,
            }
        for exp, r in recomputed.items():
            if exp in deberta:
                deberta[exp]["pairwise_auc"] = r["pairwise_auc"]
                deberta[exp]["kstar"] = r["kstar"]
                deberta[exp]["bootstrap_resamples"] = N_BOOT

    # ── Comparison Table ──
    print("\n" + "=" * 90)
    print("DeBERTa vs LightGBM-15D  Pairwise AUC (OOF)")
    print("=" * 90)
    print_comparison_table(deberta, lgbm_raw)

    # ── F1@0.50 ──
    print("\n" + "=" * 80)
    print("F1 @ threshold=0.50 on P(depth > k)")
    print("=" * 80)
    f1_results = {}
    has_any = False
    for exp in EXPERIMENTS:
        oof_data = load_oof(exp)
        if oof_data is not None:
            probs, labels, _ = oof_data
            f1_results[exp] = compute_f1_at_050(labels, probs)
            has_any = True
    if has_any:
        print_f1_table(f1_results)
    else:
        print("  OOF prediction files (.npz) not found. F1 computation skipped.")
        print("  To enable: save OOF predictions during training as")
        print(f"    np.savez(RESULTS_DIR / '{{exp}}_oof.npz', probs=oof, labels=labels, doc_ids=doc_ids)")

    # ── K* Summary ──
    print("\n" + "=" * 60)
    print(f"K* Summary  (theta={THETA})")
    print("=" * 60)
    print(f"{'Config':<16} {'K*(DeBERTa)':>12} {'K*(LGBM)':>10}")
    print("-" * 40)
    for exp in EXPERIMENTS:
        d = deberta.get(exp, {})
        l_data = lgbm_raw.get(exp, {})
        l_mean = l_data.get("mean", l_data) if isinstance(l_data, dict) else {}
        dk = d.get("kstar", "?")
        lk = lgbm_kstar(l_mean)
        print(f"{exp:<16} {dk:>12} {lk:>10}")

    # ── Save JSON ──
    summary = {}
    for exp in EXPERIMENTS:
        d = deberta.get(exp, {})
        d_pw = d.get("pairwise_auc", {})
        l_data = lgbm_raw.get(exp, {})
        l_mean = l_data.get("mean", l_data) if isinstance(l_data, dict) else {}

        entry = {
            "deberta": {
                "pairwise_auc": {p: d_pw.get(p, {}).get("auc") for p in PAIRS},
                "ci_lower": {p: d_pw.get(p, {}).get("ci_lower") for p in PAIRS},
                "ci_upper": {p: d_pw.get(p, {}).get("ci_upper") for p in PAIRS},
                "kstar": d.get("kstar"),
            },
            "lgbm_15d": {
                "pairwise_auc": {p: l_mean.get(p) for p in PAIRS},
                "kstar": lgbm_kstar(l_mean),
            },
            "delta_auc": {},
        }
        for p in PAIRS:
            da = d_pw.get(p, {}).get("auc")
            la = l_mean.get(p)
            if da is not None and la is not None:
                entry["delta_auc"][p] = round(da - la, 4)
        if exp in f1_results:
            entry["deberta"]["f1_at_050"] = f1_results[exp]
        summary[exp] = entry

    out_path = DEBERTA_DIR / "comparison.json"
    json.dump(summary, open(out_path, "w"), indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
