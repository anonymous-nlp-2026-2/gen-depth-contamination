import json, glob, os, re, sys

RESULTS_DIR = "/root/autodl-tmp/gen-depth-contamination/results"

def parse_pair_key(key):
    m = re.match(r"(\d+)v(\d+)", key)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None

def extract_pairwise_aucs(data):
    """Extract {pair_key: auc_value} from various JSON formats."""
    sources = []

    # Format 1: top-level "mean" dict (pairwise_auc.json)
    if "mean" in data and isinstance(data["mean"], dict):
        pairs = {}
        for k, v in data["mean"].items():
            p = parse_pair_key(k)
            if p and isinstance(v, (int, float)):
                pairs[k] = float(v)
        if pairs:
            sources.append(("log-likelihood (mean)", pairs))

    # Format 2: "pairs" dict with nested mean (pairwise_auc_t_dist.json)
    if "pairs" in data and isinstance(data["pairs"], dict):
        pairs = {}
        for k, v in data["pairs"].items():
            p = parse_pair_key(k)
            if p and isinstance(v, dict) and "mean" in v:
                pairs[k] = float(v["mean"])
        if pairs:
            sources.append(("log-likelihood (t-dist)", pairs))

    # Format 3: "pairwise_auc" dict with nested auc (DeBERTa)
    if "pairwise_auc" in data and isinstance(data["pairwise_auc"], dict):
        pairs = {}
        for k, v in data["pairwise_auc"].items():
            p = parse_pair_key(k)
            if p and isinstance(v, dict) and "auc" in v:
                pairs[k] = float(v["auc"])
        if pairs:
            sources.append(("DeBERTa", pairs))

    # Format 4: "pairwise" dict with nested auc (DeBERTa cross-transfer)
    if "pairwise" in data and isinstance(data["pairwise"], dict):
        pairs = {}
        for k, v in data["pairwise"].items():
            p = parse_pair_key(k)
            if p and isinstance(v, dict) and "auc" in v:
                pairs[k] = float(v["auc"])
        if pairs:
            sources.append(("DeBERTa-cross", pairs))

    # Format 5: "obd_baseline" flat dict (in DeBERTa files)
    if "obd_baseline" in data and isinstance(data["obd_baseline"], dict):
        pairs = {}
        for k, v in data["obd_baseline"].items():
            p = parse_pair_key(k)
            if p and isinstance(v, (int, float)):
                pairs[k] = float(v)
        if pairs:
            sources.append(("obd_baseline", pairs))

    return sources

def check_monotonic(pairs):
    sorted_keys = sorted(pairs.keys(), key=lambda k: parse_pair_key(k))
    aucs = [(k, pairs[k]) for k in sorted_keys]
    violations = []
    for i in range(len(aucs) - 1):
        if aucs[i][1] < aucs[i+1][1]:
            violations.append({
                "pair_a": aucs[i][0], "auc_a": aucs[i][1],
                "pair_b": aucs[i+1][0], "auc_b": aucs[i+1][1],
                "increase": round(aucs[i+1][1] - aucs[i][1], 6)
            })
    return aucs, violations

# Collect all JSON files
all_jsons = set()
for pat in ["**/pairwise_auc.json", "**/*_auc*.json"]:
    all_jsons.update(glob.glob(f"{RESULTS_DIR}/{pat}", recursive=True))

# Also include DeBERTa result files
for ddir in ["exp_deberta_upperbound", "deberta_cross_transfer", "exp_deberta_c4_1b5", "exp_deberta_arxiv", "exp_deberta_arxiv_qwen1b5"]:
    dpath = os.path.join(RESULTS_DIR, ddir)
    if os.path.isdir(dpath):
        for f in os.listdir(dpath):
            if f.endswith(".json") and "theta_sensitivity" not in f and f != "summary.json":
                all_jsons.add(os.path.join(dpath, f))

output = {"monotonic": [], "non_monotonic": [], "errors": []}

for json_path in sorted(all_jsons):
    rel = json_path.replace(RESULTS_DIR + "/", "")
    try:
        with open(json_path) as f:
            data = json.load(f)

        sources = extract_pairwise_aucs(data)
        if not sources:
            continue

        for method, pairs in sources:
            if len(pairs) < 2:
                continue
            aucs_sorted, violations = check_monotonic(pairs)
            entry = {
                "file": rel,
                "method": method,
                "pairs": {k: v for k, v in aucs_sorted},
                "is_monotonic": len(violations) == 0
            }
            if violations:
                entry["violations"] = violations
                output["non_monotonic"].append(entry)
            else:
                output["monotonic"].append(entry)
    except Exception as e:
        output["errors"].append({"file": rel, "error": str(e)})

output["summary"] = {
    "total_series": len(output["monotonic"]) + len(output["non_monotonic"]),
    "monotonic": len(output["monotonic"]),
    "non_monotonic": len(output["non_monotonic"]),
    "errors": len(output["errors"])
}

with open(f"{RESULTS_DIR}/monotonicity_check.json", "w") as f:
    json.dump(output, f, indent=2)

print(json.dumps(output["summary"], indent=2))
if output["non_monotonic"]:
    print("\n=== Non-monotonic cases ===")
    for case in output["non_monotonic"]:
        print(f"\n  {case['file']} [{case['method']}]")
        seq = "  AUC: " + " > ".join(f"{k}={v:.4f}" for k, v in case["pairs"].items())
        print(seq)
        for v in case["violations"]:
            print(f"    VIOLATION: {v['pair_a']}={v['auc_a']:.4f} < {v['pair_b']}={v['auc_b']:.4f} (Δ+{v['increase']:.4f})")
else:
    print("\nAll series are monotonically decreasing.")
