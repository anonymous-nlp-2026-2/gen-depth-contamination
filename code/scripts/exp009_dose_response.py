#!/usr/bin/env python3
"""exp-009: Contamination dose-response experiment.

Compares graduated vs binary K* filtering at varying contamination ratios (10/30/50%).
Each configuration: continued pretraining Qwen2.5-1.5B on contaminated corpus + lm-eval.

Usage:
    python scripts/exp009_dose_response.py --ratio 0.1 --device cuda:1 --num_seeds 5
"""

import argparse
import gc
import json
import logging
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/root/autodl-tmp/gen-depth-contamination")
MODEL_PATH = "/root/autodl-tmp/.hf_cache/Qwen/Qwen2___5-1___5B"
HUMAN_DATA = PROJECT_ROOT / "data_exp020_pythia_greedy" / "depth_0.jsonl"
AI_DATA_DIR = PROJECT_ROOT / "data_exp020_pythia_greedy"
AI_DEPTHS = [1, 2, 3, 4, 5]

STRATEGIES = ["no_filter", "binary_ours", "graduated"]
GRADUATED_ALPHA = 0.9

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def save_jsonl(records, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Phase 1: Data Mixing
# ---------------------------------------------------------------------------

def prepare_mixed_data(ratio, seed, data_dir):
    """Mix human and AI text at specified contamination ratio.
    
    ratio: fraction of final corpus that is AI-generated (e.g. 0.1 = 10%)
    Returns path to mixed JSONL file.
    """
    out_path = data_dir / f"mixed_seed{seed}.jsonl"
    if out_path.exists():
        n = sum(1 for _ in open(out_path))
        if n > 0:
            log.info(f"Mixed data exists ({n} samples), skipping")
            return out_path

    rng = np.random.RandomState(seed)

    human_records = load_jsonl(HUMAN_DATA)
    n_human = len(human_records)

    # Number of AI samples needed for target ratio
    n_ai = int(n_human * ratio / (1 - ratio))
    
    # Pool all AI depth data and sample
    ai_pool = []
    for d in AI_DEPTHS:
        ai_pool.extend(load_jsonl(AI_DATA_DIR / f"depth_{d}.jsonl"))
    
    if n_ai > len(ai_pool):
        log.warning(f"Requested {n_ai} AI samples but only {len(ai_pool)} available, using all")
        n_ai = len(ai_pool)
    
    ai_indices = rng.choice(len(ai_pool), n_ai, replace=False)
    ai_records = [ai_pool[i] for i in ai_indices]

    # Tag records with source label (for oracle evaluation, not used in training)
    mixed = []
    for r in human_records:
        mixed.append({"text": r["text"], "doc_id": r["doc_id"], "source": "human", "depth": 0})
    for r in ai_records:
        mixed.append({"text": r["text"], "doc_id": r.get("doc_id", -1), "source": "ai", "depth": r.get("depth", 1)})
    
    rng.shuffle(mixed)
    save_jsonl(mixed, out_path)
    log.info(f"Mixed data: {n_human} human + {n_ai} AI = {len(mixed)} total (ratio={n_ai/len(mixed):.3f})")
    return out_path


# ---------------------------------------------------------------------------
# Phase 2: K* Scoring (Surprisal Features + LightGBM)
# ---------------------------------------------------------------------------

def compute_surprisal_batch(model, tokenizer, texts, batch_size=16, device="cuda:0"):
    """Compute mean surprisal for each text."""
    surprisals = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits

        log_probs = torch.log_softmax(logits, dim=-1)
        input_ids = inputs["input_ids"]

        for i in range(len(batch_texts)):
            mask = inputs["attention_mask"][i].bool()
            seq_len = mask.sum().item()
            if seq_len < 3:
                surprisals.append(np.zeros(9))
                continue

            offset = input_ids.shape[1] - seq_len
            token_ids = input_ids[i, offset:]
            token_surp = []
            for t in range(1, seq_len):
                lp = log_probs[i, offset + t - 1, token_ids[t]].item()
                token_surp.append(-lp)

            arr = np.array(token_surp)
            d1 = np.diff(arr)
            d2 = np.diff(d1) if len(d1) > 1 else np.array([0.0])

            feats = np.array([
                np.mean(arr), np.std(arr),
                float(stats.skew(arr)), float(stats.kurtosis(arr)),
                np.mean(d1), np.std(d1),
                float(stats.skew(d1)) if len(d1) >= 3 else 0.0,
                np.mean(d2), np.std(d2),
            ])
            surprisals.append(feats)

        torch.cuda.empty_cache()

    return np.array(surprisals)


def train_filter_classifier(features_human, features_ai):
    """Train LightGBM binary classifier: human(0) vs AI(1)."""
    import lightgbm as lgb

    X = np.vstack([features_human, features_ai])
    y = np.array([0] * len(features_human) + [1] * len(features_ai))

    clf = lgb.LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        num_leaves=31, verbose=-1, n_jobs=-1,
    )
    clf.fit(X, y)
    return clf


def apply_filtering(mixed_records, strategy, model, tokenizer, device, seed):
    """Apply filtering strategy and return (filtered_records, sample_weights).
    
    Returns:
        records: list of text records to train on
        weights: list of float weights (1.0 for unweighted)
    """
    if strategy == "no_filter":
        return mixed_records, [1.0] * len(mixed_records)

    log.info(f"Computing surprisal features for {len(mixed_records)} texts...")
    texts = [r["text"] for r in mixed_records]
    features = compute_surprisal_batch(model, tokenizer, texts, batch_size=16, device=device)

    # Train classifier on known labeled data (use depth_0 vs depth_1 as training signal)
    log.info("Training K* classifier on reference data...")
    rng = np.random.RandomState(seed + 1000)
    
    # Use 500 samples from each class for classifier training
    n_train = 500
    human_train = load_jsonl(HUMAN_DATA)
    rng.shuffle(human_train)
    human_train_texts = [r["text"] for r in human_train[:n_train]]
    
    ai_train_pool = load_jsonl(AI_DATA_DIR / "depth_1.jsonl")
    rng.shuffle(ai_train_pool)
    ai_train_texts = [r["text"] for r in ai_train_pool[:n_train]]

    feat_human = compute_surprisal_batch(model, tokenizer, human_train_texts, batch_size=16, device=device)
    feat_ai = compute_surprisal_batch(model, tokenizer, ai_train_texts, batch_size=16, device=device)

    clf = train_filter_classifier(feat_human, feat_ai)

    # Predict on mixed corpus
    proba_ai = clf.predict_proba(features)[:, 1]  # P(AI)

    if strategy == "binary_ours":
        # Remove texts classified as AI (threshold 0.5)
        keep_mask = proba_ai < 0.5
        filtered = [r for r, keep in zip(mixed_records, keep_mask) if keep]
        weights = [1.0] * len(filtered)
        log.info(f"Binary filter: kept {len(filtered)}/{len(mixed_records)} "
                 f"({len(filtered)/len(mixed_records)*100:.1f}%)")
        return filtered, weights
    
    elif strategy == "graduated":
        # Weight by P(human)^alpha
        p_human = 1 - proba_ai
        weights = [float(p ** GRADUATED_ALPHA) for p in p_human]
        log.info(f"Graduated weights: mean={np.mean(weights):.3f}, "
                 f"min={np.min(weights):.3f}, max={np.max(weights):.3f}")
        return mixed_records, weights

    raise ValueError(f"Unknown strategy: {strategy}")


# ---------------------------------------------------------------------------
# Phase 3: Continued Pretraining
# ---------------------------------------------------------------------------

def continued_pretrain(records, weights, model_path, output_dir, device, seed,
                       max_length=256, batch_size=4, grad_accum=4, lr=2e-5):
    """Continued pretraining with optional sample weights."""
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        Trainer, TrainingArguments, DataCollatorForLanguageModeling,
    )
    from torch.utils.data import Dataset

    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    class WeightedTextDataset(Dataset):
        def __init__(self, texts, weights, tokenizer, max_length):
            self.encodings = []
            self.weights = []
            for text, w in zip(texts, weights):
                enc = tokenizer(text, truncation=True, max_length=max_length,
                                padding="max_length", return_tensors="pt")
                self.encodings.append({k: v.squeeze(0) for k, v in enc.items()})
                self.weights.append(w)

        def __len__(self):
            return len(self.encodings)

        def __getitem__(self, idx):
            item = {k: v.clone() for k, v in self.encodings[idx].items()}
            item["labels"] = item["input_ids"].clone()
            # Mask padding tokens in labels
            item["labels"][item["attention_mask"] == 0] = -100
            item["weight"] = torch.tensor(self.weights[idx], dtype=torch.float32)
            return item

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            weights = inputs.pop("weight", None)
            outputs = model(**inputs)
            loss = outputs.loss
            if weights is not None and weights.numel() > 0:
                # Per-sample weighting (approximate: scale batch loss by mean weight)
                loss = loss * weights.mean()
            return (loss, outputs) if return_outputs else loss

    texts = [r["text"] for r in records]
    dataset = WeightedTextDataset(texts, weights, tokenizer, max_length)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="no",
        bf16=True,
        dataloader_pin_memory=False,
        seed=seed,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    log.info(f"Starting continued pretraining: {len(dataset)} samples, device={device}")
    trainer.train()

    # Save model
    model.save_pretrained(output_dir / "model")
    tokenizer.save_pretrained(output_dir / "model")
    log.info(f"Model saved to {output_dir / 'model'}")

    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()

    return output_dir / "model"


# ---------------------------------------------------------------------------
# Phase 4: Evaluation (lm-eval-harness)
# ---------------------------------------------------------------------------

def evaluate_model(model_path, device, results_path):
    """Run lm-eval-harness on MMLU and HellaSwag."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    log.info(f"Evaluating {model_path} on MMLU + HellaSwag...")

    lm = HFLM(
        pretrained=str(model_path),
        device=device,
        dtype="bfloat16",
        batch_size="auto:4",
        trust_remote_code=True,
        max_length=2048,
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            results = lm_eval.simple_evaluate(
                model=lm,
                tasks=["mmlu", "hellaswag"],
                num_fewshot=5,
            )
            break
        except (RuntimeError, ConnectionError, OSError) as e:
            if attempt < max_retries - 1:
                wait = 30 * (attempt + 1)
                log.warning(f"Eval attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

    # Extract scores
    scores = {}
    if "mmlu" in results["results"]:
        scores["mmlu"] = results["results"]["mmlu"].get("acc,none", 
                         results["results"]["mmlu"].get("acc", 0))
    if "hellaswag" in results["results"]:
        scores["hellaswag"] = results["results"]["hellaswag"].get("acc_norm,none",
                              results["results"]["hellaswag"].get("acc_norm", 0))

    # Save full results
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({"scores": scores, "full": {k: v for k, v in results["results"].items()}}, 
                  f, indent=2, default=str)

    log.info(f"Eval results: MMLU={scores.get('mmlu', 'N/A'):.4f}, "
             f"HellaSwag={scores.get('hellaswag', 'N/A'):.4f}")

    del lm
    torch.cuda.empty_cache()
    gc.collect()

    return scores


# ---------------------------------------------------------------------------
# Phase 5: Statistical Analysis
# ---------------------------------------------------------------------------

def run_statistical_analysis(results_dir, ratio):
    """Compute paired t-test, Holm-Bonferroni, Cohen's d across seeds."""
    all_results = {}
    for strategy in STRATEGIES:
        all_results[strategy] = {"mmlu": [], "hellaswag": []}
        for f in sorted(results_dir.glob(f"{strategy}_seed*_eval.json")):
            with open(f) as fh:
                data = json.load(fh)
            scores = data["scores"]
            all_results[strategy]["mmlu"].append(scores.get("mmlu", 0))
            all_results[strategy]["hellaswag"].append(scores.get("hellaswag", 0))

    analysis = {"ratio": ratio, "strategies": {}}
    
    for strategy in STRATEGIES:
        analysis["strategies"][strategy] = {
            "mmlu_mean": float(np.mean(all_results[strategy]["mmlu"])),
            "mmlu_std": float(np.std(all_results[strategy]["mmlu"])),
            "hellaswag_mean": float(np.mean(all_results[strategy]["hellaswag"])),
            "hellaswag_std": float(np.std(all_results[strategy]["hellaswag"])),
            "n_seeds": len(all_results[strategy]["mmlu"]),
        }

    # Paired comparisons: binary_ours vs no_filter, graduated vs no_filter
    comparisons = []
    for metric in ["mmlu", "hellaswag"]:
        baseline = np.array(all_results["no_filter"][metric])
        for strategy in ["binary_ours", "graduated"]:
            treatment = np.array(all_results[strategy][metric])
            if len(baseline) < 2 or len(treatment) < 2:
                continue
            t_stat, p_val = stats.ttest_rel(treatment, baseline)
            diff = treatment - baseline
            cohens_d = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))
            comparisons.append({
                "metric": metric,
                "comparison": f"{strategy}_vs_no_filter",
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "cohens_d": cohens_d,
                "mean_diff": float(np.mean(diff)),
            })

    # Holm-Bonferroni correction
    if comparisons:
        p_values = [c["p_value"] for c in comparisons]
        sorted_indices = np.argsort(p_values)
        m = len(p_values)
        for rank, idx in enumerate(sorted_indices):
            adjusted_alpha = 0.05 / (m - rank)
            comparisons[idx]["holm_bonferroni_significant"] = p_values[idx] < adjusted_alpha
            comparisons[idx]["adjusted_alpha"] = adjusted_alpha

    analysis["comparisons"] = comparisons

    out_path = results_dir / "statistical_analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    log.info(f"Statistical analysis saved to {out_path}")
    return analysis


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="exp-009: Contamination dose-response")
    parser.add_argument("--ratio", type=float, required=True,
                        choices=[0.1, 0.3, 0.5], help="Contamination ratio")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="CUDA device")
    parser.add_argument("--seed_start", type=int, default=0,
                        help="Starting seed")
    parser.add_argument("--num_seeds", type=int, default=5,
                        help="Number of seeds to run")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only prepare data, skip training/eval")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip lm-eval (useful for debugging training)")
    args = parser.parse_args()

    ratio_pct = int(args.ratio * 100)
    base_data_dir = PROJECT_ROOT / "data" / "exp_009_dose_response" / f"ratio_{ratio_pct}"
    results_dir = PROJECT_ROOT / "results" / "exp_009_dose_response" / f"ratio_{ratio_pct}"
    base_data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"=== exp-009: ratio={args.ratio}, device={args.device}, "
             f"seeds={args.seed_start}-{args.seed_start + args.num_seeds - 1} ===")

    if args.dry_run:
        log.info("DRY RUN: preparing data only")
        for seed in range(args.seed_start, args.seed_start + args.num_seeds):
            prepare_mixed_data(args.ratio, seed, base_data_dir)
        log.info("Dry run complete.")
        return

    # Load scorer model once for all filtering
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log.info("Loading scorer model...")
    scorer_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if scorer_tokenizer.pad_token is None:
        scorer_tokenizer.pad_token = scorer_tokenizer.eos_token
        scorer_tokenizer.pad_token_id = scorer_tokenizer.eos_token_id
    scorer_tokenizer.padding_side = "left"

    scorer_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True,
    ).to(args.device)
    scorer_model.eval()
    log.info(f"Scorer loaded on {args.device}")

    for seed in range(args.seed_start, args.seed_start + args.num_seeds):
        log.info(f"\n{'='*60}\n  SEED {seed}\n{'='*60}")

        # Prepare mixed data
        mixed_path = prepare_mixed_data(args.ratio, seed, base_data_dir)
        mixed_records = load_jsonl(mixed_path)

        for strategy in STRATEGIES:
            eval_path = results_dir / f"{strategy}_seed{seed}_eval.json"
            if eval_path.exists():
                log.info(f"[seed={seed}, {strategy}] Already done, skipping")
                continue

            log.info(f"\n--- seed={seed}, strategy={strategy} ---")

            # Filter
            filtered_records, weights = apply_filtering(
                mixed_records, strategy, scorer_model, scorer_tokenizer,
                args.device, seed
            )

            # Continued pretraining
            train_dir = base_data_dir / f"{strategy}_seed{seed}_train"
            model_dir = continued_pretrain(
                filtered_records, weights, MODEL_PATH, train_dir,
                args.device, seed
            )

            # Free scorer model memory for eval
            scorer_model.cpu()
            torch.cuda.empty_cache()
            gc.collect()

            # Evaluate
            if not args.skip_eval:
                scores = evaluate_model(model_dir, args.device, eval_path)
            else:
                log.info("Skipping eval (--skip_eval)")
                scores = {"mmlu": 0, "hellaswag": 0}
                eval_path.parent.mkdir(parents=True, exist_ok=True)
                with open(eval_path, "w") as f:
                    json.dump({"scores": scores}, f)

            # Move scorer back to GPU
            scorer_model.to(args.device)

            # Clean up training checkpoint to save disk
            import shutil
            if (train_dir / "model").exists():
                shutil.rmtree(train_dir, ignore_errors=True)
                log.info(f"Cleaned up {train_dir}")

    # Cleanup scorer
    del scorer_model
    torch.cuda.empty_cache()
    gc.collect()

    # Statistical analysis
    log.info("\n" + "="*60 + "\n  STATISTICAL ANALYSIS\n" + "="*60)
    analysis = run_statistical_analysis(results_dir, args.ratio)

    # Print summary
    log.info("\n=== SUMMARY ===")
    for strategy, data in analysis["strategies"].items():
        log.info(f"  {strategy}: MMLU={data['mmlu_mean']:.4f}±{data['mmlu_std']:.4f}, "
                 f"HellaSwag={data['hellaswag_mean']:.4f}±{data['hellaswag_std']:.4f}")
    for comp in analysis.get("comparisons", []):
        sig = "***" if comp.get("holm_bonferroni_significant") else "n.s."
        log.info(f"  {comp['comparison']} ({comp['metric']}): "
                 f"d={comp['cohens_d']:.3f}, p={comp['p_value']:.4f} {sig}")

    log.info("\nDone!")


if __name__ == "__main__":
    main()
