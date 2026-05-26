# Generative Depth Contamination

Code and paper source for the EMNLP 2026 ARR submission: **"Generational Depth Estimation of Recursive Synthetic Text: Measuring the Discrimination Boundary."**

## Repository Structure

```
code/                          # Experiment code
├── scripts/
│   ├── run_pipeline.py        # End-to-end pipeline: depth-chain generation → feature extraction → OBD classification
│   ├── run_multi_source*.py   # Multi-source OBD experiments (C4, Wiki, arXiv, Code)
│   ├── run_exp015_neural_probe.py   # Neural probe validation (frozen LLM embeddings + MLP)
│   ├── run_exp013_cross_transfer*.py  # Cross-model transfer experiments
│   ├── exp_*_retrain_chain.py       # Retrain-chain experiments (OLMo, Qwen, Pythia)
│   ├── exp_*_filtering*.py          # Filtering strategy experiments
│   ├── feature_importance*.py       # Feature importance analysis (LOFO, permutation)
│   ├── *_bootstrap_ci.py            # Bootstrap confidence interval computation
│   ├── plot_*.py / draw_*.py        # Figure generation scripts
│   ├── download_*_seed.py           # Seed data downloaders (arXiv, Wiki, Code)
│   └── *.sh                         # Experiment launch scripts
├── borderline_analysis*.py    # Borderline depth-pair analysis
└── run_pythia14b_nuc.sh

paper/                         # LaTeX source files
├── main.tex
├── introduction.tex
├── method.tex
├── experiments_c1.tex         # Experiments (discrimination boundary, transfer, domain)
├── experiments_c3.tex         # K*-Guided Filtering (consistency check)
├── related_work.tex
├── discussion.tex
├── conclusion.tex
├── appendix.tex
├── appendix_extended.tex
├── references.bib
├── tables/
└── figures/
```

## Requirements

- Python 3.10+
- PyTorch 2.x with CUDA support
- Key packages: `transformers`, `datasets`, `scikit-learn`, `lightgbm`, `scipy`, `numpy`, `pandas`, `tqdm`

```bash
pip install torch transformers datasets scikit-learn lightgbm scipy numpy pandas tqdm
```

## Reproduction

### 1. Data Generation (Depth-Chain)

Generate multi-depth synthetic text via iterative prompting with a language model:

```bash
python code/scripts/run_pipeline.py \
    --model Qwen/Qwen2.5-1.5B \
    --max_depth 5 \
    --n_docs 2000 \
    --seed 42
```

This produces `depth_0.jsonl` through `depth_K.jsonl` and extracts 15D OBD features.

### 2. OBD Classification (Multi-Source)

Run pairwise depth classifiers across multiple generator models:

```bash
python code/scripts/run_multi_source.py
```

### 3. Neural Probe Validation

Validate K* with frozen LLM embeddings + MLP classifier:

```bash
python code/scripts/run_exp015_neural_probe.py \
    --model Qwen/Qwen2.5-1.5B \
    --data_dir <generated_data>
```

### 4. Cross-Model Transfer

Evaluate whether OBD features transfer across generator models:

```bash
python code/scripts/run_exp013_cross_transfer_v2.py
```

### 5. Feature Importance

Analyze which features drive depth detection:

```bash
python code/scripts/feature_importance_analysis.py
```

## Citation

Paper under review. Citation will be provided upon acceptance.
