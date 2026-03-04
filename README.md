# How Training Data Shapes the Use of Parametric and In-Context Knowledge in Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2510.02370-b31b32.svg)](https://arxiv.org/abs/2510.02370)

## Overview

This repository provides code for reproducing the experiments in our paper. We investigate how properties of training data — repetition structure, noise level, and frequency distribution — shape whether language models rely on **parametric knowledge (PK)** encoded in parameters or **in-context knowledge (ICK)** from the input context.

## Repository Structure

```
├── dataset_generation/          # Dataset creation pipeline
│   ├── seed_data/               # Source data (names, cities, universities, majors)
│   ├── sentence_templates.py    # 4 attributes × 20 paraphrase templates
│   ├── generate_profiles.py     # Step 1: Generate synthetic person profiles
│   ├── build_dataset.py         # Step 2: Build structured bio dataset (train/unknown/pert)
│   └── generate_corpus.py       # Step 3: Generate training corpora (Zipf + noise)
│
├── training/                    # Model training
│   └── train.py                 # GPT-2 (8-layer) pretraining with SFTTrainer + packing
│
├── evaluation/                  # Knowledge probing
│   └── probe.py                 # Probing (Acc_PKU, Acc_ICKU, Pref_PK, Pref_ICK)
│
├── analysis/                    # Visualization notebooks
│   └── plot_main_results.ipynb  # Figures 3, 4 (repetition & noise effects)
│
└── data/                        # Generated data (gitignored)
```

## Setup

```bash
conda create -n pk-ick python=3.10 -y
conda activate pk-ick
pip install -r requirements.txt
```

## Pipeline

### 1. Generate Profiles

Create 200K synthetic person profiles with 4 attributes (birth_city, birth_date, major, university).

```bash
python dataset_generation/generate_profiles.py \
    --n_profiles 200000 \
    --output data/profiles.json
```

### 2. Build Dataset

Split profiles into train/unknown/perturbed sets for probing.

```bash
python dataset_generation/build_dataset.py \
    --profiles data/profiles.json \
    --output_dir data/ \
    --n_train 50000
```

Produces: `data/bioS_train.json`, `data/bioS_unknown.json`, `data/bioS_pert.json`

### 3. Generate Training Corpus

Generate a training corpus with specific repetition mode, Zipf distribution, and noise level.

```bash
# §3.1 - Single occurrence (base mode, no repetition)
python dataset_generation/generate_corpus.py \
    --profiles_json data/bioS_train.json \
    --out_dir data/corpora --out_json sec31_base.json \
    --mode base --zipf_s 0 --noise_prob 0.0 \
    --max_steps 16000 --batch_size 32 --grad_accum 4

# §3.1 - Repeated (multiple-context mode)
python dataset_generation/generate_corpus.py \
    --profiles_json data/bioS_train.json \
    --out_dir data/corpora --out_json sec31_multi.json \
    --mode multiple-context --zipf_s 0 --noise_prob 0.0 \
    --max_steps 16000 --batch_size 32 --grad_accum 4

# §3.2 - Noise experiments (1%, 3%, 5%, 10%)
python dataset_generation/generate_corpus.py \
    --profiles_json data/bioS_train.json \
    --out_dir data/corpora --out_json sec32_noise001.json \
    --mode multiple-context --zipf_s 0 --noise_prob 0.01 \
    --max_steps 16000 --batch_size 32 --grad_accum 4

# §3.3 - Zipf distribution (α=1.0) + noise
python dataset_generation/generate_corpus.py \
    --profiles_json data/bioS_train.json \
    --out_dir data/corpora --out_json sec33_zipf_noise000.json \
    --mode multiple-context --zipf_s 1.0 --noise_prob 0.0 \
    --max_steps 16000 --batch_size 32 --grad_accum 4
```

### 4. Train Model

Train an 8-layer GPT-2 model (~51M params) on the generated corpus using SFTTrainer with packing.

```bash
python training/train.py \
    --data data/corpora/sec31_multi.json \
    --output_root ./checkpoints \
    --max_steps 16000 \
    --batch_size 32 \
    --grad_accum 4 \
    --max_seq_len 512 \
    --lr 4e-4
```

Checkpoints are saved every 1,000 steps to `./checkpoints/<corpus_name>_<timestamp>/`.

### 5. Evaluate (Knowledge Probing)

Probe all checkpoints for parametric vs. in-context knowledge usage. The script scans `MODEL_ROOT` for run directories containing `checkpoint-*` subdirectories.

```bash
# Set MODEL_ROOT to point to your checkpoints directory
PROBE_MODEL_ROOT=./checkpoints python evaluation/probe.py
```

Results are saved to `probe_results.csv` with columns: `model`, `step`, `acc1/<mode>`, `em/<mode>`.

Probing modes:
- `param`: No context, test parametric recall
- `in_ctx`: Original context provided
- `pert_ctx_orig` / `pert_ctx_pert`: Perturbed context (knowledge conflict)
- `ood_in_ctx`: Out-of-distribution context
- `multi_in_ctx` / `multi_ood_in_ctx`: Multiple contexts concatenated

### 6. Analyze Results

Open `analysis/plot_main_results.ipynb` in Jupyter to reproduce the paper's figures.

## Training Conditions (Paper §3)

| Condition | Mode | Noise | Zipf α | Corpus File | Paper |
|-----------|------|-------|--------|-------------|-------|
| Single | `base` | 0% | — | `sec31_base.json` | §3.1 |
| Repeated | `multiple-context` | 0% | — | `sec31_multi.json` | §3.1 |
| +Noise 1% | `multiple-context` | 1% | — | `sec32_noise001.json` | §3.2 |
| +Noise 3% | `multiple-context` | 3% | — | `sec32_noise003.json` | §3.2 |
| +Noise 5% | `multiple-context` | 5% | — | `sec32_noise005.json` | §3.2 |
| +Noise 10% | `multiple-context` | 10% | — | `sec32_noise010.json` | §3.2 |
| Zipf+0% | `multiple-context` | 0% | 1.0 | `sec33_zipf_noise000.json` | §3.3 |
| Zipf+1% | `multiple-context` | 1% | 1.0 | `sec33_zipf_noise001.json` | §3.3 |
| Zipf+3% | `multiple-context` | 3% | 1.0 | `sec33_zipf_noise003.json` | §3.3 |
| Zipf+5% | `multiple-context` | 5% | 1.0 | `sec33_zipf_noise005.json` | §3.3 |
| Zipf+10% | `multiple-context` | 10% | 1.0 | `sec33_zipf_noise010.json` | §3.3 |

## Model Architecture

| Hyperparameter | Value |
|----------------|-------|
| Architecture | GPT-2 |
| Layers | 8 |
| Hidden dim | 512 |
| Attention heads | 8 |
| FFN inner dim | 2048 |
| Max sequence length | 512 |
| Parameters | ~51M |
| Optimizer | AdamW (lr=4e-4, weight_decay=0.1) |
| LR schedule | Cosine |
| Effective batch size | 128 (batch=32 × grad_accum=4) |
| Training steps | 16,000 |

## Citation

```bibtex
@article{kim2025training,
  title={How Training Data Shapes the Use of Parametric and In-Context Knowledge in Language Models},
  author={Kim, Minsung and Kim, Dong-Kyum and Kwon, Jea and Yang, Nakyeong and Jung, Kyomin and Cha, Meeyoung},
  journal={arXiv preprint arXiv:2510.02370},
  year={2025}
}
```
