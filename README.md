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
│   ├── generate_corpus.py       # Step 3: Generate training corpora (Zipf + noise)
│   └── generate_corpus.sh       # SLURM wrapper
│
├── training/                    # Model training
│   ├── train.py                 # GPT-2 (8-layer) pretraining with on-the-fly Zipf sampling
│   └── train.sh                 # SLURM wrapper
│
├── evaluation/                  # Knowledge probing
│   ├── probe.py                 # Probing (Acc_PKU, Acc_ICKU, Pref_PK, Pref_ICK)
│   └── probe.sh                 # SLURM wrapper
│
├── analysis/                    # Visualization notebooks
│   ├── plot_main_results.ipynb  # Figures 3, 4 (repetition & noise effects)
│   ├── plot_zipf_results.ipynb  # Figure 5 (frequency-dependent behavior)
│   └── plot_correlation.ipynb   # Figure 6 (confidence vs preference)
│
└── data/                        # Generated data (gitignored)
```

## Setup

```bash
conda create -n pk-ick python=3.9 -y
conda activate pk-ick
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Profiles

```bash
python dataset_generation/generate_profiles.py \
    --n_profiles 200000 \
    --output data/profiles.json
```

### 2. Build Dataset

```bash
python dataset_generation/build_dataset.py \
    --profiles data/profiles.json \
    --output_dir data/ \
    --n_train 50000
```

This produces `data/bioS_train.json`, `data/bioS_unknown.json`, and `data/bioS_pert.json`.

### 3. Generate Training Corpus

```bash
# Zipf distribution (α=1.0) + 1% noise, multiple-context mode
python dataset_generation/generate_corpus.py \
    --profiles_json data/bioS_train.json \
    --out_dir data/corpora \
    --out_json zipf_100_1.json \
    --mode multiple-context \
    --zipf_s 1.0 \
    --noise_prob 0.01 \
    --do_build_dataset
```

### 4. Train Model

```bash
python training/train.py \
    --profiles_json data/bioS_train.json \
    --mode multiple-context \
    --zipf_s 1.0 \
    --noise_prob 0.01 \
    --samples_per_epoch 300000
```

### 5. Evaluate

```bash
python evaluation/probe.py
```

Results are saved to `probe_results.csv`.

## Training Conditions (Paper §3)

| Condition | Mode | Noise | Zipf | Paper Section |
|-----------|------|-------|------|---------------|
| Single | `base` | 0% | No | §3.1 |
| Repeated | `context` | 0% | No | §3.1 |
| Repeated+Mix | `multiple-context` | 0% | No | §3.1 |
| +Noise | `multiple-context` | 1-10% | No | §3.2 |
| +Zipf | `multiple-context` | 0-10% | α=1.0 | §3.3 |

## Citation

```bibtex
@article{kim2025training,
  title={How Training Data Shapes the Use of Parametric and In-Context Knowledge in Language Models},
  author={Kim, Minsung and Kim, Dong-Kyum and Kwon, Jea and Yang, Nakyeong and Jung, Kyomin and Cha, Meeyoung},
  journal={arXiv preprint arXiv:2510.02370},
  year={2025}
}
```
