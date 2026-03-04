# How Training Data Shapes the Use of Parametric and In-Context Knowledge in Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2510.02370-b31b32.svg)](https://arxiv.org/abs/2510.02370)

## Overview

This repository provides code for reproducing the controlled experiments in our paper. We investigate how three properties of training data — **(i) intra-document repetition**, **(ii) within-document inconsistency**, and **(iii) skewed knowledge frequency distribution** — jointly enable robust utilization of both parametric knowledge (PK) and in-context knowledge (ICK) in language models.

## Repository Structure

```
├── dataset_generation/          # Dataset creation pipeline
│   ├── seed_data/               # Source data (names, cities, universities, majors)
│   ├── sentence_templates.py    # 4 attributes × 20 paraphrase templates
│   ├── generate_profiles.py     # Step 1: Generate synthetic person profiles
│   ├── build_dataset.py         # Step 2: Build structured bio dataset (train/unseen/pert)
│   └── generate_corpus.py       # Step 3: Generate training corpora
│
├── training/
│   └── train.py                 # GPT-2 (8-layer) pretraining with SFTTrainer + packing
│
├── evaluation/
│   └── probe.py                 # Knowledge probing (AccPKU, AccICKU, PrefPK, PrefICK)
│
├── analysis/
│   └── plot_main_results.ipynb  # Result visualization
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

Create 200K synthetic person profiles with 4 attributes (`birth_city`, `birth_date`, `major`, `university`).

```bash
python dataset_generation/generate_profiles.py \
    --n_profiles 200000 \
    --output data/profiles.json
```

### 2. Build Dataset

Split profiles into E_train (50K) / E_unseen (50K) / perturbed sets for probing.

```bash
python dataset_generation/build_dataset.py \
    --profiles data/profiles.json \
    --output_dir data/ \
    --n_train 50000
```

Produces: `data/bioS_train.json`, `data/bioS_unseen.json`, `data/bioS_pert.json`

### 3. Generate Training Corpus

Generate a training corpus with controlled repetition structure, noise level, and frequency distribution.

```bash
# §3.1 — SINGLE (each entity appears once per document)
python dataset_generation/generate_corpus.py \
    --profiles_json data/bioS_train.json \
    --out_dir data/corpora --out_json single.json \
    --mode single --zipf_alpha 0 --noise_prob 0.0 \
    --max_steps 16000 --batch_size 32 --grad_accum 4

# §3.1 — REPEATED (two paraphrased paragraphs per entity, mixed with other entities)
python dataset_generation/generate_corpus.py \
    --profiles_json data/bioS_train.json \
    --out_dir data/corpora --out_json repeated.json \
    --mode repeated --zipf_alpha 0 --noise_prob 0.0 \
    --max_steps 16000 --batch_size 32 --grad_accum 4

# §3.2 — REPEATED + within-document inconsistency (noise=1%)
python dataset_generation/generate_corpus.py \
    --profiles_json data/bioS_train.json \
    --out_dir data/corpora --out_json repeated_noise001.json \
    --mode repeated --zipf_alpha 0 --noise_prob 0.01 \
    --max_steps 16000 --batch_size 32 --grad_accum 4

# §3.3 — REPEATED + Zipfian (α=1.0) + noise=1%
python dataset_generation/generate_corpus.py \
    --profiles_json data/bioS_train.json \
    --out_dir data/corpora --out_json zipf_noise001.json \
    --mode repeated --zipf_alpha 1.0 --noise_prob 0.01 \
    --max_steps 16000 --batch_size 32 --grad_accum 4
```

### 4. Train Model

Train an 8-layer GPT-2 model (~51M params) on the generated corpus.

```bash
python training/train.py \
    --data data/corpora/repeated.json \
    --output_root ./checkpoints \
    --max_steps 16000 \
    --batch_size 32 \
    --grad_accum 4 \
    --max_seq_len 512 \
    --lr 4e-4
```

Checkpoints are saved every 1,000 steps to `./checkpoints/<corpus_name>_<timestamp>/`.

### 5. Evaluate (Knowledge Probing)

Probe all checkpoints for parametric vs. in-context knowledge utilization.

```bash
PROBE_MODEL_ROOT=./checkpoints python evaluation/probe.py
```

Results are saved to `probe_results.csv` with columns: `model`, `step`, and per-mode accuracy metrics.

**Evaluation metrics (§2.3):**

| Metric | CSV Column | Description |
|--------|-----------|-------------|
| Acc_PKU | `em/pku` | Parametric recall without context (E_train) |
| Acc_ICKU | `em/icku` | In-context extraction (E_unseen, multi-entity context) |
| Pref_PK | `em/pref_pk` | Parametric preference under knowledge conflict |
| Pref_ICK | `em/pref_ick` | In-context preference under knowledge conflict |

### 6. Analyze Results

Open `analysis/plot_main_results.ipynb` in Jupyter to visualize the results.

## Experimental Conditions

### §3.1 — Intra-Document Repetition (Figure 3)

| Condition | Mode | Description |
|-----------|------|-------------|
| SINGLE | `single` | One paragraph per entity; attributes appear once |
| REPEATED | `repeated` | Two paraphrased paragraphs per entity, mixed with other entities |

### §3.2 — Within-Document Inconsistency (Figure 4)

Starting from REPEATED, inject inconsistency by perturbing attribute values in the leading paragraph.

| Noise | Corpus |
|-------|--------|
| 1% | `repeated_noise001.json` |
| 5% | `repeated_noise005.json` |
| 10% | `repeated_noise010.json` |

### §3.3 — Skewed Knowledge Distribution (Table 1, Figures 5–6)

Entity occurrences follow a Zipfian distribution (α=1.0) with inconsistency noise.

| Noise | Corpus |
|-------|--------|
| 0% | `zipf_noise000.json` |
| 1% | `zipf_noise001.json` |
| 5% | `zipf_noise005.json` |
| 10% | `zipf_noise010.json` |

## Model Architecture

| Hyperparameter | Value |
|----------------|-------|
| Architecture | GPT-2 (decoder-only Transformer) |
| Layers | 8 |
| Hidden dim | 512 |
| Attention heads | 8 |
| FFN inner dim | 2,048 |
| Max sequence length | 512 |
| Parameters | ~51M |
| Optimizer | AdamW (lr=4e-4, weight_decay=0.1) |
| LR schedule | Cosine |
| Effective batch size | 128 (batch=32 × grad_accum=4) |
| Training steps | 16,000 |

## Citation

```bibtex
@misc{kim2026trainingdatashapesuse,
      title={How Training Data Shapes the Use of Parametric and In-Context Knowledge in Language Models},
      author={Minsung Kim and Dong-Kyum Kim and Jea Kwon and Nakyeong Yang and Kyomin Jung and Meeyoung Cha},
      year={2026},
      eprint={2510.02370},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.02370},
}
```
