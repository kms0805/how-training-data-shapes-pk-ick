#!/bin/bash
#SBATCH --gpus 0
#SBATCH --job-name gen-corpus
#SBATCH --time 4-00:00:00
#SBATCH --output ./slurm_out/%x_%j.out
#SBATCH -c 16
#SBATCH --mem 100G

# Example: generate Zipf corpus with 1% noise
python dataset_generation/generate_corpus.py \
    --profiles_json data/bioS_train.json \
    --out_dir data/corpora \
    --out_json zipf_100_1.json \
    --mode multiple-context \
    --zipf_s 1.0 \
    --noise_prob 0.01 \
    --do_build_dataset
