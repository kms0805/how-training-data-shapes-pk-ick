#!/bin/bash
#SBATCH --gpus 1
#SBATCH --job-name train-gpt2
#SBATCH --time 4-00:00:00
#SBATCH --output ./slurm_out/%x_%j.out
#SBATCH -c 8
#SBATCH --mem 100G

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Example: multiple-context, Zipf s=1.0, 1% noise
python training/train.py \
    --profiles_json data/bioS_train.json \
    --mode multiple-context \
    --zipf_s 1.0 \
    --noise_prob 0.01 \
    --samples_per_epoch 300000
