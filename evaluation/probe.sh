#!/bin/bash
#SBATCH --gpus 1
#SBATCH --job-name probe
#SBATCH --time 4-00:00:00
#SBATCH --output ./slurm_out/%x_%j.out
#SBATCH -c 8
#SBATCH --mem 100G

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

python evaluation/probe.py
