#!/bin/bash
#SBATCH --job-name=smollm2-360m-coconut
#SBATCH --output=logs/smollm2_360m_coconut_%j.out
#SBATCH --error=logs/smollm2_360m_coconut_%j.err
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=long

cd /home/mila/m/michael.rizvi-martel/repos/coconut
source .venv/bin/activate

torchrun --nnodes 1 --nproc_per_node 4 \
    --master_port=$((29500 + SLURM_JOB_ID % 10000)) \
    run.py args/prosqa_smollm2_360m_coconut.yaml
