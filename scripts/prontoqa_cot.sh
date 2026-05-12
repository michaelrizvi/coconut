#!/bin/bash
#SBATCH --job-name=prontoqa-cot
#SBATCH --output=logs/prontoqa_cot_%j.out
#SBATCH --error=logs/prontoqa_cot_%j.err
#SBATCH --gres=gpu:48gb:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3:00:00

cd /home/mila/m/michael.rizvi-martel/repos/coconut
source .venv/bin/activate

torchrun --nnodes 1 --nproc_per_node 4 --master_port=$((29500 + SLURM_JOB_ID % 10000)) run.py args/prontoqa_cot.yaml
