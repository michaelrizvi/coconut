#!/bin/bash
#SBATCH --job-name=gsm-cot
#SBATCH --output=logs/gsm_cot_%j.out
#SBATCH --error=logs/gsm_cot_%j.err
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00

cd /home/mila/m/michael.rizvi-martel/repos/coconut
source .venv/bin/activate

torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_cot.yaml
