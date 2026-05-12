#!/bin/bash
#SBATCH --job-name=prosqa-cot
#SBATCH --output=logs/prosqa_cot_%j.out
#SBATCH --error=logs/prosqa_cot_%j.err
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3:00:00

cd /home/mila/m/michael.rizvi-martel/repos/coconut
source .venv/bin/activate

torchrun --nnodes 1 --nproc_per_node 4 run.py args/prosqa_cot.yaml
