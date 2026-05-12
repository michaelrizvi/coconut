#!/bin/bash
#SBATCH --job-name=prosqa-coconut-gradnorm
#SBATCH --output=logs/prosqa_coconut_gradnorm_%j.out
#SBATCH --error=logs/prosqa_coconut_gradnorm_%j.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=long

cd /home/mila/m/michael.rizvi-martel/repos/coconut
source .venv/bin/activate

torchrun --nnodes 1 --nproc_per_node 4 --master_port=$((29500 + SLURM_JOB_ID % 10000)) \
    run.py args/prosqa_coconut_gradnorm.yaml
