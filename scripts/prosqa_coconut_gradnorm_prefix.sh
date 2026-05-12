#!/bin/bash
#SBATCH --job-name=gradnorm-prefix
#SBATCH --output=logs/prosqa_coconut_gradnorm_prefix_%j.out
#SBATCH --error=logs/prosqa_coconut_gradnorm_prefix_%j.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --partition=long

cd /home/mila/m/michael.rizvi-martel/repos/coconut
source .venv/bin/activate

torchrun --nnodes 1 --nproc_per_node 4 --master_port=$((29500 + SLURM_JOB_ID % 10000)) \
    run.py args/prosqa_coconut_gradnorm_prefix.yaml
