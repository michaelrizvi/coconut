#!/bin/bash
#SBATCH --job-name=test-gradnorm-multigpu
#SBATCH --output=logs/test_gradnorm_multigpu_%j.out
#SBATCH --error=logs/test_gradnorm_multigpu_%j.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --partition=main

cd /home/mila/m/michael.rizvi-martel/repos/coconut
source .venv/bin/activate

# Test gradient norm logging with actual multi-GPU FSDP (not NO_SHARD fallback)
# This reproduces the real training setup: torchrun + FSDP + Coconut wrapper
torchrun --nnodes 1 --nproc_per_node 2 --master_port=$((29500 + SLURM_JOB_ID % 10000)) \
    scripts/test_gradnorm_multigpu.py
