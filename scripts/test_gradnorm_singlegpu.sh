#!/bin/bash
#SBATCH --job-name=test-gradnorm-1gpu
#SBATCH --output=logs/test_gradnorm_singlegpu_%j.out
#SBATCH --error=logs/test_gradnorm_singlegpu_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --partition=unkillable

cd /home/mila/m/michael.rizvi-martel/repos/coconut
source .venv/bin/activate

# Single-GPU test: FSDP falls back to NO_SHARD, but summon_full_params should still work
torchrun --nnodes 1 --nproc_per_node 1 --master_port=$((29500 + SLURM_JOB_ID % 10000)) \
    scripts/test_gradnorm_multigpu.py
