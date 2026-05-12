#!/bin/bash
#SBATCH --job-name=smollm2-coconut-debug
#SBATCH --output=logs/smollm2_coconut_debug_%j.out
#SBATCH --error=logs/smollm2_coconut_debug_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --partition=long

cd /home/mila/m/michael.rizvi-martel/repos/coconut
source .venv/bin/activate

torchrun --nnodes 1 --nproc_per_node 1 \
    --master_port=$((29500 + SLURM_JOB_ID % 10000)) \
    run.py args/prosqa_smollm2_coconut_debug.yaml
