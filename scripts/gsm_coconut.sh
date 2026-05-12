#!/bin/bash
#SBATCH --job-name=gsm-coconut
#SBATCH --output=logs/gsm_coconut_%j.out
#SBATCH --error=logs/gsm_coconut_%j.err
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00

# NOTE: Before submitting, update load_model_path in args/gsm_coconut.yaml
# to point at the best CoT checkpoint from the gsm-cot run.
# e.g. /network/scratch/m/michael.rizvi-martel/coconut_checkpoints/gsm-cot/checkpoint_X

cd /home/mila/m/michael.rizvi-martel/repos/coconut
source .venv/bin/activate

torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut.yaml
