#!/bin/bash
#SBATCH --job-name=logit-lens-perpass
#SBATCH --output=logs/logit_lens_perpass_%j.out
#SBATCH --error=logs/logit_lens_perpass_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --partition=unkillable

cd /home/mila/m/michael.rizvi-martel/repos/coconut
source .venv/bin/activate

CKPT=/network/scratch/m/michael.rizvi-martel/coconut_checkpoints/prosqa-coconut/checkpoint_50

python3 analysis/logit_lens_perpass.py \
    --checkpoint $CKPT \
    --data_path data/prosqa_valid.json \
    --n_examples 20 \
    --n_latent 6
