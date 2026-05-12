#!/bin/bash
#SBATCH --job-name=coconut-no-latent
#SBATCH --output=logs/coconut_no_latent_%j.out
#SBATCH --error=logs/coconut_no_latent_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00

cd /home/mila/m/michael.rizvi-martel/repos/coconut
source .venv/bin/activate

python analysis/eval_coconut_no_latent.py \
    --checkpoint /network/scratch/m/michael.rizvi-martel/coconut_checkpoints/prosqa-coconut/checkpoint_50 \
    --data_path data/prosqa_test.json \
    --n_examples 500 \
    --output_path analysis/results/test/coconut_no_latent_eval.json
