#!/bin/bash
#SBATCH --job-name=analyze-coconut-test
#SBATCH --output=logs/analyze_coconut_test_%j.out
#SBATCH --error=logs/analyze_coconut_test_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00

cd /home/mila/m/michael.rizvi-martel/repos/coconut
source .venv/bin/activate

python analysis/analyze_coconut.py \
    --checkpoint /network/scratch/m/michael.rizvi-martel/coconut_checkpoints/prosqa-coconut/checkpoint_50 \
    --data_path data/prosqa_test.json \
    --n_examples 500 \
    --n_latent 6 \
    --mode per_pass \
    --output_dir analysis/results/test

python analysis/analyze_question_only.py \
    --coconut_checkpoint /network/scratch/m/michael.rizvi-martel/coconut_checkpoints/prosqa-coconut/checkpoint_50 \
    --cot_checkpoint /network/scratch/m/michael.rizvi-martel/coconut_checkpoints/prosqa-cot/checkpoint_49 \
    --data_path data/prosqa_test.json \
    --n_examples 500 \
    --output_dir analysis/results/test
