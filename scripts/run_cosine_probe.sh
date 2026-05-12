#!/bin/bash
#SBATCH --job-name=cosine-probe
#SBATCH --output=analysis/logs/cosine_probe_%j.out
#SBATCH --error=analysis/logs/cosine_probe_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=unkillable

cd /home/mila/m/michael.rizvi-martel/repos/coconut
source .venv/bin/activate

CKPT=/network/scratch/m/michael.rizvi-martel/coconut_checkpoints/prosqa-coconut/checkpoint_50

# Run on test set (500 examples) to match existing entity probing data
python analysis/analyze_coconut.py \
    --checkpoint $CKPT \
    --data_path data/prosqa_test.json \
    --output_dir analysis/results/test_cosine \
    --n_examples 500 \
    --n_latent 6 \
    --mode per_pass

echo "Done"
