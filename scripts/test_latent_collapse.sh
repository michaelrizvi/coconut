#!/bin/bash
#SBATCH --job-name=test-collapse-gpt2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=analysis/logs/test_collapse_%j.out
#SBATCH --error=analysis/logs/test_collapse_%j.err

cd /home/mila/m/michael.rizvi-martel/repos/coconut
source .venv/bin/activate

echo "=== Testing Coconut (fine-tuned GPT-2) ==="
python analysis/analyze_latent_collapse.py \
    --model_type coconut \
    --checkpoint_path /network/scratch/m/michael.rizvi-martel/coconut_checkpoints/prosqa-coconut/checkpoint_50 \
    --n_examples 5 \
    --output_dir analysis/results

echo ""
echo "=== Testing CoT (fine-tuned GPT-2) ==="
python analysis/analyze_latent_collapse.py \
    --model_type cot \
    --checkpoint_path /network/scratch/m/michael.rizvi-martel/coconut_checkpoints/prosqa-cot/checkpoint_49 \
    --n_examples 5 \
    --output_dir analysis/results

echo ""
echo "=== Done ==="
