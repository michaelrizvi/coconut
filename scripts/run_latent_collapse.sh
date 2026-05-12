#!/bin/bash
#SBATCH --job-name=collapse-gpt2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=analysis/logs/collapse_gpt2_%j.out
#SBATCH --error=analysis/logs/collapse_gpt2_%j.err

cd /home/mila/m/michael.rizvi-martel/repos/coconut
source .venv/bin/activate

echo "=== Coconut (fine-tuned GPT-2) ==="
python analysis/analyze_latent_collapse.py \
    --model_type coconut \
    --checkpoint_path /network/scratch/m/michael.rizvi-martel/coconut_checkpoints/prosqa-coconut/checkpoint_50 \
    --output_dir analysis/results

echo ""
echo "=== CoT (fine-tuned GPT-2) ==="
python analysis/analyze_latent_collapse.py \
    --model_type cot \
    --checkpoint_path /network/scratch/m/michael.rizvi-martel/coconut_checkpoints/prosqa-cot/checkpoint_49 \
    --output_dir analysis/results

echo ""
echo "=== Done ==="
