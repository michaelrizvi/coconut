#!/bin/bash
#SBATCH --job-name=smollm2-135m-no-latent-eval
#SBATCH --output=logs/smollm2_135m_no_latent_eval_%j.out
#SBATCH --error=logs/smollm2_135m_no_latent_eval_%j.err
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --partition=long

cd /home/mila/m/michael.rizvi-martel/repos/coconut
source .venv/bin/activate

python analysis/eval_coconut_no_latent.py \
    --checkpoint /network/scratch/m/michael.rizvi-martel/coconut_checkpoints/prosqa-smollm2-135m-coconut/checkpoint_32 \
    --data_path data/prosqa_valid.json \
    --model_id HuggingFaceTB/SmolLM2-135M \
    --n_examples 300 \
    --output_path analysis/results/smollm2_135m_no_latent_eval.json
