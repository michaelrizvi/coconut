#!/bin/bash
#SBATCH --job-name=layerwise-belief
#SBATCH --output=logs/layerwise_belief_%j.out
#SBATCH --error=logs/layerwise_belief_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --partition=long

COCONUT_CKPT=/network/scratch/m/michael.rizvi-martel/coconut_checkpoints/prosqa-coconut/checkpoint_50
COT_CKPT=/network/scratch/m/michael.rizvi-martel/coconut_checkpoints/prosqa-cot/checkpoint_49
DATA=data/prosqa_test.json
OUTDIR=analysis/results/prosqa
FIGDIR=analysis/figures/layerwise

cd /home/mila/m/michael.rizvi-martel/repos/coconut
source .venv/bin/activate

mkdir -p "$OUTDIR" "$FIGDIR"

echo "=== Coconut layer-wise belief ==="
python analysis/analyze_layerwise_belief.py \
    --model_type coconut \
    --checkpoint "$COCONUT_CKPT" \
    --data_path "$DATA" \
    --n_examples 500 \
    --n_latent 6 \
    --output_dir "$OUTDIR" \
    --output_name layerwise_belief_coconut

echo "=== CoT layer-wise belief ==="
python analysis/analyze_layerwise_belief.py \
    --model_type cot \
    --checkpoint "$COT_CKPT" \
    --data_path "$DATA" \
    --n_examples 500 \
    --output_dir "$OUTDIR" \
    --output_name layerwise_belief_cot

echo "=== Generating figures (all steps) ==="
python analysis/visualize_layerwise.py \
    --coconut_json "$OUTDIR/layerwise_belief_coconut.json" \
    --cot_json     "$OUTDIR/layerwise_belief_cot.json" \
    --output_dir   "$FIGDIR"

echo "=== Generating figures (4-step only) ==="
python analysis/visualize_layerwise.py \
    --coconut_json "$OUTDIR/layerwise_belief_coconut.json" \
    --cot_json     "$OUTDIR/layerwise_belief_cot.json" \
    --output_dir   "$FIGDIR" \
    --n_steps 4

echo "=== Generating figures (5-step only) ==="
python analysis/visualize_layerwise.py \
    --coconut_json "$OUTDIR/layerwise_belief_coconut.json" \
    --cot_json     "$OUTDIR/layerwise_belief_cot.json" \
    --output_dir   "$FIGDIR" \
    --n_steps 5

echo "Done."
