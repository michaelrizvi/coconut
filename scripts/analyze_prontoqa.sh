#!/bin/bash
#SBATCH --job-name=analyze-prontoqa
#SBATCH --output=logs/analyze_prontoqa_%j.out
#SBATCH --error=logs/analyze_prontoqa_%j.err
#SBATCH --gres=gpu:48gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00

cd /home/mila/m/michael.rizvi-martel/repos/coconut
source .venv/bin/activate

COCONUT_CKPT=/network/scratch/m/michael.rizvi-martel/coconut_checkpoints/prontoqa-coconut/checkpoint_50
COT_CKPT=/network/scratch/m/michael.rizvi-martel/coconut_checkpoints/prontoqa-cot/checkpoint_25
DATA=data/prontoqa_valid.json
TEST_DATA=data/prontoqa_test.json
OUT=analysis/results/prontoqa

echo "=== 1/4: eval_coconut_no_latent ==="
python analysis/eval_coconut_no_latent.py \
    --checkpoint $COCONUT_CKPT \
    --data_path $TEST_DATA \
    --output_path $OUT/no_latent_accuracy.json

echo "=== 2/4: analyze_coconut (per-pass) ==="
python analysis/analyze_coconut.py \
    --checkpoint $COCONUT_CKPT \
    --data_path $DATA \
    --n_latent 6 \
    --n_examples 200 \
    --output_dir $OUT

echo "=== 3/4: analyze_cot ==="
python analysis/analyze_cot.py \
    --checkpoint $COT_CKPT \
    --data_path $DATA \
    --n_examples 200 \
    --output_dir $OUT

echo "=== 4/4: analyze_question_only ==="
python analysis/analyze_question_only.py \
    --coconut_checkpoint $COCONUT_CKPT \
    --cot_checkpoint $COT_CKPT \
    --data_path $DATA \
    --n_examples 200 \
    --output_dir $OUT

echo "=== All analyses complete ==="
