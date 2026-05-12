#!/bin/bash
#SBATCH --job-name=tf-belief-prontoqa
#SBATCH --output=logs/tf_belief_prontoqa_%j.out
#SBATCH --error=logs/tf_belief_prontoqa_%j.err
#SBATCH --gres=gpu:48gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00

cd /home/mila/m/michael.rizvi-martel/repos/coconut
source .venv/bin/activate

python analysis/analyze_true_false_prontoqa.py
