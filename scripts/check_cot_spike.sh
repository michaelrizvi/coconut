#!/bin/bash
#SBATCH --job-name=cot-spike-check
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=analysis/logs/cot_spike_%j.out
#SBATCH --error=analysis/logs/cot_spike_%j.err

cd /home/mila/m/michael.rizvi-martel/repos/coconut
source .venv/bin/activate

python analysis/check_cot_last_layer_spike.py
