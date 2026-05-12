#!/bin/bash
#SBATCH --job-name=test-gradnorm-smoke
#SBATCH --output=logs/test_gradnorm_smoke_%j.out
#SBATCH --error=logs/test_gradnorm_smoke_%j.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --partition=main

cd /home/mila/m/michael.rizvi-martel/repos/coconut
source .venv/bin/activate

# Smoke test: runs run.py itself (not a synthetic test) with the same 4-GPU FSDP
# setup as the real experiment. Uses debug=True (small data, no W&B, no checkpoints)
# for 2 epochs, then validates that grad_norms.json was written with non-zero values.
#
# This exercises the FULL code path:
#   - dataset loading + MyCollator
#   - Coconut forward pass (stage 0 = CoT warmup in epoch 1, stage 0 still in epoch 2)
#   - gradient accumulation loop
#   - summon_full_params(with_grads=True) inside training loop on all 4 ranks
#   - incremental JSON write after each epoch
#   - eval generation loop with FSDP.summon_full_params for generate()

SAVE_DIR=$SLURM_TMPDIR/gradnorm-smoke

cat > $SLURM_TMPDIR/smoke.yaml <<EOF
project: coconut
save_path: $SLURM_TMPDIR
name: gradnorm-smoke

only_eval: False
coconut: True
cot: False
no_thoughts: False
no_cot: False

c_thought: 1
epochs_per_stage: 1
max_latent_stage: 2
pad_latent_to_max: True

save_only_improve: False
uniform_prob: 0.0
model_id: openai-community/gpt2
load_model_path: None
seed: 0
resume: 0
bf16: False
train_path: data/prosqa_train.json
val_path: data/prosqa_valid.json
reset_optimizer: True
batch_size_training: 16
debug: True
gradient_accumulation_steps: 2
num_epochs: 2
lr: !!float "1e-4"
weight_decay: 0.01

log_grad_norms: True
EOF

echo "=== Smoke test: 4-GPU run.py with log_grad_norms, 2 epochs ==="
torchrun --nnodes 1 --nproc_per_node 4 --master_port=$((29500 + SLURM_JOB_ID % 10000)) \
    run.py $SLURM_TMPDIR/smoke.yaml

EXIT_CODE=$?
echo "=== torchrun exit code: $EXIT_CODE ==="

if [ $EXIT_CODE -ne 0 ]; then
    echo "FAIL: training crashed"
    exit 1
fi

# Validate grad_norms.json
echo "=== Validating grad_norms.json ==="
python3 - <<'PYEOF'
import json, sys

path = __import__('os').environ['SLURM_TMPDIR'] + '/gradnorm-smoke/grad_norms.json'

try:
    data = json.load(open(path))
except FileNotFoundError:
    print(f"FAIL: {path} not found")
    sys.exit(1)

print(f"Records: {len(data)}")
print(f"Epochs covered: {sorted(set(r['epoch'] for r in data))}")

if len(data) == 0:
    print("FAIL: no records written")
    sys.exit(1)

for group in ['wte', 'wpe', 'attn', 'mlp', 'ln']:
    vals = [r[group] for r in data]
    any_nonzero = any(v > 0 for v in vals)
    max_val = max(vals)
    status = "OK" if any_nonzero else "FAIL (all zero)"
    print(f"  {group}: max={max_val:.4f} -- {status}")
    if not any_nonzero:
        print(f"FAIL: {group} gradient norms are all zero")
        sys.exit(1)

print("PASS: all groups have nonzero gradient norms")
PYEOF
