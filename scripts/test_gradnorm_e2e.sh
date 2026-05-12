#!/bin/bash
#SBATCH --job-name=test-gradnorm-e2e
#SBATCH --output=logs/test_gradnorm_e2e_%j.out
#SBATCH --error=logs/test_gradnorm_e2e_%j.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --partition=main

cd /home/mila/m/michael.rizvi-martel/repos/coconut
source .venv/bin/activate

# End-to-end test: 1 epoch of actual Coconut training with grad norm logging
# Uses debug mode (small data subset, no W&B, no checkpoint saving)
# Writes a temp config that exits after 1 epoch

TMPDIR_TEST=$(mktemp -d)
cat > "$TMPDIR_TEST/test_gradnorm_e2e.yaml" <<EOF
project: coconut
save_path: $TMPDIR_TEST
name: gradnorm-e2e-test

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

echo "=== Config ==="
cat "$TMPDIR_TEST/test_gradnorm_e2e.yaml"
echo ""
echo "=== Running 2 epochs with grad norm logging ==="

torchrun --nnodes 1 --nproc_per_node 2 --master_port=$((29500 + SLURM_JOB_ID % 10000)) \
    run.py "$TMPDIR_TEST/test_gradnorm_e2e.yaml"

echo ""
echo "=== Checking grad_norms.json ==="
GRAD_FILE="$TMPDIR_TEST/gradnorm-e2e-test/grad_norms.json"
if [ -f "$GRAD_FILE" ]; then
    echo "File exists: $GRAD_FILE"
    python3 -c "
import json, sys
data = json.load(open('$GRAD_FILE'))
print(f'Total records: {len(data)}')
if len(data) == 0:
    print('FAIL: no records')
    sys.exit(1)
epochs = sorted(set(r['epoch'] for r in data))
print(f'Epochs: {epochs}')
for r in data[:5]:
    print(f'  epoch={r[\"epoch\"]} stage={r[\"stage\"]} wte={r[\"wte\"]:.4f} attn={r[\"attn\"]:.4f} mlp={r[\"mlp\"]:.4f} ln={r[\"ln\"]:.4f} wpe={r[\"wpe\"]:.4f}')
n_nonzero = sum(1 for r in data if any(r[k] > 0 for k in ['wte','wpe','attn','mlp','ln']))
n_total = len(data)
print(f'Records with any nonzero grad: {n_nonzero}/{n_total}')
if n_nonzero == 0:
    print('FAIL: all gradient norms are zero')
    sys.exit(1)
# Check all groups have nonzero at least once
for group in ['wte','wpe','attn','mlp','ln']:
    has_nonzero = any(r[group] > 0 for r in data)
    status = 'OK' if has_nonzero else 'NEVER NONZERO'
    print(f'  {group}: {status}')
    if not has_nonzero:
        print(f'FAIL: {group} never had nonzero gradient')
        sys.exit(1)
print('PASS: all parameter groups have nonzero gradients')
"
else
    echo "FAIL: $GRAD_FILE not found"
    exit 1
fi

# Cleanup
rm -rf "$TMPDIR_TEST"
