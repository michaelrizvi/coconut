#!/bin/bash
#SBATCH --job-name=test-gradnorm-fix
#SBATCH --output=logs/test_gradnorm_fix_%j.out
#SBATCH --error=logs/test_gradnorm_fix_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --partition=unkillable

cd /home/mila/m/michael.rizvi-martel/repos/coconut
source .venv/bin/activate

# Quick test: 1 GPU, debug mode, 1 epoch, check if grad norms are non-zero
python -c "
import torch
import torch.optim as optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import os, json

# Minimal single-process setup
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = str(29500 + int(os.environ.get('SLURM_JOB_ID', 0)) % 10000)
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
torch.distributed.init_process_group('nccl')
torch.cuda.set_device(0)

from run import compute_grad_norms
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('openai-community/gpt2').to(0)

# Wrap in FSDP with use_orig_params=True (the fix)
fsdp_model = FSDP(model, device_id=0, use_orig_params=True)

# Dummy forward/backward
tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
inputs = tokenizer('Hello world test', return_tensors='pt').to(0)
outputs = fsdp_model(**inputs, labels=inputs['input_ids'])
outputs.loss.backward()

gnorms = compute_grad_norms(fsdp_model)
print('=== GRAD NORM TEST RESULTS ===')
for k, v in gnorms.items():
    status = 'OK' if v > 0 else 'ZERO (BUG STILL PRESENT)'
    print(f'  {k}: {v:.6f} -- {status}')

all_nonzero = all(v > 0 for v in gnorms.values())
print(f'\\nVERDICT: {\"PASS - fix works\" if all_nonzero else \"FAIL - still broken\"}')

torch.distributed.destroy_process_group()
"
