"""Test that gradient norms are non-zero under real multi-GPU FSDP wrapping.

Reproduces the exact wrapping used in run.py: Coconut(GPT2) -> FSDP with
use_orig_params=True, LlamaDecoderLayer auto-wrap policy (no GPT2 sub-wrapping).
Uses summon_full_params to read per-parameter gradients.
"""

import os
import sys
import functools
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# Add parent dir so we can import from coconut repo
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from coconut import Coconut
from run import compute_grad_norms


def main():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"=== Multi-GPU Gradient Norm Test (world_size={world_size}) ===")

    # Build model exactly as run.py does
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    model.resize_token_embeddings(len(tokenizer))
    embeddings = model.get_input_embeddings()
    target_id = tokenizer.convert_tokens_to_ids("<<")
    for token_id in [latent_id, start_id, end_id]:
        embeddings.weight.data[token_id] = embeddings.weight.data[target_id]
        model.lm_head.weight.data[token_id] = model.lm_head.weight.data[target_id]

    # Wrap in Coconut, then FSDP — same as run.py
    model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)
    model = model.to(local_rank)

    llama_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer},
    )
    fsdp_model = FSDP(
        model, auto_wrap_policy=llama_auto_wrap_policy, device_id=local_rank,
        use_orig_params=True,
    )

    if rank == 0:
        print(f"FSDP wrapping: {type(fsdp_model)}")
        print(f"Sharding strategy: {fsdp_model.sharding_strategy}")

    # Dummy forward/backward through Coconut (no latent tokens = stage 0 CoT path)
    text = "Hello world this is a test of gradient norms"
    inputs = tokenizer(text, return_tensors="pt").to(local_rank)
    input_ids = inputs["input_ids"]
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    position_ids = torch.arange(input_ids.shape[1], device=local_rank).unsqueeze(0)

    outputs = fsdp_model(input_ids, attention_mask, labels, position_ids)
    outputs.loss.backward()

    # Test 1: Without summon_full_params (the old broken way)
    if rank == 0:
        print("\n--- Test 1: WITHOUT summon_full_params (old behavior) ---")
        gnorms_broken = compute_grad_norms(fsdp_model)
        for k, v in gnorms_broken.items():
            status = "OK" if v > 0 else "ZERO"
            print(f"  {k}: {v:.6f} -- {status}")
        n_broken_nonzero = sum(1 for v in gnorms_broken.values() if v > 0)

    # Test 2: With summon_full_params (the fix)
    if rank == 0:
        print("\n--- Test 2: WITH summon_full_params (new behavior) ---")
    with FSDP.summon_full_params(fsdp_model, writeback=False):
        gnorms_fixed = compute_grad_norms(fsdp_model)
    if rank == 0:
        for k, v in gnorms_fixed.items():
            status = "OK" if v > 0 else "ZERO (BUG STILL PRESENT)"
            print(f"  {k}: {v:.6f} -- {status}")
        n_fixed_nonzero = sum(1 for v in gnorms_fixed.values() if v > 0)

        print(f"\n=== VERDICT ===")
        print(f"Without summon_full_params: {n_broken_nonzero}/5 groups non-zero")
        print(f"With summon_full_params:    {n_fixed_nonzero}/5 groups non-zero")
        if n_fixed_nonzero == 5:
            print("PASS — summon_full_params fix works on multi-GPU FSDP")
        else:
            print("FAIL — gradient norms still broken")
            sys.exit(1)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
