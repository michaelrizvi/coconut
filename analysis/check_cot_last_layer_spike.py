"""
Quick check: does the CoT model's top-1 prediction change between layer 10 and 11?
The entropy spikes at layer 11 (0.08 -> 0.62); this checks whether the model
stays committed to the same token or switches.
"""

import json
import os
import sys

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import get_dataset, get_cot_latent_dataset


def logit_lens_topk(hidden_state, ln_f, lm_head, k=5):
    """Apply logit lens and return top-k tokens with probs + entropy."""
    if hidden_state.dim() == 1:
        hidden_state = hidden_state.unsqueeze(0)
    normed = ln_f(hidden_state.float())
    logits = F.linear(normed, lm_head.weight.float(), getattr(lm_head, 'bias', None))
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).item()
    topk_probs, topk_ids = torch.topk(probs, k, dim=-1)
    return entropy, topk_ids[0].tolist(), topk_probs[0].tolist()


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["<|start-latent|>", "<|latent|>", "<|end-latent|>"]
    })

    ckpt = "/network/scratch/m/michael.rizvi-martel/coconut_checkpoints/prosqa-cot/checkpoint_49"
    state_dict = torch.load(ckpt, map_location="cpu", weights_only=False)
    ckpt_vocab_size = state_dict["transformer.wte.weight"].shape[0]
    model.resize_token_embeddings(ckpt_vocab_size)
    model.load_state_dict(state_dict, strict=False)
    if ckpt_vocab_size < len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.eval()

    ln_f = model.transformer.ln_f
    lm_head = model.lm_head

    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    class MinConfig:
        c_thought = 1
        max_latent_stage = 6
        pad_latent_to_max = True
        uniform_prob = 0.0
        no_cot = False
        debug = False

    base_dataset = get_dataset("data/prosqa_valid.json", tokenizer)
    dataset = get_cot_latent_dataset(
        scheduled_stage=0, base_dataset=base_dataset, configs=MinConfig(),
        start_id=start_id, latent_id=latent_id, end_id=end_id,
        no_special_marker=True,
    )

    # Check first 50 examples
    n_examples = min(50, len(dataset))
    same_top1 = 0
    diff_top1 = 0
    total = 0

    for i in range(n_examples):
        sample = dataset[i]
        input_ids = torch.tensor([sample["input_ids"]], device=device)
        label_seq = sample["labels"]
        reasoning_positions = [pos for pos, lab in enumerate(label_seq) if lab != -100]

        if not reasoning_positions:
            continue

        outputs = model(input_ids=input_ids, output_hidden_states=True)
        hs = outputs.hidden_states

        for pos in reasoning_positions:
            h10 = hs[11][0, pos, :]  # layer 10 output
            h11 = hs[12][0, pos, :]  # layer 11 output

            ent10, top5_ids_10, top5_probs_10 = logit_lens_topk(h10, ln_f, lm_head)
            ent11, top5_ids_11, top5_probs_11 = logit_lens_topk(h11, ln_f, lm_head)

            if top5_ids_10[0] == top5_ids_11[0]:
                same_top1 += 1
            else:
                diff_top1 += 1
                if total < 10:  # Print first few examples where they differ
                    tok10 = tokenizer.decode([top5_ids_10[0]])
                    tok11 = tokenizer.decode([top5_ids_11[0]])
                    print(f"  Pos {pos}: L10 top1='{tok10}' (p={top5_probs_10[0]:.3f}, H={ent10:.3f}) "
                          f"-> L11 top1='{tok11}' (p={top5_probs_11[0]:.3f}, H={ent11:.3f})")
                    print(f"    L10 top5: {[tokenizer.decode([t]) for t in top5_ids_10]}")
                    print(f"    L11 top5: {[tokenizer.decode([t]) for t in top5_ids_11]}")
            total += 1

    print(f"\nSummary ({total} positions across {n_examples} examples):")
    print(f"  Same top-1 at L10 and L11: {same_top1} ({100*same_top1/total:.1f}%)")
    print(f"  Different top-1:           {diff_top1} ({100*diff_top1/total:.1f}%)")


if __name__ == "__main__":
    main()
