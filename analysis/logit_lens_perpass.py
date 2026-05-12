"""Logit lens in per-pass mode: probe the full next-token distribution at each
intermediate Coconut forward pass, not after all passes are done.

At thought_k, only k passes have completed. This mirrors the actual inference
dynamics rather than the final-state view.

Outputs:
  - Top-10 tokens + entropy at each step for first 5 examples (printed)
  - Entropy vs thinking step plot (saved)
  - JSON with full results
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, SCRIPT_DIR)

from coconut import Coconut
from transformers import AutoModelForCausalLM
from analysis_utils import setup_tokenizer


def load_coconut_model(checkpoint_path, model_id, tokenizer, device):
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.resize_token_embeddings(len(tokenizer))
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    embeddings = model.get_input_embeddings()
    target_id = tokenizer.convert_tokens_to_ids("<<")
    for token_id in [latent_id, start_id, end_id]:
        embeddings.weight.data[token_id] = embeddings.weight.data[target_id].clone()
    coconut_model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    coconut_model.load_state_dict(state_dict, strict=False)
    coconut_model.to(device)
    coconut_model.eval()
    return coconut_model


@torch.no_grad()
def run_perpass_logit_lens(model, tokenizer, data, n_examples, n_latent, device, top_k=10):
    """Mirror Coconut's multi-pass forward, probing logits after each pass."""
    base_model = model.base_causallm
    embedding = base_model.transformer.get_input_embeddings()

    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    results = []
    for idx in tqdm(range(min(n_examples, len(data))), desc="Per-pass logit lens"):
        example = data[idx]

        question_tokens = tokenizer.encode(
            example["question"] + "\n", add_special_tokens=True
        )
        input_ids = question_tokens + [start_id] + [latent_id] * n_latent + [end_id]
        input_ids_tensor = torch.tensor([input_ids], device=device)

        inputs_embeds = embedding(input_ids_tensor)
        attention_mask = torch.ones_like(input_ids_tensor)
        position_ids = torch.arange(len(input_ids), device=device).unsqueeze(0)

        start_latent_pos = len(question_tokens)
        first_latent_pos = start_latent_pos + 1
        latent_positions = [first_latent_pos + i for i in range(n_latent)]

        # Mirror Coconut multi-pass forward, probing after each pass
        next_compute_range = (0, first_latent_pos)
        kv_cache = None
        step_results = []

        for pass_idx in range(n_latent):
            if kv_cache is None:
                outputs = base_model(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0]: next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[
                        :, next_compute_range[0]: next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :, next_compute_range[0]: next_compute_range[1]
                    ],
                    output_hidden_states=True,
                    use_cache=True,
                )
                hidden_states_offset = 0
            else:
                past_kv = [
                    (k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :])
                    for k, v in kv_cache
                ]
                outputs = base_model(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0]: next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[:, :next_compute_range[1]],
                    position_ids=position_ids[
                        :, next_compute_range[0]: next_compute_range[1]
                    ],
                    past_key_values=past_kv,
                    output_hidden_states=True,
                    use_cache=True,
                )
                hidden_states_offset = next_compute_range[0]

            hidden_states = outputs.hidden_states[-1]
            kv_cache = outputs.past_key_values

            # Probe at the last position of this compute range
            probe_abs_pos = next_compute_range[1] - 1
            probe_rel_pos = probe_abs_pos - hidden_states_offset
            logits = outputs.logits[0, probe_rel_pos, :]
            probs = F.softmax(logits.float(), dim=-1)

            # Entropy
            entropy = -torch.sum(probs * torch.log(probs + 1e-12)).item()

            # Top-k tokens
            topk_probs, topk_ids = torch.topk(probs, k=top_k)
            top_tokens = [
                {"token": tokenizer.decode([tid.item()]), "token_id": tid.item(), "prob": p.item()}
                for tid, p in zip(topk_ids, topk_probs)
            ]

            step_results.append({
                "pass": pass_idx,
                "label": f"thought_{pass_idx + 1}" if pass_idx > 0 else "before_thinking",
                "entropy": entropy,
                "top_tokens": top_tokens,
            })

            # Feed continuous thought into next latent position
            token_idx = latent_positions[pass_idx]
            thought = hidden_states[0, token_idx - 1 - hidden_states_offset, :]
            tensor_list = [inputs_embeds[0, p, :] for p in range(inputs_embeds.shape[1])]
            tensor_list[token_idx] = thought
            inputs_embeds = torch.stack(tensor_list).unsqueeze(0)

            next_compute_range = (
                next_compute_range[1],
                len(input_ids) if pass_idx + 1 >= n_latent else next_compute_range[1] + 1,
            )

        # Final pass
        past_kv = [
            (k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :])
            for k, v in kv_cache
        ]
        outputs = base_model(
            inputs_embeds=inputs_embeds[
                :, next_compute_range[0]: next_compute_range[1], :
            ],
            attention_mask=attention_mask[:, :next_compute_range[1]],
            position_ids=position_ids[
                :, next_compute_range[0]: next_compute_range[1]
            ],
            past_key_values=past_kv,
            output_hidden_states=True,
            use_cache=True,
        )
        hidden_states_offset = next_compute_range[0]

        # Probe at the last latent position after final pass
        last_latent_rel = latent_positions[-1] - hidden_states_offset
        logits = outputs.logits[0, last_latent_rel, :]
        probs = F.softmax(logits.float(), dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12)).item()
        topk_probs, topk_ids = torch.topk(probs, k=top_k)
        top_tokens = [
            {"token": tokenizer.decode([tid.item()]), "token_id": tid.item(), "prob": p.item()}
            for tid, p in zip(topk_ids, topk_probs)
        ]
        step_results.append({
            "pass": n_latent,
            "label": f"thought_{n_latent}",
            "entropy": entropy,
            "top_tokens": top_tokens,
        })

        results.append({
            "idx": idx,
            "question": example["question"][:200],
            "answer": example["answer"],
            "steps": example["steps"],
            "step_results": step_results,
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/prosqa_valid.json")
    parser.add_argument("--n_examples", type=int, default=20)
    parser.add_argument("--n_latent", type=int, default=6)
    parser.add_argument("--model_id", type=str, default="openai-community/gpt2")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = setup_tokenizer(args.model_id)
    print("Loading Coconut model...")
    model = load_coconut_model(args.checkpoint, args.model_id, tokenizer, args.device)

    with open(args.data_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples")

    results = run_perpass_logit_lens(model, tokenizer, data, args.n_examples, args.n_latent, args.device)

    # Save JSON
    os.makedirs("analysis/results", exist_ok=True)
    out_json = "analysis/results/logit_lens_coconut_perpass.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_json}")

    # Print top-10 for first 5 examples
    print("\n" + "=" * 70)
    print("TOP-10 TOKENS AT EACH INTERMEDIATE PASS (first 5 examples)")
    print("=" * 70)
    for ex in results[:5]:
        print(f"\n{'=' * 60}")
        print(f"Example {ex['idx']}: {ex['question'][:100]}...")
        print(f"Answer: {ex['answer']}")
        print(f"CoT steps: {ex['steps']}")
        print(f"{'=' * 60}")
        for sr in ex["step_results"]:
            tok_str = "  ".join(f"{t['token']!r}({t['prob']:.3f})" for t in sr["top_tokens"])
            print(f"  {sr['label']} (H={sr['entropy']:.3f}): {tok_str}")

    # Plot entropy vs thinking step
    n_steps = len(results[0]["step_results"])
    entropies_by_step = [[] for _ in range(n_steps)]
    for ex in results:
        for i, sr in enumerate(ex["step_results"]):
            entropies_by_step[i].append(sr["entropy"])

    steps = np.arange(1, n_steps + 1)
    means = np.array([np.mean(e) for e in entropies_by_step])
    stds = np.array([np.std(e) for e in entropies_by_step])

    labels = [results[0]["step_results"][i]["label"] for i in range(n_steps)]

    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.errorbar(steps, means, yerr=stds, fmt="o-", color=sns.color_palette("colorblind")[0],
                capsize=4, linewidth=1.8, markersize=6)
    ax.set_xlabel("Latent Thought Step", fontsize=12)
    ax.set_ylabel("Entropy (nats)", fontsize=12)
    ax.set_title("Coconut Latent Token Entropy\n(Per-Pass Logit Lens)", fontsize=13, fontweight="bold")
    ax.set_xticks(steps)
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.tick_params(labelsize=10)
    sns.despine()

    plt.tight_layout()
    os.makedirs("analysis/figures", exist_ok=True)
    out_fig = "analysis/figures/latent_entropy_vs_step_perpass.pdf"
    fig.savefig(out_fig, bbox_inches="tight", dpi=150)
    fig.savefig(out_fig.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print(f"\nSaved plot to {out_fig}")


if __name__ == "__main__":
    main()
