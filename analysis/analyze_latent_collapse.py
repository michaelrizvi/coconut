"""
Measure logit-lens entropy at every layer at reasoning positions (latent/CoT).

Tests whether pretrained models show last-layer entropy collapse at latent
positions despite no direct prediction loss there.

Usage:
    # Coconut model (latent positions)
    python analysis/analyze_latent_collapse.py \
        --model_type coconut \
        --checkpoint_path /network/scratch/.../prosqa-coconut/checkpoint_50

    # CoT model (reasoning step positions)
    python analysis/analyze_latent_collapse.py \
        --model_type cot \
        --checkpoint_path /network/scratch/.../prosqa-cot/checkpoint_49
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent dir so we can import coconut, dataset
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from coconut import Coconut
from dataset import get_dataset, get_cot_latent_dataset, MyCollator


def apply_logit_lens(hidden_state, ln_f, lm_head):
    """Apply logit lens: layernorm -> lm_head -> softmax -> entropy.

    Args:
        hidden_state: [hidden_dim] or [1, hidden_dim]
        ln_f: Final layer norm (GPT-2: model.transformer.ln_f)
        lm_head: Unembedding head (GPT-2: model.lm_head)

    Returns:
        entropy: Shannon entropy in nats
    """
    if hidden_state.dim() == 1:
        hidden_state = hidden_state.unsqueeze(0)

    normed = ln_f(hidden_state.float())
    logits = F.linear(normed, lm_head.weight.float(), getattr(lm_head, 'bias', None))
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).item()
    return entropy


@torch.no_grad()
def analyze_coconut(model, tokenizer, val_path, device, n_examples=None):
    """Run Coconut forward and capture hidden states at latent positions via hook."""

    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    # Wrap in Coconut
    coconut_model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)
    coconut_model.eval()
    coconut_model.to(device)

    ln_f = model.transformer.ln_f
    lm_head = model.lm_head
    num_layers = model.config.n_layer

    # Build dataset at max stage (all latent tokens)
    class MinConfig:
        c_thought = 1
        max_latent_stage = 6
        pad_latent_to_max = True
        uniform_prob = 0.0
        no_cot = False
        debug = False

    base_dataset = get_dataset(val_path, tokenizer)
    dataset = get_cot_latent_dataset(
        scheduled_stage=6,  # max stage: all CoT replaced by latent
        base_dataset=base_dataset,
        configs=MinConfig(),
        start_id=start_id,
        latent_id=latent_id,
        end_id=end_id,
    )

    if n_examples is not None:
        # Dataset is a HF Dataset, select subset
        dataset = dataset.select(range(min(n_examples, len(dataset))))

    # Hook to capture hidden states from each base_causallm forward call
    captured_passes = []

    def hook_fn(module, args, output):
        if hasattr(output, 'hidden_states') and output.hidden_states is not None:
            captured_passes.append(tuple(h.detach().cpu() for h in output.hidden_states))

    handle = coconut_model.base_causallm.register_forward_hook(hook_fn)

    all_results = []

    for i in range(len(dataset)):
        sample = dataset[i]
        input_ids = torch.tensor([sample["input_ids"]], device=device)
        attention_mask = torch.tensor([sample["attention_mask"]], device=device)
        labels = torch.tensor([sample["labels"]], device=device)
        position_ids = torch.tensor([sample["position_ids"]], device=device)

        # Find latent positions
        latent_positions = (input_ids[0] == latent_id).nonzero(as_tuple=True)[0].tolist()

        if len(latent_positions) == 0:
            continue

        # Run Coconut forward (hook captures hidden states from each pass)
        captured_passes.clear()
        coconut_model.forward(input_ids, attention_mask, labels, position_ids)

        # captured_passes[0] = pre-latent pass
        # captured_passes[k] for k=1..n_latent = pass computing latent k
        # captured_passes[-1] = final pass (post-latent)
        # For latent k, the hidden state used as its embedding is the last position
        # of captured_passes[k] (the pass that ends just before latent position k).

        n_latent = len(latent_positions)
        example_results = {"n_latent": n_latent, "positions": []}

        for lat_idx in range(n_latent):
            pass_idx = lat_idx + 1  # pass 0 is pre-latent
            if pass_idx >= len(captured_passes):
                break

            hidden_states_tuple = captured_passes[pass_idx]

            layer_entropies = []
            for layer_idx in range(num_layers):
                # hidden_states_tuple[layer_idx+1] = output of layer layer_idx
                # Last position of this pass = the representation fed back as latent embedding
                h = hidden_states_tuple[layer_idx + 1][0, -1, :]
                entropy = apply_logit_lens(h.to(device), ln_f, lm_head)
                layer_entropies.append(entropy)

            example_results["positions"].append({
                "latent_idx": lat_idx,
                "layer_entropies": layer_entropies,
            })

        all_results.append(example_results)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(dataset)} examples")

    handle.remove()
    return all_results


@torch.no_grad()
def analyze_cot(model, tokenizer, val_path, device, n_examples=None):
    """Run CoT forward and capture hidden states at reasoning step positions."""

    model.eval()
    model.to(device)

    ln_f = model.transformer.ln_f
    lm_head = model.lm_head
    num_layers = model.config.n_layer

    # Build CoT dataset (stage 0 = no latent tokens, pure CoT)
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

    base_dataset = get_dataset(val_path, tokenizer)
    # Stage 0 = no latent tokens, full CoT kept as text
    dataset = get_cot_latent_dataset(
        scheduled_stage=0,
        base_dataset=base_dataset,
        configs=MinConfig(),
        start_id=start_id,
        latent_id=latent_id,
        end_id=end_id,
        no_special_marker=True,  # CoT mode: no <|start-latent|> markers
    )

    if n_examples is not None:
        dataset = dataset.select(range(min(n_examples, len(dataset))))

    all_results = []

    for i in range(len(dataset)):
        sample = dataset[i]
        input_ids = torch.tensor([sample["input_ids"]], device=device)

        # Reasoning positions = where labels != -100
        label_seq = sample["labels"]
        reasoning_positions = [
            pos for pos, lab in enumerate(label_seq) if lab != -100
        ]

        if len(reasoning_positions) == 0:
            continue

        # Forward pass with hidden states
        outputs = model(input_ids=input_ids, output_hidden_states=True)
        hidden_states_tuple = outputs.hidden_states

        example_results = {"n_positions": len(reasoning_positions), "positions": []}

        for pos in reasoning_positions:
            layer_entropies = []
            for layer_idx in range(num_layers):
                h = hidden_states_tuple[layer_idx + 1][0, pos, :]
                entropy = apply_logit_lens(h, ln_f, lm_head)
                layer_entropies.append(entropy)

            example_results["positions"].append({
                "position": pos,
                "layer_entropies": layer_entropies,
            })

        all_results.append(example_results)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(dataset)} examples")

    return all_results


def summarize_results(results, num_layers):
    """Aggregate per-layer entropy across all positions and examples."""
    layer_entropies = [[] for _ in range(num_layers)]

    for example in results:
        for pos_data in example["positions"]:
            for layer_idx, ent in enumerate(pos_data["layer_entropies"]):
                layer_entropies[layer_idx].append(ent)

    summary = {}
    for layer_idx in range(num_layers):
        vals = layer_entropies[layer_idx]
        if vals:
            summary[str(layer_idx)] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n": len(vals),
            }

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["coconut", "cot"], required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--val_path", default="data/prosqa_valid.json")
    parser.add_argument("--output_dir", default="analysis/results")
    parser.add_argument("--n_examples", type=int, default=None,
                        help="Limit number of examples (for testing)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading {args.model_type} model from {args.checkpoint_path}")
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    # Add special tokens (needed for tokenizer in both modes)
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["<|start-latent|>", "<|latent|>", "<|end-latent|>"]
    })

    # Load checkpoint
    state_dict = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    # Handle coconut wrapper keys
    if any(k.startswith("base_causallm") for k in state_dict.keys()):
        state_dict = {
            k.replace("base_causallm.", ""): v
            for k, v in state_dict.items()
            if k.startswith("base_causallm.")
        }

    # Resize embeddings to match checkpoint (CoT checkpoints have 50257, Coconut have 50260)
    ckpt_vocab_size = state_dict["transformer.wte.weight"].shape[0]
    model.resize_token_embeddings(ckpt_vocab_size)
    print(model.load_state_dict(state_dict, strict=False))

    # If CoT checkpoint had smaller vocab, re-resize to include special tokens
    if ckpt_vocab_size < len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)

    # Run analysis
    num_layers = model.config.n_layer
    print(f"\nModel: {num_layers} layers")
    print(f"Running {args.model_type} analysis...")

    if args.model_type == "coconut":
        results = analyze_coconut(model, tokenizer, args.val_path, args.device, args.n_examples)
    else:
        results = analyze_cot(model, tokenizer, args.val_path, args.device, args.n_examples)

    # Summarize
    summary = summarize_results(results, num_layers)

    print(f"\nEntropy by layer ({args.model_type}, {len(results)} examples):")
    for layer_idx in range(num_layers):
        s = summary.get(str(layer_idx), {})
        if s:
            print(f"  Layer {layer_idx:2d}: {s['mean']:.3f} +/- {s['std']:.3f} (n={s['n']})")

    # Save
    output = {
        "model_type": args.model_type,
        "checkpoint_path": args.checkpoint_path,
        "num_layers": num_layers,
        "num_examples": len(results),
        "summary": summary,
        "raw_results": results,
    }

    out_path = os.path.join(args.output_dir, f"latent_collapse_{args.model_type}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
