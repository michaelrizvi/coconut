#!/usr/bin/env python3
"""Logit lens analysis for Coconut and CoT models.

At each thinking position (latent tokens for Coconut, step-end tokens for CoT),
extracts hidden states at every transformer layer, projects through the final
layer norm + LM head, and computes the entropy of the resulting distribution.

Works with both ProsQA and GSM8K datasets.

Usage:
    # Coconut on ProsQA
    python run_logit_lens.py --model_type coconut \
        --checkpoint <path> --data_path data/prosqa_valid.json --n_latent 6

    # CoT on GSM8K
    python run_logit_lens.py --model_type cot \
        --checkpoint <path> --data_path data/gsm_valid.json
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, SCRIPT_DIR)

from coconut import Coconut
from transformers import AutoModelForCausalLM

from analysis_utils import setup_tokenizer


# ---- Model loading (reused from existing analysis scripts) ----


def load_coconut_model(checkpoint_path, model_id, tokenizer, device):
    """Load a Coconut model from checkpoint."""
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


def load_cot_model(checkpoint_path, model_id, device):
    """Load a CoT baseline model from checkpoint."""
    model = AutoModelForCausalLM.from_pretrained(model_id)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


# ---- Logit lens core ----


def apply_logit_lens(hidden_state, ln_f, lm_head, tokenizer, top_k=5):
    """Project a hidden state to vocabulary space via final LN + LM head.

    Args:
        hidden_state: (hidden_dim,) or (1, hidden_dim)
        ln_f: final layer norm module
        lm_head: vocabulary projection module
        tokenizer: for decoding top-k token ids
        top_k: number of top tokens to record

    Returns:
        entropy: float (nats)
        top_tokens: list of {"token": str, "token_id": int, "prob": float}
    """
    if hidden_state.dim() == 1:
        hidden_state = hidden_state.unsqueeze(0)

    logits = lm_head(ln_f(hidden_state))  # (1, vocab_size)
    probs = F.softmax(logits.float(), dim=-1)

    entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).item()

    topk_probs, topk_ids = torch.topk(probs[0], k=top_k)
    top_tokens = [
        {
            "token": tokenizer.decode([tid.item()]),
            "token_id": tid.item(),
            "prob": p.item(),
        }
        for tid, p in zip(topk_ids, topk_probs)
    ]
    return entropy, top_tokens


def get_gpt2_components(model):
    """Extract ln_f and lm_head from a GPT-2 model or Coconut wrapper."""
    base = model.base_causallm if isinstance(model, Coconut) else model
    return base.transformer.ln_f, base.lm_head


# ---- Coconut logit lens ----


@torch.no_grad()
def run_coconut_logit_lens(model, tokenizer, data, n_examples, n_latent, device):
    """Run logit lens on Coconut model at each latent thought position.

    For each example:
      1. Construct input: question + <start_latent> + n_latent * <latent> + <end_latent>
      2. Run Coconut.forward() to get final inputs_embeds (continuous thoughts plugged in)
      3. Re-run base GPT-2 on final embeds with output_hidden_states=True
      4. At each thought position × each layer, apply logit lens → entropy
    """
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    ln_f, lm_head = get_gpt2_components(model)

    results = []
    for idx in tqdm(range(min(n_examples, len(data))), desc="Coconut logit lens"):
        example = data[idx]

        question_tokens = tokenizer.encode(
            example["question"] + "\n", add_special_tokens=True
        )
        input_ids = question_tokens + [start_id] + [latent_id] * n_latent + [end_id]
        input_ids_tensor = torch.tensor([input_ids], device=device)

        # Thought positions (indices of <latent> tokens)
        start_latent_pos = len(question_tokens)
        thought_positions = [start_latent_pos + 1 + i for i in range(n_latent)]

        # Run Coconut multi-pass forward to get final inputs_embeds
        attention_mask = torch.ones_like(input_ids_tensor)
        labels = input_ids_tensor.clone()
        position_ids = torch.arange(len(input_ids), device=device).unsqueeze(0)

        outputs = model.forward(input_ids_tensor, attention_mask, labels, position_ids)
        final_embeds = outputs.inputs_embeds

        # Re-run base model to get hidden states at every layer
        base_outputs = model.base_causallm(
            inputs_embeds=final_embeds, output_hidden_states=True
        )
        # hidden_states: tuple of (n_layers + 1) tensors, each (1, seq_len, hidden_dim)
        # [0] = embedding output, [1..n_layers] = transformer block outputs
        hidden_states = base_outputs.hidden_states
        n_layers = len(hidden_states) - 1

        # Apply logit lens at each thought position × layer
        entropy_matrix = []  # (n_positions, n_layers)
        top_tokens_matrix = []
        for pos in thought_positions:
            pos_entropies = []
            pos_top_tokens = []
            for layer_idx in range(n_layers):
                h = hidden_states[layer_idx + 1][0, pos, :]
                ent, top_toks = apply_logit_lens(h, ln_f, lm_head, tokenizer)
                pos_entropies.append(ent)
                pos_top_tokens.append(top_toks)
            entropy_matrix.append(pos_entropies)
            top_tokens_matrix.append(pos_top_tokens)

        results.append(
            {
                "idx": idx,
                "question": example["question"][:200],
                "answer": example["answer"],
                "n_steps": len(example["steps"]),
                "n_positions": len(thought_positions),
                "position_labels": [f"thought_{i+1}" for i in range(n_latent)],
                "entropy": entropy_matrix,
                "top_tokens": top_tokens_matrix,
            }
        )

        if idx == 0:
            print(f"\n--- Sanity check (example 0) ---")
            print(f"Sequence length: {len(input_ids)}")
            print(f"Thought positions: {thought_positions}")
            print(f"Num layers: {n_layers}")
            print(f"Entropy at thought_1, layer 0: {entropy_matrix[0][0]:.4f}")
            print(f"Entropy at thought_1, last layer: {entropy_matrix[0][-1]:.4f}")

    return results, n_layers


# ---- CoT logit lens ----


@torch.no_grad()
def run_cot_logit_lens(model, tokenizer, data, n_examples, device):
    """Run logit lens on CoT model at the last token of each reasoning step.

    For each example:
      1. Construct teacher-forced input: question + all CoT step tokens
      2. Single forward pass with output_hidden_states=True
      3. At each step's last token × each layer, apply logit lens → entropy
    """
    ln_f = model.transformer.ln_f
    lm_head = model.lm_head

    results = []
    for idx in tqdm(range(min(n_examples, len(data))), desc="CoT logit lens"):
        example = data[idx]

        question_tokens = tokenizer.encode(
            example["question"] + "\n", add_special_tokens=True
        )
        steps_tokens = []
        for step in example["steps"]:
            step_toks = tokenizer.encode(step + "\n", add_special_tokens=False)
            steps_tokens.append(step_toks)

        all_step_tokens = [tok for step in steps_tokens for tok in step]
        input_ids = question_tokens + all_step_tokens
        input_ids_tensor = torch.tensor([input_ids], device=device)

        # Step-end positions: last token of each reasoning step
        step_end_positions = []
        pos = len(question_tokens)
        for step_toks in steps_tokens:
            pos += len(step_toks)
            step_end_positions.append(pos - 1)

        # Forward pass with hidden states
        outputs = model(input_ids_tensor, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        n_layers = len(hidden_states) - 1

        # Apply logit lens at each step-end position × layer
        entropy_matrix = []
        top_tokens_matrix = []
        for pos in step_end_positions:
            pos_entropies = []
            pos_top_tokens = []
            for layer_idx in range(n_layers):
                h = hidden_states[layer_idx + 1][0, pos, :]
                ent, top_toks = apply_logit_lens(h, ln_f, lm_head, tokenizer)
                pos_entropies.append(ent)
                pos_top_tokens.append(top_toks)
            entropy_matrix.append(pos_entropies)
            top_tokens_matrix.append(pos_top_tokens)

        results.append(
            {
                "idx": idx,
                "question": example["question"][:200],
                "answer": example["answer"],
                "n_steps": len(example["steps"]),
                "n_positions": len(step_end_positions),
                "position_labels": [
                    f"step_{i+1}" for i in range(len(steps_tokens))
                ],
                "entropy": entropy_matrix,
                "top_tokens": top_tokens_matrix,
            }
        )

        if idx == 0:
            print(f"\n--- Sanity check (example 0) ---")
            print(f"Sequence length: {len(input_ids)}")
            print(f"Step-end positions: {step_end_positions}")
            print(f"Steps: {example['steps']}")
            print(f"Num layers: {n_layers}")
            print(f"Entropy at step_1_end, layer 0: {entropy_matrix[0][0]:.4f}")
            print(f"Entropy at step_1_end, last layer: {entropy_matrix[0][-1]:.4f}")

    return results, n_layers


# ---- Main ----


def main():
    parser = argparse.ArgumentParser(
        description="Logit lens analysis for Coconut/CoT models"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["coconut", "cot"],
        help="Model type to analyze",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to dataset JSON (prosqa_valid.json or gsm_valid.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis/results",
        help="Directory to save output JSON",
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=50,
        help="Number of examples to analyze",
    )
    parser.add_argument(
        "--n_latent",
        type=int,
        default=6,
        help="Number of latent tokens (c_thought * max_latent_stage). Coconut only.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="openai-community/gpt2",
        help="Base HuggingFace model ID",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Custom output filename (without extension). "
        "Default: logit_lens_{model_type}",
    )
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    with open(args.data_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples from {args.data_path}")

    # Setup tokenizer
    tokenizer = setup_tokenizer(args.model_id)

    # Load model and run analysis
    if args.model_type == "coconut":
        print("Loading Coconut model...")
        model = load_coconut_model(
            args.checkpoint, args.model_id, tokenizer, args.device
        )
        results, n_layers = run_coconut_logit_lens(
            model, tokenizer, data, args.n_examples, args.n_latent, args.device
        )
    else:
        print("Loading CoT model...")
        model = load_cot_model(args.checkpoint, args.model_id, args.device)
        results, n_layers = run_cot_logit_lens(
            model, tokenizer, data, args.n_examples, args.device
        )

    # Save results
    output = {
        "model_type": args.model_type,
        "checkpoint": args.checkpoint,
        "data_path": args.data_path,
        "n_layers": n_layers,
        "n_examples": len(results),
        "examples": results,
    }
    name = args.output_name or f"logit_lens_{args.model_type}"
    output_path = os.path.join(args.output_dir, f"{name}.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
