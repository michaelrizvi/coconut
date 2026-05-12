#!/usr/bin/env python3
"""Layer-wise entity belief analysis for Coconut and CoT models (logit lens).

For each thinking position (latent tokens for Coconut, step-end tokens for CoT),
extracts hidden states at every transformer layer, projects through the final
layer norm + LM head (logit lens), and computes P(entity) for all graph entities.

Entity probability = P(first BPE token of entity name) from the logit-lens
projected distribution. This is the standard logit lens approximation; for
multi-token entities the remaining tokens are not conditioned on.

Coconut uses the final_state approach: run all n_latent passes to bake in
continuous thoughts, then one fresh causal forward with output_hidden_states=True.
This is equivalent to the per-pass computation (same weights, same embeddings,
same causal mask) but cheaper.

Output JSON schema (per example):
  position_labels:      list[str]  — e.g. ["thought_1", ..., "thought_6"]
  entity_layer_probs:   dict[entity -> list[list[float]]]
                        shape: (n_positions, n_layers)
                        raw (unnormalized) probabilities; normalize at plot time.

Usage:
    # Coconut on ProsQA
    python analyze_layerwise_belief.py --model_type coconut \\
        --checkpoint /path/to/checkpoint --data_path data/prosqa_valid.json

    # CoT on ProsQA
    python analyze_layerwise_belief.py --model_type cot \\
        --checkpoint /path/to/checkpoint --data_path data/prosqa_valid.json
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

from analysis_utils import setup_tokenizer, tokenize_entities


# ---- Model loading (mirrors existing analysis scripts) ----


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


def load_cot_model(checkpoint_path, model_id, device):
    model = AutoModelForCausalLM.from_pretrained(model_id)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


# ---- Logit lens helpers ----


def get_gpt2_components(model):
    """Return (ln_f, lm_head) from a GPT-2 model or Coconut wrapper."""
    base = model.base_causallm if isinstance(model, Coconut) else model
    return base.transformer.ln_f, base.lm_head


def logit_lens_entity_probs(hidden_state, ln_f, lm_head, entity_token_ids):
    """Project hidden_state to vocab via logit lens, return P(first token) per entity.

    Args:
        hidden_state: (hidden_dim,) tensor at a single position and layer
        ln_f:         final layer norm module
        lm_head:      unembedding matrix module
        entity_token_ids: dict[entity_name -> list[int]]

    Returns:
        dict[entity_name -> float]  — raw (unnormalized) probabilities
    """
    logits = lm_head(ln_f(hidden_state.unsqueeze(0)))[0]  # (vocab_size,)
    probs = F.softmax(logits.float(), dim=-1)
    return {entity: probs[token_ids[0]].item()
            for entity, token_ids in entity_token_ids.items()}


# ---- Coconut layer-wise analysis ----


@torch.no_grad()
def analyze_coconut_layerwise(model, tokenizer, data, n_examples, n_latent, device):
    """Logit lens entity belief at every (thought_position, layer) pair.

    For each example:
      1. Build input: question + <start_latent> + n_latent * <latent> + <end_latent>
      2. Run Coconut.forward() to bake continuous thoughts into final_embeds
      3. Re-run base GPT-2 on final_embeds with output_hidden_states=True
      4. At each thought position t and each layer l:
             h = hidden_states[l+1][0, thought_t, :]
             entity_probs = logit_lens(h)
    """
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    ln_f, lm_head = get_gpt2_components(model)

    results = []
    for idx in tqdm(range(min(n_examples, len(data))), desc="Coconut layerwise"):
        example = data[idx]

        question_tokens = tokenizer.encode(
            example["question"] + "\n", add_special_tokens=True
        )
        input_ids = question_tokens + [start_id] + [latent_id] * n_latent + [end_id]
        input_ids_tensor = torch.tensor([input_ids], device=device)

        start_latent_pos = len(question_tokens)
        thought_positions = [start_latent_pos + 1 + i for i in range(n_latent)]

        attention_mask = torch.ones_like(input_ids_tensor)
        labels = input_ids_tensor.clone()
        position_ids = torch.arange(len(input_ids), device=device).unsqueeze(0)

        # Bake continuous thoughts into final_embeds via Coconut multi-pass forward
        outputs = model.forward(input_ids_tensor, attention_mask, labels, position_ids)
        final_embeds = outputs.inputs_embeds

        # One fresh causal forward to get all-layer hidden states
        base_outputs = model.base_causallm(
            inputs_embeds=final_embeds, output_hidden_states=True
        )
        # hidden_states: tuple of (n_layers+1) tensors, each (1, seq_len, hidden_dim)
        # index 0 = embedding layer output; indices 1..n_layers = transformer block outputs
        hidden_states = base_outputs.hidden_states
        n_layers = len(hidden_states) - 1

        entities = example["idx_to_symbol"]
        entity_token_ids = tokenize_entities(entities, tokenizer)

        # entity_layer_probs[entity][position_idx][layer_idx] = float
        entity_layer_probs = {ent: [] for ent in entities}
        for pos in thought_positions:
            pos_layer_probs = {ent: [] for ent in entities}
            for layer_idx in range(n_layers):
                h = hidden_states[layer_idx + 1][0, pos, :]
                probs = logit_lens_entity_probs(h, ln_f, lm_head, entity_token_ids)
                for ent in entities:
                    pos_layer_probs[ent].append(probs[ent])
            for ent in entities:
                entity_layer_probs[ent].append(pos_layer_probs[ent])

        results.append({
            "idx": idx,
            "question": example["question"],
            "answer": example["answer"],
            "steps": example["steps"],
            "entities": entities,
            "edges": example["edges"],
            "root": example["root"],
            "target": example["target"],
            "neg_target": example["neg_target"],
            "position_labels": [f"thought_{i+1}" for i in range(n_latent)],
            "entity_layer_probs": entity_layer_probs,
        })

        if idx == 0:
            target_ent = entities[example["target"]]
            print(f"\n--- Example 0 sanity check (Coconut) ---")
            print(f"Target entity: {target_ent}")
            print(f"n_layers: {n_layers}, n_thought_positions: {n_latent}")
            print(f"P({target_ent}) at thought_1 across layers: "
                  f"{[f'{p:.3f}' for p in entity_layer_probs[target_ent][0]]}")

    return results, n_layers


# ---- CoT layer-wise analysis ----


@torch.no_grad()
def analyze_cot_layerwise(model, tokenizer, data, n_examples, device):
    """Logit lens entity belief at every (step_end_position, layer) pair for CoT.

    For each example:
      1. Build teacher-forced input: question + all CoT step tokens
      2. Single forward pass with output_hidden_states=True
      3. At the last token of each reasoning step and each layer l:
             h = hidden_states[l+1][0, step_end_pos, :]
             entity_probs = logit_lens(h)
    """
    ln_f = model.transformer.ln_f
    lm_head = model.lm_head

    results = []
    for idx in tqdm(range(min(n_examples, len(data))), desc="CoT layerwise"):
        example = data[idx]

        question_tokens = tokenizer.encode(
            example["question"] + "\n", add_special_tokens=True
        )
        steps_tokens = [
            tokenizer.encode(step + "\n", add_special_tokens=False)
            for step in example["steps"]
        ]
        all_step_tokens = [tok for step in steps_tokens for tok in step]
        input_ids = question_tokens + all_step_tokens
        input_ids_tensor = torch.tensor([input_ids], device=device)

        # Last token of each reasoning step
        step_end_positions = []
        pos = len(question_tokens)
        for step_toks in steps_tokens:
            pos += len(step_toks)
            step_end_positions.append(pos - 1)

        outputs = model(input_ids_tensor, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        n_layers = len(hidden_states) - 1

        entities = example["idx_to_symbol"]
        entity_token_ids = tokenize_entities(entities, tokenizer)

        entity_layer_probs = {ent: [] for ent in entities}
        for pos in step_end_positions:
            pos_layer_probs = {ent: [] for ent in entities}
            for layer_idx in range(n_layers):
                h = hidden_states[layer_idx + 1][0, pos, :]
                probs = logit_lens_entity_probs(h, ln_f, lm_head, entity_token_ids)
                for ent in entities:
                    pos_layer_probs[ent].append(probs[ent])
            for ent in entities:
                entity_layer_probs[ent].append(pos_layer_probs[ent])

        results.append({
            "idx": idx,
            "question": example["question"],
            "answer": example["answer"],
            "steps": example["steps"],
            "entities": entities,
            "edges": example["edges"],
            "root": example["root"],
            "target": example["target"],
            "neg_target": example["neg_target"],
            "position_labels": [f"step_{i+1}_end" for i in range(len(steps_tokens))],
            "entity_layer_probs": entity_layer_probs,
        })

        if idx == 0:
            target_ent = entities[example["target"]]
            print(f"\n--- Example 0 sanity check (CoT) ---")
            print(f"Target entity: {target_ent}")
            print(f"n_layers: {n_layers}, n_steps: {len(steps_tokens)}")
            print(f"P({target_ent}) at step_1_end across layers: "
                  f"{[f'{p:.3f}' for p in entity_layer_probs[target_ent][0]]}")

    return results, n_layers


# ---- Main ----


def main():
    parser = argparse.ArgumentParser(
        description="Layer-wise logit lens entity belief for Coconut / CoT"
    )
    parser.add_argument(
        "--model_type", type=str, required=True, choices=["coconut", "cot"]
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to dataset JSON (prosqa_valid.json or prosqa_test.json)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="analysis/results/prosqa",
        help="Directory to save output JSON"
    )
    parser.add_argument(
        "--n_examples", type=int, default=100,
        help="Number of examples to analyze"
    )
    parser.add_argument(
        "--n_latent", type=int, default=6,
        help="Number of latent tokens (Coconut only)"
    )
    parser.add_argument(
        "--n_steps_filter", type=int, default=None,
        help="If set, only analyze examples with exactly this many reasoning steps"
    )
    parser.add_argument(
        "--model_id", type=str, default="openai-community/gpt2"
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output_name", type=str, default=None,
        help="Custom output filename stem (default: layerwise_belief_{model_type})"
    )
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.data_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples from {args.data_path}")

    if args.n_steps_filter is not None:
        data = [ex for ex in data if len(ex["steps"]) == args.n_steps_filter]
        print(f"Filtered to {len(data)} examples with {args.n_steps_filter} steps")

    tokenizer = setup_tokenizer(args.model_id)

    if args.model_type == "coconut":
        print("Loading Coconut model...")
        model = load_coconut_model(
            args.checkpoint, args.model_id, tokenizer, args.device
        )
        results, n_layers = analyze_coconut_layerwise(
            model, tokenizer, data, args.n_examples, args.n_latent, args.device
        )
    else:
        print("Loading CoT model...")
        model = load_cot_model(args.checkpoint, args.model_id, args.device)
        results, n_layers = analyze_cot_layerwise(
            model, tokenizer, data, args.n_examples, args.device
        )

    output = {
        "model_type": args.model_type,
        "checkpoint": args.checkpoint,
        "data_path": args.data_path,
        "n_layers": n_layers,
        "n_latent": args.n_latent if args.model_type == "coconut" else None,
        "n_steps_filter": args.n_steps_filter,
        "n_examples": len(results),
        "examples": results,
    }
    name = args.output_name or f"layerwise_belief_{args.model_type}"
    output_path = os.path.join(args.output_dir, f"{name}.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
