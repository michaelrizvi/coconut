#!/usr/bin/env python3
"""Analyze latent thought representations in a trained Coconut model on ProsQA.

At each latent token position, projects the hidden state through the LM head
and computes P(entity) for all graph entities via autoregressive rollout.

Two modes:
  - per_pass (default): Probes after each individual forward pass in Coconut's
    multi-pass loop. This gives the entity distribution at each *intermediate*
    stage of latent reasoning, making it a fair comparison with CoT (both have
    done k steps of reasoning when probed at step k).
  - final_state: Probes all positions using the final hidden states after all
    passes have completed (the original approach).

Outputs a JSON file with entity probability matrices for visualization.
"""

import argparse
import json
import sys
import os

import torch
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, SCRIPT_DIR)

from coconut import Coconut
from transformers import AutoModelForCausalLM

from analysis_utils import (
    setup_tokenizer, tokenize_entities,
    compute_entity_probs_at_position, compute_entity_cosine_sims,
)


def load_coconut_model(checkpoint_path, model_id, tokenizer, device):
    """Load a Coconut model from checkpoint."""
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.resize_token_embeddings(len(tokenizer))

    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    # Initialize special token embeddings (same as training)
    embeddings = model.get_input_embeddings()
    target_id = tokenizer.convert_tokens_to_ids("<<")
    for token_id in [latent_id, start_id, end_id]:
        embeddings.weight.data[token_id] = embeddings.weight.data[target_id].clone()

    coconut_model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    load_result = coconut_model.load_state_dict(state_dict, strict=False)
    if load_result.unexpected_keys:
        print(f"Warning: unexpected keys: {load_result.unexpected_keys}")

    coconut_model.to(device)
    coconut_model.eval()
    return coconut_model


@torch.no_grad()
def analyze_coconut_per_pass(model, tokenizer, data, n_examples, n_latent, device):
    """Probe entity distributions after each individual latent pass.

    Mirrors Coconut's multi-pass forward loop, intercepting the KV cache and
    logits after each pass to probe entity distributions at that intermediate
    stage.  This is the fair comparison with CoT: Coconut thought_k is probed
    after k forward passes, comparable to CoT step_k after processing step k.

    Probe positions (same labels as final_state mode):
      - before_thinking: after pass 0 (question processed, no latent thinking yet)
      - thought_1..thought_n: after pass 1..n (k steps of latent reasoning)
      - after_thinking: after final pass (all thinking done, at end_latent position)
    """
    base_model = model.base_causallm
    embedding = base_model.transformer.get_input_embeddings()
    embed_matrix = embedding.weight.detach()

    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    results = []
    for idx in tqdm(range(min(n_examples, len(data))), desc="Coconut per-pass"):
        example = data[idx]

        question_tokens = tokenizer.encode(
            example["question"] + "\n", add_special_tokens=True
        )
        input_ids = question_tokens + [start_id] + [latent_id] * n_latent + [end_id]
        input_ids_tensor = torch.tensor([input_ids], device=device)

        inputs_embeds = embedding(input_ids_tensor)
        attention_mask = torch.ones_like(input_ids_tensor)
        position_ids = torch.arange(len(input_ids), device=device).unsqueeze(0)

        # Positions in the sequence
        start_latent_pos = len(question_tokens)       # <start_latent>
        first_latent_pos = start_latent_pos + 1       # first <latent>
        latent_positions = [first_latent_pos + i for i in range(n_latent)]
        end_latent_pos = first_latent_pos + n_latent  # <end_latent>

        # Entity setup
        entities = example["idx_to_symbol"]
        entity_token_ids = tokenize_entities(entities, tokenizer)
        entity_probs = {ent: [] for ent in entities}
        entity_cosine_sims = {ent: [] for ent in entities}

        # ---- Mirror Coconut's multi-pass forward ----
        # First compute range: everything before the first latent token
        next_compute_range = (0, first_latent_pos)
        kv_cache = None

        for pass_idx in range(n_latent):
            if kv_cache is None:
                outputs = base_model(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    output_hidden_states=True,
                    use_cache=True,
                )
                hidden_states_offset = 0
            else:
                past_kv = [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]
                outputs = base_model(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[:, : next_compute_range[1]],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    past_key_values=past_kv,
                    output_hidden_states=True,
                    use_cache=True,
                )
                hidden_states_offset = next_compute_range[0]

            hidden_states = outputs.hidden_states[-1]  # last layer
            kv_cache = outputs.past_key_values

            # --- Probe at the last position of this compute range ---
            probe_abs_pos = next_compute_range[1] - 1
            probe_rel_pos = probe_abs_pos - hidden_states_offset
            logits_at_pos = outputs.logits[0, probe_rel_pos, :]

            probs = compute_entity_probs_at_position(
                base_model, kv_cache, probe_abs_pos,
                logits_at_pos, entity_token_ids, device,
            )
            h_at_probe = hidden_states[0, probe_rel_pos, :]
            cosine_sims = compute_entity_cosine_sims(
                h_at_probe, entity_token_ids, embed_matrix,
            )
            for ent in entities:
                entity_probs[ent].append(probs[ent])
                entity_cosine_sims[ent].append(cosine_sims[ent])

            # --- Feed continuous thought into next latent position ---
            # (mirrors coconut.py lines 127-158)
            token_idx = latent_positions[pass_idx]
            thought = hidden_states[0, token_idx - 1 - hidden_states_offset, :]

            tensor_list = [
                inputs_embeds[0, p, :] for p in range(inputs_embeds.shape[1])
            ]
            tensor_list[token_idx] = thought
            inputs_embeds = torch.stack(tensor_list).unsqueeze(0)

            # Update compute range
            next_compute_range = (
                next_compute_range[1],
                (
                    len(input_ids)
                    if pass_idx + 1 >= n_latent
                    else next_compute_range[1] + 1
                ),
            )

        # --- Final pass: process remaining positions (last latent + end_latent) ---
        past_kv = [
            (
                k[:, :, : next_compute_range[0], :],
                v[:, :, : next_compute_range[0], :],
            )
            for k, v in kv_cache
        ]
        outputs = base_model(
            inputs_embeds=inputs_embeds[
                :, next_compute_range[0] : next_compute_range[1], :
            ],
            attention_mask=attention_mask[:, : next_compute_range[1]],
            position_ids=position_ids[
                :, next_compute_range[0] : next_compute_range[1]
            ],
            past_key_values=past_kv,
            output_hidden_states=True,
            use_cache=True,
        )
        kv_cache = outputs.past_key_values
        hidden_states_offset = next_compute_range[0]

        # Probe at last latent position (thought_n) and end_latent (after_thinking)
        for probe_pos in [latent_positions[-1], end_latent_pos]:
            if probe_pos < next_compute_range[0] or probe_pos >= len(input_ids):
                continue
            rel_pos = probe_pos - hidden_states_offset
            logits_at_pos = outputs.logits[0, rel_pos, :]
            probs = compute_entity_probs_at_position(
                base_model, kv_cache, probe_pos,
                logits_at_pos, entity_token_ids, device,
            )
            h_final = outputs.hidden_states[-1][0, rel_pos, :]
            cosine_sims = compute_entity_cosine_sims(
                h_final, entity_token_ids, embed_matrix,
            )
            for ent in entities:
                entity_probs[ent].append(probs[ent])
                entity_cosine_sims[ent].append(cosine_sims[ent])

        # Position labels (same format as final_state mode)
        probe_positions_list = (
            [start_latent_pos] + latent_positions + [end_latent_pos]
        )
        position_labels = (
            ["before_thinking"]
            + [f"thought_{i+1}" for i in range(n_latent)]
            + ["after_thinking"]
        )

        results.append(
            {
                "idx": idx,
                "question": example["question"],
                "answer": example["answer"],
                "steps": example["steps"],
                "entities": entities,
                "edges": example["edges"],
                "root": example["root"],
                "target": example["target"],
                "neg_target": example["neg_target"],
                "positions": probe_positions_list,
                "position_labels": position_labels,
                "entity_probs": entity_probs,
                "entity_cosine_sims": entity_cosine_sims,
            }
        )

        if idx == 0:
            print(f"\n--- Example 0 sanity check (per-pass) ---")
            print(f"Question (truncated): {example['question'][:100]}...")
            print(f"Answer: {example['answer']}")
            target_ent = entities[example["target"]]
            print(f"Target: {target_ent}")
            print(f"P({target_ent}) per pass: {entity_probs[target_ent]}")
            n_probs = len(entity_probs[target_ent])
            print(f"Total probe positions: {n_probs} "
                  f"(expected {n_latent + 2})")

    return results


@torch.no_grad()
def analyze_coconut_final_state(model, tokenizer, data, n_examples, n_latent, device):
    """Probe all positions using final hidden states (original approach).

    Runs the full Coconut multi-pass forward, then re-runs the base model on
    the final inputs_embeds to get a complete KV cache. All positions are probed
    with the same final-state representations.

    NOTE: This means thought_1 is probed AFTER all n_latent passes have completed,
    not after just 1 pass. For a fair comparison with CoT, use per_pass mode.
    """
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    results = []
    for idx in tqdm(range(min(n_examples, len(data))), desc="Coconut final-state"):
        example = data[idx]

        question_tokens = tokenizer.encode(
            example["question"] + "\n", add_special_tokens=True
        )
        input_ids = (
            question_tokens + [start_id] + [latent_id] * n_latent + [end_id]
        )
        input_ids_tensor = torch.tensor([input_ids], device=device)

        start_latent_pos = len(question_tokens)
        latent_positions = [start_latent_pos + 1 + i for i in range(n_latent)]
        end_latent_pos = start_latent_pos + 1 + n_latent

        probe_positions = [start_latent_pos] + latent_positions + [end_latent_pos]
        position_labels = (
            ["before_thinking"]
            + [f"thought_{i+1}" for i in range(n_latent)]
            + ["after_thinking"]
        )

        attention_mask = torch.ones_like(input_ids_tensor)
        labels = input_ids_tensor.clone()
        position_ids = torch.arange(len(input_ids), device=device).unsqueeze(0)

        outputs = model.forward(input_ids_tensor, attention_mask, labels, position_ids)
        final_embeds = outputs.inputs_embeds
        coconut_logits = outputs.logits

        base_outputs = model.base_causallm(
            inputs_embeds=final_embeds, use_cache=True
        )
        full_kv = base_outputs.past_key_values

        entities = example["idx_to_symbol"]
        entity_token_ids = tokenize_entities(entities, tokenizer)

        entity_probs = {ent: [] for ent in entities}
        for pos in probe_positions:
            logits_at_pos = coconut_logits[0, pos, :]
            probs = compute_entity_probs_at_position(
                model.base_causallm, full_kv, pos,
                logits_at_pos, entity_token_ids, device,
            )
            for ent in entities:
                entity_probs[ent].append(probs[ent])

        results.append(
            {
                "idx": idx,
                "question": example["question"],
                "answer": example["answer"],
                "steps": example["steps"],
                "entities": entities,
                "edges": example["edges"],
                "root": example["root"],
                "target": example["target"],
                "neg_target": example["neg_target"],
                "positions": probe_positions,
                "position_labels": position_labels,
                "entity_probs": entity_probs,
            }
        )

        if idx == 0:
            print(f"\n--- Example 0 sanity check (final-state) ---")
            print(f"Question (truncated): {example['question'][:100]}...")
            print(f"Answer: {example['answer']}")
            target_ent = entities[example["target"]]
            print(f"P({target_ent}) across thoughts: {entity_probs[target_ent]}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze latent thoughts in Coconut model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to ProsQA Coconut checkpoint",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/prosqa_valid.json",
        help="Path to ProsQA validation data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis/results",
        help="Directory to save output JSON",
    )
    parser.add_argument(
        "--n_examples", type=int, default=50, help="Number of examples to analyze"
    )
    parser.add_argument(
        "--n_latent",
        type=int,
        default=6,
        help="Number of latent tokens (c_thought * max_latent_stage)",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="openai-community/gpt2",
        help="Base model ID",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["per_pass", "final_state"],
        default="per_pass",
        help="Probing mode: per_pass (probe at each intermediate pass) or "
        "final_state (probe all positions with final hidden states)",
    )
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    with open(args.data_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples from {args.data_path}")

    # Setup tokenizer and model
    tokenizer = setup_tokenizer(args.model_id)
    print("Loading Coconut model...")
    model = load_coconut_model(args.checkpoint, args.model_id, tokenizer, args.device)

    # Run analysis
    print(f"Running in {args.mode} mode")
    if args.mode == "per_pass":
        results = analyze_coconut_per_pass(
            model, tokenizer, data, args.n_examples, args.n_latent, args.device
        )
    else:
        results = analyze_coconut_final_state(
            model, tokenizer, data, args.n_examples, args.n_latent, args.device
        )

    # Save results
    output = {
        "model_type": "coconut",
        "mode": args.mode,
        "checkpoint": args.checkpoint,
        "n_latent": args.n_latent,
        "n_examples": len(results),
        "examples": results,
    }
    output_path = os.path.join(args.output_dir, "coconut_entity_probs.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
