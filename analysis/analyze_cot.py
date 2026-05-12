#!/usr/bin/env python3
"""Analyze output distributions in a trained CoT baseline model on ProsQA.

At each token position in the teacher-forced reasoning trace, projects the
hidden state through the LM head and computes P(entity) for all graph
entities via autoregressive rollout.

This allows direct comparison with the Coconut latent thought analysis:
does the CoT model also spread probability mass across candidate entities
at intermediate reasoning steps, or does discrete decoding collapse this?

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

from transformers import AutoModelForCausalLM

from analysis_utils import setup_tokenizer, tokenize_entities, compute_entity_probs_at_position


def load_cot_model(checkpoint_path, model_id, device):
    """Load a CoT baseline model from checkpoint.

    CoT models do NOT resize token embeddings (the special latent tokens
    are added to the tokenizer but not used during CoT training).
    """
    model = AutoModelForCausalLM.from_pretrained(model_id)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.unexpected_keys:
        print(f"Warning: unexpected keys: {load_result.unexpected_keys}")
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def analyze_cot(model, tokenizer, data, n_examples, device):
    """Run analysis on CoT model with teacher-forced reasoning traces.

    For each example:
      1. Construct input: question + CoT step tokens (teacher forced).
      2. Single forward pass with use_cache=True to get logits + KV cache.
      3. At every token position in the CoT trace, compute P(entity)
         via autoregressive rollout.
    """
    results = []
    for idx in tqdm(range(min(n_examples, len(data))), desc="CoT analysis"):
        example = data[idx]

        # Construct teacher-forced input: question + CoT steps
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

        # Positions of interest: every token in the CoT trace
        cot_start = len(question_tokens)
        cot_end = len(input_ids)
        positions_of_interest = list(range(cot_start, cot_end))

        # Position labels: the decoded token at each position
        position_labels = [
            tokenizer.decode([input_ids[pos]]) for pos in positions_of_interest
        ]

        # Also record which step each position belongs to
        step_boundaries = []
        pos = cot_start
        for step_idx, step_toks in enumerate(steps_tokens):
            step_boundaries.append(
                {"step_idx": step_idx, "start": pos, "end": pos + len(step_toks)}
            )
            pos += len(step_toks)

        # Forward pass
        outputs = model(input_ids_tensor, use_cache=True)
        logits = outputs.logits
        full_kv = outputs.past_key_values

        # Tokenize entities
        entities = example["idx_to_symbol"]
        entity_token_ids = tokenize_entities(entities, tokenizer)

        # Compute entity probs at each CoT position
        entity_probs = {ent: [] for ent in entities}
        for pos in positions_of_interest:
            logits_at_pos = logits[0, pos, :]
            probs = compute_entity_probs_at_position(
                model, full_kv, pos, logits_at_pos, entity_token_ids, device
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
                "positions": positions_of_interest,
                "position_labels": position_labels,
                "step_boundaries": step_boundaries,
                "entity_probs": entity_probs,
            }
        )

        # Print first example for sanity check
        if idx == 0:
            print(f"\n--- Example 0 sanity check ---")
            print(f"Question (truncated): {example['question'][:100]}...")
            print(f"Answer: {example['answer']}")
            print(f"Steps: {example['steps']}")
            print(f"CoT trace length: {len(positions_of_interest)} tokens")
            target_ent = entities[example["target"]]
            print(f"P({target_ent}) at first CoT token: {entity_probs[target_ent][0]:.6f}")
            print(f"P({target_ent}) at last CoT token: {entity_probs[target_ent][-1]:.6f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze output distributions in CoT baseline model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to ProsQA CoT checkpoint",
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
        "--model_id",
        type=str,
        default="openai-community/gpt2",
        help="Base model ID",
    )
    parser.add_argument("--device", type=str, default=None)
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
    print("Loading CoT model...")
    model = load_cot_model(args.checkpoint, args.model_id, args.device)

    # Run analysis
    results = analyze_cot(model, tokenizer, data, args.n_examples, args.device)

    # Save results
    output = {
        "model_type": "cot",
        "checkpoint": args.checkpoint,
        "n_examples": len(results),
        "examples": results,
    }
    output_path = os.path.join(args.output_dir, "cot_entity_probs.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
