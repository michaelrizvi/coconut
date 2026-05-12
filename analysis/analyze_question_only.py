#!/usr/bin/env python3
"""Question-only baseline: probe entity beliefs immediately after the question.

For both Coconut and CoT models, feeds ONLY the question tokens (no latent
tokens, no CoT steps) and probes the output distribution at the last token.
This reveals each model's "prior belief" over entities before any reasoning.

Usage:
    python analyze_question_only.py \
        --coconut_checkpoint <path> \
        --cot_checkpoint <path> \
        --data_path data/prosqa_valid.json \
        --n_examples 50
"""

import argparse
import json
import os
import sys

import torch
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, SCRIPT_DIR)

from coconut import Coconut
from transformers import AutoModelForCausalLM

from analysis_utils import setup_tokenizer, tokenize_entities, compute_entity_probs_at_position


def load_coconut_model(checkpoint_path, model_id, tokenizer, device):
    """Load Coconut model, return the base GPT-2 inside it."""
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
    # Return the base GPT-2 — we only need a plain forward pass
    return coconut_model.base_causallm


def load_cot_model(checkpoint_path, model_id, device):
    """Load CoT model."""
    model = AutoModelForCausalLM.from_pretrained(model_id)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def probe_question_only(model, tokenizer, data, n_examples, device, label="model"):
    """Feed question-only, probe at last token.

    For each example:
      1. Tokenize question (same as training: question + "\\n")
      2. Single forward pass with use_cache=True
      3. Take logits at last position, compute P(entity) via rollout
    """
    results = []
    for idx in tqdm(range(min(n_examples, len(data))), desc=f"{label} question-only"):
        example = data[idx]

        question_tokens = tokenizer.encode(
            example["question"] + "\n", add_special_tokens=True
        )
        input_ids = torch.tensor([question_tokens], device=device)

        outputs = model(input_ids, use_cache=True)
        full_kv = outputs.past_key_values

        probe_pos = len(question_tokens) - 1  # last question token
        logits_at_pos = outputs.logits[0, probe_pos, :]

        entities = example["idx_to_symbol"]
        entity_token_ids = tokenize_entities(entities, tokenizer)

        probs = compute_entity_probs_at_position(
            model, full_kv, probe_pos, logits_at_pos, entity_token_ids, device,
        )

        entity_probs = {ent: [probs[ent]] for ent in entities}

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
            "positions": [probe_pos],
            "position_labels": ["question_end"],
            "entity_probs": entity_probs,
        })

        if idx == 0:
            print(f"\n--- {label} example 0 sanity check ---")
            print(f"Question (truncated): {example['question'][:100]}...")
            print(f"Answer: {example['answer']}")
            target_ent = entities[example["target"]]
            print(f"P({target_ent}) at question end: {probs[target_ent]:.6e}")
            # Show top 3 entities by prob
            sorted_ents = sorted(entities, key=lambda e: -probs[e])
            for e in sorted_ents[:3]:
                print(f"  {e}: {probs[e]:.6e}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Question-only entity belief probing"
    )
    parser.add_argument(
        "--coconut_checkpoint", type=str, required=True,
        help="Path to Coconut checkpoint",
    )
    parser.add_argument(
        "--cot_checkpoint", type=str, required=True,
        help="Path to CoT checkpoint",
    )
    parser.add_argument(
        "--data_path", type=str, default="data/prosqa_valid.json",
    )
    parser.add_argument(
        "--output_dir", type=str, default="analysis/results",
    )
    parser.add_argument(
        "--n_examples", type=int, default=50,
    )
    parser.add_argument(
        "--model_id", type=str, default="openai-community/gpt2",
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.data_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples from {args.data_path}")

    tokenizer = setup_tokenizer(args.model_id)

    # --- Coconut ---
    print("Loading Coconut model...")
    coconut_model = load_coconut_model(
        args.coconut_checkpoint, args.model_id, tokenizer, args.device
    )
    coconut_results = probe_question_only(
        coconut_model, tokenizer, data, args.n_examples, args.device, "Coconut"
    )
    del coconut_model
    torch.cuda.empty_cache()

    # --- CoT ---
    print("Loading CoT model...")
    cot_model = load_cot_model(args.cot_checkpoint, args.model_id, args.device)
    cot_results = probe_question_only(
        cot_model, tokenizer, data, args.n_examples, args.device, "CoT"
    )
    del cot_model
    torch.cuda.empty_cache()

    # Save
    for model_type, results in [("coconut", coconut_results), ("cot", cot_results)]:
        output = {
            "model_type": model_type,
            "mode": "question_only",
            "n_examples": len(results),
            "examples": results,
        }
        path = os.path.join(args.output_dir, f"question_only_{model_type}.json")
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved {model_type} results to {path}")


if __name__ == "__main__":
    main()
