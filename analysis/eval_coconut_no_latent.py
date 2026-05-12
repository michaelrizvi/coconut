#!/usr/bin/env python3
"""Evaluate a trained Coconut model WITHOUT latent passes.

Loads the Coconut checkpoint, feeds only the question (no latent tokens),
and generates answers autoregressively. This tests whether the model can
solve ProsQA without iterative latent reasoning.

Usage:
    python analysis/eval_coconut_no_latent.py \
        --checkpoint /path/to/coconut/checkpoint \
        --data_path data/prosqa_test.json \
        --n_examples 500
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
from analysis_utils import setup_tokenizer


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="openai-community/gpt2")
    parser.add_argument("--n_examples", type=int, default=None,
                        help="Max examples to evaluate (None = all)")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--output_path", type=str, default=None,
                        help="Save per-example results as JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = setup_tokenizer(args.model_id)
    model = load_coconut_model(args.checkpoint, args.model_id, tokenizer, device)

    with open(args.data_path) as f:
        data = json.load(f)
    if args.n_examples is not None:
        data = data[:args.n_examples]

    correct = 0
    total = 0
    results = []

    for sample in tqdm(data, desc="Evaluating (no latent)"):
        question = sample["question"]
        expected = sample["answer"].replace(",", "").strip()

        # Tokenize question only — NO latent tokens
        input_ids = tokenizer.encode(question + "\n", add_special_tokens=True,
                                     return_tensors="pt").to(device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids, attention_mask,
                max_new_tokens=args.max_new_tokens,
            )

        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Extract answer after "###"
        answer = text.split("#")[-1].replace(",", "").strip()

        is_correct = (answer == expected)
        correct += int(is_correct)
        total += 1

        results.append({
            "idx": sample.get("idx", total - 1),
            "expected": expected,
            "predicted": answer,
            "correct": is_correct,
            "full_output": text,
        })

        if total <= 5 or (not is_correct and total <= 20):
            status = "CORRECT" if is_correct else "WRONG"
            print(f"  [{status}] expected='{expected}' got='{answer}'")

    accuracy = correct / total if total > 0 else 0
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.4f}")

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump({"accuracy": accuracy, "correct": correct, "total": total,
                        "results": results}, f, indent=2)
        print(f"Saved results to {args.output_path}")


if __name__ == "__main__":
    main()
