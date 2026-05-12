#!/usr/bin/env python3
"""Plot step-aware entity belief using cosine similarity (no LM head).

Produces the same stacked-area plot as stepwise_all_entity_norm, but using
cosine similarity between hidden states and wte embeddings instead of LM head
probabilities. This validates that the LM head is not a confound.

Usage:
    python analysis/plot_cosine_probe.py \
        --coconut_json analysis/results/test_cosine/coconut_entity_probs.json \
        --cot_json analysis/results/test/cot_entity_probs.json \
        --output_dir analysis/figures
"""

import argparse
import json
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({"font.size": 12})

# Reuse graph-traversal helpers from visualize.py
from visualize import (
    get_step_info,
    find_is_a_positions,
    plot_stepwise_all_entity_norm,
)


def softmax_normalize(entity_values, entities):
    """Softmax-normalize entity values (cosine sims can be negative)."""
    n_positions = len(entity_values[entities[0]])
    normalized = {ent: [] for ent in entities}
    for i in range(n_positions):
        vals = np.array([entity_values[ent][i] for ent in entities])
        # Softmax for numerical stability
        exp_vals = np.exp(vals - vals.max())
        total = exp_vals.sum()
        for j, ent in enumerate(entities):
            normalized[ent].append(float(exp_vals[j] / total) if total > 0 else 0.0)
    return normalized


def swap_cosine_for_probs(data):
    """Replace entity_probs with softmax-normalized entity_cosine_sims.

    Returns a modified copy of the data dict.
    """
    data = dict(data)
    data["examples"] = []
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coconut_json",
                        default="analysis/results/test_cosine/coconut_entity_probs.json")
    parser.add_argument("--cot_json",
                        default="analysis/results/test/cot_entity_probs.json")
    parser.add_argument("--output_dir", default="analysis/figures")
    parser.add_argument("--filter_n_steps", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.coconut_json) as f:
        coconut_data = json.load(f)
    with open(args.cot_json) as f:
        cot_data = json.load(f)

    # Verify cosine sim data exists
    ex0 = coconut_data["examples"][0]
    if "entity_cosine_sims" not in ex0:
        raise ValueError("No entity_cosine_sims in coconut data. "
                         "Re-run analyze_coconut.py with cosine sim support.")

    # Replace entity_probs with softmax-normalized cosine sims
    for ex in coconut_data["examples"]:
        entities = ex["entities"]
        ex["entity_probs"] = softmax_normalize(ex["entity_cosine_sims"], entities)

    # Generate the plot (reuse existing function)
    fns = f"_{args.filter_n_steps}step" if args.filter_n_steps else ""
    output_path = os.path.join(
        args.output_dir, f"stepwise_cosine_entity_norm{fns}.pdf"
    )

    plot_stepwise_all_entity_norm(
        coconut_data, cot_data, output_path,
        filter_n_steps=args.filter_n_steps,
    )

    # Also generate without step filter
    if args.filter_n_steps is not None:
        output_path_all = os.path.join(
            args.output_dir, "stepwise_cosine_entity_norm.pdf"
        )
        plot_stepwise_all_entity_norm(
            coconut_data, cot_data, output_path_all,
            filter_n_steps=None,
        )


if __name__ == "__main__":
    main()
