#!/usr/bin/env python3
"""Plot step-aware entity belief for examples the model gets wrong without latents.

Cross-references no-latent eval results with entity probing data to isolate the
~3% of examples where removing latent tokens causes failure, then generates the
stepwise_all_entity_norm plot for only those examples.

Usage:
    python analysis/plot_hard_examples.py [--filter_n_steps N]
"""

import argparse
import json

from visualize import plot_stepwise_all_entity_norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_latent_json",
                        default="analysis/results/test/coconut_no_latent_eval.json")
    parser.add_argument("--coconut_json",
                        default="analysis/results/test/coconut_entity_probs.json")
    parser.add_argument("--cot_json",
                        default="analysis/results/test/cot_entity_probs.json")
    parser.add_argument("--output_path",
                        default="analysis/figures/stepwise_hard_examples.pdf")
    parser.add_argument("--filter_n_steps", type=int, default=None)
    args = parser.parse_args()

    # Load no-latent eval to find hard examples
    with open(args.no_latent_json) as f:
        no_latent = json.load(f)

    hard_idxs = {r["idx"] for r in no_latent["results"] if not r["correct"]}
    print(f"Hard examples (wrong without latents): {len(hard_idxs)}")
    print(f"  Indices: {sorted(hard_idxs)}")

    # Load entity probing data and filter
    with open(args.coconut_json) as f:
        coconut_data = json.load(f)
    with open(args.cot_json) as f:
        cot_data = json.load(f)

    coconut_data["examples"] = [
        ex for ex in coconut_data["examples"] if ex["idx"] in hard_idxs
    ]
    cot_data["examples"] = [
        ex for ex in cot_data["examples"] if ex["idx"] in hard_idxs
    ]
    print(f"Matched coconut examples: {len(coconut_data['examples'])}")
    print(f"Matched CoT examples: {len(cot_data['examples'])}")

    # Show step distribution
    step_counts = {}
    for ex in coconut_data["examples"]:
        n = len(ex["steps"])
        step_counts[n] = step_counts.get(n, 0) + 1
    print(f"Step distribution: {dict(sorted(step_counts.items()))}")

    suffix = f"_{args.filter_n_steps}step" if args.filter_n_steps else ""
    output_path = args.output_path.replace(".pdf", f"{suffix}.pdf")

    plot_stepwise_all_entity_norm(
        coconut_data, cot_data, output_path,
        filter_n_steps=args.filter_n_steps,
    )


if __name__ == "__main__":
    main()
