#!/usr/bin/env python3
"""Plot logit lens results: entropy heatmaps and layer-wise comparison.

Reads JSON outputs from run_logit_lens.py and produces:
  1. Per-model entropy heatmap (thinking position x layer)
  2. Overlay line plot comparing average entropy per layer (Coconut vs CoT)

Usage:
    # Both heatmaps + comparison line plot
    python plot_logit_lens.py \
        --coconut_results analysis/results/logit_lens_coconut.json \
        --cot_results analysis/results/logit_lens_cot.json \
        --output_dir analysis/figures/logit_lens

    # Single model heatmap only
    python plot_logit_lens.py \
        --coconut_results analysis/results/logit_lens_coconut.json \
        --output_dir analysis/figures/logit_lens
"""

import argparse
import json
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
    }
)


def load_results(path):
    with open(path) as f:
        return json.load(f)


# ---- Heatmap ----


def plot_entropy_heatmap(results, title, output_path):
    """Heatmap of entropy at (thinking position x layer), averaged across examples."""
    max_pos = max(ex["n_positions"] for ex in results["examples"])
    n_layers = results["n_layers"]

    # Accumulate with counts (examples may have different n_positions for CoT)
    entropy_sum = np.zeros((max_pos, n_layers))
    counts = np.zeros((max_pos, n_layers))

    for ex in results["examples"]:
        for p_idx, row in enumerate(ex["entropy"]):
            for l_idx, val in enumerate(row):
                entropy_sum[p_idx, l_idx] += val
                counts[p_idx, l_idx] += 1

    entropy_avg = np.divide(entropy_sum, counts, where=counts > 0)
    entropy_avg[counts == 0] = np.nan

    fig, ax = plt.subplots(figsize=(max(8, n_layers * 0.7), max(4, max_pos * 0.6)))
    im = ax.imshow(entropy_avg, aspect="auto", cmap="viridis", interpolation="nearest")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Thinking Position")
    ax.set_title(title)

    # Y-axis labels from first example
    ylabels = results["examples"][0]["position_labels"][:max_pos]
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xticks(range(n_layers))

    plt.colorbar(im, ax=ax, label="Entropy (nats)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to {output_path}")


# ---- Line plot ----


def avg_entropy_per_layer(results):
    """Compute mean entropy at each layer, averaged over all positions and examples."""
    n_layers = results["n_layers"]
    layer_sums = np.zeros(n_layers)
    layer_counts = np.zeros(n_layers)
    for ex in results["examples"]:
        for row in ex["entropy"]:
            for l_idx, val in enumerate(row):
                layer_sums[l_idx] += val
                layer_counts[l_idx] += 1
    return layer_sums / np.maximum(layer_counts, 1)


def avg_entropy_per_layer_with_std(results):
    """Compute mean and std of entropy at each layer.

    Per-example entropy at a layer = mean across that example's positions.
    Then mean/std across examples.
    """
    n_layers = results["n_layers"]
    per_example = []  # (n_examples, n_layers)
    for ex in results["examples"]:
        # Mean across positions for this example
        layer_means = np.zeros(n_layers)
        for row in ex["entropy"]:
            layer_means += np.array(row)
        layer_means /= max(len(ex["entropy"]), 1)
        per_example.append(layer_means)

    per_example = np.array(per_example)  # (n_examples, n_layers)
    return per_example.mean(axis=0), per_example.std(axis=0)


def plot_entropy_lineplot(coconut_results, cot_results, output_path):
    """Line plot comparing average entropy per layer for both models."""
    coconut_mean, coconut_std = avg_entropy_per_layer_with_std(coconut_results)
    cot_mean, cot_std = avg_entropy_per_layer_with_std(cot_results)

    n_layers = coconut_results["n_layers"]
    layers = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(layers, coconut_mean, "o-", label="Coconut (latent)", color="#2196F3", linewidth=2)
    ax.fill_between(
        layers, coconut_mean - coconut_std, coconut_mean + coconut_std,
        color="#2196F3", alpha=0.15,
    )

    ax.plot(layers, cot_mean, "s--", label="CoT (discrete)", color="#FF5722", linewidth=2)
    ax.fill_between(
        layers, cot_mean - cot_std, cot_mean + cot_std,
        color="#FF5722", alpha=0.15,
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Average Entropy (nats)")
    ax.set_title("Logit Lens Entropy: Coconut vs CoT")
    ax.legend()
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved line plot to {output_path}")


# ---- Main ----


def main():
    parser = argparse.ArgumentParser(description="Plot logit lens results")
    parser.add_argument(
        "--coconut_results",
        type=str,
        default=None,
        help="Path to Coconut logit lens JSON",
    )
    parser.add_argument(
        "--cot_results",
        type=str,
        default=None,
        help="Path to CoT logit lens JSON",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis/figures/logit_lens",
        help="Directory to save plots",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Optional suffix for output filenames (e.g. '_prosqa', '_gsm')",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sfx = args.suffix

    coconut = None
    cot = None

    if args.coconut_results:
        coconut = load_results(args.coconut_results)
        plot_entropy_heatmap(
            coconut,
            "Coconut: Entropy at Latent Thought Positions",
            os.path.join(args.output_dir, f"entropy_heatmap_coconut{sfx}.png"),
        )

    if args.cot_results:
        cot = load_results(args.cot_results)
        plot_entropy_heatmap(
            cot,
            "CoT: Entropy at Step-End Positions",
            os.path.join(args.output_dir, f"entropy_heatmap_cot{sfx}.png"),
        )

    if coconut and cot:
        plot_entropy_lineplot(
            coconut,
            cot,
            os.path.join(args.output_dir, f"entropy_comparison{sfx}.png"),
        )


if __name__ == "__main__":
    main()
