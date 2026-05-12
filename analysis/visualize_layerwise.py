#!/usr/bin/env python3
"""Visualize layer-wise entity belief from analyze_layerwise_belief.py output.

Produces two figures:
  1. Heatmap: (layer x thinking_step), color = mean normalized P(target entity)
     — one panel per model (Coconut, CoT), side by side
  2. Line plot: mean normalized P(target) vs layer, one curve per thinking step
     — Coconut and CoT in separate subplots

Normalization: at each (example, position, layer), raw P(entity) values are
divided by their sum over all entities to get a proper distribution. Then
averaged over examples.

Usage:
    python visualize_layerwise.py \\
        --coconut_json analysis/results/prosqa/layerwise_belief_coconut.json \\
        --cot_json     analysis/results/prosqa/layerwise_belief_cot.json \\
        --output_dir   analysis/figures/layerwise \\
        --n_steps      4
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

matplotlib.rcParams.update({"font.size": 12})


# ---- Data loading and normalization ----


def load_json(path):
    with open(path) as f:
        return json.load(f)


def compute_normalized_target_probs(data, filter_n_steps=None):
    """Extract normalized P(target entity) for each example.

    Returns:
        matrix: np.ndarray of shape (n_examples, n_positions, n_layers)
                each value is normalized P(target) at that (position, layer)
        position_labels: list[str]
        n_layers: int
    """
    examples = data["examples"]
    if filter_n_steps is not None:
        examples = [ex for ex in examples if len(ex["steps"]) == filter_n_steps]

    if not examples:
        raise ValueError(f"No examples with {filter_n_steps} steps found.")

    position_labels = examples[0]["position_labels"]
    n_positions = len(position_labels)
    n_layers = data["n_layers"]

    all_probs = []  # (n_examples, n_positions, n_layers)
    for ex in examples:
        entities = ex["entities"]
        target_ent = entities[ex["target"]]
        entity_layer_probs = ex["entity_layer_probs"]

        # Normalize at each (position, layer): divide P(entity) by sum over all entities
        example_probs = np.zeros((n_positions, n_layers))
        for pos_idx in range(n_positions):
            for layer_idx in range(n_layers):
                raw = {ent: entity_layer_probs[ent][pos_idx][layer_idx] for ent in entities}
                total = sum(raw.values())
                example_probs[pos_idx, layer_idx] = (
                    raw[target_ent] / total if total > 0 else 0.0
                )
        all_probs.append(example_probs)

    return np.array(all_probs), position_labels, n_layers


def compute_normalized_all_entity_probs(data, filter_n_steps=None):
    """Extract full normalized entity distribution for each example.

    Returns:
        probs_by_entity: dict[entity -> np.ndarray (n_examples, n_positions, n_layers)]
        entities: list[str]  — union of all entity names (varies per example, so
                               per-example normalization is done independently)
        position_labels: list[str]
        n_layers: int
    """
    examples = data["examples"]
    if filter_n_steps is not None:
        examples = [ex for ex in examples if len(ex["steps"]) == filter_n_steps]

    position_labels = examples[0]["position_labels"]
    n_positions = len(position_labels)
    n_layers = data["n_layers"]

    # Each example has its own entity set — we track target and neg_target
    # as canonical "roles" rather than names
    target_probs = []
    neg_target_probs = []

    for ex in examples:
        entities = ex["entities"]
        target_ent = entities[ex["target"]]
        neg_target_ent = entities[ex["neg_target"]]
        elp = ex["entity_layer_probs"]

        t_probs = np.zeros((n_positions, n_layers))
        n_probs = np.zeros((n_positions, n_layers))
        for pos_idx in range(n_positions):
            for layer_idx in range(n_layers):
                raw = {ent: elp[ent][pos_idx][layer_idx] for ent in entities}
                total = sum(raw.values())
                if total > 0:
                    t_probs[pos_idx, layer_idx] = raw[target_ent] / total
                    n_probs[pos_idx, layer_idx] = raw[neg_target_ent] / total
        target_probs.append(t_probs)
        neg_target_probs.append(n_probs)

    return (
        np.array(target_probs),    # (n_examples, n_positions, n_layers)
        np.array(neg_target_probs),
        position_labels,
        n_layers,
    )


# ---- Plotting ----


def plot_heatmap(coconut_json, cot_json, output_path, filter_n_steps=None):
    """Side-by-side heatmap: layer (y) x thinking step (x), color = mean P(target).

    Rows = layers (0 = shallowest, n_layers-1 = deepest).
    Columns = thinking steps (thought_1..N for Coconut, step_1..N for CoT).
    """
    coconut_data = load_json(coconut_json)
    cot_data = load_json(cot_json)

    coconut_probs, coconut_labels, coconut_n_layers = compute_normalized_target_probs(
        coconut_data, filter_n_steps
    )
    cot_probs, cot_labels, cot_n_layers = compute_normalized_target_probs(
        cot_data, filter_n_steps
    )

    # Mean over examples: (n_positions, n_layers) -> transpose to (n_layers, n_positions)
    coconut_mean = coconut_probs.mean(axis=0).T   # (n_layers, n_positions)
    cot_mean = cot_probs.mean(axis=0).T

    # Shared color scale
    vmin = 0.0
    vmax = max(coconut_mean.max(), cot_mean.max())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    for ax, mean_matrix, labels, title in [
        (axes[0], coconut_mean, coconut_labels, "Coconut (latent thoughts)"),
        (axes[1], cot_mean, cot_labels, "CoT (step-end tokens)"),
    ]:
        im = ax.imshow(
            mean_matrix,
            aspect="auto",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
            interpolation="nearest",
        )
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(
            [lbl.replace("_end", "").replace("thought_", "t").replace("step_", "s")
             for lbl in labels],
            fontsize=11,
        )
        ax.set_yticks(range(mean_matrix.shape[0]))
        ax.set_yticklabels([f"L{i+1}" for i in range(mean_matrix.shape[0])], fontsize=10)
        ax.set_xlabel("Thinking position", fontsize=13)
        ax.set_ylabel("Transformer layer", fontsize=13)
        ax.set_title(title, fontsize=14, fontweight="bold")

    # Shared colorbar
    cbar = fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap="viridis"),
        ax=axes,
        label="Mean normalized P(target entity)",
        shrink=0.8,
        pad=0.02,
    )

    n_steps_str = f" ({filter_n_steps}-step)" if filter_n_steps is not None else ""
    fig.suptitle(
        f"Layer-wise entity belief{n_steps_str}: when does the model commit?",
        fontsize=15,
    )

    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved heatmap to {output_path}")
    plt.close()


def plot_layer_curves(coconut_json, cot_json, output_path, filter_n_steps=None):
    """Line plot: mean P(target) vs layer, one curve per thinking step.

    Shows how entity belief builds (or doesn't) across transformer depth.
    Coconut and CoT in separate subplots for direct comparison.
    """
    coconut_data = load_json(coconut_json)
    cot_data = load_json(cot_json)

    coconut_probs, coconut_labels, coconut_n_layers = compute_normalized_target_probs(
        coconut_data, filter_n_steps
    )
    cot_probs, cot_labels, cot_n_layers = compute_normalized_target_probs(
        cot_data, filter_n_steps
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    layer_ticks = None

    for ax, probs, labels, title, color_map in [
        (axes[0], coconut_probs, coconut_labels, "Coconut (latent thoughts)", plt.cm.Blues),
        (axes[1], cot_probs, cot_labels, "CoT (step-end tokens)", plt.cm.Oranges),
    ]:
        n_positions = len(labels)
        n_layers = probs.shape[2]
        x = np.arange(1, n_layers + 1)
        layer_ticks = x

        # Color gradient: earlier steps lighter, later steps darker
        colors = color_map(np.linspace(0.35, 0.9, n_positions))

        for pos_idx, (label, color) in enumerate(zip(labels, colors)):
            # probs: (n_examples, n_positions, n_layers)
            pos_probs = probs[:, pos_idx, :]  # (n_examples, n_layers)
            mean = pos_probs.mean(axis=0)
            std = pos_probs.std(axis=0)

            short_label = (label.replace("_end", "")
                               .replace("thought_", "t")
                               .replace("step_", "s"))
            ax.plot(x, mean, "-o", color=color, label=short_label, linewidth=1.8,
                    markersize=4)
            ax.fill_between(x, np.maximum(mean - std, 0), mean + std,
                            color=color, alpha=0.12)

        ax.axhline(1.0 / 8, color="gray", linestyle=":", linewidth=1,
                   label="uniform (1/8 entities)")
        ax.set_xlabel("Transformer layer", fontsize=13)
        ax.set_ylabel("Mean normalized P(target entity)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(layer_ticks)
        ax.set_xticklabels([str(i) for i in layer_ticks], fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=10, ncol=2, frameon=False)

    n_steps_str = f" ({filter_n_steps}-step)" if filter_n_steps is not None else ""
    fig.suptitle(
        f"P(target entity) across transformer layers{n_steps_str}",
        fontsize=15,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved layer curves to {output_path}")
    plt.close()


def plot_target_vs_negtarget(coconut_json, cot_json, output_path, filter_n_steps=None):
    """Line plot: P(target) and P(neg_target) vs layer, averaged over steps.

    Averaged over all thinking steps and examples. Shows whether the model's
    commitment is to the target or to a competing entity.
    """
    coconut_data = load_json(coconut_json)
    cot_data = load_json(cot_json)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, data, title, c_target, c_neg in [
        (axes[0], coconut_data, "Coconut (latent thoughts)", "#2196F3", "#FF5722"),
        (axes[1], cot_data, "CoT (step-end tokens)", "#1976D2", "#E64A19"),
    ]:
        t_probs, nt_probs, labels, n_layers = compute_normalized_all_entity_probs(
            data, filter_n_steps
        )
        # Average over all positions, then over examples
        # t_probs: (n_examples, n_positions, n_layers)
        t_mean_over_steps = t_probs.mean(axis=1)   # (n_examples, n_layers)
        nt_mean_over_steps = nt_probs.mean(axis=1)

        x = np.arange(1, n_layers + 1)

        t_mean = t_mean_over_steps.mean(axis=0)
        t_std = t_mean_over_steps.std(axis=0)
        nt_mean = nt_mean_over_steps.mean(axis=0)
        nt_std = nt_mean_over_steps.std(axis=0)

        ax.plot(x, t_mean, "-o", color=c_target, label="target", linewidth=2, markersize=4)
        ax.fill_between(x, np.maximum(t_mean - t_std, 0), t_mean + t_std,
                        color=c_target, alpha=0.15)

        ax.plot(x, nt_mean, "s", color=c_neg, label="neg_target", linewidth=2,
                markersize=4, linestyle="--")
        ax.fill_between(x, np.maximum(nt_mean - nt_std, 0), nt_mean + nt_std,
                        color=c_neg, alpha=0.15)

        ax.axhline(1.0 / 8, color="gray", linestyle=":", linewidth=1,
                   label="uniform")
        ax.set_xlabel("Transformer layer", fontsize=13)
        ax.set_ylabel("Mean normalized P(entity)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=11, frameon=False)

    n_steps_str = f" ({filter_n_steps}-step)" if filter_n_steps is not None else ""
    fig.suptitle(
        f"Target vs neg-target belief across layers{n_steps_str}",
        fontsize=15,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved target vs neg_target plot to {output_path}")
    plt.close()


# ---- Main ----


def main():
    parser = argparse.ArgumentParser(
        description="Visualize layer-wise entity belief (logit lens)"
    )
    parser.add_argument(
        "--coconut_json", type=str, required=True,
        help="Path to layerwise_belief_coconut.json"
    )
    parser.add_argument(
        "--cot_json", type=str, required=True,
        help="Path to layerwise_belief_cot.json"
    )
    parser.add_argument(
        "--output_dir", type=str, default="analysis/figures/layerwise"
    )
    parser.add_argument(
        "--n_steps", type=int, default=None,
        help="Filter to examples with exactly this many reasoning steps"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    suffix = f"_{args.n_steps}step" if args.n_steps is not None else ""

    plot_heatmap(
        args.coconut_json, args.cot_json,
        os.path.join(args.output_dir, f"heatmap{suffix}.pdf"),
        filter_n_steps=args.n_steps,
    )
    plot_layer_curves(
        args.coconut_json, args.cot_json,
        os.path.join(args.output_dir, f"layer_curves{suffix}.pdf"),
        filter_n_steps=args.n_steps,
    )
    plot_target_vs_negtarget(
        args.coconut_json, args.cot_json,
        os.path.join(args.output_dir, f"target_vs_negtarget{suffix}.pdf"),
        filter_n_steps=args.n_steps,
    )


if __name__ == "__main__":
    main()
