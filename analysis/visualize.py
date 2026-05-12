#!/usr/bin/env python3
"""Visualize entity probability analysis for Coconut vs CoT models.

Produces:
  1. Entropy plot: mean entropy of normalized entity distribution vs reasoning step
  2. Stacked bar charts: per-example entity probability breakdown (cherry-picked)
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({"font.size": 12})


def normalize_entity_probs(entity_probs, entities):
    """Normalize raw entity probs to sum to 1 at each position."""
    n_positions = len(entity_probs[entities[0]])
    normalized = {ent: [] for ent in entities}
    for i in range(n_positions):
        total = sum(entity_probs[ent][i] for ent in entities)
        for ent in entities:
            normalized[ent].append(
                entity_probs[ent][i] / total if total > 0 else 0.0
            )
    return normalized


def compute_entropy(normalized_probs, entities, position_idx):
    """Compute entropy H = -sum(p * log(p)) at a given position."""
    h = 0.0
    for ent in entities:
        p = normalized_probs[ent][position_idx]
        if p > 0:
            h -= p * np.log2(p)
    return h


def find_is_a_positions(example):
    """Find the 'is a' positions in a CoT trace (one per reasoning step).

    These are the positions where the model predicts the next entity,
    which is the most meaningful probe point in the CoT trace.
    """
    positions = []
    for step_info in example["step_boundaries"]:
        start, end = step_info["start"], step_info["end"]
        # Search backwards from end for the " a" token within this step
        for local_idx in range(end - 1, start - 1, -1):
            # Map to position_labels index (offset by positions_of_interest[0])
            label_idx = local_idx - example["positions"][0]
            if 0 <= label_idx < len(example["position_labels"]):
                if example["position_labels"][label_idx] == " a":
                    positions.append(label_idx)
                    break
    return positions


def plot_entropy(coconut_data, cot_data, n_steps, output_path):
    """Plot entropy vs reasoning step for both models."""
    # Filter to examples with exactly n_steps
    coconut_examples = [
        ex for ex in coconut_data["examples"] if len(ex["steps"]) == n_steps
    ]
    cot_examples = [
        ex for ex in cot_data["examples"] if len(ex["steps"]) == n_steps
    ]

    # Match examples by idx
    cot_by_idx = {ex["idx"]: ex for ex in cot_examples}
    matched_idxs = [ex["idx"] for ex in coconut_examples if ex["idx"] in cot_by_idx]

    print(f"Matched {len(matched_idxs)} examples with {n_steps} steps")

    # Compute entropy for Coconut at each thought step
    # Positions: before_thinking, thought_1, ..., thought_6, after_thinking
    # Use thought_1 through thought_6 (indices 1-6 in position_labels)
    n_thoughts = coconut_data.get("n_latent", 6)
    coconut_entropies = []  # shape: (n_examples, n_thoughts)
    for idx in matched_idxs:
        ex = next(e for e in coconut_examples if e["idx"] == idx)
        entities = ex["entities"]
        normed = normalize_entity_probs(ex["entity_probs"], entities)
        # Use thoughts 1..n_thoughts (skip before_thinking and after_thinking)
        entropies = [compute_entropy(normed, entities, i) for i in range(1, n_thoughts + 1)]
        coconut_entropies.append(entropies)

    coconut_entropies = np.array(coconut_entropies)

    # Compute entropy for CoT at each "is a" position
    cot_entropies = []  # shape: (n_examples, n_steps)
    for idx in matched_idxs:
        ex = cot_by_idx[idx]
        entities = ex["entities"]
        normed = normalize_entity_probs(ex["entity_probs"], entities)
        is_a_positions = find_is_a_positions(ex)
        if len(is_a_positions) != n_steps:
            continue
        entropies = [compute_entropy(normed, entities, pos) for pos in is_a_positions]
        cot_entropies.append(entropies)

    cot_entropies = np.array(cot_entropies)
    print(f"Coconut: {coconut_entropies.shape}, CoT: {cot_entropies.shape}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    # Coconut
    x_coconut = np.arange(1, n_thoughts + 1)
    mean_c = coconut_entropies.mean(axis=0)
    std_c = coconut_entropies.std(axis=0)
    ax.plot(x_coconut, mean_c, "o-", color="#2196F3", label="Coconut (latent)", linewidth=2)
    ax.fill_between(x_coconut, np.maximum(mean_c - std_c, 0), mean_c + std_c, color="#2196F3", alpha=0.15)

    # CoT
    x_cot = np.arange(1, n_steps + 1)
    mean_t = cot_entropies.mean(axis=0)
    std_t = cot_entropies.std(axis=0)
    ax.plot(x_cot, mean_t, "s--", color="#FF5722", label="CoT (discrete)", linewidth=2)
    ax.fill_between(x_cot, np.maximum(mean_t - std_t, 0), mean_t + std_t, color="#FF5722", alpha=0.15)

    ax.set_xlabel("Reasoning Step")
    ax.set_ylabel("Entropy (bits)")
    ax.set_title(f"Entity Distribution Entropy During Reasoning ({n_steps}-step examples)")
    ax.legend()
    ax.set_xticks(np.arange(1, max(n_thoughts, n_steps) + 1))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved entropy plot to {output_path}")
    plt.close()


def plot_stacked_bars(coconut_data, cot_data, example_idx, n_top, output_path):
    """Plot stacked bar chart for a single example, Coconut vs CoT side by side."""
    coconut_ex = next(
        ex for ex in coconut_data["examples"] if ex["idx"] == example_idx
    )
    cot_ex = next(ex for ex in cot_data["examples"] if ex["idx"] == example_idx)

    entities = coconut_ex["entities"]
    target = entities[coconut_ex["target"]]
    neg_target = entities[coconut_ex["neg_target"]]
    n_thoughts = coconut_data.get("n_latent", 6)

    # Normalize
    coconut_normed = normalize_entity_probs(coconut_ex["entity_probs"], entities)
    cot_normed = normalize_entity_probs(cot_ex["entity_probs"], entities)

    # Find "is a" positions for CoT
    is_a_positions = find_is_a_positions(cot_ex)
    n_steps = len(is_a_positions)

    # Select entities to display: path entities + target + neg_target
    # Extract path entities from reasoning steps (e.g. "lempus is a zumpus" -> lempus, zumpus)
    path_ents = set()
    for step in coconut_ex["steps"]:
        parts = step.rstrip(".").split(" is a ")
        for part in parts:
            part = part.strip()
            if part in entities:
                path_ents.add(part)
    path_ents.add(target)
    path_ents.add(neg_target)
    # Order: path entities sorted by max Coconut prob, then fill remaining slots
    top_ents = sorted(path_ents, key=lambda e: -max(coconut_normed[e][1 : n_thoughts + 1]))
    other_ents = [e for e in entities if e not in top_ents]

    # Color map: target gets a distinct color, neg_target another
    colors = plt.cm.Set2(np.linspace(0, 1, n_top + 1))
    ent_colors = {}
    for i, ent in enumerate(top_ents):
        ent_colors[ent] = colors[i]
    ent_colors["other"] = colors[n_top]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Coconut panel ---
    x_c = np.arange(n_thoughts)
    bottom = np.zeros(n_thoughts)
    for ent in top_ents:
        vals = [coconut_normed[ent][i + 1] for i in range(n_thoughts)]
        label = f"{ent}" + (" *" if ent == target else "")
        ax1.bar(x_c, vals, bottom=bottom, label=label, color=ent_colors[ent], width=0.7)
        bottom += vals
    # Other
    other_vals = np.zeros(n_thoughts)
    for ent in other_ents:
        other_vals += [coconut_normed[ent][i + 1] for i in range(n_thoughts)]
    ax1.bar(x_c, other_vals, bottom=bottom, label="other", color=ent_colors["other"], width=0.7)

    ax1.set_xlabel("Latent Thought Step")
    ax1.set_ylabel("Normalized P(entity)")
    ax1.set_title("Coconut (Latent)")
    ax1.set_xticks(x_c)
    ax1.set_xticklabels([f"T{i+1}" for i in range(n_thoughts)])
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=9, loc="upper right")

    # --- CoT panel ---
    x_t = np.arange(n_steps)
    bottom = np.zeros(n_steps)
    for ent in top_ents:
        vals = [cot_normed[ent][pos] for pos in is_a_positions]
        label = f"{ent}" + (" *" if ent == target else "")
        ax2.bar(x_t, vals, bottom=bottom, label=label, color=ent_colors[ent], width=0.7)
        bottom += vals
    other_vals = np.zeros(n_steps)
    for ent in other_ents:
        other_vals += [cot_normed[ent][pos] for pos in is_a_positions]
    ax2.bar(x_t, other_vals, bottom=bottom, label="other", color=ent_colors["other"], width=0.7)

    ax2.set_xlabel("Reasoning Step")
    ax2.set_ylabel("Normalized P(entity)")
    ax2.set_title("CoT (Discrete)")
    ax2.set_xticks(x_t)
    ax2.set_xticklabels([f"S{i+1}" for i in range(n_steps)])
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=9, loc="upper right")

    # Suptitle with example info
    fig.suptitle(
        f"Example {example_idx}: {entities[coconut_ex['root']]} → {target} "
        f"({len(coconut_ex['steps'])} steps)\n"
        f"Path: {' → '.join(s.rstrip('.').split(' is a ')[-1] if 'is a' in s else s.split(' is a ')[0] for s in coconut_ex['steps'])}",
        fontsize=11,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved stacked bar plot to {output_path}")
    plt.close()


def get_root_neighbors(example):
    """Get the set of entity names that are direct neighbors of the root."""
    entities = example["entities"]
    root_idx = example["root"]
    neighbors = set()
    for src, dst in example["edges"]:
        if src == root_idx:
            neighbors.add(entities[dst])
        if dst == root_idx:
            neighbors.add(entities[src])
    return neighbors


def get_correct_first_hop(example):
    """Extract the correct first-hop entity from the reasoning steps."""
    if not example["steps"]:
        return None
    first_step = example["steps"][0].rstrip(".")
    parts = first_step.split(" is a ")
    if len(parts) == 2:
        return parts[1].strip()
    return None


def get_correct_entity_at_step(example, step_idx):
    """Extract the correct entity the model should predict at reasoning step step_idx.

    step_idx=0 → first hop (RHS of steps[0]),
    step_idx=1 → second hop (RHS of steps[1]), etc.
    Returns None if step_idx >= len(steps).
    """
    if step_idx >= len(example["steps"]):
        return None
    step = example["steps"][step_idx].rstrip(".")
    parts = step.split(" is a ")
    if len(parts) == 2:
        return parts[1].strip()
    return None


def get_node_neighbors_outgoing(example, node_name):
    """Get outgoing neighbors of a node (directed edges only: src → dst)."""
    entities = example["entities"]
    name_to_idx = {name: i for i, name in enumerate(entities)}
    node_idx = name_to_idx.get(node_name)
    if node_idx is None:
        return set()
    neighbors = set()
    for src, dst in example["edges"]:
        if src == node_idx:
            neighbors.add(entities[dst])
    return neighbors


def get_step_info(example, step_idx):
    """Get (current_node, correct_next, outgoing_neighbors) at reasoning step step_idx.

    At step 0, current node is the root.
    At step k>0, current node is the RHS of steps[k-1].
    Returns (None, None, set()) if step_idx is out of range.
    """
    entities = example["entities"]
    if step_idx >= len(example["steps"]):
        return None, None, set()
    if step_idx == 0:
        current_node = entities[example["root"]]
    else:
        prev = example["steps"][step_idx - 1].rstrip(".").split(" is a ")
        current_node = prev[1].strip() if len(prev) == 2 else None
    if current_node is None:
        return None, None, set()
    correct_next = get_correct_entity_at_step(example, step_idx)
    neighbors = get_node_neighbors_outgoing(example, current_node)
    return current_node, correct_next, neighbors


def plot_question_only_summary(coconut_qonly, cot_qonly, output_path):
    """2-panel stacked bar: entity belief right after reading the question.

    Same category breakdown as branching_summary (correct 1-hop, other 1-hops,
    target, other) but probed at the last question token before any reasoning.
    """
    from collections import defaultdict

    coconut_by_idx = {ex["idx"]: ex for ex in coconut_qonly["examples"]}
    cot_by_idx = {ex["idx"]: ex for ex in cot_qonly["examples"]}

    degree_groups = defaultdict(lambda: {"coconut": [], "cot": []})

    for ex in coconut_qonly["examples"]:
        idx = ex["idx"]
        if idx not in cot_by_idx:
            continue
        cot_ex = cot_by_idx[idx]

        entities = ex["entities"]
        target = entities[ex["target"]]
        neighbors = get_root_neighbors(ex)
        correct_hop = get_correct_first_hop(ex)
        deg = len(neighbors)

        coconut_normed = normalize_entity_probs(ex["entity_probs"], entities)
        cot_normed = normalize_entity_probs(cot_ex["entity_probs"], entities)

        # Both have a single probe position at index 0
        for model_key, normed in [
            ("coconut", coconut_normed),
            ("cot", cot_normed),
        ]:
            p_correct_hop = normed[correct_hop][0] if correct_hop else 0.0
            p_other_hops = sum(
                normed[e][0] for e in neighbors if e != correct_hop
            )
            p_target = normed[target][0] if target != correct_hop else 0.0
            p_other = 1.0 - p_correct_hop - p_other_hops - p_target

            degree_groups[deg][model_key].append({
                "correct_hop": p_correct_hop,
                "other_hops": p_other_hops,
                "target": p_target,
                "other": max(p_other, 0.0),
            })

    bins = [
        ("Low (1-2)", lambda d: d <= 2),
        ("Med (3)", lambda d: d == 3),
        ("High (4+)", lambda d: d >= 4),
    ]
    binned = {label: {"coconut": [], "cot": []} for label, _ in bins}
    for deg in degree_groups:
        for label, pred in bins:
            if pred(deg):
                binned[label]["coconut"].extend(degree_groups[deg]["coconut"])
                binned[label]["cot"].extend(degree_groups[deg]["cot"])
                break

    bin_labels = [label for label, _ in bins]
    categories = ["correct_hop", "other_hops", "target", "other"]
    cat_labels = ["Correct 1-hop", "Other 1-hops", "Target (final)", "Other"]
    cat_colors = ["#3a86a8", "#e8a838", "#c44e52", "#bdbdbd"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5),
                                    sharey=True, gridspec_kw={"wspace": 0.08})

    bar_handles = []
    for ax, model_key, model_label in [
        (ax1, "coconut", "Coconut"),
        (ax2, "cot", "CoT"),
    ]:
        x = np.arange(len(bin_labels))
        bottom = np.zeros(len(bin_labels))
        for cat, label, color in zip(categories, cat_labels, cat_colors):
            vals = []
            for bl in bin_labels:
                examples = binned[bl][model_key]
                vals.append(np.mean([e[cat] for e in examples]) if examples else 0.0)
            vals = np.array(vals)
            h = ax.bar(x, vals, bottom=bottom, label=label, color=color,
                       width=0.55, edgecolor="white", linewidth=0.5)
            if ax == ax1:
                bar_handles.append(h)
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels([f"{bl}\n$n$={len(binned[bl][model_key])}"
                            for bl in bin_labels], fontsize=9)
        ax.set_xlabel("Root Out-Degree", fontsize=11)
        ax.set_title(model_label, fontsize=12, fontweight="bold", pad=8)
        ax.set_ylim(0, 1.02)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=9)

    ax1.set_ylabel("Mean Normalized P(entity)", fontsize=11)

    fig.legend(
        bar_handles, cat_labels,
        loc="lower center", ncol=4, fontsize=9,
        frameon=False, bbox_to_anchor=(0.5, -0.01),
        columnspacing=1.5, handlelength=1.2,
    )

    fig.suptitle(
        "Question-Only Belief: Entity Distribution Before Reasoning",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.subplots_adjust(bottom=0.18)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved question-only summary to {output_path}")
    plt.close()


def _collect_stepwise_probs(coconut_data, cot_data, normalize_among_neighbors,
                            filter_n_steps=None):
    """Shared logic for stepwise plots.

    At each reasoning step k:
      - Determine current node (root at k=0, RHS of steps[k-1] at k>0)
      - Find outgoing neighbors of current node
      - Compute P(correct next) and P(wrong neighbors)

    If normalize_among_neighbors=True, probabilities are normalized among the
    current node's outgoing neighbors only.  Otherwise, among all entities.
    If filter_n_steps is set, only include examples with exactly that many steps.

    Returns:
        coconut_steps: list of dicts per step, each with keys
            'correct', 'wrong', 'n' (lists of per-example values, and count)
        cot_steps: same for CoT
        max_coconut_steps, max_cot_steps: int
    """
    n_latent = coconut_data.get("n_latent", 6)
    cot_by_idx = {ex["idx"]: ex for ex in cot_data["examples"]}

    matched = []
    for ex in coconut_data["examples"]:
        idx = ex["idx"]
        if idx not in cot_by_idx:
            continue
        if filter_n_steps is not None and len(ex["steps"]) != filter_n_steps:
            continue
        cot_ex = cot_by_idx[idx]
        is_a_pos = find_is_a_positions(cot_ex)
        matched.append((ex, cot_ex, is_a_pos))

    # Coconut: positions 0..n_latent-1 map to steps 0..n_latent-1
    # (skip "after_thinking" which doesn't map to a graph step)
    max_coconut_steps = n_latent
    max_cot_steps = max((len(m[2]) for m in matched), default=0)

    def compute_probs_at_step(normed, entities, correct_next, neighbors, target,
                              pos_idx):
        if normalize_among_neighbors:
            neighbor_raw = {n: normed[n][pos_idx] for n in neighbors}
            total = sum(neighbor_raw.values())
            if total == 0:
                return None
            p_correct = neighbor_raw.get(correct_next, 0.0) / total
            p_wrong = sum(v for n, v in neighbor_raw.items()
                          if n != correct_next) / total
            return {"correct": p_correct, "wrong": p_wrong,
                    "n_neighbors": len(neighbors)}
        else:
            # When correct_next IS the target (final step), attribute mass
            # to "target" rather than "correct next" so the red area continues.
            if correct_next and correct_next == target:
                p_correct = 0.0
                p_target = normed[target][pos_idx]
                p_wrong = sum(normed[e][pos_idx] for e in neighbors
                              if e != correct_next)
            else:
                p_correct = normed[correct_next][pos_idx] if correct_next else 0.0
                p_wrong = sum(normed[e][pos_idx] for e in neighbors
                              if e != correct_next)
                accounted = set(neighbors) | {correct_next}
                p_target = normed[target][pos_idx] if target not in accounted else 0.0
            p_other = max(1.0 - p_correct - p_wrong - p_target, 0.0)
            return {"correct": p_correct, "wrong": p_wrong,
                    "target": p_target, "other": p_other}

    # --- Coconut ---
    coconut_steps = []
    for step_idx in range(max_coconut_steps):
        step_records = []
        for coconut_ex, cot_ex, is_a_pos in matched:
            entities = coconut_ex["entities"]
            if step_idx >= len(coconut_ex["steps"]):
                continue
            if step_idx >= len(coconut_ex["entity_probs"][entities[0]]):
                continue
            normed = normalize_entity_probs(coconut_ex["entity_probs"], entities)
            _, correct_next, neighbors = get_step_info(coconut_ex, step_idx)
            if correct_next is None or len(neighbors) == 0:
                continue
            target = entities[coconut_ex["target"]]
            rec = compute_probs_at_step(
                normed, entities, correct_next, neighbors, target, step_idx
            )
            if rec is not None:
                step_records.append(rec)
        coconut_steps.append({"records": step_records, "n": len(step_records)})

    # --- CoT ---
    cot_steps = []
    for step_idx in range(max_cot_steps):
        step_records = []
        for coconut_ex, cot_ex, is_a_pos in matched:
            if step_idx >= len(is_a_pos):
                continue
            if step_idx >= len(cot_ex["steps"]):
                continue
            entities = cot_ex["entities"]
            normed = normalize_entity_probs(cot_ex["entity_probs"], entities)
            _, correct_next, neighbors = get_step_info(cot_ex, step_idx)
            if correct_next is None or len(neighbors) == 0:
                continue
            target = entities[cot_ex["target"]]
            cot_pos = is_a_pos[step_idx]
            rec = compute_probs_at_step(
                normed, entities, correct_next, neighbors, target, cot_pos
            )
            if rec is not None:
                step_records.append(rec)
        cot_steps.append({"records": step_records, "n": len(step_records)})

    # Trim trailing steps with no data
    while coconut_steps and coconut_steps[-1]["n"] == 0:
        coconut_steps.pop()
    while cot_steps and cot_steps[-1]["n"] == 0:
        cot_steps.pop()
    max_coconut_steps = len(coconut_steps)
    max_cot_steps = len(cot_steps)

    return coconut_steps, cot_steps, max_coconut_steps, max_cot_steps


def _plot_stepwise(coconut_steps, cot_steps, max_coconut_steps, max_cot_steps,
                   output_path, title, ylabel):
    """Render the 1x2 stacked area plot for stepwise neighbor-norm analysis."""
    cat_labels = ["Correct next", "Wrong neighbors"]
    cat_colors = ["#3a86a8", "#e8a838"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, steps, max_steps, model_label in [
        (ax1, coconut_steps, max_coconut_steps, "Coconut"),
        (ax2, cot_steps, max_cot_steps, "CoT"),
    ]:
        x = np.arange(max_steps)
        mean_c = np.array([np.mean([r["correct"] for r in s["records"]]) if s["n"] > 0 else 0.0 for s in steps])
        mean_w = np.array([np.mean([r["wrong"] for r in s["records"]]) if s["n"] > 0 else 0.0 for s in steps])
        ns = [s["n"] for s in steps]

        # Chance line: mean(1/n_neighbors) per step
        chance = np.array([
            np.mean([1.0 / r["n_neighbors"] for r in s["records"]])
            if s["n"] > 0 else 0.0
            for s in steps
        ])

        # Stacked area
        bottom = np.zeros(max_steps)
        for vals, label, color in [(mean_c, cat_labels[0], cat_colors[0]),
                                    (mean_w, cat_labels[1], cat_colors[1])]:
            ax.fill_between(x, bottom, bottom + vals, color=color,
                            alpha=0.85, label=label, linewidth=0)
            ax.plot(x, bottom + vals, color=color, linewidth=0.8)
            bottom = bottom + vals

        # Overlay chance level
        ax.plot(x, chance, "k--", linewidth=1.5, label="Chance (1/degree)",
                zorder=5)

        ax.set_xticks(x)
        ax.set_xticklabels([f"Step {i}\n$n$={ns[i]}" for i in range(max_steps)],
                           fontsize=8)
        ax.set_xlabel("Reasoning Step", fontsize=11)
        ax.set_title(model_label, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.2, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax1.set_ylabel(ylabel, fontsize=11)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02),
               columnspacing=1.5, handlelength=1.2)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved {output_path}")
    plt.close()


def plot_stepwise_neighbor_norm(coconut_data, cot_data, output_path,
                                filter_n_steps=None):
    """P(correct next) vs P(wrong neighbors), normalized among current-node neighbors."""
    coconut_steps, cot_steps, mc, mt = _collect_stepwise_probs(
        coconut_data, cot_data, normalize_among_neighbors=True,
        filter_n_steps=filter_n_steps,
    )
    suffix = f" ({filter_n_steps}-step examples)" if filter_n_steps else ""
    _plot_stepwise(
        coconut_steps, cot_steps, mc, mt, output_path,
        title=f"Step-Aware Neighbor Belief (Normalized Among Current-Node Neighbors){suffix}",
        ylabel="P (among neighbors)",
    )


def plot_stepwise_all_entity_norm(coconut_data, cot_data, output_path,
                                  filter_n_steps=None):
    """Stacked area: correct next, wrong neighbors, target, other (sums to 1).

    Produces individual square plots per model. If output_path ends in a known
    extension, saves <stem>_coconut.<ext> and <stem>_cot.<ext>.
    """
    coconut_steps, cot_steps, mc, mt = _collect_stepwise_probs(
        coconut_data, cot_data, normalize_among_neighbors=False,
        filter_n_steps=filter_n_steps,
    )

    categories = ["correct", "wrong", "target", "other"]
    cat_labels = ["Correct next", "Wrong neighbors", "Target (final)", "Other"]
    cat_colors = ["#3a86a8", "#e8a838", "#c44e52", "#bdbdbd"]

    stem, ext = os.path.splitext(output_path)

    for steps, max_steps, model_label in [
        (coconut_steps, mc, "Coconut"),
        (cot_steps, mt, "CoT"),
    ]:
        fig, ax = plt.subplots(figsize=(5, 5))

        x = np.arange(max_steps)
        means = {}
        for cat in categories:
            means[cat] = np.array([
                np.mean([r[cat] for r in s["records"]]) if s["n"] > 0 else 0.0
                for s in steps
            ])
        ns = [s["n"] for s in steps]

        # Stacked area
        bottom = np.zeros(max_steps)
        for cat, label, color in zip(categories, cat_labels, cat_colors):
            ax.fill_between(x, bottom, bottom + means[cat], color=color,
                            alpha=0.85, label=label, linewidth=0)
            ax.plot(x, bottom + means[cat], color=color, linewidth=0.8)
            bottom = bottom + means[cat]

        ax.set_xticks(x)
        ax.set_xticklabels([f"{i}" for i in range(max_steps)], fontsize=13)
        ax.set_xlabel("Reasoning Step", fontsize=14)
        ax.set_ylabel("Normalized P(entity)", fontsize=14)
        ax.tick_params(axis='y', labelsize=13)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.2, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Legend below plot
        ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.15),
                  ncol=2, frameon=False, columnspacing=1.0)

        plt.tight_layout()
        save_path = f"{stem}_{model_label.lower()}{ext}"
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved {save_path}")
        plt.close()


def plot_stepwise_all_entity_norm_grid(coconut_data, cot_data, output_path,
                                       filter_n_steps_list=(4, 5)):
    """2x2 grid: rows = n_steps, cols = Coconut/CoT. One shared legend at bottom."""
    categories = ["correct", "wrong", "target", "other"]
    cat_labels = ["Correct next", "Wrong neighbors", "Target (final)", "Other"]
    cat_colors = ["#3a86a8", "#e8a838", "#c44e52", "#bdbdbd"]

    nrows = len(filter_n_steps_list)
    fig, axes = plt.subplots(nrows, 2, figsize=(10, 5 * nrows), sharey=True)
    if nrows == 1:
        axes = axes[np.newaxis, :]

    for row_idx, n_steps in enumerate(filter_n_steps_list):
        coconut_steps, cot_steps, mc, mt = _collect_stepwise_probs(
            coconut_data, cot_data, normalize_among_neighbors=False,
            filter_n_steps=n_steps,
        )
        for col_idx, (steps, max_steps, model_label) in enumerate([
            (coconut_steps, mc, "Coconut"),
            (cot_steps, mt, "CoT"),
        ]):
            ax = axes[row_idx, col_idx]
            x = np.arange(max_steps)
            means = {}
            for cat in categories:
                means[cat] = np.array([
                    np.mean([r[cat] for r in s["records"]]) if s["n"] > 0 else 0.0
                    for s in steps
                ])

            bottom = np.zeros(max_steps)
            for cat, label, color in zip(categories, cat_labels, cat_colors):
                ax.fill_between(x, bottom, bottom + means[cat], color=color,
                                alpha=0.85, label=label if row_idx == 0 and col_idx == 0 else None,
                                linewidth=0)
                ax.plot(x, bottom + means[cat], color=color, linewidth=0.8)
                bottom = bottom + means[cat]

            ax.set_xticks(x)
            ax.set_xticklabels([f"{i}" for i in range(max_steps)], fontsize=15)
            ax.tick_params(axis='y', labelsize=15)
            ax.set_ylim(0, 1.0)
            ax.grid(True, alpha=0.2, axis="y")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Column titles on top row only
            if row_idx == 0:
                ax.set_title(model_label, fontsize=17, fontweight="bold")

            # Row labels on left column only (with n_steps integrated)
            if col_idx == 0:
                ax.set_ylabel(f"{n_steps}-step — Normalized P(entity)", fontsize=16)

            # X-axis label on bottom row only
            if row_idx == nrows - 1:
                ax.set_xlabel("Reasoning Step", fontsize=16)

    # Shared legend at bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=14,
               frameon=False, bbox_to_anchor=(0.5, -0.03),
               columnspacing=1.5, handlelength=1.5)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08, hspace=0.25)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize entity probability analysis")
    parser.add_argument(
        "--coconut_json",
        type=str,
        default="analysis/results/coconut_entity_probs.json",
    )
    parser.add_argument(
        "--cot_json", type=str, default="analysis/results/cot_entity_probs.json"
    )
    parser.add_argument(
        "--output_dir", type=str, default="analysis/figures"
    )
    parser.add_argument(
        "--n_steps", type=int, default=4,
        help="Filter to examples with this many reasoning steps",
    )
    parser.add_argument(
        "--bar_examples",
        type=int,
        nargs="+",
        default=None,
        help="Example indices for stacked bar plots (auto-picks if not specified)",
    )
    parser.add_argument(
        "--n_top", type=int, default=5, help="Number of top entities to show in bars"
    )
    parser.add_argument(
        "--coconut_qonly_json",
        type=str,
        default=None,
        help="Path to question-only Coconut results (enables question-only plot)",
    )
    parser.add_argument(
        "--cot_qonly_json",
        type=str,
        default=None,
        help="Path to question-only CoT results (enables question-only plot)",
    )
    parser.add_argument(
        "--filter_n_steps", type=int, default=None,
        help="Only include examples with exactly this many steps in stepwise plots",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.coconut_json) as f:
        coconut_data = json.load(f)
    with open(args.cot_json) as f:
        cot_data = json.load(f)

    # Plot 1: Entropy
    plot_entropy(
        coconut_data,
        cot_data,
        args.n_steps,
        os.path.join(args.output_dir, f"entropy_{args.n_steps}step.png"),
    )

    # Plot 2: Stacked bars for cherry-picked examples
    if args.bar_examples is None:
        # Auto-pick: first 2 examples with the right step count
        candidates = [
            ex["idx"]
            for ex in coconut_data["examples"]
            if len(ex["steps"]) == args.n_steps
        ]
        bar_examples = candidates[:2]
    else:
        bar_examples = args.bar_examples

    for ex_idx in bar_examples:
        plot_stacked_bars(
            coconut_data,
            cot_data,
            ex_idx,
            args.n_top,
            os.path.join(args.output_dir, f"bars_example_{ex_idx}.png"),
        )

    # Plot 3: Step-aware neighbor belief (normalized among current-node neighbors)
    fns = f"_{args.filter_n_steps}step" if args.filter_n_steps else ""
    plot_stepwise_neighbor_norm(
        coconut_data,
        cot_data,
        os.path.join(args.output_dir, f"stepwise_neighbor_norm{fns}.png"),
        filter_n_steps=args.filter_n_steps,
    )

    # Plot 4: Step-aware neighbor belief (normalized among all entities)
    plot_stepwise_all_entity_norm(
        coconut_data,
        cot_data,
        os.path.join(args.output_dir, f"stepwise_all_entity_norm{fns}.png"),
        filter_n_steps=args.filter_n_steps,
    )

    # Plot 5: Question-only belief (if data provided)
    if args.coconut_qonly_json and args.cot_qonly_json:
        with open(args.coconut_qonly_json) as f:
            coconut_qonly = json.load(f)
        with open(args.cot_qonly_json) as f:
            cot_qonly = json.load(f)
        plot_question_only_summary(
            coconut_qonly,
            cot_qonly,
            os.path.join(args.output_dir, "question_only_belief.png"),
        )



if __name__ == "__main__":
    main()
