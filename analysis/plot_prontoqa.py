#!/usr/bin/env python3
"""Visualize stepwise entity belief analysis for ProntoQA Coconut vs CoT.

Produces:
  1. entropy.pdf     — entropy of normalized entity distribution vs reasoning step
  2. chain_track.pdf — P(correct chain concept) at each step, Coconut vs CoT
  3. prob_mass.pdf   — total entity probability mass across latent positions
  4. bars_N.pdf      — stacked bar chart for selected examples

Usage:
    python analysis/plot_prontoqa.py
"""

import json
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

matplotlib.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

# Colorblind-safe colors (consistent with existing visualize.py)
C_COCONUT = "#2196F3"  # blue
C_COT     = "#FF5722"  # orange-red

RESULTS_DIR = "analysis/results/prontoqa"
FIGURES_DIR = "analysis/figures/prontoqa"
os.makedirs(FIGURES_DIR, exist_ok=True)


# ── Data loading ────────────────────────────────────────────────────────────

def load_data():
    coconut = json.load(open(f"{RESULTS_DIR}/coconut_entity_probs.json"))
    cot     = json.load(open(f"{RESULTS_DIR}/cot_entity_probs.json"))
    valid   = json.load(open("data/prontoqa_valid.json"))
    return coconut, cot, valid


# ── Shared helpers ───────────────────────────────────────────────────────────

def concept_entities(ex):
    """Return entity list excluding the proper-noun root (e.g. 'Alex', 'Sam').

    The root entity is a common English name that GPT-2 inherently assigns high
    probability to, dominating the normalized distribution and drowning out the
    fictional concept names we actually care about.
    """
    root_idx = ex["root"]
    return [e for i, e in enumerate(ex["entities"]) if i != root_idx]


def normalize_entity_probs(entity_probs, entities):
    """Normalize raw entity probs to sum to 1 at each position."""
    n_pos = len(entity_probs[entities[0]])
    normed = {ent: [] for ent in entities}
    for i in range(n_pos):
        total = sum(entity_probs[ent][i] for ent in entities)
        for ent in entities:
            normed[ent].append(entity_probs[ent][i] / total if total > 0 else 0.0)
    return normed


def entropy_bits(probs_dict, entities, pos_idx):
    h = 0.0
    for ent in entities:
        p = probs_dict[ent][pos_idx]
        if p > 0:
            h -= p * np.log2(p)
    return h


def find_is_a_positions(cot_ex):
    """Find the ' a' probe position (first occurrence) within each CoT step.
    ProntoQA steps: 'X is a Y. Ys are Zs.' — ' a' appears once per step (steps 0-4).
    Step 5 is a property step ('X is [property]') with no ' a'.
    Returns list of relative indices into entity_probs.
    """
    cot_start = cot_ex["positions"][0]
    positions = []
    for sb in cot_ex["step_boundaries"]:
        start_rel = sb["start"] - cot_start
        end_rel   = sb["end"]   - cot_start
        labels    = cot_ex["position_labels"][start_rel:end_rel]
        # Find first ' a' in this step
        for local_idx, label in enumerate(labels):
            if label == " a":
                positions.append(start_rel + local_idx)
                break
    return positions  # length = n_steps with ' a' (5 for 5-hop)


# ── Plot 1: Entropy of normalized entity distribution ────────────────────────

def plot_entropy(coconut_data, cot_data, valid_data):
    n_latent = coconut_data.get("n_latent", 6)

    coconut_entropies = []
    cot_entropies     = []

    cot_by_idx = {ex["idx"]: ex for ex in cot_data["examples"]}

    for ex in coconut_data["examples"]:
        idx     = ex["idx"]
        if idx not in cot_by_idx:
            continue

        # Coconut: thoughts 1..n_latent (skip before/after thinking)
        c_ents    = concept_entities(ex)
        c_normed  = normalize_entity_probs(ex["entity_probs"], c_ents)
        c_entropy = [entropy_bits(c_normed, c_ents, i) for i in range(1, n_latent + 1)]
        coconut_entropies.append(c_entropy)

        # CoT: ' a' probe positions (one per concept step)
        cot_ex   = cot_by_idx[idx]
        t_ents   = concept_entities(cot_ex)
        t_normed = normalize_entity_probs(cot_ex["entity_probs"], t_ents)
        is_a_pos = find_is_a_positions(cot_ex)
        if len(is_a_pos) == 0:
            continue
        t_entropy = [entropy_bits(t_normed, t_ents, pos) for pos in is_a_pos]
        cot_entropies.append(t_entropy)

    # Filter to uniform length (5 is_a positions for 5-hop)
    n_chain = 5
    coconut_entropies = np.array(coconut_entropies)
    cot_entropies     = np.array([e for e in cot_entropies if len(e) == n_chain])

    fig, ax = plt.subplots(figsize=(7, 4))

    x_c = np.arange(1, n_latent + 1)
    mean_c, std_c = coconut_entropies.mean(0), coconut_entropies.std(0)
    ax.plot(x_c, mean_c, "o-", color=C_COCONUT, label="Coconut (latent)", lw=2)
    ax.fill_between(x_c, np.maximum(mean_c - std_c, 0), mean_c + std_c,
                    color=C_COCONUT, alpha=0.15)

    n_cot_steps = cot_entropies.shape[1]
    x_t = np.arange(1, n_cot_steps + 1)
    mean_t, std_t = cot_entropies.mean(0), cot_entropies.std(0)
    ax.plot(x_t, mean_t, "s--", color=C_COT, label="CoT (discrete)", lw=2)
    ax.fill_between(x_t, np.maximum(mean_t - std_t, 0), mean_t + std_t,
                    color=C_COT, alpha=0.15)

    ax.set_xlabel("Reasoning Step")
    ax.set_ylabel("Entropy (bits)")
    ax.set_title("Entity Distribution Entropy During Reasoning (ProntoQA)")
    ax.set_xticks(np.arange(1, max(n_latent, n_cot_steps) + 1))
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = f"{FIGURES_DIR}/entropy.pdf"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


# ── Plot 2: Chain concept tracking ──────────────────────────────────────────

def plot_chain_tracking(coconut_data, cot_data, valid_data):
    """P(chain_concepts[k]) normalized among all entities at reasoning step k.

    For Coconut: thought_{k+1} is compared to chain_concepts[k].
    For CoT: ' a' position in step k is compared to chain_concepts[k].
    Both use 5 concept steps (5-hop chain, step 5 is property — excluded).
    """
    n_latent = coconut_data.get("n_latent", 6)
    n_chain  = 5  # ProntoQA 5-hop: 5 concept steps

    # Build lookup for valid_data chain_concepts by (question prefix)
    valid_by_question = {ex["question"][:80]: ex for ex in valid_data}

    cot_by_idx = {ex["idx"]: ex for ex in cot_data["examples"]}

    coconut_chain_probs = []  # (N, n_chain) — normalized P(chain_concepts[k]) at thought_k+1
    cot_chain_probs     = []  # (N, n_chain) — normalized P(chain_concepts[k]) at is_a step k

    for ex in coconut_data["examples"]:
        idx      = ex["idx"]
        if idx not in cot_by_idx:
            continue

        # Get chain_concepts from valid_data by matching question
        q_key = ex["question"][:80]
        if q_key not in valid_by_question:
            continue
        chain_concepts = valid_by_question[q_key]["chain_concepts"]
        if len(chain_concepts) != n_chain:
            continue

        # Filter to chain concepts that exist in entity list
        if not all(c in ex["entities"] for c in chain_concepts):
            continue

        c_ents   = concept_entities(ex)
        c_normed = normalize_entity_probs(ex["entity_probs"], c_ents)
        cot_ex   = cot_by_idx[idx]
        t_ents   = concept_entities(cot_ex)
        t_normed = normalize_entity_probs(cot_ex["entity_probs"], t_ents)
        is_a_pos = find_is_a_positions(cot_ex)
        if len(is_a_pos) < n_chain:
            continue

        # Coconut: thought_{k+1} → chain_concepts[k] (thoughts indexed 1..n_latent)
        c_row = [c_normed[chain_concepts[k]][k + 1] for k in range(n_chain)]
        coconut_chain_probs.append(c_row)

        # CoT: is_a_pos[k] → chain_concepts[k]
        t_row = [t_normed[chain_concepts[k]][is_a_pos[k]] for k in range(n_chain)]
        cot_chain_probs.append(t_row)

    coconut_chain_probs = np.array(coconut_chain_probs)
    cot_chain_probs     = np.array(cot_chain_probs)
    print(f"Chain tracking: {coconut_chain_probs.shape[0]} matched examples")

    fig, ax = plt.subplots(figsize=(7, 4))

    x = np.arange(1, n_chain + 1)
    mean_c, std_c = coconut_chain_probs.mean(0), coconut_chain_probs.std(0)
    ax.plot(x, mean_c, "o-", color=C_COCONUT, label="Coconut (latent)", lw=2)
    ax.fill_between(x, np.maximum(mean_c - std_c, 0), mean_c + std_c,
                    color=C_COCONUT, alpha=0.15)

    mean_t, std_t = cot_chain_probs.mean(0), cot_chain_probs.std(0)
    ax.plot(x, mean_t, "s--", color=C_COT, label="CoT (discrete)", lw=2)
    ax.fill_between(x, np.maximum(mean_t - std_t, 0), mean_t + std_t,
                    color=C_COT, alpha=0.15)

    ax.set_xlabel("Reasoning Step $k$")
    ax.set_ylabel("Normalized P(chain concept $k$)")
    ax.set_title("Chain Concept Tracking During Reasoning (ProntoQA)")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = f"{FIGURES_DIR}/chain_track.pdf"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


# ── Plot 3: Entity probability mass across latent positions ──────────────────

def plot_prob_mass(coconut_data):
    """Total entity probability mass at each latent position.

    Shows that latent tokens activate concept representations significantly
    compared to before/after thinking (where model is in True/False mode).
    """
    n_latent = coconut_data.get("n_latent", 6)

    # Positions: before_thinking=0, thought_1..n_latent=1..n_latent, after_thinking=n_latent+1
    all_masses = []  # (N, n_latent+2)
    for ex in coconut_data["examples"]:
        c_ents = concept_entities(ex)
        n_pos = n_latent + 2
        row = []
        for i in range(n_pos):
            if i < len(ex["entity_probs"][c_ents[0]]):
                row.append(sum(ex["entity_probs"][ent][i] for ent in c_ents))
            else:
                row.append(0.0)
        all_masses.append(row)

    all_masses = np.array(all_masses)
    # Log-scale: add small epsilon
    log_masses = np.log10(all_masses + 1e-20)

    labels = ["pre"] + [f"T{k+1}" for k in range(n_latent)] + ["post"]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, 4))
    mean_log = log_masses.mean(0)
    std_log  = log_masses.std(0)
    ax.bar(x, mean_log, color=C_COCONUT, alpha=0.8, width=0.6, label="Mean log₁₀(entity mass)")
    ax.errorbar(x, mean_log, yerr=std_log, fmt="none", color="gray", capsize=3, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Position")
    ax.set_ylabel("log₁₀(Σ P(entity))")
    ax.set_title("Entity Probability Mass Across Latent Positions (Coconut, ProntoQA)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = f"{FIGURES_DIR}/prob_mass.pdf"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


# ── Plot 4: Stacked bar charts ───────────────────────────────────────────────

def plot_stacked_bars(coconut_data, cot_data, valid_data, example_idxs=(0, 1, 2)):
    """Stacked bar chart for selected examples showing normalized entity probs."""
    n_latent    = coconut_data.get("n_latent", 6)
    cot_by_idx  = {ex["idx"]: ex for ex in cot_data["examples"]}
    valid_by_q  = {ex["question"][:80]: ex for ex in valid_data}
    palette     = sns.color_palette("Set2", 8)

    for target_idx in example_idxs:
        ex_c = next((e for e in coconut_data["examples"] if e["idx"] == target_idx), None)
        if ex_c is None or target_idx not in cot_by_idx:
            continue
        ex_t = cot_by_idx[target_idx]

        entities    = ex_c["entities"]
        target_ent  = entities[ex_c["target"]]
        neg_target  = entities[ex_c["neg_target"]] if ex_c["neg_target"] >= 0 else None

        # Get chain_concepts
        q_key          = ex_c["question"][:80]
        chain_concepts = valid_by_q.get(q_key, {}).get("chain_concepts", [])

        c_ents   = concept_entities(ex_c)
        t_ents   = concept_entities(ex_t)
        c_normed = normalize_entity_probs(ex_c["entity_probs"], c_ents)
        t_normed = normalize_entity_probs(ex_t["entity_probs"], t_ents)
        is_a_pos = find_is_a_positions(ex_t)

        # Select entities to display: chain + target + neg_target, then "other"
        # (concept_entities already excludes the proper-noun root)
        display_ents = list(dict.fromkeys(chain_concepts + [target_ent] +
                                          ([neg_target] if neg_target else [])))
        other_ents   = [e for e in c_ents if e not in display_ents]

        # Assign colors
        ent_colors = {ent: palette[i % len(palette)] for i, ent in enumerate(display_ents)}
        ent_colors["other"] = (0.8, 0.8, 0.8)

        n_steps = len(is_a_pos)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

        # --- Coconut panel ---
        x_c    = np.arange(n_latent)
        bottom = np.zeros(n_latent)
        for ent in display_ents:
            vals  = [c_normed[ent][i + 1] for i in range(n_latent)]
            label = ent + (" ★" if ent == target_ent else "")
            ax1.bar(x_c, vals, bottom=bottom, label=label, color=ent_colors[ent], width=0.7)
            bottom += np.array(vals)
        other_vals = np.zeros(n_latent)
        for ent in other_ents:
            other_vals += [c_normed[ent][i + 1] for i in range(n_latent)]
        ax1.bar(x_c, other_vals, bottom=bottom, label="other", color=ent_colors["other"], width=0.7)
        ax1.set_xticks(x_c)
        ax1.set_xticklabels([f"T{i+1}" for i in range(n_latent)])
        ax1.set_xlabel("Latent Thought Step")
        ax1.set_ylabel("Normalized P(entity)")
        ax1.set_title("Coconut (Latent)")
        ax1.set_ylim(0, 1.05)
        ax1.legend(fontsize=8, loc="upper right")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # --- CoT panel ---
        x_t    = np.arange(n_steps)
        bottom = np.zeros(n_steps)
        for ent in display_ents:
            vals  = [t_normed[ent][pos] for pos in is_a_pos]
            label = ent + (" ★" if ent == target_ent else "")
            ax2.bar(x_t, vals, bottom=bottom, label=label, color=ent_colors[ent], width=0.7)
            bottom += np.array(vals)
        other_vals = np.zeros(n_steps)
        for ent in other_ents:
            other_vals += [t_normed[ent][pos] for pos in is_a_pos]
        ax2.bar(x_t, other_vals, bottom=bottom, label="other", color=ent_colors["other"], width=0.7)
        ax2.set_xticks(x_t)
        ax2.set_xticklabels([f"S{i+1}" for i in range(n_steps)])
        ax2.set_xlabel("Reasoning Step")
        ax2.set_ylabel("Normalized P(entity)")
        ax2.set_title("CoT (Discrete)")
        ax2.set_ylim(0, 1.05)
        ax2.legend(fontsize=8, loc="upper right")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        chain_str = " → ".join(chain_concepts) if chain_concepts else "?"
        root_name = entities[ex_c["root"]]
        fig.suptitle(
            f"Example {target_idx} | {root_name} → {target_ent} | "
            f"Chain: {chain_str}",
            fontsize=10,
        )
        plt.tight_layout()
        out = f"{FIGURES_DIR}/bars_{target_idx}.pdf"
        plt.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
        plt.close()


# ── Plot 5: Stepwise all-entity-norm stacked area ────────────────────────────

def _get_outgoing_neighbors(ex, node_name):
    """Outgoing neighbors of node_name via directed graph edges."""
    entities = ex["entities"]
    name_to_idx = {name: i for i, name in enumerate(entities)}
    node_idx = name_to_idx.get(node_name)
    if node_idx is None:
        return set()
    return {entities[dst] for src, dst in ex["edges"] if src == node_idx}


def _compute_all_entity_record(normed, correct_next, neighbors, target, pos_idx):
    """Return {correct, wrong, target, other} normalized over all entities."""
    if correct_next == target:
        p_correct = 0.0
        p_target  = normed[target][pos_idx]
        p_wrong   = sum(normed[e][pos_idx] for e in neighbors if e != correct_next)
    else:
        p_correct = normed[correct_next][pos_idx]
        p_wrong   = sum(normed[e][pos_idx] for e in neighbors if e != correct_next)
        accounted = set(neighbors) | {correct_next}
        p_target  = normed[target][pos_idx] if target not in accounted else 0.0
    p_other = max(1.0 - p_correct - p_wrong - p_target, 0.0)
    return {"correct": p_correct, "wrong": p_wrong, "target": p_target, "other": p_other}


def _collect_prontoqa_stepwise(coconut_data, cot_data, valid_data):
    """Per-step records using chain_concepts for ProntoQA.

    At step k (k = 0 .. n_chain-1):
      - correct_next = chain_concepts[k]
      - current_node = chain_concepts[k-1] if k>0 else entities[root]
      - neighbors    = outgoing graph edges from current_node
      - Coconut probe: entity_probs position k+1  (thought_{k+1})
      - CoT probe:     is_a_pos[k]

    Returns (coconut_steps, cot_steps) — each a list of n_chain lists of records.
    """
    n_chain = 5
    valid_by_q  = {ex["question"][:80]: ex for ex in valid_data}
    cot_by_idx  = {ex["idx"]: ex for ex in cot_data["examples"]}

    coconut_steps = [[] for _ in range(n_chain)]
    cot_steps     = [[] for _ in range(n_chain)]

    for ex in coconut_data["examples"]:
        idx = ex["idx"]
        if idx not in cot_by_idx:
            continue
        q_key = ex["question"][:80]
        if q_key not in valid_by_q:
            continue
        chain_concepts = valid_by_q[q_key].get("chain_concepts", [])
        if len(chain_concepts) != n_chain:
            continue
        entities = ex["entities"]
        if not all(c in entities for c in chain_concepts):
            continue

        cot_ex   = cot_by_idx[idx]
        is_a_pos = find_is_a_positions(cot_ex)
        if len(is_a_pos) < n_chain:
            continue

        target   = entities[ex["target"]]
        c_ents   = concept_entities(ex)
        t_ents   = concept_entities(cot_ex)
        c_normed = normalize_entity_probs(ex["entity_probs"], c_ents)
        t_normed = normalize_entity_probs(cot_ex["entity_probs"], t_ents)
        n_coc_pos = len(c_normed[c_ents[0]])

        for k in range(n_chain):
            correct_next = chain_concepts[k]
            current_node = chain_concepts[k - 1] if k > 0 else entities[ex["root"]]
            neighbors    = _get_outgoing_neighbors(ex, current_node)

            # Coconut: thought_{k+1} is at position index k+1
            coc_pos = k + 1
            if coc_pos < n_coc_pos:
                rec = _compute_all_entity_record(c_normed, correct_next, neighbors,
                                                 target, coc_pos)
                coconut_steps[k].append(rec)

            # CoT: first ' a' token in step k
            rec = _compute_all_entity_record(t_normed, correct_next, neighbors,
                                             target, is_a_pos[k])
            cot_steps[k].append(rec)

    return coconut_steps, cot_steps


def plot_stepwise_all_entity_norm(coconut_data, cot_data, valid_data):
    """Stacked area: correct next / wrong neighbors / target / other.

    One square plot per model (Coconut and CoT), saved separately.
    Matches the style of analysis/figures/stepwise_all_entity_norm_{model}.pdf.
    """
    coconut_steps, cot_steps = _collect_prontoqa_stepwise(
        coconut_data, cot_data, valid_data
    )
    n_chain = 5

    categories = ["correct", "wrong", "target", "other"]
    cat_labels  = ["Correct next", "Wrong neighbors", "Target (final)", "Other"]
    cat_colors  = ["#3a86a8", "#e8a838", "#c44e52", "#bdbdbd"]

    for model_label, steps in [("coconut", coconut_steps), ("cot", cot_steps)]:
        fig, ax = plt.subplots(figsize=(5, 5))
        x = np.arange(n_chain)

        means = {}
        for cat in categories:
            means[cat] = np.array([
                np.mean([r[cat] for r in step_recs]) if step_recs else 0.0
                for step_recs in steps
            ])
        ns = [len(s) for s in steps]

        bottom = np.zeros(n_chain)
        for cat, label, color in zip(categories, cat_labels, cat_colors):
            ax.fill_between(x, bottom, bottom + means[cat],
                            color=color, alpha=0.85, label=label, linewidth=0)
            ax.plot(x, bottom + means[cat], color=color, linewidth=0.8)
            bottom += means[cat]

        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in range(n_chain)], fontsize=13)
        ax.set_xlabel("Reasoning Step", fontsize=14)
        ax.set_ylabel("Normalized P(entity)", fontsize=14)
        ax.tick_params(axis="y", labelsize=13)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.2, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.15),
                  ncol=2, frameon=False, columnspacing=1.0)

        print(f"  n per step ({model_label}): {ns}")

        plt.tight_layout()
        out = f"{FIGURES_DIR}/stepwise_all_entity_norm_{model_label}.pdf"
        plt.savefig(out, bbox_inches="tight", facecolor="white", dpi=200)
        print(f"Saved {out}")
        plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    coconut_data, cot_data, valid_data = load_data()
    print(f"  Coconut: {len(coconut_data['examples'])} examples")
    print(f"  CoT:     {len(cot_data['examples'])} examples")
    print(f"  Valid:   {len(valid_data)} examples")

    print("\n--- Plot 1: Entropy ---")
    plot_entropy(coconut_data, cot_data, valid_data)

    print("\n--- Plot 2: Chain tracking ---")
    plot_chain_tracking(coconut_data, cot_data, valid_data)

    print("\n--- Plot 3: Entity probability mass ---")
    plot_prob_mass(coconut_data)

    print("\n--- Plot 4: Stacked bars ---")
    plot_stacked_bars(coconut_data, cot_data, valid_data, example_idxs=[0, 1, 2, 5, 10])

    print("\n--- Plot 5: Stepwise all-entity-norm ---")
    plot_stepwise_all_entity_norm(coconut_data, cot_data, valid_data)

    print("\nDone. Figures saved to", FIGURES_DIR)
