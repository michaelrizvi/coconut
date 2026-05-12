#!/usr/bin/env python3
"""Probe P(' True') and P(' False') at each Coconut latent position on ProntoQA.

At each probe position (before_thinking, thought_1..6, after_thinking), reads
logits[TRUE_TOKEN] and logits[FALSE_TOKEN] and computes:
    P_norm = P(' True') / (P(' True') + P(' False'))

Plots P_norm across latent positions, split by ground-truth answer and by
whether the no-latent baseline already got the example right.

Usage (run from repo root):
    python analysis/analyze_true_false_prontoqa.py
"""

import json
import os
import sys

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR   = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, SCRIPT_DIR)

from coconut import Coconut
from transformers import AutoModelForCausalLM
from analysis_utils import setup_tokenizer

# ── Config ────────────────────────────────────────────────────────────────────

CHECKPOINT     = "/network/scratch/m/michael.rizvi-martel/coconut_checkpoints/prontoqa-coconut/checkpoint_50"
DATA_PATH      = "data/prontoqa_test.json"
NO_LATENT_PATH = "analysis/results/prontoqa/no_latent_accuracy.json"
N_EXAMPLES     = 200
N_LATENT       = 6
MODEL_ID       = "openai-community/gpt2"
FIGURES_DIR    = "analysis/figures/prontoqa"
RESULTS_DIR    = "analysis/results/prontoqa"

TRUE_TOKEN_ID  = 6407   # ' True'  (GPT-2 tokenizer)
FALSE_TOKEN_ID = 10352  # ' False'

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Model loading (mirrors analyze_coconut.py) ────────────────────────────────

def load_coconut_model(checkpoint_path, model_id, tokenizer, device):
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.resize_token_embeddings(len(tokenizer))
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id  = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id    = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    embeddings = model.get_input_embeddings()
    target_id  = tokenizer.convert_tokens_to_ids("<<")
    for tid in [latent_id, start_id, end_id]:
        embeddings.weight.data[tid] = embeddings.weight.data[target_id].clone()
    coconut_model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    coconut_model.load_state_dict(state_dict, strict=False)
    coconut_model.to(device)
    coconut_model.eval()
    return coconut_model


# ── Probe loop (adapted from analyze_coconut.py per-pass logic) ───────────────

@torch.no_grad()
def probe_true_false(model, tokenizer, data, n_examples, n_latent, device):
    """Return per-example P_norm at each of n_latent+2 probe positions.

    Probe positions: before_thinking, thought_1..n_latent, after_thinking.
    Uses the same per-pass forward loop as analyze_coconut.py with
    output_hidden_states=True to update latent embeddings between passes.
    """
    base_model = model.base_causallm
    embedding  = base_model.transformer.get_input_embeddings()
    latent_id  = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id   = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id     = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    def p_norm(logits_1d):
        probs = F.softmax(logits_1d.float(), dim=-1)
        p_t = probs[TRUE_TOKEN_ID].item()
        p_f = probs[FALSE_TOKEN_ID].item()
        denom = p_t + p_f
        return p_t / denom if denom > 0 else 0.5

    results = []
    for idx in tqdm(range(min(n_examples, len(data))), desc="Probing True/False"):
        example = data[idx]
        question_tokens  = tokenizer.encode(example["question"] + "\n", add_special_tokens=True)
        input_ids        = question_tokens + [start_id] + [latent_id] * n_latent + [end_id]
        input_ids_tensor = torch.tensor([input_ids], device=device)

        inputs_embeds  = embedding(input_ids_tensor)
        attention_mask = torch.ones_like(input_ids_tensor)
        position_ids   = torch.arange(len(input_ids), device=device).unsqueeze(0)

        first_latent_pos = len(question_tokens) + 1
        latent_positions = [first_latent_pos + i for i in range(n_latent)]

        p_norm_series      = []
        next_compute_range = (0, first_latent_pos)
        kv_cache           = None

        for pass_idx in range(n_latent):
            if kv_cache is None:
                outputs = base_model(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                    attention_mask=attention_mask[:, next_compute_range[0]:next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                    output_hidden_states=True, use_cache=True,
                )
                hs_offset = 0
            else:
                past_kv = [(k[:, :, :next_compute_range[0], :],
                            v[:, :, :next_compute_range[0], :]) for k, v in kv_cache]
                outputs = base_model(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                    attention_mask=attention_mask[:, :next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                    past_key_values=past_kv,
                    output_hidden_states=True, use_cache=True,
                )
                hs_offset = next_compute_range[0]

            kv_cache      = outputs.past_key_values
            probe_rel_pos = next_compute_range[1] - 1 - hs_offset
            p_norm_series.append(p_norm(outputs.logits[0, probe_rel_pos, :]))

            # Update embedding of next latent token with current hidden state
            hs         = outputs.hidden_states[-1]
            token_idx  = latent_positions[pass_idx]
            thought    = hs[0, token_idx - 1 - hs_offset, :]
            tensor_list = [inputs_embeds[0, p, :] for p in range(inputs_embeds.shape[1])]
            tensor_list[token_idx] = thought
            inputs_embeds = torch.stack(tensor_list).unsqueeze(0)

            next_compute_range = (
                next_compute_range[1],
                len(input_ids) if pass_idx + 1 >= n_latent else next_compute_range[1] + 1,
            )

        # Final pass: last latent + end_latent
        past_kv = [(k[:, :, :next_compute_range[0], :],
                    v[:, :, :next_compute_range[0], :]) for k, v in kv_cache]
        outputs = base_model(
            inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
            attention_mask=attention_mask[:, :next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
            past_key_values=past_kv,
            use_cache=True,
        )
        probe_rel_pos = next_compute_range[1] - next_compute_range[0] - 1
        p_norm_series.append(p_norm(outputs.logits[0, probe_rel_pos, :]))

        results.append({
            "idx":    idx,
            "answer": example["answer"],
            "p_norm": p_norm_series,  # length n_latent+2: [before, t1..t6, after]
        })

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_true_false_belief(results, no_latent_results, n_latent):
    """Stacked area plots (stepwise_all_entity_norm style) showing mean
    P(correct) and P(wrong) across latent positions.

    One square plot each for: all examples, easy (no-latent correct), hard.
    """
    no_latent_correct = {r["idx"]: r["correct"] for r in no_latent_results["results"]}

    n_positions = len(results[0]["p_norm"])  # 7: pre, T1..T5, post
    labels = ["pre"] + [f"T{k+1}" for k in range(n_positions - 2)] + ["post"]
    x      = np.arange(n_positions)

    matplotlib.rcParams.update({
        "font.size": 12, "axes.labelsize": 14, "axes.titlesize": 14,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 10,
    })

    # Build per-example correct-answer p_norm (high = confident in correct label)
    def correct_series(subset):
        return np.array([
            r["p_norm"] if r["answer"] == "True" else [1 - v for v in r["p_norm"]]
            for r in subset
        ])

    easy = [r for r in results if no_latent_correct.get(r["idx"], True)]
    hard = [r for r in results if not no_latent_correct.get(r["idx"], True)]
    print(f"Easy (no-latent correct): {len(easy)}, Hard (no-latent wrong): {len(hard)}")

    subsets = [
        (results, "All examples",               f"all_examples"),
        (easy,    f"No-latent correct (n={len(easy)})", "easy"),
        (hard,    f"No-latent wrong (n={len(hard)})",   "hard"),
    ]

    C_CORRECT = "#3a86a8"  # blue — correct label
    C_WRONG   = "#c44e52"  # red  — wrong label

    for subset, title, fname in subsets:
        if not subset:
            continue
        arr     = correct_series(subset)   # (N, n_positions), each value = P(correct)
        mean_c  = arr.mean(0)              # mean P(correct) per position
        mean_w  = 1.0 - mean_c            # mean P(wrong)   per position

        fig, ax = plt.subplots(figsize=(5, 5))

        # Stacked area: correct on bottom, wrong on top
        ax.fill_between(x, 0,      mean_c,           color=C_CORRECT, alpha=0.85,
                        label="Correct label", linewidth=0)
        ax.plot(x, mean_c,                            color=C_CORRECT, linewidth=0.8)
        ax.fill_between(x, mean_c, mean_c + mean_w,  color=C_WRONG,   alpha=0.85,
                        label="Wrong label",   linewidth=0)
        ax.plot(x, mean_c + mean_w,                   color=C_WRONG,   linewidth=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=13)
        ax.set_xlabel("Position", fontsize=14)
        ax.set_ylabel("Normalized P(label)", fontsize=14)
        ax.tick_params(axis="y", labelsize=13)
        ax.set_ylim(0, 1.0)
        ax.set_title(title, fontsize=13)
        ax.grid(True, alpha=0.2, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.15),
                  ncol=2, frameon=False)

        plt.tight_layout()
        out = f"{FIGURES_DIR}/true_false_belief_{fname}.pdf"
        plt.savefig(out, bbox_inches="tight", facecolor="white", dpi=200)
        plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
        print(f"Saved {out}")
        plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    with open(DATA_PATH) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples")

    with open(NO_LATENT_PATH) as f:
        no_latent_results = json.load(f)
    print(f"No-latent results: {no_latent_results['accuracy']:.1%} accuracy "
          f"({no_latent_results['correct']}/{no_latent_results['total']})")

    tokenizer = setup_tokenizer(MODEL_ID)
    print("Loading Coconut model...")
    model = load_coconut_model(CHECKPOINT, MODEL_ID, tokenizer, device)

    results = probe_true_false(model, tokenizer, data, N_EXAMPLES, N_LATENT, device)

    out_path = f"{RESULTS_DIR}/true_false_belief.json"
    with open(out_path, "w") as f:
        json.dump({"n_latent": N_LATENT, "n_examples": len(results), "results": results}, f)
    print(f"Saved raw results to {out_path}")

    plot_true_false_belief(results, no_latent_results, N_LATENT)
    print("Done.")
