#!/bin/bash
#SBATCH --job-name=logit-lens-quick
#SBATCH --output=logs/logit_lens_quick_%j.out
#SBATCH --error=logs/logit_lens_quick_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --partition=unkillable

cd /home/mila/m/michael.rizvi-martel/repos/coconut
source .venv/bin/activate

CKPT=/network/scratch/m/michael.rizvi-martel/coconut_checkpoints/prosqa-coconut/checkpoint_50

# Run logit lens on 20 examples, top 10 tokens, 6 latent positions
python analysis/run_logit_lens.py \
    --model_type coconut \
    --checkpoint $CKPT \
    --data_path data/prosqa_valid.json \
    --n_examples 20 \
    --n_latent 6 \
    --output_name logit_lens_coconut_quick

echo "=== Done. Printing top tokens at last layer for first 5 examples ==="
python3 - <<'PYEOF'
import json

data = json.load(open("analysis/results/logit_lens_coconut_quick.json"))
n_layers = data["n_layers"]

for ex in data["examples"][:5]:
    print(f"\n{'='*60}")
    print(f"Example {ex['idx']}: {ex['question'][:100]}...")
    print(f"Answer: {ex['answer']}")
    print(f"{'='*60}")
    for pos_idx, pos_label in enumerate(ex["position_labels"]):
        # Last layer top tokens
        top_toks = ex["top_tokens"][pos_idx][n_layers - 1]
        entropy = ex["entropy"][pos_idx][n_layers - 1]
        tok_str = "  ".join(f"{t['token']!r}({t['prob']:.2f})" for t in top_toks[:10])
        print(f"  {pos_label} (H={entropy:.2f}): {tok_str}")
PYEOF

echo "=== Plotting entropy vs thinking step ==="
python3 - <<'PYEOF'
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = json.load(open("analysis/results/logit_lens_coconut_quick.json"))
n_layers = data["n_layers"]

# Collect last-layer entropy per thinking step across all examples
n_positions = max(ex["n_positions"] for ex in data["examples"])
entropies_by_step = [[] for _ in range(n_positions)]

for ex in data["examples"]:
    for pos_idx in range(ex["n_positions"]):
        ent = ex["entropy"][pos_idx][n_layers - 1]  # last layer
        entropies_by_step[pos_idx].append(ent)

steps = np.arange(1, n_positions + 1)
means = np.array([np.mean(e) for e in entropies_by_step])
stds = np.array([np.std(e) for e in entropies_by_step])

sns.set_style("white")
fig, ax = plt.subplots(figsize=(5, 5))
ax.errorbar(steps, means, yerr=stds, fmt="o-", color=sns.color_palette("colorblind")[0],
            capsize=4, linewidth=1.8, markersize=6)
ax.set_xlabel("Latent Thought Step", fontsize=12)
ax.set_ylabel("Entropy (nats, last layer)", fontsize=12)
ax.set_title("Coconut Latent Token Entropy\n(Logit Lens, Final Layer)", fontsize=13, fontweight="bold")
ax.set_xticks(steps)
ax.set_xticklabels([f"t{i}" for i in steps])
ax.tick_params(labelsize=10)
sns.despine()

plt.tight_layout()
out = "analysis/figures/latent_entropy_vs_step.pdf"
fig.savefig(out, bbox_inches="tight", dpi=150)
fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
print(f"Saved to {out}")
PYEOF
