"""Plot per-group gradient norms across training stages."""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

# Load data
data = json.load(open(
    "/network/scratch/m/michael.rizvi-martel/coconut_checkpoints/"
    "prosqa-coconut-gradnorm-v2/grad_norms_merged.json"
))

groups = ["wte", "attn", "mlp", "ln", "wpe"]
labels = {
    "wte": "Embeddings\n(wte + lm_head)",
    "attn": "Attention",
    "mlp": "MLP",
    "ln": "LayerNorm",
    "wpe": "Position\nEmbeddings",
}

# Collect per-epoch means
by_epoch = defaultdict(lambda: defaultdict(list))
for r in data:
    for g in groups:
        by_epoch[r["epoch"]][g].append(r[g])

epochs = sorted(by_epoch.keys())
stage_for_epoch = {e: (e - 1) // 5 for e in epochs}  # epochs_per_stage=5, 1-indexed

# Build arrays
epoch_arr = np.array(epochs)
means = {g: np.array([np.mean(by_epoch[e][g]) for e in epochs]) for g in groups}

# Plot
sns.set_style("white")
palette = sns.color_palette("colorblind", 5)
colors = {g: palette[i] for i, g in enumerate(groups)}

fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=False)

for ax, g in zip(axes, groups):
    ax.plot(epoch_arr, means[g], color=colors[g], linewidth=1.8)
    ax.set_title(labels[g], fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_box_aspect(1)
    ax.tick_params(labelsize=9)

    # Alternating stage bands
    for s in range(0, 11):
        start = s * 5 + 1
        end = (s + 1) * 5 + 1
        if s % 2 == 1:
            ax.axvspan(start, end, alpha=0.10, color="slategray", zorder=0)


axes[0].set_ylabel("Gradient L2 Norm", fontsize=11)

fig.suptitle(
    "Per-Group Gradient Norms During Coconut Fine-Tuning (ProsQA, GPT-2)",
    fontsize=13, fontweight="bold", y=1.02,
)

plt.tight_layout()
out = "/home/mila/m/michael.rizvi-martel/repos/coconut/analysis/figures/grad_norms_by_group.pdf"
fig.savefig(out, bbox_inches="tight", dpi=150)
fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
print(f"Saved to {out}")
