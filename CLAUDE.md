# Coconut — Project Context

## Paper: "The Illusion of Superposition"

This repo is one of three contributing to a paper investigating whether latent/continuous chain-of-thought models genuinely reason or just learn shortcuts. The central claim: fine-tuned LLMs do NOT reason in latent space — they extract answers directly from the question embedding, making latent tokens decorative.

**This repo's role:** Reproduce Meta's Coconut (Hao et al., 2024) on ProsQA and run interpretability analyses showing latent tokens are not load-bearing.

### Sibling repos
- `~/repos/Soft-Thinking/` — Inference-time soft tokens (weighted top-k embeddings) on reasoning LLMs. Shows soft tokens collapse to argmax through transformer layers.
- `~/repos/reasoning-by-superposition/` — GPT-2 trained from scratch on ProsQA. Contrast case: here latent tokens DO matter (accuracy drops 85%→20% without them).

### Cross-repo task tracker
**`~/repos/illusion-of-superposition-tasks.md`** — shared TODO list for all three repos. Read this first to understand what work is pending. Update it when completing tasks.

## Key findings from this repo
- Coconut achieves **96.6% accuracy WITHOUT latent tokens** — the smoking gun
- P(target) = 0.82 before any reasoning (87% of examples already >50% at step 0)
- Belief is flat across latent steps (ΔP = -0.01); CoT belief increases (ΔP = +0.10)
- Original paper's "final_state" probing inflated results via KV cache leakage

## Technical documentation
- **`REPO_SUMMARY.md`** — Full codebase walkthrough: file map, training modes, forward pass, dataset pipeline, config reference, gotchas. Read this before modifying code.
- **`analysis/`** — All interpretability scripts and results. See MEMORY.md for details on each script.

## Checkpoints
Located on `/network/scratch/m/michael.rizvi-martel/coconut_checkpoints/`:
- `prosqa-cot/checkpoint_49`: best CoT (85.3% val acc)
- `prosqa-coconut/checkpoint_50`: best Coconut (99% val acc)
- `gsm-cot/checkpoint_21`: best GSM CoT (44.6% val acc)

## Environment
- Python 3.12, venv in `.venv/`
- PyTorch 2.5.1, Transformers 4.46.2
- All training runs require `torchrun` (even single-GPU)
