# Layer-wise Entity Belief Analysis — Implementation Summary

This document summarises the logit-lens entity belief analysis added to this repo so
that the same analysis can be ported to the **reasoning-by-superposition (RBS)** repo
with minimal effort.

---

## What the analysis does

For each thinking token position (latent thought or CoT step-end token), at every
transformer layer, we ask: *what entity does the model's internal representation
predict at this position?*

Concretely, at thinking position `t` and transformer layer `l`:

1. Extract the hidden state `h = hidden_states[l+1][0, pos_t, :]`
2. Apply the **logit lens**: `logits = lm_head(ln_f(h))` — project through the final
   layer norm and unembedding matrix, bypassing all subsequent layers
3. Take `softmax(logits)` and read off `P(first BPE token of each entity name)`
4. Normalize over all entities so probabilities sum to 1
5. Average across examples

The output is a `(n_thinking_positions × n_layers)` matrix of normalized
P(target entity), showing at which layer and which step belief crystallises.

### Why final-state (not per-pass) for Coconut

Coconut's multi-pass loop injects continuous thoughts into the embedding at each
latent position. After all `n_latent` passes, `final_embeds` contains these baked-in
thoughts. Re-running the base model once on `final_embeds` with
`output_hidden_states=True` yields **identical** hidden states to the per-pass
computation (same weights, same causal mask, same input embeddings) — it is simply
cheaper. See `analyze_layerwise_belief.py:analyze_coconut_layerwise` for the
implementation.

### Entity probability approximation

P(entity) is approximated as P(first BPE token of the entity name) from the
logit-lens projected distribution. For the multi-token case the remaining tokens
are not conditioned on — this is standard logit lens practice and sufficient for
entities that are predominantly single tokens (as in ProsQA/ProntoQA names).

---

## Files created

| File | Purpose |
|---|---|
| `analysis/analyze_layerwise_belief.py` | Main analysis script — runs the logit lens for Coconut and CoT, saves JSON |
| `analysis/visualize_layerwise.py` | Plotting — heatmap, layer curves, target vs neg_target |
| `scripts/analyze_layerwise_belief.sh` | SLURM job script (1 GPU, 32 GB, 2 h) |

---

## JSON output schema

`analysis/results/prosqa/layerwise_belief_{coconut,cot}.json`:

```json
{
  "model_type": "coconut",
  "n_layers": 12,
  "n_latent": 6,
  "n_examples": 500,
  "examples": [
    {
      "idx": 0,
      "question": "...",
      "answer": "...",
      "steps": ["...", "..."],
      "entities": {"0": "Fae", "1": "Wren", ...},
      "target": "0",
      "neg_target": "1",
      "position_labels": ["thought_1", "thought_2", ..., "thought_6"],
      "entity_layer_probs": {
        "Fae": [[p_l1, p_l2, ..., p_l12], ...],
        "Wren": [[...], ...]
      }
    }
  ]
}
```

`entity_layer_probs[entity][position_idx][layer_idx]` — raw unnormalized
P(first token). Normalization (dividing by sum over entities) happens in the
plotting code so the granular data is preserved.

---

## Porting to the RBS repo

The logic is model-architecture-agnostic; only the model loading and hidden-state
extraction need to change.

### What to reuse unchanged
- `visualize_layerwise.py` — pure numpy/matplotlib, reads the JSON schema above,
  no model dependencies
- The JSON schema — produce the same format and the visualisation just works
- The normalization and averaging logic

### What to change

**1. Model loading**

The RBS model is a GPT-2 trained from scratch with a symbolic tokenizer (small
vocabulary, ~40 tokens). Replace `load_coconut_model` / `load_cot_model` with
whatever loading code the RBS repo uses. The key requirement is that after loading
you can call:

```python
outputs = model(input_ids_tensor, output_hidden_states=True)
hidden_states = outputs.hidden_states  # tuple: (n_layers+1) x (1, seq_len, hidden_dim)
```

**2. Logit lens components**

`get_gpt2_components` extracts `(ln_f, lm_head)` from a GPT-2 model. If the RBS
model uses a different architecture, find the equivalent final layer norm and
unembedding projection. Then `logit_lens_entity_probs` is unchanged.

**3. Entity tokenization**

`tokenize_entities` from `analysis_utils.py` uses BPE tokenization (prepends a
space for correct subword splitting). With a symbolic tokenizer the entities are
likely single tokens; replace with a direct `tokenizer.convert_tokens_to_ids`
lookup. The rest of `logit_lens_entity_probs` (reads `token_ids[0]`) stays the same.

**4. Probe positions**

For the RBS latent model, the probe positions are the latent token positions —
the same as Coconut. If the RBS model uses a different number of latent tokens or a
different sequence format, update `thought_positions` accordingly.

**5. No Coconut wrapper**

The RBS model does not use the `Coconut` class. There is no multi-pass loop and no
`outputs.inputs_embeds`. The baked-in latent representations are handled differently
— check the RBS training code to understand how latent positions are populated, then
run a single forward with `output_hidden_states=True` on the final input.

---

## Running the analysis

```bash
# Quick sanity check (3 examples, interactive)
python analysis/analyze_layerwise_belief.py \
    --model_type coconut \
    --checkpoint /network/scratch/m/michael.rizvi-martel/coconut_checkpoints/prosqa-coconut/checkpoint_50 \
    --data_path data/prosqa_test.json \
    --n_examples 3 --n_latent 6

# Full run via SLURM
sbatch scripts/analyze_layerwise_belief.sh
```
