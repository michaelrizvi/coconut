"""Shared utilities for latent thought analysis."""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


def setup_tokenizer(model_id):
    """Setup tokenizer with special tokens (same as training)."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    return tokenizer


def tokenize_entities(entities, tokenizer):
    """Tokenize entity names, returning dict of entity -> token_ids.

    Entities appear after spaces in the text, so we prepend a space
    for correct BPE tokenization.
    """
    entity_tokens = {}
    for entity in entities:
        token_ids = tokenizer.encode(" " + entity, add_special_tokens=False)
        entity_tokens[entity] = token_ids
    return entity_tokens


def compute_entity_cosine_sims(hidden_state, entity_token_ids, embed_matrix):
    """Compute cosine similarity between a hidden state and each entity's embedding.

    For multi-token entities, the entity embedding is the mean of token embeddings.

    Args:
        hidden_state: (hidden_dim,) tensor at the probe position (last layer).
        entity_token_ids: dict of entity_name -> list[int].
        embed_matrix: wte weight matrix (vocab_size, hidden_dim).

    Returns:
        dict of entity_name -> float cosine similarity.
    """
    h = hidden_state.float()
    sims = {}
    for entity_name, token_ids in entity_token_ids.items():
        entity_embed = embed_matrix[token_ids].float().mean(dim=0)
        sim = F.cosine_similarity(h.unsqueeze(0), entity_embed.unsqueeze(0)).item()
        sims[entity_name] = sim
    return sims


def compute_entity_probs_at_position(
    base_model, full_kv, position, logits_at_pos, entity_token_ids, device
):
    """Compute P(entity) for all entities at a given position.

    Uses the logits at `position` for P(first_token), then does a short
    autoregressive rollout via KV cache for subsequent entity tokens.

    P(entity) = P(tok1) * P(tok2|tok1) * P(tok3|tok1,tok2) * ...

    Args:
        base_model: GPT-2 CausalLM (not the Coconut wrapper).
        full_kv: full KV cache from a forward pass over the entire input.
        position: index into the sequence to probe.
        logits_at_pos: logits vector at this position (vocab_size,).
        entity_token_ids: dict of entity_name -> list[int].
        device: torch device.

    Returns:
        dict of entity_name -> float probability.
    """
    probs_at_pos = F.softmax(logits_at_pos.float(), dim=-1)

    # Truncate KV cache to position (inclusive) so the next fed token
    # is treated as position+1 by the model.
    truncated_kv = tuple(
        (k[:, :, : position + 1, :], v[:, :, : position + 1, :])
        for k, v in full_kv
    )

    entity_probs = {}
    for entity_name, token_ids in entity_token_ids.items():
        # P(first token) directly from the logits
        prob = probs_at_pos[token_ids[0]].item()

        if len(token_ids) > 1:
            # Autoregressive rollout for remaining tokens.
            # truncated_kv is safe to reuse across entities because
            # torch.cat inside GPT-2 attention creates new tensors.
            current_kv = truncated_kv
            for i in range(1, len(token_ids)):
                prev_token = torch.tensor([[token_ids[i - 1]]], device=device)
                with torch.no_grad():
                    out = base_model(
                        prev_token, past_key_values=current_kv, use_cache=True
                    )
                current_kv = out.past_key_values
                next_probs = F.softmax(out.logits[0, -1, :].float(), dim=-1)
                prob *= next_probs[token_ids[i]].item()

        entity_probs[entity_name] = prob

    return entity_probs
