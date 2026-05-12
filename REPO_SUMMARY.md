# Coconut Repository Summary

> Official implementation of "Training Large Language Models to Reason in a Continuous Latent Space" (Hao et al., 2024). Paper: https://arxiv.org/abs/2412.06769

## Core Idea

Instead of chain-of-thought (CoT) reasoning in discrete token space, Coconut feeds the model's **last hidden states** back as input embeddings for subsequent "thought" steps — reasoning happens in **continuous latent space** rather than being decoded into text tokens. The model learns to replace explicit CoT steps with these latent "continuous thoughts."

## File Map

```
coconut/
├── run.py                  # Entry point. Distributed training loop + eval.
├── coconut.py              # Coconut model wrapper (forward pass with latent recurrence, generate)
├── dataset.py              # Dataset construction: tokenization, staged CoT→latent conversion, custom collator
├── utils.py                # Config class, set_seed
├── args/                   # YAML config files for each experiment
│   ├── gsm_cot.yaml              # GSM8K CoT baseline (stage 0 pretraining)
│   ├── gsm_coconut.yaml          # GSM8K Coconut training (stages 1+)
│   ├── gsm_coconut_eval.yaml     # GSM8K Coconut eval-only
│   ├── prontoqa_coconut.yaml     # ProntoQA Coconut training
│   ├── prontoqa_coconut_eval.yaml
│   ├── prosqa_coconut.yaml       # ProsQA Coconut training
│   └── prosqa_coconut_eval.yaml
├── preprocessing/
│   ├── gsm_icot.py         # Converts GSM8K iCoT text files → JSON
│   └── prontoqa.py         # Converts ProntoQA raw JSON → train/valid/test splits
├── data/                   # Data directory (ProsQA included; GSM/ProntoQA need preprocessing)
└── requirements.txt
```

## Training Modes (Mutually Exclusive Flags)

Controlled by four boolean config flags: `coconut`, `cot`, `no_thoughts`, `no_cot`.

| Mode | Flags | What it does |
|------|-------|-------------|
| **CoT** | `cot=True` | Standard chain-of-thought SFT. All reasoning steps are explicit text tokens. No special latent tokens added. |
| **Coconut** | `coconut=True` | Latent reasoning. CoT steps progressively replaced by `<latent>` tokens whose embeddings are filled by last hidden states (the core method). |
| **No-thoughts** | `no_thoughts=True` | Coconut architecture but with `c_thought=0` — no latent tokens inserted. Ablation baseline. |
| **No-CoT** | `no_cot=True` | No reasoning at all. All CoT steps are stripped; model directly predicts the answer. Ablation baseline. |

## Special Tokens

Three tokens are added to the tokenizer (initialized from the embedding of `<<`):

| Token | Purpose |
|-------|---------|
| `<\|start-latent\|>` | Marks the boundary before latent tokens begin |
| `<\|latent\|>` | Placeholder token whose embedding gets **replaced** by the last hidden state from the previous forward pass |
| `<\|end-latent\|>` | Marks the boundary after latent tokens end |

These are only used in Coconut mode. CoT/no-cot/no-thoughts modes set `no_special_marker=True` and skip them.

## Staged Training Schedule

This is the key training curriculum. Coconut doesn't go from full CoT to full latent in one shot — it progressively replaces CoT steps with latent tokens over multiple **stages**.

**Stage calculation** (`run.py:244`):
```
scheduled_stage = epoch // epochs_per_stage
```

At stage `k`:
- The first `k` CoT reasoning steps are **removed** from the text
- `k * c_thought` latent tokens (`<|latent|>`) are inserted in their place
- The remaining CoT steps (after the first `k`) stay as text
- The model is trained to predict the remaining text steps + final answer

**Example progression** (ProntoQA, `epochs_per_stage=5`, `c_thought=1`):
- Epochs 0-4 (stage 0): Full CoT, no latent tokens (equivalent to CoT SFT)
- Epochs 5-9 (stage 1): 1 latent token replaces step 1, steps 2-5 remain as text
- Epochs 10-14 (stage 2): 2 latent tokens replace steps 1-2, steps 3-5 as text
- ...
- Epochs 25-29 (stage 5): 5 latent tokens replace all steps, only answer remains as text

**Key parameters:**
- `c_thought`: Number of latent tokens **per** replaced reasoning step (GSM uses 2, ProntoQA/ProsQA use 1)
- `max_latent_stage`: Cap on how many stages to go through
- `pad_latent_to_max`: If True, once past `max_latent_stage`, pad latent count to `max_latent_stage * c_thought` even if the sample has fewer steps
- `uniform_prob`: Probability of randomly sampling a different stage for a given training example (data augmentation, 0.0 in standard runs, 0.3 in analysis)

## Coconut Forward Pass (`coconut.py`)

The `Coconut` class wraps a HuggingFace causal LM. Its `forward()` implements the multi-pass latent recurrence:

1. **Find latent positions**: Locate all `<|latent|>` token positions in the batch
2. **Segment the sequence** at the earliest latent token position
3. **Iterative forward passes** (one per latent token):
   - Run the base LM on the current segment, collecting KV cache
   - Extract the **last hidden state** at position `latent_pos - 1` (the token just before the latent)
   - **Replace** the embedding at `latent_pos` with this hidden state (the "continuous thought")
   - Advance the compute window to the next segment
4. **Final forward pass**: Process remaining tokens (post-latent text + answer) using accumulated KV cache
5. **Compute loss**: Concatenate all logits, apply shifted cross-entropy against labels

The KV cache is sliced and reused between passes to avoid redundant computation. The `hidden_states_offset` variable tracks the offset introduced by KV cache reuse.

**Important implementation detail**: To avoid in-place operations (which break autograd), `inputs_embeds` is decomposed into a list-of-lists of 1D tensors, modified, then reassembled via `torch.stack` (`coconut.py:134-158`).

## Coconut Generation (`coconut.py:201`)

Generation is **batch_size=1 only**. Flow:

1. Run `self.forward()` on the input (including latent tokens) to get the final hidden state
2. Greedy-decode: `argmax` on last logit → embed → append → repeat
3. In FSDP mode, pads with dummy forward passes (`synced_gpus=True`) to keep all devices synchronized

Note: Generation does **not** use HuggingFace's `generate()` for the autoregressive part — it manually loops, concatenating embeddings (not token IDs) because the latent reasoning produced continuous embeddings that can't be represented as discrete tokens.

## Dataset Pipeline (`dataset.py`)

### `get_dataset(path, tokenizer)`
Loads raw JSON, tokenizes each sample into:
- `question_tokenized`: Token IDs for the question
- `steps_tokenized`: List of token ID lists, one per reasoning step
- `answer_tokenized`: Token IDs for `"### " + answer + eos`

### `get_cot_latent_dataset(scheduled_stage, ...)` — Training data
Constructs the training sequences for a given stage:
- Prepends question tokens
- Inserts `[start_id] + [latent_id] * (n_latent_tokens) + [end_id]`
- Appends remaining (non-replaced) CoT steps + answer tokens
- **Labels**: `-100` (ignored) for question + latent tokens; actual token IDs for the remaining CoT + answer portion

### `get_question_latent_dataset(scheduled_stage, ...)` — Eval/generation data
Same as above but **without** CoT steps or answer — just question + latent tokens. Used as the prompt for generation at eval time.

### `MyCollator` — Custom padding
Pads batches so that **latent token positions are aligned across the batch** (left-pads with pad tokens to align the earliest `<|latent|>` position). This maximizes KV cache reuse in the multi-pass forward. Position IDs are explicitly constructed to handle the left-padding correctly.

## Training Loop (`run.py`)

### Distributed Setup
- Uses `torchrun` with NCCL backend
- FSDP for training (wraps `LlamaDecoderLayer`; GPT2 is small enough that FSDP acts like DDP)
- DDP for eval-only mode (avoids FSDP bugs during generation)

### Per-Epoch Flow
1. Compute `scheduled_stage` from epoch number
2. Build train/val datasets for this stage via `get_cot_latent_dataset` / `get_question_latent_dataset`
3. **Train**: Standard loop with gradient accumulation, AdamW optimizer
4. **Checkpoint**: Save model state dict (either every epoch or only on accuracy improvement)
5. **Eval loss**: Forward pass on validation set, aggregate loss across ranks
6. **Eval generation accuracy**: Generate answers with `model.generate()`, extract answer after `#`, compare exact match with ground truth

### Checkpoint Resumption
- Auto-detects previous checkpoints in `save_dir` and resumes from the latest
- `resume` config param can skip early epochs (e.g., skip stage 0 CoT when loading a CoT-pretrained model)

### Eval Answer Extraction
- Output is decoded, split on `#`, last segment is the answer
- CoT portion is extracted as everything between line 1 and `#`
- Both answer accuracy and CoT exact match are logged

## GSM8K Two-Phase Training

GSM8K requires a **two-phase** approach:
1. **Phase 1** (`gsm_cot.yaml`): Train GPT-2 on full CoT data (`coconut=False, cot=True`). Get a checkpoint at ~40% val accuracy.
2. **Phase 2** (`gsm_coconut.yaml`): Load that checkpoint into Coconut (`coconut=True, load_model_path=<cot_ckpt>, resume=3`). `resume=3` skips the first 3 epochs (stage 0) since the model is already CoT-trained.

ProntoQA and ProsQA train from scratch in a single phase (stage 0 serves as the CoT warmup).

## Config Parameters Quick Reference

| Parameter | Type | Description |
|-----------|------|-------------|
| `coconut` | bool | Enable Coconut latent reasoning |
| `cot` | bool | Standard CoT training (no latent tokens) |
| `no_thoughts` | bool | Ablation: Coconut structure but 0 latent tokens |
| `no_cot` | bool | Ablation: No reasoning steps at all |
| `c_thought` | int | Latent tokens per reasoning step (1 or 2) |
| `epochs_per_stage` | int | Epochs before advancing to next stage |
| `max_latent_stage` | int | Maximum number of CoT steps to replace |
| `pad_latent_to_max` | bool | Pad latent count when sample has fewer steps than stage |
| `uniform_prob` | float | Probability of random stage sampling per example |
| `reset_optimizer` | bool | Reinitialize optimizer each epoch (True for Coconut) |
| `save_only_improve` | bool | Only save on val accuracy improvement |
| `only_eval` | bool | Skip training, run eval only |
| `resume` | int | Skip first N epochs |
| `model_id` | str | HuggingFace model ID (e.g., `openai-community/gpt2`) |
| `load_model_path` | str | Path to checkpoint ("None" for no loading) |
| `bf16` | bool | Use bfloat16 |
| `debug` | bool | Small data subset, no W&B, no saving |

## Running

```bash
# All runs require torchrun (distributed)
torchrun --nnodes 1 --nproc_per_node N_GPUS run.py args/<config>.yaml
```

## Logging

Uses Weights & Biases. Logs:
- `train/loss`, `train/epoch`, `train/step`
- `eval/loss`, `eval/acc`, `eval/cot_em`
- `data_table`: First batch token-level input/label pairs (for debugging)

## Key Gotchas for Future Agents

1. **All runs are distributed** — even single-GPU needs `torchrun --nproc_per_node 1`. Raw `python run.py` will crash.
2. **`load_model_path` uses string `"None"`**, not Python `None`. Check against the string.
3. **GSM8K needs two training phases** — you must first train CoT, then load that checkpoint for Coconut.
4. **Generation is batch_size=1 only** in Coconut mode (`coconut.py:213`).
5. **Optimizer is reset every epoch** when `reset_optimizer=True` (default for Coconut configs). This is intentional for the staged curriculum.
6. **FSDP wrapping**: Only `LlamaDecoderLayer` is wrapped. GPT-2 effectively runs as DDP under FSDP.
7. **The collator left-pads** to align latent positions across the batch. This is critical for KV cache reuse in the multi-pass forward.
8. **Labels are `-100`** (ignored by CrossEntropyLoss) for question + latent tokens. The model only trains on predicting remaining CoT steps + answer.
9. **New token embeddings** are initialized from the `<<` token embedding, not random — this stabilizes early training.
10. **`synced_gpus`** in generation: Coconut pads dummy forward passes to keep FSDP ranks synchronized. This is disabled in `only_eval` mode (which uses DDP).
