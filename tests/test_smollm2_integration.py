"""
Integration tests for Coconut finetuning codebase with SmolLM2-135M.

Run from the coconut/ directory:
    python tests/test_smollm2_integration.py

Requirements:
    - CPU-only (no CUDA/NCCL required)
    - HuggingFaceTB/SmolLM2-135M must be downloadable (or already cached)
    - ProsQA data at data/prosqa_train.json
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import itertools

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"

RESULTS = {"passed": [], "failed": []}

SEPARATOR = "-" * 70


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_model_and_tokenizer(add_special_tokens=True, resize=True, init_embeddings=True):
    """
    Load a fresh SmolLM2-135M model and tokenizer.

    Args:
        add_special_tokens: whether to add the 3 Coconut special tokens.
        resize: whether to resize token embeddings after adding tokens.
        init_embeddings: whether to initialize new token embeddings from '<<'.

    Returns:
        (model, tokenizer, latent_id, start_id, end_id)
        Token IDs are None when add_special_tokens=False.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

    latent_id = start_id = end_id = None

    if add_special_tokens:
        tokenizer.add_tokens("<|start-latent|>")
        tokenizer.add_tokens("<|end-latent|>")
        tokenizer.add_tokens("<|latent|>")
        start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
        end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
        latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")

    if add_special_tokens and resize:
        model.resize_token_embeddings(len(tokenizer))

    if add_special_tokens and resize and init_embeddings:
        embeddings = model.get_input_embeddings()
        target_id = tokenizer.convert_tokens_to_ids("<<")
        for tid in [latent_id, start_id, end_id]:
            embeddings.weight.data[tid] = embeddings.weight.data[target_id].clone()
            model.lm_head.weight.data[tid] = model.lm_head.weight.data[target_id].clone()

    return model, tokenizer, latent_id, start_id, end_id


def run_test(name, fn):
    """Run a single test, recording pass/fail."""
    print(SEPARATOR)
    print(f"Running: {name}")
    try:
        fn()
        RESULTS["passed"].append(name)
        print(f"[PASS] {name}")
    except Exception as e:
        RESULTS["failed"].append(name)
        print(f"[FAIL] {name}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Test 1: decoder layer class lookup
# ---------------------------------------------------------------------------

def test_decoder_layer_cls():
    """get_decoder_layer_cls returns LlamaDecoderLayer for SmolLM2."""
    from run import get_decoder_layer_cls
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    model, tokenizer, *_ = setup_model_and_tokenizer(
        add_special_tokens=False, resize=False, init_embeddings=False
    )

    cls = get_decoder_layer_cls(model)
    assert cls is LlamaDecoderLayer, (
        f"Expected LlamaDecoderLayer, got {cls}"
    )

    # Verify model actually contains LlamaDecoderLayer instances
    llama_layers = [m for m in model.modules() if isinstance(m, LlamaDecoderLayer)]
    assert len(llama_layers) > 0, "No LlamaDecoderLayer found in model"
    assert len(llama_layers) == 30, f"Expected 30 layers, found {len(llama_layers)}"


# ---------------------------------------------------------------------------
# Test 2: special token setup
# ---------------------------------------------------------------------------

def test_special_token_setup():
    """Adding 3 special tokens gives unique, non-unk IDs; embeddings init from <<."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    unk_id = tokenizer.unk_token_id  # 0 for SmolLM2

    lt_id = tokenizer.convert_tokens_to_ids("<<")
    assert lt_id != unk_id, f"'<<' token maps to unk ({unk_id})"

    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")

    special_ids = [start_id, end_id, latent_id]

    # All IDs should be unique
    assert len(set(special_ids)) == 3, f"Special token IDs not unique: {special_ids}"

    # None of them should be unk
    for tid in special_ids:
        assert tid != unk_id, f"Special token maps to unk: {tid}"

    # Resize and init embeddings
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.resize_token_embeddings(len(tokenizer))

    embeddings = model.get_input_embeddings()
    source_emb = embeddings.weight.data[lt_id].clone()

    for tid in special_ids:
        embeddings.weight.data[tid] = source_emb.clone()
        model.lm_head.weight.data[tid] = model.lm_head.weight.data[lt_id].clone()

    # Verify embedding weights now match << source
    for tid in special_ids:
        diff = (embeddings.weight.data[tid] - source_emb).abs().max().item()
        assert diff == 0.0, f"Token {tid} embedding doesn't match source after init (diff={diff})"


# ---------------------------------------------------------------------------
# Test 3: Coconut forward with no latent tokens == direct model call
# ---------------------------------------------------------------------------

def test_coconut_forward_no_latent():
    """Coconut forward with no latent tokens is bit-exact with direct model call."""
    from coconut import Coconut

    model, tokenizer, latent_id, start_id, end_id = setup_model_and_tokenizer()
    coconut = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)
    coconut.base_causallm.eval()

    # Short synthetic sequence with no latent tokens
    input_ids = torch.tensor([[5, 100, 200, 300, 400, 50]])
    attn = torch.ones_like(input_ids)
    pos = torch.arange(input_ids.shape[1]).unsqueeze(0)
    labels = input_ids.clone()

    with torch.no_grad():
        coco_out = coconut(input_ids, attn, labels, pos)

        # Direct call via model
        emb = coconut.embedding(input_ids)
        direct_out = model(inputs_embeds=emb, attention_mask=attn, position_ids=pos)

    max_diff = (coco_out.logits - direct_out.logits).abs().max().item()
    assert max_diff < 1e-5, (
        f"Coconut no-latent logits differ from direct call: max_diff={max_diff}"
    )


# ---------------------------------------------------------------------------
# Test 4: Coconut forward with latents — shape and finite loss
# ---------------------------------------------------------------------------

def test_coconut_forward_with_latents_shape():
    """Coconut forward with 1/2/3 latent tokens produces correct logit shape + finite loss."""
    from coconut import Coconut

    for n_latents in [1, 2, 3]:
        model, tokenizer, latent_id, start_id, end_id = setup_model_and_tokenizer()
        coconut = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)
        coconut.base_causallm.eval()

        q_tokens = [100, 101, 102]
        a_tokens = [200, 201]
        token_seq = q_tokens + [start_id] + [latent_id] * n_latents + [end_id] + a_tokens
        input_ids = torch.tensor([token_seq])
        attn = torch.ones_like(input_ids)
        pos = torch.arange(input_ids.shape[1]).unsqueeze(0)
        labels = input_ids.clone()
        seq_len = input_ids.shape[1]
        vocab_size = len(tokenizer)

        with torch.no_grad():
            out = coconut(input_ids, attn, labels, pos)

        assert torch.isfinite(out.loss), (
            f"n_latents={n_latents}: loss is not finite ({out.loss.item()})"
        )
        assert out.loss.item() > 0, (
            f"n_latents={n_latents}: expected positive loss, got {out.loss.item()}"
        )
        assert out.logits.shape == (1, seq_len, vocab_size), (
            f"n_latents={n_latents}: logits shape {out.logits.shape} != "
            f"(1, {seq_len}, {vocab_size})"
        )


# ---------------------------------------------------------------------------
# Test 5: KV cache no-latent path matches direct call
# ---------------------------------------------------------------------------

def test_kv_cache_no_latent_match():
    """With no latents, Coconut final pass (kv_cache=None path) matches direct call."""
    from coconut import Coconut

    model, tokenizer, latent_id, start_id, end_id = setup_model_and_tokenizer()
    coconut = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)
    coconut.base_causallm.eval()

    input_ids = torch.tensor([[10, 20, 30, 40, 60, 70, 80]])
    attn = torch.ones_like(input_ids)
    pos = torch.arange(input_ids.shape[1]).unsqueeze(0)
    labels = input_ids.clone()

    with torch.no_grad():
        coco_out = coconut(input_ids, attn, labels, pos)
        emb = coconut.embedding(input_ids)
        direct_out = model(inputs_embeds=emb, attention_mask=attn, position_ids=pos)

    max_diff = (coco_out.logits - direct_out.logits).abs().max().item()
    assert max_diff < 1e-5, (
        f"KV cache no-latent logits differ: max_diff={max_diff}"
    )


# ---------------------------------------------------------------------------
# Test 6: backward through Coconut with masked labels
# ---------------------------------------------------------------------------

def test_backward_coconut():
    """Forward+backward with 2 latent tokens: finite loss, finite embedding grad, no NaN."""
    from coconut import Coconut

    model, tokenizer, latent_id, start_id, end_id = setup_model_and_tokenizer()
    coconut = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)

    q_tokens = [100, 101, 102]
    a_tokens = [200, 201, 202]
    # question + start + latent*2 + end + answer
    input_ids = torch.tensor([[*q_tokens, start_id, latent_id, latent_id, end_id, *a_tokens]])
    # -100 masks question + special markers + latent tokens, real labels for answer
    labels = torch.tensor([[-100] * (len(q_tokens) + 1 + 2 + 1) + a_tokens])
    attn = torch.ones_like(input_ids)
    pos = torch.arange(input_ids.shape[1]).unsqueeze(0)

    out = coconut(input_ids, attn, labels, pos)
    out.loss.backward()

    assert torch.isfinite(out.loss), f"Loss not finite: {out.loss.item()}"

    emb = coconut.embedding
    assert emb.weight.grad is not None, "Embedding has no gradient"
    assert torch.isfinite(emb.weight.grad).all(), "Embedding gradient contains inf"
    assert not torch.isnan(emb.weight.grad).any(), "Embedding gradient contains NaN"

    # Check all gradients for NaN
    for name, param in coconut.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), (
                f"NaN gradient in parameter: {name}"
            )


# ---------------------------------------------------------------------------
# Test 7: Coconut generation
# ---------------------------------------------------------------------------

def test_generation():
    """Coconut.generate on input with 2 latent tokens produces valid output."""
    from coconut import Coconut

    model, tokenizer, latent_id, start_id, end_id = setup_model_and_tokenizer()
    coconut = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)
    coconut.base_causallm.eval()

    input_ids = torch.tensor([[100, 200, 300, start_id, latent_id, latent_id, end_id, 50]])
    attn = torch.ones_like(input_ids)
    input_len = input_ids.shape[1]
    vocab_size = len(tokenizer)

    with torch.no_grad():
        output = coconut.generate(input_ids, attn, max_new_tokens=8)

    assert output.shape[0] == 1, f"Expected batch size 1, got {output.shape[0]}"
    assert output.shape[1] >= input_len, (
        f"Output length {output.shape[1]} < input length {input_len}"
    )
    tokens = output[0].tolist()
    for t in tokens:
        assert 0 <= t < vocab_size, f"Token ID {t} out of vocab range [0, {vocab_size})"


# ---------------------------------------------------------------------------
# Test 8: get_dataset ProsQA verify assert
# ---------------------------------------------------------------------------

def test_dataset_prosqa_verification_assert():
    """get_dataset on ProsQA: len==5, internal BPE assert holds, all IDs valid."""
    import itertools
    from dataset import get_dataset

    _, tokenizer, *_ = setup_model_and_tokenizer(
        add_special_tokens=False, resize=False, init_embeddings=False
    )
    tokenizer.pad_token = tokenizer.eos_token

    ds = get_dataset("data/prosqa_train.json", tokenizer, max_size=5)

    assert len(ds) == 5, f"Expected 5 samples, got {len(ds)}"

    vocab_size = tokenizer.vocab_size  # before special token expansion
    # SmolLM2 base vocab is 49152; all ProsQA tokens must be in [0, vocab_size)
    for i in range(len(ds)):
        sample = ds[i]
        all_ids = (
            sample["question_tokenized"]
            + list(itertools.chain.from_iterable(sample["steps_tokenized"]))
            + sample["answer_tokenized"][:-1]  # exclude appended eos
        )
        for tid in all_ids:
            assert 0 <= tid < vocab_size, (
                f"Sample {i}: token ID {tid} out of range [0, {vocab_size})"
            )

    # Re-run the internal concatenation assert manually for sample 0
    import json
    data = json.load(open("data/prosqa_train.json"))[:5]
    d = data[0]
    complete = d["question"] + "\n" + "\n".join(d["steps"]) + "\n### " + d["answer"]
    complete_tokenized = tokenizer.encode(complete, add_special_tokens=True) + [
        tokenizer.eos_token_id
    ]
    reconstructed = (
        ds[0]["question_tokenized"]
        + list(itertools.chain.from_iterable(ds[0]["steps_tokenized"]))
        + ds[0]["answer_tokenized"]
    )
    assert complete_tokenized == reconstructed, (
        "BPE boundary assert failed: piece-wise tokenization mismatch"
    )


# ---------------------------------------------------------------------------
# Test 9: get_cot_latent_dataset stages
# ---------------------------------------------------------------------------

def test_cot_latent_dataset_stages():
    """
    get_cot_latent_dataset at stages 0..3:
      - n_latent == min(stage, max_latent_stage)*c_thought (for sample 0, 3 steps)
      - labels are -100 for question+latent portion
      - at least some non-masked labels exist for stage >= 0
    """
    from dataset import get_dataset, get_cot_latent_dataset
    from utils import Config

    _, tokenizer, latent_id, start_id, end_id = setup_model_and_tokenizer(
        resize=False, init_embeddings=False
    )

    configs = Config(
        {
            "c_thought": 1,
            "max_latent_stage": 4,
            "pad_latent_to_max": False,
            "uniform_prob": 0.0,
            "no_cot": False,
        }
    )
    ds = get_dataset("data/prosqa_train.json", tokenizer, max_size=5)
    n_steps_sample0 = len(ds[0]["steps_tokenized"])  # 3 for ProsQA sample 0

    for stage in [0, 1, 2, 3]:
        cot_ds = get_cot_latent_dataset(
            stage, ds, configs, start_id, latent_id, end_id
        )
        sample = cot_ds[0]
        ids = sample["input_ids"]
        labels = sample["labels"]

        n_latent = sum(1 for t in ids if t == latent_id)
        expected_latent = min(stage, min(configs.max_latent_stage, n_steps_sample0)) * configs.c_thought

        assert n_latent == expected_latent, (
            f"stage={stage}: expected {expected_latent} latent tokens, got {n_latent}"
        )

        # The first (n_question + n_latent + n_special_markers) labels must be -100
        n_question = len(ds[0]["question_tokenized"])
        n_additional = 2  # start + end tokens
        masked_prefix = n_question + n_latent + n_additional
        for i in range(masked_prefix):
            assert labels[i] == -100, (
                f"stage={stage}: label at pos {i} should be -100, got {labels[i]}"
            )

        # At least some non-masked labels should exist (answer is always present)
        n_unmasked = sum(1 for l in labels if l != -100)
        assert n_unmasked > 0, f"stage={stage}: no unmasked labels found"


# ---------------------------------------------------------------------------
# Test 10: collator alignment
# ---------------------------------------------------------------------------

def test_collator_alignment():
    """
    MyCollator left-pads shorter sequences so all samples have their first
    latent token at the same position; padding tokens have attention_mask=0.
    """
    from dataset import MyCollator

    _, tokenizer, latent_id, start_id, end_id = setup_model_and_tokenizer(
        resize=False, init_embeddings=False
    )
    tokenizer.padding_side = "right"

    # feat1: latent at position 2 (shorter prefix)
    feat1 = {
        "input_ids": [10, 11, latent_id, 12, 13],
        "labels": [-100, -100, -100, 12, 13],
        "attention_mask": [1, 1, 1, 1, 1],
        "position_ids": [0, 1, 2, 3, 4],
    }
    # feat2: latent at position 4 (longer prefix)
    feat2 = {
        "input_ids": [20, 21, 22, 23, latent_id, 24, 25],
        "labels": [-100, -100, -100, -100, -100, 24, 25],
        "attention_mask": [1, 1, 1, 1, 1, 1, 1],
        "position_ids": [0, 1, 2, 3, 4, 5, 6],
    }

    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)
    batch = collator([feat1, feat2])

    ids0 = batch["input_ids"][0].tolist()
    ids1 = batch["input_ids"][1].tolist()

    # Both should have their first latent token at the same index
    pos0 = ids0.index(latent_id)
    pos1 = ids1.index(latent_id)
    assert pos0 == pos1, (
        f"First latent positions differ: sample0={pos0}, sample1={pos1}"
    )

    # The left-padded portion of feat1 should have attention_mask=0
    attn0 = batch["attention_mask"][0].tolist()
    n_pad = pos0 - 2  # feat1 had latent at position 2 originally
    for i in range(n_pad):
        assert attn0[i] == 0, (
            f"Padding token at position {i} has attention_mask={attn0[i]}, expected 0"
        )


# ---------------------------------------------------------------------------
# Test 11: full training step, Coconut mode
# ---------------------------------------------------------------------------

def test_training_step_coconut():
    """Full forward+backward on a real ProsQA sample (stage=2, coconut mode): finite loss, no NaN."""
    from coconut import Coconut
    from dataset import get_dataset, get_cot_latent_dataset, MyCollator
    from utils import Config

    model, tokenizer, latent_id, start_id, end_id = setup_model_and_tokenizer()
    coconut = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)

    configs = Config(
        {
            "c_thought": 1,
            "max_latent_stage": 4,
            "pad_latent_to_max": False,
            "uniform_prob": 0.0,
            "no_cot": False,
        }
    )
    ds = get_dataset("data/prosqa_train.json", tokenizer, max_size=5)
    cot_ds = get_cot_latent_dataset(2, ds, configs, start_id, latent_id, end_id)

    sample = cot_ds[0]
    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)
    batch = collator(
        [
            {
                "input_ids": sample["input_ids"],
                "labels": sample["labels"],
                "attention_mask": sample["attention_mask"],
                "position_ids": sample["position_ids"],
            }
        ]
    )

    out = coconut(**batch)
    out.loss.backward()

    assert torch.isfinite(out.loss), f"Loss not finite: {out.loss.item()}"

    for name, param in coconut.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), (
                f"NaN gradient in: {name}"
            )
            assert not torch.isinf(param.grad).any(), (
                f"Inf gradient in: {name}"
            )


# ---------------------------------------------------------------------------
# Test 12: full training step, CoT mode
# ---------------------------------------------------------------------------

def test_training_step_cot():
    """Full forward+backward on ProsQA sample (cot mode, stage=0, base model): finite loss."""
    from transformers import AutoModelForCausalLM
    from dataset import get_dataset, get_cot_latent_dataset, MyCollator
    from utils import Config

    _, tokenizer, *_ = setup_model_and_tokenizer(
        add_special_tokens=False, resize=False, init_embeddings=False
    )
    tokenizer.pad_token = tokenizer.eos_token

    configs = Config(
        {
            "c_thought": 1,
            "max_latent_stage": 4,
            "pad_latent_to_max": False,
            "uniform_prob": 0.0,
            "no_cot": False,
        }
    )
    ds = get_dataset("data/prosqa_train.json", tokenizer, max_size=5)
    # cot mode: no special markers, stage=0 → no latent tokens, all steps preserved
    cot_ds = get_cot_latent_dataset(
        0, ds, configs, -1, -1, -1, no_special_marker=True
    )

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

    sample = cot_ds[0]
    collator = MyCollator(tokenizer, latent_id=None, label_pad_token_id=-100)
    batch = collator(
        [
            {
                "input_ids": sample["input_ids"],
                "labels": sample["labels"],
                "attention_mask": sample["attention_mask"],
                "position_ids": sample["position_ids"],
            }
        ]
    )
    # Standard HuggingFace model: drop position_ids (it's fine to keep, but let's be explicit)
    model_batch = {k: v for k, v in batch.items() if k in ("input_ids", "attention_mask", "labels")}

    out = model(**model_batch)
    out.loss.backward()

    assert torch.isfinite(out.loss), f"CoT loss not finite: {out.loss.item()}"


# ---------------------------------------------------------------------------
# Test 13: full training step, no_thoughts mode
# ---------------------------------------------------------------------------

def test_training_step_no_thoughts():
    """Full forward+backward on ProsQA sample (no_thoughts mode: c_thought=0, no Coconut): finite loss."""
    from transformers import AutoModelForCausalLM
    from dataset import get_dataset, get_cot_latent_dataset, MyCollator
    from utils import Config

    _, tokenizer, *_ = setup_model_and_tokenizer(
        add_special_tokens=False, resize=False, init_embeddings=False
    )
    tokenizer.pad_token = tokenizer.eos_token

    # no_thoughts: c_thought=0, no_cot=True removes all steps
    configs = Config(
        {
            "c_thought": 0,
            "max_latent_stage": 4,
            "pad_latent_to_max": False,
            "uniform_prob": 0.0,
            "no_cot": True,
        }
    )
    ds = get_dataset("data/prosqa_train.json", tokenizer, max_size=5)
    no_thoughts_ds = get_cot_latent_dataset(
        0, ds, configs, -1, -1, -1, no_special_marker=True
    )

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

    sample = no_thoughts_ds[0]
    # Verify c_thought=0 produced no latent tokens
    n_latent = sum(1 for t in sample["input_ids"] if t == -1)
    assert n_latent == 0, f"no_thoughts mode should have 0 latent tokens, got {n_latent}"

    collator = MyCollator(tokenizer, latent_id=None, label_pad_token_id=-100)
    batch = collator(
        [
            {
                "input_ids": sample["input_ids"],
                "labels": sample["labels"],
                "attention_mask": sample["attention_mask"],
                "position_ids": sample["position_ids"],
            }
        ]
    )
    model_batch = {k: v for k, v in batch.items() if k in ("input_ids", "attention_mask", "labels")}

    out = model(**model_batch)
    out.loss.backward()

    assert torch.isfinite(out.loss), f"no_thoughts loss not finite: {out.loss.item()}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    tests = [
        ("test_decoder_layer_cls", test_decoder_layer_cls),
        ("test_special_token_setup", test_special_token_setup),
        ("test_coconut_forward_no_latent", test_coconut_forward_no_latent),
        ("test_coconut_forward_with_latents_shape", test_coconut_forward_with_latents_shape),
        ("test_kv_cache_no_latent_match", test_kv_cache_no_latent_match),
        ("test_backward_coconut", test_backward_coconut),
        ("test_generation", test_generation),
        ("test_dataset_prosqa_verification_assert", test_dataset_prosqa_verification_assert),
        ("test_cot_latent_dataset_stages", test_cot_latent_dataset_stages),
        ("test_collator_alignment", test_collator_alignment),
        ("test_training_step_coconut", test_training_step_coconut),
        ("test_training_step_cot", test_training_step_cot),
        ("test_training_step_no_thoughts", test_training_step_no_thoughts),
    ]

    for name, fn in tests:
        run_test(name, fn)

    print(SEPARATOR)
    n_passed = len(RESULTS["passed"])
    n_failed = len(RESULTS["failed"])
    n_total = n_passed + n_failed
    print(f"\nResults: {n_passed}/{n_total} passed")
    if RESULTS["failed"]:
        print("Failed tests:")
        for name in RESULTS["failed"]:
            print(f"  - {name}")
    else:
        print("All tests passed.")

    sys.exit(1 if RESULTS["failed"] else 0)


if __name__ == "__main__":
    main()
