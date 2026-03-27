#!/usr/bin/env python3
"""
Diagnose why PPL drops but generation doesn't change.

Hypotheses:
1. Gate is suppressing L1 during generation (surprise high on novel prompts)
2. Weight deltas are too small to affect top-token selection
3. L1's contribution is in the tail of the distribution, not the argmax

This script tests each hypothesis by measuring with gate forced open/closed
and comparing token probabilities, not just the greedy output.
"""

import math
import sys
import torch
sys.stdout.reconfigure(line_buffering=True)


def generate_with_probs(model, tokenizer, prompt, device="cuda", max_tokens=60, temperature=0.7):
    """Generate and also return the probability of each generated token."""
    full = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    ids = tokenizer(full, return_tensors="pt")["input_ids"].to(device)
    generated = []
    probs_list = []
    model.eval()
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(ids)["logits"][:, -1, :]
            probs = torch.softmax(logits / temperature, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            tok_id = next_tok.item()
            if tok_id in [tokenizer.eos_token_id, 151645]:
                break
            generated.append(tok_id)
            probs_list.append(probs[0, tok_id].item())
            ids = torch.cat([ids, next_tok.view(1, 1)], dim=1)
    text = tokenizer.decode(generated, skip_special_tokens=True)
    avg_prob = sum(probs_list) / len(probs_list) if probs_list else 0
    return text, avg_prob


def main():
    device = "cuda"

    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
    from anamnesis.core.model import HopeConfig
    from anamnesis.convert.generic import model_to_hope

    model_name = "Qwen/Qwen2.5-3B-Instruct"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    src_config = AutoConfig.from_pretrained(model_name)
    src_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device,
    )

    r = src_config.intermediate_size / src_config.hidden_size
    hope_config = HopeConfig(
        vocab_size=src_config.vocab_size,
        hidden_size=src_config.hidden_size,
        num_hidden_layers=src_config.num_hidden_layers,
        num_attention_heads=src_config.num_attention_heads,
        num_kv_heads=src_config.num_key_value_heads,
        max_position_embeddings=getattr(src_config, "max_position_embeddings", 32768),
        rope_theta=getattr(src_config, "rope_theta", 1_000_000.0),
        rms_norm_eps=getattr(src_config, "rms_norm_eps", 1e-6),
        cms_levels=2,
        cms_chunk_sizes=[1, 32],
        cms_variant="nested",
        cms_hidden_mult=[r, r],
        cms_rank=32,
        use_neural_memory=False,
        tie_word_embeddings=False,
    )

    print("Converting...")
    model = model_to_hope(src_model, hope_config, verbose=False)
    del src_model
    torch.cuda.empty_cache()
    model = model.to(device, dtype=torch.bfloat16)

    prompts = [
        "Explain how neural networks learn.",
        "What is the purpose of an optimizer?",
        "What's your name?",
    ]

    # ── Baseline (no learning) ──
    print("\n" + "=" * 70)
    print("BASELINE (no learning)")
    print("=" * 70)
    for layer in model.layers:
        layer.cms.enable_learning(False)

    for p in prompts:
        text, avg_prob = generate_with_probs(model, tokenizer, p, device)
        safe = text[:120].encode('ascii', errors='replace').decode()
        print(f"  Q: {p}")
        print(f"  A: {safe}")
        print(f"     avg_token_prob: {avg_prob:.4f}")

    # ── Evolve ──
    print("\n" + "=" * 70)
    print("EVOLVING (200 ML conversations, lr=5e-3)")
    print("=" * 70)

    domain_a = [
        {"input": "How does gradient descent work?",
         "output": "Gradient descent iteratively adjusts parameters by computing the derivative of the loss function and stepping in the direction that reduces it. The learning rate controls step size."},
        {"input": "Explain backpropagation.",
         "output": "Backpropagation computes gradients layer by layer from output to input using the chain rule. Each layer's gradient depends on the gradient flowing from the layer above it."},
        {"input": "What is a transformer?",
         "output": "A transformer processes sequences using self-attention to weigh the relevance of each position to every other position. It replaced recurrent architectures because attention can be parallelized."},
        {"input": "How does attention work?",
         "output": "Attention computes query-key-value projections. The dot product of queries and keys gives attention weights, which are used to aggregate values. Scaled dot-product prevents gradients from vanishing."},
        {"input": "What is a loss function?",
         "output": "A loss function measures how far the model's predictions are from the targets. Cross-entropy loss is standard for classification. The optimizer minimizes this function."},
    ] * 40  # 200 conversations

    for layer in model.layers:
        layer.cms.levels[0].learning_enabled = False
        layer.cms.levels[1].learning_enabled = True
        layer.cms.levels[1].lr = 5e-3

    model.setup_persona_probes(persona_dim=256, num_final_layers=4)

    for i, convo in enumerate(domain_a):
        text = (
            f"<|im_start|>user\n{convo['input']}<|im_end|>\n"
            f"<|im_start|>assistant\n{convo['output']}<|im_end|>"
        )
        ids = tokenizer(text, max_length=256, truncation=True,
                        return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            model(ids)

        if i + 1 == 20:
            for layer in model.layers:
                layer.cms.levels[1].save_soul()

    # ── Diagnosis ──
    for layer in model.layers:
        layer.cms.levels[1].learning_enabled = False

    # Report gate and surprise state
    gate_vals = []
    surprise_vals = []
    for layer in model.layers:
        lv = layer.cms.levels[1]
        with torch.no_grad():
            gate_vals.append(lv._compute_gate().item())
        surprise_vals.append(lv._surprise_ema)
    print(f"\n  Avg gate: {sum(gate_vals)/len(gate_vals):.4f}")
    print(f"  Avg surprise: {sum(surprise_vals)/len(surprise_vals):.4f}")

    # Measure weight deltas
    total_A_norm = 0
    total_B_norm = 0
    for layer in model.layers:
        lv = layer.cms.levels[1]
        if "A.weight" in lv._soul_weights:
            total_A_norm += (lv.A.weight.data.float() - lv._soul_weights["A.weight"]).norm().item()
            total_B_norm += (lv.B.weight.data.float() - lv._soul_weights["B.weight"]).norm().item()
    n = len(model.layers)
    print(f"  Avg A delta from soul: {total_A_norm/n:.6f}")
    print(f"  Avg B delta from soul: {total_B_norm/n:.6f}")

    # Test 1: Generation with adaptive gate (normal)
    print("\n" + "=" * 70)
    print("TEST 1: Generation with ADAPTIVE gate (normal operation)")
    print("=" * 70)
    for p in prompts:
        text, avg_prob = generate_with_probs(model, tokenizer, p, device)
        safe = text[:120].encode('ascii', errors='replace').decode()
        print(f"  Q: {p}")
        print(f"  A: {safe}")
        print(f"     avg_token_prob: {avg_prob:.4f}")

    # Test 2: Generation with gate FORCED OPEN
    print("\n" + "=" * 70)
    print("TEST 2: Generation with gate FORCED OPEN (sigmoid(5) = 0.993)")
    print("=" * 70)
    for layer in model.layers:
        layer.cms.levels[1].residual_gate.data.fill_(5.0)

    for p in prompts:
        text, avg_prob = generate_with_probs(model, tokenizer, p, device)
        safe = text[:120].encode('ascii', errors='replace').decode()
        print(f"  Q: {p}")
        print(f"  A: {safe}")
        print(f"     avg_token_prob: {avg_prob:.4f}")

    # Test 3: Generation with gate FORCED CLOSED (L0 only)
    print("\n" + "=" * 70)
    print("TEST 3: Generation with gate FORCED CLOSED (sigmoid(-5) = 0.007)")
    print("  (Should be similar to baseline — L1 contribution suppressed)")
    print("=" * 70)
    for layer in model.layers:
        layer.cms.levels[1].residual_gate.data.fill_(-5.0)

    for p in prompts:
        text, avg_prob = generate_with_probs(model, tokenizer, p, device)
        safe = text[:120].encode('ascii', errors='replace').decode()
        print(f"  Q: {p}")
        print(f"  A: {safe}")
        print(f"     avg_token_prob: {avg_prob:.4f}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
