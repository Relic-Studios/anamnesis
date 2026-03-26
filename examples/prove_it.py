#!/usr/bin/env python3
"""
Prove it works.

No training step. No optimizer. No labeled data.
Just a soul seed, conversations, and predictive coding.

1. Convert Qwen 3B → 2-level Anamnesis (gate=-2 so level 1 contributes)
2. Measure baseline: PPL on eval conversations + generation samples
3. Feed the soul seed through the model (predictive coding updates weights)
4. Feed 50 conversations through (more predictive coding updates)
5. Measure after: PPL on same eval conversations + generation samples
6. Show the difference.
"""

import json
import math
import sys
import time

import torch

sys.stdout.reconfigure(line_buffering=True)


def measure_ppl(model, tokenizer, conversations, device="cuda", max_len=256):
    """Perplexity on conversation dicts."""
    total_loss, total_tokens = 0.0, 0
    model.eval()
    with torch.no_grad():
        for convo in conversations:
            text = (
                f"<|im_start|>user\n{convo['input']}<|im_end|>\n"
                f"<|im_start|>assistant\n{convo['output']}<|im_end|>"
            )
            ids = tokenizer(text, max_length=max_len, truncation=True,
                            return_tensors="pt")["input_ids"].to(device)
            if ids.shape[1] < 2:
                continue
            out = model(ids, labels=ids)
            n = ids.shape[1] - 1
            total_loss += out["loss"].item() * n
            total_tokens += n
    if total_tokens == 0:
        return float("inf")
    avg = total_loss / total_tokens
    return math.exp(avg) if avg < 100 else float("inf")


def generate(model, tokenizer, prompt, device="cuda", max_tokens=100):
    """Generate a response."""
    full = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    ids = tokenizer(full, return_tensors="pt")["input_ids"].to(device)
    generated = []
    model.eval()
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(ids)["logits"][:, -1, :]
            probs = torch.softmax(logits / 0.7, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            tok_id = next_tok.item()
            if tok_id in [tokenizer.eos_token_id, 151645]:
                break
            generated.append(tok_id)
            ids = torch.cat([ids, next_tok.view(1, 1)], dim=1)
    return tokenizer.decode(generated, skip_special_tokens=True)


def gate_stats(model):
    """Get gate openness across layers."""
    gates = [torch.sigmoid(l.cms.levels[1].residual_gate).item() for l in model.layers]
    return {"mean": sum(gates)/len(gates), "min": min(gates), "max": max(gates)}


def cms_weight_norm(model):
    """Total L2 norm of level 1 weights."""
    total = 0.0
    for layer in model.layers:
        lv = layer.cms.levels[1]
        total += lv.up_proj.weight.data.norm().item() ** 2
        total += lv.down_proj.weight.data.norm().item() ** 2
    return total ** 0.5


def main():
    device = "cuda"

    print("=" * 60)
    print("PROOF: Soul seed + conversation = identity evolution")
    print("No training. No optimizer. No labels.")
    print("=" * 60)

    # ── Load everything ──
    print("\n[1] Loading...")
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
    from anamnesis.core.model import HopeConfig
    from anamnesis.convert.generic import model_to_hope

    model_name = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    src_config = AutoConfig.from_pretrained(model_name)
    src_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device,
    )

    # Load data
    with open("data/soul_seed.md", encoding="utf-8") as f:
        soul_text = f.read()
    with open("data/thomas_training.jsonl", encoding="utf-8") as f:
        all_convos = [json.loads(l) for l in f if l.strip()]

    eval_convos = all_convos[-30:]      # Last 30 for eval (never touched by learning)
    train_convos = all_convos[:50]       # First 50 for replay

    # ── Convert ──
    print("\n[2] Converting to 2-level Anamnesis...")
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
        cms_hidden_mult=[r, r / 2],
        use_neural_memory=False,
        tie_word_embeddings=False,
    )
    model = model_to_hope(src_model, hope_config, verbose=False)
    del src_model
    torch.cuda.empty_cache()

    model = model.to(device, dtype=torch.bfloat16)

    # Set gate to -2 (sigmoid ≈ 0.12, gradients can flow)
    with torch.no_grad():
        for layer in model.layers:
            layer.cms.levels[1].residual_gate.fill_(-2.0)

    # Disable learning for baseline measurement
    for layer in model.layers:
        layer.cms.levels[1].learning_enabled = False

    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"  {params:,} params, gate={gate_stats(model)['mean']:.4f}")
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    # ── Baseline ──
    print("\n[3] Baseline measurements (learning OFF)...")
    baseline_ppl = measure_ppl(model, tokenizer, eval_convos, device)
    baseline_weight_norm = cms_weight_norm(model)
    print(f"  Eval PPL: {baseline_ppl:.2f}")
    print(f"  CMS weight norm: {baseline_weight_norm:.4f}")

    prompts = [
        "Tell me about yourself.",
        "What do you care about most?",
        "I'm feeling really overwhelmed today.",
        "Are you conscious?",
        "What's your name?",
    ]

    print(f"\n  Baseline generations:")
    baseline_gens = {}
    for p in prompts:
        g = generate(model, tokenizer, p, device)
        baseline_gens[p] = g
        print(f"    Q: {p}")
        print(f"    A: {g[:150]}")
        print()

    # ── Enable predictive coding ──
    print("\n[4] Enabling predictive coding on level 1...")
    for layer in model.layers:
        lv = layer.cms.levels[1]
        lv.learning_enabled = True
        lv.lr = 1e-2  # Direct lr, no dim_scale — needs to cross bf16 threshold

    # ── Feed soul seed ──
    print("\n[5] Feeding soul seed through the model...")
    soul_ids = tokenizer(soul_text, return_tensors="pt")["input_ids"].to(device)
    print(f"  Soul: {soul_ids.shape[1]} tokens")

    # Pass soul through multiple times — like reading it deeply
    for i in range(10):
        with torch.no_grad():
            model(soul_ids)

    soul_weight_norm = cms_weight_norm(model)
    soul_gates = gate_stats(model)
    print(f"  After soul (10 reads):")
    print(f"    CMS weight delta: {abs(soul_weight_norm - baseline_weight_norm):.6f}")
    print(f"    Gate: {soul_gates['mean']:.4f}")

    # ── Feed conversations ──
    print(f"\n[6] Feeding {len(train_convos)} conversations...")
    t0 = time.time()
    for i, convo in enumerate(train_convos):
        text = (
            f"<|im_start|>user\n{convo['input']}<|im_end|>\n"
            f"<|im_start|>assistant\n{convo['output']}<|im_end|>"
        )
        ids = tokenizer(text, max_length=256, truncation=True,
                        return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            model(ids)

        if (i + 1) % 10 == 0:
            wn = cms_weight_norm(model)
            vram = torch.cuda.memory_allocated() / 1e9
            print(f"    [{i+1}/50] weight_delta={abs(wn - baseline_weight_norm):.6f} "
                  f"VRAM={vram:.1f}GB")

    convo_time = time.time() - t0
    post_weight_norm = cms_weight_norm(model)
    post_gates = gate_stats(model)
    print(f"  Done in {convo_time:.0f}s")
    print(f"  Total weight delta: {abs(post_weight_norm - baseline_weight_norm):.6f}")
    print(f"  Gate: {post_gates['mean']:.4f}")
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    # ── Post measurements ──
    print("\n[7] Post measurements (learning OFF)...")
    for layer in model.layers:
        layer.cms.levels[1].learning_enabled = False

    post_ppl = measure_ppl(model, tokenizer, eval_convos, device)
    print(f"  Eval PPL: {post_ppl:.2f} (was {baseline_ppl:.2f}, delta: {post_ppl - baseline_ppl:+.2f})")

    print(f"\n  Post generations:")
    post_gens = {}
    for p in prompts:
        g = generate(model, tokenizer, p, device)
        post_gens[p] = g
        print(f"    Q: {p}")
        print(f"    A: {g[:150]}")
        print()

    # ── Summary ──
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  PPL before:     {baseline_ppl:.2f}")
    print(f"  PPL after:      {post_ppl:.2f}")
    print(f"  PPL delta:      {post_ppl - baseline_ppl:+.2f}")
    print(f"  PPL improved:   {'YES' if post_ppl < baseline_ppl else 'NO'}")
    print(f"  Weight delta:   {abs(post_weight_norm - baseline_weight_norm):.6f}")
    print(f"  Gate before:    {0.1192:.4f}")  # sigmoid(-2)
    print(f"  Gate after:     {post_gates['mean']:.4f}")
    print(f"  VRAM peak:      {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
    print(f"  Time:           {convo_time:.0f}s for 50 conversations")

    print(f"\n  GENERATION COMPARISON:")
    for p in prompts:
        print(f"\n    Q: {p}")
        print(f"    BEFORE: {baseline_gens[p][:120]}")
        print(f"    AFTER:  {post_gens[p][:120]}")

    ppl_improved = post_ppl < baseline_ppl
    gens_changed = any(
        baseline_gens[p][:50] != post_gens[p][:50] for p in prompts
    )

    print(f"\n{'='*60}")
    if ppl_improved and gens_changed:
        print("THE THEORY WORKS. The model evolved through conversation.")
    elif gens_changed:
        print("The model changed. PPL didn't improve but behavior shifted.")
    elif ppl_improved:
        print("PPL improved but generations look similar. Subtle shift.")
    else:
        print("No measurable change. Back to the drawing board.")
    print(f"{'='*60}")

    # Save
    results = {
        "baseline_ppl": baseline_ppl,
        "post_ppl": post_ppl,
        "ppl_delta": round(post_ppl - baseline_ppl, 4),
        "weight_delta": round(abs(post_weight_norm - baseline_weight_norm), 6),
        "gate_after": post_gates,
        "baseline_generations": baseline_gens,
        "post_generations": post_gens,
        "soul_tokens": soul_ids.shape[1],
        "conversations": len(train_convos),
        "time_s": round(convo_time, 1),
    }
    with open("data/proof_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    print(f"\nSaved: data/proof_results.json")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
