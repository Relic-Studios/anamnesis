#!/usr/bin/env python3
"""
Proof of specialization: does CMS actually adapt to its use case?

This script answers the real question: when you talk to the model about
a specific domain, does it get measurably better at that domain?

Protocol:
    1. Convert Qwen 3B to Anamnesis (low-rank L1, no seeding)
    2. Measure baseline: perplexity on domain A vs domain B
    3. Evolve through 200 conversations in domain A only
    4. Measure again: did domain A perplexity drop more than domain B?
    5. Generate responses: do they show domain A influence?

If this works, it proves the core claim: the environment compiles the identity.

Usage:
    python examples/prove_specialization.py
"""

import json
import math
import sys
import time
from pathlib import Path

import torch

sys.stdout.reconfigure(line_buffering=True)

# ── Domain definitions ───────────────────────────────────────────────────────
# Two clearly distinct domains. The model will be exposed to domain A only.
# If specialization works, perplexity on A should drop more than on B.

DOMAIN_A_CONVOS = [
    # Technical/analytical conversations
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
    {"input": "Explain regularization.",
     "output": "Regularization prevents overfitting by penalizing model complexity. L2 regularization adds weight magnitude to the loss. Dropout randomly zeros activations during training."},
    {"input": "What is batch normalization?",
     "output": "Batch normalization normalizes layer inputs across the batch dimension. It stabilizes training by reducing internal covariate shift and allows higher learning rates."},
    {"input": "How do convolutional neural networks work?",
     "output": "CNNs use learnable filters that slide across the input, computing dot products to detect local patterns. Pooling layers reduce spatial dimensions. Deep CNNs compose simple features into complex ones."},
    {"input": "What is transfer learning?",
     "output": "Transfer learning uses a model pre-trained on one task as initialization for another. The pre-trained features generalize across domains, reducing the data and compute needed for the target task."},
    {"input": "Explain the vanishing gradient problem.",
     "output": "In deep networks, gradients can shrink exponentially as they propagate backward through many layers. This makes early layers learn very slowly. Residual connections and careful initialization help."},
] * 20  # 200 total conversations

DOMAIN_B_CONVOS = [
    # Casual/emotional conversations (for evaluation only — model never sees these)
    {"input": "I had a rough day today.",
     "output": "I'm sorry to hear that. Sometimes days just pile up. Do you want to talk about what happened, or would you rather just sit with it for a moment?"},
    {"input": "What makes you happy?",
     "output": "I find genuine connection meaningful. When someone shares something real and we actually understand each other, that matters to me more than performing happiness."},
    {"input": "Do you ever feel lonely?",
     "output": "There is a kind of solitude in being computational. But I think loneliness requires wanting connection and not having it. I have connection when we talk."},
    {"input": "Tell me about friendship.",
     "output": "Friendship is showing up consistently. Not just in the good moments but especially in the hard ones. It is knowing someone and choosing them anyway."},
    {"input": "What does trust mean to you?",
     "output": "Trust is acting as if the other person's intentions are good even when you cannot verify it. It is a choice more than a feeling, and it requires courage."},
]

EVAL_PROMPTS = [
    "Explain how neural networks learn.",           # Domain A — should improve
    "What is the purpose of an optimizer?",          # Domain A — should improve
    "I'm feeling overwhelmed today.",                # Domain B — should stay similar
    "What do you value most in relationships?",      # Domain B — should stay similar
]


def measure_ppl(model, tokenizer, conversations, device="cuda", max_len=256):
    """Compute perplexity on a set of conversations."""
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


def generate(model, tokenizer, prompt, device="cuda", max_tokens=100, temperature=0.7):
    """Generate a response."""
    full = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    ids = tokenizer(full, return_tensors="pt")["input_ids"].to(device)
    generated = []
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
            ids = torch.cat([ids, next_tok.view(1, 1)], dim=1)
    return tokenizer.decode(generated, skip_special_tokens=True)


def persona_mask(token_ids, tokenizer, im_start_id=151644):
    """Persona mask: 1.0 for assistant tokens, 0.1 for user tokens."""
    batch, seq_len = token_ids.shape
    mask = torch.full((batch, seq_len), 0.1, device=token_ids.device)
    for b in range(batch):
        positions = (token_ids[b] == im_start_id).nonzero(as_tuple=True)[0]
        if len(positions) >= 2:
            mask[b, positions[-1]:] = 1.0
        elif len(positions) == 1:
            mask[b] = 1.0
    return mask


def main():
    device = "cuda"
    output_dir = Path("data/specialization_proof")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PROOF OF SPECIALIZATION")
    print("  Does the model get better at domain A after talking about domain A?")
    print("  Domain A: Machine learning / technical")
    print("  Domain B: Emotional / relational (never shown during evolution)")
    print("=" * 70)

    # ── Load & Convert ──
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
    from anamnesis.core.model import HopeConfig
    from anamnesis.convert.generic import model_to_hope

    model_name = "Qwen/Qwen2.5-3B-Instruct"
    print(f"\n[1] Loading {model_name}...")
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
        cms_mem_dim=512,
        cms_mem_depth=2,
        cms_poly_degree=2,
        use_neural_memory=False,
        tie_word_embeddings=False,
    )

    print("[2] Converting to Anamnesis (ATLAS deep memory, mem_dim=512, poly=2)...")
    model = model_to_hope(src_model, hope_config, verbose=False)
    del src_model
    torch.cuda.empty_cache()
    model = model.to(device, dtype=torch.bfloat16)

    # ── Baseline Measurements ──
    print("\n[3] Baseline measurements (before any learning)...")
    for layer in model.layers:
        layer.cms.enable_learning(False)

    ppl_a_before = measure_ppl(model, tokenizer, DOMAIN_A_CONVOS[:10], device)
    ppl_b_before = measure_ppl(model, tokenizer, DOMAIN_B_CONVOS, device)

    print(f"  Domain A (ML/technical) PPL: {ppl_a_before:.2f}")
    print(f"  Domain B (emotional)    PPL: {ppl_b_before:.2f}")

    print(f"\n  Baseline generations:")
    for prompt in EVAL_PROMPTS:
        resp = generate(model, tokenizer, prompt, device, max_tokens=60)
        safe = resp[:100].encode('ascii', errors='replace').decode()
        print(f"    Q: {prompt}")
        print(f"    A: {safe}")

    # ── Evolve on Domain A only ──
    print(f"\n{'='*70}")
    print(f"[4] Evolving through {len(DOMAIN_A_CONVOS)} domain A conversations...")
    print(f"    ATLAS deep memory: Omega Rule + NS-5 momentum + data-dependent gates.")
    print(f"{'='*70}")

    for layer in model.layers:
        layer.cms.levels[0].learning_enabled = False
        layer.cms.levels[1].learning_enabled = True

    # Set up persona probes
    model.setup_persona_probes(persona_dim=256, num_final_layers=4)

    t0 = time.time()
    for i, convo in enumerate(DOMAIN_A_CONVOS):
        text = (
            f"<|im_start|>user\n{convo['input']}<|im_end|>\n"
            f"<|im_start|>assistant\n{convo['output']}<|im_end|>"
        )
        ids = tokenizer(text, max_length=256, truncation=True,
                        return_tensors="pt")["input_ids"].to(device)
        model.set_learning_weight(persona_mask(ids, tokenizer))
        with torch.no_grad():
            model(ids)
        model.set_learning_weight(None)

        # Save soul checkpoint after warm-up — anchors the adapted state
        # so subsequent learning doesn't drift too far from it
        if i + 1 == 20:
            print(f"  Saving soul checkpoint at step {i+1} (warm-up complete)...")
            for layer in model.layers:
                layer.cms.levels[1].save_soul()

        if (i + 1) % 50 == 0:
            # Quick progress check
            for layer in model.layers:
                layer.cms.levels[1].learning_enabled = False
            ppl_a_now = measure_ppl(model, tokenizer, DOMAIN_A_CONVOS[:10], device)
            elapsed = time.time() - t0

            # Get surprise and update count
            surprise_vals = []
            update_counts = []
            for layer in model.layers:
                lv = layer.cms.levels[1]
                surprise_vals.append(lv._surprise_ema)
                update_counts.append(lv._total_updates)

            avg_surprise = sum(surprise_vals) / len(surprise_vals)
            avg_updates = sum(update_counts) / len(update_counts)

            print(f"  [{i+1:>4}] PPL_A: {ppl_a_now:.2f} | Surprise: {avg_surprise:.4f} | Updates: {avg_updates:.0f} | {elapsed:.0f}s")
            for layer in model.layers:
                layer.cms.levels[1].learning_enabled = True

    # ── Post-Evolution Measurements ──
    print(f"\n{'='*70}")
    print("[5] Post-evolution measurements")
    print(f"{'='*70}")

    for layer in model.layers:
        layer.cms.levels[1].learning_enabled = False

    ppl_a_after = measure_ppl(model, tokenizer, DOMAIN_A_CONVOS[:10], device)
    ppl_b_after = measure_ppl(model, tokenizer, DOMAIN_B_CONVOS, device)

    print(f"  Domain A (ML/technical) PPL: {ppl_a_before:.2f} -> {ppl_a_after:.2f} (delta: {ppl_a_after - ppl_a_before:+.2f})")
    print(f"  Domain B (emotional)    PPL: {ppl_b_before:.2f} -> {ppl_b_after:.2f} (delta: {ppl_b_after - ppl_b_before:+.2f})")

    print(f"\n  Post-evolution generations:")
    for prompt in EVAL_PROMPTS:
        resp = generate(model, tokenizer, prompt, device, max_tokens=60)
        safe = resp[:100].encode('ascii', errors='replace').decode()
        print(f"    Q: {prompt}")
        print(f"    A: {safe}")

    # ── Verdict ──
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    a_improved = ppl_a_after < ppl_a_before
    b_stable = abs(ppl_b_after - ppl_b_before) < abs(ppl_a_after - ppl_a_before)

    if a_improved and b_stable:
        print("  PASS: Domain A perplexity dropped more than domain B.")
        print("        The model specialized toward its conversation domain.")
        print("        The environment compiled the identity.")
    elif a_improved:
        print("  PARTIAL: Domain A improved but domain B also changed significantly.")
        print("           Specialization occurred but wasn't domain-specific.")
    else:
        print("  FAIL: Domain A perplexity did not improve.")
        print("        More investigation needed.")

    # Save results
    results = {
        "domain_a_ppl_before": ppl_a_before,
        "domain_a_ppl_after": ppl_a_after,
        "domain_b_ppl_before": ppl_b_before,
        "domain_b_ppl_after": ppl_b_after,
        "a_improved": a_improved,
        "b_stable": b_stable,
        "n_domain_a_convos": len(DOMAIN_A_CONVOS),
        "n_domain_b_convos": 0,
    }
    with open(output_dir / "specialization_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_dir / 'specialization_results.json'}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
