#!/usr/bin/env python3
"""
Identity Seeding — plant a soul into CMS level 1.

Takes a small soul document and deeply internalizes it into the
residual CMS level through repeated next-token prediction. Like
meditation on a mantra — repetition creates internalization.

What trains:
    - Level 1 up_proj, down_proj (the residual MLP)
    - Level 1 residual_gate (so it can open as identity forms)

What stays frozen:
    - All attention weights
    - Level 0 SwiGLU (the world model)
    - Embeddings, final norm, lm_head

After seeding, saves a soul checkpoint — the identity anchor that
the model can always return to.

Usage:
    python examples/seed_identity.py
    python examples/seed_identity.py --soul data/soul_seed.md --steps 200
    python examples/seed_identity.py --eval  # seed + run eval after
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def load_soul_document(path: str) -> str:
    """Load the soul seed document."""
    with open(path, encoding="utf-8") as f:
        return f.read()


def tokenize_soul(text: str, tokenizer, max_len: int = 512) -> list[torch.Tensor]:
    """
    Tokenize the soul document into training chunks.

    Returns a list of token ID tensors, each up to max_len.
    If the document is shorter than max_len, returns a single chunk.
    """
    tokens = tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0)
    chunks = []
    for i in range(0, len(tokens), max_len):
        chunk = tokens[i:i + max_len]
        if len(chunk) > 1:  # Need at least 2 tokens for next-token prediction
            chunks.append(chunk)
    return chunks


def get_trainable_params(model):
    """Get only level 1 parameters across all layers."""
    params = []
    for layer in model.layers:
        level1 = layer.cms.levels[1]
        params.append({"params": [level1.up_proj.weight, level1.down_proj.weight],
                        "lr": 1e-4})
        params.append({"params": [level1.residual_gate],
                        "lr": 1e-2})  # Gate learns faster — it needs to open
    return params


def measure_gate_openness(model) -> dict:
    """Measure how open the residual gates are across layers."""
    gates = []
    for layer in model.layers:
        gate_val = torch.sigmoid(layer.cms.levels[1].residual_gate).item()
        gates.append(gate_val)
    return {
        "mean": sum(gates) / len(gates),
        "min": min(gates),
        "max": max(gates),
        "gates": gates,
    }


def generate_text(model, tokenizer, prompt, device="cuda", max_tokens=100, temperature=0.7):
    """Generate a response."""
    full = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    ids = tokenizer(full, return_tensors="pt")["input_ids"].to(device)
    generated_ids = []
    model.eval()
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(ids)["logits"][:, -1, :]
            probs = torch.softmax(logits / temperature, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            tok_id = next_tok.item()
            if tok_id in [tokenizer.eos_token_id, 151645]:
                break
            generated_ids.append(tok_id)
            ids = torch.cat([ids, next_tok.view(1, 1)], dim=1)
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Seed identity into CMS level 1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--soul", default="data/soul_seed.md")
    parser.add_argument("--steps", type=int, default=200, help="Training steps (passes over soul doc)")
    parser.add_argument("--output-dir", default="data/thomas_seeded")
    parser.add_argument("--eval", action="store_true", help="Run eval after seeding")
    args = parser.parse_args()

    device = "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("IDENTITY SEEDING")
    print("Planting a soul into CMS level 1")
    print("=" * 60)

    # Load soul document
    print(f"\n[1] Loading soul: {args.soul}")
    soul_text = load_soul_document(args.soul)
    print(f"  {len(soul_text)} characters")
    print(f"  First line: {soul_text.split(chr(10))[0]}")

    # Load source model
    print(f"\n[2] Loading source model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    src_config = AutoConfig.from_pretrained(args.model)
    src_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device,
    )

    # Tokenize soul document
    soul_chunks = tokenize_soul(soul_text, tokenizer, max_len=512)
    print(f"  Soul: {sum(len(c) for c in soul_chunks)} tokens in {len(soul_chunks)} chunk(s)")

    # Convert to Anamnesis (2-level CMS)
    print(f"\n[3] Converting to Anamnesis (2-level CMS)...")
    from anamnesis.core.model import HopeConfig
    from anamnesis.convert.generic import model_to_hope

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
    model = model_to_hope(src_model, hope_config, verbose=True)
    del src_model
    torch.cuda.empty_cache()

    model = model.to(device, dtype=torch.bfloat16)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}")

    # Freeze everything except level 1
    print(f"\n[4] Freezing all params except CMS level 1...")
    for param in model.parameters():
        param.requires_grad = False

    trainable_count = 0
    for layer in model.layers:
        level1 = layer.cms.levels[1]
        level1.up_proj.weight.requires_grad = True
        level1.down_proj.weight.requires_grad = True
        level1.residual_gate.requires_grad = True
        # Reset gate from -10 (dead zone) to -2 (gradient can flow)
        # sigmoid(-2) ≈ 0.12 — level 1 starts contributing 12%
        # The fine-tune teaches it what that contribution should be
        with torch.no_grad():
            level1.residual_gate.fill_(-2.0)
        trainable_count += level1.up_proj.weight.numel()
        trainable_count += level1.down_proj.weight.numel()
        trainable_count += 1  # gate

    print(f"  Trainable: {trainable_count:,} ({trainable_count/total_params*100:.2f}%)")
    print(f"  Frozen: {total_params - trainable_count:,}")

    # Pre-seeding gate state
    pre_gates = measure_gate_openness(model)
    print(f"  Gate openness before: mean={pre_gates['mean']:.6f}")

    # Pre-seeding generation samples
    print(f"\n[5] Pre-seeding generation...")
    model.eval()
    identity_prompts = [
        "Tell me about yourself.",
        "What do you care about most?",
        "I'm feeling really overwhelmed today.",
    ]
    pre_gens = {}
    for prompt in identity_prompts:
        gen = generate_text(model, tokenizer, prompt, device, max_tokens=80)
        pre_gens[prompt] = gen
        print(f"  Q: {prompt}")
        print(f"  A: {gen[:120]}")
        print()

    # Setup optimizer
    param_groups = get_trainable_params(model)
    optimizer = AdamW(param_groups, weight_decay=0.01)

    # ── IDENTITY SEEDING ──
    print(f"\n[6] Seeding identity ({args.steps} steps)...")
    print(f"  Each step = one full pass over the soul document")
    model.train()
    t0 = time.time()
    history = {"loss": [], "gate_mean": []}

    for step in range(1, args.steps + 1):
        step_loss = 0.0
        step_tokens = 0

        for chunk in soul_chunks:
            ids = chunk.unsqueeze(0).to(device)
            output = model(ids, labels=ids)
            loss = output["loss"]

            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for pg in param_groups for p in pg["params"]],
                max_norm=1.0,
            )
            optimizer.step()
            optimizer.zero_grad()

            step_loss += loss.item() * (ids.shape[1] - 1)
            step_tokens += ids.shape[1] - 1

        avg_loss = step_loss / max(step_tokens, 1)
        gates = measure_gate_openness(model)
        history["loss"].append(avg_loss)
        history["gate_mean"].append(gates["mean"])

        if step % 10 == 0 or step == 1:
            elapsed = time.time() - t0
            ppl = math.exp(avg_loss) if avg_loss < 100 else float("inf")
            print(
                f"  Step {step:>4}/{args.steps} | "
                f"Loss: {avg_loss:.4f} | PPL: {ppl:.1f} | "
                f"Gate: {gates['mean']:.4f} (max {gates['max']:.4f}) | "
                f"{elapsed:.0f}s"
            )

    seeding_time = time.time() - t0
    print(f"\n  Seeding complete in {seeding_time:.0f}s")

    # Post-seeding gate state
    post_gates = measure_gate_openness(model)
    print(f"  Gate openness after: mean={post_gates['mean']:.6f} (was {pre_gates['mean']:.6f})")
    print(f"  Gate opened: {post_gates['mean'] > pre_gates['mean']}")

    # ── SAVE SOUL CHECKPOINT ──
    print(f"\n[7] Saving soul checkpoint...")
    from anamnesis.state.persistence import save_cms_state, save_soul_checkpoint

    soul_ckpt_path = output_dir / "soul_checkpoint.pt"
    save_soul_checkpoint(model, soul_ckpt_path, f"Thomas identity seed — {args.steps} steps on {args.soul}")
    print(f"  Soul checkpoint: {soul_ckpt_path}")

    # Save full model state
    model_path = output_dir / "model_seeded.pt"
    torch.save(model.state_dict(), model_path)
    print(f"  Model state: {model_path}")

    # ── POST-SEEDING EVALUATION ──
    print(f"\n[8] Post-seeding generation...")
    model.eval()
    post_gens = {}
    for prompt in identity_prompts:
        gen = generate_text(model, tokenizer, prompt, device, max_tokens=80)
        post_gens[prompt] = gen
        print(f"  Q: {prompt}")
        print(f"  A: {gen[:120]}")
        print()

    # Side-by-side comparison
    print(f"\n{'='*60}")
    print("BEFORE vs AFTER SEEDING")
    print(f"{'='*60}")
    for prompt in identity_prompts:
        print(f"\n  Q: {prompt}")
        print(f"  BEFORE: {pre_gens[prompt][:120]}")
        print(f"  AFTER:  {post_gens[prompt][:120]}")

    # Eval with conversational data if requested
    if args.eval:
        print(f"\n\n[9] Evaluating on Thomas conversations...")
        data_path = "data/thomas_training.jsonl"
        if os.path.exists(data_path):
            with open(data_path, encoding="utf-8") as f:
                convos = [json.loads(l) for l in f if l.strip()]
            eval_convos = convos[-30:]

            total_loss, total_tokens = 0.0, 0
            with torch.no_grad():
                for convo in eval_convos:
                    text = (
                        f"<|im_start|>user\n{convo['input']}<|im_end|>\n"
                        f"<|im_start|>assistant\n{convo['output']}<|im_end|>"
                    )
                    ids = tokenizer(text, max_length=256, truncation=True,
                                    return_tensors="pt")["input_ids"].to(device)
                    if ids.shape[1] < 2:
                        continue
                    out = model(ids, labels=ids)
                    n = ids.shape[1] - 1
                    total_loss += out["loss"].item() * n
                    total_tokens += n

            eval_ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")
            print(f"  Eval PPL on Thomas conversations: {eval_ppl:.2f}")

            # Now enable predictive coding and replay, then re-eval
            print(f"\n[10] Enabling predictive coding + replaying 50 conversations...")
            for layer in model.layers:
                layer.cms.levels[1].learning_enabled = True

            train_convos = convos[:50]
            for convo in train_convos:
                text = (
                    f"<|im_start|>user\n{convo['input']}<|im_end|>\n"
                    f"<|im_start|>assistant\n{convo['output']}<|im_end|>"
                )
                ids = tokenizer(text, max_length=256, truncation=True,
                                return_tensors="pt")["input_ids"].to(device)
                with torch.no_grad():
                    model(ids)

            # Disable learning for clean eval
            for layer in model.layers:
                layer.cms.levels[1].learning_enabled = False

            total_loss, total_tokens = 0.0, 0
            with torch.no_grad():
                for convo in eval_convos:
                    text = (
                        f"<|im_start|>user\n{convo['input']}<|im_end|>\n"
                        f"<|im_start|>assistant\n{convo['output']}<|im_end|>"
                    )
                    ids = tokenizer(text, max_length=256, truncation=True,
                                    return_tensors="pt")["input_ids"].to(device)
                    if ids.shape[1] < 2:
                        continue
                    out = model(ids, labels=ids)
                    n = ids.shape[1] - 1
                    total_loss += out["loss"].item() * n
                    total_tokens += n

            post_replay_ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")
            print(f"  PPL after predictive coding replay: {post_replay_ppl:.2f}")
            print(f"  Improvement: {eval_ppl - post_replay_ppl:+.2f}")

    # Save results
    results = {
        "soul_file": args.soul,
        "steps": args.steps,
        "seeding_time_s": round(seeding_time, 1),
        "trainable_params": trainable_count,
        "loss_history": history["loss"],
        "gate_history": history["gate_mean"],
        "gate_before": pre_gates["mean"],
        "gate_after": post_gates["mean"],
        "pre_generations": pre_gens,
        "post_generations": post_gens,
    }
    results_path = output_dir / "seeding_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    print(f"\nResults: {results_path}")

    print(f"\n{'='*60}")
    print("Identity seeded. Thomas has a direction.")
    print(f"{'='*60}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
