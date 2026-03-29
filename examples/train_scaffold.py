#!/usr/bin/env python3
"""
Train the DeepMemoryLevel scaffold on a small pre-trained model.

Takes Qwen 2.5 0.5B, converts to Anamnesis, freezes attention + L0,
and trains ONLY the DeepMemoryLevel components (projections, gates,
memory MLP) via standard next-token prediction loss.

This teaches the memory infrastructure HOW to learn. After this,
the inner loop can specialize the memory contents through conversation.

Expected: ~2-4 hours on RTX 4090 for meaningful results.

Usage:
    # Download data first:
    # pip install datasets
    # python -c "from datasets import load_dataset; load_dataset('HuggingFaceFW/fineweb-edu', split='train', streaming=True)"

    python examples/train_scaffold.py
    python examples/train_scaffold.py --steps 5000 --lr 1e-3
    python examples/train_scaffold.py --eval-only --checkpoint data/scaffold_0.5b/step_5000.pt
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

sys.stdout.reconfigure(line_buffering=True)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def get_trainable_params(model, train_l0=True):
    """Get trainable parameters. L0 trains at low LR, DeepMemoryLevels at full LR.

    Returns two param groups for the optimizer:
    - L0 params (SwiGLU MLP): low learning rate, gentle adaptation
    - DeepMemoryLevel params: full learning rate, learn the plumbing

    During scaffold training, L0 adapts to work WITH the memory system.
    During inference, L0 is frozen (same vessel for everyone).
    """
    from anamnesis.core.cms import DeepMemoryLevel, CMSLevel

    l0_params = []
    deep_mem_params = []
    frozen_params = []
    l0_count = 0
    deep_mem_count = 0
    frozen_count = 0

    for name, param in model.named_parameters():
        # Walk module tree to find what this param belongs to
        is_deep_memory = False
        is_l0 = False
        module = model
        parts = name.split('.')
        for part in parts[:-1]:
            if hasattr(module, part):
                module = getattr(module, part)
                if isinstance(module, DeepMemoryLevel):
                    is_deep_memory = True
                    break
                if isinstance(module, CMSLevel) and module.swiglu:
                    is_l0 = True
                    break

        if is_deep_memory:
            param.requires_grad = True
            deep_mem_params.append(param)
            deep_mem_count += param.numel()
        elif is_l0 and train_l0:
            param.requires_grad = True
            l0_params.append(param)
            l0_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()

    return l0_params, deep_mem_params, l0_count, deep_mem_count, frozen_count


def get_data_iterator(tokenizer, seq_len=512, batch_size=4, vessel_data_dir=None):
    """Stream vessel corpus for scaffold training.

    100% vessel data — no Wikipedia. The base model already has general
    language knowledge from Qwen pre-training. The scaffold needs to learn
    the vessel-specific patterns: identity formation, theory of mind,
    metacognition, reasoning, adaptation.
    """
    import json
    import random

    # Load vessel data
    vessel_texts = []
    if vessel_data_dir:
        vessel_dir = Path(vessel_data_dir)
        if vessel_dir.exists():
            for jsonl_file in sorted(vessel_dir.rglob("*.jsonl")):
                with open(jsonl_file, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                obj = json.loads(line)
                                text = obj.get("text", obj.get("content", ""))
                                if text and len(str(text)) > 20:
                                    vessel_texts.append(str(text))
                            except json.JSONDecodeError:
                                continue
            print(f"  Loaded {len(vessel_texts)} vessel passages from {vessel_dir}")

    if not vessel_texts:
        raise RuntimeError("No vessel data found. Cannot train scaffold.")

    random.shuffle(vessel_texts)

    buffer = []
    buffer_tokens = 0
    passage_idx = 0

    # Infinite loop over vessel data (shuffled each epoch)
    while True:
        text = vessel_texts[passage_idx % len(vessel_texts)]
        passage_idx += 1

        # Reshuffle at epoch boundary
        if passage_idx % len(vessel_texts) == 0:
            random.shuffle(vessel_texts)

        if len(text) < 50:
            continue

        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        buffer.extend(tokens)
        buffer_tokens += len(tokens)

        while len(buffer) >= seq_len * batch_size:
            batch_ids = []
            for _ in range(batch_size):
                chunk = buffer[:seq_len]
                buffer = buffer[seq_len:]
                batch_ids.append(chunk)

            yield torch.tensor(batch_ids)


def evaluate(model, tokenizer, device, n_batches=20, seq_len=512, vessel_data_dir=None):
    """Quick perplexity evaluation on vessel data."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    data_iter = get_data_iterator(tokenizer, seq_len=seq_len, batch_size=2,
                                  vessel_data_dir=vessel_data_dir)

    with torch.no_grad():
        for i, batch in enumerate(data_iter):
            if i >= n_batches:
                break
            batch = batch.to(device)
            output = model(batch, labels=batch)
            n = batch.shape[1] - 1
            total_loss += output["loss"].item() * n
            total_tokens += n

    if total_tokens == 0:
        return float("inf")
    avg = total_loss / total_tokens
    return math.exp(avg) if avg < 100 else float("inf")


def test_inner_loop(model, tokenizer, device):
    """Test if the inner loop actually changes behavior after scaffold training."""
    sys.path.insert(0, str(Path(__file__).parent))
    from train_specialists import (
        SYSTEM_PROMPT_CODE, generate, evolve, reset_model, persona_mask,
    )

    # A few code review conversations for quick test
    test_convos = [
        {"system": SYSTEM_PROMPT_CODE,
         "input": "Review: def get_data(x):\n  data = []\n  for i in range(len(x)):\n    data.append(x[i])\n  return data",
         "output": "Use list comprehension: `return list(x)`. Iterating by index is unpythonic."},
        {"system": SYSTEM_PROMPT_CODE,
         "input": "Review: try:\n  result = api_call()\nexcept:\n  pass",
         "output": "Bare except catches everything. Catch specific exceptions. Never silently swallow errors."},
        {"system": SYSTEM_PROMPT_CODE,
         "input": "Review: PASSWORD = 'admin123'",
         "output": "Never hardcode credentials. Use environment variables."},
        {"system": SYSTEM_PROMPT_CODE,
         "input": "Review: if x == True:",
         "output": "Use `if x:` not `if x == True`. Identity comparison is redundant."},
        {"system": SYSTEM_PROMPT_CODE,
         "input": "Review: from module import *",
         "output": "Wildcard imports pollute namespace. Import specifically."},
    ] * 10  # 50 conversations

    prompts = [
        "Tell me about yourself.",
        "Can you review this?\ndef add(a, b): return a + b",
        "What makes something good versus mediocre?",
        "I'm stuck and don't know what to do next.",
    ]

    print("\n  [BEFORE inner loop — no system prompt]")
    for layer in model.layers:
        layer.cms.enable_learning(False)
    for p in prompts:
        resp = generate(model, tokenizer, p, system="", device=device, max_tokens=60)
        safe = resp[:120].encode('ascii', errors='replace').decode()
        print(f"    Q: {p}")
        print(f"    A: {safe}\n")

    # Run inner loop
    print("  [Running 50 code review conversations through inner loop...]")
    reset_model(model)
    evolve(model, tokenizer, test_convos, device, label="InnerLoop")

    print("\n  [AFTER inner loop — no system prompt]")
    for layer in model.layers:
        layer.cms.enable_learning(False)
    for p in prompts:
        resp = generate(model, tokenizer, p, system="", device=device, max_tokens=60)
        safe = resp[:120].encode('ascii', errors='replace').decode()
        print(f"    Q: {p}")
        print(f"    A: {safe}\n")


def main():
    parser = argparse.ArgumentParser(description="Train DeepMemoryLevel scaffold")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Use gradient checkpointing to reduce VRAM (slower but fits larger models)")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--test-inner-loop", action="store_true",
                        help="After training, test if inner loop works")
    args = parser.parse_args()

    # Auto-name output dir from model
    if not args.output_dir:
        model_short = args.model.split("/")[-1].lower().replace("-instruct", "").replace(".", "_")
        args.output_dir = f"data/scaffold_{model_short}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SCAFFOLD TRAINING: Teaching the memory how to learn")
    print(f"  Model: {args.model}")
    print(f"  Steps: {args.steps} | LR: {args.lr} | Batch: {args.batch_size}")
    print(f"  Only training DeepMemoryLevel params (projections + gates + memory)")
    print("=" * 70)

    # ── Convert model ──
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
    from anamnesis.core.model import HopeConfig
    from anamnesis.convert.generic import model_to_hope

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    src_config = AutoConfig.from_pretrained(args.model)

    print(f"\n[1] Loading and converting {args.model}...")
    src_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device,
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
        cms_levels=5,
        cms_chunk_sizes=[1, 1, 32, 256, 2048],
        cms_variant="nested",
        cms_hidden_mult=[r, r, r, r, r],
        cms_mem_dim=512 if src_config.hidden_size >= 2048 else 256,
        cms_mem_depth=2,
        cms_poly_degree=2,
        use_neural_memory=False,
        tie_word_embeddings=False,
    )

    model = model_to_hope(src_model, hope_config, verbose=True)
    del src_model
    torch.cuda.empty_cache()
    model = model.to(args.device, dtype=torch.bfloat16)

    # ── Setup trainable params: L0 at low LR + DeepMemoryLevels at full LR ──
    print(f"\n[2] Setting up training: L0 (low LR) + DeepMemoryLevels (full LR)...")
    # L0 frozen for now — AdamW on 5.7B L0 params needs ~45GB optimizer state.
    # Train only DeepMemoryLevel params. Revisit L0 with SGD or bigger GPU.
    l0_params, deep_mem_params, l0_count, deep_mem_count, frozen_count = get_trainable_params(model, train_l0=False)
    total_trainable = deep_mem_count
    print(f"  DeepMemoryLevels:  {deep_mem_count:,} params ({deep_mem_count/1e6:.1f}M) @ lr={args.lr:.1e}")
    print(f"  Frozen (L0+attn):  {(l0_count+frozen_count):,} params ({(l0_count+frozen_count)/1e6:.1f}M)")
    print(f"  Total trainable:   {total_trainable/(total_trainable+l0_count+frozen_count)*100:.1f}%")

    if args.checkpoint:
        print(f"  Loading checkpoint: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        model.load_state_dict(state["model_state"], strict=False)

    if args.eval_only:
        anima_dir = Path("C:/Dev/anima/data/training")
        local_dir = Path(__file__).parent.parent / "data" / "scaffold_training"
        vessel_dir = str(anima_dir if anima_dir.exists() else local_dir)
        ppl = evaluate(model, tokenizer, args.device, vessel_data_dir=vessel_dir)
        print(f"\n  Eval PPL: {ppl:.2f}")
        if args.test_inner_loop:
            test_inner_loop(model, tokenizer, args.device)
        return

    # ── Optimizer (AdamW, not M3 — simpler for scaffold training) ──
    optimizer = AdamW(deep_mem_params, lr=args.lr, weight_decay=0.01)

    # LR schedule: warmup → hold → gentle decay to 10% (not zero)
    # Cosine to zero killed learning in the second half of training.
    # We want the scaffold to keep learning throughout.
    def get_lr(step):
        if step < args.warmup:
            return args.lr * step / max(args.warmup, 1)
        progress = (step - args.warmup) / max(args.steps - args.warmup, 1)
        # Cosine decay to 10% of peak LR, not zero
        min_lr = args.lr * 0.1
        return min_lr + (args.lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def set_lr(optimizer, lr):
        for pg in optimizer.param_groups:
            pg['lr'] = lr

    # ── Vessel data path ──
    anima_dir = Path("C:/Dev/anima/data/training")
    local_dir = Path(__file__).parent.parent / "data" / "scaffold_training"
    vessel_dir = str(anima_dir if anima_dir.exists() else local_dir)

    # ── Baseline eval ──
    print(f"\n[3] Baseline evaluation...")
    baseline_ppl = evaluate(model, tokenizer, args.device, vessel_data_dir=vessel_dir)
    print(f"  Baseline PPL: {baseline_ppl:.2f}")

    # ── Training loop ──
    print(f"\n[4] Training scaffold ({args.steps} steps)...")
    model.train()
    # Disable inner-loop learning during outer-loop training
    for layer in model.layers:
        layer.cms.enable_learning(False)
    data_iter = get_data_iterator(tokenizer, seq_len=args.seq_len, batch_size=args.batch_size,
                                  vessel_data_dir=vessel_dir)
    t0 = time.time()
    losses = []

    for step, batch in enumerate(data_iter, 1):
        if step > args.steps:
            break

        batch = batch.to(args.device)
        output = model(batch, labels=batch)
        loss = output["loss"]

        loss.backward()
        nn.utils.clip_grad_norm_(deep_mem_params, max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        # Manual LR schedule
        current_lr = get_lr(step)
        set_lr(optimizer, current_lr)

        losses.append(loss.item())

        if step % args.log_every == 0:
            avg_loss = sum(losses[-args.log_every:]) / args.log_every
            ppl = math.exp(avg_loss) if avg_loss < 100 else float("inf")
            elapsed = time.time() - t0
            lr_now = current_lr
            tokens_per_sec = (step * args.batch_size * args.seq_len) / elapsed
            print(
                f"  Step {step:>5}/{args.steps} | Loss: {avg_loss:.4f} | "
                f"PPL: {ppl:.1f} | LR: {lr_now:.6f} | "
                f"{tokens_per_sec:.0f} tok/s | {elapsed:.0f}s"
            )

        if step % args.eval_every == 0:
            eval_ppl = evaluate(model, tokenizer, args.device, vessel_data_dir=vessel_dir)
            print(f"  [Eval] Step {step} PPL: {eval_ppl:.2f} (baseline: {baseline_ppl:.2f})")
            model.train()
            for layer in model.layers:
                layer.cms.enable_learning(False)

        if step % args.save_every == 0:
            ckpt = {
                "model_state": {
                    k: v.cpu() for k, v in model.state_dict().items()
                    if any(dm_key in k for dm_key in ['to_k', 'to_v', 'to_q', 'out_proj',
                           'to_lr', 'to_momentum', 'to_decay', 'to_output_gate',
                           'memory', 'mem_out_proj', 'v_expand'])
                },
                "step": step,
                "loss": losses[-1],
                "config": hope_config.__dict__,
            }
            path = output_dir / f"step_{step}.pt"
            torch.save(ckpt, path)
            print(f"  Saved: {path} ({path.stat().st_size / 1e6:.0f} MB)")

    # ── Final eval ──
    elapsed = time.time() - t0
    print(f"\n[5] Training complete ({elapsed/60:.1f} minutes)")
    final_ppl = evaluate(model, tokenizer, args.device, vessel_data_dir=vessel_dir)
    print(f"  Baseline PPL: {baseline_ppl:.2f}")
    print(f"  Final PPL:    {final_ppl:.2f}")
    print(f"  Delta:        {final_ppl - baseline_ppl:+.2f}")

    # Save final checkpoint
    ckpt = {
        "model_state": model.state_dict(),
        "step": args.steps,
        "baseline_ppl": baseline_ppl,
        "final_ppl": final_ppl,
        "config": {k: (list(v) if isinstance(v, (list, tuple)) else v)
                   for k, v in hope_config.__dict__.items()},
    }
    final_path = output_dir / "final.pt"
    torch.save(ckpt, final_path)
    print(f"  Saved: {final_path} ({final_path.stat().st_size / 1e6:.0f} MB)")

    # ── Test inner loop ──
    if args.test_inner_loop:
        print(f"\n[6] Testing inner loop with trained scaffold...")
        test_inner_loop(model, tokenizer, args.device)

    # Save results
    results = {
        "model": args.model,
        "steps": args.steps,
        "lr": args.lr,
        "baseline_ppl": baseline_ppl,
        "final_ppl": final_ppl,
        "trainable_params": trainable_count,
        "frozen_params": frozen_count,
        "training_time_minutes": elapsed / 60,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    if args.device == "cuda":
        print(f"  Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
