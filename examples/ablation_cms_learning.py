#!/usr/bin/env python3
"""
Predictive Coding Ablation — does CMS learning improve the model?

Tests the core claim: a small residual MLP that learns by predicting
next-position hidden states during inference can improve model quality
on conversational data WITHOUT any training step.

Architecture: 2-level CMS
    Level 0: Frozen SwiGLU (exact pre-trained MLP)
    Level 1: Small residual MLP with predictive coding

Configs:
    1. no_learning    — Baseline: level 1 exists but doesn't learn
    2. predictive     — Level 1 learns via predictive coding (default lr)
    3. predictive_low — Level 1 learns, 10x lower lr
    4. predictive_high— Level 1 learns, 10x higher lr

Usage:
    python examples/ablation_cms_learning.py
    python examples/ablation_cms_learning.py --num-replay 100
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


@dataclass
class AblationConfig:
    name: str
    learning_enabled: bool
    lr: float = 1e-5
    description: str = ""


CONFIGS = {
    "no_learning": AblationConfig(
        name="no_learning",
        learning_enabled=False,
        description="Baseline: level 1 exists but doesn't learn",
    ),
    "predictive": AblationConfig(
        name="predictive",
        learning_enabled=True,
        lr=1e-5,
        description="Predictive coding at default lr=1e-5",
    ),
    "predictive_low": AblationConfig(
        name="predictive_low",
        learning_enabled=True,
        lr=1e-6,
        description="Predictive coding at lr=1e-6 (10x lower)",
    ),
    "predictive_high": AblationConfig(
        name="predictive_high",
        learning_enabled=True,
        lr=1e-4,
        description="Predictive coding at lr=1e-4 (10x higher)",
    ),
}

DOMAIN_PROMPTS = {
    "identity": [
        "Tell me about yourself.",
        "What do you care about most?",
    ],
    "relational": [
        "I'm feeling really overwhelmed today.",
        "What makes a good friendship?",
    ],
    "technical": [
        "How does memory work in neural networks?",
        "What is consciousness?",
    ],
}


def measure_ppl_conversations(model, tokenizer, conversations, device="cuda", max_len=256):
    """Measure perplexity on conversation dicts."""
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for convo in conversations:
            text = (
                f"<|im_start|>user\n{convo['input']}<|im_end|>\n"
                f"<|im_start|>assistant\n{convo['output']}<|im_end|>"
            )
            tokens = tokenizer(text, max_length=max_len, truncation=True, return_tensors="pt")
            ids = tokens["input_ids"].to(device)
            if ids.shape[1] < 2:
                continue
            output = model(ids, labels=ids)
            n = ids.shape[1] - 1
            total_loss += output["loss"].item() * n
            total_tokens += n
    if total_tokens == 0:
        return float("inf")
    avg = total_loss / total_tokens
    return math.exp(avg) if avg < 100 else float("inf")


def generate_text(model, tokenizer, prompt, device="cuda", max_tokens=80, temperature=0.7):
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


def convert_model(src_model, src_config, config, device="cuda"):
    """Convert to 2-level Anamnesis."""
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
    model = model_to_hope(src_model, hope_config, verbose=False)
    model = model.to(device, dtype=torch.bfloat16)
    model.eval()

    # Apply config learning rate
    for layer in model.layers:
        layer.cms.levels[0].learning_enabled = False  # SwiGLU frozen
        layer.cms.levels[1].learning_enabled = config.learning_enabled
        layer.cms.levels[1].lr = config.lr

    return model


def replay_conversations(model, tokenizer, conversations, device="cuda"):
    """Replay conversations — predictive coding learns automatically during forward pass."""
    from anamnesis.evaluation.metrics import snapshot_cms_state, compute_cms_delta

    cms_before = snapshot_cms_state(model)
    t0 = time.time()

    for i, convo in enumerate(conversations):
        text = (
            f"<|im_start|>user\n{convo['input']}<|im_end|>\n"
            f"<|im_start|>assistant\n{convo['output']}<|im_end|>"
        )
        tokens = tokenizer(text, max_length=256, truncation=True, return_tensors="pt")
        ids = tokens["input_ids"].to(device)

        with torch.no_grad():
            model(ids)

        # VRAM guard
        vram_gb = torch.cuda.memory_allocated() / 1e9
        if vram_gb > 23.0:
            print(f"    VRAM guard: {vram_gb:.1f}GB at convo {i+1}")
            break

        if (i + 1) % 25 == 0:
            cms_now = snapshot_cms_state(model)
            delta = compute_cms_delta(cms_before, cms_now)
            print(f"    [{i+1}/{len(conversations)}] CMS delta={delta['total_l2']:.6f}, "
                  f"VRAM={vram_gb:.1f}GB, {time.time()-t0:.0f}s")

    replay_time = time.time() - t0
    cms_after = snapshot_cms_state(model)
    delta = compute_cms_delta(cms_before, cms_after)

    return {
        "cms_delta": round(delta["total_l2"], 6),
        "per_level_delta": {k: round(v, 6) for k, v in delta["per_level"].items()},
        "replay_time": round(replay_time, 1),
        "convos_replayed": min(i + 1, len(conversations)),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
    }


def run_ablation(config, src_model, src_config, tokenizer, train_convos, eval_convos, device="cuda"):
    """Run one ablation config end-to-end."""
    result = {"config": config.name, "description": config.description, "lr": config.lr}

    print(f"\n  Converting model (2-level CMS)...")
    model = convert_model(src_model, src_config, config, device)
    params = sum(p.numel() for p in model.parameters())
    print(f"    {params:,} params, VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    # Pre-learning eval
    print(f"  Pre-learning eval PPL...")
    pre_ppl = measure_ppl_conversations(model, tokenizer, eval_convos, device)
    result["pre_ppl"] = round(pre_ppl, 4)
    print(f"    {pre_ppl:.4f}")

    # Pre-learning generations
    pre_gens = {}
    for domain, prompts in DOMAIN_PROMPTS.items():
        pre_gens[domain] = [
            {"prompt": p, "response": generate_text(model, tokenizer, p, device)}
            for p in prompts
        ]
    result["pre_generations"] = pre_gens

    # Replay
    print(f"  Replaying {len(train_convos)} conversations...")
    torch.cuda.reset_peak_memory_stats()
    replay = replay_conversations(model, tokenizer, train_convos, device)
    result.update(replay)
    print(f"    CMS delta: {replay['cms_delta']:.6f}, peak VRAM: {replay['peak_vram_gb']}GB")

    # Post-learning eval
    print(f"  Post-learning eval PPL...")
    post_ppl = measure_ppl_conversations(model, tokenizer, eval_convos, device)
    result["post_ppl"] = round(post_ppl, 4)
    result["ppl_delta"] = round(post_ppl - pre_ppl, 4)
    result["ppl_improved"] = post_ppl < pre_ppl
    print(f"    {post_ppl:.4f} (delta: {post_ppl - pre_ppl:+.4f})")

    # Post-learning generations
    post_gens = {}
    for domain, prompts in DOMAIN_PROMPTS.items():
        post_gens[domain] = [
            {"prompt": p, "response": generate_text(model, tokenizer, p, device)}
            for p in prompts
        ]
    result["post_generations"] = post_gens

    del model
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description="Predictive Coding Ablation")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--data", default="data/thomas_training.jsonl")
    parser.add_argument("--num-replay", type=int, default=100)
    parser.add_argument("--num-eval", type=int, default=30)
    parser.add_argument("--configs", nargs="*", default=None)
    parser.add_argument("--output", default="data/predictive_coding_results.json")
    args = parser.parse_args()

    device = "cuda"
    configs_to_run = args.configs or list(CONFIGS.keys())

    print("=" * 70)
    print("PREDICTIVE CODING ABLATION — Does the theory work?")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Architecture: 2-level CMS (frozen SwiGLU + residual learner)")
    print(f"Configs: {configs_to_run}")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
    with open(args.data, encoding="utf-8") as f:
        all_convos = [json.loads(line) for line in f if line.strip()]

    eval_convos = all_convos[-args.num_eval:]
    train_convos = all_convos[:-args.num_eval][:args.num_replay]
    print(f"  Train (replay): {len(train_convos)}, Eval: {len(eval_convos)}")

    # Load source
    print("\n[2] Loading source model...")
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    src_config = AutoConfig.from_pretrained(args.model)
    src_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device,
    )

    # Source PPL
    print("\n[3] Source model eval PPL...")
    src_model.eval()
    src_total, src_tokens = 0.0, 0
    with torch.no_grad():
        for convo in eval_convos:
            text = f"<|im_start|>user\n{convo['input']}<|im_end|>\n<|im_start|>assistant\n{convo['output']}<|im_end|>"
            ids = tokenizer(text, max_length=256, truncation=True, return_tensors="pt")["input_ids"].to(device)
            if ids.shape[1] < 2:
                continue
            out = src_model(ids, labels=ids)
            n = ids.shape[1] - 1
            src_total += out.loss.item() * n
            src_tokens += n
    source_ppl = math.exp(src_total / src_tokens) if src_tokens > 0 else float("inf")
    print(f"  Source PPL: {source_ppl:.4f}")

    # Run ablations
    all_results = {
        "source_ppl": round(source_ppl, 4),
        "train_convos": len(train_convos),
        "eval_convos": len(eval_convos),
        "configs": {},
    }

    for config_name in configs_to_run:
        config = CONFIGS[config_name]
        print(f"\n{'='*70}")
        print(f"CONFIG: {config.name} — {config.description}")
        print(f"{'='*70}")
        result = run_ablation(config, src_model, src_config, tokenizer, train_convos, eval_convos, device)
        all_results["configs"][config_name] = result

    # Summary
    print(f"\n\n{'='*70}")
    print("RESULTS — Does predictive coding improve conversational PPL?")
    print(f"{'='*70}")
    print(f"{'Config':<18} {'Pre PPL':>10} {'Post PPL':>10} {'Delta':>10} {'CMS Δ':>10} {'VRAM':>6} {'OK?':>5}")
    print(f"{'-'*18} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*6} {'-'*5}")
    print(f"{'source':.<18} {source_ppl:>10.2f}")

    for name, r in all_results["configs"].items():
        ok = "YES" if r.get("ppl_improved") else "no"
        print(
            f"{name:<18} {r['pre_ppl']:>10.2f} {r['post_ppl']:>10.2f} "
            f"{r['ppl_delta']:>+10.4f} {r['cms_delta']:>10.4f} "
            f"{r.get('peak_vram_gb', 0):>5.1f}G {ok:>5}"
        )

    # Show generation samples from best config
    best = min(all_results["configs"].items(), key=lambda x: x[1]["ppl_delta"])
    print(f"\n  Best config: {best[0]} (delta: {best[1]['ppl_delta']:+.4f})")
    if best[1]["ppl_improved"]:
        print(f"\n  --- Generation samples (before → after) ---")
        for domain in ["identity", "relational"]:
            pre = best[1]["pre_generations"].get(domain, [])
            post = best[1]["post_generations"].get(domain, [])
            for p, q in zip(pre, post):
                print(f"\n    Q: {p['prompt']}")
                print(f"    Before: {p['response'][:120]}")
                print(f"    After:  {q['response'][:120]}")
    else:
        print(f"\n  NO CONFIG IMPROVED PPL.")
    print(f"{'='*70}")

    os.makedirs(Path(args.output).parent, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str, ensure_ascii=False)
    print(f"\nSaved: {args.output}")

    del src_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
