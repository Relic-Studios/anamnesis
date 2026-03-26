#!/usr/bin/env python3
"""
Benchmark Qwen2.5-3B-Instruct with Anamnesis CMS.

Three-part benchmark:
1. Conversion fidelity: does PPL match between source and Anamnesis?
2. CMS learning at inference: does replaying conversations change CMS weights?
3. Generation quality: coherent output after conversion?

Uses fused CMS kernel path for efficient inference-time learning.

Requirements: RTX 4090 (25GB+ VRAM), Qwen/Qwen2.5-3B-Instruct

Usage:
    python examples/benchmark_qwen3b.py
    python examples/benchmark_qwen3b.py --num-replay 50
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch import Tensor

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from anamnesis.core.model import HopeModel, HopeConfig
from anamnesis.convert.generic import model_to_hope
from anamnesis.evaluation.metrics import snapshot_cms_state, compute_cms_delta, compute_surprise_profile
from anamnesis.state.persistence import save_cms_state


def measure_perplexity_texts(model, tokenizer, texts, device="cuda", max_len=256):
    """Measure perplexity on a list of raw text strings."""
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for text in texts:
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
    return math.exp(total_loss / total_tokens)


def generate_text(model, tokenizer, prompt, device="cuda", max_tokens=100, temperature=0.7):
    """Generate a response from a prompt."""
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
    parser = argparse.ArgumentParser(description="Benchmark Qwen2.5-3B with Anamnesis")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--data", default="data/thomas_training.jsonl")
    parser.add_argument("--num-replay", type=int, default=30)
    parser.add_argument("--output", default="data/benchmark_3b_results.json")
    parser.add_argument("--save-state", default="data/benchmark_3b_cms_state.pt")
    args = parser.parse_args()

    device = "cuda"
    results = {"model": args.model, "device": torch.cuda.get_device_name(0)}

    print("=" * 70)
    print(f"ANAMNESIS BENCHMARK: {args.model}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 70)

    # =====================================================================
    # PART 1: CONVERSION FIDELITY
    # =====================================================================
    print("\n--- PART 1: CONVERSION FIDELITY ---")

    print("\n[1] Loading source model...")
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    src_config = AutoConfig.from_pretrained(args.model)
    src_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device,
    )
    src_params = sum(p.numel() for p in src_model.parameters())
    print(f"  Source: {src_params:,} params, {time.time()-t0:.1f}s")

    # Measure source PPL on short test strings
    test_strings = [
        "The capital of France is Paris.",
        "Machine learning models can be trained on large datasets to perform various tasks.",
        "In the beginning was the Word, and the Word was with God.",
        "The quick brown fox jumps over the lazy dog.",
        "Neural networks consist of layers of interconnected nodes that process information.",
    ]
    print("[2] Measuring source perplexity...")
    src_ppl = measure_perplexity_texts(src_model, tokenizer, test_strings, device)
    results["source_perplexity"] = round(src_ppl, 2)
    print(f"  Source PPL: {src_ppl:.2f}")

    # Generate from source for comparison
    print("[3] Source generation sample...")
    src_gen = generate_text(src_model, tokenizer, "What is consciousness?", device, max_tokens=60)
    results["source_generation"] = src_gen
    print(f"  Source: {src_gen[:150]}...")

    # Convert
    print("\n[4] Converting to Anamnesis...")
    t0 = time.time()
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
        cms_levels=3,
        cms_chunk_sizes=[1, 32, 256],
        cms_variant="nested",
        cms_hidden_mult=[r, r / 2, r / 4],
        use_neural_memory=False,
        tie_word_embeddings=False,
    )
    model = model_to_hope(src_model, hope_config, verbose=True)
    del src_model
    torch.cuda.empty_cache()

    model = model.to(device, dtype=torch.bfloat16)
    model.eval()

    anamnesis_params = model.num_parameters(trainable_only=False)
    results["anamnesis_params"] = anamnesis_params
    results["param_ratio"] = round(anamnesis_params / src_params, 3)
    results["conversion_time_s"] = round(time.time() - t0, 1)
    results["vram_gb"] = round(torch.cuda.memory_allocated() / 1e9, 1)
    print(f"  Anamnesis: {anamnesis_params:,} params ({results['param_ratio']}x)")
    print(f"  VRAM: {results['vram_gb']} GB, converted in {results['conversion_time_s']}s")

    # Measure converted PPL (learning disabled for clean comparison)
    for layer in model.layers:
        layer.cms.enable_learning(False)

    print("\n[5] Measuring Anamnesis perplexity (learning off)...")
    anamnesis_ppl = measure_perplexity_texts(model, tokenizer, test_strings, device)
    results["anamnesis_perplexity"] = round(anamnesis_ppl, 2)
    results["ppl_delta"] = round(anamnesis_ppl - src_ppl, 2)
    print(f"  Anamnesis PPL: {anamnesis_ppl:.2f} (delta: {anamnesis_ppl - src_ppl:+.2f})")

    print("\n[6] Anamnesis generation sample...")
    gen = generate_text(model, tokenizer, "What is consciousness?", device, max_tokens=60)
    results["anamnesis_generation"] = gen
    print(f"  Anamnesis: {gen[:150]}...")

    # =====================================================================
    # PART 2: CMS INFERENCE-TIME LEARNING
    # =====================================================================
    print("\n\n--- PART 2: CMS INFERENCE-TIME LEARNING ---")

    # Re-enable learning (only level 0, sequential mode for VRAM safety)
    for layer in model.layers:
        layer.cms.enable_learning(True, levels=[0])
    model.enable_drift(True)
    torch.cuda.reset_peak_memory_stats()

    # Load conversation data
    with open(args.data, encoding="utf-8") as f:
        all_convos = [json.loads(line) for line in f if line.strip()]
    replay_convos = all_convos[:args.num_replay]
    print(f"\n[7] Replaying {len(replay_convos)} conversations (level-0 CMS learning)...")

    cms_before = snapshot_cms_state(model)
    signal_trajectory = []
    t0 = time.time()

    for i, convo in enumerate(replay_convos):
        text = f"<|im_start|>user\n{convo['input']}<|im_end|>\n<|im_start|>assistant\n{convo['output']}<|im_end|>"
        tokens = tokenizer(text, max_length=128, truncation=True, return_tensors="pt")
        ids = tokens["input_ids"].to(device)

        with torch.no_grad():
            model(ids)

        # VRAM guard
        vram_gb = torch.cuda.memory_allocated() / 1e9
        if vram_gb > 16.0:
            print(f"  WARNING: VRAM at {vram_gb:.1f}GB, stopping replay early")
            break

        signal = convo.get("signal_health", 0.5)
        signal_trajectory.append(signal)

        if (i + 1) % 10 == 0:
            cms_now = snapshot_cms_state(model)
            delta = compute_cms_delta(cms_before, cms_now)
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(replay_convos)}] CMS delta={delta['total_l2']:.4f}, "
                  f"signal={signal:.3f}, {elapsed:.0f}s")

    replay_time = time.time() - t0
    cms_after = snapshot_cms_state(model)
    delta = compute_cms_delta(cms_before, cms_after)
    results["replay_conversations"] = len(replay_convos)
    results["replay_time_s"] = round(replay_time, 1)
    results["cms_delta_total"] = round(delta["total_l2"], 6)
    results["cms_delta_per_level"] = {k: round(v, 6) for k, v in delta["per_level"].items()}
    results["cms_max_param_delta"] = round(delta["max_param_delta"], 8)
    print(f"\n  Total CMS delta: {delta['total_l2']:.6f}")
    for level, d in delta["per_level"].items():
        print(f"    {level}: {d:.6f}")
    print(f"  Max param delta: {delta['max_param_delta']:.8f}")
    print(f"  Replay time: {replay_time:.1f}s ({replay_time/len(replay_convos):.1f}s/convo)")

    # Post-learning PPL
    for layer in model.layers:
        layer.cms.enable_learning(False)
    print("\n[8] Measuring post-learning perplexity...")
    post_ppl = measure_perplexity_texts(model, tokenizer, test_strings, device)
    results["post_learning_perplexity"] = round(post_ppl, 2)
    results["ppl_improvement"] = round(anamnesis_ppl - post_ppl, 2)
    print(f"  Post-learning PPL: {post_ppl:.2f} (improvement: {anamnesis_ppl - post_ppl:+.2f})")

    # =====================================================================
    # PART 3: GENERATION QUALITY
    # =====================================================================
    print("\n\n--- PART 3: GENERATION QUALITY ---")

    prompts = [
        "Tell me about yourself.",
        "What is the meaning of life?",
        "Explain how neural networks learn.",
    ]

    print("[9] Post-learning generation samples...")
    generations = {}
    for prompt in prompts:
        gen = generate_text(model, tokenizer, prompt, device, max_tokens=80)
        generations[prompt] = gen
        print(f"\n  Q: {prompt}")
        print(f"  A: {gen[:200]}")

    results["post_learning_generations"] = generations

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print(f"\n\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"  Model:              {args.model}")
    print(f"  Params:             {src_params:,} -> {anamnesis_params:,} ({results['param_ratio']}x)")
    print(f"  VRAM:               {results['vram_gb']} GB")
    print(f"  Source PPL:         {src_ppl:.2f}")
    print(f"  Anamnesis PPL:      {anamnesis_ppl:.2f} (delta: {anamnesis_ppl - src_ppl:+.2f})")
    print(f"  Post-learning PPL:  {post_ppl:.2f} (improvement: {anamnesis_ppl - post_ppl:+.2f})")
    print(f"  CMS delta:          {delta['total_l2']:.6f}")
    print(f"  Replay:             {len(replay_convos)} convos in {replay_time:.1f}s")
    print(f"  Peak VRAM:          {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    print(f"{'='*70}")

    # Save
    if args.save_state:
        save_cms_state(model, args.save_state, {
            "conversations": len(replay_convos),
            "baseline_ppl": anamnesis_ppl,
            "post_ppl": post_ppl,
        })
        print(f"\n  CMS state: {args.save_state}")

    os.makedirs(Path(args.output).parent, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results: {args.output}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
