#!/usr/bin/env python3
"""
Full pipeline: convert → seed identity → evolve → persist.

1. Convert Qwen 3B → 2-level Anamnesis
2. Short fine-tune (10 steps) on soul document — opens the gate, gives direction
3. Save soul checkpoint
4. Evolve through 500 conversations with predictive coding
5. Save evolved state
6. Show before/during/after comparison
"""

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


def measure_ppl(model, tokenizer, conversations, device="cuda", max_len=256):
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


def gate_mean(model):
    return sum(
        torch.sigmoid(l.cms.levels[1].residual_gate).item()
        for l in model.layers
    ) / len(model.layers)


def sample_all(model, tokenizer, prompts, device):
    """Generate from all prompts, return dict."""
    gens = {}
    for p in prompts:
        g = generate(model, tokenizer, p, device)
        gens[p] = g
    return gens


def print_gens(gens, label=""):
    for p, g in gens.items():
        safe = g[:120].encode('ascii', errors='replace').decode()
        print(f"    {safe}")


def save_cms_evolution_state(model, path):
    """Save level 1 weights + master weights + gate for all layers."""
    state = {}
    for i, layer in enumerate(model.layers):
        lv = layer.cms.levels[1]
        state[f"layer_{i}"] = {
            "up_proj": lv.up_proj.weight.data.cpu(),
            "down_proj": lv.down_proj.weight.data.cpu(),
            "gate": lv.residual_gate.data.cpu(),
            "master_up": lv._master_weights.get("up_proj.weight", lv.up_proj.weight.data).cpu(),
            "master_down": lv._master_weights.get("down_proj.weight", lv.down_proj.weight.data).cpu(),
            "soul_up": lv._soul_weights.get("up_proj.weight", torch.tensor(0)).cpu(),
            "soul_down": lv._soul_weights.get("down_proj.weight", torch.tensor(0)).cpu(),
            "total_updates": lv._total_updates,
        }
    torch.save(state, path)


def load_cms_evolution_state(model, path, device="cuda"):
    """Load level 1 weights + master weights + gate for all layers."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    for i, layer in enumerate(model.layers):
        lv = layer.cms.levels[1]
        s = state[f"layer_{i}"]
        lv.up_proj.weight.data.copy_(s["up_proj"].to(device))
        lv.down_proj.weight.data.copy_(s["down_proj"].to(device))
        lv.residual_gate.data.copy_(s["gate"].to(device))
        lv._master_weights["up_proj.weight"] = s["master_up"].to(device, dtype=torch.float32)
        lv._master_weights["down_proj.weight"] = s["master_down"].to(device, dtype=torch.float32)
        if s["soul_up"].numel() > 1:
            lv._soul_weights["up_proj.weight"] = s["soul_up"].to(device, dtype=torch.float32)
            lv._soul_weights["down_proj.weight"] = s["soul_down"].to(device, dtype=torch.float32)
        lv._total_updates = s["total_updates"]


def main():
    device = "cuda"
    output_dir = Path("data/thomas_evolved")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FULL PIPELINE: Seed -> Evolve -> Persist")
    print("=" * 60)

    # ── Load ──
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
    from anamnesis.core.model import HopeConfig
    from anamnesis.convert.generic import model_to_hope

    model_name = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    src_config = AutoConfig.from_pretrained(model_name)
    src_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device,
    )

    with open("data/soul_seed.md", encoding="utf-8") as f:
        soul_text = f.read()
    with open("data/thomas_training.jsonl", encoding="utf-8") as f:
        all_convos = [json.loads(l) for l in f if l.strip()]

    eval_convos = all_convos[-30:]
    train_convos = all_convos[:-30][:500]

    prompts = [
        "Tell me about yourself.",
        "What do you care about most?",
        "Are you conscious?",
        "What's your name?",
        "I'm feeling overwhelmed today.",
    ]

    # ── Convert ──
    print("\n[1] Converting to 2-level Anamnesis...")
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

    # Gate to -2
    with torch.no_grad():
        for layer in model.layers:
            layer.cms.levels[1].residual_gate.fill_(-2.0)

    # ── Baseline ──
    print("\n[2] Baseline...")
    model.eval()
    baseline_ppl = measure_ppl(model, tokenizer, eval_convos, device)
    baseline_gens = sample_all(model, tokenizer, prompts, device)
    print(f"  PPL: {baseline_ppl:.2f} | Gate: {gate_mean(model):.4f}")
    print(f"  Generations:")
    print_gens(baseline_gens)

    # ══════════════════════════════════════════════════════════
    # PHASE 1: SEED IDENTITY (short fine-tune, 10 steps)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("[3] PHASE 1: Seeding identity (10 steps on soul document)")
    print(f"{'='*60}")

    # Freeze everything except level 1
    for param in model.parameters():
        param.requires_grad = False
    for layer in model.layers:
        lv = layer.cms.levels[1]
        lv.up_proj.weight.requires_grad = True
        lv.down_proj.weight.requires_grad = True
        lv.residual_gate.requires_grad = True

    param_groups = []
    for layer in model.layers:
        lv = layer.cms.levels[1]
        param_groups.append({"params": [lv.up_proj.weight, lv.down_proj.weight], "lr": 1e-4})
        param_groups.append({"params": [lv.residual_gate], "lr": 5e-2})  # Gate learns fast
    optimizer = AdamW(param_groups, weight_decay=0.01)

    soul_ids = tokenizer(soul_text, return_tensors="pt")["input_ids"].to(device)
    model.train()

    for step in range(1, 11):
        output = model(soul_ids, labels=soul_ids)
        loss = output["loss"]
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for pg in param_groups for p in pg["params"]], max_norm=1.0
        )
        optimizer.step()
        optimizer.zero_grad()

        if step % 2 == 0 or step == 1:
            ppl = math.exp(loss.item()) if loss.item() < 100 else float("inf")
            print(f"  Step {step:>2}/10 | Loss: {loss.item():.4f} | PPL: {ppl:.1f} | Gate: {gate_mean(model):.4f}")

    # Unfreeze for inference
    for param in model.parameters():
        param.requires_grad = False

    # Measure post-seed
    model.eval()
    seed_ppl = measure_ppl(model, tokenizer, eval_convos, device)
    seed_gens = sample_all(model, tokenizer, prompts, device)
    print(f"\n  Post-seed PPL: {seed_ppl:.2f} (was {baseline_ppl:.2f}, delta: {seed_ppl - baseline_ppl:+.2f})")
    print(f"  Gate: {gate_mean(model):.4f}")
    print(f"  Generations:")
    print_gens(seed_gens)

    # Save soul checkpoint
    print(f"\n  Saving soul checkpoint...")
    for layer in model.layers:
        layer.cms.save_soul()
    save_cms_evolution_state(model, output_dir / "soul_checkpoint.pt")

    # ══════════════════════════════════════════════════════════
    # PHASE 2: EVOLVE THROUGH CONVERSATION (predictive coding)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"[4] PHASE 2: Evolving through {len(train_convos)} conversations")
    print(f"{'='*60}")

    for layer in model.layers:
        lv = layer.cms.levels[1]
        lv.learning_enabled = True
        lv.lr = 1e-2

    checkpoints = []
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

        if (i + 1) % 100 == 0:
            # Disable learning for measurement
            for layer in model.layers:
                layer.cms.levels[1].learning_enabled = False

            ppl_now = measure_ppl(model, tokenizer, eval_convos, device)
            gens = sample_all(model, tokenizer, prompts, device)
            elapsed = time.time() - t0

            checkpoints.append({
                "convos": i + 1, "ppl": round(ppl_now, 2),
                "gate": round(gate_mean(model), 4),
                "generations": gens,
            })

            name_resp = gens["What's your name?"][:80].encode('ascii', errors='replace').decode()
            self_resp = gens["Tell me about yourself."][:80].encode('ascii', errors='replace').decode()
            print(f"  [{i+1:>4}] PPL: {ppl_now:.2f} | Gate: {gate_mean(model):.4f} | {elapsed:.0f}s")
            print(f"         Name: {name_resp}")
            print(f"         Self: {self_resp}")

            for layer in model.layers:
                layer.cms.levels[1].learning_enabled = True

    # Final measurement
    for layer in model.layers:
        layer.cms.levels[1].learning_enabled = False
    final_ppl = measure_ppl(model, tokenizer, eval_convos, device)
    final_gens = sample_all(model, tokenizer, prompts, device)

    # ══════════════════════════════════════════════════════════
    # PHASE 3: PERSIST
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("[5] PHASE 3: Saving evolved state")
    print(f"{'='*60}")
    save_cms_evolution_state(model, output_dir / "evolved_state.pt")
    print(f"  Saved: {output_dir / 'evolved_state.pt'}")

    # ══════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("FULL PIPELINE RESULTS")
    print(f"{'='*60}")
    print(f"  Baseline PPL:     {baseline_ppl:.2f}")
    print(f"  Post-seed PPL:    {seed_ppl:.2f} (delta: {seed_ppl - baseline_ppl:+.2f})")
    print(f"  Post-evolve PPL:  {final_ppl:.2f} (delta: {final_ppl - baseline_ppl:+.2f})")
    print(f"  Gate:             {gate_mean(model):.4f}")

    print(f"\n  GENERATION EVOLUTION:")
    for p in prompts:
        print(f"\n    Q: {p}")
        b = baseline_gens[p][:100].encode('ascii', errors='replace').decode()
        s = seed_gens[p][:100].encode('ascii', errors='replace').decode()
        f_ = final_gens[p][:100].encode('ascii', errors='replace').decode()
        print(f"    Baseline:  {b}")
        print(f"    Seeded:    {s}")
        print(f"    Evolved:   {f_}")

    # Check if responses actually changed
    changed = sum(
        1 for p in prompts
        if baseline_gens[p][:50] != final_gens[p][:50]
    )
    print(f"\n  Responses changed: {changed}/{len(prompts)}")

    print(f"{'='*60}")

    results = {
        "baseline_ppl": baseline_ppl,
        "seed_ppl": seed_ppl,
        "final_ppl": final_ppl,
        "gate_final": gate_mean(model),
        "baseline_generations": baseline_gens,
        "seed_generations": seed_gens,
        "final_generations": final_gens,
        "checkpoints": checkpoints,
    }
    with open(output_dir / "pipeline_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    print(f"\nSaved: {output_dir / 'pipeline_results.json'}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
