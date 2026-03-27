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
    """Mean base gate value (static sigmoid of residual_gate parameter)."""
    return sum(
        torch.sigmoid(l.cms.levels[1].residual_gate).item()
        for l in model.layers
    ) / len(model.layers)


def effective_gate_mean(model):
    """Mean effective gate value (surprise-modulated, what actually runs during inference)."""
    total = 0.0
    with torch.no_grad():
        for l in model.layers:
            lv = l.cms.levels[1]
            total += lv._compute_gate().item()
    return total / len(model.layers)


def surprise_mean(model):
    """Mean surprise EMA across all layers, both levels."""
    vals = []
    for l in model.layers:
        for lv in l.cms.levels:
            vals.append(lv._surprise_ema)
    return sum(vals) / len(vals) if vals else 0.0


def _level_params(lv):
    """Get named parameters for a CMS level (handles both full and low-rank)."""
    if hasattr(lv, 'A'):  # LowRankLevel
        return [("A", lv.A.weight), ("B", lv.B.weight)]
    else:  # CMSLevel
        params = [("up", lv.up_proj.weight), ("down", lv.down_proj.weight)]
        if hasattr(lv, 'gate_proj'):
            params.append(("gate", lv.gate_proj.weight))
        return params


def weight_delta(model, snapshots):
    """Compute L2 norm of weight change since snapshot across all layers, both levels."""
    totals = {"lv0": 0.0, "lv1": 0.0}
    counts = {"lv0": 0, "lv1": 0}
    for i, layer in enumerate(model.layers):
        for lvl_idx, lv in enumerate(layer.cms.levels):
            tag = f"lv{lvl_idx}"
            for pname, param in _level_params(lv):
                snap_key = f"layer_{i}_{tag}_{pname}"
                if snap_key in snapshots:
                    totals[tag] += (param.data.float() - snapshots[snap_key]).norm().item()
                    counts[tag] += 1
    lv0 = totals["lv0"] / max(counts["lv0"], 1)
    lv1 = totals["lv1"] / max(counts["lv1"], 1)
    return lv0, lv1


def snapshot_weights(model):
    """Snapshot both level weights for delta tracking."""
    snaps = {}
    for i, layer in enumerate(model.layers):
        for lvl_idx, lv in enumerate(layer.cms.levels):
            tag = f"lv{lvl_idx}"
            for pname, param in _level_params(lv):
                snaps[f"layer_{i}_{tag}_{pname}"] = param.data.float().clone()
    return snaps


def persona_mask(token_ids, im_start_id=151644):
    """Create a mask weighting assistant tokens 1.0, user tokens 0.1.

    In the Qwen chat format:
        <|im_start|>user\\n...user...<|im_end|>\\n<|im_start|>assistant\\n...assistant...<|im_end|>

    The second <|im_start|> marks the assistant turn. Everything from there
    onward is persona-relevant and gets full learning weight.
    """
    batch, seq_len = token_ids.shape
    mask = torch.full((batch, seq_len), 0.1, device=token_ids.device)
    for b in range(batch):
        positions = (token_ids[b] == im_start_id).nonzero(as_tuple=True)[0]
        if len(positions) >= 2:
            mask[b, positions[-1]:] = 1.0
        elif len(positions) == 1:
            mask[b] = 1.0
    return mask


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
    """Save Level 1 (low-rank) weights only — Level 0 is frozen.

    At rank=32 for a 3B model, each save is ~15MB. This is the "task vector"
    that encodes everything this instance learned from its environment.
    """
    state = {}
    for i, layer in enumerate(model.layers):
        lv = layer.cms.levels[1]
        lvl_state = {
            "A": lv.A.weight.data.cpu(),
            "B": lv.B.weight.data.cpu(),
            "residual_gate": lv.residual_gate.data.cpu(),
            "total_updates": lv._total_updates,
            "surprise_ema": lv._surprise_ema,
        }
        for wname in ["A.weight", "B.weight"]:
            if wname in lv._master_weights:
                lvl_state[f"master_{wname}"] = lv._master_weights[wname].cpu()
            if wname in lv._soul_weights:
                lvl_state[f"soul_{wname}"] = lv._soul_weights[wname].cpu()
        state[f"layer_{i}"] = lvl_state
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
    print("FULL PIPELINE: Convert -> Evolve -> Persist")
    print("  No seeding phase. L1 starts at zero-output, gate at 0.5.")
    print("  Competence-based gating: L1 earns its contribution.")
    print("  The environment compiles the identity.")
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
    print("\n[1] Converting to 2-level Anamnesis (low-rank L1, rank=32)...")
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
        cms_hidden_mult=[r, r],  # L1 reuses L0's hidden dim for feature extraction
        cms_rank=32,  # Low-rank L1: ~15MB per specialist
        use_neural_memory=False,
        tie_word_embeddings=False,
    )
    model = model_to_hope(src_model, hope_config, verbose=False)
    del src_model
    torch.cuda.empty_cache()
    model = model.to(device, dtype=torch.bfloat16)

    # L1 starts with zero-output init (down_proj = 0) and gate at sigmoid(0) = 0.5.
    # Competence-based gating means the gate will close initially (L1 is confused)
    # and open naturally as L1 learns useful representations from conversation.
    # No seeding phase needed.

    # ── Baseline ──
    print("\n[1] Baseline (pre-evolution)...")
    model.eval()
    # Ensure no predictive coding runs during baseline measurement
    for layer in model.layers:
        layer.cms.enable_learning(False)
    baseline_ppl = measure_ppl(model, tokenizer, eval_convos, device)
    baseline_gens = sample_all(model, tokenizer, prompts, device)
    print(f"  PPL: {baseline_ppl:.2f} | Gate: {gate_mean(model):.4f}")
    print(f"  Generations:")
    print_gens(baseline_gens)

    # Save initial soul checkpoint — the undifferentiated state.
    # This is what the model returns to if it drifts too far.
    print(f"\n  Saving soul checkpoint (undifferentiated state)...")
    for layer in model.layers:
        layer.cms.save_soul()
    save_cms_evolution_state(model, output_dir / "soul_checkpoint.pt")

    # Set up persona probes — focus predictive coding on output-relevant directions
    print(f"\n  Setting up persona probes (SVD of LM head)...")
    model.setup_persona_probes(persona_dim=256, num_final_layers=4)
    print(f"  Persona probes active on final 4 layers (256 principal directions)")

    # ══════════════════════════════════════════════════════════
    # EVOLVE THROUGH CONVERSATION (predictive coding)
    # No seeding. The environment compiles the identity.
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"[2] Evolving through {len(train_convos)} conversations")
    print(f"    Competence gate: opens as L1 proves accurate, closes when confused")
    print(f"    Persona probe: error focused on token-selection directions")
    print(f"{'='*60}")

    for layer in model.layers:
        # Level 0: frozen. Pre-trained MLP provides the base intelligence.
        layer.cms.levels[0].learning_enabled = False

        # Level 1: the adaptation layer — learns via predictive coding.
        # Starts at zero output. Gate opens as L1 earns competence.
        layer.cms.levels[1].learning_enabled = True
        layer.cms.levels[1].lr = 1e-2

    checkpoints = []
    t0 = time.time()
    pre_evolve_snap = snapshot_weights(model)

    for i, convo in enumerate(train_convos):
        text = (
            f"<|im_start|>user\n{convo['input']}<|im_end|>\n"
            f"<|im_start|>assistant\n{convo['output']}<|im_end|>"
        )
        ids = tokenizer(text, max_length=256, truncation=True,
                        return_tensors="pt")["input_ids"].to(device)
        # Persona mask: learn 10x more from assistant responses than user prompts
        model.set_learning_weight(persona_mask(ids))
        with torch.no_grad():
            model(ids)
        model.set_learning_weight(None)

        if (i + 1) % 100 == 0:
            # Disable learning for measurement
            for layer in model.layers:
                layer.cms.levels[1].learning_enabled = False

            ppl_now = measure_ppl(model, tokenizer, eval_convos, device)
            gens = sample_all(model, tokenizer, prompts, device)
            dw0, dw1 = weight_delta(model, pre_evolve_snap)
            elapsed = time.time() - t0

            eff_gate = effective_gate_mean(model)
            surp = surprise_mean(model)
            checkpoints.append({
                "convos": i + 1, "ppl": round(ppl_now, 2),
                "gate_base": round(gate_mean(model), 4),
                "gate_effective": round(eff_gate, 4),
                "surprise_ema": round(surp, 4),
                "weight_delta_lv0": round(dw0, 6),
                "weight_delta_lv1": round(dw1, 6),
                "generations": gens,
            })

            name_resp = gens["What's your name?"][:80].encode('ascii', errors='replace').decode()
            self_resp = gens["Tell me about yourself."][:80].encode('ascii', errors='replace').decode()
            print(f"  [{i+1:>4}] PPL: {ppl_now:.2f} | Gate(eff): {eff_gate:.4f} | Surp: {surp:.4f} | dw0: {dw0:.4f} dw1: {dw1:.4f} | {elapsed:.0f}s")
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
    # PERSIST
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("[3] Saving evolved state")
    print(f"{'='*60}")
    save_cms_evolution_state(model, output_dir / "evolved_state.pt")
    print(f"  Saved: {output_dir / 'evolved_state.pt'}")

    # ══════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PIPELINE RESULTS (no seeding — pure evolution)")
    print(f"{'='*60}")
    final_dw0, final_dw1 = weight_delta(model, pre_evolve_snap)
    final_eff_gate = effective_gate_mean(model)
    final_surprise = surprise_mean(model)
    print(f"  Baseline PPL:     {baseline_ppl:.2f}")
    print(f"  Post-evolve PPL:  {final_ppl:.2f} (delta: {final_ppl - baseline_ppl:+.2f})")
    print(f"  Gate (base):      {gate_mean(model):.4f}")
    print(f"  Gate (effective): {final_eff_gate:.4f}")
    print(f"  Surprise EMA:     {final_surprise:.4f}")
    print(f"  dw level 0 (MLP): {final_dw0:.6f}")
    print(f"  dw level 1 (res): {final_dw1:.6f}")

    print(f"\n  GENERATION EVOLUTION:")
    for p in prompts:
        print(f"\n    Q: {p}")
        b = baseline_gens[p][:100].encode('ascii', errors='replace').decode()
        f_ = final_gens[p][:100].encode('ascii', errors='replace').decode()
        print(f"    Baseline:  {b}")
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
        "final_ppl": final_ppl,
        "gate_base_final": gate_mean(model),
        "gate_effective_final": final_eff_gate,
        "surprise_ema_final": final_surprise,
        "persona_probe": {"dim": 256, "num_layers": 4},
        "competence_gate": {"scale": model.layers[0].cms.levels[1].gate_surprise_scale},
        "baseline_generations": baseline_gens,
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
