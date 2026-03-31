#!/usr/bin/env python3
"""
Merge trained DeepMemoryLevel weights back into the full Qwen 2.5 7B base.

Takes:
  - Qwen 2.5 7B from HuggingFace (frozen base weights)
  - step_25000.pt (trained DeepMemoryLevel params from scaffold training)

Produces:
  - A complete HopeModel checkpoint with ALL weights (base + trained memory)
  - Saved as safetensors for reliability (no ZIP64 bugs)

Strategy: Skip SVD initialization entirely since trained weights overwrite it.
Build HopeModel structure, copy base weights, overlay trained weights.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.stdout.reconfigure(line_buffering=True)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    parser = argparse.ArgumentParser(description="Merge trained memory weights into full model")
    parser.add_argument("--checkpoint", default="C:/Users/zappa/Downloads/step_25000.pt")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--output", default="C:/Dev/hope-didymus/checkpoints/anamnesis_7b_vessel.safetensors")
    parser.add_argument("--save-config", default="C:/Dev/hope-didymus/checkpoints/anamnesis_7b_vessel_config.json")
    args = parser.parse_args()

    t0 = time.time()

    # ── Load trained checkpoint ──
    print(f"[1/5] Loading trained checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    trained_state = ckpt["model_state"]
    config_dict = ckpt["config"]
    step = ckpt["step"]
    loss = ckpt["loss"]
    trained_total = sum(v.numel() for v in trained_state.values())
    print(f"  Step: {step} | Loss: {loss:.4f} | Trained: {trained_total/1e6:.1f}M params")

    # ── Load base model ──
    print(f"\n[2/5] Loading base model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoConfig
    from anamnesis.core.model import HopeModel, HopeConfig

    src_model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cpu",
    )
    src_params = sum(p.numel() for p in src_model.parameters())
    print(f"  Loaded: {src_params:,} params")

    # ── Build HopeModel (skip SVD — we have trained weights) ──
    print(f"\n[3/5] Building HopeModel structure (skipping SVD init)...")
    hope_config = HopeConfig(**config_dict)
    model = HopeModel(hope_config)

    # Get source internal model
    src = src_model.model if hasattr(src_model, "model") else src_model

    with torch.no_grad():
        # Copy embeddings
        model.embed_tokens.weight.copy_(src.embed_tokens.weight)
        print(f"  Copied embeddings: {model.embed_tokens.weight.shape}")

        # Copy final norm
        model.norm.weight.copy_(src.norm.weight)

        # Copy LM head
        model.lm_head.weight.copy_(src_model.lm_head.weight)
        print(f"  Copied lm_head: {model.lm_head.weight.shape}")

        # Copy per-layer base weights (attention + layernorms + L0 SwiGLU)
        src_layers = src.layers
        for i, (src_layer, tgt_block) in enumerate(zip(src_layers, model.layers)):
            if i % 7 == 0:
                print(f"  Copying layer {i}/{len(src_layers)}...")

            # Layer norms
            tgt_block.input_layernorm.weight.copy_(src_layer.input_layernorm.weight)
            tgt_block.post_attention_layernorm.weight.copy_(src_layer.post_attention_layernorm.weight)

            # Attention projections
            attn = src_layer.self_attn
            tgt_block.q_proj.weight.copy_(attn.q_proj.weight)
            if attn.q_proj.bias is not None:
                tgt_block.q_proj.bias.copy_(attn.q_proj.bias)
            tgt_block.k_proj.weight.copy_(attn.k_proj.weight)
            if attn.k_proj.bias is not None:
                tgt_block.k_proj.bias.copy_(attn.k_proj.bias)
            tgt_block.v_proj.weight.copy_(attn.v_proj.weight)
            if attn.v_proj.bias is not None:
                tgt_block.v_proj.bias.copy_(attn.v_proj.bias)
            tgt_block.o_proj.weight.copy_(attn.o_proj.weight)

            # L0 SwiGLU weights
            mlp = src_layer.mlp
            level0 = tgt_block.cms.levels[0]
            level0.gate_proj.weight.copy_(mlp.gate_proj.weight)
            level0.up_proj.weight.copy_(mlp.up_proj.weight)
            level0.down_proj.weight.copy_(mlp.down_proj.weight)

    # Free source model
    del src_model, src
    import gc; gc.collect()
    print(f"  Base weights copied, source model freed")

    # ── Overlay trained DeepMemoryLevel weights ──
    print(f"\n[4/5] Overlaying {len(trained_state)} trained tensors...")
    missing, unexpected = model.load_state_dict(trained_state, strict=False)
    print(f"  Missing (frozen base, expected): {len(missing)}")
    print(f"  Unexpected: {len(unexpected)}")
    if unexpected:
        print(f"  WARNING: {unexpected[:5]}")

    # Verify
    full_state = model.state_dict()
    test_key = list(trained_state.keys())[0]
    match = torch.equal(full_state[test_key], trained_state[test_key])
    print(f"  Verify {test_key}: {'MATCH' if match else 'MISMATCH!'}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total merged model: {total_params:,} params ({total_params/1e9:.2f}B)")

    # ── Save ──
    print(f"\n[5/5] Saving merged model...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from safetensors.torch import save_file
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(output_path))

    # Save config
    config_path = Path(args.save_config)
    config_out = {
        "architecture": "HopeModel",
        "base_model": args.model,
        "training_step": step,
        "training_loss": float(loss),
        "trained_params": trained_total,
        "total_params": total_params,
        "hope_config": config_dict,
    }
    with open(config_path, "w") as f:
        json.dump(config_out, f, indent=2)

    elapsed = time.time() - t0
    file_size = output_path.stat().st_size
    print(f"\n  Done in {elapsed:.0f}s")
    print(f"  Model: {output_path} ({file_size/1e9:.2f} GB)")
    print(f"  Config: {config_path}")
    print(f"  {args.model} + {trained_total/1e6:.0f}M trained DeepMemoryLevel weights merged")


if __name__ == "__main__":
    main()
