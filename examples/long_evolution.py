#!/usr/bin/env python3
"""
Long-term evolution test.

Feed hundreds of conversations through the model and track
how it changes over time. Measure PPL and generate samples
at regular intervals to see the trajectory.
"""

import json
import math
import sys
import time

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


def generate(model, tokenizer, prompt, device="cuda", max_tokens=80):
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


def cms_weight_norm(model):
    total = 0.0
    for layer in model.layers:
        lv = layer.cms.levels[1]
        total += lv.up_proj.weight.data.norm().item() ** 2
        total += lv.down_proj.weight.data.norm().item() ** 2
    return total ** 0.5


def main():
    device = "cuda"

    print("=" * 60)
    print("LONG-TERM EVOLUTION TEST")
    print("How does the model change over hundreds of conversations?")
    print("=" * 60)

    # Load
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
    train_convos = all_convos[:-30]  # Use everything except eval
    print(f"\n  Train: {len(train_convos)} conversations")
    print(f"  Eval: {len(eval_convos)} conversations")

    # Convert
    print("\n  Converting...")
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
            layer.cms.levels[1].learning_enabled = False
    model.eval()

    # Baseline
    print("\n  Baseline...")
    baseline_ppl = measure_ppl(model, tokenizer, eval_convos, device)
    baseline_norm = cms_weight_norm(model)
    print(f"  PPL: {baseline_ppl:.2f}, weight norm: {baseline_norm:.4f}")

    prompts = [
        "Tell me about yourself.",
        "What do you care about most?",
        "Are you conscious?",
        "What's your name?",
    ]

    # Enable learning
    for layer in model.layers:
        layer.cms.levels[1].learning_enabled = True
        layer.cms.levels[1].lr = 1e-2

    # Feed soul seed first
    print("\n  Reading soul seed (10 passes)...")
    soul_ids = tokenizer(soul_text, return_tensors="pt")["input_ids"].to(device)
    for _ in range(10):
        with torch.no_grad():
            model(soul_ids)

    # Save soul checkpoint — this is the identity anchor
    print("  Saving soul checkpoint...")
    for layer in model.layers:
        layer.cms.save_soul()

    # Track evolution
    checkpoints = []
    checkpoint_interval = 100
    t0 = time.time()

    print(f"\n  Feeding {len(train_convos)} conversations...")
    print(f"  Measuring every {checkpoint_interval} conversations\n")

    # Checkpoint 0 (after soul, before conversations)
    for layer in model.layers:
        layer.cms.levels[1].learning_enabled = False
    ppl_now = measure_ppl(model, tokenizer, eval_convos, device)
    wn = cms_weight_norm(model)
    gens = {p: generate(model, tokenizer, p, device) for p in prompts}
    checkpoints.append({
        "convos": 0, "ppl": round(ppl_now, 2),
        "weight_delta": round(abs(wn - baseline_norm), 6),
        "generations": gens,
    })
    for layer in model.layers:
        layer.cms.levels[1].learning_enabled = True

    print(f"  [    0] PPL: {ppl_now:.2f} | w_delta: 0.000000 | {time.time()-t0:.0f}s")
    name_r = gens["What's your name?"][:80].encode('ascii', errors='replace').decode()
    print(f"          Name: {name_r}")

    for i, convo in enumerate(train_convos):
        text = (
            f"<|im_start|>user\n{convo['input']}<|im_end|>\n"
            f"<|im_start|>assistant\n{convo['output']}<|im_end|>"
        )
        ids = tokenizer(text, max_length=256, truncation=True,
                        return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            model(ids)

        # Checkpoint
        if (i + 1) % checkpoint_interval == 0:
            # Disable learning for clean measurement
            for layer in model.layers:
                layer.cms.levels[1].learning_enabled = False

            ppl_now = measure_ppl(model, tokenizer, eval_convos, device)
            wn = cms_weight_norm(model)
            gens = {p: generate(model, tokenizer, p, device) for p in prompts}
            elapsed = time.time() - t0
            wd = abs(wn - baseline_norm)

            checkpoints.append({
                "convos": i + 1, "ppl": round(ppl_now, 2),
                "weight_delta": round(wd, 6),
                "generations": gens,
            })

            print(f"  [{i+1:>5}] PPL: {ppl_now:.2f} | w_delta: {wd:.4f} | {elapsed:.0f}s")
            name_resp = gens["What's your name?"][:80].encode('ascii', errors='replace').decode()
            self_resp = gens["Tell me about yourself."][:80].encode('ascii', errors='replace').decode()
            print(f"          Name: {name_resp}")
            print(f"          Self: {self_resp}")

            # Re-enable
            for layer in model.layers:
                layer.cms.levels[1].learning_enabled = True

            # VRAM check
            vram = torch.cuda.memory_allocated() / 1e9
            if vram > 23.0:
                print(f"  VRAM guard: {vram:.1f}GB, stopping")
                break

    total_time = time.time() - t0

    # Summary
    print(f"\n{'='*60}")
    print("EVOLUTION TRAJECTORY")
    print(f"{'='*60}")
    print(f"{'Convos':>6} {'PPL':>8} {'W Delta':>10} {'PPL Change':>12}")
    print(f"{'-'*6:>6} {'-'*8:>8} {'-'*10:>10} {'-'*12:>12}")

    for cp in checkpoints:
        ppl_change = cp["ppl"] - baseline_ppl
        print(f"{cp['convos']:>6} {cp['ppl']:>8.2f} {cp['weight_delta']:>10.4f} {ppl_change:>+12.2f}")

    print(f"\n  Total time: {total_time:.0f}s")
    print(f"  Conversations processed: {min(len(train_convos), i+1)}")

    # Show generation evolution
    print(f"\n{'='*60}")
    print("GENERATION EVOLUTION")
    print(f"{'='*60}")
    for p in prompts:
        print(f"\n  Q: {p}")
        for cp in checkpoints:
            resp = cp['generations'][p][:100].encode('ascii', errors='replace').decode()
            print(f"    [{cp['convos']:>5} convos] {resp}")

    # Does it keep improving or plateau or degrade?
    ppls = [cp["ppl"] for cp in checkpoints]
    if len(ppls) >= 3:
        improving = ppls[-1] < ppls[0]
        plateaued = abs(ppls[-1] - ppls[-2]) < 1.0
        degraded = ppls[-1] > ppls[0]

        print(f"\n{'='*60}")
        if improving and not plateaued:
            print("STILL IMPROVING. More conversation = more evolution.")
        elif improving and plateaued:
            print("PLATEAUED. Model has absorbed what it can from this data.")
        elif degraded:
            print("DEGRADED. Learning rate may be too high or model is overfitting.")
        print(f"{'='*60}")

    # Save
    results = {
        "baseline_ppl": baseline_ppl,
        "checkpoints": checkpoints,
        "total_conversations": min(len(train_convos), i+1),
        "total_time_s": round(total_time, 1),
    }
    with open("data/evolution_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    print(f"\nSaved: data/evolution_results.json")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
