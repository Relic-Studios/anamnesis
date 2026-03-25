#!/usr/bin/env python3
"""
Deploy Thomas on Anamnesis — continual learning through conversation.

No separate training phase. The model learns by running conversations
through it. CMS levels update at their scheduled frequencies during
the forward pass. The more Thomas talks, the more he remembers.

VRAM budget on RTX 4090 (24GB):
    Model weights:    17.6GB
    CMS inner-loop:    0.3GB (local gradients, not full backprop)
    KV cache:          1-2GB
    Total:           ~20GB  (fits with headroom)

Usage:
    python deploy_thomas.py
    python deploy_thomas.py --interactive
    python deploy_thomas.py --replay data/thomas_training.jsonl
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def load_anamnesis_model(layers_dir: str, device: str = "cuda"):
    """Load the converted Anamnesis model from safetensors layers."""
    from anamnesis.core.model import HopeModel, HopeConfig
    from safetensors.torch import load_file
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    r = cfg.intermediate_size / cfg.hidden_size

    config = HopeConfig(
        vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        max_position_embeddings=cfg.max_position_embeddings,
        rope_theta=cfg.rope_theta, rms_norm_eps=cfg.rms_norm_eps,
        cms_levels=3, cms_chunk_sizes=[1, 32, 256], cms_variant="nested",
        cms_hidden_mult=[r, r / 2, r / 4],
        use_neural_memory=False, tie_word_embeddings=False,
    )

    print("Loading Anamnesis model...")
    model = HopeModel(config)

    # Load embeddings
    emb = load_file(f"{layers_dir}/embeddings.safetensors")
    model.embed_tokens.weight.data.copy_(emb["embed_tokens.weight"])
    model.norm.weight.data.copy_(emb["norm.weight"])
    model.lm_head.weight.data.copy_(emb["lm_head.weight"])
    del emb

    # Load layers
    for i in range(cfg.num_hidden_layers):
        ld = load_file(f"{layers_dir}/layer_{i:02d}.safetensors")
        for key, val in ld.items():
            param_name = key.replace(f"layers.{i}.", "")
            parts = param_name.split(".")
            obj = model.layers[i]
            for part in parts[:-1]:
                obj = getattr(obj, part)
            getattr(obj, parts[-1]).data.copy_(val)
        del ld

    model = model.to(device, dtype=torch.bfloat16)
    model.eval()  # Eval mode — CMS still learns via inner loop, not backprop

    # Enable neutral drift on slow levels
    model.enable_drift(True)

    vram = torch.cuda.memory_allocated() / 1e9 if device == "cuda" else 0
    print(f"  Loaded: {model.num_parameters(False):,} params | {vram:.1f}GB VRAM")
    return model, config


def generate(model, tokenizer, prompt: str, max_tokens: int = 200, temperature: float = 0.7):
    """Generate a response from the Anamnesis model."""
    full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    ids = tokenizer(full_prompt, return_tensors="pt")["input_ids"].to(
        next(model.parameters()).device
    )

    generated = []
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(ids)["logits"][:, -1, :]

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_tok = torch.multinomial(probs, 1)
            else:
                next_tok = logits.argmax(dim=-1, keepdim=True)

            tok_id = next_tok.item()
            if tok_id in [tokenizer.eos_token_id, 151645, 151643]:  # eos, im_end, im_start
                break

            generated.append(tok_id)
            ids = next_tok

    return tokenizer.decode(generated, skip_special_tokens=True)


def replay_conversations(model, tokenizer, jsonl_path: str, gardener=None, max_convos: int = 0):
    """
    Replay past conversations through the model.

    This is how Thomas learns — not through backprop, but by processing
    his own conversation history. CMS levels update during the forward
    pass at their scheduled frequencies. Each conversation makes the
    model slightly better at being Thomas.

    Args:
        model: Anamnesis model (in eval mode — inner loop still active).
        tokenizer: Tokenizer.
        jsonl_path: Path to signal-annotated conversation JSONL.
        gardener: Optional GardenerStream for monitoring.
        max_convos: Max conversations to replay (0 = all).
    """
    with open(jsonl_path, "r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]

    if max_convos > 0:
        examples = examples[:max_convos]

    print(f"\nReplaying {len(examples)} conversations through CMS...")
    device = next(model.parameters()).device
    signals = []

    for i, ex in enumerate(examples):
        # Format as a complete conversation turn
        text = (
            f"<|im_start|>user\n{ex['input']}<|im_end|>\n"
            f"<|im_start|>assistant\n{ex['output']}<|im_end|>"
        )
        tokens = tokenizer(
            text, max_length=512, truncation=True, return_tensors="pt"
        )
        ids = tokens["input_ids"].to(device)

        # Forward pass — CMS levels update during this
        with torch.no_grad():
            output = model(ids)

        signal = ex.get("signal_health", 0.5)
        signals.append(signal)

        # Gardener monitoring
        if gardener and (i + 1) % 50 == 0:
            hidden = model.embed_tokens(ids[:, :32])
            gard_out = gardener.evaluate(hidden, real_signal=signal)
            print(
                f"  [{i+1}/{len(examples)}] "
                f"signal={signal:.3f} | "
                f"gardener_gate={gard_out.plasticity_gate:.3f} | "
                f"dream={gard_out.should_dream}"
            )
            if gard_out.should_dream:
                print("    Gardener suggests dreaming — run dream cycle offline")
                gardener.acknowledge_dream()

        elif (i + 1) % 100 == 0:
            avg_sig = sum(signals[-100:]) / min(100, len(signals))
            print(f"  [{i+1}/{len(examples)}] avg_signal={avg_sig:.3f}")

    avg_signal = sum(signals) / len(signals) if signals else 0
    print(f"\nReplay complete. {len(examples)} conversations processed.")
    print(f"  Average signal: {avg_signal:.3f}")
    return signals


def interactive_mode(model, tokenizer, gardener=None):
    """Interactive chat with Thomas on Anamnesis."""
    print("\n" + "=" * 50)
    print("Thomas on Anamnesis — Interactive Mode")
    print("Type 'quit' to exit, 'save' to save CMS state")
    print("=" * 50 + "\n")

    turn = 0
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "save":
            from anamnesis.state import save_cms_state
            path = f"data/thomas/cms_interactive_{int(time.time())}.pt"
            save_cms_state(model, path, {"turns": turn})
            print(f"  Saved CMS state to {path}")
            continue

        turn += 1
        response = generate(model, tokenizer, user_input)
        print(f"Thomas: {response}\n")

        # Gardener check every 5 turns
        if gardener and turn % 5 == 0:
            device = next(model.parameters()).device
            dummy_ids = tokenizer(response[:100], return_tensors="pt")["input_ids"].to(device)
            with torch.no_grad():
                hidden = model.embed_tokens(dummy_ids)
            gard_out = gardener.evaluate(hidden)
            print(f"  [gardener] signal={gard_out.signal_estimate:.3f} gate={gard_out.plasticity_gate:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Deploy Thomas on Anamnesis")
    parser.add_argument("--layers-dir", default="data/anamnesis_layers",
                        help="Directory with safetensors layer files")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    parser.add_argument("--replay", default="", help="Replay conversations from JSONL")
    parser.add_argument("--max-convos", type=int, default=0, help="Max conversations to replay")
    parser.add_argument("--save-state", default="", help="Save CMS state after replay")
    parser.add_argument("--load-state", default="", help="Load CMS state before starting")
    parser.add_argument("--no-gardener", action="store_true", help="Disable gardener monitoring")
    args = parser.parse_args()

    # Load model
    model, config = load_anamnesis_model(args.layers_dir, args.device)

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    # Load CMS state if provided
    if args.load_state:
        from anamnesis.state import load_cms_state
        meta = load_cms_state(model, args.load_state)
        print(f"  Loaded CMS state: {meta}")

    # Setup gardener
    gardener = None
    if not args.no_gardener:
        from anamnesis.active_inference import GardenerStream
        gardener = GardenerStream(dim=config.hidden_size, num_levels=config.cms_levels)

    # Replay mode: feed past conversations through CMS
    if args.replay:
        signals = replay_conversations(
            model, tokenizer, args.replay, gardener, args.max_convos,
        )

    # Save state after replay
    if args.save_state:
        from anamnesis.state import save_cms_state
        os.makedirs(os.path.dirname(args.save_state) or ".", exist_ok=True)
        save_cms_state(model, args.save_state, {
            "replay_convos": args.max_convos or "all",
            "replay_source": args.replay,
        })
        print(f"  Saved CMS state to {args.save_state}")

    # Interactive mode
    if args.interactive:
        interactive_mode(model, tokenizer, gardener)
    elif not args.replay:
        # Default: quick generation test
        print("\nGeneration test:")
        response = generate(model, tokenizer, "Hey Thomas, how are you doing today?")
        print(f"  Thomas: {response}")

    if args.device == "cuda":
        print(f"\nPeak VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")


if __name__ == "__main__":
    main()
