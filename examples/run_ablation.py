#!/usr/bin/env python3
"""
Run ablation study for Anamnesis active inference extensions.

Tests each of the 7 extensions independently and compares against
the full system and a baseline with all extensions disabled.

Usage:
    python examples/run_ablation.py                       # Quick (CPU, tiny model)
    python examples/run_ablation.py --steps 500           # Longer training
    python examples/run_ablation.py --data data/thomas_training.jsonl  # Real data
"""

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from anamnesis.core.model import HopeModel, HopeConfig
from anamnesis.evaluation.ablation import AblationRunner, ABLATION_CONFIGS


def make_random_dataloader(batch_size, num_batches, seq_len, vocab_size):
    """Create a dataloader with random token IDs and signal scores."""
    batches = []
    for _ in range(num_batches):
        batches.append({
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "signal_health": torch.rand(batch_size) * 0.5 + 0.3,  # 0.3-0.8 range
        })
    return DataLoader(batches, batch_size=None)


def make_jsonl_dataloader(path, batch_size, seq_len, vocab_size):
    """Create a dataloader from JSONL training data (token-level proxy)."""
    import hashlib

    batches = []
    with open(path) as f:
        lines = [json.loads(line) for line in f if line.strip()]

    # Simple tokenization: hash characters to vocab range
    for i in range(0, len(lines) - batch_size + 1, batch_size):
        batch_lines = lines[i:i + batch_size]
        ids = []
        signals = []
        for line in batch_lines:
            text = line.get("input", "") + " " + line.get("output", "")
            tokens = [ord(c) % vocab_size for c in text[:seq_len]]
            tokens += [0] * (seq_len - len(tokens))
            ids.append(tokens[:seq_len])
            signals.append(line.get("signal_health", 0.5))

        batches.append({
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "signal_health": torch.tensor(signals),
        })

    return DataLoader(batches, batch_size=None)


def main():
    parser = argparse.ArgumentParser(description="Anamnesis Ablation Study")
    parser.add_argument("--steps", type=int, default=100, help="Training steps per config")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence length")
    parser.add_argument("--data", type=str, default="", help="JSONL data path (optional)")
    parser.add_argument("--output", type=str, default="data/ablation_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    vocab_size = 256

    print("=" * 60)
    print("ANAMNESIS ABLATION STUDY")
    print("=" * 60)
    print(f"Model: {args.hidden}d, {args.layers}L, {vocab_size}V")
    print(f"Training: {args.steps} steps, batch {args.batch_size}, seq {args.seq_len}")
    print(f"Data: {'random' if not args.data else args.data}")

    # Model factory
    def model_factory():
        torch.manual_seed(args.seed)  # Same init for every ablation
        config = HopeConfig(
            vocab_size=vocab_size,
            hidden_size=args.hidden,
            num_hidden_layers=args.layers,
            num_attention_heads=4,
            num_kv_heads=2,
            cms_levels=3,
            cms_chunk_sizes=[1, 8, 32],
            cms_hidden_mult=4.0,
        )
        return HopeModel(config), config

    # Data
    num_batches = args.steps + 20  # Extra for eval
    if args.data and Path(args.data).exists():
        train_dl = make_jsonl_dataloader(args.data, args.batch_size, args.seq_len, vocab_size)
        eval_dl = make_random_dataloader(args.batch_size, 10, args.seq_len, vocab_size)
    else:
        train_dl = make_random_dataloader(args.batch_size, num_batches, args.seq_len, vocab_size)
        eval_dl = make_random_dataloader(args.batch_size, 10, args.seq_len, vocab_size)

    # Run ablations
    runner = AblationRunner(
        model_factory=model_factory,
        train_loader=train_dl,
        eval_loader=eval_dl,
        train_steps=args.steps,
    )

    results = runner.run_all(verbose=True)
    runner.print_table(results)

    # Save results
    os.makedirs(Path(args.output).parent, exist_ok=True)
    output = {r.config_name: r.to_dict() for r in results}
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
