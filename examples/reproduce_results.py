#!/usr/bin/env python3
"""
Reproduce all Anamnesis paper results.

A single script that validates the full system end-to-end:

1. Creates a small Anamnesis model
2. Runs the ablation study (Table 1: Extension Impact)
3. Tests CMS inner-loop learning (Table 2: Learning Dynamics)
4. Tests state persistence across sessions (Table 3: Continuity)
5. Outputs all results as JSON

Usage:
    python examples/reproduce_results.py             # Quick mode (~2 min CPU)
    python examples/reproduce_results.py --thorough  # Full mode (~10 min)
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from anamnesis.core.model import HopeModel, HopeConfig
from anamnesis.evaluation.metrics import (
    compute_perplexity,
    snapshot_cms_state,
    compute_cms_delta,
    compute_surprise_profile,
)
from anamnesis.evaluation.ablation import AblationRunner, ABLATION_CONFIGS
from anamnesis.state.persistence import save_cms_state, load_cms_state
from anamnesis.training.trainer import AnamnesisTrainer, TrainerConfig


VOCAB_SIZE = 256
HIDDEN = 64
LAYERS = 2
HEADS = 4
KV_HEADS = 2
CMS_LEVELS = 3
CMS_CHUNKS = [1, 8, 32]


def make_config():
    return HopeConfig(
        vocab_size=VOCAB_SIZE, hidden_size=HIDDEN,
        num_hidden_layers=LAYERS, num_attention_heads=HEADS,
        num_kv_heads=KV_HEADS, cms_levels=CMS_LEVELS,
        cms_chunk_sizes=CMS_CHUNKS, cms_hidden_mult=4.0,
    )


def model_factory():
    torch.manual_seed(42)
    config = make_config()
    return HopeModel(config), config


def make_dataloader(n, batch_size=4, seq_len=32):
    batches = []
    for _ in range(n):
        batches.append({
            "input_ids": torch.randint(0, VOCAB_SIZE, (batch_size, seq_len)),
            "signal_health": torch.rand(batch_size) * 0.5 + 0.3,
        })
    return DataLoader(batches, batch_size=None)


def table_1_ablation(steps):
    """Table 1: Impact of each active inference extension."""
    print("\n" + "=" * 70)
    print("TABLE 1: ABLATION STUDY — Impact of Active Inference Extensions")
    print("=" * 70)

    train_dl = make_dataloader(steps + 20)
    eval_dl = make_dataloader(10)

    runner = AblationRunner(
        model_factory=model_factory,
        train_loader=train_dl,
        eval_loader=eval_dl,
        train_steps=steps,
        # Lower lr for tiny model (M3 default 0.02 is for full-scale)
        trainer_overrides={"lr": 3e-4, "adam_lr": 3e-4},
    )

    results = runner.run_all(verbose=False)
    runner.print_table(results)
    return {r.config_name: r.to_dict() for r in results}


def table_2_learning_dynamics(n_conversations):
    """Table 2: CMS inner-loop learning dynamics during inference."""
    print("\n" + "=" * 70)
    print("TABLE 2: CMS LEARNING DYNAMICS — Weight change during inference")
    print("=" * 70)

    model, _ = model_factory()
    model.eval()

    before = snapshot_cms_state(model)
    surprise_trajectory = []

    for i in range(n_conversations):
        x = torch.randint(0, VOCAB_SIZE, (1, 32))
        with torch.no_grad():
            model(x)
        profile = compute_surprise_profile(model)
        avg_surprise = sum(
            sum(v for v in layer.values())
            for layer in profile
        ) / (LAYERS * CMS_LEVELS)
        surprise_trajectory.append(avg_surprise)

        if (i + 1) % max(1, n_conversations // 5) == 0:
            after = snapshot_cms_state(model)
            delta = compute_cms_delta(before, after)
            print(f"  After {i+1} conversations: CMS d={delta['total_l2']:.6f}, "
                  f"avg surprise={avg_surprise:.6f}")

    after = snapshot_cms_state(model)
    delta = compute_cms_delta(before, after)

    result = {
        "conversations": n_conversations,
        "total_cms_delta": delta["total_l2"],
        "per_level_delta": delta["per_level"],
        "max_param_delta": delta["max_param_delta"],
        "surprise_trajectory": surprise_trajectory[-10:],  # Last 10
    }

    print(f"\n  Total CMS state change: {delta['total_l2']:.6f}")
    for level, d in delta["per_level"].items():
        print(f"  {level}: d={d:.6f}")

    return result


def table_3_continuity(tmp_dir):
    """Table 3: State persistence — learning survives across sessions."""
    print("\n" + "=" * 70)
    print("TABLE 3: CONTINUITY — CMS state persistence across sessions")
    print("=" * 70)

    # Session 1: train
    model1, _ = model_factory()
    train_dl = make_dataloader(30)
    config = TrainerConfig(
        lr=3e-4, adam_lr=3e-4,
        max_steps=20, enable_gardener=False, enable_thompson=False,
        enable_toroidal=False, enable_drift=False, enable_dreaming=False,
        log_every=100, save_every=100,
    )
    trainer = AnamnesisTrainer(model1, config)
    history = trainer.train(train_dl)

    # Save state
    state_path = tmp_dir / "session1_cms.pt"
    save_cms_state(model1, state_path, {"session": 1, "steps": 20})
    cms_after_train = snapshot_cms_state(model1)

    # Session 2: load into fresh model
    model2, _ = model_factory()
    cms_before_load = snapshot_cms_state(model2)
    metadata = load_cms_state(model2, state_path)
    cms_after_load = snapshot_cms_state(model2)

    # Compare
    delta_train = compute_cms_delta(snapshot_cms_state(HopeModel(make_config())), cms_after_train)
    delta_load = compute_cms_delta(cms_before_load, cms_after_load)

    # Verify loaded matches trained
    match = True
    for key in cms_after_train:
        if key in cms_after_load:
            if not torch.allclose(cms_after_train[key], cms_after_load[key]):
                match = False
                break

    result = {
        "session_1_steps": 20,
        "session_1_cms_delta": delta_train["total_l2"],
        "session_2_loaded_delta": delta_load["total_l2"],
        "state_match": match,
        "metadata_preserved": metadata.get("session") == 1,
    }

    print(f"  Session 1 training d: {delta_train['total_l2']:.6f}")
    print(f"  Session 2 loaded d:   {delta_load['total_l2']:.6f}")
    print(f"  State match: {'YES' if match else 'NO'}")
    print(f"  Metadata preserved: {'YES' if result['metadata_preserved'] else 'NO'}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Reproduce Anamnesis Paper Results")
    parser.add_argument("--thorough", action="store_true", help="Run thorough experiments")
    parser.add_argument("--output", default="data/paper_results.json")
    args = parser.parse_args()

    steps = 200 if args.thorough else 50
    n_convos = 100 if args.thorough else 20


    print("=" * 70)
    print("ANAMNESIS — Reproducing Paper Results")
    print(f"Mode: {'thorough' if args.thorough else 'quick'}")
    print(f"Model: {HIDDEN}d, {LAYERS}L, {CMS_LEVELS} CMS levels")
    print("=" * 70)

    start = time.time()
    all_results = {}

    # Table 1: Ablation
    all_results["table_1_ablation"] = table_1_ablation(steps)

    # Table 2: Learning dynamics
    all_results["table_2_learning"] = table_2_learning_dynamics(n_convos)

    # Table 3: Continuity
    tmp_dir = Path("data/reproduce_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    all_results["table_3_continuity"] = table_3_continuity(tmp_dir)

    elapsed = time.time() - start
    all_results["meta"] = {
        "mode": "thorough" if args.thorough else "quick",
        "elapsed_seconds": round(elapsed, 1),
        "model_config": {
            "hidden": HIDDEN, "layers": LAYERS, "cms_levels": CMS_LEVELS,
            "cms_chunks": CMS_CHUNKS,
        },
    }

    print(f"\n{'='*70}")
    print(f"All experiments complete in {elapsed:.1f}s")
    print(f"{'='*70}")

    # Save
    os.makedirs(Path(args.output).parent, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
