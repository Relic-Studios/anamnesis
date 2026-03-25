#!/usr/bin/env python3
"""
Train a small Hope-Didymus model from scratch.

Demonstrates the full system working together:
1. CMS with multi-timescale updates
2. Neural memory learning during inference
3. M3 optimizer with dual momentum
4. Signal-aware composite loss
5. Gardener evaluation and precision modulation
6. Thompson sampling learning rates
7. Neutral drift on dormant levels
8. Dream cycle when signal drops
9. Toroidal flow between CMS levels

This trains a tiny model on random data — the goal is to verify
the architecture works end-to-end, not to produce useful outputs.
"""

import torch
from torch import Tensor

from anamnesis.core.model import HopeModel, HopeConfig
from anamnesis.optim.m3 import M3
from anamnesis.active_inference import (
    CompositeHopeLoss,
    NeutralDrift,
    GardenerStream,
    ThompsonLearningRate,
    ToroidalFlow,
    DreamCycle,
)
from anamnesis.state.persistence import save_cms_state, load_cms_state, save_soul_checkpoint


def main():
    print("=" * 60)
    print("Anamnesis: End-to-End Training Demo")
    print("=" * 60)

    # ── Config ──
    config = HopeConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_kv_heads=2,
        cms_levels=3,
        cms_chunk_sizes=[1, 8, 32],
        cms_hidden_mult=4.0,
        use_neural_memory=True,
        mem_heads=2,
        mem_depth=1,
    )

    print(f"\nModel config:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  CMS levels: {config.cms_levels} ({config.cms_chunk_sizes})")
    print(f"  Neural memory: {config.use_neural_memory}")

    # ── Build model ──
    model = HopeModel(config)
    print(f"  Parameters: {model.num_parameters():,}")

    # ── Optimizer (M3) ──
    optimizer = M3(
        model.parameters(),
        lr=0.01,
        beta1=0.95, beta2=0.999, beta3=0.95,
        alpha=0.1, weight_decay=0.01,
        slow_freq=10,
    )

    # ── Loss function (composite) ──
    loss_fn = CompositeHopeLoss(
        lambda_recon=1.0,
        lambda_signal=0.0,  # start with reconstruction only
        lambda_identity=0.0,
        dim=config.hidden_size,
        use_proxy=True,
    )

    # ── Active Inference components ──
    gardener = GardenerStream(
        dim=config.hidden_size,
        num_levels=config.cms_levels,
        dream_signal_threshold=0.3,
        dream_time_threshold=20,
    )
    thompson = ThompsonLearningRate(
        num_levels=config.cms_levels, lr_max=0.01,
    )
    toroidal = ToroidalFlow(
        num_levels=config.cms_levels,
        surprise_threshold=0.7, sustained_chunks=3,
    )
    drift = NeutralDrift(sigma_base=1e-5, enabled=False)  # enable after warmup
    dreamer = DreamCycle(
        rem_noise_scale=0.01, rem_perturbations=3,
        rem_min_level=1, rem_max_level=1,
    )

    # ── Training loop ──
    print(f"\n--- Phase 1: Pure Reconstruction (10 steps) ---")
    model.train()

    for step in range(10):
        input_ids = torch.randint(0, 256, (4, 32))
        labels = torch.randint(0, 256, (4, 32))

        output = model(input_ids, labels=labels)
        recon_loss = output["loss"]

        result = loss_fn(recon_loss)
        total_loss = result["total"]

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 5 == 0:
            print(f"  Step {step:3d} | Loss: {total_loss.item():.4f}")

    # ── Save soul checkpoint ──
    print(f"\n--- Saving soul checkpoint ---")
    save_soul_checkpoint(model, "/tmp/hope_demo_soul.pt", "Demo identity")
    print("  Saved to /tmp/hope_demo_soul.pt")

    # ── Phase 2: Anneal in signal loss ──
    print(f"\n--- Phase 2: Signal-Aware Training (20 steps) ---")
    drift.enabled = True  # enable neutral drift
    model.enable_drift(True)

    for step in range(20):
        input_ids = torch.randint(0, 256, (4, 32))
        labels = torch.randint(0, 256, (4, 32))

        output = model(input_ids, labels=labels)
        recon_loss = output["loss"]

        # Get hidden states for gardener/proxy
        with torch.no_grad():
            hidden = model.embed_tokens(input_ids)

        # Gardener evaluation
        gard_out = gardener.evaluate(
            hidden, surprise=recon_loss.item(), real_signal=None,
        )

        # Thompson sampling for learning rates
        sampled_rates = thompson.sample_rates()

        # Toroidal flow: update surprise per level
        for level_idx in range(config.cms_levels):
            toroidal.update_surprise(level_idx, recon_loss.item())

        # Check cross-level signals
        signals = toroidal.check_signals()
        if signals:
            print(f"  Step {step:3d} | Toroidal signal: {signals[0].message}")

        # Compute composite loss
        loss_fn.anneal_signal(target=0.3, step=0.02)
        result = loss_fn(recon_loss, hidden_states=hidden)
        total_loss = result["total"]

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Apply neutral drift to CMS
        for layer in model.layers:
            drift.apply_to_cms(layer.cms, plasticity_gate=gard_out.plasticity_gate)

        # Update Thompson posteriors
        signal_improving = step > 0 and recon_loss.item() < 6.0
        thompson.update_posteriors(signal_improving, signal_delta=0.01)

        if step % 5 == 0:
            sig_loss = result.get("signal", torch.tensor(0.0))
            print(
                f"  Step {step:3d} | Total: {total_loss.item():.4f} | "
                f"Recon: {recon_loss.item():.4f} | "
                f"Signal w: {loss_fn.lambda_signal:.3f} | "
                f"Gardener gate: {gard_out.plasticity_gate:.3f} | "
                f"Thompson: {[f'{m:.3f}' for m in thompson.get_diagnostics()['means']]}"
            )

        # Check if gardener wants to dream
        if gard_out.should_dream:
            print(f"  Step {step:3d} | [DREAM]")
            def eval_fn(module):
                return 0.5 + torch.randn(1).item() * 0.1
            dream_result = dreamer.dream(
                model.layers[0].cms.levels, eval_fn,
            )
            print(
                f"           | NREM pruned: {dream_result.nrem_pruned_params} | "
                f"REM bridges: {dream_result.rem_bridges_discovered}/{dream_result.rem_perturbations_tested}"
            )
            gardener.acknowledge_dream()

    # ── Save final state ──
    print(f"\n--- Saving CMS state ---")
    save_cms_state(model, "/tmp/hope_demo_state.pt", {
        "tokens_processed": 30 * 4 * 32,
        "session_count": 1,
    })
    print("  Saved to /tmp/hope_demo_state.pt")

    # ── Load state to verify persistence ──
    print(f"\n--- Verifying state persistence ---")
    model2 = HopeModel(config)
    metadata = load_cms_state(model2, "/tmp/hope_demo_state.pt")
    print(f"  Loaded state with metadata: {metadata}")

    # Verify weights match
    for layer_idx in range(config.num_hidden_layers):
        for level_idx in range(config.cms_levels):
            for name, p1 in model.layers[layer_idx].cms.levels[level_idx].named_parameters():
                p2 = dict(model2.layers[layer_idx].cms.levels[level_idx].named_parameters())[name]
                assert torch.allclose(p1.cpu(), p2.cpu(), atol=1e-6), (
                    f"Mismatch at layer {layer_idx} level {level_idx} param {name}"
                )
    print("  State persistence verified!")

    # ── Final diagnostics ──
    print(f"\n--- Diagnostics ---")
    print(f"  Toroidal flow: {toroidal.get_diagnostics()}")
    print(f"  Thompson posteriors: {thompson.get_diagnostics()}")

    print(f"\n{'=' * 60}")
    print(f"Hope-Didymus end-to-end demo complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
