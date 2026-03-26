"""
Anamnesis Trainer — the main training loop for Hope-Didymus models.

Orchestrates:
1. Standard next-token prediction (reconstruction loss)
2. Signal-aware composite loss with staged annealing
3. Gardener evaluation after each step
4. Thompson sampling learning rate exploration
5. Toroidal cross-level signaling
6. Neutral drift on dormant levels
7. Dream cycle when triggered by gardener
8. CMS state checkpointing

Usage:
    trainer = AnamnesisTrainer(model, config)
    trainer.train(dataset)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from anamnesis.core.model import HopeModel
from anamnesis.optim.m3 import M3
from anamnesis.active_inference import (
    CompositeHopeLoss,
    NeutralDrift,
    GardenerStream,
    ThompsonLearningRate,
    ToroidalFlow,
    DreamCycle,
)
from anamnesis.state.persistence import save_cms_state, save_soul_checkpoint


@dataclass
class TrainerConfig:
    """Configuration for the Anamnesis trainer."""
    # Optimizer
    lr: float = 0.02
    adam_lr: float = 3e-4
    weight_decay: float = 0.01
    m3_slow_freq: int = 100

    # Loss
    lambda_recon: float = 1.0
    lambda_signal_target: float = 0.3
    lambda_identity_target: float = 0.01
    anneal_signal_step: float = 0.01
    anneal_identity_step: float = 0.001
    warmup_steps: int = 100  # Steps before annealing begins

    # Training
    batch_size: int = 4
    max_steps: int = 1000
    gradient_accumulation: int = 1
    max_grad_norm: float = 1.0
    log_every: int = 10
    save_every: int = 100

    # Active inference
    enable_gardener: bool = True
    enable_thompson: bool = True
    enable_toroidal: bool = True
    enable_drift: bool = True
    enable_dreaming: bool = True
    drift_after_step: int = 50  # Enable drift after warmup

    # Paths
    output_dir: str = "./checkpoints"
    soul_checkpoint_path: str = ""  # If empty, saves after warmup


class AnamnesisTrainer:
    """
    Main training loop for Anamnesis models.

    Implements the full Active Inference training paradigm:
    - Phase 1 (warmup): Pure reconstruction loss
    - Phase 2 (annealing): Gradually introduce signal and identity losses
    - Phase 3 (full): All losses + all extensions active

    Args:
        model: The HopeModel to train.
        config: TrainerConfig.
    """

    def __init__(self, model: HopeModel, config: TrainerConfig):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device

        # Optimizer
        self.optimizer = M3(
            model.parameters(),
            lr=config.lr,
            adam_lr=config.adam_lr,
            weight_decay=config.weight_decay,
            slow_freq=config.m3_slow_freq,
        )

        # Loss
        self.loss_fn = CompositeHopeLoss(
            lambda_recon=config.lambda_recon,
            lambda_signal=0.0,
            lambda_identity=0.0,
            dim=model.config.hidden_size,
            use_proxy=config.enable_gardener,
        )

        # Active inference components
        self.gardener: GardenerStream | None = None
        if config.enable_gardener:
            self.gardener = GardenerStream(
                dim=model.config.hidden_size,
                num_levels=model.config.cms_levels,
            )

        self.thompson: ThompsonLearningRate | None = None
        if config.enable_thompson:
            self.thompson = ThompsonLearningRate(
                num_levels=model.config.cms_levels,
            )

        self.toroidal: ToroidalFlow | None = None
        if config.enable_toroidal:
            self.toroidal = ToroidalFlow(
                num_levels=model.config.cms_levels,
            )

        self.drift = NeutralDrift(enabled=False)  # Enabled after warmup

        self.dreamer: DreamCycle | None = None
        if config.enable_dreaming:
            self.dreamer = DreamCycle(
                rem_noise_scale=0.01,
                rem_perturbations=5,
            )

        # State
        self._step = 0
        self._prev_loss = float("inf")
        self._replay_buffer: list[Tensor] = []
        self._replay_buffer_max = 10

    def train(
        self,
        dataloader: DataLoader,
        eval_fn: Callable | None = None,
    ) -> dict[str, list[float]]:
        """
        Run the full training loop.

        Args:
            dataloader: Training data loader.
            eval_fn: Optional evaluation function called every save_every steps.

        Returns:
            Dict of metric histories (loss, signal, etc.).
        """
        self.model.train()
        config = self.config
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        history = {
            "loss": [], "recon_loss": [], "signal_loss": [],
            "identity_loss": [], "signal_health": [],
        }

        for batch in dataloader:
            if self._step >= config.max_steps:
                break

            self._step += 1

            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            signal_health = batch.get("signal_health")

            # Forward
            output = self.model(input_ids, labels=input_ids)
            recon_loss = output["loss"]

            # Maintain replay buffer for dream evaluation
            self._replay_buffer.append(input_ids.detach())
            if len(self._replay_buffer) > self._replay_buffer_max:
                self._replay_buffer.pop(0)

            # Composite loss
            result = self.loss_fn(
                recon_loss,
                precomputed_signal=signal_health,
            )
            total_loss = result["total"]

            # Backward + optimize
            total_loss.backward()

            if self._step % config.gradient_accumulation == 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), config.max_grad_norm,
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

            # ── Active Inference Loop ──

            # Anneal losses after warmup
            if self._step > config.warmup_steps:
                self.loss_fn.anneal_signal(
                    config.lambda_signal_target, config.anneal_signal_step,
                )
                self.loss_fn.anneal_identity(
                    config.lambda_identity_target, config.anneal_identity_step,
                )

            # Enable drift after warmup
            if self._step == config.drift_after_step and config.enable_drift:
                self.drift.enabled = True
                self.model.enable_drift(True)

            # Save soul checkpoint after warmup (identity anchor)
            if self._step == config.warmup_steps and not config.soul_checkpoint_path:
                soul_path = output_dir / "soul_checkpoint.pt"
                save_soul_checkpoint(self.model, soul_path, "Post-warmup identity")
                config.soul_checkpoint_path = str(soul_path)

            # Gardener evaluation
            if self.gardener and self._step % 5 == 0:
                with torch.no_grad():
                    hidden = self.model.embed_tokens(input_ids[:1])
                gard_out = self.gardener.evaluate(
                    hidden,
                    surprise=recon_loss.item(),
                    real_signal=signal_health[0].item() if signal_health is not None else None,
                )

                # Apply drift with gardener's plasticity gate
                if self.drift.enabled:
                    for layer in self.model.layers:
                        self.drift.apply_to_cms(layer.cms, gard_out.plasticity_gate)

                # Thompson sampling
                if self.thompson:
                    self.thompson.sample_rates()
                    improving = recon_loss.item() < self._prev_loss
                    self.thompson.update_posteriors(improving, recon_loss.item() - self._prev_loss)

                # Toroidal flow — apply cross-level signals to plasticity
                if self.toroidal:
                    per_level_surprises = []
                    for layer in self.model.layers:
                        per_level_surprises.append(layer.cms.get_surprise())
                    # Average surprise across layers, per CMS level
                    for lvl in range(self.model.config.cms_levels):
                        avg_surprise = sum(
                            s[lvl] for s in per_level_surprises
                        ) / len(per_level_surprises) if per_level_surprises else 0.0
                        self.toroidal.update_surprise(lvl, avg_surprise)
                    signals = self.toroidal.check_signals()
                    if signals:
                        base_gates = [gard_out.plasticity_gate] * self.model.config.cms_levels
                        modified_gates = self.toroidal.apply_signals(signals, base_gates)
                        # Apply modified per-level gates to drift
                        if self.drift.enabled:
                            for layer in self.model.layers:
                                for lvl_idx, level in enumerate(layer.cms.levels):
                                    gate = modified_gates[lvl_idx] if lvl_idx < len(modified_gates) else gard_out.plasticity_gate
                                    self.drift.apply(level.parameters(), gate)

                # Dream cycle — use replay buffer for evaluation
                if self.dreamer and gard_out.should_dream and self._replay_buffer:
                    def dream_eval(level_module):
                        """Evaluate level quality on replay buffer."""
                        total_loss = 0.0
                        count = 0
                        for replay_ids in self._replay_buffer[-3:]:
                            out = self.model(replay_ids, labels=replay_ids)
                            total_loss += out["loss"].item()
                            count += 1
                        return 1.0 / (1.0 + total_loss / max(count, 1))

                    # Dream across all layers' CMS levels
                    for layer in self.model.layers:
                        dream_result = self.dreamer.dream(
                            layer.cms.levels, dream_eval,
                        )
                    self.gardener.acknowledge_dream()

                history["signal_health"].append(gard_out.signal_estimate)

            # Track metrics
            history["loss"].append(total_loss.item())
            history["recon_loss"].append(recon_loss.item())
            history["signal_loss"].append(result.get("signal", torch.tensor(0)).item())
            history["identity_loss"].append(result.get("identity", torch.tensor(0)).item())

            self._prev_loss = recon_loss.item()

            # Logging
            if self._step % config.log_every == 0:
                print(
                    f"Step {self._step}/{config.max_steps} | "
                    f"Loss: {total_loss.item():.4f} | "
                    f"Recon: {recon_loss.item():.4f} | "
                    f"Signal w: {self.loss_fn.lambda_signal:.3f}"
                )

            # Checkpointing
            if self._step % config.save_every == 0:
                ckpt_path = output_dir / f"cms_state_step_{self._step}.pt"
                save_cms_state(self.model, ckpt_path, {
                    "step": self._step,
                    "loss": total_loss.item(),
                })

        return history
