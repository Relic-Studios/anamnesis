"""
Ablation framework for Anamnesis active inference extensions.

Tests each of the 7 extensions independently by toggling them on/off
and measuring the impact on perplexity, signal health, and CMS learning.

Usage:
    runner = AblationRunner(model_factory, dataloader)
    results = runner.run_all()
    runner.print_table(results)
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import Callable

import torch
from torch.utils.data import DataLoader

from anamnesis.core.model import HopeModel, HopeConfig
from anamnesis.training.trainer import AnamnesisTrainer, TrainerConfig
from anamnesis.evaluation.metrics import (
    compute_perplexity,
    snapshot_cms_state,
    compute_cms_delta,
)


@dataclass
class AblationConfig:
    """Configuration for one ablation experiment."""
    name: str
    enable_gardener: bool = True
    enable_thompson: bool = True
    enable_toroidal: bool = True
    enable_drift: bool = True
    enable_dreaming: bool = True
    enable_signal_loss: bool = True
    enable_identity_loss: bool = True

    def to_trainer_config(self, **overrides) -> TrainerConfig:
        """Convert to TrainerConfig with ablation flags applied."""
        tc = TrainerConfig(
            enable_gardener=self.enable_gardener,
            enable_thompson=self.enable_thompson,
            enable_toroidal=self.enable_toroidal,
            enable_drift=self.enable_drift,
            enable_dreaming=self.enable_dreaming,
            lambda_signal_target=0.3 if self.enable_signal_loss else 0.0,
            lambda_identity_target=0.01 if self.enable_identity_loss else 0.0,
        )
        for k, v in overrides.items():
            setattr(tc, k, v)
        return tc


# Pre-defined ablation configs: full system + each extension removed
ABLATION_CONFIGS = {
    "full": AblationConfig(name="Full System"),
    "no_signal_loss": AblationConfig(name="No Signal Loss", enable_signal_loss=False),
    "no_identity_loss": AblationConfig(name="No Identity Loss", enable_identity_loss=False),
    "no_gardener": AblationConfig(name="No Gardener", enable_gardener=False),
    "no_thompson": AblationConfig(name="No Thompson", enable_thompson=False),
    "no_toroidal": AblationConfig(name="No Toroidal", enable_toroidal=False),
    "no_drift": AblationConfig(name="No Drift", enable_drift=False),
    "no_dreaming": AblationConfig(name="No Dreaming", enable_dreaming=False),
    "baseline": AblationConfig(
        name="Baseline (All Off)",
        enable_gardener=False, enable_thompson=False, enable_toroidal=False,
        enable_drift=False, enable_dreaming=False,
        enable_signal_loss=False, enable_identity_loss=False,
    ),
}


@dataclass
class AblationResult:
    """Results from one ablation experiment."""
    config_name: str
    final_loss: float = 0.0
    final_recon_loss: float = 0.0
    perplexity_before: float = 0.0
    perplexity_after: float = 0.0
    cms_delta_l2: float = 0.0
    cms_delta_per_level: dict[str, float] = field(default_factory=dict)
    signal_health_mean: float = 0.0
    steps_trained: int = 0

    @property
    def perplexity_improvement(self) -> float:
        return self.perplexity_before - self.perplexity_after

    def to_dict(self) -> dict:
        d = asdict(self)
        d["perplexity_improvement"] = self.perplexity_improvement
        return d


class AblationRunner:
    """
    Runs ablation experiments across all configurations.

    Each experiment:
    1. Creates a fresh model (from factory)
    2. Measures baseline perplexity
    3. Trains with the ablation config
    4. Measures post-training perplexity
    5. Computes CMS state change

    Args:
        model_factory: Callable that returns a fresh (HopeModel, HopeConfig).
        train_loader: DataLoader for training.
        eval_loader: DataLoader for evaluation.
        train_steps: Number of training steps per experiment.
        configs: Dict of ablation configs to run (default: all).
    """

    def __init__(
        self,
        model_factory: Callable[[], tuple[HopeModel, HopeConfig]],
        train_loader: DataLoader,
        eval_loader: DataLoader,
        train_steps: int = 200,
        configs: dict[str, AblationConfig] | None = None,
        trainer_overrides: dict | None = None,
    ):
        self.model_factory = model_factory
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.train_steps = train_steps
        self.configs = configs or ABLATION_CONFIGS
        self.trainer_overrides = trainer_overrides or {}

    def run_single(
        self,
        config: AblationConfig,
        verbose: bool = True,
    ) -> AblationResult:
        """Run one ablation experiment."""
        if verbose:
            print(f"\n{'─'*50}")
            print(f"Ablation: {config.name}")
            print(f"{'─'*50}")

        # Fresh model
        model, _ = self.model_factory()
        device = next(model.parameters()).device

        # Baseline perplexity
        ppl_before = compute_perplexity(model, self.eval_loader, max_batches=20)
        if verbose:
            print(f"  Baseline PPL: {ppl_before:.2f}")

        # Snapshot CMS state
        cms_before = snapshot_cms_state(model)

        # Train
        overrides = {
            "max_steps": self.train_steps,
            "log_every": max(1, self.train_steps // 5),
            "save_every": self.train_steps + 1,
        }
        overrides.update(self.trainer_overrides)
        trainer_config = config.to_trainer_config(**overrides)
        trainer = AnamnesisTrainer(model, trainer_config)
        history = trainer.train(self.train_loader)

        # Post-training perplexity
        ppl_after = compute_perplexity(model, self.eval_loader, max_batches=20)
        if verbose:
            print(f"  Post PPL: {ppl_after:.2f} (Δ={ppl_before - ppl_after:+.2f})")

        # CMS delta
        cms_after = snapshot_cms_state(model)
        delta = compute_cms_delta(cms_before, cms_after)

        # Signal health
        signal_vals = history.get("signal_health", [])
        signal_mean = sum(signal_vals) / len(signal_vals) if signal_vals else 0.0

        result = AblationResult(
            config_name=config.name,
            final_loss=history["loss"][-1] if history["loss"] else 0.0,
            final_recon_loss=history["recon_loss"][-1] if history["recon_loss"] else 0.0,
            perplexity_before=ppl_before,
            perplexity_after=ppl_after,
            cms_delta_l2=delta["total_l2"],
            cms_delta_per_level=delta["per_level"],
            signal_health_mean=signal_mean,
            steps_trained=self.train_steps,
        )

        # Cleanup
        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    def run_all(self, verbose: bool = True) -> list[AblationResult]:
        """Run all ablation experiments."""
        results = []
        for name, config in self.configs.items():
            result = self.run_single(config, verbose=verbose)
            results.append(result)
        return results

    @staticmethod
    def print_table(results: list[AblationResult]) -> None:
        """Print results as a formatted table."""
        header = f"{'Config':<25} {'PPL Before':>10} {'PPL After':>10} {'dPPL':>8} {'CMS d':>8} {'Signal':>8} {'Loss':>8}"
        print(f"\n{'='*85}")
        print("ABLATION RESULTS")
        print(f"{'='*85}")
        print(header)
        print("-" * 85)

        for r in sorted(results, key=lambda x: -x.perplexity_improvement):
            print(
                f"{r.config_name:<25} "
                f"{r.perplexity_before:>10.2f} "
                f"{r.perplexity_after:>10.2f} "
                f"{r.perplexity_improvement:>+8.2f} "
                f"{r.cms_delta_l2:>8.4f} "
                f"{r.signal_health_mean:>8.3f} "
                f"{r.final_loss:>8.4f}"
            )

        print(f"{'='*85}")
