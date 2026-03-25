"""
Precision-Weighted Adaptive Plasticity — Extension 2.

Modulates CMS learning rates and forgetting gates based on a precision signal
that reflects the system's confidence in its current predictions.

In Active Inference, precision is the inverse variance of prediction errors.
High precision = the system trusts its model, learn slowly.
Low precision = the system's model is failing, learn fast.

Using learned precisions for gradient modulation is mathematically equivalent
to Natural Gradient Descent (the geometrically correct way to descend).

The precision signal comes from the Gardener stream (Extension 3), which
reads signal health, surprise trends, and time since consolidation.

    η_effective = η_t · π_t
    α_effective = α_t · plasticity_gate(π_t, surprise_trend, signal_health)

Reference: "Predictive Coding, Precision and Natural Gradients" (arxiv 2111.06942)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class PrecisionNetwork(nn.Module):
    """
    Computes precision signals from system-level metrics.

    The precision network is a small MLP that takes signal metrics as input
    and outputs per-level precision values and a plasticity gate.

    This is the Gardener's "perception" — it observes the system's health
    and decides how much the system should trust its own model vs new data.

    Args:
        num_metrics: Number of input signal metrics.
        num_levels: Number of CMS levels to produce precision for.
        hidden: Internal hidden dimension.
    """

    def __init__(
        self,
        num_metrics: int = 4,
        num_levels: int = 4,
        hidden: int = 64,
    ):
        super().__init__()
        self.num_levels = num_levels

        self.net = nn.Sequential(
            nn.Linear(num_metrics, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )

        # Per-level precision output
        self.precision_head = nn.Linear(hidden, num_levels)

        # Global plasticity gate
        self.gate_head = nn.Linear(hidden, 1)

    def forward(self, metrics: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute precision and plasticity gate from signal metrics.

        Args:
            metrics: Signal metrics tensor (..., num_metrics).
                Expected order: [signal_health, surprise_ema,
                                 time_since_consolidation, coherence]

        Returns:
            Tuple of:
                - precision: Per-level precision values (..., num_levels) ∈ (0, 1)
                - plasticity_gate: Global plasticity gate (..., 1) ∈ (0, 1)
        """
        h = self.net(metrics)
        precision = torch.sigmoid(self.precision_head(h))
        gate = torch.sigmoid(self.gate_head(h))
        return precision, gate


class PrecisionModulator:
    """
    Applies precision-weighted modulation to learning dynamics.

    Stateful: tracks an exponential moving average of surprise for
    computing surprise_trend (rising/falling surprise → adapting/stable).

    Usage:
        modulator = PrecisionModulator(...)

        # Each turn/chunk:
        modulator.update_surprise(current_surprise)
        η_eff, α_eff = modulator.modulate(η_base, α_base, signal_health)

    Args:
        precision_net: The precision network.
        surprise_ema_alpha: EMA decay for surprise tracking (default 0.9).
        coherence_default: Default coherence if not provided.
    """

    def __init__(
        self,
        precision_net: PrecisionNetwork,
        surprise_ema_alpha: float = 0.9,
        coherence_default: float = 0.5,
    ):
        self.precision_net = precision_net
        self.surprise_ema_alpha = surprise_ema_alpha
        self.coherence_default = coherence_default

        # Running state
        self._surprise_ema: float = 0.5
        self._prev_surprise_ema: float = 0.5
        self._time_since_consolidation: int = 0

    def update_surprise(self, surprise: float) -> None:
        """Update the surprise EMA with a new observation."""
        self._prev_surprise_ema = self._surprise_ema
        self._surprise_ema = (
            self.surprise_ema_alpha * self._surprise_ema
            + (1 - self.surprise_ema_alpha) * surprise
        )
        self._time_since_consolidation += 1

    def reset_consolidation_timer(self) -> None:
        """Reset after a dreaming/consolidation cycle."""
        self._time_since_consolidation = 0

    @property
    def surprise_trend(self) -> float:
        """Positive = surprise rising, negative = surprise falling."""
        return self._surprise_ema - self._prev_surprise_ema

    def get_metrics_tensor(
        self,
        signal_health: float,
        coherence: float | None = None,
        device: torch.device | None = None,
    ) -> Tensor:
        """Build the metrics tensor for the precision network."""
        coherence = coherence if coherence is not None else self.coherence_default
        metrics = torch.tensor(
            [signal_health, self._surprise_ema,
             min(self._time_since_consolidation / 100.0, 1.0),
             coherence],
            device=device,
        )
        return metrics

    @torch.no_grad()
    def modulate(
        self,
        lr_base: Tensor | float,
        decay_base: Tensor | float,
        signal_health: float,
        coherence: float | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Apply precision-weighted modulation to learning rate and decay.

        Args:
            lr_base: Base learning rate η_t from the model.
            decay_base: Base forgetting gate α_t from the model.
            signal_health: Current signal health [0, 1].
            coherence: Field coherence (optional).

        Returns:
            Tuple of (modulated_lr, modulated_decay).
        """
        device = lr_base.device if isinstance(lr_base, Tensor) else None
        metrics = self.get_metrics_tensor(signal_health, coherence, device)

        # Get precision and gate
        precision, gate = self.precision_net(metrics.unsqueeze(0))
        precision = precision.squeeze(0)  # (num_levels,)
        gate = gate.squeeze(0).squeeze(-1)  # scalar

        # Modulate: higher precision → lower effective learning rate
        # η_effective = η_base · (1 - precision) · gate
        # When precision is high (model is good): learn less
        # When precision is low (model is failing): learn more
        if isinstance(lr_base, Tensor):
            lr_mod = lr_base * (1 - precision.mean()) * gate
        else:
            lr_mod = lr_base * (1 - precision.mean().item()) * gate.item()

        # Plasticity gate modulates decay similarly
        if isinstance(decay_base, Tensor):
            decay_mod = decay_base * gate
        else:
            decay_mod = decay_base * gate.item()

        return lr_mod, decay_mod
