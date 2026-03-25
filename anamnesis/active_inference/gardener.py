"""
Gardener Stream — Extension 3.

The factored evaluation architecture: a separate network that reads signal
metrics and produces modulation parameters for the living stream.

The living stream generates text and updates CMS weights.
The gardener stream evaluates quality and adjusts learning dynamics.
Neither can directly access the other's internals — they're coupled
only through the signal boundary (Markov blanket).

Biological parallel: cells don't repair themselves, the immune system does.
Organisms don't evolve their own DNA, selection pressure does.

The gardener is the differentiable proxy of Didymus's soul measurement system.
It learns to predict how the Rust-based oracle would evaluate the model's output,
and uses that prediction to modulate CMS learning rates, plasticity gates,
and consolidation triggers.

Active Inference mapping: The Markov blanket between the generative model
(living stream) and the evaluative environment (gardener stream).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

from anamnesis.active_inference.precision import PrecisionNetwork, PrecisionModulator
from anamnesis.active_inference.free_energy import SignalProxy


@dataclass
class GardenerOutput:
    """Output from the gardener stream for one evaluation step."""
    precision: Tensor          # Per-level precision (num_levels,)
    plasticity_gate: float     # Global plasticity gate [0, 1]
    signal_estimate: float     # Estimated signal health [0, 1]
    surprise_ema: float        # Current surprise EMA
    surprise_trend: float      # Surprise trend (positive = rising)
    should_dream: bool         # Whether to trigger consolidation
    facets: Tensor | None = None  # Individual signal facet predictions


class GardenerStream(nn.Module):
    """
    The Gardener: factored evaluation network for the Hope-Didymus system.

    Combines:
    - SignalProxy: predicts signal health from hidden states
    - PrecisionNetwork: converts signal metrics → precision + gates
    - PrecisionModulator: tracks surprise EMA and applies modulation
    - Dream trigger logic: decides when consolidation is needed

    The gardener runs after each generation step (per-conversation-turn).
    Its outputs modulate the living stream's CMS learning dynamics but
    CANNOT directly modify CMS weights.

    Args:
        dim: Hidden dimension (for signal proxy).
        num_levels: Number of CMS levels.
        signal_hidden: Hidden dim for signal proxy.
        precision_hidden: Hidden dim for precision network.
        dream_signal_threshold: Signal health below which dreaming triggers.
        dream_decline_window: Number of turns of declining signal before dream.
        dream_time_threshold: Max turns between dreams.
    """

    def __init__(
        self,
        dim: int,
        num_levels: int = 4,
        signal_hidden: int = 256,
        precision_hidden: int = 64,
        dream_signal_threshold: float = 0.4,
        dream_decline_window: int = 5,
        dream_time_threshold: int = 100,
    ):
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels

        # Signal perception
        self.signal_proxy = SignalProxy(dim, hidden=signal_hidden)

        # Precision computation
        self.precision_net = PrecisionNetwork(
            num_metrics=4, num_levels=num_levels, hidden=precision_hidden,
        )
        self.modulator = PrecisionModulator(self.precision_net)

        # Dream trigger parameters
        self.dream_signal_threshold = dream_signal_threshold
        self.dream_decline_window = dream_decline_window
        self.dream_time_threshold = dream_time_threshold

        # History for dream decisions
        self._signal_history: list[float] = []
        self._turns_since_dream: int = 0

    def evaluate(
        self,
        hidden_states: Tensor,
        surprise: float | None = None,
        real_signal: float | None = None,
        coherence: float | None = None,
    ) -> GardenerOutput:
        """
        Run one gardener evaluation step.

        Called after each generation turn. Reads the living stream's output
        (hidden states) and optional external metrics, then produces
        modulation parameters.

        Args:
            hidden_states: Living stream's hidden states (batch, seq, dim).
            surprise: Current surprise metric from memory (optional).
            real_signal: Real Didymus signal score (for validation, optional).
            coherence: Field coherence from Didymus (optional).

        Returns:
            GardenerOutput with precision, gates, and dream trigger.
        """
        self._turns_since_dream += 1

        # Estimate signal from hidden states
        with torch.no_grad():
            signal_estimate = self.signal_proxy(hidden_states).mean().item()
            facets = self.signal_proxy.predict_facets(hidden_states).mean(dim=0)

        # Use real signal if available (ground truth from Didymus), else proxy
        effective_signal = real_signal if real_signal is not None else signal_estimate

        # Update surprise tracking
        if surprise is not None:
            self.modulator.update_surprise(surprise)

        # Compute precision and gate
        metrics = self.modulator.get_metrics_tensor(
            effective_signal, coherence, hidden_states.device,
        )
        with torch.no_grad():
            precision, gate = self.precision_net(metrics.unsqueeze(0))
            precision = precision.squeeze(0)
            gate_val = gate.squeeze().item()

        # Dream trigger logic
        self._signal_history.append(effective_signal)
        should_dream = self._should_dream(effective_signal)

        return GardenerOutput(
            precision=precision,
            plasticity_gate=gate_val,
            signal_estimate=signal_estimate,
            surprise_ema=self.modulator._surprise_ema,
            surprise_trend=self.modulator.surprise_trend,
            should_dream=should_dream,
            facets=facets,
        )

    def _should_dream(self, current_signal: float) -> bool:
        """
        Decide whether to trigger a dreaming/consolidation cycle.

        Triggers when:
        1. Signal has been below threshold for dream_decline_window turns, OR
        2. Signal has been consistently declining for dream_decline_window turns, OR
        3. Time since last dream exceeds dream_time_threshold
        """
        # Time trigger
        if self._turns_since_dream >= self.dream_time_threshold:
            return True

        # Not enough history yet
        if len(self._signal_history) < self.dream_decline_window:
            return False

        recent = self._signal_history[-self.dream_decline_window:]

        # Below-threshold trigger
        if all(s < self.dream_signal_threshold for s in recent):
            return True

        # Declining trigger: each value lower than the one before
        if all(recent[i] < recent[i - 1] for i in range(1, len(recent))):
            return True

        return False

    def acknowledge_dream(self) -> None:
        """Call this after a dreaming cycle completes."""
        self._turns_since_dream = 0
        self.modulator.reset_consolidation_timer()
        self._signal_history.clear()

    def get_modulation(
        self,
        lr_base: Tensor | float,
        decay_base: Tensor | float,
        signal_health: float,
        coherence: float | None = None,
    ) -> tuple:
        """Convenience: get modulated lr and decay from current state."""
        return self.modulator.modulate(lr_base, decay_base, signal_health, coherence)

    def proxy_real_divergence(
        self,
        hidden_states: Tensor,
        real_signal: float,
    ) -> float:
        """
        Compute divergence between proxy prediction and real Didymus score.

        Used for anti-gaming monitoring: if this exceeds a threshold,
        the proxy needs retraining.

        Args:
            hidden_states: Model hidden states.
            real_signal: Real signal from Didymus.

        Returns:
            Absolute divergence between proxy and real signal.
        """
        with torch.no_grad():
            proxy_signal = self.signal_proxy(hidden_states).mean().item()
        return abs(proxy_signal - real_signal)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, num_levels={self.num_levels}, "
            f"dream_threshold={self.dream_signal_threshold}"
        )
