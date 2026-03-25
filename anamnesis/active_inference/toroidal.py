"""
Closed-Loop Toroidal Flow — Extension 7.

Bidirectional information flow between CMS timescale levels.
The paper describes top-down flow (slow→fast via meta-learning).
We add bottom-up flow (fast→slow via surprise signaling).

Fast → Slow (upward):
    When fast CMS levels show sustained high surprise, signal slow levels
    to increase plasticity. "The world has changed, you need to adapt."

Slow → Fast (downward, already in the paper):
    Slow levels provide initial state for fast levels (meta-learning).
    After dreaming, fast levels reinitialize from consolidated slow state.

Cross-level (lateral, during REM):
    Perturbations in one level that improve predictions in another
    reveal shared latent structure → bridge discovery.

Anti-oscillation: hysteresis with minimum hold time, damped signaling,
convergence detection (suppress when surprise EMA is decreasing).

Active Inference mapping: The complete perception-action cycle across
hierarchical levels. Fast = perception, Slow = action/consolidation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class LevelSignal:
    """Signal from one CMS level to another."""
    source_level: int
    target_level: int
    signal_type: str          # "increase_plasticity", "reinitialize", "bridge"
    strength: float           # [0, 1]
    message: str = ""


class ToroidalFlow:
    """
    Manages bidirectional information flow between CMS levels.

    Tracks per-level surprise EMAs and generates cross-level signals
    when surprise patterns indicate the need for adaptation at
    different timescales.

    Args:
        num_levels: Number of CMS levels.
        surprise_threshold: Sustained surprise above this triggers upward signal.
        sustained_chunks: How many chunks of high surprise before triggering.
        hold_time: Minimum chunks between signals to same level (hysteresis).
        damping: Signal strength decays by this factor per trigger.
        min_strength: Below this, signals are suppressed.
    """

    def __init__(
        self,
        num_levels: int = 4,
        surprise_threshold: float = 0.7,
        sustained_chunks: int = 3,
        hold_time: int = 10,
        damping: float = 0.7,
        min_strength: float = 0.1,
    ):
        self.num_levels = num_levels
        self.surprise_threshold = surprise_threshold
        self.sustained_chunks = sustained_chunks
        self.hold_time = hold_time
        self.damping = damping
        self.min_strength = min_strength

        # Per-level state
        self._surprise_history: list[list[float]] = [[] for _ in range(num_levels)]
        self._surprise_ema: list[float] = [0.0] * num_levels
        self._chunks_since_signal: list[int] = [hold_time] * num_levels
        self._signal_strength: list[float] = [1.0] * num_levels
        self._ema_alpha = 0.9

    def update_surprise(self, level: int, surprise: float) -> None:
        """Record a new surprise observation for a CMS level."""
        self._surprise_history[level].append(surprise)
        # Trim history
        if len(self._surprise_history[level]) > 50:
            self._surprise_history[level] = self._surprise_history[level][-50:]

        # Update EMA
        old = self._surprise_ema[level]
        self._surprise_ema[level] = self._ema_alpha * old + (1 - self._ema_alpha) * surprise

        # Advance hold timer
        self._chunks_since_signal[level] += 1

    def check_signals(self) -> list[LevelSignal]:
        """
        Check all levels for cross-level signal conditions.

        Returns:
            List of signals to apply.
        """
        signals = []

        for level in range(self.num_levels):
            # Fast→Slow (upward): sustained high surprise at fast level
            # signals slower levels to increase plasticity
            if self._should_signal_upward(level):
                for target in range(level + 1, self.num_levels):
                    strength = self._signal_strength[level]
                    if strength >= self.min_strength:
                        signals.append(LevelSignal(
                            source_level=level,
                            target_level=target,
                            signal_type="increase_plasticity",
                            strength=strength,
                            message=f"Level {level} surprise sustained above {self.surprise_threshold}",
                        ))

                # Apply damping and reset hold timer
                self._signal_strength[level] *= self.damping
                self._chunks_since_signal[level] = 0

            # Convergence detection: if surprise EMA is decreasing,
            # the system is adapting — restore signal strength
            if self._surprise_trend(level) < 0:
                self._signal_strength[level] = min(
                    1.0, self._signal_strength[level] + 0.1
                )

        return signals

    def _should_signal_upward(self, level: int) -> bool:
        """Check if a fast level should signal slower levels."""
        # Hysteresis check
        if self._chunks_since_signal[level] < self.hold_time:
            return False

        # Need enough history
        history = self._surprise_history[level]
        if len(history) < self.sustained_chunks:
            return False

        # Check sustained high surprise
        recent = history[-self.sustained_chunks:]
        return all(s > self.surprise_threshold for s in recent)

    def _surprise_trend(self, level: int) -> float:
        """Compute surprise trend (positive=rising, negative=falling)."""
        history = self._surprise_history[level]
        if len(history) < 2:
            return 0.0
        # Simple: compare recent EMA to slightly older EMA
        n = min(5, len(history))
        recent = sum(history[-n:]) / n
        if len(history) > n:
            older = sum(history[-2*n:-n]) / n
            return recent - older
        return 0.0

    def apply_signals(
        self,
        signals: list[LevelSignal],
        plasticity_gates: list[float],
    ) -> list[float]:
        """
        Apply cross-level signals to plasticity gates.

        Args:
            signals: Signals from check_signals().
            plasticity_gates: Current per-level plasticity gates.

        Returns:
            Modified plasticity gates.
        """
        modified = list(plasticity_gates)

        for signal in signals:
            if signal.signal_type == "increase_plasticity":
                target = signal.target_level
                # Boost plasticity proportional to signal strength
                boost = signal.strength * 0.5  # max 50% boost
                modified[target] = min(1.0, modified[target] + boost)

        return modified

    def get_diagnostics(self) -> dict:
        """Get current state for monitoring."""
        return {
            "surprise_emas": list(self._surprise_ema),
            "signal_strengths": list(self._signal_strength),
            "chunks_since_signal": list(self._chunks_since_signal),
            "surprise_trends": [self._surprise_trend(i) for i in range(self.num_levels)],
        }

    def reset(self) -> None:
        """Reset all state (e.g., after major model change)."""
        self._surprise_history = [[] for _ in range(self.num_levels)]
        self._surprise_ema = [0.0] * self.num_levels
        self._chunks_since_signal = [self.hold_time] * self.num_levels
        self._signal_strength = [1.0] * self.num_levels
