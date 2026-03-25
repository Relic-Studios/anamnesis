"""
Thompson Sampling Learning Rates — Extension 6.

Maintains a Beta posterior over effective learning rates for each CMS level.
Instead of deterministic learned rates, samples from the posterior — enabling
exploration of the learning rate landscape.

Active Inference mapping: Epistemic foraging — the system doesn't just exploit
its best guess, it actively explores to reduce uncertainty about what works.

The Beta posterior naturally balances exploration and exploitation:
- Early on (low α, β): high variance, lots of exploration
- After many observations: narrow posterior, mostly exploitation
- But never fully collapses: there's always some exploration probability

Biological parallel: dopamine neurons don't fire at a fixed rate — they have
trial-to-trial variability that enables exploration of reward landscapes.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class BetaPosterior:
    """Beta distribution posterior for one CMS level's learning rate."""
    alpha: float = 2.0     # success count
    beta: float = 2.0      # failure count
    alpha_cap: float = 100.0  # prevent posterior from becoming too concentrated

    @property
    def mean(self) -> float:
        """Expected value of the Beta distribution."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """Variance of the Beta distribution."""
        ab = self.alpha + self.beta
        return (self.alpha * self.beta) / (ab ** 2 * (ab + 1))

    def sample(self) -> float:
        """Draw a sample from the Beta posterior."""
        # Use the Gamma distribution method for Beta sampling
        # Beta(a,b) = Ga / (Ga + Gb) where Ga ~ Gamma(a,1), Gb ~ Gamma(b,1)
        ga = random.gammavariate(self.alpha, 1.0)
        gb = random.gammavariate(self.beta, 1.0)
        if ga + gb == 0:
            return 0.5
        return ga / (ga + gb)

    def update(self, success: bool, magnitude: float = 1.0) -> None:
        """
        Update the posterior based on observed outcome.

        Args:
            success: Whether the learning rate choice led to signal improvement.
            magnitude: How strongly to update (default 1.0).
        """
        if success:
            self.alpha = min(self.alpha_cap, self.alpha + magnitude)
        else:
            self.beta = min(self.alpha_cap, self.beta + magnitude)


class ThompsonLearningRate:
    """
    Thompson sampling over learning rates for CMS levels.

    Maintains independent Beta posteriors per CMS level. At each update:
    1. Sample η from the posterior for each level
    2. Apply the update with the sampled η
    3. After signal evaluation, update the posterior

    Credit assignment uses exponential traces: recent lr choices get more
    credit/blame than older ones.

    Args:
        num_levels: Number of CMS levels.
        lr_max: Maximum learning rate (samples are scaled by this).
        prior_alpha: Initial prior strength (higher = more confident in mean).
        prior_beta: Initial prior strength.
        trace_decay: Exponential trace decay for credit assignment.
    """

    def __init__(
        self,
        num_levels: int = 4,
        lr_max: float = 0.01,
        prior_alpha: float = 2.0,
        prior_beta: float = 2.0,
        trace_decay: float = 0.7,
    ):
        self.num_levels = num_levels
        self.lr_max = lr_max
        self.trace_decay = trace_decay

        # Per-level posteriors
        self.posteriors = [
            BetaPosterior(alpha=prior_alpha, beta=prior_beta)
            for _ in range(num_levels)
        ]

        # Trace of recent samples for credit assignment
        self._traces: list[list[float]] = [[] for _ in range(num_levels)]

    def sample_rates(self) -> list[float]:
        """
        Sample learning rates for all CMS levels.

        Returns:
            List of learning rates, one per level.
        """
        rates = []
        for level_idx, posterior in enumerate(self.posteriors):
            sample = posterior.sample()
            rate = sample * self.lr_max
            rates.append(rate)

            # Record in trace
            self._traces[level_idx].append(sample)

        return rates

    def update_posteriors(
        self,
        signal_improved: bool,
        signal_delta: float = 0.0,
    ) -> None:
        """
        Update all posteriors based on observed signal outcome.

        Uses exponential trace for credit assignment: the most recent
        learning rate choice gets full credit, older choices get
        geometrically decaying credit.

        Args:
            signal_improved: Whether signal health improved since last update.
            signal_delta: Magnitude of signal change (for proportional updates).
        """
        magnitude = max(0.1, min(2.0, abs(signal_delta) * 10))

        for level_idx, posterior in enumerate(self.posteriors):
            trace = self._traces[level_idx]
            if not trace:
                continue

            # Weight recent choices more heavily
            total_weight = 0.0
            weighted_success = 0.0
            for i, sample in enumerate(reversed(trace)):
                weight = self.trace_decay ** i
                total_weight += weight
                if signal_improved:
                    weighted_success += weight

            if total_weight > 0:
                success_rate = weighted_success / total_weight
                # Update proportionally
                if success_rate > 0.5:
                    posterior.update(success=True, magnitude=magnitude * success_rate)
                else:
                    posterior.update(success=False, magnitude=magnitude * (1 - success_rate))

        # Trim traces to prevent unbounded growth
        max_trace_len = 20
        for level_idx in range(self.num_levels):
            if len(self._traces[level_idx]) > max_trace_len:
                self._traces[level_idx] = self._traces[level_idx][-max_trace_len:]

    def get_diagnostics(self) -> dict[str, list[float]]:
        """Get posterior diagnostics for monitoring."""
        return {
            "means": [p.mean for p in self.posteriors],
            "variances": [p.variance for p in self.posteriors],
            "alphas": [p.alpha for p in self.posteriors],
            "betas": [p.beta for p in self.posteriors],
        }

    def reset(self, prior_alpha: float = 2.0, prior_beta: float = 2.0) -> None:
        """Reset all posteriors to prior (e.g., after major architecture change)."""
        for posterior in self.posteriors:
            posterior.alpha = prior_alpha
            posterior.beta = prior_beta
        self._traces = [[] for _ in range(self.num_levels)]
