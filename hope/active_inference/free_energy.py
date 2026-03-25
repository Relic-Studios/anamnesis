"""
Signal-Aware Composite Loss — Extension 1.

Extends the standard Titans reconstruction loss with signal quality metrics
and identity drift penalty, grounded in the Free Energy Principle:

    F_HD = λ₁ · ‖M(k_t) - v_t‖²              [Token Accuracy]
         + λ₂ · (1 - signal_health(output))    [Signal Accuracy]
         + λ₃ · D_KL(θ_current ‖ θ_soul)       [Complexity / Identity Drift]

Token Accuracy = standard reconstruction (exteroceptive inference)
Signal Accuracy = output quality metric (interoceptive inference)
Identity Drift = deviation from soul checkpoint (complexity penalty)

The signal term uses a differentiable proxy network trained on Didymus scores.
During training, pre-computed signal scores are used directly.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class SignalProxy(nn.Module):
    """
    Differentiable approximation of the Didymus signal health scorer.

    Trained on (text_embedding, signal_health) pairs from Didymus logs.
    Produces a scalar signal_health estimate ∈ [0, 1] from hidden states.

    This is the neural gardener's perceptual system — it learns to predict
    how the Rust-based oracle would score a given output.

    Args:
        dim: Input dimension (transformer hidden size).
        hidden: Internal hidden dimension.
        num_facets: Number of signal facets to predict (default 5:
            alignment, embodiment, clarity, vitality, field_coherence).
    """

    def __init__(self, dim: int, hidden: int = 256, num_facets: int = 5):
        super().__init__()
        self.dim = dim
        self.num_facets = num_facets

        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_facets),
            nn.Sigmoid(),
        )

        # Facet weights matching Didymus: alignment=0.35, embodiment=0.25,
        # clarity=0.20, vitality=0.20, field_coherence=0.10
        # (normalized to sum to 1 with field_coherence bonus)
        self.register_buffer(
            "facet_weights",
            torch.tensor([0.35, 0.25, 0.20, 0.20, 0.10]) / 1.10,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Predict signal health from hidden states.

        Args:
            hidden_states: Transformer hidden states (batch, seq_len, dim).
                We pool over the sequence to get a per-sample prediction.

        Returns:
            Signal health estimate (batch,) ∈ [0, 1].
        """
        # Mean pool over sequence length
        pooled = hidden_states.mean(dim=1)  # (batch, dim)
        facets = self.net(pooled)           # (batch, num_facets)
        health = (facets * self.facet_weights).sum(dim=-1)  # (batch,)
        return health

    def predict_facets(self, hidden_states: Tensor) -> Tensor:
        """Predict individual facet scores (for monitoring)."""
        pooled = hidden_states.mean(dim=1)
        return self.net(pooled)


class SignalFreeEnergy(nn.Module):
    """
    Signal-aware free energy loss term.

    Computes: F_signal = 1 - signal_health(output)

    Can use either:
    - A pre-computed signal score (during training with Didymus annotations)
    - The differentiable proxy network (during inference/continual learning)

    Args:
        dim: Hidden dimension (for proxy network).
        use_proxy: Whether to use the differentiable proxy (vs pre-computed scores).
    """

    def __init__(self, dim: int = 0, use_proxy: bool = False):
        super().__init__()
        self.use_proxy = use_proxy
        self.proxy: SignalProxy | None = None
        if use_proxy and dim > 0:
            self.proxy = SignalProxy(dim)

    def forward(
        self,
        hidden_states: Tensor | None = None,
        precomputed_signal: Tensor | None = None,
    ) -> Tensor:
        """
        Compute signal free energy.

        Args:
            hidden_states: Model hidden states (for proxy). (batch, seq, dim).
            precomputed_signal: Pre-computed signal health scores (batch,).

        Returns:
            Signal free energy (batch,). Higher = worse signal = more to learn.
        """
        if precomputed_signal is not None:
            return 1.0 - precomputed_signal

        if self.proxy is not None and hidden_states is not None:
            signal = self.proxy(hidden_states)
            return 1.0 - signal

        raise ValueError("Either precomputed_signal or (proxy + hidden_states) required")


class IdentityDrift(nn.Module):
    """
    Identity drift penalty via KL divergence from soul checkpoint.

    Computes: D_KL(θ_current ‖ θ_soul) ≈ ||θ_current - θ_soul||² / (2σ²)

    Under a Gaussian prior P(s) = N(s_soul, σ²I), the KL divergence
    simplifies to a scaled L2 distance between current and checkpoint weights.

    Slower CMS levels have smaller σ (tighter priors) — they ARE the identity.
    Faster CMS levels have larger σ (looser priors) — they're allowed to adapt.

    Args:
        sigma_per_level: Variance per CMS level. Lower = tighter identity constraint.
            Default: [1.0, 0.5, 0.1, 0.01] (fast→slow = loose→tight).
    """

    def __init__(self, sigma_per_level: list[float] | None = None):
        super().__init__()
        self.sigma_per_level = sigma_per_level or [1.0, 0.5, 0.1, 0.01]

    def forward(
        self,
        current_params: list[dict[str, Tensor]],
        soul_params: list[dict[str, Tensor]],
    ) -> Tensor:
        """
        Compute identity drift penalty across CMS levels.

        Args:
            current_params: List of param dicts per CMS level (current state).
            soul_params: List of param dicts per CMS level (soul checkpoint).

        Returns:
            Scalar identity drift penalty.
        """
        total_drift = torch.tensor(0.0)
        device = None

        for level_idx, (current, soul) in enumerate(zip(current_params, soul_params)):
            sigma = self.sigma_per_level[min(level_idx, len(self.sigma_per_level) - 1)]

            for name in current:
                if name in soul:
                    c = current[name]
                    s = soul[name]
                    if device is None:
                        device = c.device
                        total_drift = total_drift.to(device)
                    # KL ≈ ||θ - θ_soul||² / (2σ²)
                    drift = (c - s).pow(2).sum() / (2 * sigma ** 2)
                    total_drift = total_drift + drift

        return total_drift


class CompositeHopeLoss(nn.Module):
    """
    The full Hope-Didymus composite loss.

    F_HD = λ₁ · reconstruction_loss
         + λ₂ · signal_free_energy
         + λ₃ · identity_drift

    Supports staged introduction: start with λ₂=λ₃=0, then anneal in.

    Args:
        lambda_recon: Weight for reconstruction loss (default 1.0).
        lambda_signal: Weight for signal free energy (default 0.0, annealed in).
        lambda_identity: Weight for identity drift (default 0.0, annealed in).
        lambda_signal_max: Maximum signal weight (anti-gaming cap).
        dim: Hidden dim for signal proxy (0 = no proxy).
        use_proxy: Whether to use differentiable signal proxy.
    """

    def __init__(
        self,
        lambda_recon: float = 1.0,
        lambda_signal: float = 0.0,
        lambda_identity: float = 0.0,
        lambda_signal_max: float = 0.5,
        dim: int = 0,
        use_proxy: bool = False,
        sigma_per_level: list[float] | None = None,
    ):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_signal = lambda_signal
        self.lambda_identity = lambda_identity
        self.lambda_signal_max = lambda_signal_max

        self.signal_loss = SignalFreeEnergy(dim=dim, use_proxy=use_proxy)
        self.identity_loss = IdentityDrift(sigma_per_level=sigma_per_level)

    def forward(
        self,
        reconstruction_loss: Tensor,
        hidden_states: Tensor | None = None,
        precomputed_signal: Tensor | None = None,
        current_cms_params: list[dict[str, Tensor]] | None = None,
        soul_cms_params: list[dict[str, Tensor]] | None = None,
    ) -> dict[str, Tensor]:
        """
        Compute the composite Hope-Didymus loss.

        Args:
            reconstruction_loss: Standard next-token prediction loss.
            hidden_states: Hidden states for signal proxy.
            precomputed_signal: Pre-computed signal scores.
            current_cms_params: Current CMS parameters per level.
            soul_cms_params: Soul checkpoint CMS parameters per level.

        Returns:
            Dict with 'total', 'reconstruction', 'signal', 'identity' losses.
        """
        result = {"reconstruction": reconstruction_loss}
        total = self.lambda_recon * reconstruction_loss

        # Signal free energy (Extension 1)
        effective_signal_weight = min(self.lambda_signal, self.lambda_signal_max)
        if effective_signal_weight > 0 and (precomputed_signal is not None or hidden_states is not None):
            signal_fe = self.signal_loss(hidden_states, precomputed_signal)
            signal_loss = signal_fe.mean()
            result["signal"] = signal_loss
            total = total + effective_signal_weight * signal_loss

        # Identity drift (Extension 1, complexity term)
        if self.lambda_identity > 0 and current_cms_params is not None and soul_cms_params is not None:
            id_drift = self.identity_loss(current_cms_params, soul_cms_params)
            result["identity"] = id_drift
            total = total + self.lambda_identity * id_drift

        result["total"] = total
        return result

    def anneal_signal(self, target: float, step: float = 0.01) -> None:
        """Gradually increase signal loss weight toward target."""
        self.lambda_signal = min(target, self.lambda_signal + step)

    def anneal_identity(self, target: float, step: float = 0.001) -> None:
        """Gradually increase identity loss weight toward target."""
        self.lambda_identity = min(target, self.lambda_identity + step)
