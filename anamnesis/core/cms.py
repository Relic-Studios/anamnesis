"""
Continuum Memory System (CMS) — Predictive coding MLP replacement.

Two-level architecture:
    Level 0: Frozen SwiGLU — exact copy of pre-trained MLP weights.
             Never updates. This is the model's stable knowledge.
    Level 1: Small residual MLP that learns via predictive coding.
             Starts near-identity (gate ≈ 0), gradually adapts.

Predictive coding mechanism (level 1 only):
    At each position t, the CMS output is used as a prediction of
    what position t+1's input will be. The error between prediction
    and actual next-position hidden state drives weight updates via
    analytical gradients. No torch.func, no extra allocations.

    error_t = output_t - input_{t+1}
    ∇W = analytical_grad(error_t)
    W = W - lr * ∇W

The conversation IS the training data. The model learns by predicting
what comes next in the hidden state sequence.

Paper reference: Equations 70-74 (CMS), predictive coding (Rao & Ballard 1999).
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CMSVariant(Enum):
    NESTED = "nested"
    SEQUENTIAL = "sequential"
    INDEPENDENT = "independent"


class CMSLevel(nn.Module):
    """
    A single CMS level.

    Level 0 (swiglu=True): Frozen SwiGLU, exact pre-trained MLP replica.
        y = down(silu(gate(x)) * up(x))

    Level 1 (swiglu=False): Residual memory with predictive coding.
        y = x + sigmoid(gate) * down(silu(up(x)))
        Learns by predicting next-position hidden states.
    """

    def __init__(
        self,
        dim: int,
        hidden_mult: float = 4.0,
        chunk_size: int = 1,
        swiglu: bool = False,
        activation: nn.Module | None = None,
        lr: float = 1e-5,
        max_grad_norm: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = int(dim * hidden_mult)
        self.chunk_size = chunk_size
        self.swiglu = swiglu
        self.learning_enabled = False  # Off by default; enable on level 1 only
        self.lr = lr
        self.max_grad_norm = max_grad_norm

        # MLP weights
        self.up_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        self.down_proj = nn.Linear(self.hidden_dim, dim, bias=False)
        self.act = activation or nn.SiLU()

        if swiglu:
            self.gate_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        else:
            # Start near-identity: sigmoid(-10) ≈ 0, so level contributes nothing initially
            self.residual_gate = nn.Parameter(torch.tensor(-10.0))

        # Predictive coding state
        self._last_output: Tensor | None = None
        self._grad_accum: dict[str, Tensor] = {}
        self._tokens_in_chunk: int = 0
        self._total_updates: int = 0

        # Float32 master weights for precision-safe updates.
        self._master_weights: dict[str, Tensor] = {}

        # Soul checkpoint — identity anchor weights to pull back toward
        self._soul_weights: dict[str, Tensor] = {}
        self.soul_pull_strength: float = 0.01  # How strongly to pull back toward soul
        self.max_drift: float = 0.5  # Max L2 drift before pull-back activates

        # Neutral drift
        self.drift_enabled = False
        self.drift_sigma = 1e-5 / max(chunk_size, 1)

    def _mlp_forward(self, x: Tensor) -> Tensor:
        """Raw MLP computation."""
        if self.swiglu:
            return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
        else:
            return self.down_proj(self.act(self.up_proj(x)))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass. Level 0 is pure SwiGLU, level 1 is gated residual.
        Predictive coding updates happen on level 1 during inference.
        """
        batch, seq_len, dim = x.shape

        if self.swiglu:
            out = self._mlp_forward(x)
        else:
            delta = self._mlp_forward(x) * torch.sigmoid(self.residual_gate)
            out = x + delta

        # Predictive coding: learn from next-position prediction error
        if self.learning_enabled and not torch.is_grad_enabled() and seq_len > 1:
            self._predictive_coding_update(x, out)

        # Neutral drift
        if self.drift_enabled and not self.training and self.chunk_size > 1:
            with torch.no_grad():
                out = out + torch.randn_like(out) * self.drift_sigma

        return out

    @torch.no_grad()
    def _predictive_coding_update(self, x: Tensor, out: Tensor) -> None:
        """
        Predictive coding: output at position t should predict input at position t+1.

        error_t = out[:, t, :] - x[:, t+1, :]

        We compute analytical gradients of ||error||² w.r.t. the residual MLP
        weights (up_proj, down_proj) and apply them.

        This only operates on the residual MLP path (non-swiglu levels).
        The gate modulates how much the level contributes, so gradients
        flow through the MLP weights that produce the delta.
        """
        batch, seq_len, dim = x.shape

        # Prediction targets: position t predicts position t+1
        predictions = out[:, :-1, :]   # (batch, seq-1, dim)
        targets = x[:, 1:, :]          # (batch, seq-1, dim)
        n = batch * (seq_len - 1)

        # Error signal
        err = 2.0 * (predictions - targets)  # (batch, seq-1, dim)

        # We need the intermediate activations from the MLP for the gradient
        # Recompute on the prediction positions (cheap — just matmuls)
        x_pred = x[:, :-1, :].reshape(n, dim)
        err_flat = err.reshape(n, dim)

        gate_val = torch.sigmoid(self.residual_gate)

        # Forward through residual MLP
        up_pre = x_pred @ self.up_proj.weight.T    # (n, hidden)
        up_act = F.silu(up_pre)                     # (n, hidden)
        h = up_act                                  # (n, hidden)
        # delta = gate * (down @ h), error is on (x + delta) - target

        # Gradient of error w.r.t. delta is just err * gate (since out = x + gate*delta)
        # But err already includes the full output, so d(loss)/d(delta) = err * gate_val
        d_delta = err_flat * gate_val  # (n, dim)

        # d(loss)/d(down_proj.weight) = d_delta^T @ h / n
        grad_down = (d_delta.T @ h) / n  # (dim, hidden)

        # Backprop through down_proj into h
        d_h = d_delta @ self.down_proj.weight  # (n, hidden)

        # SiLU derivative: silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        sig_up = torch.sigmoid(up_pre)
        d_up_pre = d_h * sig_up * (1.0 + up_pre * (1.0 - sig_up))

        # d(loss)/d(up_proj.weight) = d_up_pre^T @ x_pred / n
        grad_up = (d_up_pre.T @ x_pred) / n  # (hidden, dim)

        # Clip gradients
        grads = {"up_proj.weight": grad_up, "down_proj.weight": grad_down}
        for name, g in grads.items():
            g_norm = g.norm()
            if g_norm > self.max_grad_norm:
                g = g * (self.max_grad_norm / g_norm)

            if name not in self._grad_accum:
                self._grad_accum[name] = g
            else:
                self._grad_accum[name].add_(g)

        self._tokens_in_chunk += seq_len - 1

        if self._tokens_in_chunk >= self.chunk_size:
            self._apply_update()

    def save_soul(self) -> None:
        """Snapshot current weights as the soul anchor."""
        self._soul_weights = {
            "up_proj.weight": self.up_proj.weight.data.float().clone(),
            "down_proj.weight": self.down_proj.weight.data.float().clone(),
        }

    def _apply_update(self) -> None:
        """Apply accumulated gradients via float32 master weights.

        Includes lr decay (1/sqrt(t)) and soul checkpoint pull-back
        to prevent identity dissolution over long conversations.
        """
        self._total_updates += 1

        # Learning rate decay: aggressive early, gentle later
        decay_factor = 1.0 / (1.0 + 0.01 * self._total_updates)
        effective_lr = self.lr * decay_factor

        for name, param in [("up_proj.weight", self.up_proj.weight),
                            ("down_proj.weight", self.down_proj.weight)]:
            if name not in self._grad_accum:
                continue

            # Initialize master weight on first update
            if name not in self._master_weights:
                self._master_weights[name] = param.data.float().clone()

            # Update master (float32) — gradient step
            self._master_weights[name].sub_(
                self._grad_accum[name].float(), alpha=effective_lr
            )

            # Soul pull-back: if we've drifted too far, pull toward anchor
            if name in self._soul_weights:
                drift = (self._master_weights[name] - self._soul_weights[name]).norm().item()
                if drift > self.max_drift:
                    # Blend back toward soul: w = (1-α)w + α·soul
                    self._master_weights[name].lerp_(
                        self._soul_weights[name],
                        self.soul_pull_strength,
                    )

            # Sync to bf16
            param.data.copy_(self._master_weights[name].to(param.data.dtype))

        self._grad_accum.clear()
        self._tokens_in_chunk = 0

    def should_update(self, token_position: int) -> bool:
        return token_position % self.chunk_size == 0

    def reset_state(self) -> None:
        """Reset learning state (keeps soul checkpoint)."""
        self._last_output = None
        self._grad_accum.clear()
        self._master_weights.clear()
        self._tokens_in_chunk = 0
        self._total_updates = 0

    @property
    def surprise(self) -> float:
        """Last measured surprise (L2 norm of gradient accumulator)."""
        if not self._grad_accum:
            return 0.0
        return sum(g.norm().item() for g in self._grad_accum.values())

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, hidden={self.hidden_dim}, "
            f"chunk={self.chunk_size}, swiglu={self.swiglu}, "
            f"learning={self.learning_enabled}"
        )


class ContinuumMemorySystem(nn.Module):
    """
    Continuum Memory System — 2-level MLP replacement with predictive coding.

    Level 0: Frozen SwiGLU (exact pre-trained MLP, never updates)
    Level 1: Residual memory (learns via predictive coding during inference)
    """

    def __init__(
        self,
        dim: int,
        num_levels: int = 2,
        chunk_sizes: list[int] | None = None,
        hidden_mult: float | list[float] = 4.0,
        variant: CMSVariant = CMSVariant.NESTED,
        activation: nn.Module | None = None,
        lr: float = 1e-5,
        max_grad_norm: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels
        self.variant = variant

        if chunk_sizes is None:
            chunk_sizes = [1, 32, 256, 2048][:num_levels]
        assert len(chunk_sizes) == num_levels
        assert chunk_sizes == sorted(chunk_sizes)
        self.chunk_sizes = chunk_sizes

        if isinstance(hidden_mult, (int, float)):
            hidden_mults = [hidden_mult] * num_levels
        else:
            assert len(hidden_mult) == num_levels
            hidden_mults = list(hidden_mult)
        self.hidden_mults = hidden_mults

        self.levels = nn.ModuleList([
            CMSLevel(
                dim=dim, hidden_mult=hm, chunk_size=cs,
                swiglu=(i == 0), activation=activation,
                lr=lr, max_grad_norm=max_grad_norm,
            )
            for i, (cs, hm) in enumerate(zip(chunk_sizes, hidden_mults))
        ])

        # Level 0 is frozen by default, level 1+ learns
        self.levels[0].learning_enabled = False
        for level in self.levels[1:]:
            level.learning_enabled = True

        if variant == CMSVariant.INDEPENDENT:
            self.agg_weights = nn.Parameter(torch.ones(num_levels) / num_levels)

        self._token_position: int = 0

    def forward(self, x: Tensor) -> Tensor:
        if self.variant == CMSVariant.INDEPENDENT:
            return self._forward_independent(x)
        return self._forward_chain(x)

    def _forward_chain(self, x: Tensor) -> Tensor:
        current = x
        for level in self.levels:
            current = level(current)
        return current

    def _forward_independent(self, x: Tensor) -> Tensor:
        weights = torch.softmax(self.agg_weights, dim=0)
        outputs = [level(x) for level in self.levels]
        result = torch.zeros_like(x)
        for w, out in zip(weights, outputs):
            result = result + w * out
        return result

    def save_soul(self) -> None:
        """Save current weights as soul anchor across all levels."""
        for level in self.levels:
            level.save_soul()

    def enable_drift(self, enabled: bool = True) -> None:
        for level in self.levels:
            level.drift_enabled = enabled

    def enable_learning(self, enabled: bool = True, levels: list[int] | None = None) -> None:
        """Enable/disable learning for specific levels."""
        for i, level in enumerate(self.levels):
            if levels is None or i in levels:
                level.learning_enabled = enabled

    def get_surprise(self) -> list[float]:
        """Get per-level surprise values."""
        return [level.surprise for level in self.levels]

    def reset_learning_state(self) -> None:
        """Reset all levels' learning state."""
        for level in self.levels:
            level.reset_state()

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, levels={self.num_levels}, "
            f"variant={self.variant.value}, chunks={self.chunk_sizes}"
        )
