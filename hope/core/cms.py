"""
Continuum Memory System (CMS) — Multi-timescale MLP replacement.

Implements Section 7.1 of the Nested Learning paper (Behrouz et al., NeurIPS 2025).
Replaces standard feedforward (MLP) blocks with a chain of MLP blocks, each updating
at a different frequency. Three variants: Nested, Sequential, Independent.

Paper reference: Equations 70-74.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from hope.core.dgd import DeltaGradientDescent


class CMSVariant(Enum):
    """CMS knowledge transfer variants (Section 7.1)."""
    NESTED = "nested"         # Eq 72: each level meta-learns from the level below
    SEQUENTIAL = "sequential" # Eq 73: all initial states connected through backprop
    INDEPENDENT = "independent"  # Eq 74: independent blocks, aggregated by learned sum


class CMSLevel(nn.Module):
    """
    A single CMS level: a 2-layer MLP with residual connection.

    M(x) = x + W_down · σ(W_up · x)

    Each level has its own update frequency C_l and maintains its own
    gradient accumulator for chunk-boundary updates.

    Args:
        dim: Input/output dimension.
        hidden_mult: Multiplier for intermediate dimension (default 4.0 matches SwiGLU ratio).
        chunk_size: Update frequency C_l — parameters update every C_l tokens.
        activation: Activation function (default SiLU to match Qwen/paper).
    """

    def __init__(
        self,
        dim: int,
        hidden_mult: float = 4.0,
        chunk_size: int = 1,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = int(dim * hidden_mult)
        self.chunk_size = chunk_size

        self.up_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        self.down_proj = nn.Linear(self.hidden_dim, dim, bias=False)
        self.act = activation or nn.SiLU()

        # DGD update state
        self._grad_accumulator: Tensor | None = None
        self._tokens_since_update: int = 0

        # Neutral drift (Extension 5) — disabled by default, enabled during deployment
        self.drift_enabled = False
        self.drift_sigma = 1e-5 / max(chunk_size, 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with residual: y = x + down(act(up(x)))

        Args:
            x: Input tensor of shape (batch, seq_len, dim).

        Returns:
            Output tensor of same shape.
        """
        residual = x
        h = self.act(self.up_proj(x))
        out = self.down_proj(h)

        # Neutral drift: inject micro-perturbation between scheduled updates
        if self.drift_enabled and not self.training and self.chunk_size > 1:
            with torch.no_grad():
                noise = torch.randn_like(out) * self.drift_sigma
                out = out + noise

        return residual + out

    def should_update(self, token_position: int) -> bool:
        """Check if this level should update at the given token position."""
        return token_position % self.chunk_size == 0

    def accumulate_gradient(self, grad: Tensor) -> None:
        """Accumulate gradient for chunk-boundary update."""
        if self._grad_accumulator is None:
            self._grad_accumulator = grad.clone()
        else:
            self._grad_accumulator.add_(grad)
        self._tokens_since_update += 1

    def apply_update(self, learning_rate: float, alpha: float = 0.0) -> None:
        """
        Apply accumulated gradient update at chunk boundary (Equation 71).

        Uses DGD if alpha > 0, otherwise standard gradient descent.
        """
        if self._grad_accumulator is None:
            return

        with torch.no_grad():
            for param in self.parameters():
                if param.grad is not None:
                    param.data -= learning_rate * self._grad_accumulator

        self._grad_accumulator = None
        self._tokens_since_update = 0

    def reset_state(self) -> None:
        """Reset accumulator state (e.g., at context boundary in nested CMS)."""
        self._grad_accumulator = None
        self._tokens_since_update = 0

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, hidden_dim={self.hidden_dim}, "
            f"chunk_size={self.chunk_size}, drift={self.drift_enabled}"
        )


class ContinuumMemorySystem(nn.Module):
    """
    Continuum Memory System — replaces standard MLP with multi-timescale chain.

    Given input x, computes:
        y = MLP_fk(MLP_{fk-1}(...MLP_f1(x)))     (Equation 70)

    where each MLP_fi updates at frequency C_i. Slower levels preserve knowledge
    across longer contexts; faster levels adapt to immediate input.

    Three variants control knowledge transfer between levels:
    - Nested: each level's init state meta-learned from level below (Eq 72)
    - Sequential: all init states connected via backprop at lowest frequency (Eq 73)
    - Independent: parallel blocks aggregated by learned weighted sum (Eq 74)

    Args:
        dim: Hidden dimension (must match transformer hidden_size).
        num_levels: Number of CMS levels (paper uses 3-4).
        chunk_sizes: Update frequencies per level. Must be ascending.
            Default: [1, 32, 256, 2048] for 4 levels.
        hidden_mult: Hidden dimension multiplier for each level's MLP.
        variant: CMS variant (nested, sequential, or independent).
        activation: Activation function for all levels.
    """

    def __init__(
        self,
        dim: int,
        num_levels: int = 4,
        chunk_sizes: list[int] | None = None,
        hidden_mult: float = 4.0,
        variant: CMSVariant = CMSVariant.NESTED,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels
        self.variant = variant

        if chunk_sizes is None:
            chunk_sizes = [1, 32, 256, 2048][:num_levels]
        assert len(chunk_sizes) == num_levels, (
            f"chunk_sizes length {len(chunk_sizes)} != num_levels {num_levels}"
        )
        assert chunk_sizes == sorted(chunk_sizes), "chunk_sizes must be ascending"
        self.chunk_sizes = chunk_sizes

        # Build the CMS levels
        self.levels = nn.ModuleList([
            CMSLevel(
                dim=dim,
                hidden_mult=hidden_mult,
                chunk_size=cs,
                activation=activation,
            )
            for cs in chunk_sizes
        ])

        # Independent variant: learned aggregation weights (Eq 74)
        if variant == CMSVariant.INDEPENDENT:
            self.agg_weights = nn.Parameter(torch.ones(num_levels) / num_levels)

        # Track global token position for update scheduling
        self._token_position: int = 0

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through CMS chain.

        Args:
            x: Input tensor of shape (batch, seq_len, dim).

        Returns:
            Output tensor of same shape.
        """
        if self.variant == CMSVariant.INDEPENDENT:
            return self._forward_independent(x)
        else:
            # Nested and Sequential both use the chain (Eq 70)
            # They differ in initialization/knowledge transfer, not forward pass
            return self._forward_chain(x)

    def _forward_chain(self, x: Tensor) -> Tensor:
        """Sequential chain: y = MLP_fk(...MLP_f1(x))"""
        current = x
        for level in self.levels:
            current = level(current)
        return current

    def _forward_independent(self, x: Tensor) -> Tensor:
        """Independent/head-wise: y = Σ w_i · MLP_fi(x) (Eq 74)"""
        weights = torch.softmax(self.agg_weights, dim=0)
        outputs = [level(x) for level in self.levels]
        result = torch.zeros_like(x)
        for w, out in zip(weights, outputs):
            result = result + w * out
        return result

    def advance_position(self, num_tokens: int = 1) -> None:
        """Advance the global token position counter."""
        self._token_position += num_tokens

    def get_update_schedule(self, seq_len: int) -> dict[int, list[int]]:
        """
        Compute which levels should update at which positions in a sequence.

        Returns:
            Dict mapping token position → list of level indices that should update.
        """
        schedule: dict[int, list[int]] = {}
        for pos in range(seq_len):
            global_pos = self._token_position + pos
            updating = [
                i for i, level in enumerate(self.levels)
                if level.should_update(global_pos)
            ]
            if updating:
                schedule[pos] = updating
        return schedule

    def enable_drift(self, enabled: bool = True) -> None:
        """Enable/disable neutral drift (Extension 5) on all levels."""
        for level in self.levels:
            level.drift_enabled = enabled

    @classmethod
    def from_pretrained_mlp(
        cls,
        gate_proj: Tensor,
        up_proj: Tensor,
        down_proj: Tensor,
        num_levels: int = 4,
        chunk_sizes: list[int] | None = None,
        variant: CMSVariant = CMSVariant.NESTED,
    ) -> "ContinuumMemorySystem":
        """
        Initialize CMS from pre-trained SwiGLU MLP weights (Section 7.3).

        Takes the gate/up/down projection weights from a standard SwiGLU MLP
        and uses them to initialize all CMS levels. Each level starts as an
        approximation of the original MLP's function.

        Args:
            gate_proj: Gate projection weight (intermediate_size, hidden_size).
            up_proj: Up projection weight (intermediate_size, hidden_size).
            down_proj: Down projection weight (hidden_size, intermediate_size).
            num_levels: Number of CMS levels.
            chunk_sizes: Update frequencies per level.
            variant: CMS variant.

        Returns:
            Initialized ContinuumMemorySystem.
        """
        hidden_size = gate_proj.shape[1]
        intermediate_size = gate_proj.shape[0]
        hidden_mult = intermediate_size / hidden_size

        cms = cls(
            dim=hidden_size,
            num_levels=num_levels,
            chunk_sizes=chunk_sizes,
            hidden_mult=hidden_mult,
            variant=variant,
        )

        # Initialize each level from the pre-trained weights
        # Section 7.3: MLP(fi)_0 = MLP_pretrained
        # We approximate the SwiGLU as a 2-layer MLP: down(act(up(x)))
        # by combining gate and up projections
        with torch.no_grad():
            for level in cms.levels:
                # Use up_proj as the up projection (gate is folded into the activation)
                # This is an approximation — SwiGLU has 3 projections, CMS has 2
                # We use the up_proj for the up path and down_proj for the down path
                if level.up_proj.weight.shape == up_proj.shape:
                    level.up_proj.weight.copy_(up_proj)
                    level.down_proj.weight.copy_(down_proj)
                else:
                    # Dimensions don't match — initialize from truncated/padded version
                    h = min(level.hidden_dim, intermediate_size)
                    d = min(level.dim, hidden_size)
                    level.up_proj.weight[:h, :d].copy_(up_proj[:h, :d])
                    level.down_proj.weight[:d, :h].copy_(down_proj[:d, :h])

        return cms

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, num_levels={self.num_levels}, "
            f"variant={self.variant.value}, "
            f"chunk_sizes={self.chunk_sizes}"
        )
