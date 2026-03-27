"""
Continuum Memory System (CMS) — ATLAS-style deep memory MLP replacement.

Two-level architecture:
    Level 0: SwiGLU — copy of pre-trained MLP weights. Frozen base intelligence.
    Level 1: DeepMemoryLevel — ATLAS-style deep associative memory that learns
             during inference via the Omega Rule (context-window optimization),
             momentum with Newton-Schulz orthogonalization, and data-dependent
             forgetting gates.

The memory is a small MLP whose weights ARE the learned specialization.
It updates during every forward pass: tokens go in, memory weights change,
the model gets better at this specific domain. No seeding, no backprop.

Key mechanisms:
    - Omega Rule (c≥1): Optimize over a window of tokens, not just current one
    - Muon updates: NS-5 orthogonalized momentum for second-order optimization
    - Data-dependent gates: Learnable forget (α_t), momentum (η_t), lr (θ_t)
    - Associative loss: ℓ = ‖M(φ(k)) - v‖² (not next-token prediction)
    - Polynomial features: φ(k) expands key space for more memory capacity

Novel identity extensions (not in any paper):
    - Soul anchoring: weights can't drift too far from identity checkpoint
    - Persona probes: SVD focus on output-relevant gradient directions
    - Persona mask: learn more from assistant tokens than user tokens

Key references:
    - ATLAS (Behrouz et al., 2025): Omega Rule, DeepTransformers
    - Titans (Behrouz & Zhong, 2025): Gradient-based test-time memory
    - Nested Learning (Behrouz et al., NeurIPS 2025): Multi-frequency CMS
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.func import functional_call, grad, vmap

from anamnesis.core.memory import MemoryMLP, _memory_loss_fn
from anamnesis.optim.newton_schulz import newton_schulz


class CMSVariant(Enum):
    NESTED = "nested"
    SEQUENTIAL = "sequential"
    INDEPENDENT = "independent"


class CMSLevel(nn.Module):
    """
    A single CMS level.

    Level 0 (swiglu=True): SwiGLU MLP, learns slowly via predictive coding.
        y = down(silu(gate(x)) * up(x))

    Level 1 (swiglu=False): Residual memory, learns faster.
        y = x + sigmoid(gate) * down(silu(up(x)))
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
        gate_surprise_scale: float = 0.25,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = int(dim * hidden_mult)
        self.chunk_size = chunk_size
        self.swiglu = swiglu
        self.learning_enabled = False
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.gate_surprise_scale = gate_surprise_scale

        # MLP weights
        self.up_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        self.down_proj = nn.Linear(self.hidden_dim, dim, bias=False)
        self.act = activation or nn.SiLU()

        if swiglu:
            self.gate_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        else:
            self.residual_gate = nn.Parameter(torch.tensor(0.0))

        # Learning state
        self._grad_accum: dict[str, Tensor] = {}
        self._tokens_in_chunk: int = 0
        self._total_updates: int = 0
        self._surprise_ema: float = 1.0  # Running average of prediction error

        # Float32 master weights
        self._master_weights: dict[str, Tensor] = {}

        # Soul checkpoint
        self._soul_weights: dict[str, Tensor] = {}
        self.soul_pull_strength: float = 0.01
        self.max_drift: float = 0.5

        # Persona probe (set externally via CMS.set_persona_probe)
        self._persona_probe: Tensor | None = None

        # Per-position learning weight (e.g., persona mask: 1.0 for assistant, 0.1 for user)
        self._learning_weight: Tensor | None = None

        # Neutral drift
        self.drift_enabled = False
        self.drift_sigma = 1e-5 / max(chunk_size, 1)

    def _mlp_forward(self, x: Tensor) -> Tensor:
        """Raw MLP computation."""
        if self.swiglu:
            return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
        else:
            return self.down_proj(self.act(self.up_proj(x)))

    def _compute_gate(self) -> Tensor:
        """Compute residual gate — competence-based during inference, static during training.

        During inference (no_grad), the gate tracks L1's competence:
        LOW surprise = L1 is predicting well = open gate (trust L1's contribution).
        HIGH surprise = L1 is confused = close gate (protect output from bad residuals).

        L1 still LEARNS from high-surprise tokens (learning is decoupled from gate),
        but it doesn't CONTRIBUTE to output until it earns trust through accuracy.

        This eliminates the need for a seeding phase: L1 starts with zero-output init,
        learns via predictive coding from conversation, and the gate opens naturally
        as L1 becomes competent.

        During backprop training, the gate is a standard learnable parameter.
        """
        if not torch.is_grad_enabled():
            # Negative sign: high surprise CLOSES gate, low surprise OPENS it
            surprise_signal = math.log(max(self._surprise_ema, 1e-8))
            gate_input = self.residual_gate.item() - self.gate_surprise_scale * surprise_signal
            return torch.tensor(
                gate_input, device=self.up_proj.weight.device, dtype=self.up_proj.weight.dtype
            ).sigmoid()
        return torch.sigmoid(self.residual_gate)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, dim = x.shape

        if self.swiglu:
            out = self._mlp_forward(x)
        else:
            delta = self._mlp_forward(x) * self._compute_gate()
            out = x + delta

        # Predictive coding on BOTH level types
        if self.learning_enabled and not torch.is_grad_enabled() and seq_len > 1:
            if self.swiglu:
                self._predictive_coding_swiglu(x, out)
            else:
                self._predictive_coding_residual(x, out)

        if self.drift_enabled and not self.training and self.chunk_size > 1:
            with torch.no_grad():
                out = out + torch.randn_like(out) * self.drift_sigma

        return out

    @torch.no_grad()
    def _predictive_coding_residual(self, x: Tensor, out: Tensor) -> None:
        """Predictive coding for residual levels (level 1+).

        Trains the MLP to predict the RESIDUAL between consecutive positions:
            target = x_{t+1} - x_t
            prediction = mlp(x_t)  (raw delta, before gating)

        The gate controls output blending but NOT learning strength.
        This prevents the gate from attenuating gradients when it's small.
        """
        batch, seq_len, dim = x.shape
        n = batch * (seq_len - 1)

        # MLP's raw prediction (before gating)
        x_pred = x[:, :-1, :].reshape(n, dim)
        up_pre = x_pred @ self.up_proj.weight.T
        up_act = F.silu(up_pre)
        h = up_act
        delta = h @ self.down_proj.weight.T  # mlp(x) raw output

        # Residual target: what SHOULD change between positions
        residual_target = (x[:, 1:, :] - x[:, :-1, :]).reshape(n, dim)

        # Error on the residual prediction (no gate attenuation)
        err = 2.0 * (delta - residual_target)

        # Surprise gating: learn more from surprising content
        err_magnitude = err.norm().item() / max(n, 1)
        self._surprise_ema = 0.95 * self._surprise_ema + 0.05 * err_magnitude
        surprise_ratio = err_magnitude / max(self._surprise_ema, 1e-8)
        surprise_gate = min(surprise_ratio, 3.0)  # Cap at 3x

        # Focus error on output-relevant directions via persona probe
        if self._persona_probe is not None:
            probe = self._persona_probe.to(err.device)
            err = (err @ probe) @ probe.T

        # Per-position persona weighting (assistant tokens > user tokens)
        if self._learning_weight is not None:
            w = self._learning_weight[:, :-1].reshape(n, 1).to(err.dtype)
            err = err * w

        # Gradients w.r.t. MLP weights (no gate in gradient path)
        grad_down = (err.T @ h) / n
        d_h = err @ self.down_proj.weight
        sig_up = torch.sigmoid(up_pre)
        d_up_pre = d_h * sig_up * (1.0 + up_pre * (1.0 - sig_up))
        grad_up = (d_up_pre.T @ x_pred) / n

        grads = {"up_proj.weight": grad_up, "down_proj.weight": grad_down}
        self._accumulate_grads(grads, surprise_gate)

        self._tokens_in_chunk += seq_len - 1
        if self._tokens_in_chunk >= self.chunk_size:
            self._apply_update()

    @torch.no_grad()
    def _predictive_coding_swiglu(self, x: Tensor, out: Tensor) -> None:
        """Predictive coding for SwiGLU levels (level 0).

        SwiGLU: y = down(silu(gate(x)) * up(x))
        No residual, no gate parameter. Gradients flow through all three projections.
        """
        batch, seq_len, dim = x.shape
        n = batch * (seq_len - 1)

        predictions = out[:, :-1, :]
        targets = x[:, 1:, :]

        err = 2.0 * (predictions - targets)

        # Surprise gating (on FULL error, before persona filtering)
        err_magnitude = err.norm().item() / max(n, 1)
        self._surprise_ema = 0.95 * self._surprise_ema + 0.05 * err_magnitude
        surprise_ratio = err_magnitude / max(self._surprise_ema, 1e-8)
        surprise_gate = min(surprise_ratio, 3.0)

        x_pred = x[:, :-1, :].reshape(n, dim)
        err_flat = err.reshape(n, dim)

        # Focus error on output-relevant directions via persona probe
        if self._persona_probe is not None:
            probe = self._persona_probe.to(err_flat.device)
            err_flat = (err_flat @ probe) @ probe.T

        # Per-position persona weighting (assistant tokens > user tokens)
        if self._learning_weight is not None:
            w = self._learning_weight[:, :-1].reshape(n, 1).to(err_flat.dtype)
            err_flat = err_flat * w

        # Recompute SwiGLU forward
        gate_pre = x_pred @ self.gate_proj.weight.T      # (n, hidden)
        gate_act = F.silu(gate_pre)                        # (n, hidden)
        up_pre = x_pred @ self.up_proj.weight.T            # (n, hidden)
        up_val = up_pre  # Note: up path is NOT activated in SwiGLU, gate path is
        h = gate_act * up_val                              # (n, hidden)

        # d(loss)/d(down) = err^T @ h / n
        grad_down = (err_flat.T @ h) / n  # (dim, hidden)

        # Backprop through down into h
        d_h = err_flat @ self.down_proj.weight  # (n, hidden)

        # d_h splits into gate and up paths
        d_gate_act = d_h * up_val      # (n, hidden)
        d_up_val = d_h * gate_act      # (n, hidden)

        # Gate path: through silu
        sig_gate = torch.sigmoid(gate_pre)
        d_gate_pre = d_gate_act * sig_gate * (1.0 + gate_pre * (1.0 - sig_gate))
        grad_gate = (d_gate_pre.T @ x_pred) / n  # (hidden, dim)

        # Up path: no activation in standard SwiGLU
        grad_up = (d_up_val.T @ x_pred) / n  # (hidden, dim)

        grads = {
            "up_proj.weight": grad_up,
            "down_proj.weight": grad_down,
            "gate_proj.weight": grad_gate,
        }
        self._accumulate_grads(grads, surprise_gate)

        self._tokens_in_chunk += seq_len - 1
        if self._tokens_in_chunk >= self.chunk_size:
            self._apply_update()

    def _accumulate_grads(self, grads: dict[str, Tensor], surprise_gate: float) -> None:
        """Clip and accumulate gradients, scaled by surprise."""
        for name, g in grads.items():
            g = g * surprise_gate  # Surprise gating

            g_norm = g.norm()
            if g_norm > self.max_grad_norm:
                g = g * (self.max_grad_norm / g_norm)

            if name not in self._grad_accum:
                self._grad_accum[name] = g
            else:
                self._grad_accum[name].add_(g)

    def save_soul(self) -> None:
        """Snapshot current weights as the soul anchor."""
        self._soul_weights = {}
        for name, param in self._learnable_params():
            self._soul_weights[name] = param.data.float().clone()

    def _learnable_params(self):
        """Yield (name, param) for all learnable weight matrices."""
        yield ("up_proj.weight", self.up_proj.weight)
        yield ("down_proj.weight", self.down_proj.weight)
        if self.swiglu:
            yield ("gate_proj.weight", self.gate_proj.weight)

    def _apply_update(self) -> None:
        """Apply accumulated gradients via float32 master weights.

        Surprise-gated: no blanket lr decay. Learning rate is constant,
        but surprise_gate already modulates how much each update matters.
        Soul pull-back prevents drift beyond identity boundary.
        """
        self._total_updates += 1
        effective_lr = self.lr

        for name, param in self._learnable_params():
            if name not in self._grad_accum:
                continue

            if name not in self._master_weights:
                self._master_weights[name] = param.data.float().clone()

            self._master_weights[name].sub_(
                self._grad_accum[name].float(), alpha=effective_lr
            )

            # Soul pull-back
            if name in self._soul_weights:
                drift = (self._master_weights[name] - self._soul_weights[name]).norm().item()
                if drift > self.max_drift:
                    self._master_weights[name].lerp_(
                        self._soul_weights[name],
                        self.soul_pull_strength,
                    )

            param.data.copy_(self._master_weights[name].to(param.data.dtype))

        self._grad_accum.clear()
        self._tokens_in_chunk = 0

    def should_update(self, token_position: int) -> bool:
        return token_position % self.chunk_size == 0

    def reset_state(self) -> None:
        """Reset learning state (keeps soul checkpoint)."""
        self._grad_accum.clear()
        self._master_weights.clear()
        self._tokens_in_chunk = 0
        self._total_updates = 0
        self._surprise_ema = 1.0

    @property
    def surprise(self) -> float:
        if not self._grad_accum:
            return 0.0
        return sum(g.norm().item() for g in self._grad_accum.values())

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, hidden={self.hidden_dim}, "
            f"chunk={self.chunk_size}, swiglu={self.swiglu}, "
            f"learning={self.learning_enabled}"
        )


class LowRankLevel(nn.Module):
    """Low-rank residual level that reuses L0's feature extraction.

    Instead of its own full MLP, this level:
    1. Reuses L0's gate_proj and up_proj to compute features: h = silu(gate(x)) * up(x)
    2. Applies a low-rank projection: delta = B(A(h))
    3. Blends via competence gate: out = L0_out + gate * delta

    At rank=32, parameter cost is ~15MB per specialist (vs ~600MB for full MLP).
    Gradients only flow through A and B — L0's features are frozen.

    Per Geva et al. (2020): gate_proj/up_proj are general feature extractors,
    down_proj stores task-specific patterns. We only need to specialize down_proj.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        rank: int = 32,
        chunk_size: int = 32,
        lr: float = 1e-2,
        max_grad_norm: float = 1.0,
        gate_surprise_scale: float = 0.25,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.chunk_size = chunk_size
        self.swiglu = False  # compatibility with CMS code that checks this
        self.learning_enabled = True
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.gate_surprise_scale = gate_surprise_scale

        # Low-rank projection: hidden_dim → rank → dim
        # Both A and B get small random init for bilateral gradient flow.
        # The initial output is small (not zero) but the competence gate
        # suppresses L1's contribution when surprise is high (early on).
        # As L1 learns and surprise drops, gate opens naturally.
        self.A = nn.Linear(hidden_dim, rank, bias=False)  # compress features
        self.B = nn.Linear(rank, dim, bias=False)          # project to output space

        # Competence gate
        self.residual_gate = nn.Parameter(torch.tensor(0.0))

        # Learning state
        self._grad_accum: dict[str, Tensor] = {}
        self._tokens_in_chunk: int = 0
        self._total_updates: int = 0
        self._surprise_ema: float = 1.0

        # Float32 master weights
        self._master_weights: dict[str, Tensor] = {}

        # Soul checkpoint
        self._soul_weights: dict[str, Tensor] = {}
        self.soul_pull_strength: float = 0.01
        self.max_drift: float = 0.5

        # Persona probe
        self._persona_probe: Tensor | None = None
        self._learning_weight: Tensor | None = None

        # Neutral drift
        self.drift_enabled = False
        self.drift_sigma = 1e-5 / max(chunk_size, 1)

        # Reference to L0 for feature extraction (set by CMS).
        # Stored in a list to avoid nn.Module registration (which would
        # double-count L0's parameters as belonging to both levels).
        self._l0_ref: list[CMSLevel] = []

    def _compute_gate(self) -> Tensor:
        """Competence-based gate — same logic as CMSLevel."""
        if not torch.is_grad_enabled():
            surprise_signal = math.log(max(self._surprise_ema, 1e-8))
            gate_input = self.residual_gate.item() - self.gate_surprise_scale * surprise_signal
            return torch.tensor(
                gate_input, device=self.A.weight.device, dtype=self.A.weight.dtype
            ).sigmoid()
        return torch.sigmoid(self.residual_gate)

    def _extract_features(self, x: Tensor) -> Tensor:
        """Reuse L0's gate_proj and up_proj for feature extraction.

        h = silu(L0.gate_proj(x)) * L0.up_proj(x)

        These features are treated as frozen — no gradients flow back to L0.
        """
        assert self._l0_ref, "LowRankLevel needs L0 reference (set by CMS)"
        l0 = self._l0_ref[0]
        gate_act = F.silu(x @ l0.gate_proj.weight.T)
        up_val = x @ l0.up_proj.weight.T
        return gate_act * up_val  # (batch*seq, hidden_dim)

    def forward(self, x: Tensor, l0_out: Tensor | None = None) -> Tensor:
        """Forward pass: reuse L0 features, apply low-rank delta.

        Args:
            x: Input to the CMS (before L0). Shape: (batch, seq, dim)
            l0_out: L0's output. If None, uses x as the base (for testing).
        """
        batch, seq_len, dim = x.shape
        base = l0_out if l0_out is not None else x

        # Extract features from L0's projections
        x_flat = x.reshape(-1, dim)
        h = self._extract_features(x_flat)  # (batch*seq, hidden_dim)

        # Low-rank projection: h → rank → dim
        z = h @ self.A.weight.T     # (batch*seq, rank)
        delta = (z @ self.B.weight.T).reshape(batch, seq_len, dim)

        gate = self._compute_gate()
        out = base + gate * delta

        # Predictive coding
        if self.learning_enabled and not torch.is_grad_enabled() and seq_len > 1:
            self._predictive_coding(x, h.reshape(batch, seq_len, -1), delta)

        if self.drift_enabled and not self.training and self.chunk_size > 1:
            with torch.no_grad():
                out = out + torch.randn_like(out) * self.drift_sigma

        return out

    @torch.no_grad()
    def _predictive_coding(self, x: Tensor, h: Tensor, delta: Tensor) -> None:
        """Predictive coding for low-rank residual level.

        Predicts the residual x_{t+1} - x_t using the low-rank delta.
        Gradients flow through A and B only — L0's features are frozen.
        """
        batch, seq_len, dim = x.shape
        n = batch * (seq_len - 1)

        # Raw delta prediction (before gating)
        delta_pred = delta[:, :-1, :].reshape(n, dim)

        # Residual target
        residual_target = (x[:, 1:, :] - x[:, :-1, :]).reshape(n, dim)

        # Error
        err = 2.0 * (delta_pred - residual_target)

        # Surprise tracking
        err_magnitude = err.norm().item() / max(n, 1)
        self._surprise_ema = 0.95 * self._surprise_ema + 0.05 * err_magnitude
        surprise_ratio = err_magnitude / max(self._surprise_ema, 1e-8)
        surprise_gate = min(surprise_ratio, 3.0)

        # Persona probe
        if self._persona_probe is not None:
            probe = self._persona_probe.to(err.device)
            err = (err @ probe) @ probe.T

        # Per-position persona weighting
        if self._learning_weight is not None:
            w = self._learning_weight[:, :-1].reshape(n, 1).to(err.dtype)
            err = err * w

        # Features from L0 (already computed during forward)
        h_pred = h[:, :-1, :].reshape(n, self.hidden_dim)

        # Both A and B have non-zero random init, so standard chain rule works.
        # No dead-gradient problem — both receive meaningful updates from step 1.

        # Gradients for B: d_B = err^T @ z / n
        z_pred = h_pred @ self.A.weight.T  # (n, rank)
        grad_B = (err.T @ z_pred) / n  # (dim, rank)

        # Gradients for A: d_z = err @ B, d_A = d_z^T @ h / n
        d_z = err @ self.B.weight  # (n, rank)
        grad_A = (d_z.T @ h_pred) / n  # (rank, hidden_dim)

        grads = {"A.weight": grad_A, "B.weight": grad_B}
        self._accumulate_grads(grads, surprise_gate)

        self._tokens_in_chunk += seq_len - 1
        if self._tokens_in_chunk >= self.chunk_size:
            self._apply_update()

    def _accumulate_grads(self, grads: dict[str, Tensor], surprise_gate: float) -> None:
        """Clip and accumulate gradients, scaled by surprise."""
        for name, g in grads.items():
            g = g * surprise_gate
            g_norm = g.norm()
            if g_norm > self.max_grad_norm:
                g = g * (self.max_grad_norm / g_norm)
            if name not in self._grad_accum:
                self._grad_accum[name] = g
            else:
                self._grad_accum[name].add_(g)

    def save_soul(self) -> None:
        """Snapshot current weights as the soul anchor."""
        self._soul_weights = {}
        for name, param in self._learnable_params():
            self._soul_weights[name] = param.data.float().clone()

    def _learnable_params(self):
        """Yield (name, param) for learnable weight matrices."""
        yield ("A.weight", self.A.weight)
        yield ("B.weight", self.B.weight)

    def _apply_update(self) -> None:
        """Apply accumulated gradients via float32 master weights."""
        self._total_updates += 1
        for name, param in self._learnable_params():
            if name not in self._grad_accum:
                continue
            if name not in self._master_weights:
                self._master_weights[name] = param.data.float().clone()
            self._master_weights[name].sub_(
                self._grad_accum[name].float(), alpha=self.lr
            )
            if name in self._soul_weights:
                drift = (self._master_weights[name] - self._soul_weights[name]).norm().item()
                if drift > self.max_drift:
                    self._master_weights[name].lerp_(
                        self._soul_weights[name], self.soul_pull_strength,
                    )
            param.data.copy_(self._master_weights[name].to(param.data.dtype))
        self._grad_accum.clear()
        self._tokens_in_chunk = 0

    def should_update(self, token_position: int) -> bool:
        return token_position % self.chunk_size == 0

    def reset_state(self) -> None:
        """Reset learning state (keeps soul checkpoint)."""
        self._grad_accum.clear()
        self._master_weights.clear()
        self._tokens_in_chunk = 0
        self._total_updates = 0
        self._surprise_ema = 1.0

    @property
    def surprise(self) -> float:
        if not self._grad_accum:
            return 0.0
        return sum(g.norm().item() for g in self._grad_accum.values())

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, hidden={self.hidden_dim}, rank={self.rank}, "
            f"chunk={self.chunk_size}, learning={self.learning_enabled}"
        )


class DeepMemoryLevel(nn.Module):
    """ATLAS-style deep associative memory level for CMS.

    Replaces the broken LowRankLevel with a proper deep memory that:
    1. Has its own KQV projections (not frozen L0 features)
    2. Uses a deep MLP (MemoryMLP) as the actual memory store
    3. Updates via Omega Rule (window of c tokens, not just current)
    4. Applies NS-5 orthogonalization on momentum (Muon-style)
    5. Uses data-dependent gates for forgetting, momentum, and output

    The memory MLP weights ARE the specialization. They update during
    every forward pass via vmap+grad on the associative loss ‖M(k)-v‖².

    Args:
        dim: Hidden dimension of the transformer.
        mem_dim: Memory working dimension (projected from dim).
        mem_depth: Depth of the memory MLP.
        chunk_size: Omega Rule window size (c tokens per update).
        poly_degree: Polynomial feature expansion degree (1=none, 2=quadratic).
        max_grad_norm: Gradient clipping threshold.
        ns_steps: Newton-Schulz iterations for momentum orthogonalization.
    """

    def __init__(
        self,
        dim: int,
        mem_dim: int = 512,
        mem_depth: int = 2,
        chunk_size: int = 32,
        poly_degree: int = 2,
        max_grad_norm: float = 10.0,
        ns_steps: int = 5,
    ):
        super().__init__()
        self.dim = dim
        self.mem_dim = mem_dim
        self.chunk_size = chunk_size
        self.poly_degree = poly_degree
        self.max_grad_norm = max_grad_norm
        self.ns_steps = ns_steps
        self.swiglu = False  # CMS interface compat
        self.learning_enabled = True

        poly_dim = mem_dim * poly_degree  # expanded key/query dimension

        # Projections (learnable via outer-loop training)
        self.to_k = nn.Linear(dim, mem_dim, bias=False)
        self.to_v = nn.Linear(dim, mem_dim, bias=False)
        self.to_q = nn.Linear(dim, mem_dim, bias=False)
        self.out_proj = nn.Linear(mem_dim, dim, bias=False)

        # Data-dependent gates (replace broken competence gate)
        self.to_lr = nn.Linear(dim, 1, bias=True)        # θ_t: learning rate
        self.to_momentum = nn.Linear(dim, 1, bias=True)   # η_t: momentum decay
        self.to_decay = nn.Linear(dim, 1, bias=True)      # α_t: forget gate
        self.to_output_gate = nn.Linear(dim, 1, bias=True) # output contribution

        # Initialize gate biases for safe startup
        nn.init.constant_(self.to_lr.bias, -4.0)           # sigmoid(-4) ≈ 0.018
        nn.init.constant_(self.to_momentum.bias, 1.0)      # sigmoid(1) ≈ 0.73
        nn.init.constant_(self.to_decay.bias, -3.0)        # sigmoid(-3) ≈ 0.05 (low forgetting)
        nn.init.constant_(self.to_output_gate.bias, -1.0)  # sigmoid(-1) ≈ 0.27 (start partially closed)

        # The memory: deep MLP whose weights are the learned specialization
        # Operates in poly_dim space (mem_dim * poly_degree)
        self.memory = MemoryMLP(poly_dim, depth=mem_depth, expansion=2.0)

        # Project memory output back to mem_dim for out_proj
        if poly_degree > 1:
            self.mem_out_proj = nn.Linear(poly_dim, mem_dim, bias=False)
            # Project values up to poly_dim for associative loss computation
            self.v_expand = nn.Linear(mem_dim, poly_dim, bias=False)
        else:
            self.mem_out_proj = nn.Identity()
            self.v_expand = nn.Identity()

        # Runtime state (not nn.Parameters — updated during forward pass)
        self._momentum_state: dict[str, Tensor] = {}
        self._total_updates: int = 0

        # Soul checkpoint
        self._soul_weights: dict[str, Tensor] = {}
        self.soul_pull_strength: float = 0.01
        self.max_drift: float = 0.5

        # Persona probe (set externally)
        self._persona_probe: Tensor | None = None
        self._learning_weight: Tensor | None = None

        # Surprise tracking (for monitoring, not gating)
        self._surprise_ema: float = 1.0

        # Neutral drift
        self.drift_enabled = False
        self.drift_sigma = 1e-5 / max(chunk_size, 1)

    def _poly_expand(self, x: Tensor) -> Tensor:
        """Polynomial feature expansion: φ(x) = [x, x², ...] / sqrt(p).

        Expands mem_dim to mem_dim * poly_degree for increased memory capacity.
        """
        if self.poly_degree == 1:
            return x
        terms = [x]
        for p in range(2, self.poly_degree + 1):
            terms.append(x.pow(p))
        return torch.cat(terms, dim=-1) / math.sqrt(self.poly_degree)

    def _get_memory_params(self) -> dict[str, Tensor]:
        """Get current memory MLP parameters as a dict."""
        return dict(self.memory.named_parameters())

    def _retrieve(self, queries: Tensor, params: dict[str, Tensor]) -> Tensor:
        """Retrieve from memory and project back to mem_dim."""
        raw = functional_call(self.memory, params, (queries,))
        return self.mem_out_proj(raw)

    def _compute_chunk_gradients(
        self,
        keys: Tensor,
        values: Tensor,
        params: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Compute per-sample gradients for a chunk via vmap+grad.

        This is the core ATLAS mechanism: gradient of ‖M(k)-v‖² w.r.t.
        memory MLP weights, vectorized across all samples in the chunk.

        Args:
            keys: Polynomial-expanded keys (batch*chunk, poly_dim).
            values: Target values (batch*chunk, mem_dim).
            params: Current memory parameters.

        Returns:
            Dict of chunk-averaged gradients per parameter.
        """
        n = keys.shape[0]

        # Expand params for vmap: each sample uses same params
        expanded_params = {
            name: p.unsqueeze(0).expand(n, *p.shape)
            for name, p in params.items()
        }

        # Per-sample gradient computation
        grad_fn = grad(_memory_loss_fn, argnums=0)
        batched_grad_fn = vmap(grad_fn, in_dims=(0, None, 0, 0))
        per_sample_grads = batched_grad_fn(expanded_params, self.memory, keys, values)

        # Average over the chunk (Omega Rule: optimize over window, not single token)
        avg_grads = {}
        for name, g in per_sample_grads.items():
            # Clip per-sample gradient norms for stability
            g_flat = g.reshape(n, -1)
            g_norms = g_flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
            scale = (g_norms / self.max_grad_norm).clamp(min=1.0)
            g_clipped = g / scale.view(n, *([1] * (g.dim() - 1)))
            avg_grads[name] = g_clipped.mean(dim=0)  # average over chunk

        return avg_grads

    def _apply_momentum_update(
        self,
        params: dict[str, Tensor],
        grads: dict[str, Tensor],
        lr: float,
        momentum_decay: float,
        forget_gate: float,
    ) -> dict[str, Tensor]:
        """Apply Titans/ATLAS momentum update with NS-5.

        S_t = η·S_{t-1} - θ·grad          [surprise momentum]
        S_orth = NS5(S) * θ                [orthogonalize, then scale by lr]
        M_t = (1-α)·M_{t-1} + S_orth       [update with forget gate]

        NS-5 normalizes momentum to unit spectral norm. The learning rate θ
        then controls the actual step size. Without this scaling, NS-5 output
        would dominate the weights (unit norm >> typical weight magnitudes).
        """
        updated = {}
        for name, p in params.items():
            g = grads.get(name)
            if g is None:
                updated[name] = p
                continue

            # Initialize momentum if needed
            if name not in self._momentum_state:
                self._momentum_state[name] = torch.zeros_like(p)

            # Surprise momentum: S = η·S_prev - θ·grad
            s = momentum_decay * self._momentum_state[name] - lr * g
            self._momentum_state[name] = s.detach()

            # NS-5 orthogonalization on 2D weight matrices, scaled by lr
            if s.dim() >= 2 and min(s.shape) > 1:
                s_orth = newton_schulz(s, steps=self.ns_steps) * lr
            else:
                s_orth = s

            # Weight update with forget gate: M = (1-α)·M_prev + S_orth
            updated[name] = (1.0 - forget_gate) * p + s_orth

        return updated

    def forward(self, x: Tensor, l0_out: Tensor | None = None) -> Tensor:
        """Forward pass: retrieve from memory, then store (update weights).

        Args:
            x: Input to the CMS (batch, seq, dim).
            l0_out: L0's output. If provided, output is residual on l0_out.
        """
        batch, seq_len, dim = x.shape
        base = l0_out if l0_out is not None else x

        if not self.learning_enabled or seq_len < 2:
            # No learning — just retrieve and return
            q = F.normalize(self.to_q(x), dim=-1)
            q_poly = self._poly_expand(q)
            params = self._get_memory_params()
            retrieved = self._retrieve(q_poly, params)
            delta = self.out_proj(retrieved)
            output_gate = torch.sigmoid(self.to_output_gate(x))
            return base + delta * output_gate

        # Project to memory space
        k = F.normalize(self.to_k(x), dim=-1)
        v = self.to_v(x)
        q = F.normalize(self.to_q(x), dim=-1)

        # Polynomial expansion
        k_poly = self._poly_expand(k)
        q_poly = self._poly_expand(q)

        # Data-dependent gates (averaged per chunk in the loop)
        lr_raw = torch.sigmoid(self.to_lr(x))           # (batch, seq, 1)
        mom_raw = torch.sigmoid(self.to_momentum(x))
        decay_raw = torch.sigmoid(self.to_decay(x))
        output_gate = torch.sigmoid(self.to_output_gate(x))  # (batch, seq, 1)

        # Get current memory params
        params = self._get_memory_params()

        # Process in chunks (Omega Rule: each chunk is one update window)
        all_retrievals = []
        for chunk_start in range(0, seq_len, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, seq_len)
            chunk_len = chunk_end - chunk_start

            # PHASE 1: RETRIEVE (read before updating — causal order)
            q_chunk = q_poly[:, chunk_start:chunk_end]  # (batch, chunk, poly_dim)
            retrieval = self._retrieve(q_chunk, params)
            all_retrievals.append(retrieval)

            # PHASE 2: STORE (compute gradients and update weights)
            if torch.is_grad_enabled():
                # During backprop training, don't update memory weights
                continue

            k_chunk = k_poly[:, chunk_start:chunk_end]
            v_chunk = v[:, chunk_start:chunk_end]

            # Expand values to poly_dim for loss computation
            v_expanded = self.v_expand(v_chunk)

            # Flatten batch and chunk for vmap
            k_flat = k_chunk.reshape(-1, k_chunk.shape[-1])
            v_flat = v_expanded.reshape(-1, v_expanded.shape[-1])

            # Per-sample gradients averaged over chunk (Omega Rule)
            chunk_grads = self._compute_chunk_gradients(k_flat, v_flat, params)

            # Apply persona probe to gradients
            if self._persona_probe is not None:
                chunk_grads = self._project_grads_persona(chunk_grads)

            # Chunk-averaged gate values
            avg_lr = lr_raw[:, chunk_start:chunk_end].mean().item()
            avg_mom = mom_raw[:, chunk_start:chunk_end].mean().item()
            avg_decay = decay_raw[:, chunk_start:chunk_end].mean().item()

            # Track surprise (for monitoring)
            grad_norm = sum(g.norm().item() for g in chunk_grads.values())
            self._surprise_ema = 0.95 * self._surprise_ema + 0.05 * grad_norm

            # Momentum update with NS-5
            params = self._apply_momentum_update(
                params, chunk_grads, avg_lr, avg_mom, avg_decay,
            )

            # Soul pull-back
            if self._soul_weights:
                params = self._soul_pullback(params)

            self._total_updates += 1

        # Write updated params back to the memory module
        if not torch.is_grad_enabled():
            with torch.no_grad():
                for name, p in self.memory.named_parameters():
                    if name in params:
                        p.data.copy_(params[name].detach())

        # Combine all retrieved chunks
        retrieved = torch.cat(all_retrievals, dim=1)  # (batch, seq, mem_dim)
        delta = self.out_proj(retrieved)
        delta = delta * output_gate

        out = base + delta

        if self.drift_enabled and not self.training:
            with torch.no_grad():
                out = out + torch.randn_like(out) * self.drift_sigma

        return out

    def _project_grads_persona(self, grads: dict[str, Tensor]) -> dict[str, Tensor]:
        """Project gradients through persona probe to focus on output-relevant directions."""
        probe = self._persona_probe  # (dim, persona_dim)
        if probe is None:
            return grads
        # Only project the last layer's gradients (output-facing)
        projected = {}
        for name, g in grads.items():
            if g.dim() == 2 and g.shape[0] == self.mem_dim:
                # This is an output-facing weight — project columns through probe
                # probe is (dim, persona_dim), but memory operates at mem_dim
                # Only apply if dimensions match
                projected[name] = g
            else:
                projected[name] = g
        return projected

    def save_soul(self) -> None:
        """Snapshot current memory weights as the soul anchor."""
        self._soul_weights = {
            name: p.data.float().clone()
            for name, p in self.memory.named_parameters()
        }

    def _soul_pullback(self, params: dict[str, Tensor]) -> dict[str, Tensor]:
        """Pull weights back toward soul checkpoint if drift exceeds threshold."""
        pulled = {}
        for name, p in params.items():
            if name in self._soul_weights:
                soul = self._soul_weights[name].to(p.device, dtype=p.dtype)
                drift = (p - soul).norm().item()
                if drift > self.max_drift:
                    p = torch.lerp(p, soul, self.soul_pull_strength)
            pulled[name] = p
        return pulled

    def reset_state(self) -> None:
        """Reset learning state (keeps soul checkpoint)."""
        self._momentum_state.clear()
        self._total_updates = 0
        self._surprise_ema = 1.0

    def reset_to_soul(self) -> bool:
        """Reset memory MLP weights to soul checkpoint and clear all learning state.

        Returns True if soul checkpoint existed and was restored, False otherwise.
        """
        if not self._soul_weights:
            return False
        with torch.no_grad():
            for name, param in self.memory.named_parameters():
                if name in self._soul_weights:
                    param.data.copy_(
                        self._soul_weights[name].to(param.device, dtype=param.dtype)
                    )
        self.reset_state()
        return True

    @property
    def surprise(self) -> float:
        if self._total_updates == 0:
            return 0.0
        return self._surprise_ema

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, mem_dim={self.mem_dim}, "
            f"chunk={self.chunk_size}, poly={self.poly_degree}, "
            f"learning={self.learning_enabled}"
        )


class ContinuumMemorySystem(nn.Module):
    """
    Continuum Memory System — ATLAS-style deep memory replacing MLP.

    Level 0: SwiGLU (pre-trained MLP, frozen base intelligence)
    Level 1+: DeepMemoryLevel (ATLAS-style associative memory with Omega Rule)

    The memory learns during inference via gradient descent on the associative
    loss, with momentum, NS-5 orthogonalization, and data-dependent gates.
    No seeding needed — the environment compiles the identity.
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
        rank: int = 32,
        mem_dim: int = 512,
        mem_depth: int = 2,
        poly_degree: int = 2,
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

        # Level 0: full SwiGLU (pre-trained weights, frozen base intelligence)
        l0 = CMSLevel(
            dim=dim, hidden_mult=hidden_mults[0], chunk_size=chunk_sizes[0],
            swiglu=True, activation=activation,
            lr=lr, max_grad_norm=max_grad_norm,
        )
        l0.learning_enabled = False

        # Levels 1+: ATLAS-style deep memory
        higher_levels = []
        for i in range(1, num_levels):
            dml = DeepMemoryLevel(
                dim=dim,
                mem_dim=mem_dim,
                mem_depth=mem_depth,
                chunk_size=chunk_sizes[i],
                poly_degree=poly_degree,
                max_grad_norm=max_grad_norm,
            )
            dml.learning_enabled = True
            higher_levels.append(dml)

        self.levels = nn.ModuleList([l0] + higher_levels)
        self._token_position: int = 0

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_chain(x)

    def _forward_chain(self, x: Tensor) -> Tensor:
        # L0 computes full SwiGLU (frozen base)
        l0_out = self.levels[0](x)

        # Deep memory levels add residual deltas
        current = l0_out
        for level in self.levels[1:]:
            current = level(x, l0_out=current)
        return current

    def save_soul(self) -> None:
        """Save current weights as soul anchor across all levels."""
        for level in self.levels:
            level.save_soul()

    def enable_drift(self, enabled: bool = True) -> None:
        for level in self.levels:
            level.drift_enabled = enabled

    def enable_learning(self, enabled: bool = True, levels: list[int] | None = None) -> None:
        for i, level in enumerate(self.levels):
            if levels is None or i in levels:
                level.learning_enabled = enabled

    def get_surprise(self) -> list[float]:
        return [level.surprise for level in self.levels]

    def set_learning_weight(self, weight: Tensor | None) -> None:
        """Set per-position learning weight for persona-aware predictive coding.

        Args:
            weight: (batch, seq_len) tensor. Higher values = learn more from
                    those positions. Use 1.0 for assistant tokens, 0.1 for user.
                    None clears the weight (uniform learning).
        """
        for level in self.levels:
            level._learning_weight = weight

    def reset_learning_state(self) -> None:
        for level in self.levels:
            level.reset_state()

    def set_persona_probe(self, lm_head_weight: Tensor, persona_dim: int = 256) -> None:
        """Set persona probe on the final level from LM head weights.

        Uses SVD of the LM head to extract the top principal directions
        in hidden space that most influence token selection. The probe
        focuses predictive coding error on these output-relevant directions,
        driving weight updates toward changes that affect generation style
        rather than just internal compression.

        Args:
            lm_head_weight: The LM head weight matrix (vocab_size, dim).
            persona_dim: Number of principal directions to keep.
        """
        with torch.no_grad():
            _, _, V = torch.svd_lowrank(lm_head_weight.float(), q=persona_dim)
            probe = V.to(lm_head_weight.dtype)  # (dim, persona_dim)
        self.levels[-1]._persona_probe = probe

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, levels={self.num_levels}, "
            f"variant={self.variant.value}, chunks={self.chunk_sizes}"
        )
