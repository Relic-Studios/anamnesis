"""
M3: Multi-scale Momentum Muon optimizer.

Implements Algorithm 1 from Section 7.2 of the Nested Learning paper.
Combines Adam's variance tracking with Muon's Newton-Schulz orthogonalization
and a Continuum Memory System for the optimizer's own momentum.

Two momentum scales:
    - Fast momentum M(1): updates every step, captures recent gradient direction
    - Slow momentum M(2): updates every f steps, captures long-term gradient landscape

Both are orthogonalized via Newton-Schulz before aggregation.
The slow momentum gives the optimizer "long context" — awareness of gradient
patterns beyond the last ~43 steps that standard momentum can see.

Paper reference: Algorithm 1, Equations 75, Section 7.2.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer

from hope.optim.newton_schulz import newton_schulz


class M3(Optimizer):
    """
    Multi-scale Momentum Muon (M3) optimizer.

    For ≥2D parameters (weight matrices): uses dual-momentum with Newton-Schulz
    orthogonalization. For 1D parameters (biases, norms, embeddings): falls back
    to AdamW.

    Algorithm:
        Outer loop (every f steps):
            M(2) += β₃ · Σ gradients over chunk
            O(2) = NewtonSchulz(M(2))
        Inner loop (every step):
            M(1) += β₁ · g_t
            V += β₂ · g_t²
            O(1) = NewtonSchulz(M(1))
            θ -= η · (O(1) + α · O(2)) / (√V + ε)

    Args:
        params: Model parameters.
        lr: Learning rate (default 0.02, in spectral norm units).
        beta1: Fast momentum factor (default 0.95).
        beta2: Second moment factor for variance tracking (default 0.999).
        beta3: Slow momentum factor (default 0.95).
        alpha: Slow momentum weight in the aggregation (default 0.1).
        weight_decay: L2 regularization (default 0.01).
        ns_steps: Newton-Schulz iterations (default 5).
        slow_freq: Update frequency for slow momentum M(2) (default 100).
        eps: Numerical stability (default 1e-8).

        # AdamW fallback for 1D params
        adam_lr: Learning rate for 1D params (default 3e-4).
        adam_betas: Beta tuple for Adam on 1D params.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        beta2: float = 0.999,
        beta3: float = 0.95,
        alpha: float = 0.1,
        weight_decay: float = 0.01,
        ns_steps: int = 5,
        slow_freq: int = 100,
        eps: float = 1e-8,
        adam_lr: float = 3e-4,
        adam_betas: tuple[float, float] = (0.9, 0.95),
    ):
        defaults = dict(
            lr=lr, beta1=beta1, beta2=beta2, beta3=beta3,
            alpha=alpha, weight_decay=weight_decay,
            ns_steps=ns_steps, slow_freq=slow_freq, eps=eps,
            adam_lr=adam_lr, adam_betas=adam_betas,
        )
        super().__init__(params, defaults)
        self._step_count = 0

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.

        Args:
            closure: Optional closure that re-evaluates the model.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        for group in self.param_groups:
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            beta3 = group["beta3"]
            alpha = group["alpha"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]
            slow_freq = group["slow_freq"]
            eps = group["eps"]

            is_slow_update = (self._step_count % slow_freq == 0)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialize state on first step
                if len(state) == 0:
                    state["step"] = 0
                    if p.dim() >= 2:
                        # M3 path: dual momentum + variance
                        state["fast_momentum"] = torch.zeros_like(p)
                        state["slow_momentum"] = torch.zeros_like(p)
                        state["variance"] = torch.zeros_like(p)
                        state["grad_chunk"] = torch.zeros_like(p)
                    else:
                        # AdamW fallback for 1D
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1

                if p.dim() >= 2:
                    self._m3_step(p, grad, state, group, is_slow_update)
                else:
                    self._adam_step(p, grad, state, group)

        return loss

    def _m3_step(
        self,
        param: Tensor,
        grad: Tensor,
        state: dict[str, Any],
        group: dict[str, Any],
        is_slow_update: bool,
    ) -> None:
        """M3 update for ≥2D parameters (weight matrices)."""
        beta1 = group["beta1"]
        beta2 = group["beta2"]
        beta3 = group["beta3"]
        alpha = group["alpha"]
        lr = group["lr"]
        wd = group["weight_decay"]
        ns_steps = group["ns_steps"]
        eps = group["eps"]

        m_fast = state["fast_momentum"]
        m_slow = state["slow_momentum"]
        v = state["variance"]
        grad_chunk = state["grad_chunk"]

        # Weight decay
        if wd > 0:
            param.mul_(1 - lr * wd)

        # Accumulate gradient for slow momentum
        grad_chunk.add_(grad)

        # Fast momentum update: M(1) = M(1) + β₁ · g_t
        m_fast.add_(grad, alpha=beta1)

        # Variance tracking: V = V + β₂ · g_t²
        v.addcmul_(grad, grad, value=beta2)

        # Orthogonalize fast momentum
        o_fast = newton_schulz(m_fast, steps=ns_steps)

        # Slow momentum update (every f steps)
        if is_slow_update:
            m_slow.add_(grad_chunk, alpha=beta3)
            grad_chunk.zero_()
            # Cache orthogonalized slow momentum
            state["o_slow_cached"] = newton_schulz(m_slow, steps=ns_steps)

        # Get cached slow momentum (or zero if not yet computed)
        o_slow = state.get("o_slow_cached", torch.zeros_like(param))

        # Combined update: θ -= η · (O(1) + α · O(2)) / (√V + ε)
        denom = v.sqrt().add_(eps)
        update = (o_fast + alpha * o_slow) / denom
        param.add_(update, alpha=-lr)

    def _adam_step(
        self,
        param: Tensor,
        grad: Tensor,
        state: dict[str, Any],
        group: dict[str, Any],
    ) -> None:
        """AdamW fallback for 1D parameters (biases, norms, embeddings)."""
        adam_lr = group["adam_lr"]
        beta1, beta2 = group["adam_betas"]
        wd = group["weight_decay"]
        eps = group["eps"]
        step = state["step"]

        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]

        # Weight decay
        if wd > 0:
            param.mul_(1 - adam_lr * wd)

        # Standard Adam updates
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # Bias correction
        bc1 = 1 - beta1 ** step
        bc2 = 1 - beta2 ** step
        corrected_avg = exp_avg / bc1
        corrected_sq = exp_avg_sq / bc2

        # Update
        param.addcdiv_(corrected_avg, corrected_sq.sqrt().add_(eps), value=-adam_lr)
