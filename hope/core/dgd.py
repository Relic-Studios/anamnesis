"""
Delta Gradient Descent (DGD) — Data-dependent weight update rule.

Implements Section 4.5 and Appendix C of the Nested Learning paper.
DGD extends standard gradient descent with a data-dependent weight decay term
that captures inter-sample dependencies.

Standard GD: W_{t+1} = W_t - η · ∇L · x_t^T
DGD:         W_{t+1} = W_t · (I - α_t · x_t · x_t^T) - β · ∇L · x_t^T

The key difference: the (I - α_t · x_t · x_t^T) term makes forgetting DIRECTIONAL.
The model forgets in the direction of the current input, not uniformly.

Paper reference: Equations 54-60, 112-121.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class DeltaGradientDescent(nn.Module):
    """
    Delta Gradient Descent update rule for CMS memory parameters.

    Computes the DGD weight update:
        W_{t+1} = W_t · (I - α_t · x_t · x_t^T) - η_t · ∇L(W_t; x_t)

    Where:
        - α_t: data-dependent decay rate (scalar or from precision network)
        - η_t: learning rate (scalar or from precision network)
        - x_t: normalized input
        - ∇L: gradient of the associative memory loss

    The data-dependent decay (I - α_t · x_t · x_t^T) is a rank-1 update to the
    identity that causes forgetting specifically in the direction of the current input.
    This captures inter-sample dependencies that standard GD misses.

    Args:
        dim: Input dimension for the memory being updated.
        default_lr: Default learning rate η if not provided per-step.
        default_alpha: Default decay rate α if not provided per-step.
        normalize_inputs: Whether to L2-normalize inputs before computing decay.
    """

    def __init__(
        self,
        dim: int,
        default_lr: float = 1e-3,
        default_alpha: float = 1e-2,
        normalize_inputs: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.default_lr = default_lr
        self.default_alpha = default_alpha
        self.normalize_inputs = normalize_inputs

    def compute_update(
        self,
        weight: Tensor,
        input_x: Tensor,
        grad: Tensor,
        lr: float | Tensor | None = None,
        alpha: float | Tensor | None = None,
    ) -> Tensor:
        """
        Compute the DGD weight update.

        Args:
            weight: Current weight matrix (out_dim, in_dim).
            input_x: Current input (batch, in_dim) or (in_dim,).
            grad: Gradient of loss w.r.t. weight (same shape as weight).
            lr: Learning rate η_t. If None, uses default.
            alpha: Decay rate α_t. If None, uses default.

        Returns:
            New weight value after DGD update.
        """
        if lr is None:
            lr = self.default_lr
        if alpha is None:
            alpha = self.default_alpha

        # Normalize input if requested (Appendix C assumes ||x_t||^2 = λ)
        if self.normalize_inputs:
            x = input_x / (input_x.norm(dim=-1, keepdim=True) + 1e-8)
        else:
            x = input_x

        # Data-dependent decay: W · (I - α · x · x^T)
        # For batched inputs, average the outer products
        if x.dim() == 1:
            outer = torch.outer(x, x)  # (in_dim, in_dim)
        else:
            # (batch, in_dim) → average outer product
            outer = torch.einsum("bi,bj->ij", x, x) / x.shape[0]

        # Apply decay and gradient step
        # W_{t+1} = W_t · (I - α · x · x^T) - η · grad
        decay_matrix = torch.eye(self.dim, device=weight.device, dtype=weight.dtype) - alpha * outer
        new_weight = weight @ decay_matrix - lr * grad

        return new_weight

    @staticmethod
    def compute_associative_loss(
        memory_output: Tensor,
        target: Tensor,
    ) -> Tensor:
        """
        Compute the associative memory loss: L = ||M(k_t) - v_t||^2

        This is the inner-loop objective for CMS/Titans memory updates.

        Args:
            memory_output: Output of the memory module M(k_t).
            target: Target values v_t.

        Returns:
            Scalar loss value.
        """
        return torch.mean((memory_output - target) ** 2)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, default_lr={self.default_lr}, "
            f"default_alpha={self.default_alpha}, normalize={self.normalize_inputs}"
        )
