"""
Neural Memory Module — Titans-style gradient-based memory.

Implements the core Titans memory mechanism (Behrouz et al., 2024) where
memory is a small MLP whose weights are updated via gradient descent during
the forward pass. The memory learns to memorize by surprise: high prediction
error triggers strong updates, low prediction error means the memory already
knows this pattern.

Paper reference: Titans arxiv 2501.00663, Equations 12-14.

Memory update rule:
    M_t = (1 - α_t) · M_{t-1} + S_t
    S_t = η_t · S_{t-1} - θ_t · ∇ℓ(M_{t-1}; x_t)

Where:
    - M_t: memory state (MLP weights) at time t
    - S_t: surprise momentum (accumulated gradient signal)
    - α_t: adaptive forgetting gate
    - η_t: momentum decay coefficient
    - θ_t: learning rate for surprise incorporation
    - ℓ: associative memory loss = ||M(k_t) - v_t||²

Reference implementation: lucidrains/titans-pytorch (NeuralMemory class).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class MemoryState:
    """Persistent state for the neural memory across sequences."""
    weights: dict[str, Tensor]   # Current memory MLP weights
    momentum: dict[str, Tensor]  # Surprise momentum per parameter
    seq_index: int = 0           # Global sequence position


class MemoryMLP(nn.Module):
    """
    The memory module: a small MLP whose weights ARE the memory.

    M(x) = x + W1 · σ(W2 · x)  (2-layer with residual)

    These weights are updated during the forward pass via gradient descent
    on the associative memory loss, NOT by standard backpropagation.

    Args:
        dim: Input/output dimension.
        depth: Number of hidden layers (paper tests 1-4, recommends 2).
        expansion: Hidden dimension multiplier.
    """

    def __init__(self, dim: int, depth: int = 2, expansion: float = 2.0):
        super().__init__()
        self.dim = dim
        self.depth = depth
        hidden = int(dim * expansion)

        layers = []
        for i in range(depth):
            in_d = dim if i == 0 else hidden
            out_d = dim if i == depth - 1 else hidden
            layers.append(nn.Linear(in_d, out_d, bias=False))
            if i < depth - 1:
                layers.append(nn.SiLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class NeuralMemory(nn.Module):
    """
    Titans-style neural long-term memory.

    The memory is a small MLP (MemoryMLP) whose weights are updated during
    both training and inference via gradient descent on the associative memory
    loss. This enables the model to memorize patterns at test time.

    The update uses momentum-based surprise:
    - Momentary surprise = gradient of reconstruction loss
    - Past surprise = exponentially decayed history (momentum)
    - Strong surprise = strong weight update
    - Weak surprise = memory already knows this, minimal update

    Args:
        dim: Hidden dimension.
        mem_dim: Memory MLP dimension (default: same as dim).
        mem_depth: Depth of memory MLP (default: 2).
        num_heads: Number of independent memory heads.
        chunk_size: Process tokens in chunks of this size for parallelization.
    """

    def __init__(
        self,
        dim: int,
        mem_dim: int | None = None,
        mem_depth: int = 2,
        num_heads: int = 1,
        chunk_size: int = 64,
    ):
        super().__init__()
        self.dim = dim
        self.mem_dim = mem_dim or dim
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        head_dim = self.mem_dim // num_heads

        # Projections for keys, values, queries
        self.to_k = nn.Linear(dim, self.mem_dim, bias=False)
        self.to_v = nn.Linear(dim, self.mem_dim, bias=False)
        self.to_q = nn.Linear(dim, self.mem_dim, bias=False)

        # Adaptive gates (data-dependent, Titans Eq 13-14)
        self.to_lr = nn.Linear(dim, num_heads, bias=False)       # θ_t: learning rate
        self.to_momentum = nn.Linear(dim, num_heads, bias=False)  # η_t: momentum decay
        self.to_decay = nn.Linear(dim, num_heads, bias=False)     # α_t: forgetting gate

        # The memory MLP itself (one per head)
        self.memory = nn.ModuleList([
            MemoryMLP(head_dim, depth=mem_depth) for _ in range(num_heads)
        ])

        # Output projection
        self.out_proj = nn.Linear(self.mem_dim, dim, bias=False)

        # L2 normalization for keys and queries (paper recommends this)
        self.norm_k = nn.LayerNorm(head_dim, elementwise_affine=False)
        self.norm_q = nn.LayerNorm(head_dim, elementwise_affine=False)

    def forward(
        self,
        x: Tensor,
        state: MemoryState | None = None,
    ) -> tuple[Tensor, MemoryState]:
        """
        Store and retrieve from neural memory.

        The forward pass has two phases:
        1. STORE: Compute keys/values, measure surprise, update memory weights
        2. RETRIEVE: Use queries to read from updated memory

        Args:
            x: Input tensor (batch, seq_len, dim).
            state: Previous memory state (for cross-sequence persistence).

        Returns:
            Tuple of (output tensor, updated memory state).
        """
        batch, seq_len, _ = x.shape

        # Project to keys, values, queries
        k = self.to_k(x)  # (batch, seq, mem_dim)
        v = self.to_v(x)
        q = self.to_q(x)

        # Compute adaptive gates
        lr = torch.sigmoid(self.to_lr(x))        # (batch, seq, heads)
        momentum = torch.sigmoid(self.to_momentum(x))
        decay = torch.sigmoid(self.to_decay(x))

        # Split into heads
        head_dim = self.mem_dim // self.num_heads
        k = k.view(batch, seq_len, self.num_heads, head_dim)
        v = v.view(batch, seq_len, self.num_heads, head_dim)
        q = q.view(batch, seq_len, self.num_heads, head_dim)

        # Normalize keys and queries
        k = F.normalize(k, dim=-1)
        q = F.normalize(q, dim=-1)

        # Retrieve from memory using queries
        outputs = []
        for h in range(self.num_heads):
            q_h = q[:, :, h, :]  # (batch, seq, head_dim)
            mem_out = self.memory[h](q_h)  # (batch, seq, head_dim)
            outputs.append(mem_out)

        # Combine heads
        output = torch.stack(outputs, dim=2)  # (batch, seq, heads, head_dim)
        output = output.view(batch, seq_len, self.mem_dim)
        output = self.out_proj(output)

        # Note: The actual weight update via gradient descent during forward pass
        # requires torch.func.vmap + grad (as in lucidrains implementation).
        # This is a structural scaffold — the full per-sample gradient computation
        # will be added when we integrate with the training loop.

        # Return current state (to be expanded with actual weight tracking)
        new_state = MemoryState(
            weights={},
            momentum={},
            seq_index=(state.seq_index if state else 0) + seq_len,
        )

        return output, new_state

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, mem_dim={self.mem_dim}, "
            f"num_heads={self.num_heads}, chunk_size={self.chunk_size}"
        )
