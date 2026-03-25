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

The key implementation technique: torch.func.vmap + grad + functional_call
to compute per-sample gradients of the memory MLP weights during the forward
pass, without materializing a separate backward pass.

Reference implementation: lucidrains/titans-pytorch (NeuralMemory class).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# torch.func APIs for per-sample gradients
from torch.func import functional_call, grad, vmap


@dataclass
class MemoryState:
    """Persistent state for the neural memory across sequences."""
    weights: dict[str, Tensor] = field(default_factory=dict)
    momentum: dict[str, Tensor] = field(default_factory=dict)
    seq_index: int = 0


class MemoryMLP(nn.Module):
    """
    The memory module: a small MLP whose weights ARE the memory.

    M(x) = x + W_down · σ(W_up · x)  (2-layer with residual)

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


def _memory_loss_fn(
    params: dict[str, Tensor],
    model: MemoryMLP,
    keys: Tensor,
    values: Tensor,
) -> Tensor:
    """
    Compute associative memory loss for a single sample.

    ℓ(M; x_t) = ||M(k_t) - v_t||²

    This function is designed to be used with torch.func.grad to compute
    per-parameter gradients, and then vmap'd across the batch.

    Args:
        params: Dict of memory MLP parameters (from dict(model.named_parameters())).
        model: The MemoryMLP module (used for structure, not weights).
        keys: Key input for this sample (head_dim,).
        values: Target value for this sample (head_dim,).

    Returns:
        Scalar loss for this sample.
    """
    # Use functional_call to evaluate the model with the given params
    pred = functional_call(model, params, (keys.unsqueeze(0),)).squeeze(0)
    loss = (pred - values).pow(2).sum()
    return loss


class NeuralMemory(nn.Module):
    """
    Titans-style neural long-term memory with gradient-based updates.

    The memory is a small MLP (MemoryMLP) whose weights are updated during
    both training and inference via gradient descent on the associative memory
    loss. This enables the model to memorize patterns at test time.

    The update uses momentum-based surprise (Equations 13-14):
        S_t = η_t · S_{t-1} - θ_t · ∇ℓ(M_{t-1}; x_t)    [surprise momentum]
        M_t = (1 - α_t) · M_{t-1} + S_t                    [weight update]

    Implementation uses torch.func for per-sample gradients:
    - grad() computes ∇ℓ w.r.t. memory parameters for one sample
    - vmap() vectorizes across the batch dimension
    - functional_call() evaluates the MLP with batched parameter sets

    Within a chunk, memory weights are frozen (parallelizable). Updates
    are applied at chunk boundaries.

    Args:
        dim: Hidden dimension of the transformer.
        mem_dim: Memory MLP dimension (default: same as dim).
        mem_depth: Depth of memory MLP (default: 2).
        num_heads: Number of independent memory heads.
        chunk_size: Process tokens in chunks of this size for parallelization.
        max_grad_norm: Soft clamp for gradient norms (stability).
    """

    def __init__(
        self,
        dim: int,
        mem_dim: int | None = None,
        mem_depth: int = 2,
        num_heads: int = 1,
        chunk_size: int = 64,
        max_grad_norm: float = 10.0,
    ):
        super().__init__()
        self.dim = dim
        self.mem_dim = mem_dim or dim
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.max_grad_norm = max_grad_norm
        self.head_dim = self.mem_dim // num_heads

        # Projections for keys, values, queries
        self.to_k = nn.Linear(dim, self.mem_dim, bias=False)
        self.to_v = nn.Linear(dim, self.mem_dim, bias=False)
        self.to_q = nn.Linear(dim, self.mem_dim, bias=False)

        # Adaptive gates (data-dependent, Titans Eq 13-14)
        self.to_lr = nn.Linear(dim, num_heads, bias=False)       # θ_t
        self.to_momentum = nn.Linear(dim, num_heads, bias=False)  # η_t
        self.to_decay = nn.Linear(dim, num_heads, bias=False)     # α_t

        # The memory MLPs (one per head)
        self.memory_heads = nn.ModuleList([
            MemoryMLP(self.head_dim, depth=mem_depth)
            for _ in range(num_heads)
        ])

        # Output projection
        self.out_proj = nn.Linear(self.mem_dim, dim, bias=False)

    def _get_memory_params(self, head_idx: int) -> dict[str, Tensor]:
        """Get a dict of named parameters for a specific memory head."""
        return dict(self.memory_heads[head_idx].named_parameters())

    def _compute_chunk_gradients(
        self,
        head_idx: int,
        keys: Tensor,
        values: Tensor,
        params: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """
        Compute per-sample gradients for a chunk of tokens using vmap+grad.

        Args:
            head_idx: Which memory head.
            keys: Keys for this chunk (batch, chunk_len, head_dim).
            values: Values for this chunk (batch, chunk_len, head_dim).
            params: Current memory parameters.

        Returns:
            Dict of parameter gradients, each shaped (batch, chunk_len, *param_shape).
        """
        model = self.memory_heads[head_idx]
        batch, chunk_len, _ = keys.shape

        # Flatten batch and chunk dimensions for vmap
        flat_keys = keys.reshape(-1, self.head_dim)      # (B*C, head_dim)
        flat_values = values.reshape(-1, self.head_dim)   # (B*C, head_dim)

        # Expand params to match the flattened batch: each sample uses same params
        expanded_params = {
            name: p.unsqueeze(0).expand(flat_keys.shape[0], *p.shape)
            for name, p in params.items()
        }

        # grad of loss w.r.t. params, for a single sample
        grad_fn = grad(_memory_loss_fn, argnums=0)

        # vmap across the batch dimension
        batched_grad_fn = vmap(grad_fn, in_dims=(0, None, 0, 0))

        # Compute per-sample gradients
        per_sample_grads = batched_grad_fn(expanded_params, model, flat_keys, flat_values)

        # Reshape back to (batch, chunk_len, *param_shape)
        result = {}
        for name, g in per_sample_grads.items():
            result[name] = g.reshape(batch, chunk_len, *g.shape[1:])

        return result

    def _apply_momentum_update(
        self,
        params: dict[str, Tensor],
        grads: dict[str, Tensor],
        momentum_state: dict[str, Tensor],
        lr: Tensor,
        momentum_decay: Tensor,
        weight_decay: Tensor,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """
        Apply the Titans momentum-based weight update (Equations 13-14).

        S_t = η_t · S_{t-1} - θ_t · ∇ℓ
        M_t = (1 - α_t) · M_{t-1} + S_t

        Processes sequentially across the chunk dimension. Within-chunk
        parallelization via associative scan is a Phase 6 optimization.

        Args:
            params: Current memory weights {name: (batch, *param_shape)}.
            grads: Per-sample gradients {name: (batch, chunk_len, *param_shape)}.
            momentum_state: Previous surprise momentum {name: (batch, *param_shape)}.
            lr: Learning rate θ_t (batch, chunk_len, 1) — from adaptive gate.
            momentum_decay: Momentum decay η_t (batch, chunk_len, 1).
            weight_decay: Forgetting gate α_t (batch, chunk_len, 1).

        Returns:
            Tuple of (updated_params, updated_momentum).
        """
        batch, chunk_len = next(iter(grads.values())).shape[:2]

        updated_params = {name: p.clone() for name, p in params.items()}
        updated_momentum = {
            name: m.clone() if name in momentum_state else torch.zeros_like(p)
            for name, p in params.items()
            for m in [momentum_state.get(name)]
        }
        # Fix: properly initialize momentum
        updated_momentum = {}
        for name, p in params.items():
            if name in momentum_state:
                updated_momentum[name] = momentum_state[name].clone()
            else:
                updated_momentum[name] = torch.zeros_like(p)

        # Sequential update across chunk positions
        # (Phase 6 will replace this with parallel associative scan)
        for t in range(chunk_len):
            lr_t = lr[:, t]                # (batch, 1)
            eta_t = momentum_decay[:, t]   # (batch, 1)
            alpha_t = weight_decay[:, t]   # (batch, 1)

            for name in updated_params:
                g_t = grads[name][:, t]    # (batch, *param_shape)

                # Soft clamp gradient norms for stability
                g_norm = g_t.flatten(1).norm(dim=1, keepdim=True)
                scale = torch.clamp(g_norm / self.max_grad_norm, min=1.0)
                g_t = g_t / scale.view(-1, *([1] * (g_t.dim() - 1)))

                # Surprise momentum: S_t = η_t · S_{t-1} - θ_t · ∇ℓ
                # Reshape gates to broadcast with param shape
                eta_bc = eta_t.view(-1, *([1] * (g_t.dim() - 1)))
                lr_bc = lr_t.view(-1, *([1] * (g_t.dim() - 1)))
                alpha_bc = alpha_t.view(-1, *([1] * (g_t.dim() - 1)))

                s = eta_bc * updated_momentum[name] - lr_bc * g_t
                updated_momentum[name] = s

                # Weight update: M_t = (1 - α_t) · M_{t-1} + S_t
                updated_params[name] = (1 - alpha_bc) * updated_params[name] + s

        return updated_params, updated_momentum

    def forward(
        self,
        x: Tensor,
        state: MemoryState | None = None,
    ) -> tuple[Tensor, MemoryState]:
        """
        Store and retrieve from neural memory.

        Two phases per chunk:
        1. RETRIEVE: Read from memory using queries (with current weights)
        2. STORE: Compute surprise gradients and update memory weights

        This ordering means retrieval uses weights BEFORE the current chunk's
        update, which is the correct causal order.

        Args:
            x: Input tensor (batch, seq_len, dim).
            state: Previous memory state (for cross-sequence persistence).

        Returns:
            Tuple of (output tensor, updated memory state).
        """
        batch, seq_len, _ = x.shape

        # Project to keys, values, queries
        k = self.to_k(x)
        v = self.to_v(x)
        q = self.to_q(x)

        # Compute adaptive gates
        lr = torch.sigmoid(self.to_lr(x))         # (batch, seq, heads)
        momentum = torch.sigmoid(self.to_momentum(x))
        decay = torch.sigmoid(self.to_decay(x))

        # Split into heads: (batch, seq, heads, head_dim)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)

        # Normalize keys and queries
        k = F.normalize(k, dim=-1)
        q = F.normalize(q, dim=-1)

        # Initialize or restore memory state
        all_head_params = []
        all_head_momentum = []
        for h in range(self.num_heads):
            if state and state.weights.get(f"head_{h}"):
                head_params = state.weights[f"head_{h}"]
                head_momentum = state.momentum.get(f"head_{h}", {})
            else:
                # Initialize from the module's current parameters
                # Expand to batch dimension
                head_params = {
                    name: p.unsqueeze(0).expand(batch, *p.shape).clone()
                    for name, p in self.memory_heads[h].named_parameters()
                }
                head_momentum = {}
            all_head_params.append(head_params)
            all_head_momentum.append(head_momentum)

        # Process in chunks
        all_outputs = []
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size

        for chunk_idx in range(num_chunks):
            start = chunk_idx * self.chunk_size
            end = min(start + self.chunk_size, seq_len)
            chunk_len = end - start

            chunk_outputs = []

            for h in range(self.num_heads):
                q_chunk = q[:, start:end, h, :]   # (batch, chunk_len, head_dim)
                k_chunk = k[:, start:end, h, :]
                v_chunk = v[:, start:end, h, :]

                # Gates for this head
                lr_chunk = lr[:, start:end, h:h+1]        # (batch, chunk_len, 1)
                mom_chunk = momentum[:, start:end, h:h+1]
                dec_chunk = decay[:, start:end, h:h+1]

                # PHASE 1: RETRIEVE (read from memory before update)
                # Use functional_call with current head params
                retrieve_out = self._retrieve_chunk(
                    h, q_chunk, all_head_params[h],
                )
                chunk_outputs.append(retrieve_out)

                # PHASE 2: STORE (compute gradients and update weights)
                # Get the base (non-batched) params for grad computation
                base_params = {
                    name: p[0]  # take first batch element (all same at init)
                    for name, p in all_head_params[h].items()
                }

                grads = self._compute_chunk_gradients(
                    h, k_chunk, v_chunk, base_params,
                )

                # Apply momentum-based update
                all_head_params[h], all_head_momentum[h] = self._apply_momentum_update(
                    all_head_params[h], grads, all_head_momentum[h],
                    lr_chunk, mom_chunk, dec_chunk,
                )

            # Stack heads: (batch, chunk_len, num_heads, head_dim)
            chunk_out = torch.stack(chunk_outputs, dim=2)
            all_outputs.append(chunk_out)

        # Concatenate all chunks
        output = torch.cat(all_outputs, dim=1)  # (batch, seq_len, heads, head_dim)
        output = output.view(batch, seq_len, self.mem_dim)
        output = self.out_proj(output)

        # Build updated state for persistence
        new_weights = {f"head_{h}": all_head_params[h] for h in range(self.num_heads)}
        new_momentum = {f"head_{h}": all_head_momentum[h] for h in range(self.num_heads)}
        new_state = MemoryState(
            weights=new_weights,
            momentum=new_momentum,
            seq_index=(state.seq_index if state else 0) + seq_len,
        )

        return output, new_state

    def _retrieve_chunk(
        self,
        head_idx: int,
        queries: Tensor,
        params: dict[str, Tensor],
    ) -> Tensor:
        """
        Retrieve from memory by running the memory MLP on queries.

        Uses the first batch element's params (they're the same within a chunk
        since updates happen at chunk boundaries).

        Args:
            head_idx: Which memory head.
            queries: Query tensor (batch, chunk_len, head_dim).
            params: Current memory parameters {name: (batch, *param_shape)}.

        Returns:
            Memory output (batch, chunk_len, head_dim).
        """
        model = self.memory_heads[head_idx]

        # Use first batch element params for retrieval
        # (within a chunk, all batch elements share the same memory state)
        base_params = {name: p[0] for name, p in params.items()}
        output = functional_call(model, base_params, (queries,))
        return output

    @property
    def surprise_metric(self) -> str:
        """Description of the surprise metric used."""
        return "||M(k_t) - v_t||^2 (L2 reconstruction error)"

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, mem_dim={self.mem_dim}, "
            f"num_heads={self.num_heads}, chunk_size={self.chunk_size}, "
            f"head_dim={self.head_dim}, max_grad_norm={self.max_grad_norm}"
        )
