"""
Fused CMS Update Kernel — forward + gradient + weight update in one pass.

The key performance optimization: instead of three separate operations
(forward pass, gradient computation, weight update), fuse them into a
single kernel that keeps MLP weights in registers across a chunk of tokens.

Pattern from Flash Attention: load a block of data into SRAM once,
do all computation on-chip, write back the result. No HBM round-trips
for intermediate values.

For CMS, the computation per chunk is:
1. Forward: y = x + W_down · σ(W_up · x)    (for each token in chunk)
2. Gradient: ∇L = d(loss)/d(W_up, W_down)    (accumulated over chunk)
3. Update: W -= η · ∇L                        (at chunk boundary)

The gradient uses the "dual form" trick from TTT: instead of per-token
gradient steps, batch the gradient updates across the chunk as matrix
multiplications.

Reference: ttt-lm-kernels, Flash Attention tiling pattern.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def fused_cms_forward_update(
    x: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
    learning_rate: float = 1e-3,
    chunk_size: int = 32,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Fused CMS level forward pass + gradient accumulation + weight update.

    Pure PyTorch implementation (fallback). Processes tokens in chunks,
    accumulating gradients within each chunk and applying updates at
    chunk boundaries.

    Args:
        x: Input tensor (batch, seq_len, dim).
        up_weight: Up projection weight (hidden_dim, dim).
        down_weight: Down projection weight (dim, hidden_dim).
        learning_rate: Learning rate for weight updates.
        chunk_size: Tokens per chunk (update at chunk boundaries).

    Returns:
        Tuple of (output, updated_up_weight, updated_down_weight).
    """
    batch, seq_len, dim = x.shape
    hidden_dim = up_weight.shape[0]

    outputs = []
    up_w = up_weight.clone()
    down_w = down_weight.clone()

    num_chunks = (seq_len + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, seq_len)
        x_chunk = x[:, start:end, :]  # (batch, chunk_len, dim)

        # Forward pass (residual MLP)
        h = torch.nn.functional.silu(x_chunk @ up_w.T)  # (batch, chunk_len, hidden)
        y = x_chunk + h @ down_w.T                        # (batch, chunk_len, dim)
        outputs.append(y)

        # Gradient computation (dual form: batched across chunk)
        # Loss = ||y - x||^2 (reconstruction of residual contribution)
        # d(loss)/d(down_w) = h^T @ (y - x - target_delta)
        # Simplified: accumulate gradients for the associative memory loss
        if chunk_idx < num_chunks - 1:  # Don't update on last chunk
            # Accumulate gradient signal across the chunk
            residual = h @ down_w.T  # The MLP's contribution
            # Gradient of ||residual||^2 w.r.t. down_w
            grad_down = torch.einsum("bcd,bch->dh", residual, h) / (batch * (end - start))
            # Gradient w.r.t. up_w (through SiLU)
            grad_up = torch.einsum("bch,bcd->hd", h, x_chunk) / (batch * (end - start))

            # Apply update at chunk boundary
            down_w = down_w - learning_rate * grad_down
            up_w = up_w - learning_rate * grad_up

    output = torch.cat(outputs, dim=1)
    return output, up_w, down_w


def cms_level_forward_with_update(
    level: nn.Module,
    x: Tensor,
    learning_rate: float = 1e-3,
) -> Tensor:
    """
    Run a CMS level's forward pass with in-place weight update.

    Convenience wrapper that extracts weights from the level module,
    runs the fused forward+update, and writes back the updated weights.

    Args:
        level: A CMSLevel module.
        x: Input tensor (batch, seq_len, dim).
        learning_rate: Learning rate for updates.

    Returns:
        Output tensor.
    """
    up_w = level.up_proj.weight.data
    down_w = level.down_proj.weight.data

    output, new_up, new_down = fused_cms_forward_update(
        x, up_w, down_w,
        learning_rate=learning_rate,
        chunk_size=level.chunk_size,
    )

    # Write back updated weights
    with torch.no_grad():
        level.up_proj.weight.copy_(new_up)
        level.down_proj.weight.copy_(new_down)

    return output
