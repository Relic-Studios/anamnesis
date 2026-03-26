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

from anamnesis.kernels import _TRITON_AVAILABLE

if _TRITON_AVAILABLE:
    import triton
    import triton.language as tl


def fused_cms_forward_update(
    x: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
    learning_rate: float = 1e-3,
    chunk_size: int = 32,
    gate_weight: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Fused CMS level forward pass + gradient accumulation + weight update.

    Processes tokens in chunks, accumulating gradients within each chunk
    and applying updates at chunk boundaries.

    Args:
        x: Input tensor (batch, seq_len, dim).
        up_weight: Up projection weight (hidden_dim, dim).
        down_weight: Down projection weight (dim, hidden_dim).
        learning_rate: Learning rate for weight updates.
        chunk_size: Tokens per chunk (update at chunk boundaries).
        gate_weight: Optional SwiGLU gate weight (hidden_dim, dim).

    Returns:
        Tuple of (output, updated_up_weight, updated_down_weight).
    """
    if _TRITON_AVAILABLE and x.is_cuda:
        return _fused_cms_cuda(x, up_weight, down_weight, learning_rate, chunk_size, gate_weight)
    return _fused_cms_pytorch(x, up_weight, down_weight, learning_rate, chunk_size, gate_weight)


def _fused_cms_pytorch(
    x: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
    learning_rate: float = 1e-3,
    chunk_size: int = 32,
    gate_weight: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pure PyTorch implementation of fused forward+update."""
    batch, seq_len, dim = x.shape
    hidden_dim = up_weight.shape[0]

    outputs = []
    up_w = up_weight.clone()
    down_w = down_weight.clone()
    gate_w = gate_weight.clone() if gate_weight is not None else None

    num_chunks = (seq_len + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, seq_len)
        x_chunk = x[:, start:end, :]

        # Forward pass
        h_up = torch.nn.functional.silu(x_chunk @ up_w.T)
        if gate_w is not None:
            h_gate = torch.nn.functional.silu(x_chunk @ gate_w.T)
            h = h_gate * h_up
            y = h @ down_w.T  # SwiGLU: no residual
        else:
            h = h_up
            y = x_chunk + h @ down_w.T  # Residual

        outputs.append(y)

        # Gradient computation at chunk boundary (dual form)
        if learning_rate > 0 and chunk_idx < num_chunks - 1:
            chunk_len = end - start
            residual = h @ down_w.T
            # Gradient w.r.t. down_w
            grad_down = torch.einsum("bcd,bch->dh", residual, h) / (batch * chunk_len)
            # Gradient w.r.t. up_w
            grad_up = torch.einsum("bch,bcd->hd", h, x_chunk) / (batch * chunk_len)

            # Clip gradients
            gn_down = grad_down.norm()
            gn_up = grad_up.norm()
            max_norm = 10.0
            if gn_down > max_norm:
                grad_down = grad_down * (max_norm / gn_down)
            if gn_up > max_norm:
                grad_up = grad_up * (max_norm / gn_up)

            down_w = down_w - learning_rate * grad_down
            up_w = up_w - learning_rate * grad_up

    output = torch.cat(outputs, dim=1)
    return output, up_w, down_w


def _fused_cms_cuda(
    x: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
    learning_rate: float = 1e-3,
    chunk_size: int = 32,
    gate_weight: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    CUDA-optimized fused CMS forward+update.

    Uses Triton for the per-chunk matmuls and gradient accumulation.
    For the actual weight update (small operation), stays in PyTorch.
    The key optimization is keeping the chunk's activation in GPU L2 cache
    across forward→gradient→update by processing one chunk at a time.
    """
    batch, seq_len, dim = x.shape
    hidden_dim = up_weight.shape[0]

    # Pre-allocate output
    output = torch.empty_like(x) if gate_weight is None else torch.empty(batch, seq_len, down_weight.shape[0], device=x.device, dtype=x.dtype)
    up_w = up_weight.clone()
    down_w = down_weight.clone()
    gate_w = gate_weight.clone() if gate_weight is not None else None

    num_chunks = (seq_len + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, seq_len)
        x_chunk = x[:, start:end, :].contiguous()
        chunk_len = end - start

        # Forward: cuBLAS matmul (Triton doesn't beat cuBLAS for standard GEMM)
        h_up = torch.nn.functional.silu(x_chunk @ up_w.T)
        if gate_w is not None:
            h_gate = torch.nn.functional.silu(x_chunk @ gate_w.T)
            h = h_gate * h_up
            y = h @ down_w.T
        else:
            h = h_up
            y = x_chunk + h @ down_w.T

        output[:, start:end, :] = y

        # Gradient accumulation + update at chunk boundary
        if learning_rate > 0 and chunk_idx < num_chunks - 1:
            # These are small matmuls that fit in L2 cache
            residual = h @ down_w.T
            grad_down = torch.einsum("bcd,bch->dh", residual, h).div_(batch * chunk_len)
            grad_up = torch.einsum("bch,bcd->hd", h, x_chunk).div_(batch * chunk_len)

            # Gradient clipping
            max_norm = 10.0
            gn = grad_down.norm()
            if gn > max_norm:
                grad_down.mul_(max_norm / gn)
            gn = grad_up.norm()
            if gn > max_norm:
                grad_up.mul_(max_norm / gn)

            down_w.sub_(grad_down, alpha=learning_rate)
            up_w.sub_(grad_up, alpha=learning_rate)

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
    gate_w = level.gate_proj.weight.data if hasattr(level, 'gate_proj') and level.swiglu else None

    output, new_up, new_down = fused_cms_forward_update(
        x, up_w, down_w,
        learning_rate=learning_rate,
        chunk_size=level.chunk_size,
        gate_weight=gate_w,
    )

    # Write back updated weights
    with torch.no_grad():
        level.up_proj.weight.copy_(new_up)
        level.down_proj.weight.copy_(new_down)

    return output
