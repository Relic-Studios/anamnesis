"""
Rotary Position Embeddings (RoPE) for Hope attention.

Implements the standard RoPE mechanism used by Qwen2, Llama, and most
modern transformers. No learned parameters — positions are encoded via
rotation in the complex plane.

Reference: RoFormer (Su et al., 2021), Qwen2 implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding.

    Computes cos/sin position encodings from theta and head dimension.
    No learned parameters — registered as buffers for device/dtype tracking.
    """

    def __init__(self, head_dim: int, max_position_embeddings: int = 32768, theta: float = 1_000_000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings

    def forward(self, x: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute cos and sin for given positions.

        Args:
            x: Input tensor (used only for dtype).
            position_ids: Position indices (batch, seq_len).

        Returns:
            Tuple of (cos, sin), each shape (batch, seq_len, head_dim).
        """
        freqs = position_ids.unsqueeze(-1).float() * self.inv_freq.unsqueeze(0).unsqueeze(0)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def rotate_half(x: Tensor) -> Tensor:
    """Rotate half the hidden dims: [-x2, x1]."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """
    Apply rotary position embeddings to Q and K tensors.

    Args:
        q: Query tensor (batch, num_heads, seq_len, head_dim).
        k: Key tensor (batch, num_kv_heads, seq_len, head_dim).
        cos: Cosine embeddings (batch, seq_len, head_dim).
        sin: Sine embeddings (batch, seq_len, head_dim).

    Returns:
        Tuple of (rotated_q, rotated_k).
    """
    cos = cos.unsqueeze(1)  # (batch, 1, seq_len, head_dim)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
