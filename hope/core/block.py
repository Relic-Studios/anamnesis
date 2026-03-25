"""
HopeBlock — A single transformer block with CMS replacing MLP.

Implements the Hope architecture (Section 8, Behrouz et al., NeurIPS 2025):
    1. Pre-norm attention (standard or self-referential)
    2. Pre-norm CMS (replaces SwiGLU MLP)
    3. Optional neural memory integration (Titans-style)

The block follows the standard pre-RMSNorm residual pattern:
    x = x + Attention(RMSNorm(x))
    x = x + CMS(RMSNorm(x))
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from hope.core.cms import ContinuumMemorySystem, CMSVariant
from hope.core.memory import NeuralMemory, MemoryState


class HopeBlock(nn.Module):
    """
    A single Hope transformer block.

    Replaces the standard Transformer block's MLP with a ContinuumMemorySystem.
    Optionally includes a NeuralMemory module for long-term memory.

    Args:
        dim: Hidden dimension.
        num_attention_heads: Number of attention heads.
        num_kv_heads: Number of key-value heads (for GQA). None = MHA.
        cms_levels: Number of CMS levels.
        cms_chunk_sizes: Update frequencies per CMS level.
        cms_variant: CMS variant (nested/sequential/independent).
        cms_hidden_mult: Hidden multiplier for CMS MLPs.
        use_neural_memory: Whether to include Titans-style neural memory.
        mem_heads: Number of neural memory heads.
        mem_depth: Depth of neural memory MLP.
        norm_eps: Epsilon for RMSNorm.
        max_position_embeddings: Max sequence length (for RoPE).
        rope_theta: RoPE theta parameter.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int = 28,
        num_kv_heads: int | None = None,
        cms_levels: int = 4,
        cms_chunk_sizes: list[int] | None = None,
        cms_variant: CMSVariant = CMSVariant.NESTED,
        cms_hidden_mult: float = 4.0,
        use_neural_memory: bool = False,
        mem_heads: int = 4,
        mem_depth: int = 2,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads or num_attention_heads
        self.head_dim = dim // num_attention_heads

        # Pre-attention norm
        self.input_layernorm = nn.RMSNorm(dim, eps=norm_eps)

        # Attention projections (GQA-compatible)
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # Pre-CMS norm
        self.post_attention_layernorm = nn.RMSNorm(dim, eps=norm_eps)

        # CMS replaces MLP
        self.cms = ContinuumMemorySystem(
            dim=dim,
            num_levels=cms_levels,
            chunk_sizes=cms_chunk_sizes,
            hidden_mult=cms_hidden_mult,
            variant=cms_variant,
        )

        # Optional neural memory (Titans-style)
        self.neural_memory: NeuralMemory | None = None
        if use_neural_memory:
            self.neural_memory = NeuralMemory(
                dim=dim,
                num_heads=mem_heads,
                mem_depth=mem_depth,
            )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        memory_state: MemoryState | None = None,
    ) -> tuple[Tensor, MemoryState | None]:
        """
        Forward pass through one Hope block.

        Args:
            hidden_states: Input (batch, seq_len, dim).
            attention_mask: Causal attention mask.
            position_ids: Position IDs for RoPE.
            memory_state: Persistent neural memory state.

        Returns:
            Tuple of (output hidden states, updated memory state or None).
        """
        # ── ATTENTION PATH ──
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Compute Q, K, V
        batch, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        q = q.view(batch, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # GQA: expand KV heads
        if self.num_kv_heads < self.num_attention_heads:
            repeat_factor = self.num_attention_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        # Scaled dot-product attention (will use Flash Attention when available)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            is_causal=attention_mask is None,
        )

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.dim)
        attn_output = self.o_proj(attn_output)

        hidden_states = residual + attn_output

        # ── NEURAL MEMORY PATH (optional) ──
        new_memory_state = None
        if self.neural_memory is not None:
            mem_output, new_memory_state = self.neural_memory(hidden_states, memory_state)
            hidden_states = hidden_states + mem_output

        # ── CMS PATH (replaces MLP) ──
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.cms(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_memory_state

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, heads={self.num_attention_heads}, "
            f"kv_heads={self.num_kv_heads}, "
            f"neural_memory={'yes' if self.neural_memory else 'no'}"
        )
