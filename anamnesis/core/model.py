"""
HopeModel — Full Hope-Didymus transformer model.

Composes HopeBlocks into a complete language model with:
- Token embeddings
- Stack of HopeBlocks (attention + CMS)
- Final RMSNorm + language model head
- CMS state management for persistence across sessions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from anamnesis.core.block import HopeBlock
from anamnesis.core.cms import CMSVariant
from anamnesis.core.memory import MemoryState


@dataclass
class HopeConfig:
    """Configuration for a HopeModel."""
    vocab_size: int = 32000
    hidden_size: int = 3584
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_kv_heads: int = 4
    max_position_embeddings: int = 32768
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6

    # CMS configuration
    cms_levels: int = 4
    cms_chunk_sizes: list[int] | None = None
    cms_variant: str = "nested"
    cms_hidden_mult: float = 5.286  # Qwen's 18944/3584 ratio

    # Neural memory configuration
    use_neural_memory: bool = False
    mem_heads: int = 4
    mem_depth: int = 2

    # Tie word embeddings
    tie_word_embeddings: bool = False

    @property
    def cms_variant_enum(self) -> CMSVariant:
        return CMSVariant(self.cms_variant)

    @classmethod
    def from_qwen2_5_7b(cls) -> "HopeConfig":
        """Create a config matching Qwen 2.5 7B Instruct dimensions."""
        return cls(
            vocab_size=152064,
            hidden_size=3584,
            num_hidden_layers=28,
            num_attention_heads=28,
            num_kv_heads=4,
            max_position_embeddings=32768,
            rope_theta=1_000_000.0,
            rms_norm_eps=1e-6,
            cms_hidden_mult=18944 / 3584,  # 5.2857...
            tie_word_embeddings=False,
        )


class HopeModel(nn.Module):
    """
    Full Hope-Didymus language model.

    Architecture:
        embed_tokens → [HopeBlock × N] → RMSNorm → lm_head

    Each HopeBlock contains:
        - Grouped Query Attention (GQA)
        - Continuum Memory System (CMS) replacing SwiGLU MLP
        - Optional Neural Memory (Titans-style)

    Args:
        config: HopeConfig with all architecture parameters.
    """

    def __init__(self, config: HopeConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer blocks
        self.layers = nn.ModuleList([
            HopeBlock(
                dim=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                num_kv_heads=config.num_kv_heads,
                cms_levels=config.cms_levels,
                cms_chunk_sizes=config.cms_chunk_sizes,
                cms_variant=config.cms_variant_enum,
                cms_hidden_mult=config.cms_hidden_mult,
                use_neural_memory=config.use_neural_memory,
                mem_heads=config.mem_heads,
                mem_depth=config.mem_depth,
                norm_eps=config.rms_norm_eps,
            )
            for _ in range(config.num_hidden_layers)
        ])

        # Final norm
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Language model head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights if configured
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        memory_states: list[MemoryState | None] | None = None,
    ) -> dict[str, Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs (batch, seq_len).
            attention_mask: Attention mask.
            labels: Target token IDs for loss computation.
            memory_states: Per-layer neural memory states.

        Returns:
            Dict with 'logits' and optionally 'loss'.
        """
        hidden_states = self.embed_tokens(input_ids)

        if memory_states is None:
            memory_states = [None] * len(self.layers)

        new_memory_states = []
        for layer, mem_state in zip(self.layers, memory_states):
            hidden_states, new_mem = layer(
                hidden_states,
                attention_mask=attention_mask,
                memory_state=mem_state,
            )
            new_memory_states.append(new_mem)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        result = {"logits": logits, "memory_states": new_memory_states}

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )
            result["loss"] = loss

        return result

    def enable_drift(self, enabled: bool = True) -> None:
        """Enable/disable neutral drift across all CMS levels in all layers."""
        for layer in self.layers:
            layer.cms.enable_drift(enabled)

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
