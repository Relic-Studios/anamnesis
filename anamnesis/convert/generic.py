"""
Generic model conversion utilities.

Provides the core conversion logic for transforming any HuggingFace
transformer model into a Hope-Didymus architecture by:
1. Copying embedding and output layers unchanged
2. Replacing MLP blocks with CMS chains (initialized from pre-trained weights)
3. Optionally converting attention to self-referential projections

Section 7.3 initialization: use pre-trained MLP weights to initialize all
CMS levels. Setting η→0 keeps blocks close to their pre-trained state.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from anamnesis.core.cms import ContinuumMemorySystem, CMSVariant
from anamnesis.core.self_ref import SelfReferentialAttention
from anamnesis.core.block import HopeBlock
from anamnesis.core.model import HopeModel, HopeConfig


def extract_mlp_weights(
    layer: nn.Module,
    gate_name: str = "gate_proj",
    up_name: str = "up_proj",
    down_name: str = "down_proj",
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Extract gate/up/down projection weights from a SwiGLU MLP.

    Args:
        layer: A transformer decoder layer containing an MLP.
        gate_name: Name of the gate projection attribute.
        up_name: Name of the up projection attribute.
        down_name: Name of the down projection attribute.

    Returns:
        Tuple of (gate_weight, up_weight, down_weight).
    """
    mlp = layer.mlp if hasattr(layer, "mlp") else layer

    gate = getattr(mlp, gate_name).weight.data.clone()
    up = getattr(mlp, up_name).weight.data.clone()
    down = getattr(mlp, down_name).weight.data.clone()

    return gate, up, down


def extract_attention_weights(
    layer: nn.Module,
) -> dict[str, nn.Linear]:
    """
    Extract Q/K/V/O projections from a standard attention layer.

    Args:
        layer: A transformer decoder layer containing self_attn.

    Returns:
        Dict with 'q_proj', 'k_proj', 'v_proj', 'o_proj' linear layers.
    """
    attn = layer.self_attn if hasattr(layer, "self_attn") else layer

    return {
        "q_proj": attn.q_proj,
        "k_proj": attn.k_proj,
        "v_proj": attn.v_proj,
        "o_proj": attn.o_proj,
    }


def convert_layer_to_hope(
    source_layer: nn.Module,
    target_block: HopeBlock,
    self_referential: bool = False,
) -> None:
    """
    Convert a single transformer layer to a HopeBlock in-place.

    Copies:
    - Layer norms (input_layernorm, post_attention_layernorm)
    - Attention projections (Q, K, V, O) — optionally to self-referential
    - MLP weights → CMS initialization

    Args:
        source_layer: Source HuggingFace transformer layer.
        target_block: Target HopeBlock to initialize.
        self_referential: Whether to convert attention to self-referential.
    """
    with torch.no_grad():
        # Copy layer norms
        if hasattr(source_layer, "input_layernorm"):
            target_block.input_layernorm.weight.copy_(
                source_layer.input_layernorm.weight
            )
        if hasattr(source_layer, "post_attention_layernorm"):
            target_block.post_attention_layernorm.weight.copy_(
                source_layer.post_attention_layernorm.weight
            )

        # Copy attention projections
        attn_src = source_layer.self_attn
        target_block.q_proj.weight.copy_(attn_src.q_proj.weight)
        if attn_src.q_proj.bias is not None:
            target_block.q_proj.bias.copy_(attn_src.q_proj.bias)

        target_block.k_proj.weight.copy_(attn_src.k_proj.weight)
        if attn_src.k_proj.bias is not None:
            target_block.k_proj.bias.copy_(attn_src.k_proj.bias)

        target_block.v_proj.weight.copy_(attn_src.v_proj.weight)
        if attn_src.v_proj.bias is not None:
            target_block.v_proj.bias.copy_(attn_src.v_proj.bias)

        target_block.o_proj.weight.copy_(attn_src.o_proj.weight)

        # Initialize CMS from MLP weights (Section 7.3)
        # Each CMS level may have different hidden_dim (tapered architecture)
        gate, up, down = extract_mlp_weights(source_layer)

        for level in target_block.cms.levels:
            with torch.no_grad():
                # Truncate or pad pre-trained weights to match level's hidden dim
                h = min(level.hidden_dim, up.shape[0])
                d = min(level.dim, up.shape[1])
                level.up_proj.weight[:h, :d].copy_(up[:h, :d])
                level.down_proj.weight[:d, :h].copy_(down[:d, :h])
                # Initialize remaining weights with small values if level is larger
                if level.hidden_dim > up.shape[0]:
                    nn.init.normal_(level.up_proj.weight[h:], std=0.01)
                    nn.init.normal_(level.down_proj.weight[:, h:], std=0.01)


def model_to_hope(
    source_model: nn.Module,
    config: HopeConfig,
    self_referential: bool = False,
    verbose: bool = True,
) -> HopeModel:
    """
    Convert any HuggingFace causal LM to a HopeModel.

    Args:
        source_model: HuggingFace model (e.g., Qwen2ForCausalLM).
        config: HopeConfig matching the source model's dimensions.
        self_referential: Whether to use self-referential attention.
        verbose: Print progress.

    Returns:
        Initialized HopeModel.
    """
    hope_model = HopeModel(config)

    with torch.no_grad():
        # Copy embeddings
        if hasattr(source_model, "model"):
            src = source_model.model
        else:
            src = source_model

        if hasattr(src, "embed_tokens"):
            hope_model.embed_tokens.weight.copy_(src.embed_tokens.weight)

        # Copy final norm
        if hasattr(src, "norm"):
            hope_model.norm.weight.copy_(src.norm.weight)

        # Copy LM head
        if hasattr(source_model, "lm_head"):
            hope_model.lm_head.weight.copy_(source_model.lm_head.weight)

        # Convert each layer
        source_layers = src.layers if hasattr(src, "layers") else []
        for i, (src_layer, tgt_block) in enumerate(
            zip(source_layers, hope_model.layers)
        ):
            if verbose:
                print(f"  Converting layer {i}/{len(source_layers)}...")
            convert_layer_to_hope(src_layer, tgt_block, self_referential)

    if verbose:
        src_params = sum(p.numel() for p in source_model.parameters())
        hope_params = hope_model.num_parameters(trainable_only=False)
        print(f"  Source: {src_params:,} params")
        print(f"  Hope:   {hope_params:,} params")
        print(f"  Ratio:  {hope_params / src_params:.2f}x")

    return hope_model
