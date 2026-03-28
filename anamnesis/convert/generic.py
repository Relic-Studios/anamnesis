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


def _init_deep_memory_from_svd(
    level: nn.Module,
    up_weight: Tensor,
    down_weight: Tensor,
) -> None:
    """Initialize DeepMemoryLevel projections from L0's pre-trained SVD.

    Instead of random init (which produces meaningless keys/values), we
    extract the most important feature directions from L0's existing weights.
    This gives the memory meaningful inputs from day one — no outer-loop
    training needed.

    Args:
        level: DeepMemoryLevel to initialize.
        up_weight: L0's up_proj weight (hidden_dim, dim).
        down_weight: L0's down_proj weight (dim, hidden_dim).
    """
    mem_dim = level.mem_dim

    with torch.no_grad():
        # SVD of up_proj: extract top input feature directions
        # up_weight is (hidden_dim, dim). SVD gives U(hidden,k) S(k) V^T(k, dim)
        # V^T rows are the most important directions in input space.
        U_up, S_up, Vt_up = torch.linalg.svd(up_weight.float(), full_matrices=False)
        # Take top mem_dim directions from input space
        top_input_dirs = Vt_up[:mem_dim, :]  # (mem_dim, dim)

        # Key and value projections: project input into L0's principal subspace
        level.to_k.weight.copy_(top_input_dirs.to(level.to_k.weight.dtype))
        level.to_v.weight.copy_(top_input_dirs.to(level.to_v.weight.dtype))
        level.to_q.weight.copy_(top_input_dirs.to(level.to_q.weight.dtype))

        # SVD of down_proj: extract top output directions
        # down_weight is (dim, hidden_dim). SVD gives U(dim,k) S(k) V^T(k, hidden)
        # U columns are the most important directions in output space.
        U_down, S_down, Vt_down = torch.linalg.svd(down_weight.float(), full_matrices=False)
        # out_proj maps mem_dim → dim. Use top output directions scaled by singular values.
        # Scale gives the projection proportional weight in the residual stream.
        top_output_dirs = U_down[:, :mem_dim] * S_down[:mem_dim].unsqueeze(0)
        # Scale so L1 starts as a modest perturbation on L0.
        top_output_dirs = top_output_dirs * 0.1
        level.out_proj.weight.copy_(top_output_dirs.to(level.out_proj.weight.dtype))

        # v_expand and mem_out_proj: connect mem_dim <-> poly_dim
        # Use identity-like init so polynomial features pass through cleanly
        if hasattr(level, 'v_expand') and not isinstance(level.v_expand, nn.Identity):
            # v_expand: (poly_dim, mem_dim) — tile the top input directions
            poly_dim = level.v_expand.weight.shape[0]
            n_copies = poly_dim // mem_dim
            v_exp_init = torch.eye(mem_dim).repeat(n_copies, 1) / n_copies
            level.v_expand.weight.copy_(v_exp_init.to(level.v_expand.weight.dtype))

        if hasattr(level, 'mem_out_proj') and not isinstance(level.mem_out_proj, nn.Identity):
            # mem_out_proj: (mem_dim, poly_dim) — average across polynomial terms
            poly_dim = level.mem_out_proj.weight.shape[1]
            n_copies = poly_dim // mem_dim
            mop_init = torch.eye(mem_dim).repeat(1, n_copies) / n_copies
            level.mem_out_proj.weight.copy_(mop_init.to(level.mem_out_proj.weight.dtype))

        # Memory MLP: small init so M(x) ≈ x + small noise
        for name, param in level.memory.named_parameters():
            nn.init.normal_(param, std=0.01)


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
        gate, up, down = extract_mlp_weights(source_layer)

        for i, level in enumerate(target_block.cms.levels):
            with torch.no_grad():
                if level.swiglu:
                    # Level 0: exact copy of SwiGLU weights
                    level.gate_proj.weight.copy_(gate)
                    level.up_proj.weight.copy_(up)
                    level.down_proj.weight.copy_(down)
                elif hasattr(level, 'memory'):
                    # DeepMemoryLevel: initialize projections from L0's SVD.
                    # This gives the memory meaningful input/output directions
                    # extracted from the pre-trained MLP — no training needed.
                    _init_deep_memory_from_svd(level, up, down)
                elif hasattr(level, 'A'):
                    # Legacy LowRankLevel (backward compat)
                    nn.init.normal_(level.A.weight, std=0.02)
                    nn.init.normal_(level.B.weight, std=0.02)
                else:
                    # Legacy full-rank residual levels
                    nn.init.normal_(level.up_proj.weight, std=0.02)
                    nn.init.zeros_(level.down_proj.weight)


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
