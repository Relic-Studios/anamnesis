"""
Qwen 2.5 → Hope-Didymus conversion.

Converts a Qwen2ForCausalLM model to Hope-Didymus architecture.

Qwen 2.5 7B Instruct specifics:
    - hidden_size=3584, intermediate_size=18944, 28 layers
    - GQA: 28 query heads, 4 KV heads, head_dim=128
    - SwiGLU MLP: gate_proj(3584→18944) + up_proj(3584→18944) + down_proj(18944→3584)
    - Q/K/V have bias=True, O has bias=False
    - RoPE with theta=1_000_000

The conversion:
    1. Copy embeddings, norms, LM head unchanged
    2. Copy attention Q/K/V/O weights (optionally convert to self-referential)
    3. Replace SwiGLU MLP with CMS chain (initialized from pre-trained weights)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from anamnesis.core.cms import CMSVariant
from anamnesis.core.model import HopeModel, HopeConfig
from anamnesis.convert.generic import model_to_hope


def qwen_to_hope(
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct",
    cms_levels: int = 4,
    cms_chunk_sizes: list[int] | None = None,
    cms_variant: str = "nested",
    self_referential: bool = False,
    use_neural_memory: bool = False,
    device: str = "cpu",
    dtype: torch.dtype | None = None,
    verbose: bool = True,
) -> HopeModel:
    """
    Convert a Qwen 2.5 model to Hope-Didymus.

    Requires the `transformers` package (in the [convert] extra).

    Example:
        from anamnesis.convert import qwen_to_hope

        model = qwen_to_hope(
            "Qwen/Qwen2.5-7B-Instruct",
            cms_levels=4,
            cms_variant="nested",
        )

    Args:
        model_name_or_path: HuggingFace model name or local path.
        cms_levels: Number of CMS levels (default 4).
        cms_chunk_sizes: Update frequencies per level.
            Default: [1, 32, 256, 2048].
        cms_variant: CMS variant: "nested", "sequential", or "independent".
        self_referential: Whether to convert attention to self-referential.
        use_neural_memory: Whether to add Titans-style neural memory.
        device: Device for the converted model.
        dtype: Data type (default: model's native dtype).
        verbose: Print progress.

    Returns:
        Initialized HopeModel with pre-trained weights.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoConfig
    except ImportError:
        raise ImportError(
            "Converting models requires the `transformers` package. "
            "Install with: pip install hope-didymus[convert]"
        )

    if verbose:
        print(f"Loading {model_name_or_path}...")

    # Load the source model
    source_config = AutoConfig.from_pretrained(model_name_or_path)
    source_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype or torch.bfloat16,
        device_map=device,
    )

    if verbose:
        print(f"Source model loaded: {source_config.model_type}")
        print(f"  hidden_size={source_config.hidden_size}")
        print(f"  num_layers={source_config.num_hidden_layers}")
        print(f"  num_heads={source_config.num_attention_heads}")
        print(f"  num_kv_heads={source_config.num_key_value_heads}")

    # Build Hope config from source
    hope_config = HopeConfig(
        vocab_size=source_config.vocab_size,
        hidden_size=source_config.hidden_size,
        num_hidden_layers=source_config.num_hidden_layers,
        num_attention_heads=source_config.num_attention_heads,
        num_kv_heads=source_config.num_key_value_heads,
        max_position_embeddings=source_config.max_position_embeddings,
        rope_theta=getattr(source_config, "rope_theta", 1_000_000.0),
        rms_norm_eps=getattr(source_config, "rms_norm_eps", 1e-6),
        cms_levels=cms_levels,
        cms_chunk_sizes=cms_chunk_sizes,
        cms_variant=cms_variant,
        cms_hidden_mult=source_config.intermediate_size / source_config.hidden_size,
        use_neural_memory=use_neural_memory,
        tie_word_embeddings=getattr(source_config, "tie_word_embeddings", False),
    )

    if verbose:
        print(f"\nConverting to Hope-Didymus...")
        print(f"  CMS: {cms_levels} levels, {cms_chunk_sizes or 'default'}, {cms_variant}")
        print(f"  Self-referential: {self_referential}")
        print(f"  Neural memory: {use_neural_memory}")

    # Convert
    hope_model = model_to_hope(
        source_model=source_model,
        config=hope_config,
        self_referential=self_referential,
        verbose=verbose,
    )

    # Move to target device if needed
    if device != "cpu":
        hope_model = hope_model.to(device)

    # Free source model memory
    del source_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if verbose:
        print(f"\nConversion complete!")
        print(f"  Total params: {hope_model.num_parameters(trainable_only=False):,}")

    return hope_model
