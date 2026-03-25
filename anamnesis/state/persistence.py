"""
CMS State Persistence — save/load CMS weights across sessions.

The CMS state represents accumulated learning — it's what makes
the model continuously improve. Losing it means losing everything
the model has learned since conversion.

State format (v1):
    {
        "version": 1,
        "config": { CMS configuration },
        "layers": {
            "0": { "level_0": {...weights}, "level_1": {...}, ... },
            ...
        },
        "metadata": {
            "tokens_processed": int,
            "signal_health_avg": float,
            "created_at": str,
            "session_count": int,
        }
    }

Soul checkpoints are a special case: the CMS state at the time of
initial identity training. They serve as the prior P(s) for the
identity drift penalty in the composite loss.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from anamnesis.core.model import HopeModel


STATE_VERSION = 1


def save_cms_state(
    model: HopeModel,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Save CMS state from all layers of a HopeModel.

    Args:
        model: The HopeModel to save state from.
        path: File path to save to (.pt).
        metadata: Optional metadata (tokens_processed, signal_health, etc.).
    """
    state = {
        "version": STATE_VERSION,
        "config": {
            "cms_levels": model.config.cms_levels,
            "cms_chunk_sizes": model.config.cms_chunk_sizes,
            "cms_variant": model.config.cms_variant,
            "hidden_size": model.config.hidden_size,
            "num_layers": model.config.num_hidden_layers,
        },
        "layers": {},
        "metadata": metadata or {},
    }

    # Add timestamp if not present
    if "created_at" not in state["metadata"]:
        state["metadata"]["created_at"] = datetime.now(timezone.utc).isoformat()

    # Extract CMS weights from each layer
    for layer_idx, layer in enumerate(model.layers):
        layer_state = {}
        for level_idx, level in enumerate(layer.cms.levels):
            level_state = {
                name: param.detach().cpu().clone()
                for name, param in level.named_parameters()
            }
            layer_state[f"level_{level_idx}"] = level_state
        state["layers"][str(layer_idx)] = layer_state

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_cms_state(
    model: HopeModel,
    path: str | Path,
    strict: bool = True,
) -> dict[str, Any]:
    """
    Load CMS state into a HopeModel.

    Args:
        model: The HopeModel to load state into.
        path: File path to load from (.pt).
        strict: If True, raise on version or shape mismatch.

    Returns:
        Metadata dict from the saved state.
    """
    state = torch.load(Path(path), map_location="cpu", weights_only=True)

    # Version check
    version = state.get("version", 0)
    if strict and version != STATE_VERSION:
        raise ValueError(
            f"State version mismatch: file has v{version}, expected v{STATE_VERSION}"
        )

    # Load weights into model
    with torch.no_grad():
        for layer_idx, layer in enumerate(model.layers):
            layer_key = str(layer_idx)
            if layer_key not in state["layers"]:
                if strict:
                    raise KeyError(f"Missing layer {layer_idx} in state file")
                continue

            layer_state = state["layers"][layer_key]
            for level_idx, level in enumerate(layer.cms.levels):
                level_key = f"level_{level_idx}"
                if level_key not in layer_state:
                    if strict:
                        raise KeyError(
                            f"Missing level {level_idx} in layer {layer_idx}"
                        )
                    continue

                level_state = layer_state[level_key]
                for name, param in level.named_parameters():
                    if name in level_state:
                        saved = level_state[name]
                        if param.shape != saved.shape:
                            if strict:
                                raise ValueError(
                                    f"Shape mismatch for layer {layer_idx} "
                                    f"level {level_idx} param {name}: "
                                    f"{param.shape} vs {saved.shape}"
                                )
                            continue
                        param.copy_(saved.to(param.device))

    return state.get("metadata", {})


def save_soul_checkpoint(
    model: HopeModel,
    path: str | Path,
    description: str = "",
) -> None:
    """
    Save a soul checkpoint — the identity anchor.

    This is the CMS state after initial identity training.
    It serves as the prior P(s) for the identity drift penalty.
    Losing this means losing the ability to anchor identity.

    Args:
        model: HopeModel at the point of identity establishment.
        path: File path to save to (.pt).
        description: Human-readable description of this checkpoint.
    """
    save_cms_state(
        model, path,
        metadata={
            "type": "soul_checkpoint",
            "description": description,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def load_soul_checkpoint(
    path: str | Path,
) -> list[dict[str, dict[str, Tensor]]]:
    """
    Load a soul checkpoint as nested dicts (for identity drift computation).

    Does NOT load into a model — returns the raw per-layer, per-level weights
    for use in IdentityDrift loss computation.

    Args:
        path: File path to the soul checkpoint.

    Returns:
        List of dicts, one per layer. Each dict maps level names to
        dicts of parameter tensors.
    """
    state = torch.load(Path(path), map_location="cpu", weights_only=True)

    layers = []
    num_layers = state["config"]["num_layers"]
    for layer_idx in range(num_layers):
        layer_key = str(layer_idx)
        if layer_key in state["layers"]:
            layers.append(state["layers"][layer_key])
        else:
            layers.append({})

    return layers
