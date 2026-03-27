"""
Evaluation metrics for Anamnesis models.

Provides standardized measurements for:
- Perplexity (language modeling quality)
- CMS state delta (how much did learning change the weights)
- Surprise profile (per-level surprise distribution)
- Signal trajectory (signal health over time during inference)
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from anamnesis.core.model import HopeModel


@torch.no_grad()
def compute_perplexity(
    model: HopeModel,
    dataloader: DataLoader,
    max_batches: int = 0,
) -> float:
    """
    Compute perplexity on a dataset.

    Args:
        model: HopeModel in eval mode.
        dataloader: DataLoader yielding dicts with 'input_ids'.
        max_batches: Limit evaluation to N batches (0 = all).

    Returns:
        Perplexity (exp of average cross-entropy loss).
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i, batch in enumerate(dataloader):
        if max_batches and i >= max_batches:
            break

        input_ids = batch["input_ids"]
        if hasattr(input_ids, "to"):
            input_ids = input_ids.to(next(model.parameters()).device)

        if input_ids.shape[1] < 2:
            continue

        output = model(input_ids, labels=input_ids)
        n_tokens = input_ids.shape[1] - 1
        total_loss += output["loss"].item() * n_tokens
        total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    if avg_loss > 100:  # Prevent overflow in math.exp
        return float("inf")
    return math.exp(avg_loss)


def snapshot_cms_state(model: HopeModel) -> dict[str, Tensor]:
    """Snapshot all CMS parameters (detached, on CPU)."""
    snapshot = {}
    for name, param in model.named_parameters():
        if "cms" in name:
            snapshot[name] = param.detach().cpu().clone()
    return snapshot


def compute_cms_delta(
    before: dict[str, Tensor],
    after: dict[str, Tensor],
) -> dict[str, float]:
    """
    Compute CMS weight change statistics.

    Args:
        before: Snapshot from before learning.
        after: Snapshot from after learning.

    Returns:
        Dict with total_l2 (total L2 norm of change), per_level (list of
        per-level L2 norms), and max_param_delta.
    """
    total_sq = 0.0
    max_delta = 0.0
    level_deltas: dict[str, float] = {}

    for key in before:
        if key not in after:
            continue
        delta = (after[key].float() - before[key].float())
        d_norm = delta.norm().item()
        total_sq += d_norm ** 2
        max_delta = max(max_delta, delta.abs().max().item())

        # Extract level index from param name (e.g., "layers.0.cms.levels.1.up_proj.weight")
        parts = key.split(".")
        for j, p in enumerate(parts):
            if p == "levels" and j + 1 < len(parts):
                level_key = f"level_{parts[j+1]}"
                level_deltas[level_key] = level_deltas.get(level_key, 0.0) + d_norm ** 2
                break

    per_level = {k: v ** 0.5 for k, v in sorted(level_deltas.items())}

    return {
        "total_l2": total_sq ** 0.5,
        "per_level": per_level,
        "max_param_delta": max_delta,
    }


@torch.no_grad()
def compute_surprise_profile(model: HopeModel) -> list[dict[str, float]]:
    """
    Get per-layer, per-level surprise values from the CMS.

    Returns:
        List of dicts (one per layer), each mapping level index to surprise value.
    """
    profile = []
    for layer in model.layers:
        surprises = layer.cms.get_surprise()
        profile.append({
            f"level_{i}": s for i, s in enumerate(surprises)
        })
    return profile


@torch.no_grad()
def compute_signal_trajectory(
    model: HopeModel,
    dataloader: DataLoader,
    signal_proxy: nn.Module,
    max_batches: int = 0,
) -> list[float]:
    """
    Compute signal health trajectory as the model processes data.

    The model's CMS learns during inference (inner-loop), so signal
    may improve over successive batches.

    Args:
        model: HopeModel in eval mode.
        dataloader: DataLoader yielding dicts with 'input_ids'.
        signal_proxy: SignalProxy network for quality estimation.
        max_batches: Limit to N batches (0 = all).

    Returns:
        List of signal health estimates, one per batch.
    """
    model.eval()
    trajectory = []

    for i, batch in enumerate(dataloader):
        if max_batches and i >= max_batches:
            break

        input_ids = batch["input_ids"]
        device = next(model.parameters()).device
        if hasattr(input_ids, "to"):
            input_ids = input_ids.to(device)

        # Forward pass (CMS learns during this)
        output = model(input_ids)

        # Estimate signal from hidden states
        hidden = model.embed_tokens(input_ids)
        signal = signal_proxy(hidden).mean().item()
        trajectory.append(signal)

    return trajectory


def _hsic(K: Tensor, L: Tensor) -> Tensor:
    """Compute the Hilbert-Schmidt Independence Criterion (unbiased estimator)."""
    n = K.shape[0]
    H = torch.eye(n, device=K.device, dtype=K.dtype) - 1.0 / n
    KH = K @ H
    LH = L @ H
    return (KH * LH.T).sum() / max((n - 1) ** 2, 1)


@torch.no_grad()
def compute_cka(
    representations_a: Tensor,
    representations_b: Tensor,
) -> float:
    """
    Compute Centered Kernel Alignment (CKA) between two sets of representations.

    CKA measures representational similarity between two networks (Kornblith et al., 2019).
    Use this to measure specialization distance: run the same input through the
    base Qwen model and the evolved Anamnesis model, compare hidden representations.

    CKA ≈ 0.95 → model hasn't specialized (representations are nearly identical).
    CKA ≈ 0.70 → structural adaptation has occurred.
    CKA < 0.50 → significant representational divergence.

    Args:
        representations_a: Hidden states from model A, shape (n_samples, dim).
        representations_b: Hidden states from model B, shape (n_samples, dim).

    Returns:
        CKA similarity score in [0, 1].
    """
    # Ensure float32 for numerical stability
    X = representations_a.float()
    Y = representations_b.float()

    # Center
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Linear kernels
    K = X @ X.T
    L = Y @ Y.T

    hsic_kl = _hsic(K, L)
    hsic_kk = _hsic(K, K)
    hsic_ll = _hsic(L, L)

    denom = (hsic_kk * hsic_ll).sqrt()
    if denom < 1e-10:
        return 0.0

    return (hsic_kl / denom).clamp(0, 1).item()


@torch.no_grad()
def compute_layer_cka(
    model_a: HopeModel,
    model_b: HopeModel,
    input_ids: Tensor,
    layers: list[int] | None = None,
) -> dict[int, float]:
    """
    Compute per-layer CKA between two models on the same input.

    Runs input through both models, extracts hidden states after each
    transformer block, and computes CKA per layer.

    Args:
        model_a: Base model (e.g., freshly converted).
        model_b: Evolved model (after N conversations).
        input_ids: Shared input, shape (batch, seq_len).
        layers: Which layers to compare (None = all).

    Returns:
        Dict mapping layer index to CKA score.
    """
    model_a.eval()
    model_b.eval()

    # Collect hidden states per layer via hooks
    states_a: dict[int, Tensor] = {}
    states_b: dict[int, Tensor] = {}

    def make_hook(storage, idx):
        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            # Flatten to (n_tokens, dim) for CKA
            storage[idx] = h.reshape(-1, h.shape[-1]).detach()
        return hook

    hooks = []
    target_layers = layers or list(range(len(model_a.layers)))

    for idx in target_layers:
        hooks.append(model_a.layers[idx].register_forward_hook(make_hook(states_a, idx)))
        hooks.append(model_b.layers[idx].register_forward_hook(make_hook(states_b, idx)))

    # Forward pass through both
    model_a(input_ids)
    model_b(input_ids)

    # Clean up hooks
    for h in hooks:
        h.remove()

    # Compute CKA per layer
    cka_scores = {}
    for idx in target_layers:
        if idx in states_a and idx in states_b:
            cka_scores[idx] = compute_cka(states_a[idx], states_b[idx])

    return cka_scores


@torch.no_grad()
def evaluate_generation(
    model: HopeModel,
    prompts: list[Tensor],
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    num_samples: int = 3,
) -> dict[str, float]:
    """
    Evaluate generation quality via consistency and diversity.

    For each prompt, generates N samples and measures:
    - Consistency: average pairwise token overlap (Jaccard)
    - Diversity: 1 - consistency (variety of responses)

    Args:
        model: HopeModel in eval mode.
        prompts: List of tokenized prompt tensors.
        max_new_tokens: Maximum tokens to generate per response.
        temperature: Sampling temperature.
        num_samples: Number of samples per prompt.

    Returns:
        Dict with 'consistency' and 'diversity' scores.
    """
    model.eval()
    device = next(model.parameters()).device
    consistencies = []

    for prompt_ids in prompts:
        prompt_ids = prompt_ids.to(device)
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)

        samples = []
        for _ in range(num_samples):
            ids = prompt_ids.clone()
            generated = []
            for _ in range(max_new_tokens):
                logits = model(ids)["logits"][:, -1, :]
                probs = torch.softmax(logits / temperature, dim=-1)
                next_tok = torch.multinomial(probs, 1)
                generated.append(next_tok.item())
                ids = torch.cat([ids, next_tok], dim=-1)
            samples.append(set(generated))

        # Pairwise Jaccard similarity
        if len(samples) >= 2:
            pairs = []
            for i in range(len(samples)):
                for j in range(i + 1, len(samples)):
                    if samples[i] and samples[j]:
                        overlap = len(samples[i] & samples[j])
                        union = len(samples[i] | samples[j])
                        pairs.append(overlap / union if union > 0 else 0)
            if pairs:
                consistencies.append(sum(pairs) / len(pairs))

    avg_consistency = sum(consistencies) / len(consistencies) if consistencies else 0.0
    return {
        "consistency": avg_consistency,
        "diversity": 1.0 - avg_consistency,
    }
